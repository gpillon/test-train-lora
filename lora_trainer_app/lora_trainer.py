"""
LoRA Trainer module for fine-tuning language models with LoRA/QLoRA.
Supports Hugging Face transformers, PEFT, and TRL libraries.
"""

import os
import gc
import signal
import sys
import torch
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

from .config import LoRAConfig
from .utils import find_latest_jsonl, ensure_dir, is_gguf_file, resolve_model_from_gguf
from .training_visualizer import MetricsCallback, generate_training_plots


def cleanup_gpu_memory():
    """Clean up GPU memory by clearing cache and running garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


class LoRATrainer:
    """Manages LoRA fine-tuning of language models."""

    def __init__(self, config: LoRAConfig, dataset_path: Optional[str] = None):
        """
        Initialize LoRA trainer.

        Args:
            config: LoRA training configuration
            dataset_path: Path to training dataset JSONL file
        """
        self.config = config
        self.dataset_path = dataset_path
        if self.dataset_path:
            print(f"📁 Using dataset: {self.dataset_path}")
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.metrics_callback = None
        self._interrupted = False

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

        # Ensure output directory exists
        ensure_dir(config.output_dir)

        print("Initialized LoRATrainer:")
        print(f"  - Model: {config.model_id}")
        print(f"  - Output: {config.output_dir}")
        print(f"  - LoRA rank: {config.lora_r}")
        print(f"  - 4-bit quantization: {config.use_4bit}")

    def _signal_handler(self, signum, frame):
        """Handle interruption signals (Ctrl+C)."""
        if not self._interrupted:
            self._interrupted = True
            print("\n\n⚠️  Training interrupted by user! Cleaning up...")
            print("   Press Ctrl+C again to force quit (not recommended)")
            self.cleanup()
            sys.exit(0)
        else:
            print("\n\n❌ Force quit! GPU memory may not be released properly.")
            sys.exit(1)

    def cleanup(self):
        """Clean up resources and free GPU memory."""
        print("\n🧹 Releasing GPU resources...")
        
        # Delete trainer
        if self.trainer is not None:
            del self.trainer
            self.trainer = None
        
        # Delete model
        if self.model is not None:
            # Move model to CPU before deleting (helps with cleanup)
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            del self.model
            self.model = None
        
        # Delete tokenizer
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clean up GPU memory
        cleanup_gpu_memory()
        
        print("   ✓ GPU resources released")

    def discover_dataset(self, data_dir: str = "outputs") -> str:
        """
        Find the most recent dataset file.

        Args:
            data_dir: Base directory to search for datasets

        Returns:
            Path to the most recent dataset

        Raises:
            FileNotFoundError: If no dataset is found
        """
        if self.dataset_path and os.path.exists(self.dataset_path):
            return self.dataset_path

        latest = find_latest_jsonl(data_dir)
        if not latest:
            raise FileNotFoundError(
                f"No JSONL dataset found in {data_dir}. "
                "Please specify dataset_path or ensure a dataset exists."
            )

        self.dataset_path = latest
        print(f"📁 Using dataset: {self.dataset_path}")
        return latest

    def load_and_format_dataset(self, test_size: float = 0.05) -> tuple[Dataset, Dataset]:
        """
        Load and format the training dataset.
        Supports both local JSONL files and HuggingFace datasets.

        Args:
            test_size: Fraction of data to use for validation

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        print("\n📊 Loading dataset...")

        # Check if dataset_path is a HuggingFace dataset ID
        # HuggingFace dataset IDs typically have format "username/dataset-name"
        # and don't have file extensions or exist as local files
        is_hf_dataset = (
            self.dataset_path 
            and '/' in self.dataset_path 
            and not self.dataset_path.endswith(('.jsonl', '.json'))
            and not os.path.isfile(self.dataset_path)
            and not os.path.isdir(self.dataset_path)
        )
        
        if is_hf_dataset:
            # Try loading as HuggingFace dataset
            print(f"  - Loading HuggingFace dataset: {self.dataset_path}")
            try:
                raw_dataset = load_dataset(self.dataset_path, split="train", token=self.config.hf_token)
                print(f"  - Loaded {len(raw_dataset)} examples from HuggingFace")
            except Exception as e:
                print(f"  ⚠️  Failed to load as HuggingFace dataset: {e}")
                print(f"  - Falling back to local file...")
                raw_dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        else:
            # Load from local JSONL file
            raw_dataset = load_dataset("json", data_files=self.dataset_path, split="train")
        
        print(f"  - Loaded {len(raw_dataset)} examples")

        # Initialize tokenizer
        # Resolve model ID if it's a GGUF file
        model_id = self.config.model_id
        if is_gguf_file(model_id):
            try:
                resolved_model_id, _ = resolve_model_from_gguf(model_id)
                model_id = resolved_model_id
            except Exception:
                # If resolution fails, use original (will fail later with better error)
                pass
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, token=self.config.hf_token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Format dataset for training
        def format_example(example):
            """Convert JSONL record to formatted text."""
            # Handle both old and new format
            if "instruction" in example:
                # New format: instruction/input/output
                instruction = example.get("instruction", "")
                context = example.get("input", "")
                output = example.get("output", "")

                # Build user message
                if context:
                    user_message = f"{instruction}\n\nContext: {context}"
                else:
                    user_message = instruction

                messages = [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": output},
                ]
            else:
                # Old format: input (list of messages) / output
                messages = example.get("input", [])
                output = example.get("output", "")
                messages = list(messages) + [{"role": "assistant", "content": output}]

            # Apply chat template
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            
            # Truncate to max_seq_length if specified (to save memory)
            if self.config.max_seq_length:
                # Tokenize to check length
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.config.max_seq_length:
                    # Truncate tokens and decode back
                    truncated_tokens = tokens[:self.config.max_seq_length]
                    text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)

            return {"text": text}

        formatted_dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

        # Split into train/validation
        split = formatted_dataset.train_test_split(test_size=test_size, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

        print(f"  - Training examples: {len(train_dataset)}")
        print(f"  - Validation examples: {len(eval_dataset)}")

        return train_dataset, eval_dataset

    def setup_model(self) -> AutoModelForCausalLM:
        """
        Load and configure the base model with optional quantization.
        Supports both Hugging Face model IDs and GGUF files.

        Returns:
            Loaded model
        """
        model_id = self.config.model_id
        
        # Check if model_id is a GGUF file
        if is_gguf_file(model_id):
            print(f"\n📦 Detected GGUF file: {model_id}")
            print("  ℹ️  GGUF files are optimized for inference and cannot be used directly for training.")
            print("  🔍 Attempting to resolve original Hugging Face model from GGUF metadata...")
            
            try:
                resolved_model_id, metadata = resolve_model_from_gguf(model_id)
                print(f"  ✓ Resolved base model: {resolved_model_id}")
                if metadata:
                    print(f"  ℹ️  GGUF metadata: {metadata.get('name', 'N/A')} ({metadata.get('architecture', 'N/A')})")
                print(f"  ⚠️  Note: Training will use the original Hugging Face model weights, not the quantized GGUF file.")
                model_id = resolved_model_id
            except Exception as e:
                print(f"  ❌ Failed to resolve model from GGUF: {e}")
                raise ValueError(
                    f"Cannot use GGUF file {model_id} for training. "
                    f"Please specify the original Hugging Face model ID in config.yaml "
                    f"(e.g., 'model_id: google/gemma-2-2b-it' instead of the GGUF file path)."
                )
        else:
            print(f"\n🤖 Loading model: {model_id}")

        # Configure quantization if enabled
        bnb_config = None
        if self.config.use_4bit:
            print("  - Using 4-bit quantization (QLoRA)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        # Multi-GPU with quantization: each rank loads on its specific GPU
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        
        if local_rank >= 0:
            # DDP: pin model to the GPU corresponding to this process rank
            # CRITICAL: use local_rank directly, not torch.cuda.current_device()
            device_map = {"": local_rank}
            max_memory = {local_rank: "100%"}  # Use 100% of GPU memory, no CPU offload
            print(f"  - DDP rank {local_rank}: loading on cuda:{local_rank}")
        else:
            # Single GPU: force GPU-only, no CPU offload
            # Use "cuda:0" explicitly instead of "auto" to prevent RAM fallback
            device_map = "cuda:0"
            max_memory = {0: "100%"}  # Use 100% of GPU 0, no CPU offload
            print("  - Single GPU mode: forcing GPU-only (no CPU offload)")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            max_memory=max_memory if local_rank >= 0 or torch.cuda.is_available() else None,
            dtype=torch.bfloat16,  # Use dtype instead of deprecated torch_dtype
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            use_cache=False,  # Disable KV cache during training to save memory
            token=self.config.hf_token,
        )

        print("  ✓ Model loaded")
        return self.model

    def _detect_target_modules(self) -> list[str]:
        """
        Auto-detect target modules for LoRA based on model architecture.
        
        Returns:
            List of target module names
        """
        model_id_lower = self.config.model_id.lower()
        
        # Model-specific defaults
        if "granite" in model_id_lower:
            # Granite models use similar architecture to Mistral/Llama
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "gemma" in model_id_lower:
            # Gemma models
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "llama" in model_id_lower or "mistral" in model_id_lower:
            # Llama/Mistral models
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "qwen" in model_id_lower:
            # Qwen models
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            # Generic detection: find linear layers in attention modules
            print("  🔍 Auto-detecting target modules from model architecture...")
            target_modules = []
            
            # Common attention module patterns
            attention_patterns = [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Standard attention
                "query", "key", "value", "dense",  # Alternative naming
                "qkv", "out_proj",  # Other patterns
            ]
            
            # Check model structure
            for name, module in self.model.named_modules():
                module_type = type(module).__name__
                if "Linear" in module_type or "Linear4bit" in module_type:
                    # Check if it's an attention-related module
                    for pattern in attention_patterns:
                        if pattern in name.lower() and name not in target_modules:
                            target_modules.append(name)
                            break
            
            if target_modules:
                print(f"  ✓ Detected {len(target_modules)} target modules")
                return target_modules[:8]  # Limit to reasonable number
            else:
                # Fallback: use all linear layers in attention blocks
                print("  ⚠️  Could not auto-detect, using common defaults")
                return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def setup_lora_config(self) -> LoraConfig:
        """
        Create LoRA configuration.

        Returns:
            LoRA configuration object
        """
        # Determine target modules
        if self.config.target_modules:
            target_modules = self.config.target_modules
            print(f"\n🔧 LoRA Configuration:")
            print(f"  - Target modules (from config): {target_modules}")
        else:
            target_modules = self._detect_target_modules()
            print(f"\n🔧 LoRA Configuration:")
            print(f"  - Target modules (auto-detected): {target_modules}")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        print(f"  - Rank (r): {self.config.lora_r}")
        print(f"  - Alpha: {self.config.lora_alpha}")
        print(f"  - Dropout: {self.config.lora_dropout}")

        return lora_config

    def setup_trainer(self, train_dataset: Dataset, eval_dataset: Dataset) -> SFTTrainer:
        """
        Create and configure the SFT trainer.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset

        Returns:
            Configured trainer
        """
        print("\n⚙️  Configuring trainer...")

        # Create LoRA config
        peft_config = self.setup_lora_config()

        # Create metrics callback for visualization
        self.metrics_callback = MetricsCallback(self.config.output_dir)

        # Detect multi-GPU configuration
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Create training arguments with multi-GPU DDP support
        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            bf16=True,
            packing=False,  # Disable packing to have more control over memory usage
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            # Memory optimizations
            gradient_checkpointing=True,  # Trade compute for memory (~30-40% memory savings)
            optim="adamw_torch_fused",  # Fused optimizer is more memory efficient
            dataloader_pin_memory=False,  # Disable pin memory to save RAM
            dataloader_num_workers=0,  # Single-threaded data loading to save memory
            dataloader_prefetch_factor=None,
            remove_unused_columns=False,
            group_by_length=False,
            max_grad_norm=1.0,
            # Multi-GPU DDP optimizations (accelerate handles automatically)
            ddp_find_unused_parameters=False,  # Faster with LoRA
            ddp_bucket_cap_mb=25,  # Optimize gradient sync
            # Prevent evaluation slowdown
            eval_do_concat_batches=False,  # Don't accumulate eval batches
        )
        
        # Print GPU info
        if world_size > 1:
            if local_rank == 0 or local_rank == -1:
                print(f"\n🚀 Multi-GPU Training (DDP):")
                print(f"   - World size: {world_size} GPUs")
                print(f"   - Per-GPU batch: {self.config.batch_size}")
                print(f"   - Global batch: {self.config.batch_size * world_size * self.config.gradient_accumulation_steps}")
                print(f"   - Gradient sync every {self.config.gradient_accumulation_steps} steps")
        else:
            print(f"\n💻 Single GPU Training:")
            print(f"   - Batch: {self.config.batch_size}")
            print(f"   - Effective: {self.config.batch_size * self.config.gradient_accumulation_steps}")

        # Create trainer
        # Note: Different TRL versions have different APIs
        # max_seq_length is handled in dataset formatting, not here
        # Build base kwargs
        trainer_kwargs = {
            "model": self.model,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "peft_config": peft_config,
            "args": training_args,
            "callbacks": [self.metrics_callback],
        }
        
        # Try different API variations
        try:
            # Try new API (trl 0.24+) - tokenizer auto-loaded from model
            self.trainer = SFTTrainer(**trainer_kwargs)
        except (TypeError, ValueError) as e1:
            print(f"  ⚠️  Attempt 1 failed: {e1}")
            try:
                # Try with explicit tokenizer (some versions require it)
                trainer_kwargs["tokenizer"] = self.tokenizer
                self.trainer = SFTTrainer(**trainer_kwargs)
            except (TypeError, ValueError) as e2:
                print(f"  ⚠️  Attempt 2 failed: {e2}")
                # Try older API with dataset_text_field
                trainer_kwargs.pop("tokenizer", None)
                trainer_kwargs["dataset_text_field"] = "text"
                self.trainer = SFTTrainer(**trainer_kwargs)

        print("  ✓ Trainer configured")
        return self.trainer

    def train(self) -> Dict[str, Any]:
        """
        Execute the training process.

        Returns:
            Training metrics
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not initialized. Call setup() first.")

        print("\n🚀 Starting training...\n")

        try:
            # Train
            train_result = self.trainer.train()

            # Save LoRA adapter only (not the tokenizer)
            print("\n💾 Saving LoRA adapter...")
            
            # Save only the adapter (not full model with tokenizer)
            # This saves ~3.5 MB per training run
            self.model.save_pretrained(self.config.output_dir)
            
            # Save adapter config
            print("   ✓ LoRA adapter saved")
            print(f"   ℹ️  Tokenizer not saved (saves 3.5 MB)")
            print(f"   ℹ️  Use base model tokenizer: {self.config.model_id}")

            print("\n✅ Training complete!")
            print(f"📁 LoRA adapter saved to: {self.config.output_dir}")

            # Generate visualization plots
            if self.metrics_callback:
                metrics_history = self.metrics_callback.get_metrics()
                
                # Check if we have enough data for meaningful plots
                if len(metrics_history.get('loss', [])) < 2:
                    print("\n⚠️  Not enough training data for plots (need at least 2 logged steps)")
                    print(f"   Current: {len(metrics_history.get('loss', []))} logged steps")
                    print("   Tip: Decrease 'logging_steps' in config.yaml for more frequent logging")
                else:
                    try:
                        generate_training_plots(metrics_history, self.config.output_dir)
                    except Exception as e:
                        print(f"\n⚠️  Failed to generate plots: {e}")
                        print("   Training completed successfully, but plot generation failed.")

            return train_result.metrics
        
        finally:
            # Always clean up GPU memory after training
            print("\n🧹 Cleaning up GPU memory...")
            cleanup_gpu_memory()

    def run_full_training(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Args:
            dataset_path: Optional path to dataset (uses discovery if None)

        Returns:
            Training metrics
        """
        try:
            # Set dataset path if provided
            if dataset_path:
                self.dataset_path = dataset_path

            # Discover dataset if not set
            if not self.dataset_path:
                self.discover_dataset()

            # Load and format dataset
            train_dataset, eval_dataset = self.load_and_format_dataset()

            # Setup model
            self.setup_model()

            # Setup trainer
            self.setup_trainer(train_dataset, eval_dataset)

            # Train
            metrics = self.train()

            return metrics
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user!")
            raise
        
        except Exception as e:
            print(f"\n\n❌ Training failed with error: {e}")
            raise
        
        finally:
            # Ensure cleanup happens even on error
            if not self._interrupted:
                self.cleanup()


def merge_and_save_full_model(
    base_model_id: str, lora_adapter_path: str, output_path: str, use_4bit: bool = True
) -> str:
    """
    Merge LoRA adapter with base model and save the complete model.

    Args:
        base_model_id: Hugging Face model ID of the base model
        lora_adapter_path: Path to the LoRA adapter
        output_path: Path to save the merged model
        use_4bit: Whether the model was trained with 4-bit quantization

    Returns:
        Path to the saved model
    """
    print("🔄 Merging LoRA adapter with base model...")
    print(f"  - Base model: {base_model_id}")
    print(f"  - Adapter: {lora_adapter_path}")

    # Configure quantization
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapter
    model_with_lora = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # Merge adapter into base model
    print("  - Merging...")
    merged_model = model_with_lora.merge_and_unload()

    # Save merged model
    ensure_dir(output_path)
    print(f"  - Saving to {output_path}...")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"✅ Merged model saved to: {output_path}")
    return output_path

