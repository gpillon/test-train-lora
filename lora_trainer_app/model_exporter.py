"""
Model export functionality for LoRA trainers.
Supports exporting merged models to Hugging Face format and GGUF.
"""

import json
import logging
from pathlib import Path
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import gguf

logger = logging.getLogger(__name__)


class ModelExporter:
    """Export LoRA models to various formats."""
    
    def __init__(self, base_model_id: str, adapter_path: str, output_dir: str):
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def export_merged_model(self, model_name: str = "merged_model") -> str:
        """
        Export LoRA adapter merged with base model to Hugging Face format.
        
        Args:
            model_name: Name for the exported model directory
            
        Returns:
            Path to the exported model directory
        """
        logger.info(f"🔄 Merging LoRA adapter with base model...")
        logger.info(f"  - Base model: {self.base_model_id}")
        logger.info(f"  - Adapter: {self.adapter_path}")
        
        try:
            # Load base model and tokenizer
            logger.info("📥 Loading base model...")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id,
                use_fast=False  # Avoid tiktoken issues
            )
            
            # Load LoRA adapter
            logger.info("📥 Loading LoRA adapter...")
            model = PeftModel.from_pretrained(base_model, self.adapter_path)
            
            # Merge adapter with base model
            logger.info("🔗 Merging adapter with base model...")
            merged_model = model.merge_and_unload()
            
            # Save merged model
            output_path = self.output_dir / model_name
            logger.info(f"💾 Saving merged model to: {output_path}")
            
            merged_model.save_pretrained(
                output_path,
                save_safetensors=True,
                max_shard_size="5GB"
            )
            
            tokenizer.save_pretrained(output_path)
            
            # Save model info
            model_info = {
                "base_model": self.base_model_id,
                "adapter_path": self.adapter_path,
                "merged_at": str(Path.cwd()),
                "model_type": "merged_lora",
                "architecture": merged_model.config.architectures[0] if merged_model.config.architectures else "unknown"
            }
            
            with open(output_path / "model_info.json", "w") as f:
                json.dump(model_info, f, indent=2)
            
            logger.info(f"✅ Merged model saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error merging model: {e}")
            raise
    
    def export_to_gguf(self, model_path: str, output_file: str = None) -> str:
        """
        Export merged model to GGUF format.
        
        Args:
            model_path: Path to the merged model directory
            output_file: Output GGUF file path (optional)
            
        Returns:
            Path to the exported GGUF file
        """
        if output_file is None:
            model_name = Path(model_path).name
            output_file = self.output_dir / f"{model_name}.gguf"
        else:
            output_file = Path(output_file)
            
        logger.info(f"🔄 Converting to GGUF format...")
        logger.info(f"  - Model: {model_path}")
        logger.info(f"  - Output: {output_file}")
        
        try:
            # Load the merged model
            logger.info("📥 Loading merged model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Create GGUF writer
            logger.info("📝 Creating GGUF file...")
            writer = gguf.GGUFWriter(str(output_file), model.config.architectures[0] if model.config.architectures else "llama")
            
            # Add model metadata
            writer.add_name(Path(model_path).name)
            writer.add_description(f"Merged LoRA model from {self.base_model_id}")
            # Note: add_architecture() doesn't take parameters in newer versions
            # writer.add_architecture(model.config.architectures[0] if model.config.architectures else "llama")
            
            # Add model parameters
            writer.add_uint32("vocab_size", model.config.vocab_size)
            writer.add_uint32("hidden_size", model.config.hidden_size)
            writer.add_uint32("intermediate_size", model.config.intermediate_size)
            writer.add_uint32("num_attention_heads", model.config.num_attention_heads)
            writer.add_uint32("num_hidden_layers", model.config.num_hidden_layers)
            writer.add_uint32("max_position_embeddings", model.config.max_position_embeddings)
            
            # Add tokenizer info
            if hasattr(tokenizer, 'vocab_size'):
                writer.add_uint32("vocab_size", tokenizer.vocab_size)
            
            # Write model weights
            logger.info("💾 Writing model weights...")
            self._write_model_weights(writer, model)
            
            # Write tokenizer
            logger.info("💾 Writing tokenizer...")
            self._write_tokenizer(writer, tokenizer)
            
            # Finalize
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            
            logger.info(f"✅ GGUF model saved to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"❌ Error converting to GGUF: {e}")
            raise
    
    def _write_model_weights(self, writer: gguf.GGUFWriter, model: torch.nn.Module):
        """Write model weights to GGUF writer."""
        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            # Skip non-tensor items
            if not isinstance(tensor, torch.Tensor):
                continue
                
            # Keep original dtype to preserve size
            # Only convert to float32 if necessary for compatibility
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()  # bfloat16 not supported in GGUF
            # Keep float16 as is to preserve size
            
            # Add tensor to writer
            try:
                writer.add_tensor(name, tensor.detach().cpu().numpy())
            except Exception as e:
                logger.warning(f"⚠️  Could not write tensor {name}: {e}")
    
    def _write_tokenizer(self, writer: gguf.GGUFWriter, tokenizer):
        """Write tokenizer information to GGUF writer."""
        try:
            # Add vocabulary
            if hasattr(tokenizer, 'get_vocab'):
                vocab = tokenizer.get_vocab()
                for token, idx in vocab.items():
                    writer.add_token(token, idx)
            
            # Add special tokens
            if hasattr(tokenizer, 'special_tokens_map'):
                special_tokens = tokenizer.special_tokens_map
                for token_type, token in special_tokens.items():
                    if token and token in tokenizer.get_vocab():
                        writer.add_special_token(token, tokenizer.get_vocab()[token])
            
        except Exception as e:
            logger.warning(f"⚠️  Could not write tokenizer info: {e}")
    
    def export_complete(self, model_name: str = "merged_model", export_gguf: bool = True) -> Dict[str, str]:
        """
        Complete export process: merge LoRA and optionally export to GGUF.
        
        Args:
            model_name: Name for the exported model
            export_gguf: Whether to also export to GGUF format
            
        Returns:
            Dictionary with paths to exported files
        """
        results = {}
        
        # Step 1: Merge LoRA with base model
        merged_path = self.export_merged_model(model_name)
        results["merged_model"] = merged_path
        
        # Step 2: Export to GGUF (optional)
        if export_gguf:
            gguf_path = self.export_to_gguf(merged_path)
            results["gguf_model"] = gguf_path
        
        return results


def export_lora_model(
    base_model_id: str,
    adapter_path: str, 
    output_dir: str,
    model_name: str = "merged_model",
    export_gguf: bool = True
) -> Dict[str, str]:
    """
    Convenience function to export LoRA model.
    
    Args:
        base_model_id: Hugging Face model ID
        adapter_path: Path to LoRA adapter
        output_dir: Output directory
        model_name: Name for exported model
        export_gguf: Whether to export GGUF format
        
    Returns:
        Dictionary with exported file paths
    """
    exporter = ModelExporter(base_model_id, adapter_path, output_dir)
    return exporter.export_complete(model_name, export_gguf)
