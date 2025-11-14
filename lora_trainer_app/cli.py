#!/usr/bin/env python3
"""
CLI interface for LoRA Trainer Application using Typer.
Automatically generates detailed help from type hints and docstrings.
"""

import logging
from pathlib import Path
from typing import Optional, Annotated, TYPE_CHECKING
from enum import Enum

import typer
from rich.console import Console
from rich.logging import RichHandler

# Lazy imports - only import heavy modules when needed
if TYPE_CHECKING:
    from .config_loader import ConfigManager
    from .dataset_generator import DatasetGenerator
    from .lora_trainer import LoRATrainer
    from .inference import LoRAInference
    from .model_exporter import ModelExporter


# Initialize Typer app and Rich console
app = typer.Typer(
    name="lora-trainer",
    help="🚀 LoRA Trainer Application - Dataset Generation, Training, Inference, and Export",
    add_completion=True,  # Enable shell completion support
    rich_markup_mode="rich",
)
console = Console()


class LogLevel(str, Enum):
    """Available logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def setup_logging(level: LogLevel = LogLevel.INFO):
    """Configure rich logging for the application."""
    logging.basicConfig(
        level=level.value,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)]
    )


@app.command(name="test-models")
def cmd_test_models(
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c",
        help="Path to config.yaml file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    )] = None,
    log_level: Annotated[LogLevel, typer.Option(
        "--log-level", "-l",
        help="Set logging level"
    )] = LogLevel.INFO,
):
    """
    🧪 Test connectivity to all configured API models.
    
    This command checks if all configured models in config.yaml are accessible
    and responding correctly. It's useful for debugging API configurations.
    """
    setup_logging(log_level)
    
    try:
        # Lazy import
        from .config_loader import ConfigManager
        from .api_client import test_all_models
        
        config_mgr = ConfigManager(str(config) if config else None)
        models = config_mgr.load_models()
        
        if not models:
            console.print("❌ No models configured!", style="bold red")
            raise typer.Exit(1)
        
        console.print(f"\n🔍 Testing {len(models)} configured models...\n")
        results = test_all_models(models)
        
        # Display results
        success_count = sum(1 for r in results.values() if r["success"])
        console.print(f"\n✅ {success_count}/{len(results)} models working correctly")
        
        if success_count < len(results):
            raise typer.Exit(1)
            
    except FileNotFoundError as e:
        console.print(f"❌ {e}", style="bold red")
        console.print("\nPlease create a config.yaml file or specify --config <path>")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        raise typer.Exit(1)


@app.command(name="generate-data")
def cmd_generate_data(
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c",
        help="Path to config.yaml file"
    )] = None,
    batch_size: Annotated[Optional[int], typer.Option(
        "--batch-size",
        help="Number of examples to generate (overrides config)"
    )] = None,
    max_tokens: Annotated[Optional[int], typer.Option(
        "--max-tokens",
        help="Maximum tokens per response (overrides config)"
    )] = None,
    temperature: Annotated[Optional[float], typer.Option(
        "--temperature",
        help="Generation temperature 0.0-2.0 (overrides config)"
    )] = None,
    output_dir: Annotated[Optional[Path], typer.Option(
        "--output-dir",
        help="Output directory for dataset (overrides config)"
    )] = None,
    seed: Annotated[Optional[int], typer.Option(
        "--seed",
        help="Random seed for reproducibility"
    )] = None,
    clean: Annotated[bool, typer.Option(
        "--clean",
        help="Clean and validate dataset after generation"
    )] = False,
    log_level: Annotated[LogLevel, typer.Option(
        "--log-level", "-l",
        help="Set logging level"
    )] = LogLevel.INFO,
):
    """
    📝 Generate training dataset using configured API models.
    
    This command generates a dataset in JSONL format by querying multiple
    API models. The dataset follows the "Split Personality LoRA" format
    with Analyst, Creative, and Consensus personas.
    
    Examples:
        lora-trainer generate-data --batch-size 100
        lora-trainer generate-data --batch-size 50 --temperature 0.8 --clean
    """
    setup_logging(log_level)
    
    try:
        # Lazy import
        from .config_loader import ConfigManager
        from .dataset_generator import DatasetGenerator, clean_dataset
        
        config_mgr = ConfigManager(str(config) if config else None)
        dataset_config = config_mgr.get_dataset_config()
        
        # Apply overrides
        if batch_size:
            dataset_config.batch_size = batch_size
        if max_tokens:
            dataset_config.max_tokens = max_tokens
        if temperature:
            dataset_config.temperature = temperature
        if output_dir:
            dataset_config.output_dir = str(output_dir)
        if seed:
            dataset_config.seed = seed
        
        models = config_mgr.load_models()
        generator = DatasetGenerator(models, dataset_config)
        
        console.print("📝 Generating dataset...", style="bold green")
        output_file = generator.generate()
        
        console.print(f"\n✅ Dataset generated: {output_file}", style="bold green")
        
        if clean:
            console.print("\n🧹 Cleaning dataset...")
            cleaned_file = clean_dataset(output_file)
            console.print(f"✅ Cleaned dataset: {cleaned_file}", style="bold green")
        
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        raise typer.Exit(1)


@app.command(name="train-model")
def cmd_train_model(
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c",
        help="Path to config.yaml file"
    )] = None,
    dataset: Annotated[Optional[str], typer.Option(
        "--dataset",
        help="Path to training dataset JSONL file or HuggingFace dataset ID (e.g., 'DeepMount00/pii-masking-ita')",
    )] = None,
    model_id: Annotated[Optional[str], typer.Option(
        "--model-id",
        help="Hugging Face model ID to fine-tune"
    )] = None,
    epochs: Annotated[Optional[int], typer.Option(
        "--epochs",
        help="Number of training epochs"
    )] = None,
    batch_size: Annotated[Optional[int], typer.Option(
        "--batch-size",
        help="Training batch size per device"
    )] = None,
    learning_rate: Annotated[Optional[float], typer.Option(
        "--learning-rate",
        help="Learning rate for training"
    )] = None,
    lora_r: Annotated[Optional[int], typer.Option(
        "--lora-r",
        help="LoRA rank (attention dimension)"
    )] = None,
    lora_alpha: Annotated[Optional[int], typer.Option(
        "--lora-alpha",
        help="LoRA alpha (scaling factor)"
    )] = None,
    no_4bit: Annotated[bool, typer.Option(
        "--no-4bit",
        help="Disable 4-bit quantization (use full precision)"
    )] = False,
    output_dir: Annotated[Optional[Path], typer.Option(
        "--output-dir",
        help="Output directory for trained model"
    )] = None,
    merge: Annotated[bool, typer.Option(
        "--merge",
        help="Merge LoRA adapter with base model after training"
    )] = False,
    log_level: Annotated[LogLevel, typer.Option(
        "--log-level", "-l",
        help="Set logging level"
    )] = LogLevel.INFO,
):
    """
    🎯 Train a LoRA model on your dataset.
    
    This command fine-tunes a language model using LoRA (Low-Rank Adaptation)
    with optional 4-bit quantization (QLoRA). Supports single and multi-GPU training.
    
    Examples:
        lora-trainer train-model --dataset data/dataset.jsonl --epochs 3
        lora-trainer train-model --dataset data/dataset.jsonl --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --merge
    """
    setup_logging(log_level)
    
    try:
        # Lazy import
        from .config_loader import ConfigManager
        from .lora_trainer import LoRATrainer
        
        config_mgr = ConfigManager(str(config) if config else None)
        lora_config = config_mgr.get_lora_config()
        
        # Get dataset_path from config if not provided via CLI
        lora_cfg = config_mgr.config.get("lora", {})
        dataset_path = dataset if dataset else lora_cfg.get("dataset_path")
        
        # Apply overrides
        if model_id:
            lora_config.model_id = model_id
        if epochs:
            lora_config.num_epochs = epochs
        if batch_size:
            lora_config.batch_size = batch_size
        if learning_rate:
            lora_config.learning_rate = learning_rate
        if lora_r:
            lora_config.lora_r = lora_r
        if lora_alpha:
            lora_config.lora_alpha = lora_alpha
        if no_4bit:
            lora_config.use_4bit = False
        if output_dir:
            lora_config.output_dir = str(output_dir)
        
        # dataset_path can be None - trainer will use discover_dataset() if needed
        if dataset_path:
            console.print(f"📁 Using dataset: {dataset_path}", style="bold blue")
        else:
            console.print("ℹ️  No dataset specified, will auto-discover from outputs/", style="bold yellow")
        
        console.print("🚀 Starting LoRA training...", style="bold green")
        
        trainer = LoRATrainer(lora_config, dataset_path=dataset_path)
        trainer.run_full_training()
        
        console.print("\n✅ Training complete!", style="bold green")
        console.print(f"📁 Model saved to: {lora_config.output_dir}")
        
        if merge:
            console.print("\n🔗 Merging LoRA with base model...")
            # Merge logic here
            
    except Exception as e:
        console.print(f"❌ Training failed: {e}", style="bold red")
        raise typer.Exit(1)


@app.command(name="inference")
def cmd_inference(
    model_id: Annotated[str, typer.Option(
        "--model-id",
        help="Base model ID or path to merged model"
    )],
    config: Annotated[Optional[Path], typer.Option(
        "--config", "-c",
        help="Path to config.yaml file"
    )] = None,
    adapter: Annotated[Optional[Path], typer.Option(
        "--adapter",
        help="Path to LoRA adapter directory"
    )] = None,
    merged: Annotated[bool, typer.Option(
        "--merged",
        help="Model is already merged (no adapter needed)"
    )] = False,
    prompt: Annotated[Optional[str], typer.Option(
        "--prompt",
        help="Prompt for single inference"
    )] = None,
    interactive: Annotated[bool, typer.Option(
        "--interactive", "-i",
        help="Run interactive chat mode"
    )] = False,
    system_prompt: Annotated[Optional[str], typer.Option(
        "--system-prompt",
        help="System prompt to guide the model"
    )] = None,
    max_tokens: Annotated[Optional[int], typer.Option(
        "--max-tokens",
        help="Maximum tokens to generate"
    )] = None,
    temperature: Annotated[Optional[float], typer.Option(
        "--temperature",
        help="Sampling temperature (0.0-2.0)"
    )] = None,
    top_p: Annotated[Optional[float], typer.Option(
        "--top-p",
        help="Nucleus sampling parameter"
    )] = None,
    no_4bit: Annotated[bool, typer.Option(
        "--no-4bit",
        help="Disable 4-bit quantization"
    )] = False,
    log_level: Annotated[LogLevel, typer.Option(
        "--log-level", "-l",
        help="Set logging level"
    )] = LogLevel.INFO,
):
    """
    💬 Run inference with a trained LoRA model.
    
    This command loads a model (with or without LoRA adapter) and generates
    responses. Supports both single-prompt and interactive chat modes.
    
    Examples:
        lora-trainer inference --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter ./lora_output --prompt "Hello!"
        lora-trainer inference --model-id ./merged_model --merged --interactive
    """
    setup_logging(log_level)
    
    try:
        # Lazy import
        from .config_loader import ConfigManager
        from .inference import LoRAInference, run_interactive_chat
        
        config_mgr = ConfigManager(str(config) if config else None)
        inf_config = config_mgr.get_inference_config()
        
        # Get config values with defaults
        system_message = system_prompt or inf_config.get("system_prompt", "You are a helpful assistant.")
        max_new_tokens = max_tokens or inf_config.get("max_tokens", 256)
        gen_temperature = temperature if temperature is not None else inf_config.get("temperature", 0.7)
        gen_top_p = top_p if top_p is not None else inf_config.get("top_p", 0.9)
        use_4bit_flag = not no_4bit
        
        # Handle merged model case - if merged, use model_id as base and no adapter
        adapter_path = None if merged else (str(adapter) if adapter else None)
        
        console.print("🤖 Starting inference...", style="bold green")
        
        if interactive:
            run_interactive_chat(
                base_model_id=model_id,
                adapter_path=adapter_path,
                system_message=system_message,
                use_4bit=use_4bit_flag,
                max_new_tokens=max_new_tokens,
                temperature=gen_temperature,
                top_p=gen_top_p,
            )
        elif prompt:
            # Initialize inference engine
            engine = LoRAInference(
                base_model_id=model_id,
                adapter_path=adapter_path,
                use_4bit=use_4bit_flag,
            )
            
            # Generate response
            response = engine.chat_simple(
                user_message=prompt,
                system_message=system_message,
                max_new_tokens=max_new_tokens,
                temperature=gen_temperature,
                top_p=gen_top_p,
            )
            console.print(f"\n[bold cyan]Response:[/bold cyan]\n{response}")
            
            # Cleanup
            engine.cleanup()
        else:
            console.print("❌ Either --prompt or --interactive is required!", style="bold red")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"❌ Inference failed: {e}", style="bold red")
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command(name="export-model")
def cmd_export_model(
    output_dir: Annotated[Path, typer.Option(
        "--output-dir",
        help="Output directory for exported model"
    )],
    base_model: Annotated[Optional[str], typer.Option(
        "--base-model",
        help="Base model ID (required unless --gguf-only)"
    )] = None,
    adapter_path: Annotated[Optional[Path], typer.Option(
        "--adapter-path",
        help="Path to LoRA adapter (required unless --gguf-only)"
    )] = None,
    model_name: Annotated[str, typer.Option(
        "--model-name",
        help="Name for the exported model directory"
    )] = "merged_model",
    gguf: Annotated[bool, typer.Option(
        "--gguf",
        help="Also export to GGUF format"
    )] = False,
    gguf_only: Annotated[bool, typer.Option(
        "--gguf-only",
        help="Only export to GGUF (requires --merged-model-path)"
    )] = False,
    merged_model_path: Annotated[Optional[Path], typer.Option(
        "--merged-model-path",
        help="Path to existing merged model (for --gguf-only)"
    )] = None,
    log_level: Annotated[LogLevel, typer.Option(
        "--log-level", "-l",
        help="Set logging level"
    )] = LogLevel.INFO,
):
    """
    📦 Export LoRA model to merged HuggingFace format and/or GGUF.
    
    This command merges a LoRA adapter with its base model and exports it in
    various formats. GGUF format is compatible with llama.cpp for CPU inference.
    
    Examples:
        # Full export (merge + GGUF)
        lora-trainer export-model --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter-path ./lora_output --output-dir ./exports --gguf
        
        # Only merge (HuggingFace format)
        lora-trainer export-model --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter-path ./lora_output --output-dir ./exports
        
        # Only GGUF (from already merged model)
        lora-trainer export-model --gguf-only --merged-model-path ./exports/merged_model --output-dir ./exports
    """
    setup_logging(log_level)
    
    try:
        # Lazy import
        from .model_exporter import ModelExporter
        
        if gguf_only:
            if not merged_model_path:
                console.print("❌ --merged-model-path is required for --gguf-only export", style="bold red")
                raise typer.Exit(1)
            
            console.print("🔄 Exporting to GGUF...", style="bold green")
            console.print(f"  - Merged model: {merged_model_path}")
            console.print(f"  - Output dir: {output_dir}")
            
            exporter = ModelExporter("", "", str(output_dir))
            gguf_path = exporter.export_to_gguf(str(merged_model_path))
            
            console.print("\n✅ GGUF model exported!", style="bold green")
            console.print(f"📁 {gguf_path}")
            
        else:
            if not base_model or not adapter_path:
                console.print("❌ --base-model and --adapter-path are required for full export", style="bold red")
                raise typer.Exit(1)
            
            console.print("🔄 Exporting LoRA model...", style="bold green")
            console.print(f"  - Base model: {base_model}")
            console.print(f"  - Adapter: {adapter_path}")
            console.print(f"  - Output dir: {output_dir}")
            console.print(f"  - Export GGUF: {gguf}")
            
            exporter = ModelExporter(base_model, str(adapter_path), str(output_dir))
            results = exporter.export_complete(model_name, gguf)
            
            console.print("\n✅ Export completed!", style="bold green")
            console.print(f"📁 Merged model: {results['merged_model']}")
            if 'gguf_model' in results:
                console.print(f"📁 GGUF model: {results['gguf_model']}")
                
    except Exception as e:
        console.print(f"❌ Export failed: {e}", style="bold red")
        raise typer.Exit(1)


@app.command(name="clean")
def cmd_clean_dataset(
    input_file: Annotated[Path, typer.Argument(
        help="Input JSONL file to clean"
    )],
    output: Annotated[Optional[Path], typer.Option(
        "--output", "-o",
        help="Output JSONL file (default: input_cleaned.jsonl)"
    )] = None,
    log_level: Annotated[LogLevel, typer.Option(
        "--log-level", "-l",
        help="Set logging level"
    )] = LogLevel.INFO,
):
    """
    🧹 Clean and validate a dataset file.
    
    This command removes invalid records, duplicates, and malformed JSON from
    a dataset file. Useful for cleaning datasets before training.
    
    Examples:
        lora-trainer clean dataset.jsonl
        lora-trainer clean dataset.jsonl --output clean_dataset.jsonl
    """
    setup_logging(log_level)
    
    try:
        # Lazy import
        from .dataset_generator import clean_dataset
        
        console.print(f"🧹 Cleaning dataset: {input_file}", style="bold green")
        
        output_file = clean_dataset(str(input_file), str(output) if output else None)
        
        console.print(f"\n✅ Cleaned dataset saved to: {output_file}", style="bold green")
        
    except Exception as e:
        console.print(f"❌ Error: {e}", style="bold red")
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

