#!/usr/bin/env python3
"""
CLI interface for LoRA Trainer Application.
Single executable with subcommands.
"""

import sys
import argparse
import logging
from typing import Optional

from .config_loader import ConfigManager
from .dataset_generator import DatasetGenerator, clean_dataset
from .lora_trainer import LoRATrainer, merge_and_save_full_model
from .inference import LoRAInference, run_interactive_chat
from .api_client import test_all_models
from .model_exporter import ModelExporter


# Setup logging
def setup_logging(level: str = "INFO"):
    """Configure logging for the application."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_config_manager(args) -> ConfigManager:
    """Get configuration manager from CLI args."""
    config_path = getattr(args, "config", None)
    try:
        return ConfigManager(config_path)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nPlease create a config.yaml file or specify --config <path>")
        print(
            "You can find an example at: https://github.com/yourusername/lora-trainer/blob/main/config.yaml"
        )
        sys.exit(1)


def cmd_test_models(args):
    """Test connectivity to configured models."""
    setup_logging(args.log_level)

    print("🔌 Testing model connectivity...\n")

    config_mgr = get_config_manager(args)
    models = config_mgr.get_models()

    if not models:
        print("❌ No models configured or no API keys found!")
        print("\nPlease configure models in config.yaml and set API keys as environment variables.")
        return 1

    results = test_all_models(models)

    print("Results:")
    for model_name, status in results.items():
        print(f"  {model_name}: {status}")

    working = sum(1 for s in results.values() if "✅" in s)
    print(f"\n{working}/{len(models)} models working")

    return 0 if working > 0 else 1


def cmd_generate_data(args):
    """Generate training dataset."""
    setup_logging(args.log_level)

    print("📝 Generating dataset...\n")

    # Load configuration
    config_mgr = get_config_manager(args)
    models = config_mgr.get_models()

    if not models:
        print("❌ No models configured!")
        return 1

    print(f"Using {len(models)} model(s):")
    for model in models:
        print(f"  - {model.name}")
    print()

    # Prepare CLI overrides
    cli_overrides = {
        k: v for k, v in vars(args).items() if v is not None and k not in ["config", "clean"]
    }

    # Load dataset config with CLI overrides
    dataset_config = config_mgr.get_dataset_config(cli_overrides)

    # Create generator
    generator = DatasetGenerator(dataset_config, models)

    # Generate dataset
    try:
        count = generator.run_batch()

        if count > 0:
            print("\n📊 Validating dataset...")
            validation = generator.validate_output()

            if validation.get("exists"):
                print(f"  ✓ Total records: {validation['valid_records']}")
                print(f"  ✓ Persona distribution: {validation.get('persona_distribution', {})}")

                if args.clean:
                    print("\n🧹 Cleaning dataset...")
                    clean_count = clean_dataset(generator.output_path)
                    print(f"  ✓ {clean_count} clean records")

            return 0
        else:
            print("❌ Failed to generate dataset")
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_train_model(args):
    """Train LoRA model."""
    setup_logging(args.log_level)

    print("🚀 Starting LoRA training...\n")

    # Load configuration
    config_mgr = get_config_manager(args)

    # Prepare CLI overrides
    cli_overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in ["config", "dataset", "merge"]
    }

    # Load LoRA config with CLI overrides
    lora_config = config_mgr.get_lora_config(cli_overrides)

    # Create trainer
    trainer = LoRATrainer(lora_config, dataset_path=args.dataset)

    try:
        # Run training
        metrics = trainer.run_full_training()

        print("\n📈 Training Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Merge and save full model if requested
        if args.merge:
            print("\n🔄 Merging LoRA with base model...")
            output_path = lora_config.output_dir + "_merged"
            merge_and_save_full_model(
                base_model_id=lora_config.model_id,
                lora_adapter_path=lora_config.output_dir,
                output_path=output_path,
                use_4bit=lora_config.use_4bit,
            )

        return 0

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_inference(args):
    """Run inference with trained model."""
    setup_logging(args.log_level)

    print("🤖 Starting inference...\n")

    # Load configuration
    try:
        config_mgr = get_config_manager(args)
        inf_config = config_mgr.get_inference_config()
    except:
        inf_config = {}

    # Merge config with CLI args
    system_prompt = args.system_prompt or inf_config.get(
        "system_prompt", "You are a helpful assistant."
    )
    max_tokens = args.max_tokens or inf_config.get("max_new_tokens", 256)
    temperature = args.temperature or inf_config.get("temperature", 0.7)
    top_p = args.top_p or inf_config.get("top_p", 0.9)
    use_4bit = not args.no_4bit

    engine = None
    try:
        if args.interactive:
            # Run interactive chat
            run_interactive_chat(
                base_model_id=args.model_id,
                adapter_path=args.adapter if not args.merged else None,
                system_message=system_prompt,
                use_4bit=use_4bit,
            )
        else:
            # Single prompt inference
            if not args.prompt:
                print("❌ Error: --prompt is required for non-interactive mode")
                return 1

            # Initialize inference engine
            engine = LoRAInference(
                base_model_id=args.model_id,
                adapter_path=args.adapter if not args.merged else None,
                use_4bit=use_4bit,
            )

            # Generate response
            print(f"Prompt: {args.prompt}\n")
            print("Generating response...\n")

            response = engine.chat_simple(
                user_message=args.prompt,
                system_message=system_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            print("Response:")
            print("-" * 60)
            print(response)
            print("-" * 60)

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1
    
    finally:
        # Clean up GPU resources
        if engine is not None:
            print("\n🧹 Releasing GPU resources...")
            engine.cleanup()


def cmd_clean_dataset(args):
    """Clean a dataset file."""
    setup_logging(args.log_level)

    print(f"🧹 Cleaning dataset: {args.input}\n")

    try:
        count = clean_dataset(args.input, args.output)

        if count > 0:
            print(f"\n✅ Success! {count} valid records")
            return 0
        else:
            print("\n❌ No valid records found")
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def cmd_export_model(args):
    """Export LoRA model to merged format and/or GGUF."""
    setup_logging(args.log_level)
    
    try:
        if args.gguf_only:
            # Only export GGUF from existing merged model
            if not args.merged_model_path:
                print("❌ --merged-model-path is required for --gguf-only export")
                return 1
                
            print(f"🔄 Exporting existing merged model to GGUF...")
            print(f"  - Merged model: {args.merged_model_path}")
            print(f"  - Output dir: {args.output_dir}")
            
            exporter = ModelExporter("", "", args.output_dir)
            gguf_path = exporter.export_to_gguf(args.merged_model_path)
            
            print(f"✅ GGUF model exported to: {gguf_path}")
            return 0
        
        else:
            # Full export: merge LoRA + optionally export GGUF
            if not args.base_model or not args.adapter_path:
                print("❌ --base-model and --adapter-path are required for full export")
                return 1
                
            print(f"🔄 Exporting LoRA model...")
            print(f"  - Base model: {args.base_model}")
            print(f"  - Adapter: {args.adapter_path}")
            print(f"  - Output dir: {args.output_dir}")
            print(f"  - Model name: {args.model_name}")
            print(f"  - Export GGUF: {args.gguf}")
            
            exporter = ModelExporter(args.base_model, args.adapter_path, args.output_dir)
            results = exporter.export_complete(args.model_name, args.gguf)
            
            print(f"\n✅ Export completed!")
            print(f"📁 Merged model: {results['merged_model']}")
            if 'gguf_model' in results:
                print(f"📁 GGUF model: {results['gguf_model']}")
            
            return 0
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 1


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        prog="lora-trainer",
        description="LoRA Trainer Application - Dataset Generation, Training, and Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lora-trainer generate-data --batch-size 50
  lora-trainer train-model --dataset ./outputs/dataset.jsonl --epochs 3
  lora-trainer inference --model-id google/gemma-2-2b-it --adapter ./lora_output -i
  lora-trainer export-model --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --adapter-path lora_output/ --output-dir exports/ --gguf
  lora-trainer test-models
        """,
    )

    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # test-models command
    test_parser = subparsers.add_parser(
        "test-models", help="Test connectivity to configured models"
    )
    test_parser.add_argument("--config", "-c", type=str, help="Path to config.yaml file")
    test_parser.set_defaults(func=cmd_test_models)

    # generate-data command
    gen_parser = subparsers.add_parser("generate-data", help="Generate training dataset")
    gen_parser.add_argument("--config", "-c", type=str, help="Path to config.yaml file")
    gen_parser.add_argument("--batch-size", type=int, help="Number of examples to generate")
    gen_parser.add_argument("--max-tokens", type=int, help="Max tokens per response")
    gen_parser.add_argument("--temperature", type=float, help="Generation temperature")
    gen_parser.add_argument("--output-dir", type=str, help="Output directory")
    gen_parser.add_argument("--seed", type=int, help="Random seed")
    gen_parser.add_argument("--clean", action="store_true", help="Clean dataset after generation")
    gen_parser.set_defaults(func=cmd_generate_data)

    # train-model command
    train_parser = subparsers.add_parser("train-model", help="Train LoRA model")
    train_parser.add_argument("--config", "-c", type=str, help="Path to config.yaml file")
    train_parser.add_argument("--dataset", type=str, help="Path to dataset JSONL file")
    train_parser.add_argument("--model-id", type=str, help="Base model ID")
    train_parser.add_argument("--epochs", type=int, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--lora-r", type=int, help="LoRA rank")
    train_parser.add_argument("--lora-alpha", type=int, help="LoRA alpha")
    train_parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    train_parser.add_argument("--output-dir", type=str, help="Output directory")
    train_parser.add_argument(
        "--merge", action="store_true", help="Merge LoRA with base model after training"
    )
    train_parser.set_defaults(func=cmd_train_model)

    # inference command
    inf_parser = subparsers.add_parser("inference", help="Run inference with trained model")
    inf_parser.add_argument("--config", "-c", type=str, help="Path to config.yaml file")
    inf_parser.add_argument(
        "--model-id", type=str, required=True, help="Base model ID or path to merged model"
    )
    inf_parser.add_argument("--adapter", type=str, help="Path to LoRA adapter")
    inf_parser.add_argument("--merged", action="store_true", help="Model is already merged")
    inf_parser.add_argument("--prompt", type=str, help="Prompt for single inference")
    inf_parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive chat")
    inf_parser.add_argument("--system-prompt", type=str, help="System prompt")
    inf_parser.add_argument("--max-tokens", type=int, help="Max tokens to generate")
    inf_parser.add_argument("--temperature", type=float, help="Sampling temperature")
    inf_parser.add_argument("--top-p", type=float, help="Nucleus sampling parameter")
    inf_parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    inf_parser.set_defaults(func=cmd_inference)

    # export-model command
    export_parser = subparsers.add_parser("export-model", help="Export LoRA model to merged format and/or GGUF")
    export_parser.add_argument("--base-model", type=str, help="Base model ID (required unless --gguf-only)")
    export_parser.add_argument("--adapter-path", type=str, help="Path to LoRA adapter (required unless --gguf-only)")
    export_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    export_parser.add_argument("--model-name", type=str, default="merged_model", help="Name for exported model")
    export_parser.add_argument("--gguf", action="store_true", help="Also export to GGUF format")
    export_parser.add_argument("--gguf-only", action="store_true", help="Only export GGUF (requires existing merged model)")
    export_parser.add_argument("--merged-model-path", type=str, help="Path to existing merged model (for GGUF-only export)")
    export_parser.set_defaults(func=cmd_export_model)

    # clean command
    clean_parser = subparsers.add_parser("clean", help="Clean a dataset file")
    clean_parser.add_argument("input", type=str, help="Input JSONL file")
    clean_parser.add_argument("--output", "-o", type=str, help="Output JSONL file")
    clean_parser.set_defaults(func=cmd_clean_dataset)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Run the appropriate command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
