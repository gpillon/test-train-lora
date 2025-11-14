"""
Configuration loader that reads from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import DatasetConfig, LoRAConfig, ModelConfig


def find_config_file(config_path: Optional[str] = None) -> str:
    """
    Find configuration file.

    Args:
        config_path: Explicit path to config file

    Returns:
        Path to config file

    Raises:
        FileNotFoundError: If config file not found
    """
    if config_path and os.path.exists(config_path):
        return config_path

    # Search in common locations
    search_paths = [
        "config.yaml",
        "config.yml",
        os.path.expanduser("~/.lora-trainer/config.yaml"),
        "/etc/lora-trainer/config.yaml",
    ]

    for path in search_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "No configuration file found. Please create config.yaml or specify --config"
    )


def load_yaml_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    config_file = find_config_file(config_path)

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def load_models_from_config(config: Dict[str, Any]) -> List[ModelConfig]:
    """
    Load model configurations from config dict.

    Args:
        config: Configuration dictionary

    Returns:
        List of ModelConfig objects
    """
    models = []

    api_models = config.get("api_models", [])
    for model_data in api_models:
        # Skip disabled models
        if not model_data.get("enabled", True):
            continue

        # Get API key from direct value or environment variable
        api_key = model_data.get("api_key")
        api_key_env = model_data.get("api_key_env")
        
        # Check if API key is available
        has_key = False
        if api_key and api_key.strip():
            has_key = True
        elif api_key_env and os.getenv(api_key_env):
            has_key = True
        
        if not has_key:
            key_source = f"api_key or {api_key_env}" if api_key_env else "api_key"
            print(
                f"⚠️  Skipping {model_data.get('name')}: {key_source} not found"
            )
            continue

        models.append(
            ModelConfig(
                name=model_data["name"],
                api_url=model_data["api_url"],
                model=model_data["model"],
                api_key=api_key,
                api_key_env=api_key_env,
                verify_ssl=model_data.get("verify_ssl", True),  # Default to True
            )
        )

    return models


def load_dataset_config(
    config: Dict[str, Any], cli_overrides: Optional[Dict[str, Any]] = None
) -> DatasetConfig:
    """
    Load dataset configuration from config dict with optional CLI overrides.

    Args:
        config: Configuration dictionary
        cli_overrides: CLI argument overrides

    Returns:
        DatasetConfig object
    """
    dataset_cfg = config.get("dataset", {})
    cli_overrides = cli_overrides or {}

    # Handle output_dir
    output_dir = cli_overrides.get("output_dir") or dataset_cfg.get("output_dir")
    if not output_dir:
        from datetime import datetime

        output_dir = f"outputs/{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    # Get output format config
    output_format = dataset_cfg.get("output_format", {})

    return DatasetConfig(
        batch_size=cli_overrides.get("batch_size", dataset_cfg.get("batch_size", 10)),
        max_tokens=cli_overrides.get("max_tokens", dataset_cfg.get("max_tokens", 800)),
        temperature=cli_overrides.get("temperature", dataset_cfg.get("temperature", 0.85)),
        top_p=cli_overrides.get("top_p", dataset_cfg.get("top_p", 0.95)),
        output_dir=output_dir,
        random_seed=cli_overrides.get("seed", dataset_cfg.get("random_seed")),
        output_format_type=output_format.get("type", "dialogue"),
        empty_input_percentage=output_format.get("empty_input_percentage", 70),
        prompt_template=dataset_cfg.get("prompt_template"),
        base_topics=dataset_cfg.get("base_topics", {}),
        n_topic_variants=dataset_cfg.get("n_topic_variants", 60),
    )


def load_lora_config(
    config: Dict[str, Any], cli_overrides: Optional[Dict[str, Any]] = None
) -> LoRAConfig:
    """
    Load LoRA training configuration from config dict with optional CLI overrides.

    Args:
        config: Configuration dictionary
        cli_overrides: CLI argument overrides

    Returns:
        LoRAConfig object
    """
    lora_cfg = config.get("lora", {})
    cli_overrides = cli_overrides or {}

    # Handle output_dir
    output_dir = cli_overrides.get("output_dir") or lora_cfg.get("output_dir")
    if not output_dir:
        from datetime import datetime

        output_dir = f"lora_gemma2b_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    # Handle target_modules - can be a list or None for auto-detection
    target_modules = lora_cfg.get("target_modules")
    if target_modules is not None and not isinstance(target_modules, list):
        # Convert single string to list
        target_modules = [target_modules] if isinstance(target_modules, str) else None
    
    return LoRAConfig(
        model_id=cli_overrides.get("model_id", lora_cfg.get("model_id", "google/gemma-2-2b-it")),
        hf_token=lora_cfg.get("hf_token"),
        lora_r=cli_overrides.get("lora_r", lora_cfg.get("lora_r", 16)),
        lora_alpha=cli_overrides.get("lora_alpha", lora_cfg.get("lora_alpha", 32)),
        lora_dropout=cli_overrides.get("lora_dropout", lora_cfg.get("lora_dropout", 0.05)),
        use_4bit=not cli_overrides.get("no_4bit", not lora_cfg.get("use_4bit", True)),
        target_modules=target_modules,
        num_epochs=cli_overrides.get("epochs", lora_cfg.get("num_epochs", 2)),
        batch_size=cli_overrides.get("batch_size", lora_cfg.get("batch_size", 2)),
        eval_batch_size=cli_overrides.get("eval_batch_size", lora_cfg.get("eval_batch_size", 2)),
        gradient_accumulation_steps=cli_overrides.get(
            "grad_acc", lora_cfg.get("gradient_accumulation_steps", 8)
        ),
        learning_rate=cli_overrides.get("learning_rate", lora_cfg.get("learning_rate", 2e-4)),
        max_seq_length=cli_overrides.get("max_seq_len", lora_cfg.get("max_seq_length", 2048)),
        logging_steps=lora_cfg.get("logging_steps", 10),
        eval_steps=lora_cfg.get("eval_steps", 100),
        save_steps=lora_cfg.get("save_steps", 200),
        save_total_limit=lora_cfg.get("save_total_limit", 2),
        lr_scheduler_type=lora_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=lora_cfg.get("warmup_ratio", 0.03),
        output_dir=output_dir,
    )


class ConfigManager:
    """Manages application configuration from YAML and CLI."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML config file
        """
        self.config_path = config_path
        self.config = load_yaml_config(config_path)

    def get_models(self) -> List[ModelConfig]:
        """Get configured models."""
        return load_models_from_config(self.config)

    def get_dataset_config(self, cli_overrides: Optional[Dict[str, Any]] = None) -> DatasetConfig:
        """Get dataset configuration."""
        return load_dataset_config(self.config, cli_overrides)

    def get_lora_config(self, cli_overrides: Optional[Dict[str, Any]] = None) -> LoRAConfig:
        """Get LoRA training configuration."""
        return load_lora_config(self.config, cli_overrides)

    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration."""
        return self.config.get("inference", {})

    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration."""
        return self.config.get("paths", {})
