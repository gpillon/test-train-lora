"""
LoRA Trainer Application
A modular Python application for dataset generation and LoRA fine-tuning.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Lazy imports - don't import heavy modules at package level
# This keeps CLI startup fast for --help and completion
# Modules are imported when actually needed

__all__ = [
    "DatasetConfig",
    "LoRAConfig",
    "ModelConfig",
    "DatasetGenerator",
    "LoRATrainer",
    "LoRAInference",
    "InteractiveChatSession",
]
