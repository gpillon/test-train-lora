"""
Configuration module for LoRA Training Application
Manages all configuration settings, environment variables, and constants.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class ModelConfig:
    """Configuration for an API model endpoint."""

    name: str
    api_url: str
    model: str
    api_key: Optional[str] = None  # Direct API key
    api_key_env: Optional[str] = None  # Environment variable name containing the API key
    verify_ssl: bool = True  # Whether to verify SSL certificates

    def get_api_key(self) -> Optional[str]:
        """Retrieve API key from direct value or environment variable."""
        # First try direct API key
        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        
        # Then try environment variable
        if self.api_key_env:
            key = os.getenv(self.api_key_env)
            if key and key.strip():
                return key.strip()
        
        return None


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    batch_size: int = 10
    max_tokens: int = 800
    temperature: float = 0.85
    top_p: float = 0.95
    output_dir: str = field(
        default_factory=lambda: f"outputs/{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    )
    random_seed: Optional[int] = None

    # Output format configuration
    output_format_type: str = "dialogue"  # 'dialogue' or 'single'
    empty_input_percentage: int = 70

    # Prompt template for generation
    prompt_template: Optional[str] = None
    
    # Topics for dataset generation
    base_topics: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "technical": [
                "Kubernetes scheduling and node pressure",
                "Linux networking with eBPF and XDP",
                "DevOps pipelines with GitHub Actions and ArgoCD",
                "Observability trade-offs in Prometheus/Thanos",
                "OpenShift OAuth integration and route passthrough TLS",
                "KubeVirt VM lifecycle and wake-on-LAN operator design",
                "CSI design: TopoLVM vs RBD for single-node",
                "Cilium vs Calico datapath decisions",
        ],
            "abstract": [
                "consciousness as an emergent property",
                "limits of intelligence in bounded agents",
                "alignment, corrigibility, and instrumental goals",
                "emergence and phase transitions in complex systems",
                "meaning-making under uncertainty",
        ],
            "ethical": [
                "automation and job displacement trade-offs",
                "privacy vs utility in telemetry collection",
                "model transparency vs performance",
                "responsible release strategies for powerful models",
        ],
            "metaphors": [
                "daily life metaphors for distributed consensus",
                "kitchen metaphors for CI/CD pipelines",
                "city traffic analogies for backpressure and queues",
                "sports analogies for blue/green deployments",
        ],
            "creative_eng": [
                "mixing creative reasoning with engineering constraints",
                "designing playful interfaces for serious operations",
                "storytelling for incident retrospectives",
            ],
        }
    )
    
    n_topic_variants: int = 60


@dataclass
class LoRAConfig:
    """Configuration for LoRA training."""

    model_id: str = "google/gemma-2-2b-it"
    hf_token: Optional[str] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_4bit: bool = True
    target_modules: Optional[List[str]] = None  # Auto-detect if None
    
    # Training parameters
    num_epochs: int = 2
    batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    
    # Logging and saving
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    save_total_limit: int = 2
    
    # Scheduler
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    
    # Output
    output_dir: str = field(
        default_factory=lambda: f"lora_gemma2b_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    )
    
    @classmethod
    def from_env(cls) -> "LoRAConfig":
        """Create configuration from environment variables."""
        return cls(
            model_id=os.getenv("MODEL_ID", "google/gemma-2-2b-it"),
            lora_r=int(os.getenv("LORA_R", "16")),
            lora_alpha=int(os.getenv("LORA_ALPHA", "32")),
            lora_dropout=float(os.getenv("LORA_DROPOUT", "0.05")),
            use_4bit=os.getenv("USE_4BIT", "1") == "1",
            num_epochs=int(os.getenv("EPOCHS", "2")),
            batch_size=int(os.getenv("BATCH_SIZE", "2")),
            eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "2")),
            gradient_accumulation_steps=int(os.getenv("GRAD_ACC", "8")),
            learning_rate=float(os.getenv("LR", "2e-4")),
            max_seq_length=int(os.getenv("MAX_SEQ_LEN", "2048")),
            output_dir=os.getenv(
                "OUT_DIR", f"lora_gemma2b_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            ),
        )


# Persona configurations
SYSTEM_ANALYST = "You are Analyst 🧠: precise, structured, and terse. Use bullet points, cite assumptions, include step-by-step with explicit checks."
SYSTEM_CREATIVE = "You are Creative 💡: divergent, vivid yet concise. Offer surprising angles, metaphors, and short examples."
SYSTEM_CONSENSUS = "You are Consensus 🤝: synthesize Analyst and Creative. Provide a balanced, actionable summary with trade-offs."

PERSONAE = [
    ("analyst", SYSTEM_ANALYST),
    ("creative", SYSTEM_CREATIVE),
    ("consensus", SYSTEM_CONSENSUS),
]

STYLE_HINTS = {
    "analyst": "structured, with numbered steps, pre-flight checks, and acceptance criteria",
    "creative": "playful and metaphorical but technically plausible",
    "consensus": "balanced summary with bullet-point trade-offs and a crisp recommendation",
}

USER_TEMPLATE = (
    "Topic: {topic}\n"
    "Task: produce a self-contained answer that would be *useful for a LoRA training set*, avoiding generic filler. "
    "Keep it {style_hint}. Include concrete details (commands/snippets/checklists) when relevant."
)


# Default model configurations
def get_default_models() -> List[ModelConfig]:
    """Get default model configurations from environment."""
    models = []
    
    # Check for Scout17B
    if os.getenv("API_KEY_SCOUT17B"):
        models.append(
            ModelConfig(
                name="llama-4-scout-17b-16e-w4a16",
                api_url="https://llama-4-scout-17b-16e-w4a16-maas-apicast-production.apps.prod.rhoai.rh-aiservices-bu.com:443/v1",
                api_key_env="API_KEY_SCOUT17B",
                model="llama-4-scout-17b-16e-w4a16",
            )
        )
    
    # Check for Gemini Flash
    if os.getenv("API_KEY_GEMINI_FLASH"):
        models.append(
            ModelConfig(
                name="gemini-2.5-flash",
                api_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key_env="API_KEY_GEMINI_FLASH",
                model="gemini-2.5-flash",
            )
        )
    
    # Check for Mistral Small
    if os.getenv("API_KEY_MISTRAL_SMALL"):
        models.append(
            ModelConfig(
                name="Mistral-Small-24B-W8A8",
                api_url="https://litellm-litemaas.apps.prod.rhoai.rh-aiservices-bu.com/v1",
                api_key_env="API_KEY_MISTRAL_SMALL",
                model="Mistral-Small-24B-W8A8",
            )
        )
    
    # Check for generic OpenAI-compatible endpoints
    for i in range(1, 6):  # Support up to 5 custom models
        if os.getenv(f"API_KEY_{i}"):
            models.append(
                ModelConfig(
                    name=os.getenv(f"MODEL_NAME_{i}", f"Custom-Model-{i}"),
                    api_url=os.getenv(f"API_URL_{i}", "https://api.openai.com/v1"),
                    api_key_env=f"API_KEY_{i}",
                    model=os.getenv(f"MODEL_ID_{i}", "gpt-4o-mini"),
                )
            )
    
    return models
