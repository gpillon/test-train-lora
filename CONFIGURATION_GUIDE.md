# 📋 Configuration Guide

## Overview

The LoRA Trainer uses a **two-level configuration system**:
1. **YAML file** (`config.yaml`) - Base configuration
2. **CLI arguments** - Override YAML values

## 🎯 Priority Order

```
CLI Arguments > YAML Configuration > Default Values
```

## 📝 Example Usage

### Using only YAML configuration:
```bash
# config.yaml has: batch_size: 10, temperature: 0.7
lora-trainer generate-data --config config.yaml
# Uses: batch_size=10, temperature=0.7
```

### Overriding specific values:
```bash
# Override batch_size but keep other YAML values
lora-trainer generate-data --config config.yaml --batch-size 50
# Uses: batch_size=50, temperature=0.7 (from YAML)
```

### Multiple overrides:
```bash
# Override multiple values
lora-trainer generate-data \
  --config config.yaml \
  --batch-size 100 \
  --temperature 0.9 \
  --max-tokens 2048
# Uses: batch_size=100, temperature=0.9, max_tokens=2048
```

### No config file (uses defaults):
```bash
# Must specify all required values
lora-trainer generate-data \
  --batch-size 50 \
  --output-dir ./data
```

## 🎯 Command-Specific Overrides

### `generate-data` Command

**YAML Keys:**
```yaml
dataset:
  batch_size: 10
  max_tokens: 4096
  temperature: 0.7
  output_dir: production_dataset
  seed: 42
```

**CLI Overrides:**
```bash
--batch-size        # Override dataset.batch_size
--max-tokens        # Override dataset.max_tokens
--temperature       # Override dataset.temperature
--output-dir        # Override dataset.output_dir
--seed              # Override dataset.seed
```

**Example:**
```bash
lora-trainer generate-data \
  --batch-size 100 \
  --temperature 0.8 \
  --clean
```

---

### `train-model` Command

**YAML Keys:**
```yaml
lora:
  model_id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  dataset_path: null
  num_epochs: 4
  batch_size: 4
  learning_rate: 0.0004
  lora_r: 16
  lora_alpha: 32
  use_4bit: true
  output_dir: lora_output
```

**CLI Overrides:**
```bash
--dataset           # Override lora.dataset_path
--model-id          # Override lora.model_id
--epochs            # Override lora.num_epochs
--batch-size        # Override lora.batch_size
--learning-rate     # Override lora.learning_rate
--lora-r            # Override lora.lora_r
--lora-alpha        # Override lora.lora_alpha
--no-4bit           # Set lora.use_4bit = False
--output-dir        # Override lora.output_dir
--merge             # Merge after training
```

**Example:**
```bash
lora-trainer train-model \
  --dataset ./data/dataset.jsonl \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 0.0002 \
  --merge
```

---

### `inference` Command

**YAML Keys:**
```yaml
inference:
  system_prompt: "..."
  max_tokens: 256
  temperature: 0.7
  top_p: 0.9
```

**CLI Overrides:**
```bash
--system-prompt     # Override inference.system_prompt
--max-tokens        # Override inference.max_tokens
--temperature       # Override inference.temperature
--top-p             # Override inference.top_p
--no-4bit           # Disable quantization
```

**Example:**
```bash
lora-trainer inference \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter ./lora_output \
  --prompt "Hello!" \
  --temperature 0.9 \
  --max-tokens 512
```

---

### `export-model` Command

**No YAML configuration** - all parameters required via CLI:

```bash
# Full export (merge + GGUF)
lora-trainer export-model \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --adapter-path ./lora_output \
  --output-dir ./exports \
  --model-name my_model \
  --gguf

# Only GGUF from merged model
lora-trainer export-model \
  --gguf-only \
  --merged-model-path ./exports/merged_model \
  --output-dir ./exports
```

---

## 🔧 Configuration Best Practices

### 1. **Base Configuration in YAML**
Put stable, project-wide settings in `config.yaml`:
```yaml
lora:
  model_id: TinyLlama/TinyLlama-1.1B-Chat-v1.0
  lora_r: 16
  lora_alpha: 32
  use_4bit: true

dataset:
  max_tokens: 4096
  temperature: 0.7
```

### 2. **Experiment with CLI Overrides**
Use CLI arguments for testing and iteration:
```bash
# Quick experiment with different learning rates
lora-trainer train-model --dataset data.jsonl --learning-rate 0.0001
lora-trainer train-model --dataset data.jsonl --learning-rate 0.0003
lora-trainer train-model --dataset data.jsonl --learning-rate 0.0005
```

### 3. **Multiple Config Files**
Create different configs for different scenarios:
```bash
# Development
lora-trainer generate-data --config config.dev.yaml --batch-size 10

# Production
lora-trainer generate-data --config config.prod.yaml --batch-size 1000
```

---

## 📊 View Current Configuration

To see what configuration is being used:
```bash
# Use INFO or DEBUG log level
lora-trainer train-model \
  --log-level DEBUG \
  --dataset data.jsonl
```

This will show:
- Loaded YAML values
- Applied CLI overrides
- Final configuration used for training

---

## 🔍 Configuration Resolution Examples

### Example 1: Dataset Generation
```yaml
# config.yaml
dataset:
  batch_size: 10
  temperature: 0.7
  max_tokens: 4096
```

```bash
# Command
lora-trainer generate-data --batch-size 50
```

**Result:**
- `batch_size = 50` (from CLI override)
- `temperature = 0.7` (from YAML)
- `max_tokens = 4096` (from YAML)

---

### Example 2: Training
```yaml
# config.yaml
lora:
  num_epochs: 4
  batch_size: 4
  learning_rate: 0.0004
```

```bash
# Command
lora-trainer train-model \
  --dataset data.jsonl \
  --epochs 3 \
  --learning-rate 0.0002
```

**Result:**
- `num_epochs = 3` (from CLI override)
- `batch_size = 4` (from YAML)
- `learning_rate = 0.0002` (from CLI override)
- `dataset_path = data.jsonl` (from CLI)

---

## ✨ Tips

1. **Start with YAML**: Define your stable configuration in YAML
2. **Override for experiments**: Use CLI args to test different values
3. **Use `--help`**: Every command shows available overrides
4. **Log level DEBUG**: See exactly what configuration is used
5. **Multiple configs**: Keep different YAML files for dev/prod

---

## 📚 Related Documentation

- See `lora-trainer COMMAND --help` for all available options
- See `config.yaml` for full YAML structure
- See `README.md` for complete usage guide

