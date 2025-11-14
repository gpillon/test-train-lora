# 📦 Installation Guide

## Quick Install (Recommended)

```bash
# Navigate to the project directory
cd lora-test

# Install in development mode (editable)
pip install -e .

# Or install normally
pip install .
```

After installation, you'll have these commands available globally:

- `generate-data` - Generate training datasets
- `train-model` - Train LoRA models
- `lora-inference` - Run inference
- `test-models` - Test API connectivity
- `clean-dataset` - Clean dataset files
- `lora-trainer` - Main CLI (alternative entry point)

## Step-by-Step Installation

### 1. Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- GPU with CUDA (recommended for training)
  - 12-16GB VRAM for full precision
  - 8-10GB VRAM with QLoRA 4-bit
- For CPU-only: Training will be very slow, but inference works

### 2. Clone or Download

```bash
cd /path/to/your/workspace
# If you have the project, just navigate to it
cd lora-test
```

### 3. (Optional) Create Virtual Environment

**Highly recommended** to avoid conflicts:

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 4. Install the Package

**Option A: Development/Editable Install** (recommended if you plan to modify code)

```bash
pip install -e .
```

This creates a link to your source code, so changes take effect immediately.

**Option B: Regular Install**

```bash
pip install .
```

**Option C: From GitHub** (if published)

```bash
pip install git+https://github.com/yourusername/lora-trainer.git
```

### 5. Verify Installation

```bash
# Check that commands are available
generate-data --help
train-model --help
lora-inference --help

# Check version
lora-trainer --version
```

You should see help text and no errors.

## Configuration Setup

### 1. Copy Example Configuration

```bash
# Copy the example config
cp config.yaml my-config.yaml
```

### 2. Set API Keys

The config file references environment variables for API keys. Set them:

```bash
# For Linux/Mac (add to ~/.bashrc or ~/.zshrc for persistence)
export API_KEY_GEMINI_FLASH='your-key-here'
export API_KEY_SCOUT17B='your-key-here'
export API_KEY_MISTRAL_SMALL='your-key-here'

# For Windows PowerShell
$env:API_KEY_GEMINI_FLASH='your-key-here'

# Or create a .env file and load it
# (requires python-dotenv: pip install python-dotenv)
```

### 3. Test Configuration

```bash
test-models --config config.yaml
```

You should see connection status for each configured model.

## CUDA Setup (For GPU Training)

### Check CUDA Version

```bash
nvidia-smi
```

Look for the CUDA version in the output.

### Install PyTorch with CUDA Support

The default `pip install .` includes a CUDA build, but if you need a specific version:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify CUDA

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Your GPU name
```

## Troubleshooting

### Command Not Found

If commands like `generate-data` aren't found after installation:

1. Make sure you're in the same shell/terminal where you installed
2. Try reactivating your virtual environment
3. Check that pip installed to the right location:
   ```bash
   which generate-data  # Linux/Mac
   where generate-data  # Windows
   ```
4. You may need to add `~/.local/bin` to your PATH

### Import Errors

```bash
# Reinstall with all dependencies
pip install --upgrade -e .

# Or force reinstall
pip install --force-reinstall -e .
```

### CUDA Out of Memory

- Reduce batch size: `train-model --batch-size 1`
- Use 4-bit quantization (default)
- Reduce `--max-seq-len`
- Close other GPU applications

### Config File Not Found

Commands will search for `config.yaml` in:
1. Current directory
2. `~/.lora-trainer/config.yaml`
3. `/etc/lora-trainer/config.yaml`

Or specify explicitly: `generate-data --config /path/to/config.yaml`

## Uninstallation

```bash
pip uninstall lora-trainer
```

## Development Installation

For developers who want to contribute:

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests (if available)
pytest

# Format code
black lora_trainer_app/

# Type checking
mypy lora_trainer_app/
```

## Docker Installation (Alternative)

If you prefer Docker (Dockerfile not included, but here's a template):

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y git

# Copy project
COPY . /app
WORKDIR /app

# Install
RUN pip install .

# Set config path
ENV CONFIG_PATH=/app/config.yaml

CMD ["bash"]
```

## Next Steps

After installation, see [README.md](README.md) for usage examples and [config.yaml](config.yaml) for configuration options.

Quick start:

```bash
# 1. Test your models
test-models

# 2. Generate a dataset
generate-data --batch-size 20

# 3. Train a model
train-model --epochs 2

# 4. Try inference
lora-inference --model-id google/gemma-2-2b-it --adapter ./lora_output -i
```

Enjoy training! 🚀

