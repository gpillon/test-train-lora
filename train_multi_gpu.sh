#!/bin/bash
# Multi-GPU training script using accelerate for full GPU utilization

set -e

# Activate virtual environment
source .venv/bin/activate

# Optimize CUDA memory allocation to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check GPU count
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "🔍 Detected $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "🚀 Launching multi-GPU training with accelerate..."
    echo "   - Using ALL $GPU_COUNT GPUs"
    echo "   - Data Parallel (DDP) mode"
    echo "   - Full CUDA cores utilization"
    echo ""
    
    # Use accelerate for multi-GPU training
    accelerate launch \
        --config_file accelerate_config.yaml \
        -m lora_trainer_app.cli \
        train-model \
        --config config.yaml \
        --dataset production_dataset/dataset.jsonl
else
    echo "💻 Single GPU detected, using standard training..."
    lora-trainer train-model \
        --config config.yaml \
        --dataset production_dataset/dataset.jsonl
fi


