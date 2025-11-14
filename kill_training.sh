#!/bin/bash
# Kill all training processes (multi-GPU DDP, accelerate, lora-trainer) and clean GPU memory

echo "🔍 Searching for training processes..."

# Find all training-related processes
TRAIN_PIDS=$(ps aux | grep -E "train_multi_gpu|lora-trainer|accelerate launch" | grep -v grep | awk '{print $2}')
PYTHON_PIDS=$(ps aux | grep -E "lora_trainer_app.cli.*train-model" | grep -v grep | awk '{print $2}')

ALL_PIDS="$TRAIN_PIDS $PYTHON_PIDS"
ALL_PIDS=$(echo $ALL_PIDS | tr ' ' '\n' | sort -u | tr '\n' ' ')

if [ -z "$ALL_PIDS" ]; then
    echo "✅ No training processes found"
else
    echo "📍 Found processes:"
    ps aux | head -1
    for pid in $ALL_PIDS; do
        ps aux | grep "^\S*\s*$pid\s" | grep -v grep
    done
    
    echo ""
    echo "🛑 Terminating processes..."
    
    # Try graceful kill first (SIGTERM)
    for pid in $ALL_PIDS; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "  Sending SIGTERM to $pid"
            kill -15 $pid 2>/dev/null
        fi
    done
    
    sleep 3
    
    # Force kill if still running (SIGKILL)
    for pid in $ALL_PIDS; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "  Force killing $pid (SIGKILL)"
            kill -9 $pid 2>/dev/null
        fi
    done
    
    sleep 1
    echo "✅ All processes terminated"
fi

# Aggressive GPU cleanup
echo ""
echo "🧹 Cleaning GPU memory..."
if command -v nvidia-smi &> /dev/null; then
    # Kill any remaining GPU processes
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read gpid; do
        if [ ! -z "$gpid" ]; then
            PROC_NAME=$(ps -p $gpid -o comm= 2>/dev/null)
            if [[ "$PROC_NAME" =~ python|lora ]]; then
                echo "  Killing GPU process $gpid ($PROC_NAME)"
                kill -9 $gpid 2>/dev/null
            fi
        fi
    done
fi

sleep 1

# Check GPU status
echo ""
echo "📊 GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s (%s): %s%% util | %d MB / %d MB (%.1f%%)\n", $1, $2, $3, $4, $5, ($4/$5)*100}'
    
    echo ""
    echo "🔧 Active GPU processes:"
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | wc -l)
    if [ "$GPU_PROCS" -eq 0 ]; then
        echo "  ✅ No active GPU compute processes"
    else
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | \
            awk -F', ' '{printf "  PID %s: %s (%s MB)\n", $1, $2, $3}'
    fi
else
    echo "  ⚠️  nvidia-smi not available"
fi

echo ""
echo "✅ Done!"

