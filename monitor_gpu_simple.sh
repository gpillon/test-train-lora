#!/bin/bash
# Simple GPU Monitor - Quick overview
# Usage: ./monitor_gpu_simple.sh [--watch]

WATCH=${1:-""}

show_status() {
    clear
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║  📊 GPU Status - $(date '+%H:%M:%S')                              ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    awk -F', ' '{
        printf "GPU %s: %3s%% util | %4d MB / %4d MB (%.1f%%) | %s°C | %sW\n", 
            $1, $2, $3, $4, ($3/$4)*100, $5, $6
    }'
    
    echo ""
    echo "Training processes:"
    ps aux | grep -E "(train_multi_gpu|accelerate|lora-trainer)" | grep -v grep | grep -v monitor | wc -l | xargs echo "  Active:"
    echo ""
}

if [ "$WATCH" == "--watch" ] || [ "$WATCH" == "-w" ]; then
    while true; do
        show_status
        sleep 2
    done
else
    show_status
fi



