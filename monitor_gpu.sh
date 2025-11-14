#!/bin/bash
# GPU Monitoring Script for LoRA Training
# Shows detailed GPU utilization, memory, processes, and training stats

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ nvidia-smi not found. Are you on a system with NVIDIA GPUs?${NC}"
    exit 1
fi

# Function to draw progress bar
draw_bar() {
    local percent=$1
    local width=20
    local filled=$(awk "BEGIN {printf \"%.0f\", ($percent/100)*$width}")
    local empty=$((width - filled))
    
    printf "${GREEN}%*s${NC}${YELLOW}%*s${NC}" $filled "" $empty "" | tr ' ' '█' | sed "s/█/${GREEN}█${NC}/g" | sed "s/\$${YELLOW}/${YELLOW}/g"
    printf "${YELLOW}%*s${NC}" $empty "" | tr ' ' '░'
}

# Function to get GPU status
get_gpu_status() {
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${BOLD}                    📊 GPU MONITORING - LoRA Training${NC}                        ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Get timestamp
    echo -e "${BLUE}🕐 Time: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
    
    # Detailed GPU information
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit --format=csv,noheader | while IFS=',' read -r index name mem_total mem_used mem_free util_gpu util_mem temp power power_limit; do
        # Trim whitespace
        index=$(echo "$index" | xargs)
        name=$(echo "$name" | xargs)
        mem_total=$(echo "$mem_total" | xargs | sed 's/ MiB//')
        mem_used=$(echo "$mem_used" | xargs | sed 's/ MiB//')
        mem_free=$(echo "$mem_free" | xargs | sed 's/ MiB//')
        util_gpu=$(echo "$util_gpu" | xargs | sed 's/ %//')
        util_mem=$(echo "$util_mem" | xargs | sed 's/ %//')
        temp=$(echo "$temp" | xargs | sed 's/ C//')
        power=$(echo "$power" | xargs | sed 's/ W//')
        power_limit=$(echo "$power_limit" | xargs | sed 's/ W//')
        
        # Calculate percentages
        mem_percent=$(awk "BEGIN {printf \"%.1f\", ($mem_used/$mem_total)*100}")
        power_percent=$(awk "BEGIN {printf \"%.1f\", ($power/$power_limit)*100}")
        
        # Color coding for utilization
        if [ "$util_gpu" -gt 80 ]; then
            util_color="${GREEN}"
        elif [ "$util_gpu" -gt 50 ]; then
            util_color="${YELLOW}"
        else
            util_color="${RED}"
        fi
        
        # Color coding for memory
        mem_check=$(awk "BEGIN {print ($mem_percent > 80) ? 1 : (($mem_percent > 50) ? 2 : 3)}")
        if [ "$mem_check" -eq 1 ]; then
            mem_color="${GREEN}"
        elif [ "$mem_check" -eq 2 ]; then
            mem_color="${YELLOW}"
        else
            mem_color="${RED}"
        fi
        
        # Color coding for temperature
        if [ "$temp" -lt 70 ]; then
            temp_color="${GREEN}"
        elif [ "$temp" -lt 85 ]; then
            temp_color="${YELLOW}"
        else
            temp_color="${RED}"
        fi
        
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BOLD}${CYAN}🎮 GPU ${index}: ${name}${NC}"
        echo ""
        echo -e "   ${BOLD}💻 GPU Utilization:${NC} ${util_color}${util_gpu}%${NC}"
        echo -e "      $(printf '%*s' $((util_gpu/5)) | tr ' ' '█')$(printf '%*s' $((20-util_gpu/5)) | tr ' ' '░')"
        echo ""
        echo -e "   ${BOLD}💾 Memory Usage:${NC}"
        echo -e "      Used:   ${mem_color}${mem_used} MB / ${mem_total} MB (${mem_percent}%)${NC}"
        echo -e "      Free:   ${mem_color}${mem_free} MB${NC}"
        echo -e "      Bar:    $(printf '%*s' $((mem_used*20/mem_total)) | tr ' ' '█')$(printf '%*s' $((20-mem_used*20/mem_total)) | tr ' ' '░')"
        echo ""
        echo -e "   ${BOLD}🌡️  Temperature:${NC} ${temp_color}${temp}°C${NC}"
        echo -e "   ${BOLD}⚡ Power:${NC} ${power}W / ${power_limit}W (${power_percent}%)"
        echo ""
    done
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to show GPU processes
get_gpu_processes() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${BOLD}                        🔄 GPU PROCESSES${NC}                                     ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Get processes per GPU
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    for gpu_id in $(seq 0 $((GPU_COUNT - 1))); do
        # Try to get compute processes first
        compute_processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i $gpu_id 2>/dev/null)
        
        # Also get all processes with GPU memory allocated (more comprehensive)
        # Use nvidia-smi with process info
        all_processes=$(nvidia-smi pmon -c 1 -i $gpu_id 2>/dev/null | tail -n +3 | grep -v "^#" | grep -v "^$" | awk '{if ($2 != "-" && $2 != "Id") print $1,$2,$3,$4}' 2>/dev/null)
        
        # Also try to get PIDs from fuser or lsof if available
        pids_on_gpu=""
        if command -v fuser &> /dev/null; then
            pids_on_gpu=$(fuser /dev/nvidia${gpu_id} 2>/dev/null | tr ' ' '\n' | grep -v "^$")
        fi
        
        has_processes=false
        
        if [ ! -z "$compute_processes" ]; then
            echo -e "${BOLD}${BLUE}GPU ${gpu_id} (Compute Processes):${NC}"
            # Check if this is multi-GPU DDP - same PID might appear on multiple GPUs
            total_pids=$(echo "$compute_processes" | wc -l)
            echo "$compute_processes" | while IFS=',' read -r pid name mem; do
                pid=$(echo "$pid" | xargs)
                name=$(echo "$name" | xargs)
                mem=$(echo "$mem" | xargs)
                
                # If name is not found or empty, try to get it from ps
                if [ -z "$name" ] || [ "$name" == "[Not Found]" ] || [ "$name" == "N/A" ]; then
                    if ps -p $pid > /dev/null 2>&1; then
                        name=$(ps -p $pid -o comm= 2>/dev/null | head -1)
                        # Get full command if it's python
                        if [ "$name" == "python" ] || [ "$name" == "python3" ]; then
                            name=$(ps -p $pid -o cmd= 2>/dev/null | head -c 60 | tr '\n' ' ')
                        fi
                    else
                        name="[PID $pid not found]"
                    fi
                fi
                
                # If memory is not available, try to get it from ps
                if [ -z "$mem" ] || [ "$mem" == "[N/A]" ] || [ "$mem" == "N/A" ]; then
                    if ps -p $pid > /dev/null 2>&1; then
                        mem_rss=$(ps -p $pid -o rss= 2>/dev/null | awk '{printf "%.1f MB", $1/1024}')
                        mem="$mem_rss (RAM)"
                    else
                        mem="N/A"
                    fi
                fi
                
                echo -e "   ${GREEN}•${NC} PID: ${pid} | Process: ${CYAN}${name}${NC} | Memory: ${mem}"
            done
            has_processes=true
        fi
        
        if [ ! -z "$all_processes" ]; then
            if [ "$has_processes" = false ]; then
                echo -e "${BOLD}${BLUE}GPU ${gpu_id} (All GPU Processes):${NC}"
            fi
            echo "$all_processes" | while read -r pid type sm util mem; do
                if [ "$pid" != "-" ] && [ "$pid" != "Id" ] && [ ! -z "$pid" ]; then
                    echo -e "   ${GREEN}•${NC} PID: ${pid} | Type: ${type} | SM: ${sm} | Util: ${util} | Mem: ${mem}"
                fi
            done
            has_processes=true
        fi
        
        # Fallback: try to find Python processes that might be using GPU
        if [ "$has_processes" = false ] && [ ! -z "$pids_on_gpu" ]; then
            echo -e "${BOLD}${BLUE}GPU ${gpu_id} (Processes with GPU Access):${NC}"
            for pid in $pids_on_gpu; do
                if ps -p $pid > /dev/null 2>&1; then
                    cmd=$(ps -p $pid -o comm= 2>/dev/null)
                    mem_rss=$(ps -p $pid -o rss= 2>/dev/null | awk '{printf "%.1f MB", $1/1024}')
                    # Try to get GPU memory if available
                    gpu_mem=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits -i $gpu_id 2>/dev/null | grep -w "$pid" | head -1 | xargs)
                    if [ ! -z "$gpu_mem" ]; then
                        mem="${gpu_mem} MB (GPU)"
                    else
                        mem="${mem_rss} (RAM)"
                    fi
                    # Get full command for Python processes
                    if [ "$cmd" == "python" ] || [ "$cmd" == "python3" ]; then
                        full_cmd=$(ps -p $pid -o cmd= 2>/dev/null | head -c 70)
                        echo -e "   ${GREEN}•${NC} PID: ${pid} | Process: ${CYAN}${full_cmd}${NC}"
                        echo -e "     Memory: ${mem}"
                    else
                        echo -e "   ${GREEN}•${NC} PID: ${pid} | Process: ${CYAN}${cmd}${NC} | Memory: ${mem}"
                    fi
                fi
            done
            has_processes=true
        fi
        
        if [ "$has_processes" = false ]; then
            # Check if there's memory allocated but no active compute
            mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id | xargs | sed 's/ MiB//')
            if [ "$mem_used" -gt 100 ]; then
                echo -e "${YELLOW}GPU ${gpu_id}:${NC} Memory allocated (${mem_used} MB) but no active compute processes detected"
                echo -e "   ${CYAN}Tip:${NC} This usually means the process is in setup/waiting phase"
            else
                echo -e "${YELLOW}GPU ${gpu_id}:${NC} No active processes"
            fi
        fi
        echo ""
    done
}

# Function to show training processes
get_training_processes() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${BOLD}                    🚀 Training Processes${NC}                                    ${CYAN}║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    # Find training-related processes
    processes=$(ps aux | grep -E "(train_multi_gpu|accelerate|lora-trainer|python.*train|python.*lora)" | grep -v grep | grep -v monitor)
    
    if [ -z "$processes" ]; then
        echo -e "${YELLOW}⏸️  No training processes detected${NC}"
    else
        echo "$processes" | while read -r line; do
            pid=$(echo "$line" | awk '{print $2}')
            cpu=$(echo "$line" | awk '{print $3}')
            mem=$(echo "$line" | awk '{print $4}')
            cmd=$(echo "$line" | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
            echo -e "${GREEN}•${NC} ${BOLD}PID:${NC} ${pid} | ${BOLD}CPU:${NC} ${cpu}% | ${BOLD}Memory:${NC} ${mem}%"
            echo -e "  ${CYAN}Command:${NC} ${cmd}"
            echo ""
        done
    fi
    echo ""
}

# Main execution
if [ "$1" == "--watch" ] || [ "$1" == "-w" ]; then
    # Watch mode - refresh every 2 seconds
    while true; do
        get_gpu_status
        get_gpu_processes
        get_training_processes
        echo -e "${CYAN}Press Ctrl+C to exit... (Refreshing every 2 seconds)${NC}"
        sleep 2
    done
else
    # Single run
    get_gpu_status
    get_gpu_processes
    get_training_processes
fi
