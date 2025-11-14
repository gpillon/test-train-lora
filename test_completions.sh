#!/bin/bash
# Test script to verify shell completions work correctly

echo "🧪 Testing LoRA Trainer Shell Completions"
echo "=========================================="
echo ""

# Activate venv
source .venv/bin/activate

# Load completions
if [ -f ~/.bash_completions/lora-trainer.sh ]; then
    source ~/.bash_completions/lora-trainer.sh
    echo "✅ Completion script loaded"
else
    echo "❌ Completion script not found. Installing..."
    lora-trainer --install-completion bash
    source ~/.bash_completions/lora-trainer.sh
fi

echo ""
echo "Test 1: Command completion"
echo "$ lora-trainer [TAB]"
COMP_WORDS="lora-trainer " COMP_CWORD=1 _LORA_TRAINER_COMPLETE=complete_bash lora-trainer
echo ""

echo "Test 2: train-model options"
echo "$ lora-trainer train-model --[TAB]"
COMP_WORDS="lora-trainer train-model --" COMP_CWORD=2 _LORA_TRAINER_COMPLETE=complete_bash lora-trainer
echo ""

echo "Test 3: export-model options"
echo "$ lora-trainer export-model --[TAB]"
COMP_WORDS="lora-trainer export-model --" COMP_CWORD=2 _LORA_TRAINER_COMPLETE=complete_bash lora-trainer
echo ""

echo "Test 4: log-level values"
echo "$ lora-trainer train-model --log-level [TAB]"
COMP_WORDS="lora-trainer train-model --log-level " COMP_CWORD=3 _LORA_TRAINER_COMPLETE=complete_bash lora-trainer
echo ""

echo "✅ All tests completed!"
echo ""
echo "💡 To use completions in your shell:"
echo "   1. Run: lora-trainer --install-completion bash"
echo "   2. Restart your terminal (or source ~/.bashrc)"
echo "   3. Try: lora-trainer [TAB]"

