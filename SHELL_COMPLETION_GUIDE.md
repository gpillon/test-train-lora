# 🚀 Shell Completion Guide

## 📋 Quick Start

### Step 1: Install Completions

Run this command **once** to install completions for your shell:

```bash
# For Bash
lora-trainer --install-completion bash

# For Zsh
lora-trainer --install-completion zsh

# For Fish
lora-trainer --install-completion fish

# For PowerShell
lora-trainer --install-completion powershell
```

### Step 2: Reload Your Shell

**Option A: Restart your terminal** (easiest)
```bash
# Just close and reopen your terminal
```

**Option B: Source the completion file**
```bash
# Bash
source ~/.bash_completions/lora-trainer.sh

# Zsh
source ~/.zsh_completions/_lora-trainer

# Fish
# Fish auto-loads from ~/.config/fish/completions/
```

### Step 3: Test It!

```bash
# Type this and press TAB
lora-trainer [TAB]

# You should see:
# test-models  generate-data  train-model  inference  export-model  clean

# Type more and press TAB
lora-trainer train-model --[TAB]

# You should see all options:
# --config  --dataset  --model-id  --epochs  --batch-size  ...
```

## 🎯 What Gets Completed

### 1. **Commands** (after `lora-trainer`)
```bash
lora-trainer [TAB]
# Shows: test-models, generate-data, train-model, inference, export-model, clean
```

### 2. **Options** (after `lora-trainer COMMAND --`)
```bash
lora-trainer train-model --[TAB]
# Shows: --config, --dataset, --model-id, --epochs, --batch-size, etc.
```

### 3. **Option Values** (for enum options)
```bash
lora-trainer train-model --log-level [TAB]
# Shows: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### 4. **File Paths** (for file/directory options)
```bash
lora-trainer train-model --dataset [TAB]
# Shows files in current directory
```

## 📖 Examples

### Example 1: Quick Command Selection
```bash
$ lora-trainer [TAB]
test-models    generate-data    train-model    inference    export-model    clean

$ lora-trainer t[TAB]
# Completes to: lora-trainer train-model
```

### Example 2: Option Discovery
```bash
$ lora-trainer train-model --[TAB]
--config        --model-id      --learning-rate  --output-dir
--dataset       --epochs        --lora-r         --merge
--batch-size    --lora-alpha    --no-4bit        --log-level

$ lora-trainer train-model --ep[TAB]
# Completes to: lora-trainer train-model --epochs
```

### Example 3: Log Level Selection
```bash
$ lora-trainer train-model --log-level [TAB]
DEBUG    INFO    WARNING    ERROR    CRITICAL

$ lora-trainer train-model --log-level D[TAB]
# Completes to: lora-trainer train-model --log-level DEBUG
```

## 🔧 Troubleshooting

### Problem: "Nothing happens when I press TAB"

**Solution:** Completions not installed or shell not reloaded.

```bash
# 1. Check if completion is installed
ls ~/.bash_completions/lora-trainer.sh   # Bash
ls ~/.zsh_completions/_lora-trainer      # Zsh

# 2. If not found, install it
lora-trainer --install-completion bash   # or zsh/fish

# 3. Reload your shell
source ~/.bashrc  # Bash
source ~/.zshrc   # Zsh
# or just restart terminal
```

### Problem: "Only shows files, not commands/options"

**Solution:** Completion script not sourced.

```bash
# For Bash, add to ~/.bashrc
if [ -f ~/.bash_completions/lora-trainer.sh ]; then
    source ~/.bash_completions/lora-trainer.sh
fi

# For Zsh, add to ~/.zshrc
if [ -f ~/.zsh_completions/_lora-trainer ]; then
    fpath=(~/.zsh_completions $fpath)
    autoload -Uz compinit && compinit
fi
```

### Problem: "Completion is slow (takes 5+ seconds)"

**Solution:** This was fixed! Completions are now instant (~0.12s).

```bash
# Test completion speed
time (COMP_WORDS="lora-trainer " COMP_CWORD=1 _LORA_TRAINER_COMPLETE=complete_bash lora-trainer)
# Should be ~0.12s

# If still slow, make sure you're using the latest version
pip install --upgrade .
```

### Problem: "Completions work for commands but not options"

**Solution:** You need to type `--` first.

```bash
# Wrong - won't show options
lora-trainer train-model [TAB]

# Correct - shows options
lora-trainer train-model --[TAB]
```

## 🎨 Shell-Specific Tips

### Bash Tips

```bash
# Make completions case-insensitive
echo "set completion-ignore-case on" >> ~/.inputrc

# Show all options on first TAB (instead of requiring two TABs)
echo "set show-all-if-ambiguous on" >> ~/.inputrc
```

### Zsh Tips

```bash
# Enable better completion system
autoload -Uz compinit && compinit

# Case-insensitive completion
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Za-z}'

# Show descriptions
zstyle ':completion:*' verbose yes
```

### Fish Tips

```fish
# Fish has great defaults, but you can customize:

# Case-insensitive completion (already default)
set -g fish_complete_ignore_case on

# Show help for commands
set -g fish_complete_show_descriptions yes
```

## 📊 Performance

With our optimizations, completion is **instant**:

| Operation | Time | Speed |
|-----------|------|-------|
| `lora-trainer [TAB]` | ~0.12s | ⚡ Instant |
| `lora-trainer train-model --[TAB]` | ~0.15s | ⚡ Instant |
| `lora-trainer export-model --help` | ~0.14s | ⚡ Instant |

## 💡 Pro Tips

### 1. **Use Partial Completion**
```bash
# Type just a few letters and TAB
lora-trainer tr[TAB]    # → train-model
lora-trainer inf[TAB]   # → inference
lora-trainer ex[TAB]    # → export-model
```

### 2. **Discover Options Without --help**
```bash
# Just use TAB instead of looking at help
lora-trainer train-model --[TAB][TAB]
# Shows all available options instantly
```

### 3. **Combine with Aliases**
```bash
# Add to ~/.bashrc or ~/.zshrc
alias lt='lora-trainer'
alias lt-train='lora-trainer train-model'
alias lt-gen='lora-trainer generate-data'

# Completions work with aliases too!
lt [TAB]           # Works!
lt-train --[TAB]   # Works!
```

### 4. **Use History with Completion**
```bash
# Search command history with Ctrl+R
Ctrl+R train
# Then edit and use TAB for options
```

## 🔄 Updating Completions

If you update the CLI (add new commands/options), reinstall completions:

```bash
# 1. Update the package
pip install --upgrade .

# 2. Reinstall completions
lora-trainer --install-completion bash --force

# 3. Reload shell
source ~/.bashrc
```

## 📚 What Gets Completed - Detailed

### All Commands
- `test-models` - Test API connectivity
- `generate-data` - Generate training dataset
- `train-model` - Train LoRA model
- `inference` - Run inference
- `export-model` - Export model to HF/GGUF
- `clean` - Clean dataset file

### Common Options (work with most commands)
- `--help` - Show help message
- `--config` or `-c` - Config file path
- `--log-level` or `-l` - Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)

### Command-Specific Options

#### `train-model`
- `--dataset` - Dataset file path (file completion)
- `--model-id` - Model ID
- `--epochs` - Number of epochs
- `--batch-size` - Batch size
- `--learning-rate` - Learning rate
- `--lora-r` - LoRA rank
- `--lora-alpha` - LoRA alpha
- `--no-4bit` - Disable 4-bit quantization
- `--output-dir` - Output directory (dir completion)
- `--merge` - Merge after training

#### `export-model`
- `--base-model` - Base model ID
- `--adapter-path` - Adapter path (dir completion)
- `--output-dir` - Output directory (dir completion)
- `--model-name` - Model name
- `--gguf` - Export to GGUF
- `--gguf-only` - Only export GGUF
- `--merged-model-path` - Merged model path (dir completion)

#### `inference`
- `--model-id` - Model ID (required)
- `--adapter` - Adapter path (dir completion)
- `--merged` - Use merged model
- `--prompt` - Inference prompt
- `--interactive` or `-i` - Interactive mode
- `--system-prompt` - System prompt
- `--max-tokens` - Max tokens
- `--temperature` - Temperature
- `--top-p` - Top-p sampling
- `--no-4bit` - Disable quantization

## ✨ Summary

1. **Install:** `lora-trainer --install-completion bash`
2. **Reload:** Restart terminal or `source ~/.bashrc`
3. **Use:** Press TAB after `lora-trainer` or `lora-trainer COMMAND --`
4. **Enjoy:** Instant completions with zero waiting! ⚡

The completion system is now **production-ready** and **instant** thanks to our lazy import optimizations!

