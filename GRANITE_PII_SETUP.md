# 🎯 Setup per Fine-Tuning Granite con Dataset PII-Masking-ITA

Questa guida spiega come configurare il progetto per il fine-tuning del modello **Granite-4.0-H-1B** per l'anonimizzazione di dati sensibili in italiano usando il dataset **pii-masking-ita**.

## ✅ Cosa è già configurato

Il progetto supporta già:
- ✅ **LoRA/QLoRA** fine-tuning (configurabile nel `config.yaml`)
- ✅ Caricamento modelli da HuggingFace
- ✅ Supporto per dataset HuggingFace (aggiunto)
- ✅ Quantizzazione 4-bit per risparmiare memoria

## 📋 Configurazione

### 1. Modello Granite

Il modello è già configurato nel `config.yaml`:

```yaml
lora:
  model_id: ibm-granite/granite-4.0-h-1b
  use_4bit: true  # QLoRA abilitato
```

### 2. Dataset PII-Masking-ITA

Hai **due opzioni** per usare il dataset:

#### Opzione A: Dataset HuggingFace diretto (consigliato)

Nel `config.yaml`, imposta:

```yaml
lora:
  dataset_path: DeepMount00/pii-masking-ita
```

Il sistema caricherà automaticamente il dataset da HuggingFace.

#### Opzione B: Scaricare il dataset localmente

Se preferisci scaricare il dataset prima:

```bash
python download_pii_dataset.py
```

Poi nel `config.yaml`:

```yaml
lora:
  dataset_path: outputs/pii-masking-ita/dataset.jsonl
```

## 🚀 Esecuzione del Training

### Metodo 1: Usando la CLI

```bash
# Con dataset HuggingFace diretto
python -m lora_trainer_app.cli train-model \
  --config config.yaml \
  --dataset DeepMount00/pii-masking-ita \
  --model-id ibm-granite/granite-4.0-h-1b \
  --epochs 4 \
  --batch-size 1 \
  --learning-rate 0.0004

# Oppure usando solo il config.yaml (se dataset_path è già configurato)
python -m lora_trainer_app.cli train-model --config config.yaml
```

### Metodo 2: Usando Python direttamente

```python
from lora_trainer_app.config_loader import ConfigManager
from lora_trainer_app.lora_trainer import LoRATrainer

# Carica configurazione
config_mgr = ConfigManager("config.yaml")
lora_config = config_mgr.get_lora_config()

# Imposta il dataset HuggingFace
lora_config.dataset_path = "DeepMount00/pii-masking-ita"

# Crea trainer e avvia training
trainer = LoRATrainer(lora_config)
metrics = trainer.run_full_training()

print(f"Training completato! Metriche: {metrics}")
```

## ⚙️ Parametri Consigliati per Granite-4.0-H-1B

Il modello Granite-4.0-H-1B ha ~1B parametri. Parametri consigliati:

```yaml
lora:
  # LoRA parameters
  lora_r: 16          # Rank LoRA (8-32 tipico per modelli 1B)
  lora_alpha: 32      # Alpha = 2 * r è una buona pratica
  lora_dropout: 0.05
  
  # Training hyperparameters
  num_epochs: 3-5     # Dipende dalla dimensione del dataset
  batch_size: 2-4     # Per GPU con 8-16GB VRAM
  gradient_accumulation_steps: 4-8  # Per aumentare batch size effettivo
  learning_rate: 0.0002-0.0004  # Learning rate per LoRA
  max_seq_length: 2048  # Lunghezza massima sequenza
  
  # QLoRA (4-bit quantization)
  use_4bit: true      # Abilita QLoRA per risparmiare memoria
```

## 📊 Struttura Dataset Attesa

Il sistema si aspetta che il dataset abbia uno di questi formati:

### Formato 1: Instruction/Input/Output (preferito)

```json
{
  "instruction": "Anonimizza il seguente testo",
  "input": "Ciao, sono Mario Rossi e la mia email è mario.rossi@example.com",
  "output": "Ciao, sono [PERSON] e la mia email è [EMAIL]"
}
```

### Formato 2: Solo Output

```json
{
  "text": "Formatted text with instruction and response..."
}
```

Il sistema mapperà automaticamente i campi comuni del dataset (`prompt`, `input`, `response`, `masked_text`, ecc.).

## 🔍 Verifica Setup

Prima di avviare il training, verifica:

1. **Token HuggingFace**: Assicurati che `hf_token` nel config.yaml sia valido
2. **GPU disponibile**: 
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```
3. **Dataset accessibile**:
   ```python
   from datasets import load_dataset
   ds = load_dataset("DeepMount00/pii-masking-ita", split="train")
   print(f"Dataset size: {len(ds)}")
   print(f"Columns: {ds.column_names}")
   print(f"Example: {ds[0]}")
   ```

## 🎯 Risultato Atteso

Dopo il training, otterrai:
- Un adapter LoRA salvato in `outputs/lora_granite_<timestamp>/`
- Metriche di training (loss, perplexity, ecc.)
- Possibilità di fare merge con il modello base per inference

## 📝 Note Importanti

1. **Memoria GPU**: Con QLoRA, Granite-4.0-H-1B dovrebbe funzionare su GPU con 8GB+ VRAM
2. **Tempo di training**: Dipende dalla dimensione del dataset (~41K samples) e dalla GPU
3. **Formato output**: Il modello addestrato manterrà il formato chat template di Granite

## 🐛 Troubleshooting

### Errore: "Failed to load as HuggingFace dataset"
- Verifica che il token HuggingFace sia valido
- Controlla la connessione internet
- Prova a scaricare il dataset localmente con `download_pii_dataset.py`

### Errore: "CUDA out of memory"
- Riduci `batch_size` a 1
- Aumenta `gradient_accumulation_steps`
- Verifica che `use_4bit: true` sia impostato

### Errore: "Model not found"
- Verifica che il modello `ibm-granite/granite-4.0-h-1b` sia accessibile
- Potrebbe richiedere autenticazione HuggingFace

## 🔗 Risorse

- [Modello Granite-4.0-H-1B](https://huggingface.co/ibm-granite/granite-4.0-h-1b)
- [Dataset PII-Masking-ITA](https://huggingface.co/datasets/DeepMount00/pii-masking-ita)
- [Documentazione LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)
- [Documentazione QLoRA](https://huggingface.co/docs/peft/package_reference/peft_model#peft.QuantizationConfig)

