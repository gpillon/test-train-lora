# 🦙 LoRA Training Application

Applicazione Python modulare per la generazione di dataset e il fine-tuning di modelli di linguaggio utilizzando LoRA (Low-Rank Adaptation).

## 🎯 Caratteristiche

- **Generazione Dataset**: Crea dataset di training usando modelli AI con tre personalità diverse (Analyst, Creative, Consensus)
- **Training LoRA/QLoRA**: Fine-tuning efficiente di modelli come Gemma 2B usando LoRA
- **Inference**: Interfaccia per testare modelli addestrati, con modalità interattiva
- **Architettura Modulare**: Codice pulito e ben organizzato in moduli separati
- **CLI Completa**: Interfaccia a riga di comando per tutte le operazioni

## 📁 Struttura del Progetto

```
lora-test/
├── config.py              # Configurazioni e costanti
├── api_client.py          # Client per API OpenAI-compatibili
├── dataset_generator.py   # Generazione dataset
├── lora_trainer.py        # Training LoRA
├── inference.py           # Inference con modelli addestrati
├── utils.py               # Funzioni utility
├── main.py               # Entry point CLI
├── requirements.txt       # Dipendenze Python
└── README.md             # Questa documentazione
```

## 🚀 Installazione

### Prerequisiti

- Python 3.9+
- GPU con CUDA (consigliato: 12-16GB VRAM per training, 8-10GB con QLoRA)
- Per l'inference senza GPU, è possibile usare CPU (più lento)

### Setup

```bash
# Clona o naviga nella directory del progetto
cd lora-test

# Crea un ambiente virtuale (opzionale ma consigliato)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
venv\Scripts\activate  # Windows

# Installa le dipendenze
pip install -r requirements.txt

# Per CUDA 12.x (se necessario)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## ⚙️ Configurazione

### Variabili d'Ambiente per API Models

Per la generazione del dataset, devi configurare almeno un modello API:

```bash
# Modelli predefiniti
export API_KEY_SCOUT17B='your_key_here'
export API_KEY_GEMINI_FLASH='your_key_here'
export API_KEY_MISTRAL_SMALL='your_key_here'

# Oppure modelli custom
export API_KEY_1='your_openai_key'
export API_URL_1='https://api.openai.com/v1'
export MODEL_ID_1='gpt-4o-mini'
```

### Variabili per Training (opzionali)

```bash
export MODEL_ID='google/gemma-2-2b-it'
export LORA_R=16
export LORA_ALPHA=32
export EPOCHS=2
export BATCH_SIZE=2
export LR=2e-4
```

## 📖 Utilizzo

L'applicazione fornisce una CLI con diversi comandi:

### 1. Test Connettività Modelli

Verifica che i modelli API siano accessibili:

```bash
python main.py test-models
```

Output esempio:
```
🔌 Testing model connectivity...

Results:
  llama-4-scout-17b-16e-w4a16: ✅ Connected (response: OK)
  gemini-2.5-flash: ✅ Connected (response: OK)
  Mistral-Small-24B-W8A8: ✅ Connected (response: OK)

3/3 models working
```

### 2. Generare Dataset

Genera un dataset di training con esempi multi-personalità:

```bash
# Genera 50 esempi
python main.py generate --batch-size 50

# Con parametri personalizzati
python main.py generate \
    --batch-size 100 \
    --max-tokens 1000 \
    --temperature 0.9 \
    --output-dir ./my_dataset \
    --seed 42 \
    --clean
```

**Opzioni:**
- `--batch-size`: Numero di esempi da generare (default: 10)
- `--max-tokens`: Token massimi per risposta (default: 800)
- `--temperature`: Temperatura di generazione (default: 0.85)
- `--output-dir`: Directory di output (default: auto-generata con timestamp)
- `--seed`: Seed random per riproducibilità (default: None)
- `--clean`: Pulisce il dataset dopo la generazione

### 3. Training LoRA

Addestra un modello usando il dataset generato:

```bash
# Training base
python main.py train --dataset ./outputs/20251029-022151/dataset.jsonl

# Training personalizzato
python main.py train \
    --dataset ./my_dataset/dataset.jsonl \
    --model-id google/gemma-2-2b-it \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 3e-4 \
    --lora-r 32 \
    --lora-alpha 64 \
    --output-dir ./my_lora_model
    
# Merge LoRA con modello base dopo training
python main.py train \
    --dataset ./my_dataset/dataset.jsonl \
    --merge
```

**Opzioni:**
- `--dataset`: Path al file JSONL del dataset (auto-scoperto se omesso)
- `--model-id`: ID del modello base (default: google/gemma-2-2b-it)
- `--epochs`: Numero di epoche (default: 2)
- `--batch-size`: Batch size (default: 2)
- `--learning-rate`: Learning rate (default: 2e-4)
- `--lora-r`: Rank LoRA (default: 16)
- `--lora-alpha`: Alpha LoRA (default: 32)
- `--no-4bit`: Disabilita quantizzazione 4-bit
- `--output-dir`: Directory di output (default: auto-generata)
- `--merge`: Merge LoRA con base model dopo il training
- `--config-from-env`: Carica configurazione da variabili d'ambiente

### 4. Inference

Usa il modello addestrato per generare testo:

#### Modalità Prompt Singolo

```bash
python main.py inference \
    --model-id google/gemma-2-2b-it \
    --adapter ./lora_gemma2b_20251029-021642 \
    --prompt "Spiega cosa sono le LoRA in machine learning"
```

#### Modalità Interattiva (Chat)

```bash
python main.py inference \
    --model-id google/gemma-2-2b-it \
    --adapter ./lora_gemma2b_20251029-021642 \
    --interactive
```

Comandi interattivi:
- `quit` / `exit` / `q`: Esci dalla chat
- `reset`: Cancella la cronologia conversazione
- `history`: Visualizza la cronologia

#### Con Modello Merged

Se hai fatto il merge del modello:

```bash
python main.py inference \
    --model-id ./my_lora_model_merged \
    --merged \
    --interactive
```

**Opzioni:**
- `--model-id`: ID modello base o path a modello merged (required)
- `--adapter`: Path all'adapter LoRA (opzionale se --merged)
- `--merged`: Il modello è già merged (non serve adapter)
- `--prompt`: Prompt per inference singola
- `--interactive`: Avvia chat interattiva
- `--system-prompt`: System prompt (default: "You are a helpful assistant.")
- `--max-tokens`: Token massimi da generare (default: 256)
- `--temperature`: Temperatura (default: 0.7)
- `--top-p`: Nucleus sampling (default: 0.9)
- `--no-4bit`: Disabilita quantizzazione 4-bit

### 5. Pulire Dataset

Rimuovi record duplicati o invalidi:

```bash
python main.py clean input.jsonl --output cleaned.jsonl
```

## 🔧 Uso come Libreria

Puoi anche importare i moduli direttamente nel tuo codice Python:

### Esempio: Generazione Dataset

```python
from config import DatasetConfig, get_default_models
from dataset_generator import DatasetGenerator

# Configura
config = DatasetConfig(batch_size=20, max_tokens=1000)
models = get_default_models()

# Genera dataset
generator = DatasetGenerator(config, models)
generator.run_batch()

# Valida
results = generator.validate_output()
print(results)
```

### Esempio: Training

```python
from config import LoRAConfig
from lora_trainer import LoRATrainer

# Configura
config = LoRAConfig(
    model_id='google/gemma-2-2b-it',
    num_epochs=3,
    lora_r=32
)

# Training
trainer = LoRATrainer(config, dataset_path='./dataset.jsonl')
metrics = trainer.run_full_training()
print(metrics)
```

### Esempio: Inference

```python
from inference import LoRAInference

# Inizializza
engine = LoRAInference(
    base_model_id='google/gemma-2-2b-it',
    adapter_path='./lora_output'
)

# Chat
response = engine.chat_simple(
    user_message="Explain LoRA",
    max_new_tokens=200
)
print(response)
```

## 📊 Formati Dataset

Il dataset generato è in formato JSONL, con una struttura come:

```json
{
  "id": "uuid-here",
  "persona": "analyst",
  "input": [
    {"role": "system", "content": "You are Analyst..."},
    {"role": "user", "content": "Topic: ..."}
  ],
  "output": "Generated response...",
  "meta": {
    "model_name": "gemini-2.5-flash",
    "model_id": "gemini-2.5-flash",
    "topic": "Kubernetes scheduling",
    "seed": 12345,
    "temperature": 0.85,
    "created_utc": "2025-10-29T02:21:51.123456Z"
  }
}
```

## 🎭 Personalità (Personae)

Il dataset viene generato usando tre personalità diverse:

1. **🧠 Analyst**: Preciso, strutturato, con step-by-step e bullet points
2. **💡 Creative**: Divergente, metaforico, con esempi narrativi
3. **🤝 Consensus**: Bilanciato, sintetizza le due prospettive

Questo crea un dataset più vario e ricco per il fine-tuning.

## 🐛 Troubleshooting

### Errore: "No models configured"

Assicurati di aver impostato almeno una API key:
```bash
export API_KEY_GEMINI_FLASH='your_key_here'
```

### Errore CUDA out of memory

- Riduci `--batch-size` (es. 1 invece di 2)
- Usa `--gradient-accumulation-steps` più alto (già default a 8)
- Assicurati che `use_4bit=True` (default)
- Riduci `--max-seq-length`

### Modelli non si caricano

Verifica di avere spazio disco sufficiente (i modelli possono pesare diversi GB).

### Errori di import

Reinstalla le dipendenze:
```bash
pip install --upgrade -r requirements.txt
```

## 📝 Note

- **GPU Requirement**: Il training richiede una GPU. Per inference, la CPU funziona ma è più lenta.
- **Quantizzazione**: QLoRA 4-bit è abilitata di default per ridurre l'uso di memoria.
- **Dataset Size**: Più esempi generalmente migliorano la qualità, ma aumentano il tempo di training.
- **LoRA Rank**: Valori tipici sono 8-64. Più alto = più parametri addestrabili ma più memoria.

## 🤝 Contributi

Questo è un progetto di esempio/template. Sentiti libero di modificarlo secondo le tue esigenze!

## 📄 Licenza

Questo codice è fornito "as-is" per uso educativo e di ricerca.

## 🔗 Risorse Utili

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

**Buon Training! 🚀**

