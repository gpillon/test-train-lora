# 🚀 Ottimizzazioni Memoria GPU

Questo documento descrive le ottimizzazioni applicate per ridurre l'uso della memoria GPU e prevenire il fallback sulla RAM.

## ⚙️ Modifiche Applicate

### 1. Configurazione LoRA Ridotta
- **`lora_r: 8`** (era 16) - Riduce i parametri LoRA del ~50%
- **`lora_alpha: 16`** (era 32) - Proporzionale a r
- Questo riduce significativamente la memoria usata dagli adapter LoRA

### 2. Sequenze Più Corte
- **`max_seq_length: 1024`** (era 2048) - Riduce la memoria per sequenza del ~50%
- La memoria per l'attention è quadratica rispetto alla lunghezza della sequenza
- Ridurre da 2048 a 1024 può risparmiare fino al 75% della memoria dell'attention

### 3. Batch Size e Gradient Accumulation
- **`batch_size: 1`** - Batch size minimo per evitare OOM
- **`gradient_accumulation_steps: 8`** - Aumentato per compensare
- Effective batch size = 1 × 8 = 8 (mantiene la qualità del training)

### 4. Forzare GPU-Only (No CPU Offload)
- Modificato `device_map` da `"auto"` a `"cuda:0"` per single GPU
- Aggiunto `max_memory={0: "100%"}` per prevenire offload su RAM
- Il modello ora fallisce invece di usare la RAM (meglio saperlo subito)

### 5. Ottimizzazioni Trainer
- **`packing=False`** - Disabilita il packing delle sequenze per più controllo
- **`gradient_checkpointing=True`** - Trade compute per memoria (~30-40% risparmio)
- **`dataloader_pin_memory=False`** - Risparmia RAM
- **`dataloader_num_workers=0`** - Single-threaded per risparmiare memoria

## 📊 Stima Risparmio Memoria

Con queste ottimizzazioni, il risparmio stimato è:

1. **LoRA r=8 vs r=16**: ~50% meno parametri LoRA
2. **max_seq_length 1024 vs 2048**: ~50-75% meno memoria per sequenza
3. **Gradient checkpointing**: ~30-40% meno memoria durante backward pass
4. **No CPU offload**: Tutto sulla GPU (più veloce, ma richiede GPU sufficiente)

**Totale stimato**: Riduzione del 60-70% dell'uso memoria rispetto alla configurazione originale.

## 🔍 Verifica Uso Memoria

Per monitorare l'uso della memoria GPU durante il training:

```bash
# In un altro terminale
watch -n 1 nvidia-smi
```

Oppure:

```bash
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')"
```

## ⚠️ Se Ancora Usa Troppa Memoria

Se dopo queste ottimizzazioni il modello usa ancora troppa memoria, puoi:

1. **Ridurre ulteriormente `max_seq_length`** a 512 o 256
2. **Ridurre `lora_r`** a 4 (ma potrebbe ridurre la qualità)
3. **Ridurre `batch_size`** a 1 e aumentare `gradient_accumulation_steps` a 16+
4. **Usare solo 2 target_modules** invece di 4:
   ```yaml
   target_modules: ["q_proj", "v_proj"]  # Solo query e value
   ```

## 📝 Note

- Con QLoRA (4-bit), Granite-4.0-H-1B dovrebbe usare ~2-3GB di memoria base
- LoRA adapter aggiunge ~100-200MB con r=8
- Durante il training, la memoria può aumentare del 50-100% per i gradienti
- **Totale stimato**: 4-6GB per il training con queste ottimizzazioni

Se la tua GPU ha meno di 6GB, considera di ridurre ulteriormente `max_seq_length` o `lora_r`.

