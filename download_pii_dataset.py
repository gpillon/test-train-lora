#!/usr/bin/env python3
"""
Script per scaricare il dataset pii-masking-ita da HuggingFace
e salvarlo in formato JSONL compatibile con il trainer LoRA.
"""

import json
from datasets import load_dataset
from pathlib import Path


def download_and_convert_dataset():
    """Scarica il dataset pii-masking-ita e lo converte in JSONL."""
    
    print("📥 Scaricamento dataset pii-masking-ita da HuggingFace...")
    
    # Carica il dataset da HuggingFace
    dataset = load_dataset("DeepMount00/pii-masking-ita", split="train")
    
    print(f"✅ Dataset caricato: {len(dataset)} esempi")
    print(f"📋 Colonne disponibili: {dataset.column_names}")
    
    # Mostra un esempio per capire la struttura
    if len(dataset) > 0:
        print("\n📄 Esempio di record:")
        example = dataset[0]
        print(json.dumps(example, indent=2, ensure_ascii=False))
    
    # Crea directory di output
    output_dir = Path("outputs/pii-masking-ita")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dataset.jsonl"
    
    print("\n💾 Conversione in JSONL...")
    
    # Converti in JSONL
    # Il dataset potrebbe avere diversi formati, quindi adattiamo
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            # Prova a mappare i campi comuni
            # Se il dataset ha già instruction/input/output, usali
            # Altrimenti cerca altri campi comuni
            
            record = {}
            
            # Cerca campi standard per instruction tuning
            if 'instruction' in example:
                record['instruction'] = example['instruction']
            elif 'prompt' in example:
                record['instruction'] = example['prompt']
            elif 'text' in example:
                # Se c'è solo 'text', potrebbe essere già formattato
                record['instruction'] = example['text']

            if 'input' in example:
                record['input'] = example['input']
            else:
                record['input'] = ""  # Vuoto se non presente

            if 'output' in example:
                record['output'] = example['output']
            elif 'response' in example:
                record['output'] = example['response']
            elif 'target' in example:
                record['output'] = example['target']
            elif 'masked_text' in example:
                record['output'] = example['masked_text']

            # Se non abbiamo trovato i campi standard, usa tutti i campi
            if not record.get('instruction') and not record.get('output'):
                # Usa tutti i campi come sono
                record = example

            # Scrivi il record
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

            if (i + 1) % 1000 == 0:
                print(f"  Processati {i + 1}/{len(dataset)} esempi...")

    print(f"\n✅ Dataset salvato in: {output_file}")
    print(f"📊 Totale esempi: {len(dataset)}")
    
    return str(output_file)


if __name__ == "__main__":
    try:
        output_path = download_and_convert_dataset()
        print("\n🎯 Per usare questo dataset, imposta nel config.yaml:")
        print(f"   dataset_path: {output_path}")
    except Exception as e:
        print(f"\n❌ Errore durante il download: {e}")
        import traceback
        traceback.print_exc()

