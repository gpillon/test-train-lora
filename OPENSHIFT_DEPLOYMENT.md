# 🚀 Deploy LoRA Trainer su OpenShift

Questa guida spiega come deployare il LoRA Trainer su OpenShift AI con output su PVC.

## 📋 Prerequisiti

1. Accesso a un cluster OpenShift
2. `oc` CLI installato e configurato
3. Permessi per creare BuildConfig, Deployment e PVC nel namespace
4. (Opzionale) Nodi GPU disponibili per il training

## 📁 File Creati

- **`Dockerfile`**: Immagine Docker per il trainer
- **`openshift-training.yaml`**: Tutte le risorse OpenShift (BuildConfig, ImageStream, Deployment, PVC)
- **`openshift-setup.sh`**: Script di setup automatico

## 🚀 Deploy Rapido

### Metodo 1: Script Automatico

```bash
# Configura le variabili
export NAMESPACE="my-namespace"
export GIT_REPO="https://github.com/your-org/lora-test.git"
export GIT_REF="main"
export STORAGE_CLASS="gp3"
export STORAGE_SIZE="100Gi"
export HF_TOKEN="your-huggingface-token"

# Esegui lo script
./openshift-setup.sh
```

### Metodo 2: Manuale

1. **Modifica `openshift-training.yaml`**:
   - Sostituisci `your-namespace` con il tuo namespace
   - Aggiorna `GIT_REPO` con il tuo repository Git
   - Modifica `storageClassName` se necessario
   - Aggiorna `HF_TOKEN` nel secret

2. **Crea il secret per HuggingFace**:
   ```bash
   oc create secret generic huggingface-token \
     --from-literal=hf_token="your-token-here" \
     -n your-namespace
   ```

3. **Applica le risorse**:
   ```bash
   oc apply -f openshift-training.yaml
   ```

4. **Avvia il build**:
   ```bash
   oc start-build lora-trainer-build --follow
   ```

5. **Il deployment partirà automaticamente** quando l'immagine è pronta

## 🔧 Configurazione

### Variabili d'Ambiente nel ConfigMap

Puoi modificare il ConfigMap `lora-training-config` per cambiare i parametri:

```bash
oc edit configmap lora-training-config -n your-namespace
```

Parametri disponibili:
- `MODEL_ID`: Modello da addestrare (default: `ibm-granite/granite-4.0-h-1b`)
- `DATASET_PATH`: Dataset HuggingFace o path locale
- `MAX_SEQ_LENGTH`: Lunghezza massima sequenza (default: `1024`)
- `BATCH_SIZE`: Batch size (default: `1`)
- `GRADIENT_ACCUMULATION_STEPS`: Gradient accumulation (default: `8`)
- `NUM_EPOCHS`: Numero di epoche (default: `4`)
- `LEARNING_RATE`: Learning rate (default: `0.0004`)

### GPU Support

Per abilitare il supporto GPU, decommenta le righe nel Deployment:

```yaml
resources:
  requests:
    nvidia.com/gpu: "1"
  limits:
    nvidia.com/gpu: "1"
```

E aggiungi nodeSelector/tolerations se necessario:

```yaml
nodeSelector:
  accelerator: nvidia-tesla-v100

tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

## 📊 Monitoraggio

### Logs del Training

```bash
# Logs in tempo reale
oc logs -f deployment/lora-trainer -n your-namespace

# Logs dell'ultimo pod
oc logs -l app=lora-trainer --tail=100 -n your-namespace
```

### Status Pods

```bash
oc get pods -l app=lora-trainer -n your-namespace
oc describe pod <pod-name> -n your-namespace
```

### Uso Risorse

```bash
oc top pod -l app=lora-trainer -n your-namespace
```

### Output sul PVC

```bash
# Lista file di output
oc exec -it deployment/lora-trainer -- ls -lh /workspace/outputs

# Copia file localmente
oc cp <pod-name>:/workspace/outputs/model.tar.gz ./model.tar.gz -n your-namespace
```

## 💾 Accesso ai Risultati

I risultati del training sono salvati sul PVC `lora-training-outputs` montato in:
- `/workspace/outputs/` - Dataset e log
- `/workspace/models/` - Modelli addestrati

Per accedere ai file:

1. **Via exec**:
   ```bash
   oc exec -it deployment/lora-trainer -- bash
   cd /workspace/outputs
   ls -lh
   ```

2. **Via rsync** (se abilitato):
   ```bash
   oc rsync <pod-name>:/workspace/outputs ./local-outputs -n your-namespace
   ```

3. **Copia file specifici**:
   ```bash
   oc cp <pod-name>:/workspace/outputs/checkpoint-100 ./checkpoint-100 -n your-namespace
   ```

## 🔄 Restart e Rebuild

### Restart Deployment

```bash
oc rollout restart deployment/lora-trainer -n your-namespace
```

### Rebuild Immagine

```bash
# Rebuild dopo modifiche al codice
oc start-build lora-trainer-build --follow -n your-namespace

# Il deployment si aggiornerà automaticamente
```

### Cancella e Ricrea

```bash
# Cancella tutto
oc delete -f openshift-training.yaml

# Ricrea
oc apply -f openshift-training.yaml
```

## 🐛 Troubleshooting

### Build Fallisce

```bash
# Verifica logs del build
oc logs build/lora-trainer-build-<number> -n your-namespace

# Verifica Dockerfile
oc get buildconfig lora-trainer-build -o yaml -n your-namespace
```

### Pod Non Parte

```bash
# Verifica eventi
oc get events --sort-by='.lastTimestamp' -n your-namespace

# Verifica descrizione pod
oc describe pod <pod-name> -n your-namespace
```

### PVC Non Montato

```bash
# Verifica PVC
oc get pvc lora-training-outputs -n your-namespace
oc describe pvc lora-training-outputs -n your-namespace

# Verifica mount nel pod
oc exec deployment/lora-trainer -- df -h
```

### Out of Memory

Aumenta le risorse nel Deployment:

```yaml
resources:
  requests:
    memory: "32Gi"  # Aumenta
  limits:
    memory: "64Gi"  # Aumenta
```

### GPU Non Disponibile

Verifica:
1. Nodi GPU disponibili: `oc get nodes -l accelerator=nvidia-tesla-v100`
2. GPU operator installato
3. NodeSelector e Tolerations corretti nel Deployment

## 📝 Note Importanti

1. **Storage**: Il PVC usa `ReadWriteOnce`, quindi solo un pod può montarlo alla volta
2. **Costi**: Il training può essere costoso in termini di risorse. Monitora l'uso
3. **Backup**: I risultati sono sul PVC. Considera backup regolari
4. **Secrets**: Non committare token nel repository Git. Usa Secrets di OpenShift

## 🔗 Risorse Utili

- [OpenShift Documentation](https://docs.openshift.com/)
- [OpenShift AI Documentation](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai/)
- [PVC Management](https://docs.openshift.com/container-platform/4.15/storage/persistent_storage/persistent-storage-pod.html)

