#!/bin/bash
# Script per deployare il training su OpenShift

set -e

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Variabili configurabili
NAMESPACE="${NAMESPACE:-lora-test}"
GIT_REPO="${GIT_REPO:-https://github.com/gpillon/test-train-lora.git}"
GIT_REF="${GIT_REF:-main}"
STORAGE_CLASS="${STORAGE_CLASS:-gp3}"
STORAGE_SIZE="${STORAGE_SIZE:-50Gi}"
HF_TOKEN="${HF_TOKEN:-}"

echo -e "${GREEN}🚀 Setup LoRA Trainer su OpenShift${NC}"
echo ""

# Verifica che oc sia installato
if ! command -v oc &> /dev/null; then
    echo -e "${RED}❌ oc (OpenShift CLI) non trovato. Installa oc prima di continuare.${NC}"
    exit 1
fi

# Verifica login
echo -e "${YELLOW}📋 Verificando login OpenShift...${NC}"
if ! oc whoami &> /dev/null; then
    echo -e "${RED}❌ Non sei loggato in OpenShift. Esegui 'oc login' prima.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Loggato come: $(oc whoami)${NC}"
echo ""

# Crea namespace se non esiste
echo -e "${YELLOW}📦 Creando namespace '${NAMESPACE}'...${NC}"
oc create namespace ${NAMESPACE} --dry-run=client -o yaml | oc apply -f -
oc project ${NAMESPACE}
echo -e "${GREEN}✓ Namespace pronto${NC}"
echo ""

# Sostituisci placeholder nel file YAML
echo -e "${YELLOW}🔧 Configurando risorse OpenShift...${NC}"
sed -i.bak \
    -e "s/your-namespace/${NAMESPACE}/g" \
    -e "s|https://github.com/your-org/lora-test.git|${GIT_REPO}|g" \
    -e "s/ref: main/ref: ${GIT_REF}/g" \
    -e "s/storageClassName: gp3/storageClassName: ${STORAGE_CLASS}/g" \
    -e "s/storage: 100Gi/storage: ${STORAGE_SIZE}/g" \
    openshift-training.yaml

# Crea secret per HuggingFace token se fornito
if [ -n "${HF_TOKEN}" ]; then
    echo -e "${YELLOW}🔐 Creando secret per HuggingFace token...${NC}"
    oc create secret generic huggingface-token \
        --from-literal=hf_token="${HF_TOKEN}" \
        --dry-run=client -o yaml | oc apply -f -
    echo -e "${GREEN}✓ Secret creato${NC}"
else
    echo -e "${YELLOW}⚠️  HF_TOKEN non fornito. Crea manualmente il secret 'huggingface-token'${NC}"
    echo "   oc create secret generic huggingface-token --from-literal=hf_token='your-token'"
fi
echo ""

# Applica le risorse
echo -e "${YELLOW}📥 Applicando risorse OpenShift...${NC}"
oc apply -f openshift-training.yaml
echo -e "${GREEN}✓ Risorse applicate${NC}"
echo ""

# Avvia il build
echo -e "${YELLOW}🔨 Avviando build dell'immagine...${NC}"
oc start-build lora-trainer-build --follow
echo -e "${GREEN}✓ Build completato${NC}"
echo ""

# Mostra status
echo -e "${GREEN}✅ Setup completato!${NC}"
echo ""
echo -e "${YELLOW}📊 Status risorse:${NC}"
echo ""
echo "BuildConfig:"
oc get bc lora-trainer-build
echo ""
echo "ImageStream:"
oc get is lora-trainer
echo ""
echo "PVC:"
oc get pvc lora-training-outputs
echo ""
echo "Deployment:"
oc get deployment lora-trainer
echo ""
echo "Pods:"
oc get pods -l app=lora-trainer
echo ""
echo -e "${GREEN}📝 Comandi utili:${NC}"
echo "  - Logs: oc logs -f deployment/lora-trainer"
echo "  - Status: oc get pods -l app=lora-trainer"
echo "  - Outputs: oc exec -it deployment/lora-trainer -- ls -lh /workspace/outputs"
echo "  - Restart: oc rollout restart deployment/lora-trainer"
echo ""

