# Dockerfile per LoRA Trainer su OpenShift
#FROM registry.redhat.io/rhoai/odh-training-cuda128-torch28-py312-rhel9@sha256:851f2b31fa418d2eb172ddfa6010851bd5d4f0844d16b2b42a408f2c0b985b86
FROM image-registry.openshift-image-registry.svc:5000/lora-test/odh-training-cuda128-torch28-py312-rhel9@sha256:9bebe87278d36a16c864536cda0d77e92b95b7059b5915629498adff3fe53966

# Set working directory
WORKDIR /workspace

# Install system dependencies
USER root
RUN dnf install -y git && \
    dnf clean all

# Switch back to notebook user
USER 1001

# Copy requirements and install Python dependencies
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY lora_trainer_app/ /workspace/lora_trainer_app/
# COPY config.yaml /workspace/
COPY download_pii_dataset.py /workspace/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command (can be overridden in deployment)
CMD ["python", "-m", "lora_trainer_app.cli", "train-model", "--config", "/workspace/config.yaml"]

