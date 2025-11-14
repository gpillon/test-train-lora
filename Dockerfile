# Dockerfile per LoRA Trainer su OpenShift
FROM quay.io/opendatahub/notebooks:jupyter-pytorch-cuda-11.8.0-ubi9-python-3.11-2024.01.19

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
COPY config.yaml /workspace/
COPY download_pii_dataset.py /workspace/

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Default command (can be overridden in deployment)
CMD ["python", "-m", "lora_trainer_app.cli", "train-model", "--config", "/workspace/config.yaml"]

