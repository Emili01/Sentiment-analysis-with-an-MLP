FROM python:3.9-slim

WORKDIR /app

# Copiamos requirements.txt desde el directorio model-trainer, no de la raíz
COPY model-trainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model training files
COPY model-trainer/train_model.py .
COPY mlp.py .
COPY clean_data.py .

# Copy data directory
COPY data/ /app/data/

# Run training script when container starts
CMD ["python", "train_model.py"] 