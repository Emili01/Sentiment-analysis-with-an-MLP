FROM python:3.9-slim

WORKDIR /app

# Copiamos requirements.txt desde el directorio backend, no de la raíz
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory
RUN mkdir -p /app/models

# Copy backend files
COPY backend/backend.py .
COPY mlp.py .
# COPY ../models/glove_twitter_200.model /app/models/glove_twitter_200.model # This is handled by docker-compose volume

# Expose the API port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"] 