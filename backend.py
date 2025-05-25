from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
from mlp import load_best_model, predict_sentiment, MLP
import numpy as np
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import os

# Download NLTK resources (only first time)
nltk.download('punkt')
nltk.download('stopwords')
torch.serialization.add_safe_globals([MLP])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load embedding model
EMBEDDING_MODEL_PATH = 'glove_twitter_200.model'
if not os.path.exists(EMBEDDING_MODEL_PATH):
    raise FileNotFoundError(f"Embedding model {EMBEDDING_MODEL_PATH} not found")

print("Loading embedding model...")
embedding_model = KeyedVectors.load(EMBEDDING_MODEL_PATH)
print("Embedding model loaded")

# Load classification model
print("Loading classification model...")
try:
    model, _ = load_best_model()
    print("Classification model loaded successfully")
except Exception as e:
    print(f"Error loading classification model: {str(e)}")
    raise

# Initialize FastAPI app with English metadata
app = FastAPI(title="Sentiment Analysis API",
              description="API for classifying text into Negative(-1), Neutral(0) or Positive(1) sentiment")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: int  # -1, 0, 1
    probabilities: List[float]
    sentiment_label: str

def safe_to_numpy(tensor):
    """Convert tensor to numpy ensuring it's on CPU"""
    return tensor.cpu().detach().numpy()

def preprocess_text(text):
    """Preprocess text for embedding"""
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stopwords (handles both Spanish and English)
    stop_words = set(stopwords.words('spanish' if 'es' in text.lower() else 'english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def text_to_embedding(text, embedding_model, embedding_size=200):
    """Convert text to embedding using GloVe model"""
    tokens = preprocess_text(text)
    
    # Get embeddings for each word
    word_embeddings = []
    for token in tokens:
        if token in embedding_model:
            word_embeddings.append(embedding_model[token])
    
    # If no words with embeddings found, return zero vector
    if not word_embeddings:
        return np.zeros(embedding_size)
    
    # Average of word embeddings (common approach)
    text_embedding = np.mean(word_embeddings, axis=0)
    
    return text_embedding

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sentiment Analysis API",
        "endpoints": {
            "/predict": "POST - Analyze sentiment of a text",
            "/health": "GET - Check service status"
        }
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment1(request: TextRequest):
    """Endpoint to predict sentiment of input text"""
    try:
        # Convert text to embedding (synchronous)
        text_embedding = text_to_embedding(request.text, embedding_model)
        
        # Get the device where the model is located
        device = next(model.parameters()).device

        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(text_embedding, dtype=torch.float32)\
                      .unsqueeze(0)\
                      .to(device)

        # Ensure tensor is on correct device
        input_tensor = input_tensor.to(device)
        
        # Predict sentiment
        sentiment, probs = predict_sentiment(model,input_tensor,device)

        # Sentiment label mapping
        sentiment_labels = {
            -1: "Negative",
            0: "Neutral",
            1: "Positive"
        }
        
        return {
            "text": request.text,
            "sentiment": sentiment,
            "probabilities": probs.tolist(),
            "sentiment_label": sentiment_labels.get(sentiment, "Unknown")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embedding_loaded": embedding_model is not None,
        "model_loaded": model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)