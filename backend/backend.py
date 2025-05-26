from fastapi import FastAPI, HTTPException, BackgroundTasks
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
from nltk.stem import WordNetLemmatizer
import string
import os
import gensim.downloader as api
import threading
import time
import re

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)  # Agregado para resolver el error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

EMBEDDING_MODEL_PATH = os.environ.get('EMBEDDING_MODEL_PATH', '/app/models/glove_twitter_200.model')
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/models/best_mlp_model_full.pt')
CONFIG_PATH = os.environ.get('CONFIG_PATH', '/app/models/model_config.json')

embedding_model = None
embedding_model_status = "not_loaded"  
model = None
model_status = "not_loaded"  

app = FastAPI(title="Sentiment Analysis API",
              description="API for classifying text into Negative(-1), Neutral(0) or Positive(1) sentiment")

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
    sentiment: int  
    probabilities: List[float]
    sentiment_label: str
    models_status: dict  

def clean_text(text):
    """
    Basic text cleaning for English text
    """
    # Convert to string in case of non-text values
    text = str(text)

    # 1. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 2. Remove mentions (@user) and hashtags
    text = re.sub(r'\@\w+|\#\w+', '', text)
    
    # 3. Remove special characters (keeping basic punctuation)
    text = re.sub(r'[^\w\s.,!?]', ' ', text)
    
    # 4. Remove numbers 
    text = re.sub(r'\d+', '', text)
    
    # 5. Remove multiple spaces and line breaks
    text = ' '.join(text.split())
    
    # 6. Convert to lowercase
    text = text.lower()
    
    return text.strip()

def advanced_text_processing(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def preprocess_text(text):
    cleaned_text = clean_text(text)
    processed_text = advanced_text_processing(cleaned_text)
    return processed_text.split() 

def text_to_embedding(text, embedding_model, embedding_size=200):
    tokens = preprocess_text(text)

    word_embeddings = []
    found_words = []
    missing_words = []
    
    for token in tokens:
        if token in embedding_model:
            word_embeddings.append(embedding_model[token])
            found_words.append(token)
        else:
            missing_words.append(token)
    
    if not word_embeddings:
        print(f"Warning: No embeddings found for any words in text: '{text}'")
        print(f"Missing words: {missing_words}")
        return np.zeros(embedding_size)
    
    text_embedding = np.mean(word_embeddings, axis=0)
    
    return text_embedding

def load_embedding_model_background():
    global embedding_model, embedding_model_status
    
    embedding_model_status = "loading"
    print(f"Loading embedding model from {EMBEDDING_MODEL_PATH}...")
    
    if os.path.exists(EMBEDDING_MODEL_PATH):
        try:
            embedding_model = KeyedVectors.load(EMBEDDING_MODEL_PATH)
            embedding_model_status = "ready"
            print("Embedding model loaded successfully")
            return
        except Exception as e:
            print(f"Error loading local model: {str(e)}")
    
    try:
        embedding_model = api.load("glove-twitter-200")
        
        os.makedirs(os.path.dirname(EMBEDDING_MODEL_PATH), exist_ok=True)
        embedding_model.save(EMBEDDING_MODEL_PATH)    
        embedding_model_status = "ready"
    except Exception as e:
        print(f"Error downloading embedding model: {str(e)}")
        embedding_model_status = "error"

def load_classification_model_background():
    global model, model_status
    
    model_status = "loading"
    print(f"Loading classification model from {MODEL_PATH}...")
    
    try:
        loaded_model, config = load_best_model(model_path=MODEL_PATH, config_path=CONFIG_PATH)
        model = loaded_model
        model_status = "ready"
        print("Classification model loaded successfully")
    except Exception as e:
        print(f"Error loading classification model: {str(e)}")
        model_status = "error"

# Start background loading of models immediately
threading.Thread(target=load_embedding_model_background, daemon=True).start()
threading.Thread(target=load_classification_model_background, daemon=True).start()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Sentiment Analysis API",
        "status": "healthy",
        "models_status": {
            "embedding_model": embedding_model_status,
            "classification_model": model_status
        },
        "endpoints": {
            "/predict": "POST - Analyze sentiment of a text",
            "/health": "GET - Check service status"
        }
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict_text_sentiment(request: TextRequest):
    """Endpoint to predict sentiment of input text"""
    # Check if models are loaded
    if embedding_model_status != "ready" or model_status != "ready":
        # Return a friendly response if models are still loading
        return {
            "text": request.text,
            "sentiment": 0,  # Default neutral 
            "probabilities": [0.0, 0.85, 0.0],  # Default neutral probability
            "sentiment_label": "Loading models...",
            "models_status": {
                "embedding_model": embedding_model_status,
                "classification_model": model_status
            }
        }
    
    try:
        # Convert text to embedding
        text_embedding = text_to_embedding(request.text, embedding_model)
        
        # Get device where the model is located
        model_device = next(model.parameters()).device

        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(text_embedding, dtype=torch.float32)\
                      .unsqueeze(0)\
                      .to(model_device)
        
        # Predict sentiment
        sentiment, probs = predict_sentiment(model, input_tensor, model_device)

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
            "sentiment_label": sentiment_labels.get(sentiment, "Unknown"),
            "models_status": {
                "embedding_model": embedding_model_status,
                "classification_model": model_status
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_status": {
            "embedding_model": {
                "status": embedding_model_status,
                "path": EMBEDDING_MODEL_PATH
            },
            "classification_model": {
                "status": model_status,
                "path": MODEL_PATH
            }
        }
    }
