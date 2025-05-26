import numpy as np
import os
import shutil
from mlp import experiment_hyperparameters
import torch
import time
import gensim.downloader as api
from gensim.models import KeyedVectors

# Directory to save model
MODEL_OUTPUT_DIR = os.environ.get('MODEL_OUTPUT_DIR', '/app/models')

# Make sure output directory exists
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

print("Starting model training process...")
print(f"Model will be saved to: {MODEL_OUTPUT_DIR}")

# Verificar si existe el modelo de embeddings y descargarlo si es necesario
EMBEDDING_MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, 'glove_twitter_200.model')
if not os.path.exists(EMBEDDING_MODEL_PATH):
    print(f"Embedding model not found at {EMBEDDING_MODEL_PATH}")
    print("Downloading GloVe Twitter embeddings...")
    try:
        glove_twitter = api.load("glove-twitter-200")
        print("Embeddings downloaded successfully")
        glove_twitter.save(EMBEDDING_MODEL_PATH)
        print(f"Embeddings saved to {EMBEDDING_MODEL_PATH}")
    except Exception as e:
        print(f"Error downloading embeddings: {str(e)}")
        print("Model training will continue, but may have reduced performance")

# Load data
print("Loading training and test data...")
try:
    X_train = np.load("data/X_train.npy")  # shape (156374, 200)
    y_train = np.load("data/y_train.npy")  # shape (156374,)
    X_test = np.load("data/X_test.npy")    # shape (39094, 200)
    y_test = np.load("data/y_test.npy")    # shape (39094,)
    
    print(f"Data loaded successfully!")
    print(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit(1)

# Train model
print("\nStarting MLP hyperparameter experimentation for sentiment analysis...")
start_time = time.time()

try:
    best_model, best_params = experiment_hyperparameters(X_train, y_train, X_test, y_test)
    
    # Save model files to output directory
    model_files = [
        'best_mlp_model.pt',
        'best_mlp_model_full.pt',
        'model_config.json',
        'confusion_matrix.png',
        'metrics_report.txt',
        'learning_curves.png',
        'hyperparameter_summary.txt'
    ]
    
    for file in model_files:
        if os.path.exists(file):
            dest_path = os.path.join(MODEL_OUTPUT_DIR, file)
            shutil.copy(file, dest_path)
            print(f"Copied {file} to {dest_path}")
    
    # Save embedding model if available
    if os.path.exists('glove_twitter_200.model'):
        embed_dest = os.path.join(MODEL_OUTPUT_DIR, 'glove_twitter_200.model')
        shutil.copy('glove_twitter_200.model', embed_dest)
        print(f"Copied embedding model to {embed_dest}")
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Model files saved to {MODEL_OUTPUT_DIR}")
    print("Model training container will now exit.")
    
except Exception as e:
    print(f"Error during model training: {str(e)}")
    exit(1) 