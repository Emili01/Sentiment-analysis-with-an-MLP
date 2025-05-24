import numpy as np
from mlp import experiment_hyperparameters

# Load data
X_train = np.load("data/X_train.npy") # shape (156374, 200)
y_train = np.load("data/y_train.npy") # shape (156374,)
X_test = np.load("data/X_test.npy") # shape (39094, 200)
y_test = np.load("data/y_test.npy") # shape (39094,)

# Run hyperparameter experimentation
if __name__ == "__main__":
    print("Starting MLP hyperparameter experimentation for sentiment analysis...")
    print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    best_model, best_params = experiment_hyperparameters(X_train, y_train, X_test, y_test)
    
    print("\nExperimentation complete!")
    print(f"Best model saved as 'best_mlp_model.pt'")
    print(f"Confusion matrix saved as 'confusion_matrix.png'")
    print(f"Metrics report saved as 'metrics_report.txt'")

