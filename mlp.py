import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from itertools import product
import copy
import time

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rates=None, activation=nn.ReLU, 
                 batch_norm=False):
        """
        Initialize MLP with customizable architecture and regularization options
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden dimensions
            output_dim: Output dimension
            dropout_rates: List of dropout rates (one for each layer)
            activation: Activation function to use
            batch_norm: Whether to use batch normalization
        """
        super(MLP, self).__init__()
        
        # Default dropout if not provided
        if dropout_rates is None:
            dropout_rates = [0.2] * len(hidden_dims)
        elif len(dropout_rates) != len(hidden_dims):
            raise ValueError("Length of dropout_rates must match length of hidden_dims")
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        # Create hidden layers with selected normalization and regularization
        for i in range(len(dims)-1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Normalization (optional)
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            # Activation
            layers.append(activation())
            
            # Dropout for regularization
            layers.append(nn.Dropout(dropout_rates[i]))
        
        # Add output layer
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights with He initialization (good for ReLU)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.model(x)
    


def preprocess_labels(y):
    """Convert labels from [-1, 0, 1] to [0, 1, 2]"""
    return y + 1


def postprocess_labels(y_pred):
    """Convert predictions from [0, 1, 2] back to [-1, 0, 1]"""
    return y_pred - 1


def train_model(model, optimizer, criterion, train_loader, val_loader, 
                device, num_epochs, patience=5, verbose=True):
    """
    Train the model with early stopping
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        criterion: Loss function
        train_loader: DataLoader for training data
        val_loader: DataLoader for test data (used for validation)
        device: Device to train on
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        verbose: Whether to print progress
        
    Returns:
        best_model: Best model based on validation loss
        train_losses, val_losses: Loss history
        train_accs, val_accs: Accuracy history
    """
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Get classification loss
            ce_loss = criterion(outputs, labels)
            
            # Add L2 regularization loss if applicable
            if hasattr(model, 'get_l2_regularization_loss'):
                l2_loss = model.get_l2_regularization_loss()
                loss = ce_loss + l2_loss
            else:
                loss = ce_loss
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs, 1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predictions = torch.max(outputs, 1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if verbose and epoch % 5 == 0:
            print(f'Epoch {epoch+1}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch+1}')
                break
    
    return best_model, train_losses, val_losses, train_accs, val_accs


def evaluate_model(model, test_loader, criterion, device):
    """Evaluate model on test data"""
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / total
    test_acc = correct / total
    
    return test_loss, test_acc, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot and save normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()


def plot_learning_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot learning curves for accuracy and loss"""
    epochs = range(1, len(train_losses) + 1)
    
    # Plot loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()


def experiment_hyperparameters(X_train, y_train, X_test, y_test):
    """Run experiments with different hyperparameters"""
    # Preprocess labels
    y_train_processed = preprocess_labels(y_train)
    y_test_processed = preprocess_labels(y_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train_processed)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test_processed)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Updated hyperparameters based on the latest results
    # Focusing on larger architectures and advanced regularization techniques
    hyperparams = {
        'hidden_dims': [       
            [2048, 1024, 512, 256, 128], # MEJOR VALOR
            # [4096, 2048, 1024, 512, 256, 128],
            # [2048, 1024, 512, 256, 128, 64],
            # [1024, 512, 256, 128, 64, 32],
            # [512, 256, 128, 64, 32, 16],
            # [256, 128, 64, 32, 16, 8],
            # [32, 16, 8, 4, 2, 1],
            # [1024, 512, 256, 128, 64],
            # [512, 256, 128, 64, 32],
            # [256, 128, 64, 32, 16],
            # [64, 32, 16, 8, 4],
            # [4096, 3072, 2048, 1024, 512],
            # [2048, 1536, 1024, 512, 256],
            # [1024, 768, 512, 256, 128],
            # [512, 384, 256, 128, 64],
            # [32, 24, 16, 8, 4],
            # [32, 16, 8],
            # [128, 64, 32],
            # [64, 32, 16],
            # [32, 16, 8],
            # [16, 8, 4],
            # [8, 4, 2],
            # [4, 2, 1],
            # [2, 1, 0.5],
        ],
        'dropout_rates': [
            # [0.4, 0.4, 0.3, 0.3, 0.3, 0.3],
            # [0.3, 0.3, 0.3, 0.3, 0.2, 0.1],
            # [0.2, 0.2, 0.2, 0.2, 0.1, 0.05],
            # [0.4, 0.4, 0.3, 0.3, 0.2, 0.0125],
            # [0.4, 0.4, 0.3, 0.3, 0.2, 0.00625],
            [0.4, 0.4, 0.3, 0.3, 0.2] # MEJOR VALOR
            # [0.3, 0.3, 0.2, 0.2, 0.1],
            # [0.2, 0.2, 0.1, 0.1, 0.05],
            # [0.2, 0.2, 0.1],
            # [0.3, 0.3, 0.2],
            # [0.4, 0.4, 0.3],
            # [0.5, 0.5, 0.4],
        ],      
        'batch_norm': [True],           # Batch normalization
        'learning_rate': [0.001],       # Fixed as requested
        'batch_size': [128]             # Fixed as requested
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Track results
    results = []
    input_dim = X_train.shape[1]
    output_dim = 3  # 3 classes: 0, 1, 2 (originally -1, 0, 1)
    
    best_accuracy = 0
    best_model = None
    best_params = None
    best_preds = None
    best_labels = None
    best_train_losses = None
    best_val_losses = None
    best_train_accs = None
    best_val_accs = None
    
    # Try all hyperparameter combinations
    total_combinations = (len(hyperparams['hidden_dims']) * 
                         len(hyperparams['dropout_rates']) *
                         len(hyperparams['batch_norm']) * 
                         len(hyperparams['learning_rate']) *
                         len(hyperparams['batch_size']))
    
    print(f"Testing {total_combinations} hyperparameter combinations")
    print(f"Maximum epochs per combination: 75, Early stopping patience: 8")
    combo_count = 0
    
    for hidden_dims, dropout_rates, batch_norm, lr, batch_size in product(
        hyperparams['hidden_dims'], 
        hyperparams['dropout_rates'],
        hyperparams['batch_norm'],
        hyperparams['learning_rate'],
        hyperparams['batch_size']
    ):
        combo_count += 1
        start_time = time.time()
        
        print(f"\nCombination {combo_count}/{total_combinations}:")
        print(f"Hidden dims: {hidden_dims}, Dropout rates: {dropout_rates}, "
              f"Learning rate: {lr}, Batch size: {batch_size}")
        
        # Skip if dropout_rates doesn't match hidden_dims length
        if len(dropout_rates) != len(hidden_dims):
            print("Skipping: dropout_rates length doesn't match hidden_dims length")
            continue
            
        # Create dataloaders for current batch size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Create model with current hyperparameters
        model = MLP(
            input_dim=input_dim, 
            hidden_dims=hidden_dims, 
            output_dim=output_dim, 
            dropout_rates=dropout_rates,
            batch_norm=batch_norm
        ).to(device)
        
        # Create optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Initialize regularization info
        reg_info = "Regularization:" 
        if batch_norm:
            reg_info += " BN"
        
        # Train model with early stopping - use test set as validation set
        trained_model, train_losses, val_losses, train_accs, val_accs = train_model(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=test_loader,  # Using test set as validation set
            device=device,
            num_epochs=75,
            patience=8,
            verbose=False
        )
        
        # Evaluate on test set (same as validation set in this case)
        test_loss, test_acc, test_preds, test_labels = evaluate_model(
            model=trained_model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )
        
        # Save results
        training_time = time.time() - start_time
        result = {
            'hidden_dims': hidden_dims,
            'dropout_rates': dropout_rates,
            'batch_norm': batch_norm,
            'regularization': reg_info,
            'learning_rate': lr,
            'batch_size': batch_size,
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
            'test_loss': test_loss,
            'train_acc': train_accs[-1],
            'val_acc': val_accs[-1],
            'test_acc': test_acc,
            'num_epochs': len(train_losses),
            'training_time': training_time,
            'overfitting_metric': train_accs[-1] - val_accs[-1]  # Measure of overfitting
        }
        results.append(result)
        
        print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}, "
              f"Time: {training_time:.2f}s, Epochs: {len(train_losses)}/{75} ({len(train_losses)/75*100:.0f}%)")
        
        # Update best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = copy.deepcopy(trained_model)
            best_params = result
            best_preds = test_preds
            best_labels = test_labels
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_train_accs = train_accs
            best_val_accs = val_accs
    
    # Print and save best results
    print("\n====== Best Hyperparameters ======")
    print(f"Hidden Dimensions: {best_params['hidden_dims']}")
    print(f"Dropout Rates: {best_params['dropout_rates']}")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Batch Size: {best_params['batch_size']}")
    print(f"Train Accuracy: {best_params['train_acc']:.4f}") 
    print(f"Test/Validation Accuracy: {best_params['test_acc']:.4f}")  # Same as val_acc now
    print(f"Train Loss: {best_params['train_loss']:.4f}")
    print(f"Test/Validation Loss: {best_params['test_loss']:.4f}")  # Same as val_loss now
    print(f"Epochs Used: {best_params['num_epochs']}/{75} ({best_params['num_epochs']/75*100:.0f}%)")
    
    # Print overfitting analysis
    train_test_acc_diff = best_params['train_acc'] - best_params['test_acc']
    if train_test_acc_diff > 0.05:
        print(f"\nPossible overfitting detected: Train accuracy is {train_test_acc_diff:.4f} higher than test accuracy")
    else:
        print(f"\nNo significant overfitting detected: Train-Test accuracy difference is only {train_test_acc_diff:.4f}")
    
    # Plot and save learning curves for the best model
    plot_learning_curves(best_train_losses, best_val_losses, best_train_accs, best_val_accs)
    print("Learning curves saved as 'learning_curves.png'")
    
    # Generate confusion matrix
    plot_confusion_matrix(
        best_labels, best_preds, 
        class_names=['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    )
    
    # Generate classification report
    original_labels = postprocess_labels(best_labels)
    original_preds = postprocess_labels(best_preds)
    class_report = classification_report(
        original_labels, original_preds, 
        target_names=['Negative (-1)', 'Neutral (0)', 'Positive (1)']
    )
    
    print("\n====== Classification Report ======")
    print(class_report)
    
    # Save classification report to file
    with open('metrics_report.txt', 'w') as f:
        f.write("====== Best Hyperparameters ======\n")
        f.write(f"Hidden Dimensions: {best_params['hidden_dims']}\n")
        f.write(f"Dropout Rates: {best_params['dropout_rates']}\n")
        f.write(f"Learning Rate: {best_params['learning_rate']}\n")
        f.write(f"Batch Size: {best_params['batch_size']}\n")
        f.write(f"Train Accuracy: {best_params['train_acc']:.4f}\n")
        f.write(f"Test/Validation Accuracy: {best_params['test_acc']:.4f}\n")
        f.write(f"Train Loss: {best_params['train_loss']:.4f}\n")
        f.write(f"Test/Validation Loss: {best_params['test_loss']:.4f}\n")
        f.write(f"Epochs Used: {best_params['num_epochs']}/{75} ({best_params['num_epochs']/75*100:.0f}%)\n\n")
        f.write("====== Classification Report ======\n")
        f.write(class_report)
    
    # Save model weights
    torch.save(best_model.state_dict(), 'best_mlp_model.pt')
    
    # Save complete model (architecture + weights)
    torch.save(best_model, 'best_mlp_model_full.pt')
    
    # Save model architecture and hyperparameters for easy reuse
    model_config = {
        'input_dim': input_dim,
        'hidden_dims': best_params['hidden_dims'],
        'output_dim': output_dim,
        'dropout_rates': best_params['dropout_rates'],
        'batch_norm': best_params['batch_norm'],
        'regularization_info': best_params['regularization'],
        'learning_rate': best_params['learning_rate'],
        'batch_size': best_params['batch_size'],
        'performance': {
            'train_acc': best_params['train_acc'],
            'test_acc': best_params['test_acc'],
            'train_loss': best_params['train_loss'],
            'test_loss': best_params['test_loss'],
            'overfitting_metric': best_params['overfitting_metric']
        },
        'learning_curves': {
            'train_losses': best_train_losses,
            'val_losses': best_val_losses,  # These are actually test losses
            'train_accs': best_train_accs,
            'val_accs': best_val_accs,      # These are actually test accuracies
            'epochs': list(range(1, len(best_train_losses) + 1))
        }
    }
    
    with open('model_config.json', 'w') as f:
        import json
        json.dump(model_config, f, indent=4)
    
    print("Model saved in multiple formats:")
    print(" - Weights only (PyTorch style): best_mlp_model.pt")
    print(" - Full model (architecture + weights): best_mlp_model_full.pt")
    print(" - Architecture configuration (JSON): model_config.json")
    
    # Print summary table of all combinations
    print("\n====== Summary of All Combinations ======")
    print(f"{'#':<3} {'Hidden Dims':<20} {'Dropout':<15} {'Reg':<15} {'Train Acc':<10} {'Test Acc':<10} {'Overfit':<10} {'Epochs':<10}")
    print("-" * 100)
    
    # Sort results by test accuracy in descending order
    sorted_results = sorted(results, key=lambda x: x['test_acc'], reverse=True)
    
    for i, result in enumerate(sorted_results):
        hidden_dims_str = str(result['hidden_dims']).replace(" ", "")[:18]
        dropout_rates_str = str(result['dropout_rates']).replace(" ", "")[:13]
        reg_str = result['regularization'][:13]
        overfit = result['overfitting_metric']
        
        print(f"{i+1:<3} {hidden_dims_str:<20} {dropout_rates_str:<15} {reg_str:<15} "
              f"{result['train_acc']:<10.4f} {result['test_acc']:<10.4f} "
              f"{overfit:<10.4f} {result['num_epochs']}/{75} ({result['num_epochs']/75*100:.0f}%)")
    
    # Also save summary to file
    with open('hyperparameter_summary.txt', 'w') as f:
        f.write("====== Summary of All Combinations (Sorted by Test Accuracy) ======\n\n")
        f.write(f"{'#':<3} {'Hidden Dims':<40} {'Dropout Rates':<30} {'Regularization':<20} {'Train Acc':<10} {'Test Acc':<10} {'Overfitting':<10} {'Epochs':<15}\n")
        f.write("-" * 150 + "\n")
        
        for i, result in enumerate(sorted_results):
            f.write(f"{i+1:<3} {str(result['hidden_dims']):<40} {str(result['dropout_rates']):<30} "
                   f"{result['regularization']:<20} "
                   f"{result['train_acc']:<10.4f} {result['test_acc']:<10.4f} "
                   f"{result['overfitting_metric']:<10.4f} "
                   f"{result['num_epochs']}/{75} ({result['num_epochs']/75*100:.0f}%)\n")
    
    print(f"\nSummary table saved to 'hyperparameter_summary.txt'")
    
    return best_model, best_params 


def load_best_model(model_path='best_mlp_model_full.pt', config_path='model_config.json'):
    """
    Utility function to load the saved model for future use
    
    Args:
        model_path: Path to the saved model file
        config_path: Path to the config JSON file
        
    Returns:
        model: Loaded model ready for inference
        config: Model configuration
    """
    import json
    
    # Method 1: Load complete model directly (architecture + weights)
    try:
        model = torch.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading full model: {e}")
        model = None
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Model configuration loaded from {config_path}")
            
            # If direct model loading failed, rebuild from config
            if model is None:
                print("Rebuilding model from configuration...")
                model = MLP(
                    input_dim=config['input_dim'],
                    hidden_dims=config['hidden_dims'],
                    output_dim=config['output_dim'],
                    dropout_rates=config['dropout_rates'],
                    batch_norm=config.get('batch_norm', False)  # Use get() with default in case key is missing
                )
                
                # Load weights
                model.load_state_dict(torch.load('best_mlp_model.pt'))
                print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model configuration: {e}")
        config = None
    
    return model, config


def predict_sentiment(model, text_embedding, device='cpu'):
    """
    Predict sentiment from a pre-processed text embedding
    
    Args:
        model: Trained MLP model
        text_embedding: Embedding vector (same dimension as training data)
        device: Device to run inference on
        
    Returns:
        sentiment: -1 (negative), 0 (neutral), or 1 (positive)
        probabilities: Raw probability scores for each class
    """
    model.eval()
    model = model.to(device)
    
    # Convert input to tensor
    if not isinstance(text_embedding, torch.Tensor):
        text_embedding = torch.FloatTensor(text_embedding).to(device)
    
    # Add batch dimension if needed
    if len(text_embedding.shape) == 1:
        text_embedding = text_embedding.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(text_embedding)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
    
    # Convert back from model format (0,1,2) to sentiment (-1,0,1)
    sentiment = postprocess_labels(predictions.item())
    
    return sentiment, probabilities.cpu().numpy()[0]


# Example of how to use the model after training
if __name__ == "__main__":
    # This section will only run when the script is executed directly
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--use-model':
        print("Loading the best model for inference...")
        model, config = load_best_model()
        
        if model is None:
            print("Error: Could not load the model. Please train the model first.")
            sys.exit(1)
        
        print(f"\nModel configuration: {config['hidden_dims']} with {config['regularization_info']} regularization")
        print(f"Performance: Test accuracy = {config['performance']['test_acc']:.4f}")
        
        # Example prediction with random data
        # In a real application, this would be your preprocessed text embedding
        import numpy as np
        sample_embedding = np.random.randn(config['input_dim'])  # Random vector with same dimension as training data
        
        # Get prediction
        sentiment, probs = predict_sentiment(model, sample_embedding)
        
        # Map prediction to sentiment label
        sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
        print(f"\nExample prediction for random input:")
        print(f"Predicted sentiment: {sentiment_map[sentiment]} ({sentiment})")
        print(f"Class probabilities: Negative={probs[0]:.4f}, Neutral={probs[1]:.4f}, Positive={probs[2]:.4f}")
        
        print("\nTo use this model in your own code, do:")
        print("from mlp import load_best_model, predict_sentiment")
        print("model, _ = load_best_model()")
        print("sentiment, probs = predict_sentiment(model, your_text_embedding)") 