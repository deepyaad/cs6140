import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.preprocessing import StandardScaler
from mlp_implementation import MLP

# Create directory for visualizations
os.makedirs('results/visualizations', exist_ok=True)

# Load datasets and MLP results
print("Loading datasets and results...")
with open('datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)

with open('cv_splits.pkl', 'rb') as f:
    cv_splits = pickle.load(f)

with open('mlp_results.pkl', 'rb') as f:
    mlp_results = pickle.load(f)

# Define MLP configurations
mlp_configs = [
    # Configuration 1: Small network with sigmoid activation
    {
        'hidden_sizes': [10],
        'learning_rate': 0.01,
        'max_epochs': 500,
        'batch_size': 32,
        'activation': 'sigmoid'
    },
    # Configuration 2: Medium network with ReLU activation
    {
        'hidden_sizes': [20, 10],
        'learning_rate': 0.01,
        'max_epochs': 500,
        'batch_size': 32,
        'activation': 'relu'
    },
    # Configuration 3: Larger network with tanh activation
    {
        'hidden_sizes': [30, 15],
        'learning_rate': 0.01,
        'max_epochs': 500,
        'batch_size': 32,
        'activation': 'tanh'
    }
]

# Function to create a mesh grid for decision boundary visualization
def create_mesh_grid(X, padding=0.5):
    """
    Create a mesh grid for decision boundary visualization
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features matrix of shape (n_samples, 2)
    padding : float, default=0.5
        Padding around the data range
        
    Returns:
    --------
    tuple
        (xx, yy) mesh grid
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    return xx, yy

# Function to visualize decision boundaries
def visualize_decision_boundary(model, X, y, title, filename, scaler=None):
    """
    Visualize decision boundary for a model
    
    Parameters:
    -----------
    model : object
        Trained model with predict method
    X : numpy.ndarray
        Features matrix of shape (n_samples, 2)
    y : numpy.ndarray
        Target vector of shape (n_samples,)
    title : str
        Plot title
    filename : str
        Output filename
    scaler : object, default=None
        Scaler object for feature standardization
    """
    plt.figure(figsize=(10, 8))
    
    # Create mesh grid
    xx, yy = create_mesh_grid(X)
    
    # Create mesh points
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    
    # Scale features if scaler is provided
    if scaler:
        X_scaled = scaler.transform(X)
        X_mesh_scaled = scaler.transform(X_mesh)
    else:
        X_scaled = X
        X_mesh_scaled = X_mesh
    
    # Predict class labels
    Z = model.predict(X_mesh_scaled)
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    
    # Plot data points
    colors = ['#ff9999', '#66b3ff']
    markers = ['o', 'x']
    for i in range(2):  # Two classes
        plt.scatter(X[y == i, 0], X[y == i, 1], 
                   c=colors[i], marker=markers[i], 
                   label=f'Class {i}', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/visualizations/{filename}.png', dpi=300)
    plt.close()
    print(f"Saved visualization: {filename}.png")

# Function to train and visualize best MLP model for each dataset
def visualize_best_mlp_models():
    """
    Train and visualize best MLP model for each dataset
    """
    # Initialize scaler
    scaler = StandardScaler()
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\nVisualizing best MLP model for {dataset_name}...")
        
        # Find best MLP configuration for this dataset
        best_config_idx = 0
        best_accuracy = 0
        
        for config_idx, config_results in mlp_results[dataset_name].items():
            test_accuracy = config_results['test_metrics']['accuracy']
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_config_idx = int(config_idx.split('_')[1]) - 1
        
        best_config = mlp_configs[best_config_idx]
        print(f"Best configuration: {best_config}")
        
        # Scale features
        X_scaled = scaler.fit_transform(X)
        
        # Train MLP with best configuration
        mlp = MLP(
            input_size=X.shape[1],
            hidden_sizes=best_config['hidden_sizes'],
            output_size=len(np.unique(y)),
            learning_rate=best_config['learning_rate'],
            max_epochs=best_config['max_epochs'],
            batch_size=best_config['batch_size'],
            activation=best_config['activation'],
            random_state=42
        )
        
        mlp.fit(X_scaled, y, verbose=False)
        
        # Visualize decision boundary
        config_name = f"config_{best_config_idx+1}"
        title = f"MLP Decision Boundary - {dataset_name}\n{config_name}: {best_config['hidden_sizes']}, {best_config['activation']}"
        filename = f"mlp_{dataset_name}_{config_name}"
        
        visualize_decision_boundary(mlp, X, y, title, filename, scaler)

# Visualize best MLP models
visualize_best_mlp_models()

print("\nAll visualizations completed successfully!")
