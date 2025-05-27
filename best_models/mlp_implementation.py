import numpy as np
import pickle
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

class MLP:
    """
    Multi-Layer Perceptron implementation from scratch
    """
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, max_epochs=1000, 
                 batch_size=32, activation='sigmoid', random_state=42):
        """
        Initialize MLP with specified architecture
        
        Parameters:
        -----------
        input_size : int
            Number of input features
        hidden_sizes : list
            List of integers specifying the number of neurons in each hidden layer
        output_size : int
            Number of output classes
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        max_epochs : int, default=1000
            Maximum number of training epochs
        batch_size : int, default=32
            Mini-batch size for training
        activation : str, default='sigmoid'
            Activation function ('sigmoid', 'relu', or 'tanh')
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.activation = activation
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(self.random_state)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * 0.1)
        self.biases.append(np.zeros((1, hidden_sizes[0])))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.weights.append(np.random.randn(hidden_sizes[i-1], hidden_sizes[i]) * 0.1)
            self.biases.append(np.zeros((1, hidden_sizes[i])))
        
        # Last hidden layer to output
        self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 0.1)
        self.biases.append(np.zeros((1, output_size)))
        
        # For tracking training progress
        self.train_losses = []
        self.val_losses = []
        self.train_time = 0
        
    def _sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def _relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """Derivative of ReLU function"""
        return np.where(x > 0, 1, 0)
    
    def _tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def _tanh_derivative(self, x):
        """Derivative of tanh function"""
        return 1 - np.power(x, 2)
    
    def _softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _activate(self, x):
        """Apply activation function"""
        if self.activation == 'sigmoid':
            return self._sigmoid(x)
        elif self.activation == 'relu':
            return self._relu(x)
        elif self.activation == 'tanh':
            return self._tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
    
    def _activate_derivative(self, x):
        """Apply derivative of activation function"""
        if self.activation == 'sigmoid':
            return self._sigmoid_derivative(x)
        elif self.activation == 'relu':
            return self._relu_derivative(x)
        elif self.activation == 'tanh':
            return self._tanh_derivative(x)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")
    
    def _forward(self, X):
        """
        Forward pass through the network
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, input_size)
            
        Returns:
        --------
        list
            List of activations for each layer
        """
        activations = [X]
        
        # Hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self._activate(z)
            activations.append(a)
        
        # Output layer (softmax)
        z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a_out = self._softmax(z_out)
        activations.append(a_out)
        
        return activations
    
    def _compute_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            One-hot encoded true labels of shape (n_samples, output_size)
        y_pred : numpy.ndarray
            Predicted probabilities of shape (n_samples, output_size)
            
        Returns:
        --------
        float
            Cross-entropy loss
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    def _backward(self, X, y, activations):
        """
        Backward pass through the network
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, input_size)
        y : numpy.ndarray
            One-hot encoded true labels of shape (n_samples, output_size)
        activations : list
            List of activations from forward pass
            
        Returns:
        --------
        tuple
            Weight and bias gradients
        """
        m = X.shape[0]
        
        # Initialize gradients
        dweights = [np.zeros_like(w) for w in self.weights]
        dbiases = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        delta = activations[-1] - y
        
        # Last layer gradients
        dweights[-1] = np.dot(activations[-2].T, delta) / m
        dbiases[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l+1].T) * self._activate_derivative(activations[l+1])
            dweights[l] = np.dot(activations[l].T, delta) / m
            dbiases[l] = np.sum(delta, axis=0, keepdims=True) / m
        
        return dweights, dbiases
    
    def _one_hot_encode(self, y):
        """
        Convert class labels to one-hot encoding
        
        Parameters:
        -----------
        y : numpy.ndarray
            Class labels of shape (n_samples,)
            
        Returns:
        --------
        numpy.ndarray
            One-hot encoded labels of shape (n_samples, output_size)
        """
        m = y.shape[0]
        y_one_hot = np.zeros((m, self.output_size))
        y_one_hot[np.arange(m), y.astype(int)] = 1
        return y_one_hot
    
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        """
        Train the MLP model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, input_size)
        y : numpy.ndarray
            Training labels of shape (n_samples,)
        X_val : numpy.ndarray, default=None
            Validation data of shape (n_val_samples, input_size)
        y_val : numpy.ndarray, default=None
            Validation labels of shape (n_val_samples,)
        verbose : bool, default=True
            Whether to print training progress
            
        Returns:
        --------
        self
        """
        start_time = time.time()
        
        # Convert labels to one-hot encoding
        y_one_hot = self._one_hot_encode(y)
        
        if X_val is not None and y_val is not None:
            y_val_one_hot = self._one_hot_encode(y_val)
        
        # Training loop
        for epoch in range(self.max_epochs):
            # Mini-batch gradient descent
            indices = np.random.permutation(X.shape[0])
            for start_idx in range(0, X.shape[0], self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y_one_hot[batch_indices]
                
                # Forward pass
                activations = self._forward(X_batch)
                
                # Backward pass
                dweights, dbiases = self._backward(X_batch, y_batch, activations)
                
                # Update weights and biases
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * dweights[i]
                    self.biases[i] -= self.learning_rate * dbiases[i]
            
            # Compute training loss
            activations = self._forward(X)
            train_loss = self._compute_loss(y_one_hot, activations[-1])
            self.train_losses.append(train_loss)
            
            # Compute validation loss if validation data is provided
            if X_val is not None and y_val is not None:
                val_activations = self._forward(X_val)
                val_loss = self._compute_loss(y_val_one_hot, val_activations[-1])
                self.val_losses.append(val_loss)
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch + 1}/{self.max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{self.max_epochs}, Train Loss: {train_loss:.4f}")
        
        self.train_time = time.time() - start_time
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, input_size)
            
        Returns:
        --------
        numpy.ndarray
            Predicted probabilities of shape (n_samples, output_size)
        """
        activations = self._forward(X)
        return activations[-1]
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, input_size)
            
        Returns:
        --------
        numpy.ndarray
            Predicted class labels of shape (n_samples,)
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, input_size)
        y : numpy.ndarray
            True labels of shape (n_samples,)
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        y_pred = self.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': self.train_time
        }


def run_mlp_experiments(dataset_name, cv_splits, configs, scaler=None):
    """
    Run MLP experiments with different configurations
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    cv_splits : list
        List of (X_train, y_train, X_test, y_test) tuples for cross-validation
    configs : list
        List of dictionaries containing MLP configurations
    scaler : object, default=None
        Scaler object for feature standardization
        
    Returns:
    --------
    dict
        Dictionary containing results for each configuration
    """
    results = {}
    
    for config_idx, config in enumerate(configs):
        config_name = f"config_{config_idx+1}"
        print(f"\nRunning MLP with {config_name} on {dataset_name}")
        print(f"Configuration: {config}")
        
        # Initialize metrics storage
        train_metrics = []
        test_metrics = []
        
        # Cross-validation
        for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(cv_splits):
            print(f"  Fold {fold_idx+1}/5")
            
            # Scale features if scaler is provided
            if scaler:
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            # Initialize and train MLP
            mlp = MLP(
                input_size=X_train.shape[1],
                hidden_sizes=config['hidden_sizes'],
                output_size=len(np.unique(y_train)),
                learning_rate=config['learning_rate'],
                max_epochs=config['max_epochs'],
                batch_size=config['batch_size'],
                activation=config['activation'],
                random_state=42
            )
            
            mlp.fit(X_train, y_train, verbose=False)
            
            # Evaluate on train and test sets
            train_metric = mlp.evaluate(X_train, y_train)
            test_metric = mlp.evaluate(X_test, y_test)
            
            train_metrics.append(train_metric)
            test_metrics.append(test_metric)
        
        # Calculate mean metrics across folds
        mean_train_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in train_metrics]),
            'precision': np.mean([m['precision'] for m in train_metrics]),
            'recall': np.mean([m['recall'] for m in train_metrics]),
            'f1': np.mean([m['f1'] for m in train_metrics]),
            'train_time': np.mean([m['train_time'] for m in train_metrics])
        }
        
        mean_test_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in test_metrics]),
            'precision': np.mean([m['precision'] for m in test_metrics]),
            'recall': np.mean([m['recall'] for m in test_metrics]),
            'f1': np.mean([m['f1'] for m in test_metrics])
        }
        
        # Store results
        results[config_name] = {
            'config': config,
            'train_metrics': mean_train_metrics,
            'test_metrics': mean_test_metrics
        }
        
        print(f"  Mean Train Accuracy: {mean_train_metrics['accuracy']:.4f}")
        print(f"  Mean Test Accuracy: {mean_test_metrics['accuracy']:.4f}")
        print(f"  Mean Train Time: {mean_train_metrics['train_time']:.4f} seconds")
    
    return results


if __name__ == "__main__":
    # Load datasets and CV splits
    print("Loading datasets...")
    with open('datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    
    print(" Loading cross-validation splits...")
    with open('cv_splits.pkl', 'rb') as f:
        cv_splits = pickle.load(f)
    
    print("Experimenting with MLP configurations...")
    # Define MLP configurations to experiment with
    mlp_configs = [
        # Configuration 1: Small network with sigmoid activation
        {
            'hidden_sizes': [10],
            'learning_rate': 0.01,
            'max_epochs': 500,
            'batch_size': 32,
            'activation': 'sigmoid'
        }]
    
    # run MLP experiments for each dataset
    for dataset_name, cv_split in cv_splits.items():
        print(f"\nRunning MLP experiments for dataset: {dataset_name}")
        
        # Optionally scale features
        scaler = StandardScaler()
        
        # Run experiments
        results = run_mlp_experiments(
            dataset_name=dataset_name,
            cv_splits=cv_split,
            configs=mlp_configs,
            scaler=scaler
        )

        # Save results
        with open(f'mlp_results_{dataset_name}.pkl', 'wb') as f:
            pickle.dump(results, f)


