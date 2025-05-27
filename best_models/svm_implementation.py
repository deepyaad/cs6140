import numpy as np
import pickle
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

class SVM:
    """
    Support Vector Machine implementation from scratch using gradient descent
    """
    def __init__(self, kernel='linear', C=1.0, learning_rate=0.01, max_epochs=1000, 
                 batch_size=32, tol=1e-3, random_state=42):
        """
        Initialize SVM with specified parameters
        
        Parameters:
        -----------
        kernel : str, default='linear'
            Kernel type ('linear', 'polynomial', 'rbf', or 'sigmoid')
        C : float, default=1.0
            Regularization parameter
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        max_epochs : int, default=1000
            Maximum number of training epochs
        batch_size : int, default=32
            Mini-batch size for training
        tol : float, default=1e-3
            Tolerance for stopping criterion
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.kernel = kernel
        self.C = C
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.tol = tol
        self.random_state = random_state
        
        # Kernel parameters
        self.degree = 3  # for polynomial kernel
        self.gamma = 'scale'  # for rbf, polynomial and sigmoid kernels
        self.coef0 = 0.0  # for polynomial and sigmoid kernels
        
        # For tracking training progress
        self.train_losses = []
        self.train_time = 0
        
        # Model parameters
        self.w = None
        self.b = 0.0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_coeffs = None
        
    def _linear_kernel(self, X1, X2):
        """Linear kernel: K(x, y) = x^T y"""
        return np.dot(X1, X2.T)
    
    def _polynomial_kernel(self, X1, X2):
        """Polynomial kernel: K(x, y) = (gamma * x^T y + coef0)^degree"""
        return (self.gamma * np.dot(X1, X2.T) + self.coef0) ** self.degree
    
    def _rbf_kernel(self, X1, X2):
        """RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)"""
        # Compute squared Euclidean distance between each pair of points
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * distances)
    
    def _sigmoid_kernel(self, X1, X2):
        """Sigmoid kernel: K(x, y) = tanh(gamma * x^T y + coef0)"""
        return np.tanh(self.gamma * np.dot(X1, X2.T) + self.coef0)
    
    def _get_kernel(self, X1, X2=None):
        """Apply the selected kernel function"""
        if X2 is None:
            X2 = X1
            
        if self.kernel == 'linear':
            return self._linear_kernel(X1, X2)
        elif self.kernel == 'polynomial':
            return self._polynomial_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X1, X2)
        elif self.kernel == 'sigmoid':
            return self._sigmoid_kernel(X1, X2)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")
    
    def _compute_loss(self, X, y):
        """
        Compute hinge loss for SVM
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features)
        y : numpy.ndarray
            Class labels of shape (n_samples,)
            
        Returns:
        --------
        float
            Hinge loss
        """
        if self.kernel == 'linear':
            # For linear kernel, we can directly use w and b
            margins = y * (np.dot(X, self.w) + self.b)
        else:
            # For non-linear kernels, use the kernel trick
            K = self._get_kernel(X, self.support_vectors)
            margins = y * (np.dot(K, self.support_vector_coeffs * self.support_vector_labels) + self.b)
        
        # Hinge loss: max(0, 1 - y_i * (w^T x_i + b))
        hinge_losses = np.maximum(0, 1 - margins)
        
        # Add regularization term for linear kernel
        if self.kernel == 'linear':
            reg_term = 0.5 * np.sum(self.w ** 2)
            return np.mean(hinge_losses) + self.C * reg_term
        else:
            return np.mean(hinge_losses)
    
    def fit(self, X, y, verbose=True):
        """
        Train the SVM model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features)
        y : numpy.ndarray
            Training labels of shape (n_samples,)
            Must be in {-1, 1}
        verbose : bool, default=True
            Whether to print training progress
            
        Returns:
        --------
        self
        """
        start_time = time.time()
        
        # Set gamma if it's 'scale'
        if self.gamma == 'scale':
            self.gamma = 1.0 / (X.shape[1] * X.var())
        
        # Initialize model parameters
        n_samples, n_features = X.shape
        
        if self.kernel == 'linear':
            # For linear kernel, we directly optimize w and b
            self.w = np.zeros(n_features)
            self.b = 0.0
            
            # Set random seed
            np.random.seed(self.random_state)
            
            # Training loop
            for epoch in range(self.max_epochs):
                # Mini-batch gradient descent
                indices = np.random.permutation(n_samples)
                for start_idx in range(0, n_samples, self.batch_size):
                    batch_indices = indices[start_idx:start_idx + self.batch_size]
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]
                    
                    # Compute predictions
                    margins = y_batch * (np.dot(X_batch, self.w) + self.b)
                    
                    # Compute gradients
                    mask = margins < 1
                    dw = self.w - self.C * np.sum(y_batch[mask].reshape(-1, 1) * X_batch[mask], axis=0)
                    db = -self.C * np.sum(y_batch[mask])
                    
                    # Update parameters
                    self.w -= self.learning_rate * dw
                    self.b -= self.learning_rate * db
                
                # Compute training loss
                train_loss = self._compute_loss(X, y)
                self.train_losses.append(train_loss)
                
                # Print progress
                if verbose and (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {train_loss:.4f}")
                
                # Check for convergence
                if epoch > 0 and abs(self.train_losses[-1] - self.train_losses[-2]) < self.tol:
                    if verbose:
                        print(f"Converged at epoch {epoch + 1}")
                    break
        
        else:
            # For non-linear kernels, use the kernel trick
            # Store all training data as potential support vectors
            self.support_vectors = X.copy()
            self.support_vector_labels = y.copy()
            self.support_vector_coeffs = np.zeros(n_samples)
            
            # Compute kernel matrix
            K = self._get_kernel(X)
            
            # Set random seed
            np.random.seed(self.random_state)
            
            # Training loop
            for epoch in range(self.max_epochs):
                # Mini-batch gradient descent
                indices = np.random.permutation(n_samples)
                for start_idx in range(0, n_samples, self.batch_size):
                    batch_indices = indices[start_idx:start_idx + self.batch_size]
                    
                    # Compute predictions for the batch
                    batch_K = K[batch_indices]
                    y_batch = y[batch_indices]
                    
                    # Compute margins
                    margins = y_batch * (np.dot(batch_K, self.support_vector_coeffs * self.support_vector_labels) + self.b)
                    
                    # Compute gradients
                    mask = margins < 1
                    dalpha = np.zeros(n_samples)
                    
                    for i, idx in enumerate(batch_indices):
                        if mask[i]:
                            dalpha[idx] = self.C * y[idx]
                    
                    db = -self.C * np.sum(y_batch[mask])
                    
                    # Update parameters
                    self.support_vector_coeffs += self.learning_rate * dalpha
                    self.b -= self.learning_rate * db
                
                # Compute training loss
                train_loss = self._compute_loss(X, y)
                self.train_losses.append(train_loss)
                
                # Print progress
                if verbose and (epoch + 1) % 100 == 0:
                    print(f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {train_loss:.4f}")
                
                # Check for convergence
                if epoch > 0 and abs(self.train_losses[-1] - self.train_losses[-2]) < self.tol:
                    if verbose:
                        print(f"Converged at epoch {epoch + 1}")
                    break
            
            # Identify true support vectors (non-zero coefficients)
            sv_mask = np.abs(self.support_vector_coeffs) > 1e-5
            self.support_vectors = self.support_vectors[sv_mask]
            self.support_vector_labels = self.support_vector_labels[sv_mask]
            self.support_vector_coeffs = self.support_vector_coeffs[sv_mask]
            
            if verbose:
                print(f"Number of support vectors: {len(self.support_vectors)}")
        
        self.train_time = time.time() - start_time
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns:
        --------
        numpy.ndarray
            Predicted class labels of shape (n_samples,)
        """
        if self.kernel == 'linear':
            # For linear kernel, we can directly use w and b
            decision_values = np.dot(X, self.w) + self.b
        else:
            # For non-linear kernels, use the kernel trick
            K = self._get_kernel(X, self.support_vectors)
            decision_values = np.dot(K, self.support_vector_coeffs * self.support_vector_labels) + self.b
        
        return np.sign(decision_values)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features)
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
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_time': self.train_time
        }


def run_svm_experiments(dataset_name, cv_splits, kernels, C_values, scaler=None):
    """
    Run SVM experiments with different kernels and C values
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    cv_splits : list
        List of (X_train, y_train, X_test, y_test) tuples for cross-validation
    kernels : list
        List of kernel types to experiment with
    C_values : list
        List of C values to experiment with
    scaler : object, default=None
        Scaler object for feature standardization
        
    Returns:
    --------
    dict
        Dictionary containing results for each kernel and C value
    """
    results = {}
    
    for kernel in kernels:
        kernel_results = {}
        
        for C in C_values:
            config_name = f"{kernel}_C{C}"
            print(f"\nRunning SVM with {config_name} on {dataset_name}")
            
            # Initialize metrics storage
            train_metrics = []
            test_metrics = []

            # skip SVM with polynomial_C1.0 on half_circles_10k
            if dataset_name == 'half_circles_10k' and kernel == 'polynomial' and C >= 1.0:
                print(f"  Skipping SVM with {kernel} kernel and C={C} on {dataset_name} dataset")
                continue
            
            # Cross-validation
            for fold_idx, (X_train, y_train, X_test, y_test) in enumerate(cv_splits):
                print(f"  Fold {fold_idx+1}/5")
                
                # Convert labels to {-1, 1} for SVM
                y_train_svm = np.where(y_train == 0, -1, 1)
                y_test_svm = np.where(y_test == 0, -1, 1)
                
                # Scale features if scaler is provided
                if scaler:
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                
                # Initialize and train SVM
                svm = SVM(
                    kernel=kernel,
                    C=C,
                    learning_rate=0.01,
                    max_epochs=500,
                    batch_size=32,
                    tol=1e-3,
                    random_state=42
                )
                
                svm.fit(X_train, y_train_svm, verbose=False)
                
                # Evaluate on train and test sets
                train_metric = svm.evaluate(X_train, y_train_svm)
                test_metric = svm.evaluate(X_test, y_test_svm)
                
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
            kernel_results[C] = {
                'train_metrics': mean_train_metrics,
                'test_metrics': mean_test_metrics
            }
            
            print(f"  Mean Train Accuracy: {mean_train_metrics['accuracy']:.4f}")
            print(f"  Mean Test Accuracy: {mean_test_metrics['accuracy']:.4f}")
            print(f"  Mean Train Time: {mean_train_metrics['train_time']:.4f} seconds")
        
        results[kernel] = kernel_results
    
    return results


if __name__ == "__main__":
    # Load datasets and CV splits
    with open('datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
    
    with open('cv_splits.pkl', 'rb') as f:
        cv_splits = pickle.load(f)
    
    # Define SVM parameters
    kernels = ['linear', 'polynomial', 'rbf', 'sigmoid']
    C_values = [0.1, 1.0, 10.0]
    
    # Initialize scaler for feature standardization
    scaler = StandardScaler()

    # Run SVM experiments for each dataset
    for dataset_name, split in cv_splits.items():

        print(f"\nRunning SVM experiments for dataset: {dataset_name}")
        results = run_svm_experiments(dataset_name, split, kernels, C_values, scaler)
        
        # Save results to file
        with open(f'svm_results_{dataset_name}.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved for dataset: {dataset_name}")

  