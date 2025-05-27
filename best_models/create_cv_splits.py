import numpy as np
import pickle
from sklearn.model_selection import KFold

def create_k_fold_splits(X, y, k=5, random_state=42):
    """
    Create k-fold cross-validation splits for a dataset.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Features matrix of shape (n_samples, n_features)
    y : numpy.ndarray
        Target vector of shape (n_samples,)
    k : int, default=5
        Number of folds for cross-validation
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    list of tuples
        Each tuple contains (X_train, y_train, X_test, y_test) for a fold
    """
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    # Create splits
    splits = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        splits.append((X_train, y_train, X_test, y_test))
    
    return splits

# Load datasets
print("Loading datasets...")
with open('datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)

# Create k-fold splits for each dataset
print("Creating 5-fold cross-validation splits...")
k = 5
cv_splits = {}

for dataset_name, (X, y) in datasets.items():
    print(f"Creating splits for {dataset_name}...")
    cv_splits[dataset_name] = create_k_fold_splits(X, y, k=k)
    
    # Print information about the splits
    for i, (X_train, y_train, X_test, y_test) in enumerate(cv_splits[dataset_name]):
        print(f"  Fold {i+1}: Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# Save the splits
print("Saving cross-validation splits...")
with open('cv_splits.pkl', 'wb') as f:
    pickle.dump(cv_splits, f)

print("Cross-validation splits created and saved successfully!")
