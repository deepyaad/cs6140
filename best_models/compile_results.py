import numpy as np
import pickle # used for loading MLP and SVM results
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Create directory for results
os.makedirs('results', exist_ok=True)

# Load MLP and SVM results
print("Loading results...")
try:
    with open('mlp_results.pkl', 'rb') as f:
        mlp_results = pickle.load(f)
    print("MLP results loaded successfully.")
except FileNotFoundError:
    print("MLP results file not found.")
    mlp_results = {}

try:
    with open('svm_results.pkl', 'rb') as f:
        svm_results = pickle.load(f)
    print("SVM results loaded successfully.")
except FileNotFoundError:
    print("SVM results file not found.")
    svm_results = {}

# Function to create performance table
def create_performance_table(mlp_results, svm_results):
    """
    Create a comprehensive performance table for all models and datasets
    
    Parameters:
    -----------
    mlp_results : dict
        Dictionary containing MLP results
    svm_results : dict
        Dictionary containing SVM results
        
    Returns:
    --------
    pandas.DataFrame
        Performance table
    """
    # Initialize lists to store results
    rows = []
    
    # Process MLP results
    if mlp_results:
        for dataset_name, configs in mlp_results.items():
            for config_name, results in configs.items():
                # Extract metrics
                train_metrics = results['train_metrics']
                test_metrics = results['test_metrics']
                config = results['config']
                
                # Create row
                row = {
                    'Dataset': dataset_name,
                    'Model': 'MLP',
                    'Configuration': f"{config_name}: {config['hidden_sizes']}, {config['activation']}",
                    'Train Accuracy': train_metrics['accuracy'],
                    'Test Accuracy': test_metrics['accuracy'],
                    'Train Precision': train_metrics['precision'],
                    'Test Precision': test_metrics['precision'],
                    'Train Recall': train_metrics['recall'],
                    'Test Recall': test_metrics['recall'],
                    'Train F1': train_metrics['f1'],
                    'Test F1': test_metrics['f1'],
                    'Train Time (s)': train_metrics['train_time']
                }
                
                rows.append(row)
    
    # Process SVM results
    if svm_results:
        for dataset_name, kernels in svm_results.items():
            for kernel_name, c_values in kernels.items():
                for c_value, results in c_values.items():
                    # Extract metrics
                    train_metrics = results['train_metrics']
                    test_metrics = results['test_metrics']
                    
                    # Create row
                    row = {
                        'Dataset': dataset_name,
                        'Model': 'SVM',
                        'Configuration': f"{kernel_name} kernel, C={c_value}",
                        'Train Accuracy': train_metrics['accuracy'],
                        'Test Accuracy': test_metrics['accuracy'],
                        'Train Precision': train_metrics['precision'],
                        'Test Precision': test_metrics['precision'],
                        'Train Recall': train_metrics['recall'],
                        'Test Recall': test_metrics['recall'],
                        'Train F1': train_metrics['f1'],
                        'Test F1': test_metrics['f1'],
                        'Train Time (s)': train_metrics['train_time']
                    }
                    
                    rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort by dataset and test accuracy
    if not df.empty:
        df = df.sort_values(['Dataset', 'Test Accuracy'], ascending=[True, False])
    
    return df

# Create performance table
print("Creating performance table...")
performance_table = create_performance_table(mlp_results, svm_results)

# Save performance table to CSV
if not performance_table.empty:
    performance_table.to_csv('results/performance_table.csv', index=False)
    print("Performance table saved to results/performance_table.csv")
    
    # Display top 5 models for each dataset
    for dataset_name in performance_table['Dataset'].unique():
        print(f"\nTop 5 models for {dataset_name} (by test accuracy):")
        dataset_results = performance_table[performance_table['Dataset'] == dataset_name]
        top_5 = dataset_results.sort_values('Test Accuracy', ascending=False).head(5)
        print(top_5[['Model', 'Configuration', 'Test Accuracy', 'Train Time (s)']])
else:
    print("No results available to create performance table.")

# Function to find best model for each dataset
def find_best_models(performance_table):
    """
    Find the best model for each dataset
    
    Parameters:
    -----------
    performance_table : pandas.DataFrame
        Performance table
        
    Returns:
    --------
    dict
        Dictionary containing best model for each dataset
    """
    best_models = {}
    
    if not performance_table.empty:
        for dataset_name in performance_table['Dataset'].unique():
            dataset_results = performance_table[performance_table['Dataset'] == dataset_name]
            best_model = dataset_results.loc[dataset_results['Test Accuracy'].idxmax()]
            best_models[dataset_name] = {
                'Model': best_model['Model'],
                'Configuration': best_model['Configuration'],
                'Test Accuracy': best_model['Test Accuracy'],
                'Train Time': best_model['Train Time (s)']
            }
    
    return best_models

# Find best models
best_models = find_best_models(performance_table)

# Save best models to text file
if best_models:
    with open('results/best_models.txt', 'w') as f:
        f.write("Best Models for Each Dataset\n")
        f.write("===========================\n\n")
        
        for dataset_name, model_info in best_models.items():
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Best Model: {model_info['Model']}\n")
            f.write(f"Configuration: {model_info['Configuration']}\n")
            f.write(f"Test Accuracy: {model_info['Test Accuracy']:.4f}\n")
            f.write(f"Training Time: {model_info['Train Time']:.4f} seconds\n\n")
    
    print("Best models saved to results/best_models.txt")

print("Results compilation complete!")