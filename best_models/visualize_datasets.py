import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Load datasets
with open('datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)

# Create output directory for visualizations
os.makedirs('visualizations', exist_ok=True)

# Set up figure parameters
plt.figure(figsize=(16, 12))
colors = ['#ff9999', '#66b3ff']
markers = ['o', 'x']

# Function to visualize dataset
def visualize_dataset(X, y, title, filename):
    plt.figure(figsize=(8, 6))
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
    plt.savefig(f'visualizations/{filename}.png', dpi=300)
    plt.close()
    print(f"Saved visualization: {filename}.png")

# Visualize each dataset
print("Visualizing datasets...")

# Half-circles datasets
X_half_circles_1k, y_half_circles_1k = datasets['half_circles_1k']
visualize_dataset(X_half_circles_1k, y_half_circles_1k, 
                 'Half-circles Dataset (1000 points)', 
                 'half_circles_1k')

X_half_circles_10k, y_half_circles_10k = datasets['half_circles_10k']
visualize_dataset(X_half_circles_10k, y_half_circles_10k, 
                 'Half-circles Dataset (10000 points)', 
                 'half_circles_10k')

# Moons datasets
X_moons_1k, y_moons_1k = datasets['moons_1k']
visualize_dataset(X_moons_1k, y_moons_1k, 
                 'Moons Dataset (1000 points)', 
                 'moons_1k')

X_moons_10k, y_moons_10k = datasets['moons_10k']
visualize_dataset(X_moons_10k, y_moons_10k, 
                 'Moons Dataset (10000 points)', 
                 'moons_10k')

print("All visualizations completed successfully!")
