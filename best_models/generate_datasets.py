import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
import pickle

# Set random seed for reproducibility
np.random.seed(42)

def generate_half_circles(n_samples):    
    noisy_circles = make_circles(
      n_samples=n_samples, factor=0.5, noise=0.05
    )
    
    # noisy_circles[0] is the data X, noisy_circles[1] is the labels Y
    return noisy_circles[0], noisy_circles[1]

def generate_moons(n_samples):
    noisy_moons = make_moons(
    n_samples=n_samples, noise=0.05,random_state=6
)

    # noisy_moons[0] is the data X, noisy_moons[1] is the labels Y
    return noisy_moons[0], noisy_moons[1]

# Generate datasets
print("Generating datasets...")

# Half-circles datasets
X_half_circles_1k, y_half_circles_1k = generate_half_circles(1000)
X_half_circles_10k, y_half_circles_10k = generate_half_circles(10000)

# Moons datasets
X_moons_1k, y_moons_1k = generate_moons(1000)
X_moons_10k, y_moons_10k = generate_moons(10000)

# Save datasets
print("Saving datasets...")
datasets = {
    'half_circles_1k': (X_half_circles_1k, y_half_circles_1k),
    'half_circles_10k': (X_half_circles_10k, y_half_circles_10k),
    'moons_1k': (X_moons_1k, y_moons_1k),
    'moons_10k': (X_moons_10k, y_moons_10k)
}

with open('datasets.pkl', 'wb') as f:
    pickle.dump(datasets, f)

print("Datasets generated and saved successfully!")
print(f"Half-circles 1k: {X_half_circles_1k.shape[0]} points")
print(f"Half-circles 10k: {X_half_circles_10k.shape[0]} points")
print(f"Moons 1k: {X_moons_1k.shape[0]} points")
print(f"Moons 10k: {X_moons_10k.shape[0]} points")