# Spectral Bridges

Spectral Bridges is a Python package that implements a novel clustering algorithm combining k-means and spectral clustering techniques. It leverages efficient affinity matrix computation and merges clusters based on a connectivity measure inspired by SVM's margin concept. This package is designed to provide robust clustering solutions, particularly suited for large datasets.

## Features

- **Spectral Bridges Algorithm**: Integrates k-means and spectral clustering with efficient affinity matrix calculation for improved clustering results.
- **Scalability**: Designed to handle large datasets by optimizing cluster formation through advanced affinity matrix computations.
- **Customizable**: Parameters such as number of clusters, iterations, and random state allow flexibility in clustering configurations.

## Speed

Starting with version 1.0.0, Spectral Bridges not only utilizes FAISS's efficient k-means implementation but also uses a scikit-learn method clone for centroid initialization which is much faster (over 2x improvement).

## Installation

You can install the package via pip:

```bash
pip install spectral-bridges
```

## Usage

### Example

```python
import spectralbridges as sb
import numpy as np

# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 10)  # Replace with your dataset

# Define range of nodes to evaluate
n_nodes_range = [10, 15, 20]

# Find the optimal number of nodes
optimal_model, best_n_nodes_, best_normalized_eigengap_ = sb.best_nodes(
    X, n_clusters=5, n_nodes_range=n_nodes_range
)

print("Optimal number of nodes:", best_n_nodes_)
print("Best mean normalized eigengap:", best_normalized_eigengap_)

# Initialize and fit Spectral Bridges with a specified number of nodes and random seed
custom_model = sb.SpectralBridges(n_clusters=5, n_nodes=12, random_state=42)
custom_model.fit(X)

# Predict clusters for new data points
new_data = np.random.rand(20, 10)  # Replace with new data
predicted_clusters = custom_model.predict(new_data)

print("Predicted clusters:", predicted_clusters)
```
