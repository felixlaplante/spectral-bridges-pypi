# Spectral Bridges

Spectral Bridges is a Python package that implements a novel clustering algorithm combining k-means and spectral clustering techniques. It leverages efficient affinity matrix computation and merges clusters based on a connectivity measure inspired by SVM's margin concept. This package is designed to provide robust clustering solutions, particularly suited for large datasets.

## Features

- **Spectral Bridges Algorithm**: Integrates k-means and spectral clustering with efficient affinity matrix calculation for improved clustering results.
- **Scalability**: Designed to handle large datasets by optimizing cluster formation through advanced affinity matrix computations.
- **Customizable**: Parameters such as number of clusters, iterations, and random state allow flexibility in clustering configurations.

## Installation

You can install the package via pip:

```bash
pip install spectral-bridges
```

## Usage

### Example

```python
from spectralbridges import SpectralBridges
import numpy as np

# Generate sample data
np.random.seed(0)
X = np.random.rand(100, 10)  # Replace with your dataset

# Initialize and fit Spectral Bridges
model = SpectralBridges(n_clusters=5, n_nodes=10, random_state=42)
model.fit(X)

# Predict clusters for new data points
new_data = np.random.rand(20, 10)  # Replace with new data
predicted_clusters = model.predict(new_data)

print("Predicted clusters:", predicted_clusters)
```
