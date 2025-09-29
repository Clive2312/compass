# Cluster Search Baseline

This folder contains baseline tests for clustering algorithms and cluster-based search.

## Configuration

Dataset configurations are stored in `config.json`. This file contains the parameters and settings for each dataset used in the experiments.

## Running the Tests

### Clustering Algorithm
To run the clustering algorithm and get cluster assignments, along with testing search accuracy under these assignments:
```bash
cd build && ./kmeans_cluster d=[dataset]
```

### Cluster Search (HE-based)
To run HE-based cluster search:
```bash
cd build && ./kmeans_cluster r=[role] d=[dataset]
```

## Directory Structure
- `config.json` - Dataset configurations
- `kmeans_cluster/` - K-means clustering and accuracy testing
- `cluster_search/` - HE-based cluster search implementation