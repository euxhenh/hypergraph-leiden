import faiss
import numpy as np
from sklearn.neighbors import kneighbors_graph


def full_knn(x, n_neighbors=15, mode='connectivity'):
    """
    Compute the exact nearest neighbors graph using sklearn.
    """
    kg = kneighbors_graph(x, n_neighbors=n_neighbors, mode=mode)
    sources, targets = kg.nonzero()
    weights = np.array(kg[sources, targets])[0]

    return sources, targets, weights


def faiss_knn(x, n_neighbors=15):
    """
    Compute approximate neighbors using the faiss library.
    """
    n_samples = x.shape[0]
    n_features = x.shape[1]
    x = np.ascontiguousarray(x)  # important

    index = faiss.IndexHNSWFlat(n_features, min(15, n_samples))
    index.add(x)

    weights, targets = index.search(x, n_neighbors)

    sources = np.repeat(np.arange(n_samples), n_neighbors)
    targets = targets.flatten()
    assert -1 not in targets

    return sources, targets, np.full(len(sources), 1)


def knn_auto(x, n_neighbors=15, mode='connectivity', method='auto'):
    if method == 'auto':
        if x.shape[0] > 5000:
            print("Dataset is too large. Finding approximate neighbors.")
            return faiss_knn(x, n_neighbors=n_neighbors)
        return full_knn(x, n_neighbors=n_neighbors, mode=mode)
    elif method == 'full':
        return full_knn(x, n_neighbors=n_neighbors, mode=mode)

    return faiss_knn(x, n_neighbors=n_neighbors)
