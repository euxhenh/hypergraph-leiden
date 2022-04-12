import numpy as np
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from tqdm import tqdm


def get_hg_reduction(X, n_clusters=4):
    """Computes a connectivity matrix for X based on the hypergraph method.
    For every gene (column), connect via a hyperedge all the cells (rows)
    where the gene is highly expressed.

    Parameters
    __________
    X: ndarray of shape (n_samples, n_features)
        The data matrix.
    n_clusters: int
        Number of clusters to use for KMeans for every gene.

    Returns
    _______
    reduced_adj: ndarray of shape (n_samples, n_samples)
        The corrected adjacency matrix.
    """
    n_samples, n_features = X.shape

    # Initialize
    # incidence matrix has a 1 in (i, j) if sample i is in hyperedge j
    incidence_mat = np.zeros((n_samples, n_features))
    # hyperedge_weight matrix contains a "weight" for every hyperedge.
    hyperedge_weight_mat = np.zeros(n_features)
    km = KMeans(n_clusters=n_clusters)

    for feature in tqdm(range(n_features)):
        # Skip if there are less unique values than n_clusters
        if np.unique(X[:, feature]).size < n_clusters:
            continue
        # Find clusters for the given gene
        labels = km.fit_predict(X[:, feature].reshape(-1, 1))
        # If only 1 cluster was found, skip this gene.
        if len(km.cluster_centers_) == 1:
            continue
        # Next, we need to find which cluster has the highest mean.
        # The means here coincide with the cluster centers since
        # the samples are 1D.
        best_label = np.argmax(km.cluster_centers_.flatten())
        argidx = np.argwhere(labels==best_label).flatten()
        # Skip this gene if the top cluster contains <= 3 points, since
        # these are probably outliers.
        if argidx.size <= 3:
            continue
        # Points in the cluster with ID best_label will be assigned to
        # a hyperedge
        incidence_mat[argidx, feature] = 1
        # We assign a weight to this hyperedge based on log fold change,
        # i.e., the log ratio between the expression of this gene in the
        # top cluster vs the rest.
        hyperedge_weight_mat[feature] = (
            np.max(km.cluster_centers_) # same as X[labels==best_label, feature].mean()
            - X[labels!=best_label, feature].mean()
        )

    # The clique reduction of the hypergraph is given by
    # A = H @ W @ H.T
    # where H = incidence_mat and W = hyperedge_weight_mat.

    # In the formula above, the degree of every node is over-counted.
    # To correct it, we instead compute
    # A = H @ W @ inverse(D - I) @ H.T
    # where D = edge_degree_mat and I is identity matrix.

    # how many points are assigned to a hyperedge
    edge_degree_mat = incidence_mat.sum(axis=0)
    # remove empty hyperedges
    non_zero_idx = edge_degree_mat.nonzero()[0]
    edge_degree_mat = edge_degree_mat[non_zero_idx]
    incidence_mat = incidence_mat[:, non_zero_idx]
    hyperedge_weight_mat = hyperedge_weight_mat[non_zero_idx]
    # subtract identity
    correction_term = edge_degree_mat - 1
    assert correction_term.min() >= 0
    # invert and be careful not to invert 0 valued entries
    correction_term[correction_term > 0] = 1 / correction_term[correction_term > 0]

    # The incidence matrix is sparse, so we convert it to
    # a sparse matrix for faster multiplication.
    incidence_mat = coo_matrix(incidence_mat)

    reduced_adj = (
        (incidence_mat.multiply(hyperedge_weight_mat * correction_term))
        @ incidence_mat.T
    )
    reduced_adj[np.diag_indices(n_samples)] = 0

    return reduced_adj
