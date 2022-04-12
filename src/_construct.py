import numpy as np
from scipy.sparse import coo_matrix

def get_hg_reduction(x):
    n, m = x.shape
    incidence = np.zeros((n, m))  # shape (n, m)
    hyperedge_W = np.zeros(m)
    for i in tqdm(range(m)):
        ori = x[:, i].copy()

        row = x[:, i].copy()
        thresh = threshold_otsu(row)
        indices = np.where(ori > thresh)[0]
        if indices.size == 0:
            continue
        incidence[indices, i] = 1
        num = ori[indices].mean()
        hyperedge_W[i] = num

    # hyperedge_W[np.diag_indices(n)] = np.log2(
        #     hyperedge_W[np.diag_indices(n)] + 1)
    incidence = coo_matrix(incidence)

    hyperedge_D = np.array(incidence.sum(axis=0))  # shape (m, m)
    hyperedge_D = np.clip(hyperedge_D - 1, 0, np.Inf)
    hyperedge_D[hyperedge_D > 0] = 1 / hyperedge_D[hyperedge_D > 0]

    reduced_hypergraph = (incidence.multiply(
        hyperedge_W * hyperedge_D)) @ incidence.T
    # reduced_hypergraph = reduced_hypergraph.todense()
    reduced_hypergraph[np.diag_indices(n)] = 0

    return reduced_hypergraph
