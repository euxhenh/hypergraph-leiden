import scanpy as sc
from sklearn.decomposition import PCA
from umap import UMAP


def preprocess(
        adata,
        filter_genes=True,
        normalize=True,
        log1p=True,
        high_var=True,
        scale=True):
    """Preprocess the data.

    Parameters
    __________
    adata: AnnData object
    filter_genes: bool
        Filter genes expressed in less than 5 cells.
    normalize: bool
        Normalize so that the total count for each cell is equal.
    log1p: bool
        log(x+1) transform the data.
    high_var: bool
        Only consider highly variable genes
    scale: bool
        Scale the data to unit variance and zero mean.
    """
    adata = adata.copy()
    print(f"adata has shape {adata.shape}")
    if filter_genes:
        sc.pp.filter_genes(adata, min_cells=5)
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
    if log1p:
        sc.pp.log1p(adata)
    if high_var:
        sc.pp.highly_variable_genes(
            adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
        adata = adata[:, adata.var.highly_variable]
    if scale:
        sc.pp.scale(adata, max_value=10)
    print(f"After preprocessing, adata has shape {adata.shape}")

    return adata


def reduce(adata):
    """Add PCA and UMAP embeddings.
    """
    pca = PCA(40)
    ump = UMAP()
    adata.obsm['x_emb'] = pca.fit_transform(adata.X)
    adata.obsm['x_emb_2d'] = ump.fit_transform(adata.obsm['x_emb'])
