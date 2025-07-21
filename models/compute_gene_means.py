import scanpy as sc
import numpy as np
import pandas as pd

adata = sc.read_h5ad('data/adata_Training.h5ad', backed='r')
n_cells = adata.n_obs
sample_size = min(1000, n_cells)
np.random.seed(42)
cell_indices = np.random.choice(n_cells, sample_size, replace=False)
adata_sample = adata[cell_indices, :].to_memory()
X = adata_sample.X
if not isinstance(X, np.ndarray):
    X = X.toarray()
gene_names = list(adata_sample.var.index)
means = np.mean(X, axis=0)
pd.Series(means, index=gene_names).to_csv('data/gene_means.csv')
print(f'Done. Means saved to data/gene_means.csv (sampled from {sample_size} cells)') 