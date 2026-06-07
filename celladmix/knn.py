"""KNN graph construction and neighborhood count matrices."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def knn_adjacency_matrix(df, k, weighted=False):
    """Create a KNN adjacency matrix from spatial coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have columns 'x', 'y', 'z'.
    k : int
        Number of nearest neighbors.
    weighted : bool
        If True, use distances as weights; otherwise binary.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency matrix of shape (n, n).
    """
    coords = df[['x', 'y', 'z']].values
    nn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    n = len(coords)
    # Build COO format
    row = np.repeat(np.arange(n), k)
    col = indices.ravel()
    if weighted:
        data = distances.ravel()
    else:
        data = np.ones(len(row), dtype=np.float64)

    adj = sparse.csr_matrix((data, (row, col)), shape=(n, n))
    return adj


def knn_count_matrix(df, k, all_genes=None, include_i=False):
    """Get KNN gene counts for each molecule.

    Parameters
    ----------
    df : pandas.DataFrame
        Must have columns 'x', 'y', 'z', 'gene'.
    k : int
        Neighborhood size.
    all_genes : list or None
        All genes to include as columns.
    include_i : bool
        Whether to include self-molecule as a neighbor.

    Returns
    -------
    scipy.sparse.csr_matrix
        Molecule-by-gene matrix of KNN counts.
    """
    adj = knn_adjacency_matrix(df, k)
    if include_i:
        adj = adj + sparse.eye(adj.shape[0], format='csr')

    if all_genes is None:
        genes = pd.Categorical(df['gene'])
    else:
        genes = pd.Categorical(df['gene'], categories=all_genes)

    gene_indicator = sparse.csr_matrix(
        (np.ones(len(genes)), (np.arange(len(genes)), genes.codes)),
        shape=(len(genes), len(genes.categories))
    )

    ncm = adj @ gene_indicator
    ncm = sparse.csr_matrix(ncm)
    return ncm, list(genes.categories)


def get_knn_counts_all(df, h, include_i=False, n_cores=1):
    """Get KNN count matrices for molecules in all cells.

    Parameters
    ----------
    df : pandas.DataFrame
        Molecule coordinates with 'x', 'y', 'z', 'gene', 'cell', 'mol_id'.
    h : int
        Neighborhood size.
    include_i : bool
        Whether to include self-molecule.
    n_cores : int
        Number of cores (unused in Python, kept for API compatibility).

    Returns
    -------
    pandas.DataFrame
        KNN counts per molecule with cell, mol_id, gene_i columns.
    """
    h = int(h)
    cell_counts = df['cell'].value_counts()
    cells = cell_counts[cell_counts > h].index.tolist()
    df_sub = df[df['cell'].isin(cells)].copy()

    all_genes = sorted(df_sub['gene'].unique())
    gene_cat = pd.Categorical(all_genes)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}

    results = []
    for cell_id, df_cell in df_sub.groupby('cell'):
        ncm, cols = knn_count_matrix(df_cell, k=h, all_genes=all_genes, include_i=include_i)
        ncm_dense = np.asarray(ncm.todense())

        cell_result = df_cell[['cell', 'mol_id']].copy()
        cell_result = cell_result.rename(columns={'mol_id': 'id'})
        cell_result['gene_i'] = df_cell['gene'].values

        for i, gene in enumerate(all_genes):
            cell_result[gene] = ncm_dense[:, i]

        results.append(cell_result)

    N = pd.concat(results, ignore_index=True)
    N = N.fillna(0)
    return N
