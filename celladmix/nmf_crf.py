"""NMF factorization and CRF-based molecule segmentation."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors

from .knn import knn_adjacency_matrix, knn_count_matrix


def _get_knn_counts_for_nmf(df, h, include_i=True, all_genes=None):
    """Internal: get KNN count matrix for NMF input (optimized batch version)."""
    h = int(h)
    cell_counts = df['cell'].value_counts()
    cells = cell_counts[cell_counts > h].index.tolist()
    df_sub = df[df['cell'].isin(cells)].copy()

    if all_genes is None:
        all_genes = sorted(df_sub['gene'].unique())

    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    n_genes = len(all_genes)

    # Pre-compute gene indices for all molecules
    gene_indices = np.array([gene_to_idx.get(g, -1) for g in df_sub['gene']])

    # Batch KNN on all molecules at once
    coords = df_sub[['x', 'y', 'z']].values
    nn = NearestNeighbors(n_neighbors=h, algorithm='auto')
    nn.fit(coords)
    _, nn_indices = nn.kneighbors(coords)

    # Build sparse adjacency matrix
    n = len(coords)
    row = np.repeat(np.arange(n), h)
    col = nn_indices.ravel()
    data = np.ones(len(row), dtype=np.float64)
    adj = sparse.csr_matrix((data, (row, col)), shape=(n, n))

    if include_i:
        adj = adj + sparse.eye(n, format='csr')

    # Build gene indicator matrix
    valid_mask = gene_indices >= 0
    gene_indicator = sparse.csr_matrix(
        (np.ones(n), (np.arange(n), gene_indices)),
        shape=(n, n_genes)
    )

    # Compute KNN counts: adj @ gene_indicator
    X = adj @ gene_indicator
    X = sparse.csr_matrix(X)

    # Filter zero rows
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    mask = row_sums > 0
    X = X[mask]
    mol_ids = df_sub['mol_id'].values[mask].tolist()

    X_dense = np.asarray(X.todense(), dtype=np.float64)
    return X_dense, mol_ids, all_genes


def run_knn_nmf(df, k=5, h=20, mols_test=None, nmol_dsamp=None, n_cores=1):
    """Run KNN-based NMF on spatial transcriptomics data.

    Parameters
    ----------
    df : pandas.DataFrame
        Molecule data with columns: x, y, z, gene, cell, mol_id.
    k : int
        Number of NMF factors.
    h : int
        Neighborhood size for KNN.
    mols_test : list or None
        Specific molecules to test.
    nmol_dsamp : int or None
        Number of molecules to downsample to.
    n_cores : int
        Unused, kept for API compatibility.

    Returns
    -------
    dict
        NMF results with keys: W, H, gene_sub, feature_names.
    """
    df = df.copy()
    df.index = df['mol_id']

    if mols_test is not None:
        cells_test = df.loc[df.index.isin(mols_test), 'cell'].unique()
        df = df[df['cell'].isin(cells_test)]

    cell_counts = df['cell'].value_counts()
    cells = cell_counts[cell_counts > h].index.tolist()
    df = df[df['cell'].isin(cells)]
    genes = sorted(df['gene'].unique())

    X_dense, mol_ids, all_genes = _get_knn_counts_for_nmf(df, h, include_i=True, all_genes=genes)

    if mols_test is not None:
        mask = np.isin(mol_ids, mols_test)
        X_dense = X_dense[mask]
        mol_ids = [m for m, keep in zip(mol_ids, mask) if keep]

    if nmol_dsamp is not None and len(X_dense) > nmol_dsamp:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X_dense), nmol_dsamp, replace=False)
        X_dense = X_dense[idx]
        mol_ids = [mol_ids[i] for i in idx]

    # Filter zero rows/cols
    row_sums = X_dense.sum(axis=1)
    col_sums = X_dense.sum(axis=0)
    row_mask = row_sums > 0
    col_mask = col_sums > 0
    X_dense = X_dense[row_mask][:, col_mask]
    mol_ids = [m for m, keep in zip(mol_ids, row_mask) if keep]
    genes_filtered = [g for g, keep in zip(all_genes, col_mask) if keep]

    # Weight matrix Z (inverse column sums)
    col_sums_X = X_dense.sum(axis=0)
    Z = np.ones_like(X_dense) / col_sums_X[np.newaxis, :]

    # Run NMF with LS-NMF (multiplicative update with weight)
    # Pre-multiply by Z to approximate weighted LS-NMF (same as R's ls-nmf)
    X_weighted = X_dense * Z

    # Use sklearn's optimized NMF with multiple seeds (same as R's nrun=5)
    best_nmf = None
    best_err = np.inf

    for seed in range(5):
        nmf_model = NMF(n_components=k, init='random', max_iter=500, random_state=seed)
        W = nmf_model.fit_transform(X_weighted)
        H = nmf_model.components_
        err = np.sum((X_weighted - W @ H) ** 2)
        if err < best_err:
            best_err = err
            best_nmf = (W, H)

    W, H = best_nmf

    gene_sub = 'gene_sub' in df.columns

    return {
        'W': W,
        'H': H,
        'mol_ids': mol_ids,
        'feature_names': genes_filtered,
        'gene_sub': gene_sub,
        'k': k,
    }


def _get_graph(df_cell, h=10, distinct_edges=False):
    """Build a graph from molecule coordinates within a cell.

    Parameters
    ----------
    df_cell : pandas.DataFrame
        Molecule data for a single cell with x, y, z, gene, mol_id.
    h : int
        Neighborhood size.
    distinct_edges : bool
        Whether to simplify to distinct edges.

    Returns
    -------
    dict
        Graph with 'adj' (sparse matrix), 'vertices' (DataFrame).
    """
    h = int(h)
    if h >= len(df_cell):
        h = min(h, len(df_cell) - 1)
    if h < 2:
        raise ValueError('Need to remove cells with only two or fewer molecules')

    coords = df_cell[['x', 'y', 'z']].values
    nn = NearestNeighbors(n_neighbors=h, algorithm='auto')
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)

    n = len(coords)
    mol_ids = df_cell['mol_id'].values
    genes = df_cell['gene'].values

    # Build adjacency
    rows = np.repeat(np.arange(n), h)
    cols = indices.ravel()
    data = np.ones(len(rows))
    adj = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    if distinct_edges:
        adj = (adj > 0).astype(np.float64)

    return {
        'adj': adj,
        'mol_ids': mol_ids,
        'genes': genes,
        'n_vertices': n,
    }


def _decode_lbp(adj, node_pot, k, max_iter=200, t=0.05):
    """Loopy Belief Propagation for CRF decoding.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        Adjacency matrix.
    node_pot : ndarray of shape (n, k)
        Node potentials.
    k : int
        Number of states.
    max_iter : int
        Maximum iterations.
    t : float
        Transition probability.

    Returns
    -------
    ndarray of shape (n,)
        Decoded labels (0-indexed).
    """
    n = adj.shape[0]

    # Edge potentials: same label gets 1-(k-1)*t, different gets t
    A = np.full((k, k), t)
    np.fill_diagonal(A, 1 - (k - 1) * t)

    # Initialize messages: message[i][j] = message from i to j
    # Use dict of sparse-ish arrays for memory efficiency
    adj_coo = sparse.triu(adj, k=1).tocoo()
    edges = list(zip(adj_coo.row, adj_coo.col))

    # messages[edge_idx][direction] -> array of shape (k,)
    # direction 0: i->j, direction 1: j->i
    messages = {}
    for idx, (i, j) in enumerate(edges):
        messages[(i, j)] = np.ones(k) / k
        messages[(j, i)] = np.ones(k) / k

    # Build neighbor lists
    neighbors = {i: [] for i in range(n)}
    for idx, (i, j) in enumerate(edges):
        neighbors[i].append(j)
        neighbors[j].append(i)

    for iteration in range(max_iter):
        new_messages = {}
        for i, j in edges:
            # Message from i to j
            # product of all messages into i except from j
            incoming = np.ones(k)
            for ni in neighbors[i]:
                if ni != j:
                    incoming *= messages[(ni, i)]

            belief_i = node_pot[i] * incoming
            msg = A.T @ belief_i
            msg = msg / (msg.sum() + 1e-300)
            new_messages[(i, j)] = msg

            # Message from j to i
            incoming = np.ones(k)
            for nj in neighbors[j]:
                if nj != i:
                    incoming *= messages[(nj, j)]

            belief_j = node_pot[j] * incoming
            msg = A.T @ belief_j
            msg = msg / (msg.sum() + 1e-300)
            new_messages[(j, i)] = msg

        messages = new_messages

    # Decode
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        incoming = np.ones(k)
        for ni in neighbors[i]:
            incoming *= messages[(ni, i)]
        belief = node_pot[i] * incoming
        labels[i] = np.argmax(belief)

    return labels


def run_crf(df_cell, lds, k, h=10, t=0.05, max_iter=200):
    """Run CRF segmentation for a single cell.

    Parameters
    ----------
    df_cell : pandas.DataFrame
        Molecule data for one cell.
    lds : pandas.DataFrame
        Gene loadings with 'gene' column and factor columns.
    k : int
        Number of factors.
    h : int
        Neighborhood size.
    t : float
        Transition probability.
    max_iter : int
        Max LBP iterations.

    Returns
    -------
    pandas.DataFrame
        Columns: id, factor, cell.
    """
    G = _get_graph(df_cell, h=h)
    adj = G['adj']
    genes = G['genes']
    mol_ids = G['mol_ids']

    # Build node potentials from loadings
    gene_to_loading = {}
    for _, row in lds.iterrows():
        gene_to_loading[row['gene']] = row.iloc[1:].values.astype(np.float64)

    node_pot = np.zeros((len(genes), k))
    for i, gene in enumerate(genes):
        if gene in gene_to_loading:
            node_pot[i] = gene_to_loading[gene]
        else:
            node_pot[i] = 1.0 / k  # uniform if gene not found

    # Decode
    labels = _decode_lbp(adj, node_pot, k, max_iter=max_iter, t=t)

    df_out = pd.DataFrame({
        'id': mol_ids,
        'factor': labels + 1,  # R uses 1-indexed factors
        'cell': df_cell['cell'].iloc[0],
    })
    return df_out


def project_nmf(df, res, nmf_h, n_cores=1):
    """Project NMF loadings to all genes using NNLS.

    Parameters
    ----------
    df : pandas.DataFrame
        Molecule data.
    res : dict
        NMF results from run_knn_nmf.
    nmf_h : int
        Neighborhood size for projection.
    n_cores : int
        Unused.

    Returns
    -------
    ndarray
        Projected gene loadings matrix (genes x factors).
    """
    from scipy.optimize import nnls

    W = res['W']  # molecules x factors
    mol_ids_tested = res['mol_ids']
    all_genes = res['feature_names']

    cells_tested = df.loc[df['mol_id'].isin(mol_ids_tested), 'cell'].unique()
    df_sub = df[df['cell'].isin(cells_tested)].copy()

    cell_counts = df_sub['cell'].value_counts()
    cells = cell_counts[cell_counts > nmf_h].index.tolist()
    df_sub = df_sub[df_sub['cell'].isin(cells)]

    mols_keep = [m for m in mol_ids_tested if m in df_sub['mol_id'].values]
    mol_idx = [mol_ids_tested.index(m) for m in mols_keep]
    W_sub = W[mol_idx]

    # Build KNN count matrix for projection
    results = []
    for cell_id, df_cell in df_sub.groupby('cell'):
        nmf_h_int = int(nmf_h)
        if nmf_h_int >= len(df_cell):
            nmf_h_int = len(df_cell) - 1
        if nmf_h_int < 2:
            continue
        ncm, cols = knn_count_matrix(df_cell, k=nmf_h_int, all_genes=all_genes, include_i=True)
        # Keep only mols_keep rows
        mol_mask = df_cell['mol_id'].isin(mols_keep).values
        ncm_sub = ncm[mol_mask]
        results.append(ncm_sub)

    if not results:
        return np.zeros((len(all_genes), res['k']))

    knn_res = sparse.vstack(results)
    knn_res = sparse.csr_matrix(knn_res)

    # Filter zero columns
    col_sums = np.asarray(knn_res.sum(axis=0)).ravel()
    col_mask = col_sums > 0
    knn_res = knn_res[:, col_mask]
    genes_filtered = [g for g, keep in zip(all_genes, col_mask) if keep]

    # Project using NNLS
    knn_dense = knn_res.T.toarray()  # genes x molecules
    k = res['k']
    lds_proj = np.zeros((len(genes_filtered), k))

    for i in range(len(genes_filtered)):
        for j in range(k):
            x, _ = nnls(W_sub, knn_dense[i])
            lds_proj[i] = x
            break  # compute full vector at once below

    # Actually compute full NNLS per gene
    lds_proj = np.zeros((len(genes_filtered), k))
    for i in range(len(genes_filtered)):
        x, _ = nnls(W_sub, knn_dense[i])
        lds_proj[i] = x

    # Replace zeros with minimum non-zero value
    min_nonzero = lds_proj[lds_proj > 0].min() if np.any(lds_proj > 0) else 0
    lds_proj[lds_proj == 0] = min_nonzero

    return lds_proj


def run_crf_all(df, res, num_nn=10, same_label_ratio=5,
                normalize_by='gene', proj_h=None, n_cores=1):
    """Run CRF segmentation for all cells.

    Parameters
    ----------
    df : pandas.DataFrame
        Molecule data with x, y, z, gene, cell, mol_id.
    res : dict
        NMF results from run_knn_nmf.
    num_nn : int
        Neighborhood size for CRF.
    same_label_ratio : float
        Transition ratio.
    normalize_by : str
        'gene', 'factor', or 'gene.factor'.
    proj_h : int or None
        Neighborhood size for NMF projection.
    n_cores : int
        Unused.

    Returns
    -------
    pandas.DataFrame
        Column 'factor' with molecule factor assignments.
    """
    df = df.copy()
    df.index = df['mol_id']
    k = res['k']

    # Get loadings
    if res['gene_sub']:
        if proj_h is None:
            proj_h = num_nn * 2
        lds_raw = project_nmf(df, res, proj_h, n_cores)
        lds = pd.DataFrame(lds_raw, index=res['feature_names'],
                          columns=[str(i+1) for i in range(k)])
        lds = lds.reset_index().rename(columns={'index': 'gene'})
    else:
        H = res['H']  # factors x genes
        lds = pd.DataFrame(H.T, index=res['feature_names'],
                          columns=[str(i+1) for i in range(k)])
        lds = lds.reset_index().rename(columns={'index': 'gene'})

    # Add missing genes with minimum loading
    all_genes = df['gene'].unique()
    missing_genes = [g for g in all_genes if g not in lds['gene'].values]
    if missing_genes:
        min_val = lds.iloc[:, 1:].values.min()
        missing_df = pd.DataFrame(
            np.full((len(missing_genes), k + 1), min_val),
            columns=lds.columns
        )
        missing_df['gene'] = missing_genes
        lds = pd.concat([lds, missing_df], ignore_index=True)

    # Normalize loadings
    factor_cols = [str(i+1) for i in range(k)]
    lds_values = lds[factor_cols].values.astype(np.float64)

    if normalize_by == 'gene':
        col_sums = lds_values.sum(axis=0)
        lds_values = lds_values / col_sums[np.newaxis, :]
    elif normalize_by == 'factor':
        row_sums = lds_values.sum(axis=1, keepdims=True)
        lds_values = lds_values / row_sums
    elif normalize_by == 'gene.factor':
        row_sums = lds_values.sum(axis=1, keepdims=True)
        lds_values = lds_values / row_sums
        col_sums = lds_values.sum(axis=0)
        lds_values = lds_values / col_sums[np.newaxis, :]

    lds[factor_cols] = lds_values

    # Transition probability
    tval = 1.0 / (same_label_ratio + k - 1)
    max_iter = 200

    # Run CRF per cell
    results = []
    for cell_id, df_cell in df.groupby('cell'):
        df_cell = df_cell.copy()
        if len(df_cell) <= num_nn:
            continue
        try:
            crf_res = run_crf(df_cell, lds, k=k, h=num_nn, t=tval, max_iter=max_iter)
            results.append(crf_res)
        except Exception:
            continue

    if not results:
        return pd.DataFrame({'factor': []}, index=pd.Index([], name='mol_id'))

    crf_all = pd.concat(results, ignore_index=True)
    crf_all.index = crf_all['id']
    crf_all = crf_all.loc[df.index]
    return crf_all[['factor']]
