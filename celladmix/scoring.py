"""Admixture scoring functions — Bayesian contamination estimation."""

import numpy as np
import pandas as pd
from scipy import sparse


def estimate_cell_adjacency(df_spatial, random_shift=1e-3, edge_max_mad=4, n_cores=1):
    """Estimate cell adjacency using Delaunay triangulation.

    Parameters
    ----------
    df_spatial : pandas.DataFrame
        Must have columns: x, y, cell_type, cell.
    random_shift : float
        Small random jitter to avoid degenerate triangulation.
    edge_max_mad : float
        MAD multiplier for edge distance filtering.
    n_cores : int
        Unused.

    Returns
    -------
    pandas.DataFrame
        Columns: cell_s, cell_e, cts, cte.
    """
    from scipy.spatial import Delaunay

    required = ['x', 'y', 'cell']
    if not all(c in df_spatial.columns for c in required):
        raise ValueError(f'Required columns: {required}')

    # Accept both 'cell_type' and 'celltype'
    ct_col = 'cell_type' if 'cell_type' in df_spatial.columns else 'celltype'
    if ct_col not in df_spatial.columns:
        raise ValueError(f'Required column: cell_type or celltype')

    df = df_spatial.copy()
    df['cell_type'] = df[ct_col]
    df = df.dropna(subset=['cell_type'])

    rng = np.random.RandomState(0)
    df['x'] = df['x'] + rng.uniform(-random_shift, random_shift, len(df))
    df['y'] = df['y'] + rng.uniform(-random_shift, random_shift, len(df))

    coords = df[['x', 'y']].values
    tri = Delaunay(coords)

    # Extract edges from triangles
    simplices = tri.simplices
    edges = set()
    for s in simplices:
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            e = (min(s[i], s[j]), max(s[i], s[j]))
            edges.add(e)

    edges = np.array(list(edges))

    # Build adjacency dataframe
    adj_df = pd.DataFrame({
        'xs': coords[edges[:, 0], 0],
        'xe': coords[edges[:, 1], 0],
        'ys': coords[edges[:, 0], 1],
        'ye': coords[edges[:, 1], 1],
        'cts': df['cell_type'].values[edges[:, 0]],
        'cte': df['cell_type'].values[edges[:, 1]],
        'cell_s': df['cell'].values[edges[:, 0]],
        'cell_e': df['cell'].values[edges[:, 1]],
    })

    adj_df['dist'] = np.sqrt((adj_df['xs'] - adj_df['xe'])**2 + (adj_df['ys'] - adj_df['ye'])**2)

    # Filter by distance
    med_dist = adj_df['dist'].median()
    mad_dist = (adj_df['dist'] - med_dist).abs().median()
    adj_df = adj_df[adj_df['dist'] < med_dist + edge_max_mad * mad_dist]

    # Keep only inter-cell edges
    adj_df = adj_df[adj_df['cell_s'] != adj_df['cell_e']]
    adj_df = adj_df[['cell_s', 'cell_e', 'cts', 'cte']].drop_duplicates()

    # Add reverse edges
    reverse = adj_df[['cell_e', 'cell_s', 'cte', 'cts']].copy()
    reverse.columns = adj_df.columns
    adj_df = pd.concat([adj_df, reverse]).drop_duplicates()

    return adj_df


def estimate_cell_type_adjacency(cell_adj_df):
    """Estimate cell type adjacency matrix.

    Parameters
    ----------
    cell_adj_df : pandas.DataFrame
        From estimate_cell_adjacency.

    Returns
    -------
    pandas.DataFrame
        Cell type adjacency matrix.
    """
    grouped = cell_adj_df.groupby(['cell_s', 'cts', 'cte']).size().reset_index(name='n')
    mean_counts = grouped.groupby(['cts', 'cte'])['n'].mean().reset_index()
    adj_mat = mean_counts.pivot(index='cte', columns='cts', values='n')
    adj_mat = adj_mat.fillna(0)

    # Ensure square and symmetric ordering
    all_cts = sorted(set(adj_mat.index) | set(adj_mat.columns))
    adj_mat = adj_mat.reindex(index=all_cts, columns=all_cts, fill_value=0)

    return adj_mat


def estimate_gene_prob_per_type(cm, cell_annot, expr_thres=0, use_counts=False):
    """Estimate gene expression probabilities per cell type.

    Parameters
    ----------
    cm : pandas.DataFrame or scipy.sparse.csr_matrix or ndarray
        Count matrix (genes x cells). If DataFrame, index = gene names, columns = cell IDs.
    cell_annot : pandas.Series
        Cell type annotations indexed by cell ID.
    expr_thres : float
        Expression threshold.
    use_counts : bool
        If True, use summed counts; otherwise use fraction expressing.

    Returns
    -------
    pandas.DataFrame
        Gene probabilities per cell type (cell_types x genes).
    """
    # Preserve gene names from DataFrame index
    gene_names = None
    if isinstance(cm, pd.DataFrame):
        gene_names = cm.index.tolist()
        cm_dense = cm.values
    elif sparse.issparse(cm):
        cm_dense = cm.toarray()
    else:
        cm_dense = np.asarray(cm)

    cell_types = cell_annot.unique()
    results = {}

    for ct in cell_types:
        cids = cell_annot[cell_annot == ct].index.tolist()

        # Find columns matching cell IDs
        if isinstance(cm, pd.DataFrame):
            col_idx = [cm.columns.get_loc(c) for c in cids if c in cm.columns]
        elif hasattr(cm, 'columns'):
            col_idx = [cm.columns.get_loc(c) for c in cids if c in cm.columns]
        else:
            col_idx = list(range(cm_dense.shape[1]))

        sub = cm_dense[:, col_idx] if col_idx else np.zeros((cm_dense.shape[0], 1))

        if use_counts:
            probs = sub.sum(axis=1)
            probs = probs / probs.sum() if probs.sum() > 0 else probs
        else:
            probs = (sub > 0).mean(axis=1)

        results[ct] = probs

    # Build DataFrame with gene names
    if gene_names is not None:
        prob_df = pd.DataFrame(results, index=gene_names)
    else:
        prob_df = pd.DataFrame(results)

    return prob_df.T


def estimate_gene_scores(prob_gene_per_type, cell_type_adj_mat, cell_type):
    """Estimate gene contamination and reference scores.

    Parameters
    ----------
    prob_gene_per_type : pandas.DataFrame
        Gene probabilities per cell type.
    cell_type_adj_mat : pandas.DataFrame
        Cell type adjacency matrix.
    cell_type : str
        Current cell type.

    Returns
    -------
    pandas.DataFrame
        Columns: cont, ref.
    """
    adj_col = cell_type_adj_mat[cell_type]
    other_types = [t for t in adj_col.index if t != cell_type]
    cont_fracs = adj_col[other_types]
    cont_fracs = cont_fracs / cont_fracs.sum()

    # Contamination score: weighted sum of other types' probabilities
    prob_other = prob_gene_per_type.loc[other_types]
    weighted = prob_other.values * cont_fracs.values[:, np.newaxis]
    cont_scores = weighted.sum(axis=0)

    # Reference score: this type's probabilities
    ref_scores = prob_gene_per_type.loc[cell_type].values

    genes = prob_gene_per_type.columns
    return pd.DataFrame({'cont': cont_scores, 'ref': ref_scores}, index=genes)


def estimate_gene_contamination_probabilities(gene_scores, p_c=None, prior_cont_prob=None):
    """Estimate gene contamination probabilities using Bayesian formula.

    Parameters
    ----------
    gene_scores : pandas.DataFrame
        From estimate_gene_scores.
    p_c : float or None
        Proportion of contamination cluster.
    prior_cont_prob : float or None
        Prior contamination probability.

    Returns
    -------
    pandas.Series
        Gene contamination probabilities.
    """
    if p_c is None:
        p_cu = 0.5
    else:
        p_cu = p_c * prior_cont_prob

    cont = gene_scores['cont'].values
    ref = gene_scores['ref'].values

    probs = p_cu * cont / (p_cu * cont + (1 - p_cu) * ref)
    probs = np.nan_to_num(probs, nan=0.0)

    return pd.Series(probs, index=gene_scores.index)


def estimate_cell_admixture_fractions(gene_cont_probs, cell_type_cm, mask=None):
    """Estimate cell contamination fractions.

    Parameters
    ----------
    gene_cont_probs : pandas.Series
        Gene contamination probabilities.
    cell_type_cm : scipy.sparse.csr_matrix or ndarray
        Count matrix for cells of this type (genes x cells).
    mask : array-like or None
        Gene mask.

    Returns
    -------
    numpy.ndarray
        Per-cell admixture fractions.
    """
    if mask is not None:
        gene_cont_probs = gene_cont_probs.copy()
        gene_cont_probs[~mask] = 0

    if sparse.issparse(cell_type_cm):
        cm_dense = cell_type_cm.toarray()
    else:
        cm_dense = np.asarray(cell_type_cm)

    weighted = cm_dense * gene_cont_probs.values[:, np.newaxis]
    fracs = weighted.sum(axis=0) / cm_dense.sum(axis=0)
    fracs = np.nan_to_num(fracs, nan=0.0)

    return fracs


def _mask_out_genes(cont_probs, genes, all_name='all'):
    """Mask out specific genes from contamination probabilities."""
    if genes is None:
        return cont_probs

    if not isinstance(genes, dict):
        genes_list = list(set(cont_probs.index) & set(genes))
        cont_probs.loc[genes_list] = 0
        return cont_probs

    for name, gene_list in genes.items():
        gene_list = list(set(cont_probs.index) & set(gene_list))
        if len(gene_list) == 0:
            continue
        if name == all_name:
            cont_probs.loc[gene_list] = 0
        else:
            if name in cont_probs.columns:
                cont_probs.loc[gene_list, name] = 0

    return cont_probs


def estimate_contamination_scores(
    cm_rna, cm_spatial, annot_rna, annot_spatial, cell_type_adj_mat,
    p_c=0.25, signal_thres=0.25, min_expr_frac=0.05, max_prob_high=0.1,
    exclude_cell_types=None, exclude_genes=None, use_counts=False, adjust=True
):
    """Estimate contamination scores using Bayesian approach.

    Parameters
    ----------
    cm_rna : DataFrame or sparse matrix
        scRNA-seq count matrix (genes x cells).
    cm_spatial : DataFrame or sparse matrix
        Spatial count matrix (genes x cells).
    annot_rna : Series
        Cell type annotations for scRNA.
    annot_spatial : Series
        Cell type annotations for spatial.
    cell_type_adj_mat : DataFrame
        Cell type adjacency matrix.
    p_c : float
        Prior contamination proportion.
    signal_thres : float
        Threshold for selecting admixture genes.
    min_expr_frac : float
        Minimum expression fraction.
    max_prob_high : float
        Max probability for high expression genes.
    exclude_cell_types : list or None
        Cell types to exclude.
    exclude_genes : list or None
        Genes to exclude.
    use_counts : bool
        Whether to use counts.
    adjust : bool
        Whether to adjust scores.

    Returns
    -------
    dict
        Keys: cont_probs_per_type, doublet_scores.
    """
    if exclude_cell_types is None:
        exclude_cell_types = []

    # Find common genes
    if hasattr(cm_rna, 'index'):
        common_genes = sorted(set(cm_rna.index) & set(cm_spatial.index))
    else:
        common_genes = sorted(set(cm_rna.index) if hasattr(cm_rna, 'index') else set(cm_spatial.index))

    cm_rna = cm_rna.loc[common_genes] if hasattr(cm_rna, 'loc') else cm_rna
    cm_spatial = cm_spatial.loc[common_genes] if hasattr(cm_spatial, 'loc') else cm_spatial

    prob_gene_per_type = estimate_gene_prob_per_type(cm_rna, annot_rna, use_counts=use_counts)
    prior_cont_probs = 1 - np.diag(cell_type_adj_mat.values) / cell_type_adj_mat.values.sum(axis=0)
    prior_cont_probs = pd.Series(prior_cont_probs, index=cell_type_adj_mat.columns)

    # Expression fraction mask
    expr_frac_mask = prob_gene_per_type.values.max(axis=0) > min_expr_frac

    # High expression probabilities
    prob_gene_per_type_high = estimate_gene_prob_per_type(cm_rna, annot_rna, expr_thres=1)

    all_ctypes = [ct for ct in cell_type_adj_mat.columns if ct not in exclude_cell_types]

    cont_probs_per_type = {}
    for ct in all_ctypes:
        gene_scores = estimate_gene_scores(prob_gene_per_type, cell_type_adj_mat, cell_type=ct)
        cont_probs = estimate_gene_contamination_probabilities(
            gene_scores, p_c=p_c, prior_cont_prob=prior_cont_probs[ct]
        )

        # Apply signal threshold and expression mask
        mask = (cont_probs > signal_thres) & expr_frac_mask
        cont_probs[~mask] = 0

        cont_probs_per_type[ct] = cont_probs

    cont_probs_df = pd.DataFrame(cont_probs_per_type)

    if exclude_genes is not None:
        cont_probs_df = _mask_out_genes(cont_probs_df, exclude_genes)

    # Null contamination fractions
    cont_fracs_null = {}
    for ct in all_ctypes:
        probs_ct = prob_gene_per_type.loc[ct]
        cont_probs_ct = cont_probs_df[ct]
        total = probs_ct.sum()
        if total > 0:
            cont_fracs_null[ct] = (cont_probs_ct * probs_ct).sum() / total
        else:
            cont_fracs_null[ct] = 0

    # Doublet scores per cell type
    doublet_scores_all = []
    for ct in all_ctypes:
        cm_ct = cm_spatial[:, annot_spatial == ct] if hasattr(cm_spatial, '__getitem__') else cm_spatial
        cont_probs_ct = cont_probs_df[ct]
        cont_genes = cont_probs_ct[cont_probs_ct > signal_thres].index.tolist()

        scores = estimate_cell_admixture_fractions(cont_probs_ct, cm_ct)

        if adjust:
            scores = np.maximum(scores - cont_fracs_null[ct], 0)

        doublet_scores_all.extend(scores)

    doublet_scores = np.array(doublet_scores_all)

    return {
        'cont_probs_per_type': cont_probs_df,
        'doublet_scores': doublet_scores,
    }


def sparse_cor(x):
    """Compute correlation matrix for sparse matrix columns.

    Parameters
    ----------
    x : scipy.sparse.csr_matrix
        Sparse matrix.

    Returns
    -------
    numpy.ndarray
        Correlation matrix.
    """
    if sparse.issparse(x):
        x = sparse.csr_matrix(x)
        n = x.shape[0]
        cmean = np.asarray(x.mean(axis=0)).ravel()
        covmat = (x.T @ x - 2 * np.outer(cmean, np.asarray(x.sum(axis=0)).ravel())
                  + n * np.outer(cmean, cmean)) / (n - 1)
    else:
        x = np.asarray(x)
        n = x.shape[0]
        cmean = x.mean(axis=0)
        covmat = (x.T @ x - 2 * np.outer(cmean, x.sum(axis=0))
                  + n * np.outer(cmean, cmean)) / (n - 1)

    sdvec = np.sqrt(np.diag(covmat))
    cor_mat = covmat / np.outer(sdvec, sdvec)
    return cor_mat


def estimate_correlation_preservation(cm1, cm2):
    """Estimate preservation of gene correlations between two count matrices.

    Parameters
    ----------
    cm1 : sparse or dense matrix
        First count matrix (genes x cells).
    cm2 : sparse or dense matrix
        Second count matrix (genes x cells).

    Returns
    -------
    pandas.Series
        Per-gene correlation of correlations.
    """
    if hasattr(cm1, 'index') and hasattr(cm2, 'index'):
        common_genes = sorted(set(cm1.index) & set(cm2.index))
    else:
        common_genes = sorted(set(range(cm1.shape[0])) & set(range(cm2.shape[0])))

    # Filter genes with zero sums
    if sparse.issparse(cm1):
        sums1 = np.asarray(cm1[common_genes].sum(axis=1)).ravel()
        sums2 = np.asarray(cm2[common_genes].sum(axis=1)).ravel()
    else:
        sums1 = np.asarray(cm1[common_genes]).sum(axis=1)
        sums2 = np.asarray(cm2[common_genes]).sum(axis=1)

    mask = (sums1 > 0) & (sums2 > 0)
    common_genes = [g for g, m in zip(common_genes, mask) if m]

    sub1 = cm1[common_genes] if hasattr(cm1, '__getitem__') else cm1
    sub2 = cm2[common_genes] if hasattr(cm2, '__getitem__') else cm2

    if sparse.issparse(sub1):
        sub1 = sub1.toarray()
    if sparse.issparse(sub2):
        sub2 = sub2.toarray()

    cors1 = sparse_cor(sub1.T)
    cors2 = sparse_cor(sub2.T)

    np.fill_diagonal(cors1, np.nan)
    np.fill_diagonal(cors2, np.nan)

    cor_cors = {}
    for i in range(len(common_genes)):
        valid = ~(np.isnan(cors1[i]) | np.isnan(cors2[i]))
        if valid.sum() > 1:
            cor_cors[common_genes[i]] = np.corrcoef(cors1[i, valid], cors2[i, valid])[0, 1]
        else:
            cor_cors[common_genes[i]] = np.nan

    return pd.Series(cor_cors)
