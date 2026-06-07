"""Preprocessing and utility functions."""

import numpy as np
import pandas as pd
from scipy import sparse
import anndata


def samp_ct_equal(df, cell_annot, num_cells_samp):
    """Downsample cells to balance cell type representation.

    Parameters
    ----------
    df : pandas.DataFrame
        Molecule data with 'cell', 'celltype' columns.
    cell_annot : pandas.DataFrame
        Cell metadata with 'celltype' column, indexed by cell ID.
    num_cells_samp : int
        Target total number of cells.

    Returns
    -------
    pandas.DataFrame
        Downsampled molecule data.
    """
    ct_counts = cell_annot['celltype'].value_counts()
    total_lib = df['celltype'].value_counts()
    av_lib = total_lib / ct_counts

    probs = 1 / av_lib
    probs = probs / probs.sum()

    all_ct_sel = (num_cells_samp * probs).round().astype(int)

    cells_keep = []
    for ct, n_select in all_ct_sel.items():
        ct_cells = cell_annot[cell_annot['celltype'] == ct].index.tolist()
        if len(ct_cells) < n_select:
            cells_keep.extend(ct_cells)
        else:
            cells_keep.extend(np.random.choice(ct_cells, n_select, replace=False).tolist())

    return df[df['cell'].isin(cells_keep)].copy()


def subset_genes(df, cell_annot, top_g_thresh=None, n_hvgs=None):
    """Subset genes by expression or variance.

    Parameters
    ----------
    df : pandas.DataFrame
        Molecule data.
    cell_annot : pandas.DataFrame
        Cell metadata.
    top_g_thresh : float or None
        Min fractional expression threshold.
    n_hvgs : int or None
        Number of highly variable genes.

    Returns
    -------
    pandas.DataFrame
        Subsetted molecule data.
    """
    df = df.copy()

    if top_g_thresh is not None:
        top_g = _get_high_exp_genes(df, cell_annot, frac_thresh=top_g_thresh)
        df = df[df['gene'].isin(top_g)]

    if n_hvgs is not None:
        top_g = _get_high_var_genes(df, n_hvgs=n_hvgs)
        df = df[df['gene'].isin(top_g)]

    df['gene_sub'] = True
    return df


def _get_high_exp_genes(tx_dat, cell_annot, frac_thresh=0.1):
    """Get genes with high expression in any cell type."""
    # Build count matrix
    gene_factors = pd.Categorical(tx_dat['gene'])
    cell_factors = pd.Categorical(tx_dat['cell'])

    cm = sparse.csr_matrix(
        (np.ones(len(tx_dat)), (gene_factors.codes, cell_factors.codes)),
        shape=(len(gene_factors.categories), len(cell_factors.categories))
    )

    all_ct = cell_annot['celltype'].unique()
    all_g_keep = set()

    for ct in all_ct:
        cells_keep = cell_annot[cell_annot['celltype'] == ct].index.tolist()
        col_idx = [cell_factors.categories.get_loc(c) for c in cells_keep if c in cell_factors.categories]
        if not col_idx:
            continue
        counts_sub = cm[:, col_idx]
        frac_exp = np.asarray((counts_sub > 0).sum(axis=1)).ravel() / counts_sub.shape[1]
        g_keep = gene_factors.categories[frac_exp >= frac_thresh]
        all_g_keep.update(g_keep)

    return list(all_g_keep)


def _get_high_var_genes(tx_dat, n_hvgs=1000):
    """Get highly variable genes using variance-based selection."""
    # Build count matrix
    gene_factors = pd.Categorical(tx_dat['gene'])
    cell_factors = pd.Categorical(tx_dat['cell'])

    cm = sparse.csr_matrix(
        (np.ones(len(tx_dat)), (gene_factors.codes, cell_factors.codes)),
        shape=(len(gene_factors.categories), len(cell_factors.categories))
    )

    # Normalize by library size
    lib_sizes = np.asarray(cm.sum(axis=0)).ravel()
    lib_sizes[lib_sizes == 0] = 1
    cm_norm = cm / lib_sizes[np.newaxis, :]

    # Compute variance
    means = np.asarray(cm_norm.mean(axis=1)).ravel()
    variances = np.asarray((cm_norm.power(2)).mean(axis=1)).ravel() - means**2

    n_hvgs = min(n_hvgs, len(gene_factors.categories))
    top_idx = np.argsort(variances)[-n_hvgs:]
    return gene_factors.categories[top_idx].tolist()


def get_counts_matrix(tx_dat):
    """Transform transcript molecule dataframe into a count matrix.

    Parameters
    ----------
    tx_dat : pandas.DataFrame
        Transcript data with 'gene', 'cell' columns.

    Returns
    -------
    anndata.AnnData
        AnnData object with counts in X and cell metadata in obs.
    """
    gene_factors = pd.Categorical(tx_dat['gene'])
    cell_factors = pd.Categorical(tx_dat['cell'])

    cm = sparse.csr_matrix(
        (np.ones(len(tx_dat)), (gene_factors.codes, cell_factors.codes)),
        shape=(len(gene_factors.categories), len(cell_factors.categories))
    )

    adata = anndata.AnnData(
        X=cm.T,  # cells x genes
        obs=pd.DataFrame(index=cell_factors.categories.tolist()),
        var=pd.DataFrame(index=gene_factors.categories.tolist()),
    )

    # Add cell type metadata if available
    if 'celltype' in tx_dat.columns:
        cell_type_map = tx_dat.drop_duplicates('cell').set_index('cell')['celltype']
        adata.obs['cell_type'] = adata.obs.index.map(cell_type_map)

    return adata


def normalize_images(im):
    """Normalize image channels.

    Parameters
    ----------
    im : numpy.ndarray
        Image with dimensions (channels, height, width).

    Returns
    -------
    numpy.ndarray
        Normalized image.
    """
    # Subtract channel means
    channel_means = im.mean(axis=(1, 2), keepdims=True)
    im_centered = im - channel_means

    # Divide by channel stds
    channel_stds = im_centered.std(axis=(1, 2), keepdims=True)
    channel_stds[channel_stds == 0] = 1
    im_normalized = im_centered / channel_stds

    # Shift global min to zero
    global_min = im_normalized.min()
    im_shifted = im_normalized - global_min

    # Scale to max = 1
    global_max = im_shifted.max()
    if global_max > 0:
        im_final = im_shifted / global_max
    else:
        im_final = im_shifted

    return im_final
