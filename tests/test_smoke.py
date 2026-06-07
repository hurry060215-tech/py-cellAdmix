"""Smoke tests — does the package import and do basic operations work?"""

import pytest
import numpy as np
import pandas as pd


def test_import():
    """Package imports successfully."""
    import celladmix
    assert hasattr(celladmix, '__version__')
    assert celladmix.__version__ == '0.1.0'


def test_import_functions():
    """All public functions are importable."""
    from celladmix import (
        CellAdmix,
        knn_adjacency_matrix,
        knn_count_matrix,
        get_knn_counts_all,
        run_knn_nmf,
        run_crf_all,
        estimate_cell_adjacency,
        estimate_cell_type_adjacency,
        estimate_gene_prob_per_type,
        estimate_contamination_scores,
        sparse_cor,
        estimate_correlation_preservation,
        run_bridge_test,
        extract_bridge_res,
        run_memb_test,
        extract_memb_res,
        get_enr,
        check_f_rm,
        check_f_rm_per_ct,
        samp_ct_equal,
        subset_genes,
        get_counts_matrix,
        normalize_images,
        plot_nmf_loadings,
        plot_cell_score_ratios,
        plot_annot_hmap,
        plot_correlation_preservation,
        plot_expression_comparison,
    )


def test_knn_adjacency(small_molecule_df):
    """KNN adjacency matrix builds correctly."""
    from celladmix.knn import knn_adjacency_matrix

    adj = knn_adjacency_matrix(small_molecule_df, k=5)
    assert adj.shape[0] == len(small_molecule_df)
    assert adj.shape[1] == len(small_molecule_df)
    # Each row should have exactly k neighbors
    row_sums = np.asarray(adj.sum(axis=1)).ravel()
    assert np.all(row_sums == 5)


def test_knn_count_matrix(small_molecule_df):
    """KNN count matrix builds correctly."""
    from celladmix.knn import knn_count_matrix

    ncm, genes = knn_count_matrix(small_molecule_df, k=5)
    assert ncm.shape[0] == len(small_molecule_df)
    assert len(genes) == len(small_molecule_df['gene'].unique())


def test_sparse_cor():
    """Sparse correlation computes correctly."""
    from celladmix.scoring import sparse_cor

    np.random.seed(42)
    x = np.random.randn(100, 5)
    cor = sparse_cor(x)
    assert cor.shape == (5, 5)
    # Diagonal should be 1
    np.testing.assert_allclose(np.diag(cor), 1.0, atol=1e-10)


def test_normalize_images():
    """Image normalization works."""
    from celladmix.preprocessing import normalize_images

    im = np.random.rand(3, 100, 100) * 255
    im_norm = normalize_images(im)
    assert im_norm.min() >= 0
    assert im_norm.max() <= 1.0 + 1e-10


def test_estimate_gene_prob_per_type(small_count_matrix, small_cell_annot):
    """Gene probability estimation works."""
    from celladmix.scoring import estimate_gene_prob_per_type

    probs = estimate_gene_prob_per_type(small_count_matrix, small_cell_annot)
    assert probs.shape[0] == len(small_cell_annot.unique())
    assert probs.shape[1] == len(small_count_matrix)


def test_estimate_cell_type_adjacency(small_molecule_df):
    """Cell type adjacency estimation works."""
    from celladmix.scoring import estimate_cell_adjacency, estimate_cell_type_adjacency

    cell_adj = estimate_cell_adjacency(small_molecule_df, random_shift=0)
    adj_mat = estimate_cell_type_adjacency(cell_adj)
    assert adj_mat.shape[0] == adj_mat.shape[1]


def test_celladmix_class():
    """CellAdmix class instantiates."""
    from celladmix import CellAdmix
    ca = CellAdmix()
    assert ca._nmf_res is None
    assert ca._crf_res is None
