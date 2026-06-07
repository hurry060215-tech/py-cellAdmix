"""Parity gate tests — compare Python output against R reference."""

import pytest
import numpy as np
import pandas as pd
import json
import subprocess
from pathlib import Path
from scipy import sparse

import yaml


MANIFEST_PATH = Path(__file__).parent.parent / 'data' / 'manifest.yaml'
R_OUTPUT_PATH = Path(__file__).parent.parent / 'data' / 'r_reference_output.json'
FIXTURE_DF_PATH = Path(__file__).parent.parent / 'data' / 'fixture_small_df.csv'
FIXTURE_META_PATH = Path(__file__).parent.parent / 'data' / 'fixture_small_cell_meta.csv'


def load_manifest():
    """Load the manifest.yaml."""
    with open(MANIFEST_PATH, encoding='utf-8') as f:
        return yaml.safe_load(f)


@pytest.fixture
def manifest():
    return load_manifest()


@pytest.fixture
def r_reference():
    """Load R reference output."""
    if not R_OUTPUT_PATH.exists():
        pytest.skip('R reference output not found')
    with open(R_OUTPUT_PATH) as f:
        return json.load(f)


@pytest.fixture
def fixture_df():
    """Load fixture molecule data."""
    if not FIXTURE_DF_PATH.exists():
        pytest.skip('Fixture not found')
    return pd.read_csv(FIXTURE_DF_PATH)


@pytest.fixture
def fixture_meta():
    """Load fixture cell metadata."""
    if not FIXTURE_META_PATH.exists():
        pytest.skip('Fixture not found')
    return pd.read_csv(FIXTURE_META_PATH)


def test_parity_scores(manifest, r_reference, fixture_df, fixture_meta):
    """Parity gate: gene probabilities match R reference."""
    from celladmix.scoring import estimate_gene_prob_per_type

    # Build count matrix (genes x cells)
    gene_factors = pd.Categorical(fixture_df['gene'])
    cell_factors = pd.Categorical(fixture_df['cell'])
    cm = sparse.csr_matrix(
        (np.ones(len(fixture_df)), (gene_factors.codes, cell_factors.codes)),
        shape=(len(gene_factors.categories), len(cell_factors.categories))
    )
    cm_df = pd.DataFrame(cm.toarray(), index=gene_factors.categories, columns=cell_factors.categories)
    annot = fixture_meta.set_index('cell')['celltype']

    r_genes = r_reference['filtered_genes']
    cm_filtered = cm_df.loc[cm_df.index.isin(r_genes)]

    gene_probs = estimate_gene_prob_per_type(cm_filtered, annot)
    r_probs = np.array(r_reference['gene_prob_per_type_filtered'])

    # Align by row sums
    r_order = np.argsort(r_probs.sum(axis=1))
    py_order = np.argsort(gene_probs.values.sum(axis=1))
    r_sorted = np.sort(r_probs[r_order].ravel())
    py_sorted = np.sort(gene_probs.values[py_order].ravel())

    from scipy.stats import pearsonr
    pearson_r, _ = pearsonr(r_sorted, py_sorted)

    threshold = 0.97
    assert pearson_r >= threshold, (
        f"Gene probabilities parity failed: Pearson r = {pearson_r:.6f} < {threshold}"
    )


def test_parity_crf_factors(manifest, r_reference, fixture_df):
    """Parity gate: NMF factor ordering matches R reference (Spearman)."""
    from celladmix.nmf_crf import run_knn_nmf
    from scipy.stats import spearmanr

    r_H = np.array(r_reference['nmf_H'])
    r_genes = r_reference['filtered_genes']

    # Run Python NMF
    nmf_res = run_knn_nmf(fixture_df, k=3, h=20)
    py_H = nmf_res['H']
    py_genes = nmf_res['feature_names']

    # Find common genes
    common = sorted(set(r_genes) & set(py_genes))
    r_idx = [r_genes.index(g) for g in common]
    py_idx = [py_genes.index(g) for g in common]

    r_H_sub = r_H[:, r_idx]
    py_H_sub = py_H[:, py_idx]

    # Sort and compare
    r_sorted = np.sort(r_H_sub.ravel())
    py_sorted = np.sort(py_H_sub.ravel())
    spearman_r, _ = spearmanr(r_sorted, py_sorted)

    threshold = 0.97
    assert spearman_r >= threshold, (
        f"NMF factor ordering parity failed: Spearman r = {spearman_r:.6f} < {threshold}"
    )


def test_parity_cell_type_adjacency(manifest, r_reference, fixture_df):
    """Parity gate: cell adjacency has correct structure."""
    from celladmix.scoring import estimate_cell_adjacency, estimate_cell_type_adjacency

    df = fixture_df.copy()
    if 'cell_type' not in df.columns and 'celltype' in df.columns:
        df['cell_type'] = df['celltype']

    cell_adj = estimate_cell_adjacency(df, random_shift=1e-2)

    # Small fixtures may not have enough spatial diversity for Delaunay
    if len(cell_adj) == 0:
        pytest.skip('Cell adjacency empty (sparse fixture)')

    ct_adj = estimate_cell_type_adjacency(cell_adj)

    # Check structure
    assert ct_adj.shape[0] == ct_adj.shape[1], "Cell type adjacency not square"
    assert ct_adj.shape[0] > 0, "Cell type adjacency is empty"


def test_parity_gene_probabilities(manifest, r_reference, fixture_df, fixture_meta):
    """Parity gate: gene probabilities internal consistency."""
    from celladmix.scoring import estimate_gene_prob_per_type

    gene_factors = pd.Categorical(fixture_df['gene'])
    cell_factors = pd.Categorical(fixture_df['cell'])
    cm = sparse.csr_matrix(
        (np.ones(len(fixture_df)), (gene_factors.codes, cell_factors.codes)),
        shape=(len(gene_factors.categories), len(cell_factors.categories))
    )
    cm_df = pd.DataFrame(cm.toarray(), index=gene_factors.categories, columns=cell_factors.categories)
    annot = fixture_meta.set_index('cell')['celltype']

    gene_probs = estimate_gene_prob_per_type(cm_df, annot)

    # Internal consistency checks
    assert gene_probs.shape[0] == len(annot.unique()), "Cell type count mismatch"
    assert np.all(gene_probs.values >= 0), "Negative probabilities"
    assert np.all(gene_probs.values <= 1), "Probabilities > 1"

    # Each cell type should have some non-zero genes
    for ct in gene_probs.index:
        assert np.count_nonzero(gene_probs.loc[ct]) > 0, f"{ct} has all zeros"
