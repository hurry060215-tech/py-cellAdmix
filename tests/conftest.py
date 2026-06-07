"""Pytest fixtures for cellAdmix tests."""

import pytest
import numpy as np
import pandas as pd
from scipy import sparse


@pytest.fixture
def small_molecule_df():
    """Small molecule dataframe for testing."""
    np.random.seed(42)
    n_cells = 20
    n_mols_per_cell = 30
    genes = ['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE']

    records = []
    for cell_id in range(n_cells):
        cell_name = f'Cell_{cell_id}'
        celltype = np.random.choice(['Type1', 'Type2', 'Type3'])
        cx, cy = np.random.uniform(0, 100, 2)

        for mol_id in range(n_mols_per_cell):
            records.append({
                'x': cx + np.random.normal(0, 5),
                'y': cy + np.random.normal(0, 5),
                'z': np.random.choice([0, 1, 2]),
                'gene': np.random.choice(genes),
                'cell': cell_name,
                'celltype': celltype,
                'mol_id': f'Mol_{cell_id}_{mol_id}',
            })

    return pd.DataFrame(records)


@pytest.fixture
def small_count_matrix():
    """Small count matrix for testing."""
    np.random.seed(42)
    genes = [f'Gene{i}' for i in range(50)]
    cells = [f'Cell{i}' for i in range(100)]

    data = np.random.poisson(2, size=(50, 100)).astype(float)
    cm = pd.DataFrame(data, index=genes, columns=cells)
    return cm


@pytest.fixture
def small_cell_annot():
    """Small cell annotations."""
    np.random.seed(42)
    cells = [f'Cell{i}' for i in range(100)]
    celltypes = np.random.choice(['Type1', 'Type2', 'Type3'], size=100)
    return pd.Series(celltypes, index=cells)


@pytest.fixture
def small_adjacency_matrix():
    """Small cell type adjacency matrix."""
    ctypes = ['Type1', 'Type2', 'Type3']
    adj = pd.DataFrame(
        [[10, 3, 2], [3, 10, 4], [2, 4, 10]],
        index=ctypes, columns=ctypes
    )
    return adj
