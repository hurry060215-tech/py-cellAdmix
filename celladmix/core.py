"""CellAdmix class — method-chaining API over AnnData."""

import numpy as np
import pandas as pd
from scipy import sparse
import anndata


class CellAdmix:
    """Evaluate and correct cell admixtures in imaging-based spatial transcriptomics.

    Parameters
    ----------
    adata : anndata.AnnData
        Spatial transcriptomics data with molecule-level info in .obs.
    molecule_df : pandas.DataFrame or None
        Molecule-level data with x, y, z, gene, cell, mol_id, celltype columns.
        If None, expects molecule info in adata.obs.
    """

    def __init__(self, adata=None, molecule_df=None):
        self.adata = adata
        self.molecule_df = molecule_df
        self._nmf_res = None
        self._crf_res = None
        self._contamination_scores = None
        self._bridge_res = None
        self._memb_res = None

    def run_knn_nmf(self, k=5, h=20, mols_test=None, nmol_dsamp=None, n_cores=1):
        """Run KNN-based NMF factorization.

        Parameters
        ----------
        k : int
            Number of factors.
        h : int
            KNN neighborhood size.

        Returns
        -------
        CellAdmix
        """
        from .nmf_crf import run_knn_nmf
        self._nmf_res = run_knn_nmf(
            self.molecule_df, k=k, h=h, mols_test=mols_test,
            nmol_dsamp=nmol_dsamp, n_cores=n_cores
        )
        return self

    def run_crf_all(self, num_nn=10, same_label_ratio=5, normalize_by='gene', proj_h=None, n_cores=1):
        """Run CRF segmentation for all cells.

        Parameters
        ----------
        num_nn : int
            Neighborhood size for CRF.
        same_label_ratio : float
            Transition ratio.

        Returns
        -------
        CellAdmix
        """
        from .nmf_crf import run_crf_all
        if self._nmf_res is None:
            raise RuntimeError('Must run run_knn_nmf first')

        self._crf_res = run_crf_all(
            self.molecule_df, self._nmf_res, num_nn=num_nn,
            same_label_ratio=same_label_ratio, normalize_by=normalize_by,
            proj_h=proj_h, n_cores=n_cores
        )
        return self

    def estimate_contamination_scores(
        self, cm_rna, cm_spatial, annot_rna, annot_spatial,
        cell_type_adj_mat, p_c=0.25, signal_thres=0.25,
        min_expr_frac=0.05, max_prob_high=0.1,
        exclude_cell_types=None, exclude_genes=None,
        use_counts=False, adjust=True
    ):
        """Estimate contamination scores.

        Parameters
        ----------
        cm_rna : DataFrame
            scRNA-seq count matrix.
        cm_spatial : DataFrame
            Spatial count matrix.
        annot_rna : Series
            scRNA cell type annotations.
        annot_spatial : Series
            Spatial cell type annotations.
        cell_type_adj_mat : DataFrame
            Cell type adjacency matrix.

        Returns
        -------
        CellAdmix
        """
        from .scoring import estimate_contamination_scores
        self._contamination_scores = estimate_contamination_scores(
            cm_rna, cm_spatial, annot_rna, annot_spatial,
            cell_type_adj_mat, p_c=p_c, signal_thres=signal_thres,
            min_expr_frac=min_expr_frac, max_prob_high=max_prob_high,
            exclude_cell_types=exclude_cell_types, exclude_genes=exclude_genes,
            use_counts=use_counts, adjust=adjust
        )
        return self

    def run_bridge_test(
        self, cell_annot, ncm_ct, ncm_cell,
        ncells_samp=200, knn_k=20, n_cores=1
    ):
        """Run cell-bridging test.

        Parameters
        ----------
        cell_annot : Series
            Cell type annotations.
        ncm_ct : DataFrame
            Molecule-to-celltype KNN counts.
        ncm_cell : DataFrame
            Molecule-to-cell KNN counts.

        Returns
        -------
        CellAdmix
        """
        from .annotation import run_bridge_test
        if self._crf_res is None:
            raise RuntimeError('Must run run_crf_all first')

        self._bridge_res = run_bridge_test(
            self.molecule_df, self._crf_res, cell_annot,
            ncm_ct, ncm_cell, ncells_samp=ncells_samp,
            knn_k=knn_k, n_cores=n_cores
        )
        return self

    def extract_bridge_res(
        self, all_ctypes, n_factors, nmf_type='joint',
        p_thresh=0.01, adj_pvals=True, get_p_mat=False
    ):
        """Extract bridge test results.

        Parameters
        ----------
        all_ctypes : list
            All cell types.
        n_factors : int
            Number of factors.

        Returns
        -------
        CellAdmix
        """
        from .annotation import extract_bridge_res
        if self._bridge_res is None:
            raise RuntimeError('Must run run_bridge_test first')

        self._bridge_extracted = extract_bridge_res(
            self._bridge_res, all_ctypes, n_factors,
            nmf_type=nmf_type, p_thresh=p_thresh,
            adj_pvals=adj_pvals, get_p_mat=get_p_mat
        )
        return self

    def get_enr(self, markers, n_perm=10000, p_thresh=0.1, adj_pvals=True,
                crf_h=None, n_cores=1):
        """Run enrichment test for marker genes.

        Parameters
        ----------
        markers : DataFrame
            Columns: gene, marker_of.

        Returns
        -------
        CellAdmix
        """
        from .annotation import get_enr
        if self._nmf_res is None:
            raise RuntimeError('Must run run_knn_nmf first')

        self._enr_res = get_enr(
            self._nmf_res, markers, n_perm=n_perm, p_thresh=p_thresh,
            adj_pvals=adj_pvals, crf_h=crf_h, df=self.molecule_df,
            n_cores=n_cores
        )
        return self

    def check_f_rm(self, cell_annot, ncm_ct_full, do_clean=True, median_thresh=0.1):
        """Check for false positive removals.

        Parameters
        ----------
        cell_annot : Series
            Cell type annotations.
        ncm_ct_full : DataFrame
            Cell-to-celltype neighbor counts.

        Returns
        -------
        CellAdmix
        """
        from .annotation import check_f_rm
        if self._enr_res is None:
            raise RuntimeError('Must run get_enr first')

        f_ct_rm, f_from = self._enr_res
        self._f_rm_checked = check_f_rm(
            cell_annot, f_ct_rm, f_from,
            self._contamination_scores['doublet_scores'] if self._contamination_scores else None,
            ncm_ct_full, do_clean=do_clean, median_thresh=median_thresh
        )
        return self

    @property
    def nmf_results(self):
        """NMF results dict."""
        return self._nmf_res

    @property
    def crf_results(self):
        """CRF factor assignments DataFrame."""
        return self._crf_res

    @property
    def contamination_scores(self):
        """Contamination scores dict."""
        return self._contamination_scores

    @property
    def bridge_results(self):
        """Bridge test results."""
        return self._bridge_res

    @property
    def enr_results(self):
        """Enrichment test results."""
        return self._enr_res
