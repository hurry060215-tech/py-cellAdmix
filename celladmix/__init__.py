"""
celladmix — Python port of cellAdmix

Evaluating and correcting cell admixtures in imaging-based spatial
transcriptomics data.
"""

from .core import CellAdmix
from .knn import (
    knn_adjacency_matrix,
    knn_count_matrix,
    get_knn_counts_all,
)
from .nmf_crf import (
    run_knn_nmf,
    run_crf_all,
)
from .scoring import (
    estimate_cell_adjacency,
    estimate_cell_type_adjacency,
    estimate_gene_prob_per_type,
    estimate_contamination_scores,
    sparse_cor,
    estimate_correlation_preservation,
)
from .annotation import (
    run_bridge_test,
    extract_bridge_res,
    run_memb_test,
    extract_memb_res,
    get_enr,
    check_f_rm,
    check_f_rm_per_ct,
)
from .preprocessing import (
    samp_ct_equal,
    subset_genes,
    get_counts_matrix,
    normalize_images,
)
from .plots import (
    plot_nmf_loadings,
    plot_cell_score_ratios,
    plot_annot_hmap,
    plot_correlation_preservation,
    plot_expression_comparison,
)

__all__ = [
    "CellAdmix",
    # KNN
    "knn_adjacency_matrix",
    "knn_count_matrix",
    "get_knn_counts_all",
    # NMF / CRF
    "run_knn_nmf",
    "run_crf_all",
    # Scoring
    "estimate_cell_adjacency",
    "estimate_cell_type_adjacency",
    "estimate_gene_prob_per_type",
    "estimate_contamination_scores",
    "sparse_cor",
    "estimate_correlation_preservation",
    # Annotation
    "run_bridge_test",
    "extract_bridge_res",
    "run_memb_test",
    "extract_memb_res",
    "get_enr",
    "check_f_rm",
    "check_f_rm_per_ct",
    # Preprocessing
    "samp_ct_equal",
    "subset_genes",
    "get_counts_matrix",
    "normalize_images",
    # Plots
    "plot_nmf_loadings",
    "plot_cell_score_ratios",
    "plot_annot_hmap",
    "plot_correlation_preservation",
    "plot_expression_comparison",
]
__version__ = "0.1.0"
