#!/usr/bin/env Rscript
# Reference runner for py-cellAdmix parity tests.
# Loads the canonical fixture, runs the R cellAdmix package, writes JSON.

suppressMessages({
  library(cellAdmix)
  library(Matrix)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
fixture_path <- args[1]
output_path  <- args[2]

# Load fixture (RDS format)
input <- readRDS(fixture_path)

# Set seed for reproducibility
set.seed(42)

# Extract components
df <- input$df              # molecule data
cell_meta <- input$cell_meta  # cell metadata

# 1. Compute cell type adjacency
cell_adj <- estimate_cell_adjacency(df)
ct_adj <- estimate_cell_type_adjacency(cell_adj)

# 2. Compute gene probabilities per type (using scRNA if available)
if (!is.null(input$cm_rna) && !is.null(input$annot_rna)) {
  gene_probs <- estimate_gene_prob_per_type(input$cm_rna, input$annot_rna)
} else {
  # Use spatial data as fallback
  cm <- get_counts_meta_seurat(df)
  gene_probs <- estimate_gene_prob_per_type(cm[['RNA']]$counts, cell_meta$celltype)
}

# 3. Run NMF + CRF pipeline
nmf_res <- run_knn_nmf(df, k=5, h=20)
crf_res <- run_crf_all(df, nmf_res, num_nn=10)

# 4. Compute contamination scores
if (!is.null(input$cm_rna) && !is.null(input$annot_rna)) {
  scores <- estimate_contamination_scores(
    input$cm_rna, cm[['RNA']]$counts,
    input$annot_rna, cell_meta$celltype,
    ct_adj
  )
} else {
  scores <- NULL
}

# 5. Run bridge test
ncm_data <- get_knn_counts_all(df, h=20)
ncm_ct <- ncm_data[, unique(cell_meta$celltype)]
ncm_cell <- ncm_data[, unique(cell_meta$cell)]

bridge_res <- run_bridge_test(
  df, crf_res, cell_meta$celltype,
  ncm_ct, ncm_cell, ncells.samp=200, knn.k=20
)

# Serialize results
result <- list(
  cell_type_adjacency = as.matrix(ct_adj),
  gene_prob_per_type = as.matrix(gene_probs),
  crf_factors = crf_res$factor,
  doublet_scores = if (!is.null(scores)) scores$doublet_scores else NULL,
  cont_probs_per_type = if (!is.null(scores)) as.matrix(scores$cont.probs.per.type) else NULL,
  bridge_pvalues = bridge_res
)

write_json(result, output_path, auto_unbox = TRUE, digits = NA)
cat("R reference completed successfully.\n")
