#!/usr/bin/env Rscript
# Fixed R reference — uses same gene filtering as Python for fair comparison.

suppressMessages({
  library(Matrix)
  library(jsonlite)
  library(FNN)
  library(dplyr)
  library(magrittr)
  library(NMF)
  library(nnls)
})

args <- commandArgs(trailingOnly = TRUE)
fixture_path <- args[1]
output_path  <- args[2]

input <- readRDS(fixture_path)
df <- input$df
cell_meta <- input$cell_meta

df$cell_type <- df$celltype
cell_meta$cell_type <- cell_meta$celltype

set.seed(42)

cat("Running fixed R reference...\n")

# === 1. Build count matrix (genes x cells) ===
cat("1. Building count matrix...\n")
gene_factors <- factor(df$gene)
cell_factors <- factor(df$cell)
cm <- sparseMatrix(
  i = as.integer(gene_factors),
  j = as.integer(cell_factors),
  x = rep(1, nrow(df)),
  dims = c(length(levels(gene_factors)), length(levels(cell_factors))),
  dimnames = list(levels(gene_factors), levels(cell_factors))
)
cat("  Count matrix:", nrow(cm), "genes x", ncol(cm), "cells\n")

# === 2. Gene probabilities on ALL genes ===
cat("2. Computing gene probabilities (all genes)...\n")
annot <- setNames(cell_meta$celltype, cell_meta$cell)
cids_per_type <- split(names(annot), annot)

gene_probs_all <- sapply(cids_per_type, function(cids) {
  cids_in <- intersect(cids, colnames(cm))
  if (length(cids_in) == 0) return(rep(0, nrow(cm)))
  Matrix::rowMeans(cm[, cids_in, drop = FALSE] > 0)
})
gene_probs_all <- t(gene_probs_all)
cat("  gene_probs_all dimensions:", nrow(gene_probs_all), "x", ncol(gene_probs_all), "\n")

# === 3. NMF with gene filtering (same as Python) ===
cat("3. Running NMF with gene filtering...\n")
h <- 20
cell_counts <- table(df$cell)
cells_keep <- names(cell_counts[cell_counts > h])
df_sub <- df[df$cell %in% cells_keep, ]

# Get genes that survive NMF filtering (rowSums > 0 after KNN)
# Build KNN count matrix
cat("  Building KNN count matrix...\n")
genes_all <- sort(unique(df_sub$gene))

# Sample a subset for faster NMF (same molecules as Python)
set.seed(42)
mol_ids_all <- df_sub$mol_id
if (length(mol_ids_all) > 2000) {
  mol_sample <- sample(mol_ids_all, 2000)
  df_nmf <- df_sub[df_sub$mol_id %in% mol_sample, ]
} else {
  df_nmf <- df_sub
}

knn_results <- list()
mol_ids <- c()
for (cell_id in unique(df_nmf$cell)) {
  df_cell <- df_nmf[df_nmf$cell == cell_id, ]
  if (nrow(df_cell) <= h) next
  coords_cell <- as.matrix(df_cell[, c('x', 'y', 'z')])
  nn <- get.knn(coords_cell, k = h)
  xn <- nn$nn.index
  i <- rep(1:nrow(xn), each = ncol(xn))
  j <- c(t(xn))
  adj <- sparseMatrix(i = i, j = j, x = rep(1, length(i)),
                       dims = c(nrow(xn), nrow(xn)))
  diag(adj) <- 1

  gene_factor <- factor(df_cell$gene, levels = genes_all)
  gene_ind <- sparseMatrix(
    i = 1:length(gene_factor),
    j = as.integer(gene_factor),
    x = rep(1, length(gene_factor)),
    dims = c(length(gene_factor), length(genes_all))
  )
  counts <- adj %*% gene_ind
  knn_results[[cell_id]] <- as.matrix(counts)
  mol_ids <- c(mol_ids, df_cell$mol_id)
}

X <- do.call(rbind, knn_results)
rownames(X) <- mol_ids

# Ensure column names are set
if (is.null(colnames(X))) colnames(X) <- genes_all

# Filter zero rows/cols (same as Python)
row_mask <- rowSums(X) > 0
col_mask <- colSums(X) > 0
X <- X[row_mask, col_mask]
genes_filtered <- colnames(X)
cat("  NMF input:", nrow(X), "x", ncol(X), "\n")
cat("  Filtered genes:", length(genes_filtered), "\n")

# Weight matrix Z (inverse column sums)
Z <- matrix(rep(1 / colSums(X), nrow(X)), nrow = nrow(X), byrow = TRUE)

# Pre-multiply by Z (same as Python fix)
X_weighted <- X * Z

# Run NMF
cat("  Running NMF...\n")
nmf_res <- NMF::nmf(X_weighted, rank = 3, method = 'lee', nrun = 2, .options = 'vmp1', seed = 0)

# === 4. Gene probabilities on FILTERED genes (same as Python) ===
cat("4. Computing gene probabilities (filtered genes)...\n")
cm_filtered <- cm[genes_filtered, , drop = FALSE]
gene_probs_filtered <- sapply(cids_per_type, function(cids) {
  cids_in <- intersect(cids, colnames(cm_filtered))
  if (length(cids_in) == 0) return(rep(0, nrow(cm_filtered)))
  Matrix::rowMeans(cm_filtered[, cids_in, drop = FALSE] > 0)
})
gene_probs_filtered <- t(gene_probs_filtered)
cat("  gene_probs_filtered dimensions:", nrow(gene_probs_filtered), "x", ncol(gene_probs_filtered), "\n")

# === 5. Sparse correlation test ===
cat("5. Testing sparse_cor...\n")
set.seed(123)
test_mat <- matrix(rnorm(500), nrow = 100, ncol = 5)
cmean.x <- colMeans(test_mat)
n <- nrow(test_mat)
covmat <- (crossprod(test_mat) - 2 * (cmean.x %o% colSums(test_mat)) + n * cmean.x %o% cmean.x) / (n - 1)
sdvec <- sqrt(diag(covmat))
sc_result <- covmat / crossprod(t(sdvec))

# === 6. Normalize images test ===
cat("6. Testing normalize_images...\n")
set.seed(123)
test_im <- array(runif(300), dim = c(3, 10, 10))
channel_means <- apply(test_im, 1, mean)
im_centered <- sweep(test_im, 1, channel_means, "-")
channel_sds <- apply(im_centered, 1, sd)
im_normalized <- sweep(im_centered, 1, channel_sds, "/")
im_shifted <- im_normalized - min(im_normalized)
norm_im <- im_shifted / max(im_shifted)

# Serialize
cat("Serializing results...\n")
result <- list(
  gene_prob_per_type_all = gene_probs_all,
  gene_prob_per_type_filtered = gene_probs_filtered,
  filtered_genes = genes_filtered,
  nmf_H = nmf_res@fit@H,
  nmf_W = nmf_res@fit@W,
  sparse_cor = sc_result,
  test_mat = test_mat,
  norm_im_range = c(min(norm_im), max(norm_im)),
  n_genes_all = nrow(cm),
  n_genes_filtered = length(genes_filtered),
  n_cells = ncol(cm),
  n_mols_nmf = nrow(X)
)

write_json(result, output_path, auto_unbox = TRUE, digits = NA)
cat("R reference completed.\n")
