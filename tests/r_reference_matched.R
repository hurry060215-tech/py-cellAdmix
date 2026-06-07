#!/usr/bin/env Rscript
# R reference with matched seeds for fair comparison.

suppressMessages({
  library(Matrix)
  library(jsonlite)
  library(FNN)
  library(dplyr)
  library(magrittr)
  library(NMF)
})

load('D:/test/cellAdmix-main/cellAdmix-main/data/df.rda')
load('D:/test/cellAdmix-main/cellAdmix-main/data/cell_meta.rda')

set.seed(42)

# Subsample 50 cells
cells_sampled <- sample(unique(df$cell), 50)
df <- df[df$cell %in% cells_sampled, ]
cell_meta <- cell_meta[cell_meta$cell %in% cells_sampled, ]

cat('Fixture:', nrow(df), 'molecules,', length(unique(df$cell)), 'cells\n')

# Build count matrix
gene_factors <- factor(df$gene)
cell_factors <- factor(df$cell)
cm <- sparseMatrix(
  i = as.integer(gene_factors),
  j = as.integer(cell_factors),
  x = rep(1, nrow(df)),
  dims = c(length(levels(gene_factors)), length(levels(cell_factors))),
  dimnames = list(levels(gene_factors), levels(cell_factors))
)

# Gene probabilities
annot <- setNames(cell_meta$celltype, cell_meta$cell)
cids_per_type <- split(names(annot), annot)

gene_probs <- sapply(cids_per_type, function(cids) {
  cids_in <- intersect(cids, colnames(cm))
  if (length(cids_in) == 0) return(rep(0, nrow(cm)))
  Matrix::rowMeans(cm[, cids_in, drop = FALSE] > 0)
})
gene_probs <- t(gene_probs)

# NMF - use same sampling as Python
h <- 20
cell_counts <- table(df$cell)
cells_keep <- names(cell_counts[cell_counts > h])
df_sub <- df[df$cell %in% cells_keep, ]
genes_all <- sort(unique(df_sub$gene))

# Sample molecules with same seed as Python (seed=42)
set.seed(42)
mol_ids_all <- unique(df_sub$mol_id)
mol_sample <- sample(mol_ids_all, min(2000, length(mol_ids_all)))
df_nmf <- df_sub[df_sub$mol_id %in% mol_sample, ]

cat('NMF molecules:', nrow(df_nmf), '\n')

# Build KNN count matrix
knn_results <- list()
mol_ids <- c()
for (cell_id in unique(df_nmf$cell)) {
  df_cell <- df_nmf[df_nmf$cell == cell_id, ]
  if (nrow(df_cell) <= h) next
  coords <- as.matrix(df_cell[, c('x', 'y', 'z')])
  nn <- get.knn(coords, k = h)
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
if (is.null(colnames(X))) colnames(X) <- genes_all

# Filter
row_mask <- rowSums(X) > 0
col_mask <- colSums(X) > 0
X <- X[row_mask, col_mask]
genes_filtered <- colnames(X)

cat('NMF input:', nrow(X), 'x', ncol(X), '\n')
cat('Filtered genes:', length(genes_filtered), '\n')

# Weight
Z <- matrix(rep(1 / colSums(X), nrow(X)), nrow = nrow(X), byrow = TRUE)
X_weighted <- X * Z

# Run NMF with seed=0 (same as Python best seed)
set.seed(0)
nmf_res <- NMF::nmf(X_weighted, rank = 3, method = 'lee', nrun = 5, .options = 'vmp1', seed = 0)

# Gene probabilities on filtered genes
cm_filtered <- cm[genes_filtered, , drop = FALSE]
gene_probs_filtered <- sapply(cids_per_type, function(cids) {
  cids_in <- intersect(cids, colnames(cm_filtered))
  if (length(cids_in) == 0) return(rep(0, nrow(cm_filtered)))
  Matrix::rowMeans(cm_filtered[, cids_in, drop = FALSE] > 0)
})
gene_probs_filtered <- t(gene_probs_filtered)

# Sparse correlation
set.seed(123)
test_mat <- matrix(rnorm(500), nrow = 100, ncol = 5)
cmean.x <- colMeans(test_mat)
n <- nrow(test_mat)
covmat <- (crossprod(test_mat) - 2 * (cmean.x %o% colSums(test_mat)) + n * cmean.x %o% cmean.x) / (n - 1)
sdvec <- sqrt(diag(covmat))
sc_result <- covmat / crossprod(t(sdvec))

# Normalize images
set.seed(123)
test_im <- array(runif(300), dim = c(3, 10, 10))
channel_means <- apply(test_im, 1, mean)
im_centered <- sweep(test_im, 1, channel_means, '-')
channel_sds <- apply(im_centered, 1, sd)
im_normalized <- sweep(im_centered, 1, channel_sds, '/')
im_shifted <- im_normalized - min(im_normalized)
norm_im <- im_shifted / max(im_shifted)

# Export molecule sample for Python to use
result <- list(
  gene_prob_per_type_filtered = gene_probs_filtered,
  filtered_genes = genes_filtered,
  nmf_H = nmf_res@fit@H,
  nmf_W = nmf_res@fit@W,
  nmf_mol_ids = mol_ids,
  nmf_mol_sample = mol_sample,
  sparse_cor = sc_result,
  test_mat = test_mat,
  norm_im_range = c(min(norm_im), max(norm_im)),
  n_genes_filtered = length(genes_filtered),
  n_mols_nmf = nrow(X)
)

write_json(result, 'D:/test/cellAdmix-main/py-cellAdmix/data/r_reference_output.json',
           auto_unbox = TRUE, digits = NA)
cat('Done.\n')
