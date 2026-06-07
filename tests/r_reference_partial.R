#!/usr/bin/env Rscript
# Partial R reference — runs non-CRF functions and serializes outputs.
# CRF package unavailable on R 4.5, so CRF-dependent functions are skipped.

suppressMessages({
  library(Matrix)
  library(jsonlite)
  library(FNN)
  library(dplyr)
  library(magrittr)
  library(igraph)
  library(NMF)
  library(nnls)
})

# Source the R functions directly (bypassing package loading which needs CRF)
r_src <- 'D:/test/cellAdmix-main/cellAdmix-main/R'
source(file.path(r_src, 'admix_scores.R'))
source(file.path(r_src, 'utils.R'))

args <- commandArgs(trailingOnly = TRUE)
fixture_path <- args[1]
output_path  <- args[2]

# Load fixture
input <- readRDS(fixture_path)
df <- input$df
cell_meta <- input$cell_meta

# Add cell_type column (R function expects this name)
df$cell_type <- df$celltype
cell_meta$cell_type <- cell_meta$celltype

set.seed(42)

cat("Running partial R reference...\n")

# 1. Cell type adjacency
cat("1. Estimating cell adjacency...\n")
# Workaround: n.cores=1 causes cut(x, 1) to return NA; use n.cores=2
cell_adj <- estimate_cell_adjacency(df, n.cores=2)
ct_adj <- estimate_cell_type_adjacency(cell_adj)

cat("2. Estimating gene probabilities...\n")
# Build count matrix from spatial data
cm_spatial <- get_counts_meta_seurat(df)
annot_spatial <- cell_meta$celltype
names(annot_spatial) <- cell_meta$cell

# Use spatial data for gene probabilities (no scRNA available)
gene_probs <- estimate_gene_prob_per_type(cm_spatial[['RNA']]$counts, annot_spatial)

# 3. Gene scores for each cell type
cat("3. Computing gene scores...\n")
all_ctypes <- colnames(ct_adj)
gene_scores_list <- list()
for (ct in all_ctypes) {
  gs <- estimate_gene_scores(gene_probs, ct_adj, cell_type=ct)
  gene_scores_list[[ct]] <- gs
}

# 4. Gene contamination probabilities
cat("4. Computing contamination probabilities...\n")
prior_cont_probs <- 1 - diag(as.matrix(ct_adj)) / colSums(as.matrix(ct_adj))
cont_probs_list <- list()
for (ct in all_ctypes) {
  cp <- estimate_gene_contamination_probabilities(
    gene_scores_list[[ct]], p.c=0.25, prior.cont.prob=prior_cont_probs[ct]
  )
  cont_probs_list[[ct]] <- cp
}

# 5. NMF (without CRF)
cat("5. Running NMF...\n")
# Build KNN count matrix
h <- 20
cells_keep <- df %>% count(cell) %>% filter(n > h) %>% pull(cell)
df_sub <- df[df$cell %in% cells_keep, ]
genes <- sort(unique(df_sub$gene))

# Get KNN counts for NMF input
knn_res_list <- list()
for (cell_id in unique(df_sub$cell)) {
  df_cell <- df_sub[df_sub$cell == cell_id, ]
  if (nrow(df_cell) <= h) next
  counts <- knn_count_matrix(df_cell, k=h, all.genes=genes, include_i=TRUE)
  knn_res_list[[cell_id]] <- as.matrix(counts)
}
X <- do.call(rbind, knn_res_list)
X <- X[rowSums(X) > 0, colSums(X) > 0]

# Weight matrix
Z <- matrix(rep(1/colSums(X), nrow(X)), nrow=nrow(X), byrow=TRUE)

# Run NMF
cat("  Running LS-NMF with k=5...\n")
nmf_res <- NMF::nmf(X, rank=5, method='ls-nmf', weight=Z, nrun=10, .options='vmp1', seed=0)

# 6. Sparse correlation test
cat("6. Testing sparse_cor...\n")
test_mat <- matrix(rnorm(500), nrow=100, ncol=5)
sc_result <- sparse_cor(test_mat)

# 7. Normalize images test
cat("7. Testing normalize_images...\n")
test_im <- array(runif(300), dim=c(3, 10, 10))
norm_im <- normalize_images(test_im)

# Serialize results
cat("Serializing results...\n")
result <- list(
  cell_type_adjacency = as.matrix(ct_adj),
  gene_prob_per_type = as.matrix(gene_probs),
  prior_cont_probs = prior_cont_probs,
  cont_probs_per_type = lapply(cont_probs_list, function(x) as.numeric(x)),
  gene_scores = lapply(gene_scores_list, function(x) as.matrix(x)),
  nmf_H = nmf_res@fit@H,
  nmf_W_head = head(nmf_res@fit@W, 20),
  sparse_cor = sc_result,
  norm_im_range = c(min(norm_im), max(norm_im)),
  n_genes = length(genes),
  n_cells = length(unique(df_sub$cell))
)

write_json(result, output_path, auto_unbox=TRUE, digits=NA)
cat("R reference completed.\n")
