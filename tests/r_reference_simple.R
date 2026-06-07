#!/usr/bin/env Rscript
# Simplified R reference — implements core logic directly (no parallel/mclapply).

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

cat("  Fixture: nrow(df) =", nrow(df), ", n_cells =", length(unique(df$cell)), "\n")

set.seed(42)

cat("Running simplified R reference...\n")

# === 1. Cell adjacency (inline, no mclapply) ===
cat("1. Estimating cell adjacency...\n")
df.spatial <- df %>% filter(!is.na(celltype))
df.spatial$x <- df.spatial$x + runif(nrow(df.spatial), -1e-3, 1e-3)
df.spatial$y <- df.spatial$y + runif(nrow(df.spatial), -1e-3, 1e-3)

coords <- as.matrix(df.spatial[, c('x', 'y')])
tri <- geometry::delaunayn(coords)
edges <- rbind(tri[, c(1, 2)], tri[, c(1, 3)], tri[, c(2, 3)])
edges <- cbind(pmin(edges[, 1], edges[, 2]), pmax(edges[, 1], edges[, 2]))
edges <- unique(edges)

adj_df <- data.frame(
  xs = df.spatial$x[edges[, 1]],
  xe = df.spatial$x[edges[, 2]],
  ys = df.spatial$y[edges[, 1]],
  ye = df.spatial$y[edges[, 2]],
  cts = df.spatial$celltype[edges[, 1]],
  cte = df.spatial$celltype[edges[, 2]],
  cell_s = df.spatial$cell[edges[, 1]],
  cell_e = df.spatial$cell[edges[, 2]]
)
adj_df$dist <- sqrt((adj_df$xs - adj_df$xe)^2 + (adj_df$ys - adj_df$ye)^2)
adj_df <- adj_df %>% filter(dist < (median(dist) + 4 * mad(dist)))
adj_df <- adj_df %>% filter(cell_s != cell_e) %>% select(cell_s, cell_e, cts, cte) %>% unique()
reverse <- adj_df %>% select(cell_e, cell_s, cte, cts) %>% setNames(colnames(adj_df))
adj_df <- rbind(adj_df, reverse) %>% unique()

# Cell type adjacency
ct_adj <- adj_df %>%
  group_by(cell_s, cts, cte) %>%
  dplyr::count() %>%
  group_by(cts, cte) %>%
  dplyr::summarise(n = mean(n)) %>%
  tidyr::pivot_wider(names_from = cts, values_from = n) %>%
  tibble::column_to_rownames('cte') %>%
  as.matrix()
ct_adj[is.na(ct_adj)] <- 0
ct_adj <- ct_adj[colnames(ct_adj), colnames(ct_adj)]

cat("  ct_adj dimensions:", nrow(ct_adj), "x", ncol(ct_adj), "\n")

# === 2. Gene probabilities per type ===
cat("2. Computing gene probabilities...\n")
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

annot <- setNames(cell_meta$celltype, cell_meta$cell)
cids_per_type <- split(names(annot), annot)

gene_probs <- sapply(cids_per_type, function(cids) {
  cids_in <- intersect(cids, colnames(cm))
  if (length(cids_in) == 0) return(rep(0, nrow(cm)))
  Matrix::rowMeans(cm[, cids_in, drop = FALSE] > 0)
})
gene_probs <- t(gene_probs)

cat("  gene_probs dimensions:", nrow(gene_probs), "x", ncol(gene_probs), "\n")

# === 3. Gene scores for each cell type ===
cat("3. Computing gene scores...\n")
all_ctypes <- colnames(ct_adj)
prior_cont_probs <- 1 - diag(ct_adj) / colSums(ct_adj)

gene_scores_list <- list()
cont_probs_list <- list()

for (ct in all_ctypes) {
  # Gene scores
  cont_fracs <- ct_adj[, ct]
  cont_fracs <- cont_fracs[names(cont_fracs) != ct]
  cont_fracs <- cont_fracs / sum(cont_fracs)

  # Weight other types' probabilities by adjacency fractions
  other_types <- names(cont_fracs)
  prob_sub <- gene_probs[other_types, , drop = FALSE]
  weighted <- sweep(prob_sub, 1, cont_fracs, "*")
  cont_scores <- colSums(weighted)
  ref_scores <- gene_probs[ct, ]

  # Align genes
  common_genes <- intersect(names(cont_scores), names(ref_scores))
  gs <- data.frame(cont = cont_scores[common_genes], ref = ref_scores[common_genes],
                   row.names = common_genes)
  gene_scores_list[[ct]] <- gs

  # Contamination probabilities
  p_cu <- 0.25 * prior_cont_probs[ct]
  cp <- p_cu * gs$cont / (p_cu * gs$cont + (1 - p_cu) * gs$ref)
  cp[is.nan(cp)] <- 0
  names(cp) <- rownames(gs)
  cont_probs_list[[ct]] <- cp
}

# === 4. NMF (simplified) ===
cat("4. Running NMF...\n")
h <- 20
cell_counts <- table(df$cell)
cells_keep <- names(cell_counts[cell_counts > h])
df_sub <- df[df$cell %in% cells_keep, ]
genes <- sort(unique(df_sub$gene))

# Build KNN count matrix for NMF input
cat("  Building KNN count matrix...\n")
knn_results <- list()
mol_ids <- c()
for (cell_id in unique(df_sub$cell)) {
  df_cell <- df_sub[df_sub$cell == cell_id, ]
  if (nrow(df_cell) <= h) next
  coords_cell <- as.matrix(df_cell[, c('x', 'y', 'z')])
  nn <- get.knn(coords_cell, k = h)
  xn <- nn$nn.index
  i <- rep(1:nrow(xn), each = ncol(xn))
  j <- c(t(xn))
  adj <- sparseMatrix(i = i, j = j, x = rep(1, length(i)),
                       dims = c(nrow(xn), nrow(xn)))
  diag(adj) <- 1  # include_i = TRUE

  gene_factor <- factor(df_cell$gene, levels = genes)
  gene_ind <- sparseMatrix(
    i = 1:length(gene_factor),
    j = as.integer(gene_factor),
    x = rep(1, length(gene_factor)),
    dims = c(length(gene_factor), length(genes))
  )
  counts <- adj %*% gene_ind
  knn_results[[cell_id]] <- as.matrix(counts)
  mol_ids <- c(mol_ids, df_cell$mol_id)
}

X <- do.call(rbind, knn_results)
rownames(X) <- mol_ids
X <- X[rowSums(X) > 0, colSums(X) > 0]
Z <- matrix(rep(1 / colSums(X), nrow(X)), nrow = nrow(X), byrow = TRUE)

cat("  NMF input dimensions:", nrow(X), "x", ncol(X), "\n")
nmf_res <- NMF::nmf(X, rank = 3, method = 'ls-nmf', weight = Z, nrun = 2, .options = 'vmp1', seed = 0)

cat("  NMF H dimensions:", nrow(nmf_res@fit@H), "x", ncol(nmf_res@fit@H), "\n")
cat("  NMF W dimensions:", nrow(nmf_res@fit@W), "x", ncol(nmf_res@fit@W), "\n")

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

# Serialize results
cat("Serializing results...\n")
result <- list(
  cell_type_adjacency = ct_adj,
  gene_prob_per_type = gene_probs,
  prior_cont_probs = prior_cont_probs,
  cont_probs_per_type = lapply(cont_probs_list, function(x) as.numeric(x)),
  nmf_H = nmf_res@fit@H,
  nmf_W_head = head(nmf_res@fit@W, 20),
  sparse_cor = sc_result,
  test_mat = test_mat,
  norm_im_range = c(min(norm_im), max(norm_im)),
  n_genes = length(genes),
  n_cells = length(unique(df_sub$cell)),
  n_mols_nmf = nrow(X)
)

write_json(result, output_path, auto_unbox = TRUE, digits = NA)
cat("R reference completed.\n")
