#!/usr/bin/env Rscript
# Create a small fixture for fast parity testing.

load('D:/test/cellAdmix-main/cellAdmix-main/data/df.rda')
load('D:/test/cellAdmix-main/cellAdmix-main/data/cell_meta.rda')

set.seed(42)

# Subsample to 50 cells
cells_sampled <- sample(unique(df$cell), 50)
df_small <- df[df$cell %in% cells_sampled, ]
cell_meta_small <- cell_meta[cell_meta$cell %in% cells_sampled, ]

cat("Small fixture:", nrow(df_small), "molecules,", length(unique(df_small$cell)), "cells\n")

out_dir <- 'D:/test/cellAdmix-main/py-cellAdmix/data'
write.csv(df_small, file.path(out_dir, 'fixture_small_df.csv'), row.names = FALSE)
write.csv(cell_meta_small, file.path(out_dir, 'fixture_small_cell_meta.csv'), row.names = FALSE)
saveRDS(list(df = df_small, cell_meta = cell_meta_small), file.path(out_dir, 'fixture_small.rds'))
cat("Exported to", out_dir, "\n")
