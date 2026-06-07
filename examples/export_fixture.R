#!/usr/bin/env Rscript
# Export cellAdmix bundled data to CSV fixtures for parity testing.

suppressMessages({
  library(cellAdmix)
  library(Matrix)
  library(jsonlite)
})

out_dir <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(out_dir)) out_dir <- "D:/test/cellAdmix-main/py-cellAdmix/data"

# Load bundled data
data(df, package = "cellAdmix")
data(cell_meta, package = "cellAdmix")

cat("=== df ===\n")
cat("Dimensions:", nrow(df), "x", ncol(df), "\n")
cat("Columns:", paste(colnames(df), collapse=", "), "\n")
cat("Head:\n")
print(head(df))
cat("\nCell types:", paste(unique(df$celltype), collapse=", "), "\n")
cat("Genes:", length(unique(df$gene)), "\n")
cat("Cells:", length(unique(df$cell)), "\n")

cat("\n=== cell_meta ===\n")
cat("Dimensions:", nrow(cell_meta), "x", ncol(cell_meta), "\n")
cat("Columns:", paste(colnames(cell_meta), collapse=", "), "\n")
print(head(cell_meta))

# Export molecule data
write.csv(df, file.path(out_dir, "fixture_df.csv"), row.names = FALSE)

# Export cell metadata
write.csv(cell_meta, file.path(out_dir, "fixture_cell_meta.csv"), row.names = FALSE)

# Also export as RDS for R reference
saveRDS(list(df = df, cell_meta = cell_meta), file.path(out_dir, "fixture_cosmx.rds"))

cat("\nFixtures exported to:", out_dir, "\n")
