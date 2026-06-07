#!/usr/bin/env Rscript
# Load cellAdmix bundled data and export to CSV.

load('D:/test/cellAdmix-main/cellAdmix-main/data/df.rda')
load('D:/test/cellAdmix-main/cellAdmix-main/data/cell_meta.rda')

cat('=== df ===\n')
cat('Dimensions:', nrow(df), 'x', ncol(df), '\n')
cat('Columns:', paste(colnames(df), collapse=', '), '\n')
print(head(df))
cat('\nCell types:', paste(unique(df$celltype), collapse=', '), '\n')
cat('Genes:', length(unique(df$gene)), '\n')
cat('Cells:', length(unique(df$cell)), '\n')

cat('\n=== cell_meta ===\n')
cat('Dimensions:', nrow(cell_meta), 'x', ncol(cell_meta), '\n')
cat('Columns:', paste(colnames(cell_meta), collapse=', '), '\n')
print(head(cell_meta))

out_dir <- 'D:/test/cellAdmix-main/py-cellAdmix/data'
write.csv(df, file.path(out_dir, 'fixture_df.csv'), row.names=FALSE)
write.csv(cell_meta, file.path(out_dir, 'fixture_cell_meta.csv'), row.names=FALSE)
saveRDS(list(df=df, cell_meta=cell_meta), file.path(out_dir, 'fixture_cosmx.rds'))
cat('\nExported to', out_dir, '\n')
