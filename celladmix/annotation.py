"""Factor annotation — bridge test, membrane test, enrichment, false positive checking."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from itertools import permutations

from .knn import knn_adjacency_matrix


def _get_adj_pairs(df_sub, cell_annot, ncm_ct, ncm_cell, ct1, ct2):
    """Get adjacent cell pairs between two cell types."""
    ct1_mol_ids = df_sub.loc[df_sub['celltype'] == ct1, 'mol_id'].values
    ncm_ct1 = ncm_ct.loc[ncm_ct.index.isin(ct1_mol_ids)]

    if ct2 not in ncm_ct1.columns:
        return None

    mol_near_ct2 = ncm_ct1.index[ncm_ct1[ct2] > 0].tolist()
    if len(mol_near_ct2) < 5:
        return None

    # Get cells of type ct2
    ct2_cells = cell_annot[cell_annot == ct2].index.tolist()
    ncm_cell_sub = ncm_cell.loc[ncm_ct.index.isin(mol_near_ct2)]
    ncm_cell_sub = ncm_cell_sub[[c for c in ct2_cells if c in ncm_cell_sub.columns]]
    ncm_cell_sub = ncm_cell_sub > 0

    if ct1 == ct2:
        # Exclude self cells
        ndx_keep = ncm_cell_sub.sum(axis=1) > 1
        if ndx_keep.sum() == 0:
            return None
        ncm_cell_sub = ncm_cell_sub[ndx_keep]
        self_cells = df_sub.loc[ncm_cell_sub.index, 'cell'].values

        # Find non-self adjacent cells
        adj_cells = []
        for i, row in enumerate(ncm_cell_sub.values):
            non_zero_idx = np.where(row)[0]
            non_zero_cells = ncm_cell_sub.columns[non_zero_idx].tolist()
            non_self = [c for c in non_zero_cells if c != self_cells[i]]
            adj_cells.append(non_self[0] if non_self else None)

        valid = [i for i, a in enumerate(adj_cells) if a is not None]
        cell_pair_df = pd.DataFrame({
            'self_cells': [self_cells[i] for i in valid],
            'adj_cells': [adj_cells[i] for i in valid],
        })
    else:
        self_cells = df_sub.loc[ncm_cell_sub.index, 'cell'].values
        adj_cells = []
        for row in ncm_cell_sub.values:
            non_zero_idx = np.where(row)[0]
            adj_cells.append(ncm_cell_sub.columns[non_zero_idx[0]])
        cell_pair_df = pd.DataFrame({'self_cells': self_cells, 'adj_cells': adj_cells})

    # Deduplicate
    cell_pair_df = cell_pair_df.drop_duplicates(subset='self_cells')
    cell_pair_df = cell_pair_df.drop_duplicates(subset='adj_cells')

    if ct1 == ct2:
        mask = ~cell_pair_df['adj_cells'].isin(cell_pair_df['self_cells'])
        cell_pair_df = cell_pair_df[mask]

    return cell_pair_df


def _get_mol_link(df_cell1, df_cell2, knn_k=10):
    """Count bridging molecules between two cells."""
    df_tmp = pd.concat([df_cell2, df_cell1], ignore_index=True)
    if len(df_tmp) <= knn_k:
        knn_k = len(df_tmp) - 1

    nn = NearestNeighbors(n_neighbors=knn_k)
    nn.fit(df_tmp[['x', 'y', 'z']].values)
    _, indices = nn.kneighbors(df_cell1[['x', 'y', 'z']].values)

    index_switch = len(df_cell2)
    bridging_count = sum(
        any(idx <= index_switch for idx in row)
        for row in indices
    )
    return bridging_count


def _compute_bridge_scores(df_crf_diff_fracs, factor_test, df_crf, cell_pair_df,
                           knn_k=10, abs_diff=True):
    """Compute bridge scores for cell pairs."""
    all_scores = {}
    for _, row in cell_pair_df.iterrows():
        cell1 = row['self_cells']
        cell2 = row['adj_cells']

        fracs_c1 = df_crf_diff_fracs.loc[cell1, factor_test]
        fracs_c2 = df_crf_diff_fracs.loc[cell2, factor_test]
        fracs_diff = abs(fracs_c2 - fracs_c1) if abs_diff else fracs_c2 - fracs_c1

        df_c1 = df_crf[df_crf['cell'] == cell1][['x', 'y', 'z', 'factor']]
        df_c2 = df_crf[df_crf['cell'] == cell2][['x', 'y', 'z']]

        f_labs = df_c1['factor'] == factor_test
        df_c1_coords = df_c1[['x', 'y', 'z']]

        df_tmp = pd.concat([df_c1_coords, df_c2], ignore_index=True)
        tmp_k = min(knn_k, len(df_tmp) - 1)

        nn = NearestNeighbors(n_neighbors=tmp_k)
        nn.fit(df_tmp.values)
        _, indices = nn.kneighbors(df_c1_coords.values)

        index_switch = len(df_c1)
        f_labs_idx = np.where(f_labs.values)[0]

        bridging = sum(
            any(idx > index_switch for idx in indices[i])
            for i in f_labs_idx
        )
        frac_cross = bridging / len(f_labs_idx) if len(f_labs_idx) > 0 else 0

        all_scores[cell1] = fracs_diff * frac_cross

    return pd.Series(all_scores)


def run_bridge_test(
    df, crf_f_labs, cell_annot, ncm_ct, ncm_cell,
    ncells_samp=200, knn_k=20, n_cores=1
):
    """Run cell-bridging test for admixture detection.

    Parameters
    ----------
    df : pandas.DataFrame
        Transcript data with cell, celltype, mol_id.
    crf_f_labs : pandas.DataFrame
        CRF factor labels.
    cell_annot : pandas.Series
        Cell type annotations.
    ncm_ct : pandas.DataFrame
        Molecule-to-cell-type KNN counts.
    ncm_cell : pandas.DataFrame
        Molecule-to-cell KNN counts.
    ncells_samp : int
        Number of cells to subsample.
    knn_k : int
        KNN neighborhood size.
    n_cores : int
        Unused.

    Returns
    -------
    dict
        Results per cell type pair.
    """
    df = df.copy()
    df['cell'] = df['cell'].astype(str)

    ct_counts = cell_annot.value_counts()
    all_ctypes = ct_counts[ct_counts > 1].index.tolist()

    # Generate all pairs including same-type
    ct_perms = list(permutations(all_ctypes, 2))
    same_ct = [(ct, ct) for ct in all_ctypes]
    all_pairs = same_ct + ct_perms

    combo_res = {}
    for ct1, ct2 in all_pairs:
        pair_key = f'{ct1}_{ct2}'
        cell_pair_df = _get_adj_pairs(df, cell_annot, ncm_ct, ncm_cell, ct1, ct2)

        if cell_pair_df is None or len(cell_pair_df) == 0:
            combo_res[pair_key] = None
            continue

        # Set factor labels
        if crf_f_labs.shape[1] > 1:
            df['factor'] = crf_f_labs[ct1].values
        else:
            df['factor'] = crf_f_labs.iloc[:, 0].values

        # Compute factor fractions per cell
        cells_all = list(set(cell_pair_df['self_cells'].tolist() + cell_pair_df['adj_cells'].tolist()))
        df_crf = df[df['cell'].isin(cells_all)].copy()

        factor_counts = pd.crosstab(df_crf['cell'], df_crf['factor'])
        factor_fracs = factor_counts.div(factor_counts.sum(axis=1), axis=0)

        # Prune cells
        f_counts = factor_counts.copy()
        all_cells_test = []
        for f_col in f_counts.columns:
            cells_f = f_counts.index[f_counts[f_col] >= 5].tolist()
            cells_f = [c for c in cells_f if c not in all_cells_test]
            if len(cells_f) + len(all_cells_test) > ncells_samp:
                num_needed = ncells_samp - len(all_cells_test)
                if num_needed > 0:
                    cells_f = list(np.random.choice(cells_f, min(num_needed, len(cells_f)), replace=False))
                else:
                    cells_f = []
            all_cells_test.extend(cells_f)

        all_cells_test = list(set(all_cells_test))
        cell_pair_pruned = cell_pair_df[cell_pair_df['self_cells'].isin(all_cells_test)]

        if len(cell_pair_pruned) < 5:
            combo_res[pair_key] = None
            continue

        # Compute real scores
        all_factors = sorted(factor_counts.columns)
        factor_stats = {}

        for f in all_factors:
            scores_real = _compute_bridge_scores(
                factor_fracs, f, df_crf, cell_pair_pruned, knn_k=knn_k
            )

            # Null scores (3 iterations)
            scores_null_iter = []
            for _ in range(3):
                null_scores = _compute_null_bridge_scores(
                    factor_fracs, cell_pair_pruned, df_crf, knn_k, ncells_samp
                )
                scores_null_iter.append(null_scores)

            # Average null scores
            if scores_null_iter and scores_null_iter[0] is not None:
                common_cells = set(scores_null_iter[0].index)
                for s in scores_null_iter[1:]:
                    common_cells &= set(s.index)
                common_cells = list(common_cells)

                if common_cells:
                    null_mean = np.mean([s.loc[common_cells].values for s in scores_null_iter], axis=0)
                    null_mean = pd.Series(null_mean, index=common_cells)

                    real_aligned = scores_real.reindex(common_cells).dropna()
                    null_aligned = null_mean.reindex(real_aligned.index)

                    if len(real_aligned) > 1:
                        from scipy.stats import wilcoxon
                        try:
                            stat, pval = wilcoxon(real_aligned, null_aligned, alternative='greater')
                        except ValueError:
                            pval = 1.0
                        q3 = np.percentile(real_aligned, 75)
                        factor_stats[f] = [pval, q3]
                    else:
                        factor_stats[f] = [np.nan, np.nan]
                else:
                    factor_stats[f] = [np.nan, np.nan]
            else:
                factor_stats[f] = [np.nan, np.nan]

        combo_res[pair_key] = factor_stats

    return combo_res


def _compute_null_bridge_scores(df_crf_diff_fracs, cell_pair_df, df_crf, knn_k, ncells_samp):
    """Compute null bridge scores by random cell pairing."""
    # Find similar cells by factor fraction distance
    all_cells = cell_pair_df['adj_cells'].unique()
    if len(all_cells) < 2:
        return None

    fracs = df_crf_diff_fracs.loc[df_crf_diff_fracs.index.isin(all_cells)]
    if len(fracs) < 2:
        return None

    from scipy.spatial.distance import pdist, squareform
    dist_matrix = pd.DataFrame(
        squareform(pdist(fracs.values)),
        index=fracs.index, columns=fracs.index
    )

    null_scores = {}
    for _, row in cell_pair_df.iterrows():
        cell1 = row['self_cells']
        cell2_orig = row['adj_cells']

        if cell2_orig not in dist_matrix.index:
            continue

        # Find top 10% most similar cells
        dists = dist_matrix[cell2_orig].drop(cell2_orig).sort_values()
        top_n = max(2, int(len(dists) * 0.1) + 1)
        top_cells = dists.head(top_n).index.tolist()

        cell2_samp = np.random.choice(top_cells)
        df_c1 = df_crf[df_crf['cell'] == cell1][['x', 'y', 'z']]
        df_c2_samp = df_crf[df_crf['cell'] == cell2_samp][['x', 'y', 'z']]

        # Move cell2 close enough to cell1
        num_cross = _get_mol_link(df_c1, df_crf[df_crf['cell'] == cell2_orig][['x', 'y', 'z']], knn_k)

        df_c2_moved = _move_cells_adj(df_c1, df_c2_samp, num_cross, knn_k)
        if df_c2_moved is None:
            continue

        # Compute score
        f_labs_c1 = df_crf[df_crf['cell'] == cell1]['factor'].values
        for f in df_crf_diff_fracs.columns:
            f_mask = f_labs_c1 == f
            if f_mask.sum() == 0:
                continue

            df_tmp = pd.concat([df_c1, df_c2_moved], ignore_index=True)
            tmp_k = min(knn_k, len(df_tmp) - 1)
            nn = NearestNeighbors(n_neighbors=tmp_k)
            nn.fit(df_tmp.values)
            _, indices = nn.kneighbors(df_c1.values)

            bridging = sum(
                any(idx > len(df_c1) for idx in indices[i])
                for i in np.where(f_mask)[0]
            )
            frac_cross = bridging / f_mask.sum()

            fracs_diff = abs(
                df_crf_diff_fracs.loc[cell2_samp, f] - df_crf_diff_fracs.loc[cell1, f]
            )
            null_scores[f'{cell1}_{f}'] = fracs_diff * frac_cross

    return pd.Series(null_scores) if null_scores else None


def _move_cells_adj(df_cell1, df_cell2, num_cross_from_target, knn_k=10, step=0.02, max_iter=100):
    """Move cell2 adjacent to cell1 to match target bridging count."""
    c1_center = df_cell1[['x', 'y']].mean()
    c2_center = df_cell2[['x', 'y']].mean()

    r1 = (df_cell1['x'].max() - df_cell1['x'].min()) / 2
    r2 = (df_cell1['y'].max() - df_cell1['y'].min()) / 2
    r = max(r1, r2) + (df_cell2['x'].max() - df_cell2['x'].min()) / 2

    theta = np.random.uniform(0, 2 * np.pi)
    x2 = c1_center['x'] + r * np.cos(theta)
    y2 = c1_center['y'] + r * np.sin(theta)

    df_cell2 = df_cell2.copy()
    df_cell2['x'] = df_cell2['x'] + (x2 - c2_center['x'])
    df_cell2['y'] = df_cell2['y'] + (y2 - c2_center['y'])

    for i in range(max_iter):
        count = _get_mol_link(df_cell1, df_cell2, knn_k)
        if count >= num_cross_from_target:
            return df_cell2

        direction = df_cell1[['x', 'y']].mean() - df_cell2[['x', 'y']].mean()
        df_cell2['x'] = df_cell2['x'] + step * direction['x']
        df_cell2['y'] = df_cell2['y'] + step * direction['y']

    return None


def extract_bridge_res(
    combo_res, all_ctypes, n_factors,
    nmf_type='joint', p_thresh=0.01, adj_pvals=True, get_p_mat=False
):
    """Extract bridge test results into factor-cell type pairs for removal.

    Parameters
    ----------
    combo_res : dict
        From run_bridge_test.
    all_ctypes : list
        All cell types.
    n_factors : int
        Number of factors.
    nmf_type : str
        'joint' or 'ct'.
    p_thresh : float
        P-value threshold.
    adj_pvals : bool
        Whether to FDR-adjust p-values.
    get_p_mat : bool
        If True, return p-value matrices.

    Returns
    -------
    list or dict
        Factor-cell type pairs for removal.
    """
    all_qsc_mats = {}
    all_pval_mats = {}

    for factor_test in range(1, n_factors + 1):
        mat_qsc = pd.DataFrame(np.nan, index=all_ctypes, columns=all_ctypes)
        mat_pval = mat_qsc.copy()

        for pair_key, pair_res in combo_res.items():
            if pair_res is None:
                continue

            parts = pair_key.split('_')
            ct1 = parts[0]
            ct2 = '_'.join(parts[1:]) if len(parts) > 2 else parts[1]

            if factor_test not in pair_res:
                continue

            pval, qsc = pair_res[factor_test]
            mat_qsc.loc[ct1, ct2] = qsc
            mat_pval.loc[ct1, ct2] = pval

        all_qsc_mats[factor_test] = mat_qsc

        if adj_pvals:
            from scipy.stats import false_discovery_control
            pvals_flat = mat_pval.values.ravel()
            valid = ~np.isnan(pvals_flat)
            if valid.sum() > 0:
                pvals_adj = pvals_flat.copy()
                pvals_adj[valid] = false_discovery_control(pvals_flat[valid])
                mat_pval_fdr = pd.DataFrame(
                    -np.log10(pvals_adj.reshape(mat_pval.shape)),
                    index=mat_pval.index, columns=mat_pval.columns
                )
            else:
                mat_pval_fdr = -np.log10(mat_pval)
            all_pval_mats[factor_test] = mat_pval_fdr
        else:
            all_pval_mats[factor_test] = -np.log10(mat_pval)

    if get_p_mat:
        return all_pval_mats

    if nmf_type == 'joint':
        # Collapse to row max and column mean
        all_pval_mats_rs = {}
        all_pval_mats_cs = {}
        for f in all_pval_mats:
            mat = all_pval_mats[f]
            all_pval_mats_rs[f] = mat.max(axis=1)
            all_pval_mats_cs[f] = mat.mean(axis=0)

        bridge_res = _select_bridge_remove(all_pval_mats_rs, all_pval_mats_cs, p_thresh)
    else:
        bridge_res = {}
        for ct in all_ctypes:
            ct_from = {}
            for f in all_pval_mats:
                mat = all_pval_mats[f]
                if ct not in mat.index:
                    continue
                admix = mat.loc[ct].dropna()
                if len(admix) == 0:
                    continue
                ct_rm = admix[admix > -np.log10(p_thresh)].index.tolist()
                if ct_rm:
                    ct_from[f'f_{f}'] = admix.idxmax()
            if ct_from:
                ct_to = {k: ct for k in ct_from}
                bridge_res[ct] = [ct_to, ct_from]

    return bridge_res


def _select_bridge_remove(all_pval_mats_rs, all_pval_mats_cs, p_thresh):
    """Select factor-ctype pairs for removal."""
    f_ct_rm = {}
    f_from = {}

    for f in all_pval_mats_rs:
        pval_rs = all_pval_mats_rs[f].dropna()
        ct_rm = pval_rs[pval_rs > -np.log10(p_thresh)].index.tolist()

        pval_cs = all_pval_mats_cs[f].dropna()
        if len(pval_cs) == 0:
            continue
        ct_from = pval_cs.idxmax()

        ct_rm = [ct for ct in ct_rm if ct != ct_from]
        if ct_rm:
            f_ct_rm[f'f_{f}'] = ct_rm
        if ct_from and ct_rm:
            f_from[f'f_{f}'] = ct_from

    return [f_ct_rm, f_from]


def run_memb_test(
    df, crf_f_labs, im_load_fn, num_factors, min_mols=5,
    max_cells_per_fov=10, knn_k=10, all_fov=None,
    use_block_knn=False, c_dist_expand=0.025,
    ncm_ct=None, ncm_cell=None, n_cores=1
):
    """Run membrane test for factor-ctype pair detection.

    Parameters
    ----------
    df : pandas.DataFrame
        Molecule data with fov, cell, celltype, z columns.
    crf_f_labs : pandas.DataFrame
        CRF factor labels.
    im_load_fn : callable
        Function to load images for a FOV.
    num_factors : int
        Number of factors.
    min_mols : int
        Min admixture molecules per cell.
    max_cells_per_fov : int
        Max cells per FOV to test.
    knn_k : int
        KNN neighborhood size.
    all_fov : list or None
        FOVs to test.
    use_block_knn : bool
        Whether to use block KNN.
    c_dist_expand : float
        Distance expansion factor.
    ncm_ct : DataFrame or None
        Pre-computed molecule-to-celltype KNN counts.
    ncm_cell : DataFrame or None
        Pre-computed molecule-to-cell KNN counts.
    n_cores : int
        Unused.

    Returns
    -------
    dict
        Membrane test scores per factor-ctype combination.
    """
    all_ct = df['celltype'].unique().tolist()
    if all_fov is None:
        all_fov = sorted(df['fov'].unique())

    # Cell type pairs
    ct_perms = list(permutations(all_ct, 2))
    same_ct = [(ct, ct) for ct in all_ct]
    all_pairs = same_ct + ct_perms

    fov_res = []
    for fov in all_fov:
        df_fov = df[df['fov'] == fov].copy()
        df_fov['mol_id'] = range(1, len(df_fov) + 1)
        df_fov.index = df_fov['mol_id']
        cell_annot = df_fov[['cell', 'celltype']].drop_duplicates().set_index('cell')['celltype']

        # Get KNN counts if needed
        if ncm_ct is None or ncm_cell is None:
            from .knn import knn_adjacency_matrix as _knn_adj
            adj = _knn_adj(df_fov, k=knn_k)
            # Build ncm_ct and ncm_cell
            ct_factor = pd.Categorical(df_fov['celltype'])
            ncm_ct = adj @ sparse.csr_matrix(
                (np.ones(len(ct_factor)), (np.arange(len(ct_factor)), ct_factor.codes)),
                shape=(len(ct_factor), len(ct_factor.categories))
            )
            ncm_ct = pd.DataFrame(
                ncm_ct.toarray(),
                index=df_fov['mol_id'],
                columns=ct_factor.categories
            )

            cell_factor = pd.Categorical(df_fov['cell'])
            ncm_cell = adj @ sparse.csr_matrix(
                (np.ones(len(cell_factor)), (np.arange(len(cell_factor)), cell_factor.codes)),
                shape=(len(cell_factor), len(cell_factor.categories))
            )
            ncm_cell = pd.DataFrame(
                ncm_cell.toarray(),
                index=df_fov['mol_id'],
                columns=cell_factor.categories
            )

        # Rename z column
        if 'z' in df_fov.columns and 'z_index' not in df_fov.columns:
            df_fov = df_fov.rename(columns={'z': 'z_index'})

        # Load images
        im = im_load_fn(fov=fov, normalize=True)

        f_res = {}
        for f in range(1, num_factors + 1):
            results = {}
            for ct1, ct2 in all_pairs:
                pair_key = f'{ct1}_{ct2}'
                cell_pair_df = _get_adj_pairs(df_fov, cell_annot, ncm_ct, ncm_cell, ct1, ct2)

                if cell_pair_df is None or len(cell_pair_df) == 0:
                    results[pair_key] = []
                    continue

                # Set factor labels
                if crf_f_labs.shape[1] > 1:
                    df_fov['factor'] = crf_f_labs.loc[df_fov.index, ct1].values
                else:
                    df_fov['factor'] = crf_f_labs.iloc[:, 0].values

                df_fov['is_admixture'] = df_fov['factor'] == f

                cells_in_pairs = set(cell_pair_df['self_cells'].tolist() + cell_pair_df['adj_cells'].tolist())
                df_sub = df_fov[df_fov['cell'].isin(cells_in_pairs)]
                df_sub_split = {cell: group for cell, group in df_sub.groupby('cell')}

                pres = _compute_membrane_sep_per_cell_factor(
                    df_sub_split, im, cell_pair_df, min_mols=min_mols,
                    max_cells_per_fov=max_cells_per_fov, c_dist_expand=c_dist_expand
                )
                results[pair_key] = pres

            f_res[f] = results
        fov_res.append(f_res)

    # Stitch results
    ct_ct_all = [f'{ct1}_{ct2}' for ct1, ct2 in all_pairs]
    scores_final = {}

    for f in range(1, num_factors + 1):
        for ct_ct in ct_ct_all:
            key = f'{f}_{ct_ct}'
            all_scores = []
            for fov_data in fov_res:
                if f in fov_data and ct_ct in fov_data[f]:
                    all_scores.extend(fov_data[f][ct_ct])
            scores_final[key] = all_scores

    return scores_final


def _compute_membrane_sep_per_cell_factor(
    df_fov_sub, im, cell_pair_df, min_mols=5, max_cells_per_fov=10, c_dist_expand=0.025
):
    """Compute membrane separation scores per cell pair."""
    results = []
    non_na_count = 0

    for _, pair_row in cell_pair_df.iterrows():
        if non_na_count >= max_cells_per_fov:
            results.append(np.nan)
            continue

        cur_cell = pair_row['self_cells']
        adj_cell = pair_row['adj_cells']

        if cur_cell not in df_fov_sub or adj_cell not in df_fov_sub:
            results.append(np.nan)
            continue

        cur_df = df_fov_sub[cur_cell].copy()
        adj_df = df_fov_sub[adj_cell].copy()

        if 'is_admixture' not in cur_df.columns:
            results.append(np.nan)
            continue

        if cur_df['is_admixture'].sum() < min_mols:
            results.append(np.nan)
            continue

        cur_cent = np.array([cur_df['x'].mean(), cur_df['y'].mean()])
        adj_cent = np.array([adj_df['x'].mean(), adj_df['y'].mean()])

        # Compute end distances
        cur_xy = cur_df[['x', 'y']].values
        end_dists = np.zeros(len(cur_xy))
        for i in range(len(cur_xy)):
            direction = cur_cent - cur_xy[i]
            length = np.linalg.norm(direction)
            if length > 0:
                unit = direction / length
                cur_end = cur_xy[i] + unit
                adj_end = cur_xy[i] + (adj_cent - cur_xy[i]) / np.linalg.norm(adj_cent - cur_xy[i])
                end_dists[i] = np.linalg.norm(cur_end - adj_end) / 2

        # Compute scores for admixture molecules
        admix_mask = cur_df['is_admixture'].values
        if admix_mask.sum() == 0:
            results.append(np.nan)
            continue

        scores = []
        scores_perm = []

        for i in np.where(admix_mask)[0]:
            mol_id = cur_df.index[i]
            if end_dists[i] == 0:
                continue

            # Real score
            score = end_dists[i]
            scores.append(score)

            # Permuted score (random molecule at similar distance)
            same_z = cur_df['z_index'] == cur_df.iloc[i]['z_index']
            d_real = np.linalg.norm(cur_xy[i] - cur_cent)
            d_range = (d_real - d_real * c_dist_expand, d_real + d_real * c_dist_expand)

            candidates = np.where(same_z & (cur_df.index != mol_id))[0]
            if len(candidates) > 1:
                cand_dists = np.array([np.linalg.norm(cur_xy[j] - cur_cent) for j in candidates])
                in_range = (cand_dists > d_range[0]) & (cand_dists < d_range[1])
                if in_range.sum() > 0:
                    j = np.random.choice(candidates[in_range])
                    scores_perm.append(end_dists[j])

        if len(scores) == 0:
            results.append(np.nan)
            continue

        scores = np.array(scores)
        scores_perm = np.array(scores_perm[:len(scores)])
        if len(scores_perm) > 0:
            score_rel = np.mean(np.log(scores / scores_perm))
        else:
            score_rel = np.nan

        results.append(score_rel)
        non_na_count += 1

    return [r for r in results if not np.isnan(r)]


def extract_memb_res(
    combo_res, all_ctypes, nmf_type,
    p_thresh=0.01, adj_pvals=False, get_p_mat=False
):
    """Extract membrane test results.

    Parameters
    ----------
    combo_res : dict
        From run_memb_test.
    all_ctypes : list
        All cell types.
    nmf_type : str
        'joint' or 'ct'.
    p_thresh : float
        P-value threshold.
    adj_pvals : bool
        Whether to FDR-adjust.
    get_p_mat : bool
        If True, return p-value matrices.

    Returns
    -------
    list or dict
        Factor-cell type pairs for removal.
    """
    if combo_res is None:
        return None

    # Parse factor keys
    f_all = sorted(set(int(k.split('_')[0]) for k in combo_res.keys()))

    fmats = {}
    pmats = {}

    for f in f_all:
        scores_sub = {k: v for k, v in combo_res.items() if k.startswith(f'{f}_')}
        # Parse cell type pairs
        ct_pairs = {}
        for key, scores in scores_sub.items():
            parts = key.split('_')
            ct_from = parts[-1]
            ct_to = '_'.join(parts[1:-1])

            if not scores:
                continue

            scores_arr = np.array([s[0] for s in scores if isinstance(s, (list, tuple)) and len(s) > 0])
            scores_arr = scores_arr[np.isfinite(scores_arr)]

            if len(scores_arr) <= 1:
                continue

            from scipy.stats import ttest_1samp
            _, pval = ttest_1samp(scores_arr, 0, alternative='greater')
            myscore = (scores_arr > 0).sum() / len(scores_arr)

            if ct_to not in ct_pairs:
                ct_pairs[ct_to] = {}
            ct_pairs[ct_to][ct_from] = {'score': myscore, 'pval': pval}

        # Build matrices
        ct_from_all = list(set(ct_from for pairs in ct_pairs.values() for ct_from in pairs))
        ct_to_all = list(ct_pairs.keys())

        fmat = pd.DataFrame(np.nan, index=ct_to_all, columns=ct_from_all)
        pmat = fmat.copy()

        for ct_to, pairs in ct_pairs.items():
            for ct_from, data in pairs.items():
                fmat.loc[ct_to, ct_from] = data['score']
                pmat.loc[ct_to, ct_from] = data['pval']

        if adj_pvals:
            pvals_flat = pmat.values.ravel()
            valid = ~np.isnan(pvals_flat)
            if valid.sum() > 0:
                from scipy.stats import false_discovery_control
                pvals_adj = pvals_flat.copy()
                pvals_adj[valid] = false_discovery_control(pvals_flat[valid])
                pmat_fdr = pd.DataFrame(
                    -np.log10(pvals_adj.reshape(pmat.shape)),
                    index=pmat.index, columns=pmat.columns
                )
            else:
                pmat_fdr = -np.log10(pmat)
            pmats[f] = pmat_fdr
        else:
            pmats[f] = -np.log10(pmat)

        fmats[f] = fmat

    if get_p_mat:
        return pmats

    if nmf_type == 'joint':
        # Collapse
        all_pval_mats_rs = {}
        all_pval_mats_cs = {}
        for f in pmats:
            mat = pmats[f]
            all_pval_mats_rs[f] = mat.max(axis=1)
            all_pval_mats_cs[f] = mat.mean(axis=0)

        memb_res = _select_memb_remove(all_pval_mats_rs, all_pval_mats_cs, p_thresh)
    else:
        memb_res = {}
        for ct in all_ctypes:
            ct_from = {}
            for f in pmats:
                mat = pmats[f]
                if ct not in mat.index:
                    continue
                admix = mat.loc[ct].dropna()
                if len(admix) == 0:
                    continue
                ct_rm = admix[admix > -np.log10(p_thresh)].index.tolist()
                if ct_rm:
                    ct_from[f'f_{f}'] = admix.idxmax()
            if ct_from:
                memb_res[ct] = [{ct: ct for ct in ct_from}, ct_from]

    return memb_res


def _select_memb_remove(all_pval_mats_rs, all_pval_mats_cs, p_thresh):
    """Select membrane-based removals."""
    f_ct_rm = {}
    f_from = {}

    for f in all_pval_mats_rs:
        pval_rs = all_pval_mats_rs[f]
        ct_rm = pval_rs[pval_rs > -np.log10(p_thresh)].index.tolist()

        pval_cs = all_pval_mats_cs[f].dropna()
        if len(pval_cs) == 0:
            continue
        ct_from = pval_cs.idxmax()

        ct_rm = [ct for ct in ct_rm if ct != ct_from]
        if ct_rm:
            f_ct_rm[f'f_{f}'] = ct_rm
        if ct_from and ct_rm:
            f_from[f'f_{f}'] = ct_from

    return [f_ct_rm, f_from]


def get_enr(
    res, markers, n_perm=10000, p_thresh=0.1, adj_pvals=True,
    crf_h=None, df=None, n_cores=1
):
    """Run enrichment test for marker genes in NMF factors.

    Parameters
    ----------
    res : dict
        NMF results.
    markers : pandas.DataFrame
        Columns: gene, marker_of.
    n_perm : int
        Number of permutations.
    p_thresh : float
        P-value threshold.
    adj_pvals : bool
        Whether to FDR-adjust.
    crf_h : int or None
        CRF neighborhood size.
    df : pandas.DataFrame or None
        Molecule data (needed if genes were subsetted).
    n_cores : int
        Unused.

    Returns
    -------
    list
        [f_ct_rm, f_from] — factor-cell type pairs for removal.
    """
    # Get gene loadings
    if res['gene_sub']:
        if df is None:
            raise ValueError('Need df since genes were subsetted')
        if crf_h is None:
            raise ValueError('Need crf_h since genes were subsetted')
        from .nmf_crf import project_nmf
        H = project_nmf(df, res, nmf_h=crf_h, n_cores=n_cores)
    else:
        H = res['H']

    k = H.shape[0]

    # Build loading dataframe
    H_all = pd.DataFrame(H.T, index=res['feature_names'],
                         columns=[str(i+1) for i in range(k)])
    H_all = H_all.reset_index().rename(columns={'index': 'gene'})
    H_melted = H_all.melt(id_vars='gene', var_name='factor', value_name='loading')
    H_melted['frac'] = H_melted.groupby('factor')['loading'].transform(lambda x: x / x.sum())

    # Pivot to gene x factor fraction matrix
    V = H_melted.pivot(index='gene', columns='factor', values='frac').fillna(0)

    # Subset markers to tested genes
    markers = markers[markers['gene'].isin(V.index)]

    # Filter cell types with few markers
    marker_counts = markers['marker_of'].value_counts()
    ct_keep = marker_counts[marker_counts > 2].index
    markers = markers[markers['marker_of'].isin(ct_keep)]

    # Permutation test
    marker_n = markers.groupby('marker_of').size()

    marker_sim_list = []
    rng = np.random.RandomState(42)
    for marker_of, n in marker_n.items():
        for i in range(n_perm):
            for factor in V.columns:
                sampled = V[factor].sample(n=n, random_state=rng).sum()
                marker_sim_list.append({
                    'factor': factor, 'frac': sampled, 'i': i, 'marker_of': marker_of
                })

    marker_sim = pd.DataFrame(marker_sim_list)

    # Compute observed fractions
    dat_marker = H_melted.merge(markers, on='gene')
    dat_marker = dat_marker.groupby(['factor', 'marker_of'])['frac'].sum().reset_index()

    # Compute p-values
    all_pv = []
    for _, row in dat_marker.iterrows():
        ct = row['marker_of']
        f = row['factor']
        frac = row['frac']

        sim_sub = marker_sim[(marker_sim['factor'] == f) & (marker_sim['marker_of'] == ct)]
        pval = (sim_sub['frac'] > frac).sum() / len(sim_sub)
        if pval == 0:
            pval = 1 / len(sim_sub)
        all_pv.append(pval)

    dat_marker['pval'] = all_pv

    # FDR adjustment
    if adj_pvals:
        dat_marker['fdr'] = dat_marker.groupby('factor')['pval'].transform(
            lambda x: np.minimum(x * len(x), 1.0)  # Bonferroni-like
        )
        dat_marker_sig = dat_marker[dat_marker['fdr'] < p_thresh]
    else:
        dat_marker_sig = dat_marker[dat_marker['pval'] < p_thresh]

    all_ct = markers['marker_of'].unique()
    f_test = dat_marker_sig['factor'].unique()

    f_ct_rm = {}
    f_from = {}

    for f in f_test:
        sig_sub = dat_marker_sig[dat_marker_sig['factor'] == f]
        ct_rm = [ct for ct in all_ct if ct not in sig_sub['marker_of'].values]
        if ct_rm:
            f_ct_rm[f'f_{f}'] = ct_rm
        f_from[f'f_{f}'] = sig_sub['marker_of'].unique().tolist()

    return [f_ct_rm, f_from]


def check_f_rm(
    cell_annot, f_ct_rm, f_from, cf_fracs_all, ncm_ct_full,
    do_clean=True, median_thresh=0.1
):
    """Check for false positive removals.

    Parameters
    ----------
    cell_annot : pandas.Series
        Cell type annotations.
    f_ct_rm : dict
        Factor-cell type pairs for removal.
    f_from : dict
        Source cell types per factor.
    cf_fracs_all : pandas.DataFrame
        Factor fractions per cell.
    ncm_ct_full : pandas.DataFrame
        Cell-to-celltype neighbor counts.
    do_clean : bool
        Whether to remove false positives.
    median_thresh : float
        Median expression threshold.

    Returns
    -------
    list or str
        Cleaned factor-ctype pairs for removal.
    """
    if f_ct_rm is None:
        return None
    if len(f_ct_rm) == 0:
        return None

    all_ct = cell_annot.unique()
    f_ct_rm_clean = dict(f_ct_rm)

    for f_nm, ct_rm_list in f_ct_rm.items():
        f = f_nm.split('_')[1]
        admix_from = f_from.get(f_nm, [None])[0]
        if admix_from is None:
            continue

        all_admix_to = [ct for ct in ct_rm_list if ct != admix_from]
        all_other_ct = [ct for ct in all_ct if ct != admix_from]

        f_ct_rm_clean[f_nm] = all_admix_to

        # Get distant cell fractions
        for admix_to in all_admix_to:
            if admix_to not in all_ct:
                f_ct_rm_clean[f_nm] = [ct for ct in f_ct_rm_clean[f_nm] if ct != admix_to]
                continue

            cells_test = cell_annot[cell_annot == admix_to].index.tolist()
            ncm_sub = ncm_ct_full.loc[ncm_ct_full.index.isin(cells_test)]

            if admix_from in ncm_sub.columns:
                cells_keep = ncm_sub.index[ncm_sub[admix_from] == 0].tolist()
            else:
                cells_keep = cells_test

            if len(cells_keep) <= 1:
                f_ct_rm_clean[f_nm] = [ct for ct in f_ct_rm_clean[f_nm] if ct != admix_to]
                continue

            factor_fracs = cf_fracs_all.loc[cf_fracs_all.index.isin(cells_keep), f].values
            cells_bg = [c for c in cells_test if c not in cells_keep]

            if len(cells_bg) <= 1:
                f_ct_rm_clean[f_nm] = [ct for ct in f_ct_rm_clean[f_nm] if ct != admix_to]
                continue

            factor_fracs_bg = cf_fracs_all.loc[cf_fracs_all.index.isin(cells_bg), f].values

            if np.all(factor_fracs == 0) and np.all(factor_fracs_bg == 0):
                f_ct_rm_clean[f_nm] = [ct for ct in f_ct_rm_clean[f_nm] if ct != admix_to]
                continue

            # Outlier test
            frac_expr = {}
            for ct in all_other_ct:
                ct_cells = cell_annot[cell_annot == ct].index.tolist()
                ct_fracs = cf_fracs_all.loc[cf_fracs_all.index.isin(ct_cells), f].values
                frac_expr[ct] = (ct_fracs > 0.05).sum() / len(ct_fracs) if len(ct_fracs) > 0 else 0

            frac_arr = np.array(list(frac_expr.values()))
            iqr = np.percentile(frac_arr, 75) - np.percentile(frac_arr, 25)
            upper = np.percentile(frac_arr, 75) + 1.5 * iqr

            admix_to_idx = list(frac_expr.keys()).index(admix_to)
            is_outlier = frac_arr[admix_to_idx] > upper and frac_arr[admix_to_idx] > 0.1

            med_expr = np.median(factor_fracs)
            expr_diff = factor_fracs_bg.mean() - factor_fracs.mean()

            if do_clean:
                if med_expr > median_thresh or is_outlier or expr_diff < 0:
                    f_ct_rm_clean[f_nm] = [ct for ct in f_ct_rm_clean[f_nm] if ct != admix_to]

    # Format output
    result = []
    for f_nm, ct_list in f_ct_rm_clean.items():
        f = f_nm.split('_')[1]
        for ct in ct_list:
            result.append(f'{f}_{ct}')

    return result if result else None


def check_f_rm_per_ct(
    cell_annot, mb_res, ct_fracs_per_ct, ncm_ct_full,
    do_clean=True, median_thresh=0.1
):
    """Check false positive removals per cell type.

    Parameters
    ----------
    cell_annot : pandas.Series
        Cell type annotations.
    mb_res : dict
        Per-cell-type results.
    ct_fracs_per_ct : dict
        Factor fractions per cell type.
    ncm_ct_full : DataFrame
        Cell-to-celltype neighbor counts.
    do_clean : bool
        Whether to clean.
    median_thresh : float
        Median threshold.

    Returns
    -------
    dict
        Per-cell-type cleaned results.
    """
    if mb_res is None:
        return None

    ct_res_fin = {}
    for ct, (f_ct_rm, f_from) in mb_res.items():
        cf_fracs = ct_fracs_per_ct.get(ct)
        if cf_fracs is None:
            continue
        check_res = check_f_rm(
            cell_annot, f_ct_rm, f_from, cf_fracs, ncm_ct_full,
            do_clean=do_clean, median_thresh=median_thresh
        )
        ct_res_fin[ct] = check_res

    return ct_res_fin
