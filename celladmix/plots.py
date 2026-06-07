"""Plotting functions — matplotlib/seaborn equivalents of cellAdmix R plots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_nmf_loadings(nmf_res, ct_exp=None, f_show=None, nplt_row=2):
    """Plot top NMF loadings per factor.

    Parameters
    ----------
    nmf_res : dict
        NMF results from run_knn_nmf.
    ct_exp : pandas.DataFrame or None
        Scaled scRNA expression (columns: gene, celltype, z).
    f_show : list or None
        Factors to show.
    nplt_row : int
        Number of subplot rows.

    Returns
    -------
    matplotlib.figure.Figure
    """
    H = nmf_res['H']  # factors x genes
    k = H.shape[0]
    genes = nmf_res['feature_names']

    if f_show is None:
        f_show = list(range(1, k + 1))

    n_plots = len(f_show)
    n_cols = int(np.ceil(n_plots / nplt_row))

    if ct_exp is not None:
        fig, axes = plt.subplots(nplt_row, n_cols * 2, figsize=(4 * n_cols, 3 * nplt_row))
        if nplt_row == 1:
            axes = axes.reshape(1, -1)
    else:
        fig, axes = plt.subplots(nplt_row, n_cols, figsize=(3 * n_cols, 3 * nplt_row))
        if nplt_row == 1:
            axes = axes.reshape(1, -1)

    for plot_idx, f in enumerate(f_show):
        row = plot_idx // n_cols
        col = plot_idx % n_cols

        h = H[f - 1]
        fracs = h / h.sum() if h.sum() > 0 else h
        gene_order = np.argsort(fracs)[-10:][::-1]
        gene_names = [genes[i] for i in gene_order]
        frac_vals = fracs[gene_order]

        if ct_exp is not None:
            ax_bar = axes[row, col * 2]
            ax_ct = axes[row, col * 2 + 1]
        else:
            ax_bar = axes[row, col]
            ax_ct = None

        # Bar plot
        ax_bar.barh(range(len(gene_names)), frac_vals, color='steelblue')
        ax_bar.set_yticks(range(len(gene_names)))
        ax_bar.set_yticklabels(gene_names, fontsize=7)
        ax_bar.set_xlabel('Loadings fraction', fontsize=7)
        ax_bar.set_ylabel(f'Factor {f}', fontsize=8, fontweight='bold')
        ax_bar.invert_yaxis()

        # Cell type heatmap
        if ax_ct is not None and ct_exp is not None:
            ct_sub = ct_exp[ct_exp['gene'].isin(gene_names)]
            ct_pivot = ct_sub.pivot(index='gene', columns='celltype', values='z')
            ct_pivot = ct_pivot.reindex(gene_names)

            cmap = LinearSegmentedColormap.from_list('rwb', ['blue', 'white', 'red'])
            sns.heatmap(
                ct_pivot, ax=ax_ct, cmap=cmap, center=0, vmin=-2, vmax=2,
                cbar=False, linewidths=0.5, linecolor='gray'
            )
            ax_ct.set_xticklabels(ax_ct.get_xticklabels(), rotation=60, ha='right', fontsize=6)
            ax_ct.set_yticklabels(ax_ct.get_yticklabels(), fontsize=6)
            ax_ct.set_xlabel('')
            ax_ct.set_ylabel('')

    plt.tight_layout()
    return fig


def plot_cell_score_ratios(
    scores_orig, scores_cln, f_rm, df, crf_res,
    upper_lim=1.5, min_mean_thresh=1e-3
):
    """Plot admixture score ratios before/after cleaning.

    Parameters
    ----------
    scores_orig : pandas.DataFrame
        Original scores with 'cell', 'cell_type', 'scores'.
    scores_cln : pandas.DataFrame
        Cleaned scores.
    f_rm : list
        Factor-ctype combinations to remove.
    df : pandas.DataFrame
        Molecule data.
    crf_res : pandas.DataFrame
        CRF results.
    upper_lim : float
        Upper limit for score ratios.
    min_mean_thresh : float
        Min mean threshold.

    Returns
    -------
    list of matplotlib.figure.Figure
    """
    all_ct = scores_orig['cell_type'].unique()

    # Parse f_rm
    f_ct = [x.split('_', 1)[1] for x in f_rm]
    f_f = [x.split('_')[0] for x in f_rm]

    df = df.copy()
    df['factor'] = crf_res.iloc[:, 0].values

    score_ratios = {}
    for ct in all_ct:
        if ct not in f_ct:
            continue

        ndx = [i for i, x in enumerate(f_ct) if x == ct]
        f_ord = sorted(set(int(f_f[i]) for i in ndx))

        df_sub = df[df['cell_type'] == ct]
        f_ord = [f for f in f_ord if f in df_sub['factor'].values]

        if not f_ord:
            continue

        t2 = pd.crosstab(df_sub['cell'], df_sub['factor'])
        cells_w_f = t2.index[t2[f_ord].sum(axis=1) > 0].tolist()

        scores_orig_ct = scores_orig[scores_orig['cell_type'] == ct].set_index('cell')
        scores_cln_ct = scores_cln[scores_cln['cell_type'] == ct].set_index('cell')

        cells_both = set(scores_orig_ct.index) & set(scores_cln_ct.index)
        scores_orig_ct = scores_orig_ct.loc[list(cells_both)]
        scores_cln_ct = scores_cln_ct.loc[list(cells_both)]

        cells_fin = [c for c in cells_both if c in cells_w_f]
        if not cells_fin:
            continue

        frac_cleaned = len(cells_w_f) / len(t2)
        scores_orig_ct = scores_orig_ct.loc[cells_fin]
        scores_cln_ct = scores_cln_ct.loc[cells_fin]

        tmp = pd.DataFrame({
            'orig': scores_orig_ct['scores'].values,
            'clean': scores_cln_ct['scores'].values,
            'scores': scores_cln_ct['scores'].values / scores_orig_ct['scores'].values,
            'cell_type': ct,
            'cell': scores_cln_ct.index,
            'frac.cleaned': frac_cleaned,
        })

        mol_counts = t2.loc[cells_fin].sum(axis=1)
        mol_counts_rm = t2.loc[cells_fin, f_ord].sum(axis=1)
        tmp['mol.counts'] = mol_counts.values
        tmp['mol.counts.after'] = (mol_counts - mol_counts_rm).values

        if tmp['orig'].mean() > min_mean_thresh:
            score_ratios[ct] = tmp

    if not score_ratios:
        return []

    all_ratios = pd.concat(score_ratios.values())
    all_ratios = all_ratios[~all_ratios['scores'].isna()]
    all_ratios.loc[all_ratios['scores'] > upper_lim, 'scores'] = upper_lim

    # Order cell types by mean score
    msc = all_ratios.groupby('cell_type')['orig'].mean().sort_values()
    ct_ord = msc.index.tolist()

    figs = []

    # Plot 1: Admixture scores boxplot
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    tmp = pd.concat([
        all_ratios[['orig', 'cell_type']].assign(cln_status='unclean').rename(columns={'orig': 'scores'}),
        all_ratios[['clean', 'cell_type']].assign(cln_status='clean').rename(columns={'clean': 'scores'}),
    ])
    tmp['cell_type'] = pd.Categorical(tmp['cell_type'], categories=ct_ord, ordered=True)
    sns.boxplot(data=tmp, y='cell_type', x='scores', hue='cln_status', ax=ax1, fliersize=0)
    ax1.set_title('Admixture scores')
    ax1.set_xlabel('P_admix')
    ax1.set_ylabel('Cell type')
    figs.append(fig1)

    # Plot 2: Score ratios ridgeline
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    all_ratios['cell_type'] = pd.Categorical(all_ratios['cell_type'], categories=ct_ord, ordered=True)
    for i, ct in enumerate(ct_ord):
        ct_data = all_ratios[all_ratios['cell_type'] == ct]['scores']
        if len(ct_data) > 0:
            sns.kdeplot(data=ct_data, ax=ax2, fill=True, alpha=0.3, label=ct)
    ax2.axvline(x=1, color='red', linestyle='dashed')
    ax2.set_title('Admixture score ratios per cell')
    ax2.set_xlabel('(P_admix after) / (P_admix before)')
    ax2.legend(fontsize=6)
    figs.append(fig2)

    # Plot 3: Percent cells modified
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    f_c = all_ratios.drop_duplicates('cell_type')[['cell_type', 'frac.cleaned']]
    f_c['cell_type'] = pd.Categorical(f_c['cell_type'], categories=ct_ord, ordered=True)
    f_c = f_c.sort_values('cell_type')
    ax3.barh(f_c['cell_type'], f_c['frac.cleaned'] * 100, color='darkred', alpha=0.8)
    ax3.set_title('Percent cells modified')
    ax3.set_xlabel('%')
    figs.append(fig3)

    # Plot 4: Molecule counts
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    tmp2 = pd.concat([
        all_ratios[['mol.counts', 'cell_type']].assign(mol_type='unclean').rename(columns={'mol.counts': 'counts'}),
        all_ratios[['mol.counts.after', 'cell_type']].assign(mol_type='clean').rename(columns={'mol.counts.after': 'counts'}),
    ])
    tmp2['cell_type'] = pd.Categorical(tmp2['cell_type'], categories=ct_ord, ordered=True)
    sns.boxplot(data=tmp2, y='cell_type', x='counts', hue='mol_type', ax=ax4, fliersize=0)
    ax4.set_title('Molecule counts in modified cells')
    ax4.set_xlabel('Number of molecules per cell')
    figs.append(fig4)

    return figs


def plot_annot_hmap(rs, ad_sources, final_rm, annot_type, p_bridge=0.1, p_memb=0.01):
    """Plot annotation heatmap with p-values.

    Parameters
    ----------
    rs : pandas.DataFrame
        P-value rowmax matrix (cell_types x factors).
    ad_sources : dict
        Source cell types per factor.
    final_rm : list
        Factor-ctype pairs for removal.
    annot_type : str
        'membrane' or 'bridge'.
    p_bridge : float
        Bridge p-value threshold.
    p_memb : float
        Membrane p-value threshold.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if annot_type == 'membrane':
        p_thresh = p_memb
    else:
        p_thresh = p_bridge

    fig, ax = plt.subplots(figsize=(max(6, rs.shape[1] * 0.8), max(4, rs.shape[0] * 0.4)))

    # Replace NaN with 0 for display
    rs_display = rs.fillna(0)

    cmap = LinearSegmentedColormap.from_list('pval', ['white', 'white', 'red'])
    vmax = max(rs_display.values.max(), -np.log10(p_thresh) * 2)

    im = ax.imshow(rs_display.values, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)

    # Labels
    ax.set_xticks(range(rs.shape[1]))
    ax.set_xticklabels([f'F{i+1}' for i in range(rs.shape[1])], fontsize=8)
    ax.set_yticks(range(rs.shape[0]))
    ax.set_yticklabels(rs.index, fontsize=7)
    ax.set_xlabel('Factor')
    ax.set_ylabel('Cell type')

    # Annotations
    for i in range(rs.shape[0]):
        for j in range(rs.shape[1]):
            f_nm = f'f_{j+1}'
            ct = rs.index[i]
            f_ct = f'{j+1}_{ct}'

            if f_ct in final_rm:
                ax.text(j, i, '\u2605', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='black')
            elif f_nm in ad_sources and str(j+1) in [str(x) for x in range(rs.shape[1])]:
                if ad_sources[f_nm] and ad_sources[f_nm][0] == ct:
                    ax.text(j, i, 'S', ha='center', va='center',
                           fontsize=8, fontweight='bold', color='black')

    plt.colorbar(im, ax=ax, label='-log10(p)')
    plt.tight_layout()
    return fig


def plot_correlation_preservation(cor_cors, bins=30):
    """Plot histogram of gene correlation preservation.

    Parameters
    ----------
    cor_cors : pandas.Series or array-like
        Correlation values.
    bins : int
        Number of histogram bins.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(cor_cors, bins=bins, color='steelblue', edgecolor='white')
    ax.set_title('Preservation of gene correlation')
    ax.set_xlabel('Correlation of gene correlations')
    ax.set_ylabel('Num. genes')
    ax.set_xlim(min(cor_cors), 1.0)
    plt.tight_layout()
    return fig


def plot_expression_comparison(
    cm_orig, cm_clean, markers_native, markers_admix,
    admixture_type, native_type, combine=True
):
    """Plot expression comparison before/after cleaning.

    Parameters
    ----------
    cm_orig : sparse or dense matrix
        Original count matrix (genes x cells).
    cm_clean : sparse or dense matrix
        Cleaned count matrix.
    markers_native : list
        Native marker genes.
    markers_admix : list
        Admixture marker genes.
    admixture_type : str
        Name of admixture cell type.
    native_type : str
        Name of native cell type.
    combine : bool
        If True, combine into one figure.

    Returns
    -------
    matplotlib.figure.Figure or list
    """
    from scipy import sparse as sp

    # Normalize
    if sp.issparse(cm_orig):
        cm_orig = cm_orig.toarray()
    if sp.issparse(cm_clean):
        cm_clean = cm_clean.toarray()

    dat_orig_norm = cm_orig / cm_orig.sum(axis=0, keepdims=True)
    dat_clean_norm = cm_clean / cm_clean.sum(axis=0, keepdims=True)

    # Average expression
    orig_idx_admix = [i for i, g in enumerate(cm_orig.index) if g in markers_admix] if hasattr(cm_orig, 'index') else range(len(markers_admix))
    orig_idx_native = [i for i, g in enumerate(cm_orig.index) if g in markers_native] if hasattr(cm_orig, 'index') else range(len(markers_native))

    admix_before = dat_orig_norm[markers_admix].mean(axis=1) if hasattr(dat_orig_norm, 'loc') else dat_orig_norm[orig_idx_admix].mean(axis=1)
    native_before = dat_orig_norm[markers_native].mean(axis=1) if hasattr(dat_orig_norm, 'loc') else dat_orig_norm[orig_idx_native].mean(axis=1)
    admix_after = dat_clean_norm[markers_admix].mean(axis=1) if hasattr(dat_clean_norm, 'loc') else dat_clean_norm[orig_idx_admix].mean(axis=1)
    native_after = dat_clean_norm[markers_native].mean(axis=1) if hasattr(dat_clean_norm, 'loc') else dat_clean_norm[orig_idx_native].mean(axis=1)

    # Build dataframes
    tmp1 = pd.DataFrame({
        'gene': markers_admix * 2,
        'av_expr': list(admix_before) + list(admix_after),
        'type': ['original'] * len(markers_admix) + ['cleaned'] * len(markers_admix),
    })

    tmp2 = pd.DataFrame({
        'gene': markers_native * 2,
        'av_expr': list(native_before) + list(native_after),
        'type': ['original'] * len(markers_native) + ['cleaned'] * len(markers_native),
    })

    if combine:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        sns.barplot(data=tmp1, x='gene', y='av_expr', hue='type', ax=ax1)
        ax1.set_title(f'{admixture_type} marker expression in {native_type}')
        ax1.set_xlabel(f'{admixture_type} marker')
        ax1.set_ylabel('Average expression')
        ax1.tick_params(axis='x', rotation=45)

        sns.barplot(data=tmp2, x='gene', y='av_expr', hue='type', ax=ax2)
        ax2.set_title(f'{native_type} marker expression in {native_type}')
        ax2.set_xlabel(f'{native_type} marker')
        ax2.set_ylabel('Average expression')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig
    else:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=tmp1, x='gene', y='av_expr', hue='type', ax=ax1)
        ax1.set_title(f'{admixture_type} marker expression in {native_type}')

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(data=tmp2, x='gene', y='av_expr', hue='type', ax=ax2)
        ax2.set_title(f'{native_type} marker expression in {native_type}')

        return [fig1, fig2]
