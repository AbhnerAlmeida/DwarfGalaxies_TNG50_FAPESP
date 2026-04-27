import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DIRECT THRESHOLD-ROBUSTNESS TEST FOR THE REFEREE QUESTION
# ============================================================
# This script tests whether the *science signals* of the paper
# are stable when the DM-fraction threshold is changed.
#
# Philosophy:
#   - do NOT calibrate against an internal "truth label"
#   - instead, re-split satellites into DM-poor / DM-rich at
#     t = 0.5, 0.7, 0.8 (or any list you want)
#   - for each threshold, recompute the actual physical signals
#     used in the paper: entry epoch, host mass, orbital depth,
#     pre-processing fraction, etc.
#   - track BOTH effect size and sample size.
#
# Dependencies assumed in your environment:
#   - numpy, pandas, matplotlib
#   - TNG.extractPopulation(...)
#
# Optional:
#   - if you already have MATH.TestPermutation and want to reuse it,
#     keep it; here we implement our own permutation p-value so the
#     script is self-contained.
# ============================================================

# -----------------------------
# USER CONFIGURATION
# -----------------------------
DFNAME = 'PaperII'
NAMEARG = 'Name'

# Thresholds for the referee test.
THRESHOLDS = [0.5, 0.7, 0.8]
# Optional dense sweep for a figure.
DENSE_THRESHOLDS = np.round(np.arange(0.45, 0.851, 0.01), 2)

# Population naming convention already used in your project.
# Edit here if your names differ.
POP_PREFIX = {
    'Normals': 'Normal',
    'CompactsMB': 'MBC',
    'CompactsSB': 'SBC',
}

# Existing threshold-specific populations already defined in your catalog.
# This lets you answer the referee question immediately for 0.5 / 0.7 / 0.8.
THRESHOLD_SUFFIX = {
    0.7: ('DMpoor_Satellite', 'DMrich_Satellite'),
    0.8: ('DMpoor_08_Referee_Satellite', 'DMrich_08_Referee_Satellite'),
    0.5: ('DMpoor_05_Referee_Satellite', 'DMrich_05_Referee_Satellite'),
}

# Optional *parent* satellite populations if you want to do a dense sweep
# directly from the full satellite sample using an fDM column.
# If a parent population is not available, the dense sweep falls back to the
# union of the 0.7 DM-poor + DM-rich samples.
PARENT_SATELLITE_POP = {
    'Normals': None,      # e.g. 'Normal_Satellite'
    'CompactsMB': None,   # e.g. 'MBC_Satellite'
    'CompactsSB': None,   # e.g. 'SBC_Satellite'
}

# Optional central populations for the "0.8 is close to centrals" check.
# Set to None if unavailable.
CENTRAL_POP = {
    'Normals': None,      # e.g. 'Normal_Central'
    'CompactsMB': None,   # e.g. 'MBC_Central'
    'CompactsSB': None,   # e.g. 'SBC_Central'
}

# Column names in your dataframe/catalog.
COLS = {
    'fdm': 'DMFrac_99',
    'zentry': 'z_At_FirstEntry',      # or z_At_FinalEntry if that is what you use
    'm200': 'M200Mean',               # time-averaged host mass after entry
    'rmean': 'rOverR200Mean_New',     # time-averaged orbital depth
    'rmin': 'rOverR200Min',           # minimum pericentre in R/R200
    'preproc': 'PreProcessedFlag',    # 0/1 or bool; edit if needed
    'id': 'SubfindID_99',                       # e.g. 'SubfindID_99' if available
}

OUTDIR = './threshold_science_signal_scan'

# -----------------------------
# SCIENCE SIGNALS TO TEST
# -----------------------------
# These are the most relevant signals from the paper to decide whether the
# DM-poor/DM-rich split is scientifically stable when the threshold changes.
#
# expected_sign:
#   +1 means DM-poor should be larger than DM-rich
#   -1 means DM-poor should be smaller than DM-rich
#
# priority:
#   1 = most important for the referee answer
#   2 = useful complementary check
SIGNALS = {
    'zentry': {
        'col': COLS['zentry'],
        'label': r'$\Delta z_{\rm entry}$',
        'summary': 'median',
        'expected_sign': +1,
        'priority': 1,
    },
    'm200': {
        'col': COLS['m200'],
        'label': r'$\Delta \log M_{200}$',
        'summary': 'median',
        'expected_sign': +1,
        'priority': 1,
    },
    'rmean': {
        'col': COLS['rmean'],
        'label': r'$\Delta \langle R/R_{200}\rangle$',
        'summary': 'median',
        'expected_sign': -1,
        'priority': 1,
    },
    'rmin': {
        'col': COLS['rmin'],
        'label': r'$\Delta (R/R_{200})_{\min}$',
        'summary': 'median',
        'expected_sign': -1,
        'priority': 1,
    },
    'preproc': {
        'col': COLS['preproc'],
        'label': r'$\Delta f_{\rm preproc}$',
        'summary': 'mean',
        'expected_sign': +1,
        'priority': 1,
    },
}

# Optional secondary signals if you have them and want to show that the main
# narrative also remains qualitatively stable.
SECONDARY_SIGNALS = {
    # 'gasret': {
    #     'col': 'Mgas_z0_over_Mgas_max',
    #     'label': r'$\Delta (M_{\rm gas}(z=0)/M_{\rm gas,max})$',
    #     'summary': 'median',
    #     'expected_sign': -1,
    #     'priority': 2,
    # },
}

# -----------------------------
# HELPERS
# -----------------------------
def _ensure_dir(path):
    import os
    os.makedirs(path, exist_ok=True)


def _nan_clean(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def bootstrap_summary(values, summary='median', n_boot=5000, random_state=42):
    values = _nan_clean(values)
    n = len(values)
    if n == 0:
        return {
            'N': 0,
            'value': np.nan,
            'p16': np.nan,
            'p84': np.nan,
        }
    if summary == 'median':
        f = np.median
    elif summary == 'mean':
        f = np.mean
    else:
        raise ValueError(f'Unknown summary={summary}')

    rng = np.random.default_rng(random_state)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boots[i] = f(sample)

    return {
        'N': n,
        'value': float(f(values)),
        'p16': float(np.percentile(boots, 16)),
        'p84': float(np.percentile(boots, 84)),
    }


def permutation_pvalue(x, y, statistic='median_diff', n_perm=5000, random_state=42):
    """
    Two-sided permutation p-value.
    statistic='median_diff' or 'mean_diff'
    Returns observed statistic as x - y.
    """
    x = _nan_clean(x)
    y = _nan_clean(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan

    if statistic == 'median_diff':
        statfunc = lambda a, b: np.median(a) - np.median(b)
    elif statistic == 'mean_diff':
        statfunc = lambda a, b: np.mean(a) - np.mean(b)
    else:
        raise ValueError(f'Unknown statistic={statistic}')

    obs = statfunc(x, y)
    pooled = np.concatenate([x, y])
    nx = len(x)
    rng = np.random.default_rng(random_state)
    perm_stats = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        rng.shuffle(pooled)
        xp = pooled[:nx]
        yp = pooled[nx:]
        perm_stats[i] = statfunc(xp, yp)

    p = np.mean(np.abs(perm_stats) >= np.abs(obs))
    return float(obs), float(p)


def fisher_or_permutation_binary(x, y, n_perm=5000, random_state=42):
    """
    Binary or fractional data: compare means as proportions.
    Returns observed difference x - y and p-value.
    """
    return permutation_pvalue(x, y, statistic='mean_diff', n_perm=n_perm, random_state=random_state)


def extract_population_df(population, cols=None, dfName=DFNAME, Name=NAMEARG):
    df = TNG.extractPopulation(population, dfName=dfName, Name=Name).copy()
    if cols is not None:
        keep = [c for c in cols if c in df.columns]
        df = df[keep].copy()
    return df


def concat_and_union(dfs):
    if len(dfs) == 0:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    idcol = COLS.get('id')
    if idcol is not None and idcol in out.columns:
        out = out.drop_duplicates(subset=idcol, keep='first').reset_index(drop=True)
    return out.reset_index(drop=True)


def get_threshold_population_names(sample_class, threshold):
    if threshold not in THRESHOLD_SUFFIX:
        raise ValueError(f'No named populations configured for threshold={threshold}')
    pref = POP_PREFIX[sample_class]
    poor_suffix, rich_suffix = THRESHOLD_SUFFIX[threshold]
    return f'{pref}_{poor_suffix}', f'{pref}_{rich_suffix}'


def load_named_split(sample_class, threshold, cols=None):
    poor_pop, rich_pop = get_threshold_population_names(sample_class, threshold)
    poor = extract_population_df(poor_pop, cols=cols)
    rich = extract_population_df(rich_pop, cols=cols)
    poor['__pred__'] = 'poor'
    rich['__pred__'] = 'rich'
    return poor, rich, poor_pop, rich_pop


def load_parent_satellites(sample_class, cols=None):
    parent = PARENT_SATELLITE_POP.get(sample_class)
    if parent is not None:
        df = extract_population_df(parent, cols=cols)
        return df

    # fallback: union of the 0.7 named split
    poor, rich, _, _ = load_named_split(sample_class, 0.7, cols=cols)
    return concat_and_union([poor.drop(columns='__pred__'), rich.drop(columns='__pred__')])


def split_parent_by_threshold(df_parent, threshold):
    fdmcol = COLS['fdm']
    out = df_parent.copy()
    out = out[np.isfinite(out[fdmcol].values)].copy()
    poor = out[out[fdmcol] < threshold].copy()
    rich = out[out[fdmcol] >= threshold].copy()
    poor['__pred__'] = 'poor'
    rich['__pred__'] = 'rich'
    return poor, rich


def extract_signal_values(df, colname, binary=False):
    if colname not in df.columns:
        return np.array([])
    vals = np.asarray(df[colname].values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if binary:
        vals = (vals > 0).astype(float)
    return vals


def evaluate_signal(df_poor, df_rich, signal_cfg, n_perm=5000, n_boot=5000):
    col = signal_cfg['col']
    summary = signal_cfg['summary']
    binary = (summary == 'mean')

    x = extract_signal_values(df_poor, col, binary=binary)
    y = extract_signal_values(df_rich, col, binary=binary)

    sx = bootstrap_summary(x, summary=summary, n_boot=n_boot)
    sy = bootstrap_summary(y, summary=summary, n_boot=n_boot)

    if summary == 'median':
        delta, p = permutation_pvalue(x, y, statistic='median_diff', n_perm=n_perm)
    else:
        delta, p = fisher_or_permutation_binary(x, y, n_perm=n_perm)

    expected_sign = signal_cfg['expected_sign']
    correct_direction = np.sign(delta) == np.sign(expected_sign) if np.isfinite(delta) and delta != 0 else False

    return {
        'N_poor': sx['N'],
        'N_rich': sy['N'],
        'poor_value': sx['value'],
        'poor_p16': sx['p16'],
        'poor_p84': sx['p84'],
        'rich_value': sy['value'],
        'rich_p16': sy['p16'],
        'rich_p84': sy['p84'],
        'delta_poor_minus_rich': delta,
        'p_value': p,
        'correct_direction': bool(correct_direction),
        'significant_0p05': bool(np.isfinite(p) and p < 0.05),
        'science_success': bool(np.isfinite(p) and p < 0.05 and correct_direction),
    }


def evaluate_threshold_named_populations(sample_classes=('Normals', 'CompactsMB', 'CompactsSB'),
                                         thresholds=THRESHOLDS,
                                         signals=SIGNALS,
                                         n_perm=5000,
                                         n_boot=5000):
    rows = []
    cols_needed = list({COLS['fdm']} | {v['col'] for v in signals.values()})
    if COLS.get('id') is not None:
        cols_needed.append(COLS['id'])

    for sc in sample_classes:
        for t in thresholds:
            df_poor, df_rich, poor_pop, rich_pop = load_named_split(sc, t, cols=cols_needed)

            for sname, scfg in signals.items():
                res = evaluate_signal(df_poor, df_rich, scfg, n_perm=n_perm, n_boot=n_boot)
                row = {
                    'sample_class': sc,
                    'threshold': t,
                    'signal': sname,
                    'signal_label': scfg['label'],
                    'expected_sign': scfg['expected_sign'],
                    'poor_population': poor_pop,
                    'rich_population': rich_pop,
                    'Npoor_total': len(df_poor),
                    'Nrich_total': len(df_rich),
                }
                row.update(res)
                rows.append(row)

    return pd.DataFrame(rows)


def evaluate_threshold_dense_scan(sample_classes=('Normals', 'CompactsMB', 'CompactsSB'),
                                  thresholds=DENSE_THRESHOLDS,
                                  signals=SIGNALS,
                                  n_perm=3000,
                                  n_boot=3000):
    rows = []
    cols_needed = list({COLS['fdm']} | {v['col'] for v in signals.values()})
    if COLS.get('id') is not None:
        cols_needed.append(COLS['id'])

    for sc in sample_classes:
        parent = load_parent_satellites(sc, cols=cols_needed)
        for t in thresholds:
            df_poor, df_rich = split_parent_by_threshold(parent, t)
            for sname, scfg in signals.items():
                res = evaluate_signal(df_poor, df_rich, scfg, n_perm=n_perm, n_boot=n_boot)
                row = {
                    'sample_class': sc,
                    'threshold': float(t),
                    'signal': sname,
                    'signal_label': scfg['label'],
                    'expected_sign': scfg['expected_sign'],
                    'Npoor_total': len(df_poor),
                    'Nrich_total': len(df_rich),
                }
                row.update(res)
                rows.append(row)
    return pd.DataFrame(rows)


def build_threshold_score(df_results, signals=SIGNALS):
    """
    Direct score for the referee answer:
      - counts how many *paper signals* are recovered with the expected sign
        and p < 0.05 at each threshold
      - multiplies by a sample-size penalty so tiny DM-poor samples do not win
    """
    rows = []
    signals_primary = [k for k, v in signals.items() if v['priority'] == 1]

    for (sc, t), g in df_results.groupby(['sample_class', 'threshold']):
        g = g.copy()
        n_success = int(g['science_success'].sum())
        n_total = len(g)
        frac_success = n_success / n_total if n_total > 0 else np.nan

        # sample-size penalty based on the smallest useful side of the split
        npoor = int(np.nanmedian(g['Npoor_total'])) if len(g) else 0
        nrich = int(np.nanmedian(g['Nrich_total'])) if len(g) else 0
        nmin = min(npoor, nrich)

        # gentle penalty: 0 below 5, then rises to ~1 by 20
        size_penalty = np.clip((nmin - 5) / 15.0, 0, 1)

        score = frac_success * size_penalty if np.isfinite(frac_success) else np.nan
        rows.append({
            'sample_class': sc,
            'threshold': t,
            'n_signals_recovered': n_success,
            'n_signals_total': n_total,
            'frac_signals_recovered': frac_success,
            'Npoor': npoor,
            'Nrich': nrich,
            'Nmin': nmin,
            'size_penalty': size_penalty,
            'science_score': score,
        })
    return pd.DataFrame(rows)


def plot_signal_scan(df_results, score_df=None, sample_classes=('Normals','CompactsMB','CompactsSB'),
                     signal_order=('zentry','m200','rmean','rmin','preproc'),
                     threshold_ref=0.7, threshold_marks=(0.5,0.8),
                     figsize=(13, 12)):
    """
    Main referee figure:
      left column  = effect size (delta poor-rich) for each signal vs threshold
      right column = Npoor and science_score vs threshold
    One row per sample_class.
    """
    
    # -----------------------------
    # Make axes grid
    # -----------------------------
    plt.rcParams.update({"figure.figsize": (3. * 2, 6. * len(sample_classes))})
    fig = plt.figure()
    gs = fig.add_gridspec( len(sample_classes), 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex="col", sharey="row")

    if len(sample_classes) == 1:
        axs = np.array([axs])

    # fixed colors for signals
    colors = {
        'zentry': 'tab:blue',
        'm200': 'tab:green',
        'rmean': 'tab:orange',
        'rmin': 'tab:red',
    }

    for i, sc in enumerate(sample_classes):
        gsc = df_results[df_results['sample_class'] == sc]

        ax = axs[i, 0]
        for s in signal_order:
            gs = gsc[gsc['signal'] == s].sort_values('threshold')
            if len(gs) == 0:
                continue
            ax.plot(gs['threshold'], gs['delta_poor_minus_rich'], lw=2.0, color=colors.get(s, None), label=SIGNALS[s]['label'])
            # mark significant points
            m = gs['science_success'].values.astype(bool)
            ax.scatter(gs['threshold'][m], gs['delta_poor_minus_rich'][m], s=20, color=colors.get(s, None), zorder=5)

        ax.axhline(0, color='k', lw=0.8, alpha=0.4)
        ax.axvline(threshold_ref, color='dodgerblue', lw=2.0)
        for t in threshold_marks:
            ax.axvline(t, color='deepskyblue', ls='--', lw=1.2)
        ax.set_ylabel(sc)
        if i == 0:
            ax.set_title('Effect size: DM-poor minus DM-rich')
            ax.legend(frameon=False, fontsize=9, loc='best')

        ax2 = axs[i, 1]
        # representative Npoor from any one signal row (same split for all signals)
        rep = gsc[gsc['signal'] == signal_order[0]].sort_values('threshold')
        if len(rep):
            ax2.plot(rep['threshold'], rep['Npoor_total'], lw=2.0, color='tab:blue', label=r'$N_{\rm poor}$')
            ax2.plot(rep['threshold'], rep['Nrich_total'], lw=2.0, color='tab:green', label=r'$N_{\rm rich}$')
        if score_df is not None:
            srow = score_df[score_df['sample_class'] == sc].sort_values('threshold')
            ax2b = ax2.twinx()
            ax2b.plot(srow['threshold'], srow['science_score'], lw=2.0, color='tab:red', ls='--', label='science score')
            ax2b.set_ylim(-0.02, 1.02)
            if i == 0:
                ax2b.set_ylabel('Science score')
        ax2.axvline(threshold_ref, color='dodgerblue', lw=2.0)
        for t in threshold_marks:
            ax2.axvline(t, color='deepskyblue', ls='--', lw=1.2)
        if i == 0:
            ax2.set_title('Sample size and score')
            ax2.legend(frameon=False, fontsize=9, loc='upper left')

    axs[-1, 0].set_xlabel(r'$f_{\rm DM,z=0}$ threshold')
    axs[-1, 1].set_xlabel(r'$f_{\rm DM,z=0}$ threshold')
    fig.tight_layout()
    return fig, axs


def plot_central_overlap(sample_classes=('Normals','CompactsMB','CompactsSB'),
                         threshold_lines=(0.5,0.7,0.8), figsize=(10, 3.5)):
    """
    Optional figure answering the referee motivation for 0.8:
    show the fDM(z=0) CDFs of satellites and centrals.
    Requires CENTRAL_POP entries.
    """
    fig, axs = plt.subplots(1, len(sample_classes), figsize=figsize, sharex=True, sharey=True)
    if len(sample_classes) == 1:
        axs = [axs]

    for ax, sc in zip(axs, sample_classes):
        sat_parent = load_parent_satellites(sc, cols=[COLS['fdm']])
        x_sat = np.sort(_nan_clean(sat_parent[COLS['fdm']].values))
        y_sat = np.arange(1, len(x_sat)+1) / len(x_sat) if len(x_sat) else np.array([])
        if len(x_sat):
            ax.step(x_sat, y_sat, where='post', lw=2, label='Satellites')

        cpop = CENTRAL_POP.get(sc)
        if cpop is not None:
            cent = extract_population_df(cpop, cols=[COLS['fdm']])
            x_c = np.sort(_nan_clean(cent[COLS['fdm']].values))
            y_c = np.arange(1, len(x_c)+1) / len(x_c) if len(x_c) else np.array([])
            if len(x_c):
                ax.step(x_c, y_c, where='post', lw=2, label='Centrals')

        for t in threshold_lines:
            ax.axvline(t, ls='--' if t != 0.7 else '-', lw=2 if t == 0.7 else 1.2, color='dodgerblue' if t == 0.7 else 'deepskyblue')
        ax.set_title(sc)
        ax.set_xlabel(r'$f_{\rm DM,z=0}$')
        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', top=True, right=True)
    axs[0].set_ylabel('CDF')
    axs[0].legend(frameon=False)
    fig.tight_layout()
    return fig, axs


def summarize_referee_table(df_results, score_df=None):
    """
    Compact summary table with the quantities you can cite in the rebuttal.
    """
    rows = []
    for (sc, t), g in df_results.groupby(['sample_class', 'threshold']):
        row = {'sample_class': sc, 'threshold': t}
        for _, r in g.iterrows():
            s = r['signal']
            row[f'{s}_delta'] = r['delta_poor_minus_rich']
            row[f'{s}_p'] = r['p_value']
            row[f'{s}_success'] = int(r['science_success'])
            row[f'{s}_poor'] = r['poor_value']
            row[f'{s}_rich'] = r['rich_value']
        row['Npoor'] = int(np.nanmedian(g['Npoor_total']))
        row['Nrich'] = int(np.nanmedian(g['Nrich_total']))
        if score_df is not None:
            ss = score_df[(score_df['sample_class'] == sc) & (score_df['threshold'] == t)]
            if len(ss):
                row['science_score'] = float(ss['science_score'].iloc[0])
                row['frac_signals_recovered'] = float(ss['frac_signals_recovered'].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['sample_class', 'threshold']).reset_index(drop=True)

#%%

# -----------------------------
# MAIN EXAMPLE
# -----------------------------
if __name__ == '__main__':
    _ensure_dir(OUTDIR)

    # A) direct referee comparison for the exact thresholds requested
    results_ref = evaluate_threshold_named_populations(
        sample_classes=('Normals', 'CompactsMB', 'CompactsSB'),
        thresholds=THRESHOLDS,
        signals=SIGNALS,
        n_perm=5000,
        n_boot=5000,
    )
    score_ref = build_threshold_score(results_ref)
    tab_ref = summarize_referee_table(results_ref, score_df=score_ref)

    results_ref.to_csv(f'{OUTDIR}/results_referee_thresholds.csv', index=False)
    score_ref.to_csv(f'{OUTDIR}/science_score_referee_thresholds.csv', index=False)
    tab_ref.to_csv(f'{OUTDIR}/summary_referee_thresholds.csv', index=False)

    print('\n=== Referee threshold results (0.5, 0.7, 0.8) ===')
    print(tab_ref.to_string(index=False))

    # B) dense sweep: optional, more visual and more direct to identify the knee
    results_dense = evaluate_threshold_dense_scan(
        sample_classes=('Normals', 'CompactsMB', 'CompactsSB'),
        thresholds=DENSE_THRESHOLDS,
        signals=SIGNALS,
        n_perm=3000,
        n_boot=3000,
    )
    score_dense = build_threshold_score(results_dense)
    tab_dense = summarize_referee_table(results_dense, score_df=score_dense)

    results_dense.to_csv(f'{OUTDIR}/results_dense_threshold_scan.csv', index=False)
    score_dense.to_csv(f'{OUTDIR}/science_score_dense_threshold_scan.csv', index=False)
    tab_dense.to_csv(f'{OUTDIR}/summary_dense_threshold_scan.csv', index=False)

    fig, axs = plot_signal_scan(results_dense, score_df=score_dense)
    fig.savefig(f'{OUTDIR}/science_signal_vs_threshold.png', dpi=250, bbox_inches='tight')
    plt.close(fig)

    # Optional central overlap figure if CENTRAL_POP is configured
    if any(v is not None for v in CENTRAL_POP.values()):
        fig, axs = plot_central_overlap()
        fig.savefig(f'{OUTDIR}/fdm_cdf_satellites_vs_centrals.png', dpi=250, bbox_inches='tight')
        plt.close(fig)

    print(f'\nDone. Outputs written to: {OUTDIR}')
    
    
#%%



def plot_signal_scan(df_results, score_df=None, sample_classes=('Normals','CompactsMB','CompactsSB'),
                     signal_order=('zentry','m200','rmean','rmin'),
                     threshold_ref=0.7, threshold_marks=(0.5,0.8),
                     figsize=(13, 12)):
    """
    Main referee figure:
      left column  = effect size (delta poor-rich) for each signal vs threshold
      right column = Npoor and science_score vs threshold
    One row per sample_class.
    """
    
    # -----------------------------
    # Make axes grid
    # -----------------------------
    plt.rcParams.update({"figure.figsize": (6. * 1, 3. * len(sample_classes))})
    fig = plt.figure()
    gs = fig.add_gridspec( len(sample_classes), 1, hspace=0, wspace=0)
    axs = gs.subplots(sharex="col", sharey="row")

    if len(sample_classes) == 1:
        axs = np.array([axs])

    # fixed colors for signals
    colors = {
        'zentry': 'tab:blue',
        'm200': 'tab:green',
        'rmean': 'tab:orange',
        'rmin': 'tab:red',
    }

    for i, sc in enumerate(sample_classes):
        gsc = df_results[df_results['sample_class'] == sc]

        ax = axs[i]
        for s in signal_order:
            gs = gsc[gsc['signal'] == s].sort_values('threshold')
            if len(gs) == 0:
                continue
            ax.plot(gs['threshold'], gs['delta_poor_minus_rich'], lw=2.0, color=colors.get(s, None), label=SIGNALS[s]['label'])
            # mark significant points
            #m = gs['science_success'].values.astype(bool)
            #ax.scatter(gs['threshold'][m], gs['delta_poor_minus_rich'][m], s=20, markersize = 0,  color=colors.get(s, None), zorder=5)

        ax.axhline(0, color='k', lw=0.8, alpha=0.4)
        ax.axvline(threshold_ref, color='dodgerblue', lw=2.0)
        for t in threshold_marks:
            ax.axvline(t, color='deepskyblue', ls='--', lw=1.2)
        ax.set_ylabel(sc)
        if i == 0:
            ax.set_title('Effect size: DM-poor minus DM-rich')
            ax.legend(frameon=False, fontsize=9, loc='best')


    axs[-1].set_xlabel(r'$f_{\rm DM,z=0}$ threshold')
    
    fig.tight_layout()
    return fig, axs


#%%

import numpy as np
import matplotlib.pyplot as plt

def get_population_values(population, key='DMFrac_99', dfName='PaperII', Name='Name'):
    Sample = TNG.extractPopulation(population, dfName=dfName, Name=Name)
    values = np.asarray(Sample[key].values, dtype=float)
    values = values[np.isfinite(values)]
    return values

def ecdf(values):
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

#%%

populations = {
    "Centrals": "Central",              # ajustar se necessário
    "Normals": "Normal_Satellite",
    "CompactsMB": "MBC_Satellite",
    "CompactsSB": "SBC_Satellite",
}

#%%

def plot_fdm_hist_cdf(
    populations,
    key='DMFrac_99',
    dfName='PaperII',
    Name='Name',
    bins=np.linspace(0.0, 1.0, 41),
    thresholds=(0.5, 0.7, 0.8),
    colors=None,
    cNum=4.8,
    lNum=3.6
):
    if colors is None:
        colors = {
            "Centrals": "black",
            "Normals": "tab:orange",
            "CompactsMB": "tab:blue",
            "CompactsSB": "tab:green",
        }

    data = {}
    for label, pop in populations.items():
        vals = get_population_values(pop, key=key, dfName=dfName, Name=Name)
        data[label] = vals
        print(f"{label:12s}  N = {len(vals):4d}   median = {np.median(vals):.3f}")

    # -----------------------------
    # Make axes grid
    # -----------------------------
    plt.rcParams.update({"figure.figsize": (cNum * 2, lNum)})
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True)

    # -----------------------------
    # Left: histogram
    # -----------------------------
    ax = axs[0]
    for label, vals in data.items():
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype='step',
            lw=2.2,
            color=colors[label],
            label=label
        )

    for t in thresholds:
        if t == 0.7:
            ax.axvline(t, color='dodgerblue', lw=2.2)
        else:
            ax.axvline(t, color='deepskyblue', lw=1.2, ls='--')

    ax.set_ylabel("Density")
    ax.set_xlabel(r"$f_{\rm DM,z=0}$")
    ax.set_title("Normalized histogram")
    ax.legend(frameon=False, loc='upper left')
    ax.minorticks_on()
    ax.tick_params(direction='in', which='both', top=True, right=True)

    # -----------------------------
    # Right: CDF
    # -----------------------------
    ax = axs[1]
    for label, vals in data.items():
        x, y = ecdf(vals)
        ax.plot(
            x, y,
            lw=2.2,
            color=colors[label],
            label=label
        )

    for t in thresholds:
        if t == 0.7:
            ax.axvline(t, color='dodgerblue', lw=2.2)
        else:
            ax.axvline(t, color='deepskyblue', lw=1.2, ls='--')

    ax.set_ylabel("CDF")
    ax.set_xlabel(r"$f_{\rm DM,z=0}$")
    ax.set_title("Empirical CDF")
    ax.minorticks_on()
    ax.tick_params(direction='in', which='both', top=True, right=True)

    fig.tight_layout()
    return fig, axs, data

#%%
fig, axs, data = plot_fdm_hist_cdf(populations)
plt.show()