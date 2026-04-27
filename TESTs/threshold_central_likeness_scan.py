import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CENTRAL-LIKENESS TEST AS A FUNCTION OF fDM_z=0 THRESHOLD
# ============================================================
# Goal
# -----
# For each threshold t, split satellites into DM-poor and DM-rich and test
# whether the DM-rich subset is closer to the corresponding central population
# than the DM-poor subset is.
#
# This is useful if your physical interpretation is that DM-rich compact
# satellites behave more like centrals, while DM-poor satellites represent the
# strongly processed channel.
#
# Main metric
# -----------
# For each variable X and each threshold t we compute:
#   D_rich(t) = W1[ X_DM-rich(t), X_central ] / scale_central
#   D_poor(t) = W1[ X_DM-poor(t), X_central ] / scale_central
#   DeltaD(t) = D_poor(t) - D_rich(t)
#
# where W1 is the 1D Wasserstein distance.
# Interpretation:
#   DeltaD > 0  --> DM-rich is closer to centrals than DM-poor
#   DeltaD = 0  --> no separation in the central-like sense
#   DeltaD < 0  --> unexpected / threshold not separating the intended channel
#
# Optional outputs:
#   - CSV table with distances for each variable and threshold
#   - aggregate central-likeness score vs threshold
#   - figure with DeltaD curves + sample size and score
#
# Assumes in your environment:
#   - numpy, pandas, matplotlib
#   - TNG.extractPopulation(...)
#
# Optional but not required:
#   - scipy (NOT required; we implement Wasserstein directly)
# ============================================================

# -----------------------------
# USER CONFIGURATION
# -----------------------------
DFNAME = 'PaperII'
NAMEARG = 'Name'
OUTDIR = './threshold_central_likeness_scan'

THRESHOLDS = [0.5, 0.7, 0.8]
DENSE_THRESHOLDS = np.round(np.arange(0.45, 0.851, 0.01), 2)

POP_PREFIX = {
    'Normals': 'Normal',
    'CompactsMB': 'MBC',
    'CompactsSB': 'SBC',
}

# Existing named populations already available in your catalog.
THRESHOLD_SUFFIX = {
    0.7: ('DMpoor_Satellite', 'DMrich_Satellite'),
    0.8: ('DMpoor_08_Referee_Satellite', 'DMrich_08_Referee_Satellite'),
    0.5: ('DMpoor_05_Referee_Satellite', 'DMrich_05_Referee_Satellite'),
}

# Optional parent satellite populations if you want a dense threshold sweep
# directly from the full satellite sample. If not available, the dense sweep
# falls back to the union of the 0.7 DM-poor + DM-rich populations.
PARENT_SATELLITE_POP = {
    'Normals': None,      # e.g. 'Normal_Satellite'
    'CompactsMB': None,   # e.g. 'MBC_Satellite'
    'CompactsSB': None,   # e.g. 'SBC_Satellite'
}

# Central populations used as the reference for “central-likeness”.
CENTRAL_POP = {
    'Normals': None,      # e.g. 'Normal_Central'
    'CompactsMB': None,   # e.g. 'MBC_Central'
    'CompactsSB': None,   # e.g. 'SBC_Central'
}

# Column names in your catalog.
COLS = {
    'fdm': 'DMFrac_99',
    'id': 'SubfindID_99',  # e.g. 'SubfindID_99'
}

# Variables to use for the central-likeness test.
# Replace the column names below with the ones that exist in your catalog.
# Recommended choices are quantities where your physical picture says that
# DM-rich satellites should resemble centrals more than DM-poor satellites do.
CENTRAL_LIKE_VARS = {
    # Example placeholders — edit these to match your dataframe:
    # 'inner_ssfr_mean_zlt5': {
    #     'col': 'sSFR_In_TrueRhpkpc_Mean_zlt5',
    #     'label': r'inner sSFR (z<5)',
    #     'weight': 1.0,
    # },
    # 'exsitu_frac': {
    #     'col': 'ExSituFrac_Total',
    #     'label': r'ex-situ fraction',
    #     'weight': 1.0,
    # },
    # 'size_mean_zlt5': {
    #     'col': 'logRhalf_Mean_zlt5',
    #     'label': r'mean $\log r_{1/2}$ (z<5)',
    #     'weight': 0.6,
    # },
}

# If True, variables with no valid central scatter are normalized with an IQR
# fallback. If False, they are skipped.
ALLOW_SCALE_FALLBACK = True

# Minimum number of objects required in DM-poor, DM-rich and central samples
# to evaluate a threshold.
MIN_N = 5

# -----------------------------
# HELPERS
# -----------------------------

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _nan_clean(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def wasserstein_1d(x, y):
    """
    1D Wasserstein distance (Earth mover's distance) implemented without scipy.
    Uses the integral of |F_x - F_y| over the union grid of support points.
    """
    x = np.sort(_nan_clean(x))
    y = np.sort(_nan_clean(y))
    if len(x) == 0 or len(y) == 0:
        return np.nan

    z = np.sort(np.unique(np.concatenate([x, y])))
    if len(z) == 1:
        return 0.0

    Fx = np.searchsorted(x, z, side='right') / len(x)
    Fy = np.searchsorted(y, z, side='right') / len(y)
    dz = np.diff(z)
    return float(np.sum(np.abs(Fx[:-1] - Fy[:-1]) * dz))


def robust_scale(x):
    """
    Robust scale of a 1D array, used to normalize Wasserstein distances.
    Preference: std -> IQR -> MAD -> range.
    """
    x = _nan_clean(x)
    if len(x) < 2:
        return np.nan

    s = np.nanstd(x)
    if np.isfinite(s) and s > 0:
        return float(s)

    q75, q25 = np.nanpercentile(x, [75, 25])
    iqr = q75 - q25
    if np.isfinite(iqr) and iqr > 0:
        return float(iqr)

    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)

    xr = np.nanmax(x) - np.nanmin(x)
    if np.isfinite(xr) and xr > 0:
        return float(xr)

    return np.nan


def bootstrap_delta_distance(poor, rich, central, n_boot=2000, random_state=42):
    """
    Bootstrap CI for DeltaD = Dpoor - Drich.
    Returns median, p16, p84 of the bootstrap DeltaD distribution.
    """
    poor = _nan_clean(poor)
    rich = _nan_clean(rich)
    central = _nan_clean(central)
    if min(len(poor), len(rich), len(central)) < 2:
        return {'value': np.nan, 'p16': np.nan, 'p84': np.nan}

    rng = np.random.default_rng(random_state)
    out = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        bp = rng.choice(poor, size=len(poor), replace=True)
        br = rng.choice(rich, size=len(rich), replace=True)
        bc = rng.choice(central, size=len(central), replace=True)
        scale = robust_scale(bc)
        if not np.isfinite(scale) or scale <= 0:
            out[i] = np.nan
            continue
        dpoor = wasserstein_1d(bp, bc) / scale
        drich = wasserstein_1d(br, bc) / scale
        out[i] = dpoor - drich

    out = out[np.isfinite(out)]
    if len(out) == 0:
        return {'value': np.nan, 'p16': np.nan, 'p84': np.nan}
    return {
        'value': float(np.nanmedian(out)),
        'p16': float(np.nanpercentile(out, 16)),
        'p84': float(np.nanpercentile(out, 84)),
    }


def extract_population_df(population, cols=None, dfName=DFNAME, Name=NAMEARG):
    df = TNG.extractPopulation(population, dfName=dfName, Name=Name).copy()
    if cols is not None:
        keep = [c for c in cols if c in df.columns]
        df = df[keep].copy()
    return df


def concat_and_union(dfs, idcol=None):
    if len(dfs) == 0:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    if idcol is not None and idcol in out.columns:
        nunique = out[idcol].nunique(dropna=False)
        if nunique > 1:
            out = out.drop_duplicates(subset=idcol, keep='first').reset_index(drop=True)
    return out.reset_index(drop=True)


def get_threshold_population_names(sample_class, threshold):
    if threshold not in THRESHOLD_SUFFIX:
        raise ValueError(f'No named populations configured for threshold={threshold}')
    pref = POP_PREFIX[sample_class]
    poor_suffix, rich_suffix = THRESHOLD_SUFFIX[threshold]
    return f'{pref}_{poor_suffix}', f'{pref}_{rich_suffix}'


def required_columns():
    cols = [COLS['fdm']]
    if COLS.get('id') is not None:
        cols.append(COLS['id'])
    for cfg in CENTRAL_LIKE_VARS.values():
        cols.append(cfg['col'])
    return list(dict.fromkeys(cols))


def load_parent_satellite_sample(sample_class):
    """
    Preferred: use an explicit parent satellite population.
    Fallback: union of the 0.7 DM-poor + DM-rich satellite samples.
    """
    parent = PARENT_SATELLITE_POP.get(sample_class)
    cols = required_columns()
    if parent is not None:
        return extract_population_df(parent, cols=cols)

    poor_name, rich_name = get_threshold_population_names(sample_class, 0.7)
    dfs = [
        extract_population_df(poor_name, cols=cols),
        extract_population_df(rich_name, cols=cols),
    ]
    return concat_and_union(dfs, idcol=COLS.get('id'))


def load_central_sample(sample_class):
    pop = CENTRAL_POP.get(sample_class)
    if pop is None:
        return pd.DataFrame(columns=required_columns())
    return extract_population_df(pop, cols=required_columns())


def split_by_threshold(df, threshold):
    fdm = COLS['fdm']
    poor = df[df[fdm] < threshold].copy()
    rich = df[df[fdm] >= threshold].copy()
    return poor, rich


def evaluate_threshold_against_centrals(poor_df, rich_df, central_df, sample_class, threshold,
                                        n_boot=2000, random_state=42):
    rows = []
    score_terms = []

    npoor = len(poor_df)
    nrich = len(rich_df)
    ncent = len(central_df)

    for key, cfg in CENTRAL_LIKE_VARS.items():
        col = cfg['col']
        label = cfg.get('label', key)
        weight = float(cfg.get('weight', 1.0))

        poor = _nan_clean(poor_df[col].values) if col in poor_df.columns else np.array([])
        rich = _nan_clean(rich_df[col].values) if col in rich_df.columns else np.array([])
        cent = _nan_clean(central_df[col].values) if col in central_df.columns else np.array([])

        row = {
            'sample_class': sample_class,
            'threshold': float(threshold),
            'variable': key,
            'label': label,
            'weight': weight,
            'N_poor': len(poor),
            'N_rich': len(rich),
            'N_central': len(cent),
            'Dpoor_norm': np.nan,
            'Drich_norm': np.nan,
            'DeltaD': np.nan,
            'DeltaD_p16': np.nan,
            'DeltaD_p84': np.nan,
            'rich_is_more_central_like': False,
        }

        if min(len(poor), len(rich), len(cent)) < MIN_N:
            rows.append(row)
            continue

        scale = robust_scale(cent)
        if (not np.isfinite(scale)) or (scale <= 0):
            if not ALLOW_SCALE_FALLBACK:
                rows.append(row)
                continue
            scale = 1.0

        dpoor = wasserstein_1d(poor, cent) / scale
        drich = wasserstein_1d(rich, cent) / scale
        delta = dpoor - drich
        boot = bootstrap_delta_distance(poor, rich, cent, n_boot=n_boot, random_state=random_state)

        row.update({
            'Dpoor_norm': float(dpoor),
            'Drich_norm': float(drich),
            'DeltaD': float(delta),
            'DeltaD_p16': boot['p16'],
            'DeltaD_p84': boot['p84'],
            'rich_is_more_central_like': bool(delta > 0),
        })
        rows.append(row)

        if np.isfinite(delta):
            # Score contribution: positive only when DM-rich is closer to centrals.
            # Compressed with tanh to avoid a single variable dominating the score.
            score_terms.append(weight * np.tanh(delta))

    if len(score_terms) > 0:
        score = float(np.sum(score_terms) / np.sum([cfg.get('weight', 1.0) for cfg in CENTRAL_LIKE_VARS.values()]))
    else:
        score = np.nan

    summary = {
        'sample_class': sample_class,
        'threshold': float(threshold),
        'Npoor': npoor,
        'Nrich': nrich,
        'Ncentral': ncent,
        'central_likeness_score': score,
        'n_positive_variables': int(np.sum([r['rich_is_more_central_like'] for r in rows])),
        'n_variables_evaluated': int(np.sum([np.isfinite(r['DeltaD']) for r in rows])),
    }

    return pd.DataFrame(rows), pd.DataFrame([summary])


def evaluate_named_thresholds(sample_classes=('Normals', 'CompactsMB', 'CompactsSB')):
    all_rows = []
    all_summary = []
    cols = required_columns()

    for sample_class in sample_classes:
        central_df = load_central_sample(sample_class)

        for threshold in THRESHOLDS:
            poor_name, rich_name = get_threshold_population_names(sample_class, threshold)
            poor_df = extract_population_df(poor_name, cols=cols)
            rich_df = extract_population_df(rich_name, cols=cols)
            rows, summary = evaluate_threshold_against_centrals(
                poor_df, rich_df, central_df, sample_class, threshold
            )
            all_rows.append(rows)
            all_summary.append(summary)

    return pd.concat(all_rows, ignore_index=True), pd.concat(all_summary, ignore_index=True)


def evaluate_dense_threshold_scan(sample_classes=('Normals', 'CompactsMB', 'CompactsSB')):
    all_rows = []
    all_summary = []

    for sample_class in sample_classes:
        parent_sat = load_parent_satellite_sample(sample_class)
        central_df = load_central_sample(sample_class)

        for threshold in DENSE_THRESHOLDS:
            poor_df, rich_df = split_by_threshold(parent_sat, threshold)
            rows, summary = evaluate_threshold_against_centrals(
                poor_df, rich_df, central_df, sample_class, threshold
            )
            all_rows.append(rows)
            all_summary.append(summary)

    return pd.concat(all_rows, ignore_index=True), pd.concat(all_summary, ignore_index=True)


def make_central_likeness_figure(detail_df, summary_df, outpath,
                                 rows=('Normals', 'CompactsMB', 'CompactsSB')):
    if detail_df.empty or summary_df.empty:
        return

    # Distinct colors per variable.
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown']
    var_order = list(CENTRAL_LIKE_VARS.keys())
    color_map = {k: colors[i % len(colors)] for i, k in enumerate(var_order)}

    plt.rcParams.update({'figure.figsize': (10.5, 3.0 * len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), 2, hspace=0, wspace=0.1)
    axs = gs.subplots(sharex='col')
    if len(rows) == 1:
        axs = np.array([axs])

    for i, sample_class in enumerate(rows):
        ddet = detail_df[detail_df['sample_class'] == sample_class].copy()
        dsum = summary_df[summary_df['sample_class'] == sample_class].copy().sort_values('threshold')

        # Left: DeltaD curves per variable
        ax = axs[i, 0]
        ax.axhline(0.0, color='0.7', lw=0.9, zorder=0)
        for t in [0.5, 0.8]:
            ax.axvline(t, color='deepskyblue', lw=1.0, ls='--', zorder=0)
        ax.axvline(0.7, color='dodgerblue', lw=2.0, zorder=0)

        for var in var_order:
            dvar = ddet[ddet['variable'] == var].sort_values('threshold')
            if dvar.empty:
                continue
            ax.plot(dvar['threshold'], dvar['DeltaD'], lw=2.0,
                    color=color_map[var], label=CENTRAL_LIKE_VARS[var].get('label', var))
            if 'DeltaD_p16' in dvar.columns and 'DeltaD_p84' in dvar.columns:
                y1 = dvar['DeltaD_p16'].values
                y2 = dvar['DeltaD_p84'].values
                if np.any(np.isfinite(y1)) and np.any(np.isfinite(y2)):
                    ax.fill_between(dvar['threshold'].values, y1, y2,
                                    color=color_map[var], alpha=0.10)

        ax.set_ylabel(sample_class)
        ax.tick_params(direction='in', which='both', top=True, right=True)
        ax.minorticks_on()
        if i == 0:
            ax.set_title(r'Central-likeness contrast: $\Delta D = D_{\rm poor} - D_{\rm rich}$')
        if i == len(rows) - 1:
            ax.set_xlabel(r'$f_{\rm DM,z=0}$ threshold')

        # Right: sample size and aggregate score
        ax = axs[i, 1]
        ax2 = ax.twinx()
        for t in [0.5, 0.8]:
            ax.axvline(t, color='deepskyblue', lw=1.0, ls='--', zorder=0)
        ax.axvline(0.7, color='dodgerblue', lw=2.0, zorder=0)

        ax.plot(dsum['threshold'], dsum['Npoor'], color='tab:blue', lw=2.0, label=r'$N_{\rm poor}$')
        ax.plot(dsum['threshold'], dsum['Nrich'], color='tab:green', lw=2.0, label=r'$N_{\rm rich}$')
        ax2.plot(dsum['threshold'], dsum['central_likeness_score'], color='tab:red',
                 lw=2.0, ls='--', label='central-likeness score')

        ax.tick_params(direction='in', which='both', top=True, right=False)
        ax2.tick_params(direction='in', which='both', top=True, right=True)
        ax.minorticks_on()
        ax2.minorticks_on()

        if i == 0:
            ax.set_title('Sample size and score')
        if i == len(rows) - 1:
            ax.set_xlabel(r'$f_{\rm DM,z=0}$ threshold')

    # global legends
    h0, l0 = axs[0, 0].get_legend_handles_labels()
    h1, l1 = axs[0, 1].get_legend_handles_labels()
    h2, l2 = axs[0, 1].twinx().get_legend_handles_labels()
    handles = h0 + h1 + h2
    labels = l0 + l1 + l2
    if len(handles) > 0:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.995),
                   ncol=min(len(handles), 5), frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, dpi=250, bbox_inches='tight')
    plt.close(fig)


def make_referee_table(detail_df, summary_df, thresholds=(0.5, 0.7, 0.8)):
    rows = []
    
    for sample_class in summary_df['sample_class'].unique():
        for t in thresholds:
            ds = summary_df[(summary_df['sample_class'] == sample_class) &
                            (np.isclose(summary_df['threshold'], t))]
            if ds.empty:
                continue
            ds = ds.iloc[0]
            row = {
                'sample_class': sample_class,
                'threshold': t,
                'Npoor': ds['Npoor'],
                'Nrich': ds['Nrich'],
                'central_likeness_score': ds['central_likeness_score'],
                'n_positive_variables': ds['n_positive_variables'],
                'n_variables_evaluated': ds['n_variables_evaluated'],
            }
            dsub = detail_df[(detail_df['sample_class'] == sample_class) &
                             (np.isclose(detail_df['threshold'], t))]
            for var in CENTRAL_LIKE_VARS.keys():
                dv = dsub[dsub['variable'] == var]
                if dv.empty:
                    row[f'{var}_DeltaD'] = np.nan
                    row[f'{var}_Drich'] = np.nan
                    row[f'{var}_Dpoor'] = np.nan
                else:
                    dv = dv.iloc[0]
                    row[f'{var}_DeltaD'] = dv['DeltaD']
                    row[f'{var}_Drich'] = dv['Drich_norm']
                    row[f'{var}_Dpoor'] = dv['Dpoor_norm']
            rows.append(row)
    return pd.DataFrame(rows)


def print_config_warning_if_empty():
    if len(CENTRAL_LIKE_VARS) == 0:
        print('\n[warning] CENTRAL_LIKE_VARS is empty.')
        print('Edit the variable configuration near the top of the script before running.')
        print('Suggested variables: inner sSFR summary, ex-situ fraction, size-evolution summary.\n')


#%%
def main():
    print_config_warning_if_empty()
    _ensure_dir(OUTDIR)

    # 1) Named thresholds exactly matching the referee values.
    detail_named, summary_named = evaluate_named_thresholds()
    detail_named.to_csv(os.path.join(OUTDIR, 'central_likeness_named_thresholds_detail.csv'), index=False)
    summary_named.to_csv(os.path.join(OUTDIR, 'central_likeness_named_thresholds_summary.csv'), index=False)
    table_ref = make_referee_table(detail_named, summary_named, thresholds=THRESHOLDS)
    table_ref.to_csv(os.path.join(OUTDIR, 'central_likeness_referee_table.csv'), index=False)

    # 2) Dense sweep to locate the robust plateau around the fiducial threshold.
    detail_dense, summary_dense = evaluate_dense_threshold_scan()
    detail_dense.to_csv(os.path.join(OUTDIR, 'central_likeness_dense_scan_detail.csv'), index=False)
    summary_dense.to_csv(os.path.join(OUTDIR, 'central_likeness_dense_scan_summary.csv'), index=False)

    make_central_likeness_figure(
        detail_dense,
        summary_dense,
        outpath=os.path.join(OUTDIR, 'central_likeness_vs_threshold.png')
    )

    print('\nDone. Outputs written to:')
    print(OUTDIR)
    print('\nFiles:')
    print('  - central_likeness_named_thresholds_detail.csv')
    print('  - central_likeness_named_thresholds_summary.csv')
    print('  - central_likeness_referee_table.csv')
    print('  - central_likeness_dense_scan_detail.csv')
    print('  - central_likeness_dense_scan_summary.csv')
    print('  - central_likeness_vs_threshold.png')


if __name__ == '__main__':
    main()
