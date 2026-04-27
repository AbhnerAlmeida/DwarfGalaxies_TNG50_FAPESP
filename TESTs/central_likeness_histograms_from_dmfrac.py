import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp

# ============================================================
# Configuration (edit population names if needed)
# ============================================================
DFNAME = 'PaperII'
NAMEARG = 'Name'  # passed to TNG.extractPopulation(...)

FDM_KEY = 'DMFrac_99'

# Compare each satellite class to its corresponding central class.
# Edit the population names below to match your project.
POP_CONFIG = {
    'Normals': {
        'satellite': 'Normal_Satellite',
        'central': 'Normal_Central',
    },
    'CompactsMB': {
        'satellite': 'MBC_Satellite',
        'central': 'MBC_Central',
    },
    'CompactsSB': {
        'satellite': 'SBC_Satellite',
        'central': 'SBC_Central',
    },
}

CENTRAL_LIKE_VARS = {
    'inner_ssfr_mean_zlt5': {
        'col': 'sSFR_In_TrueRhpkpc_Mean_zlt5',
        'label': r'inner sSFR (z<5)',
        'weight': 1.0,
    },
    'exsitu_frac': {
        'col': 'ExSituFrac_Total',
        'label': r'ex-situ fraction',
        'weight': 1.0,
    },
    'size_mean_zlt5': {
        'col': 'logRhalf_Mean_zlt5',
        'label': r'mean $\log r_{1/2}$ (z<5)',
        'weight': 0.6,
    },
}

COLORS = {
    'central': 'black',
    'poor': 'tab:orange',
    'rich': 'tab:blue',
    'Normals': 'tab:orange',
    'CompactsMB': 'tab:blue',
    'CompactsSB': 'tab:green',
}


# ============================================================
# Basic helpers
# ============================================================

def _as_finite(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def ecdf(values):
    x = np.sort(_as_finite(values))
    if len(x) == 0:
        return np.array([]), np.array([])
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def safe_wasserstein(a, b):
    a = _as_finite(a)
    b = _as_finite(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(wasserstein_distance(a, b))


def normalized_wasserstein(a, b):
    """
    Wasserstein distance normalized by the central-population scatter.
    This makes different variables easier to compare.
    """
    a = _as_finite(a)
    b = _as_finite(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    wd = wasserstein_distance(a, b)
    scale = np.nanstd(b)
    if not np.isfinite(scale) or scale <= 0:
        return float(wd)
    return float(wd / scale)


def split_satellite_df(df_sat, threshold, fdm_key=FDM_KEY):
    vals = pd.to_numeric(df_sat[fdm_key], errors='coerce')
    mask = np.isfinite(vals)
    df = df_sat.loc[mask].copy()
    poor = df.loc[df[fdm_key] < threshold].copy()
    rich = df.loc[df[fdm_key] >= threshold].copy()
    return poor, rich


# ============================================================
# Data extraction
# ============================================================

def extract_population_df(population, cols, dfName=DFNAME, Name=NAMEARG):
    """
    Requires the user's analysis environment to already have TNG loaded.
    Example:
        Sample = TNG.extractPopulation('Normal_Satellite', dfName='PaperII', Name='Name')
    """
    Sample = TNG.extractPopulation(population, dfName=dfName, Name=Name)
    out = pd.DataFrame()
    for c in cols:
        if c in Sample.columns:
            out[c] = Sample[c].values
        else:
            out[c] = np.nan
    return out


def required_columns(fdm_key=FDM_KEY, var_defs=CENTRAL_LIKE_VARS):
    cols = [fdm_key]
    cols += [v['col'] for v in var_defs.values()]
    return sorted(set(cols))


def build_cache(pop_config=POP_CONFIG, var_defs=CENTRAL_LIKE_VARS,
                fdm_key=FDM_KEY, dfName=DFNAME, Name=NAMEARG):
    cols = required_columns(fdm_key, var_defs)
    cache = {}
    for cls, cfg in pop_config.items():
        cache[cls] = {
            'satellite': extract_population_df(cfg['satellite'], cols, dfName=dfName, Name=Name),
            'central': extract_population_df(cfg['central'], cols, dfName=dfName, Name=Name),
        }
    return cache


# ============================================================
# 1) Reference figure for DMFrac itself: histogram + CDF
# ============================================================

def plot_dmfrac_reference(cache, fdm_key=FDM_KEY, thresholds=(0.5, 0.7, 0.8),
                          bins=np.linspace(0, 1.02, 41), figsize=(12, 4.2)):
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True)

    # Collect values
    central_vals = []
    for cls in cache:
        central_vals.append(_as_finite(cache[cls]['central'][fdm_key].values))
    central_vals = np.concatenate([v for v in central_vals if len(v) > 0]) if len(central_vals) else np.array([])

    # Left: histogram
    axs[0].hist(central_vals, bins=bins, density=True, histtype='step', lw=2.6,
                color=COLORS['central'], label='Centrals')
    for cls in ['Normals', 'CompactsMB', 'CompactsSB']:
        vals = _as_finite(cache[cls]['satellite'][fdm_key].values)
        axs[0].hist(vals, bins=bins, density=True, histtype='step', lw=2.3,
                    color=COLORS[cls], label=cls)

    # Right: CDF
    x, y = ecdf(central_vals)
    axs[1].plot(x, y, lw=2.8, color=COLORS['central'], label='Centrals')
    for cls in ['Normals', 'CompactsMB', 'CompactsSB']:
        vals = _as_finite(cache[cls]['satellite'][fdm_key].values)
        x, y = ecdf(vals)
        axs[1].plot(x, y, lw=2.4, color=COLORS[cls], label=cls)

    for ax in axs:
        for t in thresholds:
            if t == 0.7:
                ax.axvline(t, color='dodgerblue', lw=2.2)
            else:
                ax.axvline(t, color='deepskyblue', lw=1.4, ls='--')
        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', top=True, right=True)
        ax.set_xlabel(r'$f_{\rm DM,z=0}$')

    axs[0].set_ylabel('Density')
    axs[0].set_title('Normalized histogram')
    axs[1].set_ylabel('CDF')
    axs[1].set_title('Empirical CDF')
    axs[0].legend(frameon=False, loc='upper left')
    fig.tight_layout()
    return fig, axs


# ============================================================
# 2) Histogram + CDF for CENTRAL_LIKE variables at a chosen threshold
# ============================================================

def compare_to_central_for_threshold(cache, threshold, var_name,
                                     var_defs=CENTRAL_LIKE_VARS,
                                     fdm_key=FDM_KEY):
    vcol = var_defs[var_name]['col']
    rows = []
    for cls in cache:
        dfc = cache[cls]['central']
        dfs = cache[cls]['satellite']
        poor, rich = split_satellite_df(dfs, threshold, fdm_key=fdm_key)

        c = _as_finite(dfc[vcol].values)
        p = _as_finite(poor[vcol].values)
        r = _as_finite(rich[vcol].values)

        wd_poor = safe_wasserstein(p, c)
        wd_rich = safe_wasserstein(r, c)
        nwd_poor = normalized_wasserstein(p, c)
        nwd_rich = normalized_wasserstein(r, c)
        deltaD = nwd_poor - nwd_rich if np.isfinite(nwd_poor) and np.isfinite(nwd_rich) else np.nan

        ks_poor = ks_2samp(p, c).statistic if len(p) > 1 and len(c) > 1 else np.nan
        ks_rich = ks_2samp(r, c).statistic if len(r) > 1 and len(c) > 1 else np.nan

        rows.append({
            'sample_class': cls,
            'threshold': threshold,
            'var_name': var_name,
            'var_col': vcol,
            'N_central': len(c),
            'N_poor': len(p),
            'N_rich': len(r),
            'wd_poor': wd_poor,
            'wd_rich': wd_rich,
            'nwd_poor': nwd_poor,
            'nwd_rich': nwd_rich,
            'deltaD': deltaD,
            'ks_poor': ks_poor,
            'ks_rich': ks_rich,
            'central_median': np.nanmedian(c) if len(c) else np.nan,
            'poor_median': np.nanmedian(p) if len(p) else np.nan,
            'rich_median': np.nanmedian(r) if len(r) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_variable_hist_cdf(cache, var_name, threshold=0.7, var_defs=CENTRAL_LIKE_VARS,
                           fdm_key=FDM_KEY, bins=30, figsize=(12, 8.5),
                           xlim=None, density=True):
    """
    One figure per variable:
      3 rows (Normals, CompactsMB, CompactsSB)
      2 cols (Histogram, CDF)

    In each row, compare:
      central vs DM-poor vs DM-rich
    where the split is done using DMFrac_99 and the chosen threshold.
    """
    vcol = var_defs[var_name]['col']
    vlabel = var_defs[var_name]['label']

    fig, axs = plt.subplots(3, 2, figsize=figsize, sharex='col')
    classes = ['Normals', 'CompactsMB', 'CompactsSB']

    summary = compare_to_central_for_threshold(cache, threshold, var_name,
                                               var_defs=var_defs, fdm_key=fdm_key)

    for i, cls in enumerate(classes):
        row = summary.loc[summary['sample_class'] == cls].iloc[0]
        dfc = cache[cls]['central']
        dfs = cache[cls]['satellite']
        poor, rich = split_satellite_df(dfs, threshold, fdm_key=fdm_key)

        c = _as_finite(dfc[vcol].values)
        p = _as_finite(poor[vcol].values)
        r = _as_finite(rich[vcol].values)

        # histogram
        ax = axs[i, 0]
        if isinstance(bins, int):
            allv = np.concatenate([x for x in [c, p, r] if len(x) > 0])
            if len(allv) > 0:
                lo, hi = np.nanmin(allv), np.nanmax(allv)
                bbins = np.linspace(lo, hi, bins)
            else:
                bbins = bins
        else:
            bbins = bins

        if len(c):
            ax.hist(c, bins=bbins, density=density, histtype='step', lw=2.4,
                    color=COLORS['central'], label='Centrals')
        if len(p):
            ax.hist(p, bins=bbins, density=density, histtype='step', lw=2.2,
                    color=COLORS['poor'], label=f'DM-poor (< {threshold:.2f})')
        if len(r):
            ax.hist(r, bins=bbins, density=density, histtype='step', lw=2.2,
                    color=COLORS['rich'], label=f'DM-rich (>= {threshold:.2f})')

        ax.set_ylabel(cls)
        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', top=True, right=True)
        if i == 0:
            ax.set_title(f'{vlabel} — histogram')
            ax.legend(frameon=False, loc='best')

        txt = (fr'$N_{{cen}}={row.N_central}$' + '\n' +
               fr'$N_{{poor}}={row.N_poor},\ N_{{rich}}={row.N_rich}$' + '\n' +
               fr'$\tilde D_{{poor}}={row.nwd_poor:.2f},\ \tilde D_{{rich}}={row.nwd_rich:.2f}$' + '\n' +
               fr'$\Delta D={row.deltaD:.2f}$')
        ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha='right', va='top', fontsize=10)

        # CDF
        ax = axs[i, 1]
        if len(c):
            x, y = ecdf(c)
            ax.plot(x, y, lw=2.5, color=COLORS['central'], label='Centrals')
        if len(p):
            x, y = ecdf(p)
            ax.plot(x, y, lw=2.2, color=COLORS['poor'], label='DM-poor')
        if len(r):
            x, y = ecdf(r)
            ax.plot(x, y, lw=2.2, color=COLORS['rich'], label='DM-rich')

        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', top=True, right=True)
        if i == 0:
            ax.set_title(f'{vlabel} — CDF')

        if np.isfinite(row.deltaD):
            if row.deltaD > 0:
                verdict = 'DM-rich more central-like'
            elif row.deltaD < 0:
                verdict = 'DM-poor more central-like'
            else:
                verdict = 'tie'
        else:
            verdict = 'undefined'
        ax.text(0.03, 0.06, verdict, transform=ax.transAxes, ha='left', va='bottom', fontsize=10)

    axs[-1, 0].set_xlabel(vlabel)
    axs[-1, 1].set_xlabel(vlabel)
    if xlim is not None:
        for ax in axs.flat:
            ax.set_xlim(*xlim)

    fig.suptitle(fr'Central-likeness split with {fdm_key} threshold = {threshold:.2f}', y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig, axs, summary


# ============================================================
# 3) Threshold sweep for central-likeness
# ============================================================

def sweep_central_likeness(cache, thresholds=np.linspace(0.45, 0.85, 41),
                           var_defs=CENTRAL_LIKE_VARS, fdm_key=FDM_KEY):
    rows = []
    for t in thresholds:
        for var_name in var_defs:
            df = compare_to_central_for_threshold(cache, t, var_name,
                                                 var_defs=var_defs,
                                                 fdm_key=fdm_key)
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)

    # Aggregate a weighted score over the chosen central-like variables.
    score_rows = []
    for cls in out['sample_class'].dropna().unique():
        dcls = out[out['sample_class'] == cls]
        for t in sorted(dcls['threshold'].unique()):
            d = dcls[dcls['threshold'] == t]
            weighted = 0.0
            wsum = 0.0
            for var_name, meta in var_defs.items():
                w = float(meta.get('weight', 1.0))
                row = d[d['var_name'] == var_name]
                if len(row) == 0:
                    continue
                deltaD = row['deltaD'].iloc[0]
                if np.isfinite(deltaD):
                    weighted += w * deltaD
                    wsum += w
            score = weighted / wsum if wsum > 0 else np.nan

            # also keep sizes from the first variable row (same split for all vars)
            row0 = d.iloc[0]
            score_rows.append({
                'sample_class': cls,
                'threshold': t,
                'central_likeness_score': score,
                'N_poor': row0['N_poor'],
                'N_rich': row0['N_rich'],
                'N_central': row0['N_central'],
            })
    score_df = pd.DataFrame(score_rows)
    return out, score_df


def plot_central_likeness_sweep(score_df, thresholds_ref=(0.5, 0.7, 0.8),
                                figsize=(10, 8.5)):
    fig, axs = plt.subplots(3, 2, figsize=figsize, sharex=True)
    classes = ['Normals', 'CompactsMB', 'CompactsSB']

    for i, cls in enumerate(classes):
        d = score_df[score_df['sample_class'] == cls].sort_values('threshold')
        x = d['threshold'].values

        # left: score
        ax = axs[i, 0]
        ax.axhline(0.0, color='0.75', lw=0.8)
        ax.plot(x, d['central_likeness_score'].values, lw=2.4, color='tab:red')
        for t in thresholds_ref:
            if t == 0.7:
                ax.axvline(t, color='dodgerblue', lw=2.1)
            else:
                ax.axvline(t, color='deepskyblue', lw=1.3, ls='--')
        ax.set_ylabel(cls)
        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', top=True, right=True)
        if i == 0:
            ax.set_title(r'Central-likeness score = $	ilde D_{poor} - 	ilde D_{rich}$')

        # right: sizes
        ax = axs[i, 1]
        ax.plot(x, d['N_poor'].values, lw=2.2, color='tab:orange', label=r'$N_{poor}$')
        ax.plot(x, d['N_rich'].values, lw=2.2, color='tab:blue', label=r'$N_{rich}$')
        for t in thresholds_ref:
            if t == 0.7:
                ax.axvline(t, color='dodgerblue', lw=2.1)
            else:
                ax.axvline(t, color='deepskyblue', lw=1.3, ls='--')
        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', top=True, right=True)
        if i == 0:
            ax.set_title('Sample size')
            ax.legend(frameon=False, loc='best')

    axs[-1, 0].set_xlabel(r'$f_{\rm DM,z=0}$ threshold')
    axs[-1, 1].set_xlabel(r'$f_{\rm DM,z=0}$ threshold')
    fig.tight_layout()
    return fig, axs


# ============================================================
# Example usage in a notebook
# ============================================================
EXAMPLE = r'''
# 1) Build cache from your PaperII dataframe/populations
cache = build_cache()

# 2) Reproduce the DMFrac histogram/CDF figure
fig, axs = plot_dmfrac_reference(cache)
plt.show()

# 3) For a chosen threshold, compare central-like variables
fig, axs, summary = plot_variable_hist_cdf(cache, 'inner_ssfr_mean_zlt5', threshold=0.7)
plt.show()
print(summary)

fig, axs, summary = plot_variable_hist_cdf(cache, 'exsitu_frac', threshold=0.7)
plt.show()
print(summary)

fig, axs, summary = plot_variable_hist_cdf(cache, 'size_mean_zlt5', threshold=0.7)
plt.show()
print(summary)

# 4) Sweep threshold and summarize whether DM-rich stays more central-like
all_rows, score_df = sweep_central_likeness(cache)
fig, axs = plot_central_likeness_sweep(score_df)
plt.show()
'''

if __name__ == '__main__':
    print('This module is intended to be imported into your notebook/session.')
    print(EXAMPLE)
