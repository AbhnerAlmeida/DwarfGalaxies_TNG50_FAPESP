#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:47:26 2026

@author: abhner
"""

import numpy as np
import matplotlib.pyplot as plt
import os
plt.style.use(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/src/abhner.mplstyle")

import pandas as pd
from scipy.stats import wasserstein_distance

SIGNALS = {'zentry': {'col': 'z_At_FirstEntry',
  'label': '$\\Delta z_{\\rm entry}$',
  'summary': 'median',
  'expected_sign': 1,
  'priority': 1},
 'm200': {'col': 'M200Mean',
  'label': '$\\Delta \overline{(\log M_{200})}$',
  'summary': 'median',
  'expected_sign': 1,
  'priority': 1},
 'rmean': {'col': 'rOverR200Mean_New',
  'label': '$\\Delta \overline{(R/R_{200})}$',
  'summary': 'median',
  'expected_sign': -1,
  'priority': 1},
 'rmin': {'col': 'rOverR200Min',
  'label': '$\\Delta (R/R_{200})^{\\min}_{\mathrm{pericentre}}$',
  'summary': 'median',
  'expected_sign': -1,
  'priority': 1},
 'ssfr_central': {'col': 'sSFRinHalfRadAfterz5',
  'label': r'$\Delta D_{\rm cen}$ inner sSFR',
  'summary': 'median',
  'expected_sign': 1,
  'priority': 1}}


POP_CONFIG = {
    'CompactsMB': {
        'satellite_parent': 'MBC_Satellite',
        'central_parent':   'MBC_Central',
    },
    'CompactsSB': {
        'satellite_parent': 'SBC_Satellite',
        'central_parent':   'SBC_Central',
    },
    # se quiser incluir Normals também:
    # 'Normals': {
    #     'satellite_parent': 'Normal_Satellite',
    #     'central_parent':   'Normal_Central',
    # },
}

FDM_KEY = 'DMFrac_99'

labels = [ r'Compacts$_\mathrm{MB}$', r'Compacts$_\mathrm{SB}$']

fs = 11

columnspacing = 0.4
handletextpad = 0.5
labelspacing = 0.3
fontlegend= 22
framealpha= 0.7
alphaScater=0.8
quantile=0.28
handlelength = 1.5

lwsize = 0.8

#%%

def extract_numeric_population(population, keys, dfName='PaperII', Name='Name'):
    df = TNG.extractPopulation(population, dfName=dfName, Name=Name).copy()

    keep = [k for k in keys if k in df.columns]
    out = df[keep].copy()

    for k in keep:
        out[k] = pd.to_numeric(out[k], errors='coerce')

    return out

#%%

def transform_signal_values(x, transform=None):
    x = np.asarray(x, dtype=float)

    if transform is None:
        return x

    if transform == 'log10':
        out = np.full_like(x, np.nan, dtype=float)
        m = np.isfinite(x) & (x > 0)
        out[m] = np.log10(x[m])
        return out

    raise ValueError(f"Unknown transform: {transform}")


def build_results_dense_signal_scan(
    sample_classes=('CompactsMB', 'CompactsSB'),
    thresholds=np.linspace(0.45, 0.85, 41),
    signal_order=('zentry', 'm200', 'rmean', 'rmin'),
    dfName='PaperII',
    Name='Name',
    statistic='median',
):
    rows = []

    needed_keys = [FDM_KEY] + [SIGNALS[s]['col'] for s in signal_order]

    for sc in sample_classes:
        sat_pop = POP_CONFIG[sc]['satellite_parent']

        sat_df = extract_numeric_population(
            sat_pop,
            needed_keys,
            dfName=dfName,
            Name=Name
        )

        fdm = sat_df[FDM_KEY].to_numpy(dtype=float)
        finite_fdm = np.isfinite(fdm)

        for t in thresholds:
            poor_mask = finite_fdm & (fdm < t)
            rich_mask = finite_fdm & (fdm >= t)

            Npoor_total = int(np.sum(poor_mask))
            Nrich_total = int(np.sum(rich_mask))

            for s in signal_order:
                key = SIGNALS[s]['col']
                transform = SIGNALS[s].get('transform', None)

                vals = sat_df[key].to_numpy(dtype=float)
                vals = transform_signal_values(vals, transform=transform)

                poor_vals = vals[poor_mask]
                rich_vals = vals[rich_mask]

                poor_vals = poor_vals[np.isfinite(poor_vals)]
                rich_vals = rich_vals[np.isfinite(rich_vals)]

                if len(poor_vals) == 0 or len(rich_vals) == 0:
                    poor_stat = np.nan
                    rich_stat = np.nan
                    delta = np.nan
                else:
                    if statistic == 'median':
                        poor_stat = np.nanmedian(poor_vals)
                        rich_stat = np.nanmedian(rich_vals)
                    elif statistic == 'mean':
                        poor_stat = np.nanmean(poor_vals)
                        rich_stat = np.nanmean(rich_vals)
                    else:
                        raise ValueError("statistic must be either 'median' or 'mean'")

                    delta = poor_stat - rich_stat

                rows.append({
                    'sample_class': sc,
                    'threshold': t,
                    'signal': s,
                    'delta_poor_minus_rich': delta,
                    'Npoor_total': Npoor_total,
                    'Nrich_total': Nrich_total,
                    'poor_stat': poor_stat,
                    'rich_stat': rich_stat,
                    'poor_median': np.nanmedian(poor_vals) if len(poor_vals) else np.nan,
                    'rich_median': np.nanmedian(rich_vals) if len(rich_vals) else np.nan,
                    'poor_mean': np.nanmean(poor_vals) if len(poor_vals) else np.nan,
                    'rich_mean': np.nanmean(rich_vals) if len(rich_vals) else np.nan,
                })

    return pd.DataFrame(rows)

#%%
thresholds = np.linspace(0.45, 0.85, 41)
results_dense = build_results_dense_signal_scan(
    sample_classes=('CompactsMB', 'CompactsSB'),
    thresholds=thresholds,
    signal_order=('zentry', 'm200', 'rmean', 'rmin'),
    statistic='median',
    dfName='PaperII',
    Name='Name',
)

#%%
def plot_signal_scan(df_results, score_df=None,
                     sample_classes=('CompactsMB', 'CompactsSB'),
                     signal_order=('zentry', 'm200', 'rmean', 'rmin'),
                     threshold_ref=0.7, threshold_marks=(0.5, 0.8),
                     figsize=(4.8, 4.5),
                     show_counts=True,
                     counts_as_step=True):
    """
    Main referee figure:
      - left y-axis  : effect size (DM-poor minus DM-rich) for each signal vs threshold
      - right y-axis : Npoor_total and Nrich_total vs threshold
    One row per sample_class.
    """

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(len(sample_classes), 1, hspace=0.0)
    axs = gs.subplots(sharex=True)

    if len(sample_classes) == 1:
        axs = np.array([axs])

    # fixed colors for signals
    colors = {
        'zentry': 'tab:blue',
        'm200':   'tab:green',
        'rmean':  'tab:orange',
        'rmin':   'tab:red',
        'ssfr_central': 'tab:blue',
        'exsitu_central':   'tab:red',
    }

    # neutral colors for counts, so they don't compete with the science signals
    count_colors = {
        'poor': '0.20',   # dark gray
        'rich': '0.55',   # lighter gray
    }

    axs_right = []

    for i, sc in enumerate(sample_classes):
        gsc = df_results[df_results['sample_class'] == sc].copy()
        ax = axs[i]

        ax.tick_params(labelsize=0.88 * fs)

        # -----------------------------
        # LEFT AXIS: effect sizes
        # -----------------------------
        for s in signal_order:
            gsig = gsc[gsc['signal'] == s].sort_values('threshold')
            if len(gsig) == 0:
                continue

            ax.plot(
                gsig['threshold'],
                gsig['delta_poor_minus_rich'],
                lw=1.5*lwsize,
                color=colors.get(s, None),
                label=SIGNALS[s]['label']
            )

        ax.axhline(0, color='k', lw=0.8*lwsize, alpha=0.4)
        ax.axvline(threshold_ref, color='dodgerblue', lw=1.6*lwsize)
        for t in threshold_marks:
            ax.axvline(t, color='deepskyblue', ls='--', lw=1.4*lwsize)

        ax.set_ylabel(labels[i], fontsize=fs)
        ax.minorticks_on()
        #ax.tick_params(direction='in', which='both', top=True, right=False)

        if i == 0:
            ax.set_title('DM-poor minus DM-rich', fontsize=fs*1.1)
            ax.text(0.85, 0.2, "MB", fontsize = 0.99*12)
        elif i == 1:
            ax.text(0.85, 1.2, "SB", fontsize = 0.99*12)

        # -----------------------------
        # RIGHT AXIS: population counts
        # -----------------------------
        axr = None
        if show_counts:
            axr = ax.twinx()
            axr.tick_params(labelsize=0.88 * fs)

            # Build a single population curve per threshold
            # (counts should not depend on signal)
            gpop = (
                gsc.sort_values('threshold')
                   .groupby('threshold', as_index=False)[['Npoor_total', 'Nrich_total']]
                   .first()
                   .sort_values('threshold')
            )

            if counts_as_step:
                axr.step(
                    gpop['threshold'], gpop['Npoor_total'],
                    where='mid', lw=1.3*lwsize, color=count_colors['poor'],
                    alpha=0.9, label=r'$N_{\rm poor}$'
                )
                axr.step(
                    gpop['threshold'], gpop['Nrich_total'],
                    where='mid', lw=1.3*lwsize, ls='--', color=count_colors['rich'],
                    alpha=0.95, label=r'$N_{\rm rich}$'
                )
            else:
                axr.plot(
                    gpop['threshold'], gpop['Npoor_total'],
                    lw=1.3*lwsize, color=count_colors['poor'],
                    alpha=0.9, label=r'$N_{\rm poor}$'
                )
                axr.plot(
                    gpop['threshold'], gpop['Nrich_total'],
                    lw=1.3*lwsize, ls='--', color=count_colors['rich'],
                    alpha=0.95, label=r'$N_{\rm rich}$'
                )

            ymax = np.nanmax(gpop[['Npoor_total', 'Nrich_total']].to_numpy())
            axr.set_ylim(0.5, 1.08 * ymax if ymax > 0 else 1)
            axr.tick_params(axis='y', direction='in', which='both',
                            colors='0.35', right=True)
            axr.spines['right'].set_color('0.5')
            axr.set_ylabel('Population', color='0.35', fontsize=fs)
            axr.minorticks_on()

        axs_right.append(axr)

        # -----------------------------
        # Legend only in top panel
        # -----------------------------
        if i == 1:
            h1, l1 = ax.get_legend_handles_labels()
            if axr is not None:
                h2, l2 = axr.get_legend_handles_labels()
            else:
                h2, l2 = [], []

            ax.legend(
                h1 + h2, l1 + l2,
                frameon=True, fontsize=0.88 * fs,
                ncol=2, loc='center left', framealpha=framealpha,
                columnspacing=columnspacing, handlelength=handlelength,
                handletextpad=handletextpad, labelspacing=labelspacing
                
            )

    axs[-1].set_xlabel(r'$f_{\rm DM,z=0}$ threshold', fontsize=1.1*fs)

    fig.tight_layout()
    return fig, axs, axs_right

#%%

fig, axs, axs_right = plot_signal_scan(
    results_dense,
    score_df=None,
    sample_classes=('CompactsMB', 'CompactsSB'),
    signal_order=('zentry', 'm200', 'rmean', 'rmin'),
    threshold_ref=0.7,
    threshold_marks=(0.5, 0.8),
    figsize=(4.8, 4.5),
    show_counts=True,
    counts_as_step=True,
)

pathBase =  os.getenv("HOME")+'/TNG_Analyzes/Figs/TNG50/'
 
plt.savefig(pathBase +'AppendixB' +
            '.pdf', bbox_inches='tight')


plt.savefig(pathBase +'AppendixB' +
            '.png', bbox_inches='tight', dpi=400)



#%%

def extract_numeric_population(population, keys, dfName='PaperII', Name='Name'):
    df = TNG.extractPopulation(population, dfName=dfName, Name=Name).copy()

    keep = [k for k in keys if k in df.columns]
    out = df[keep].copy()

    for k in keep:
        out[k] = pd.to_numeric(out[k], errors='coerce')

    return out


def robust_scale(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    p16, p84 = np.percentile(x, [16, 84])
    scale = 0.5 * (p84 - p16)

    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(x)

    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    return scale


def normalized_wasserstein(x, y, scale=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    wd = wasserstein_distance(x, y)

    if scale is None:
        scale = robust_scale(y)

    if not np.isfinite(scale) or scale <= 0:
        return np.nan

    return wd / scale


#%%

def build_results_dense_centrallike(
    sample_classes=('CompactsMB', 'CompactsSB'),
    thresholds=np.linspace(0.45, 0.85, 41),
    signal_order=('ssfr_central', 'exsitu_central'),
    dfName='PaperII',
    Name='Name',
):
    rows = []

    needed_keys = [FDM_KEY] + [CENTRAL_SIGNALS[s]['key'] for s in signal_order]

    for sc in sample_classes:
        sat_pop = POP_CONFIG[sc]['satellite_parent']
        cen_pop = POP_CONFIG[sc]['central_parent']

        sat_df = extract_numeric_population(sat_pop, needed_keys, dfName=dfName, Name=Name)
        cen_df = extract_numeric_population(cen_pop, needed_keys, dfName=dfName, Name=Name)

        for t in thresholds:
            poor_mask = sat_df[FDM_KEY] < t
            rich_mask = sat_df[FDM_KEY] >= t

            Npoor_total = int(np.sum(np.isfinite(sat_df[FDM_KEY]) & poor_mask))
            Nrich_total = int(np.sum(np.isfinite(sat_df[FDM_KEY]) & rich_mask))

            for s in signal_order:
                key = CENTRAL_SIGNALS[s]['key']

                cen_vals  = cen_df[key].to_numpy(dtype=float)
                poor_vals = sat_df.loc[poor_mask, key].to_numpy(dtype=float)
                rich_vals = sat_df.loc[rich_mask, key].to_numpy(dtype=float)

                cen_vals  = cen_vals[np.isfinite(cen_vals)]
                poor_vals = poor_vals[np.isfinite(poor_vals)]
                rich_vals = rich_vals[np.isfinite(rich_vals)]

                scale = robust_scale(cen_vals)

                d_poor = normalized_wasserstein(poor_vals, cen_vals, scale=scale)
                d_rich = normalized_wasserstein(rich_vals, cen_vals, scale=scale)

                rows.append({
                    'sample_class': sc,
                    'threshold': t,
                    'signal': s,
                    'delta_poor_minus_rich': d_poor - d_rich,
                    'Npoor_total': Npoor_total,
                    'Nrich_total': Nrich_total,
                    'Dpoor_to_central': d_poor,
                    'Drich_to_central': d_rich,
                    'poor_median': np.nanmedian(poor_vals) if len(poor_vals) else np.nan,
                    'rich_median': np.nanmedian(rich_vals) if len(rich_vals) else np.nan,
                    'central_median': np.nanmedian(cen_vals) if len(cen_vals) else np.nan,
                })

    return pd.DataFrame(rows)


#%%
results_dense_centrallike = build_results_dense_centrallike(
    sample_classes=('CompactsMB', 'CompactsSB'),
    thresholds=np.linspace(0.45, 0.85, 41),
    signal_order=('ssfr_central', 'exsitu_central'),
    dfName='PaperII',
    Name='Name',
)

fig, axs, axs_right = plot_signal_scan(
    results_dense_centrallike,
    score_df=None,
    sample_classes=('CompactsMB', 'CompactsSB'),
    signal_order=('ssfr_central', 'exsitu_central'),
    threshold_ref=0.7,
    threshold_marks=(0.5, 0.8),
    figsize=(4.8, 4.5),
    show_counts=True,
    counts_as_step=True,
)

plt.savefig(pathBase + 'AppendixB_centralLike.pdf', bbox_inches='tight')
plt.savefig(pathBase + 'AppendixB_centralLike.png', bbox_inches='tight', dpi=400)
plt.show()