import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# USER CONFIGURATION
# ============================================================================
# This script is designed to work in the same environment as your current
# Appendix-B script, i.e. with TNG.extractPopulation(...) already available.
# It calibrates fDM,z=0 against more direct stripping proxies and evaluates
# purity / contamination / completeness for DM-poor and DM-rich classifications.
#
# IMPORTANT:
#   1) Edit KEYS below so they match your dataframe column names.
#   2) Edit POP_CONFIG so the population names match your TNG setup.
#   3) If you do not have direct parent satellite populations, keep
#      "satellite_parent" = None and provide a split_ref with the two existing
#      threshold-0.7 populations. They will be concatenated to reconstruct the
#      parent sample.
#   4) For centrals, set the population names you want to use as negative
#      controls. If you do not have them yet, set them to None.
# ============================================================================

DFNAME = 'PaperII'
NAMECOL = 'Name'
OUTDIR = 'fdm_threshold_calibration'

KEYS = {
    # Required keys
    'fdm': 'DMFrac_99',                    # final DM fraction used as classifier
    'dm_retained': 'MDM_Norm_Max_99',   # M_DM(z=0) / M_DM,max
    'rmin': 'rOverR200Min',             # minimum pericentre in host virial units

    # Optional but strongly recommended
    'zentry': 'z_At_FirstEntry',        # or 'z_At_FinalEntry' if that is your choice

    # Optional identifier column for de-duplication (set to NAMECOL if unsure)
    'id': NAMECOL,
}

POP_CONFIG = {
    'Normals': {
        'satellite_parent': None,
        'satellite_split_ref': ['Normal_DMpoor_Satellite', 'Normal_DMrich_Satellite'],
        'central_parent': 'Normal_Central',   # edit if needed
    },
    'CompactsMB': {
        'satellite_parent': None,
        'satellite_split_ref': ['MBC_DMpoor_Satellite', 'MBC_DMrich_Satellite'],
        'central_parent': 'MBC_Central',      # edit if needed
    },
    'CompactsSB': {
        'satellite_parent': None,
        'satellite_split_ref': ['SBC_DMpoor_Satellite', 'SBC_DMrich_Satellite'],
        'central_parent': 'SBC_Central',      # edit if needed
    },
}

# Thresholds to scan for the classifier fDM,z=0 < t  => DM-poor
THRESHOLDS = np.round(np.arange(0.40, 0.86, 0.005), 3)
REPORT_THRESHOLDS = [0.50, 0.60, 0.70, 0.80]

# ---------------------------------------------------------------------------
# Truth-label calibration options
# ---------------------------------------------------------------------------
# The aim is to define a less circular “ground truth” using stronger stripping
# proxies. By default the script uses a HYBRID QUANTILE strategy:
#
#   strongly processed (DM-poor truth) if at least N votes among:
#       - low DM retained
#       - deep pericentre
#       - early entry (high zentry)
#
#   weakly processed (DM-rich truth) if at least N votes among:
#       - high DM retained
#       - shallow pericentre
#       - late entry (low zentry)
#
# Objects in between are marked as ambiguous and are excluded from the main
# confusion-matrix metrics. Centrals are evaluated separately as a negative
# control (false positives among centrals).
# ---------------------------------------------------------------------------

TRUTH_CONFIG = {
    'mode': 'hybrid_fixed',   # options: 'hybrid_quantile', 'hybrid_fixed'
    'votes_required': 2,

    # For hybrid_quantile mode
    'q_low': 0.25,
    'q_high': 0.75,

    # For hybrid_fixed mode (edit if you prefer explicit physics-motivated cuts)
    # 'fixed': {
    #     'dm_retained_low': 0.10,
    #     'dm_retained_high': 0.40,
    #     'rmin_low': 0.15,
    #     'rmin_high': 0.40,
    #     'zentry_high': 0.80,
    #     'zentry_low': 0.40,
    # },
    'fixed': {
    'dm_retained_low': 0.08,
    'dm_retained_high': 0.60,
    'rmin_low': 0.10,
    'rmin_high': 0.60,
    'zentry_high': 1.00,
    'zentry_low': 0.35,
    }
}

SCATTER_COLOR_KEY = 'rmin'  # options: 'rmin' or 'zentry'
SCATTER_THRESHOLDS = [0.50, 0.70, 0.80]

# ============================================================================
# Helper utilities
# ============================================================================

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def required_columns() -> List[str]:
    cols = [KEYS['fdm'], KEYS['dm_retained'], KEYS['rmin']]
    if KEYS.get('zentry'):
        cols.append(KEYS['zentry'])
    if KEYS.get('id'):
        cols.append(KEYS['id'])
    return list(dict.fromkeys(cols))


def extract_population_df(population: str,
                          dfName: str = DFNAME,
                          Name: str = NAMECOL,
                          cols: Optional[List[str]] = None) -> pd.DataFrame:
    sample = TNG.extractPopulation(population, dfName=dfName, Name=Name).copy()
    if cols is None:
        cols = required_columns()

    missing = [c for c in cols if c not in sample.columns]
    if missing:
        raise KeyError(
            f"Population '{population}' is missing columns: {missing}. "
            f"Available columns include: {list(sample.columns)[:20]} ..."
        )

    out = sample[cols].copy()
    out = safe_numeric(out, [c for c in cols if c != KEYS.get('id')])
    out['source_population'] = population
    return out


def concat_and_deduplicate(dfs):
    if len(dfs) == 0:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    # print('len after concat =', len(out))
    # print('columns =', list(out.columns))

    idcol = KEYS.get('id')
    # print('idcol =', idcol)

    if idcol and idcol in out.columns:
        # print('nunique(id) =', out[idcol].nunique(dropna=False))
        # print(out[idcol].value_counts(dropna=False).head(10))

        # out2 = out.drop_duplicates(subset=idcol, keep='first').reset_index(drop=True)
        # print('len after drop_duplicates(subset=idcol) =', len(out2))
        return out
    else:
        # out2 = out.drop_duplicates().reset_index(drop=True)
        # print('len after full-row drop_duplicates =', len(out2))
        return out


def load_parent_sample(class_name: str,
                       cfg: Dict,
                       dfName: str = DFNAME,
                       Name: str = NAMECOL) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    cols = required_columns()

    if cfg.get('satellite_parent'):
        sat = extract_population_df(cfg['satellite_parent'], dfName=dfName, Name=Name, cols=cols)
    else:
        split_ref = cfg.get('satellite_split_ref', [])
        sat = concat_and_deduplicate([
            extract_population_df(pop, dfName=dfName, Name=Name, cols=cols)
            for pop in split_ref
        ])
    sat['sample_class'] = class_name
    sat['sample_kind'] = 'satellite'

    cen = None
    if cfg.get('central_parent'):
        try:
            cen = extract_population_df(cfg['central_parent'], dfName=dfName, Name=Name, cols=cols)
            cen['sample_class'] = class_name
            cen['sample_kind'] = 'central'
        except Exception as exc:
            print(f"[warning] Could not load central population for {class_name}: {exc}")
            cen = None

    return sat, cen


def clean_satellite_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keep = np.isfinite(out[KEYS['fdm']]) & np.isfinite(out[KEYS['dm_retained']]) & np.isfinite(out[KEYS['rmin']])
    if KEYS.get('zentry') and KEYS['zentry'] in out.columns:
        # allow zentry to be missing if you want the truth labels to use only 2 proxies
        pass
    out = out.loc[keep].reset_index(drop=True)
    return out


def compute_truth_boundaries(df: pd.DataFrame) -> Dict[str, float]:
    mode = TRUTH_CONFIG['mode']

    if mode == 'hybrid_quantile':
        q_low = TRUTH_CONFIG['q_low']
        q_high = TRUTH_CONFIG['q_high']
        bounds = {
            'dm_retained_low': float(df[KEYS['dm_retained']].quantile(q_low)),
            'dm_retained_high': float(df[KEYS['dm_retained']].quantile(q_high)),
            'rmin_low': float(df[KEYS['rmin']].quantile(q_low)),
            'rmin_high': float(df[KEYS['rmin']].quantile(q_high)),
        }

        if KEYS.get('zentry') and KEYS['zentry'] in df.columns:
            z = df[KEYS['zentry']].copy()
            z = z[np.isfinite(z)]
            if len(z) > 0:
                bounds['zentry_low'] = float(np.quantile(z, q_low))
                bounds['zentry_high'] = float(np.quantile(z, q_high))

        return bounds

    if mode == 'hybrid_fixed':
        return TRUTH_CONFIG['fixed'].copy()

    raise ValueError(f"Unknown TRUTH_CONFIG['mode'] = {mode}")


def assign_truth_labels(df: pd.DataFrame, bounds: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()

    strong_votes = np.zeros(len(out), dtype=int)
    weak_votes = np.zeros(len(out), dtype=int)

    # Proxy 1: retained DM mass fraction
    strong_votes += (out[KEYS['dm_retained']] <= bounds['dm_retained_low']).astype(int)
    weak_votes += (out[KEYS['dm_retained']] >= bounds['dm_retained_high']).astype(int)

    # Proxy 2: orbital depth / minimum pericentre
    strong_votes += (out[KEYS['rmin']] <= bounds['rmin_low']).astype(int)
    weak_votes += (out[KEYS['rmin']] >= bounds['rmin_high']).astype(int)

    # Proxy 3: entry epoch (optional)
    zkey = KEYS.get('zentry')
    if zkey and zkey in out.columns and ('zentry_low' in bounds) and ('zentry_high' in bounds):
        zvals = out[zkey]
        strong_votes += (zvals >= bounds['zentry_high']).astype(int)
        weak_votes += (zvals <= bounds['zentry_low']).astype(int)

    votes_required = TRUTH_CONFIG['votes_required']

    truth = np.full(len(out), 'ambiguous', dtype=object)
    truth[strong_votes >= votes_required] = 'DMpoor_truth'
    truth[weak_votes >= votes_required] = 'DMrich_truth'

    # If both conditions happen simultaneously, keep ambiguous
    both = (strong_votes >= votes_required) & (weak_votes >= votes_required)
    truth[both] = 'ambiguous'

    out['truth_label'] = truth
    out['truth_votes_strong'] = strong_votes
    out['truth_votes_weak'] = weak_votes
    return out


def predict_labels(df: pd.DataFrame, threshold: float) -> pd.Series:
    return pd.Series(
        np.where(df[KEYS['fdm']] < threshold, 'DMpoor_pred', 'DMrich_pred'),
        index=df.index,
        name='pred_label'
    )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum((~y_true) & y_pred))
    fn = int(np.sum(y_true & (~y_pred)))
    tn = int(np.sum((~y_true) & (~y_pred)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    contamination = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'purity': precision,
        'completeness': recall,
        'contamination': contamination,
        'f1': f1,
        'accuracy': accuracy,
    }


def evaluate_threshold(df_sat: pd.DataFrame,
                       threshold: float,
                       df_central: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    eval_df = df_sat[df_sat['truth_label'] != 'ambiguous'].copy()
    eval_df['pred_label'] = predict_labels(eval_df, threshold)

    true_poor = (eval_df['truth_label'].values == 'DMpoor_truth')
    pred_poor = (eval_df['pred_label'].values == 'DMpoor_pred')
    poor = binary_metrics(true_poor, pred_poor)

    true_rich = (eval_df['truth_label'].values == 'DMrich_truth')
    pred_rich = (eval_df['pred_label'].values == 'DMrich_pred')
    rich = binary_metrics(true_rich, pred_rich)

    out = {
        'threshold': float(threshold),
        'N_total_sat': int(len(df_sat)),
        'N_eval_sat': int(len(eval_df)),
        'N_truth_poor': int(np.sum(true_poor)),
        'N_truth_rich': int(np.sum(true_rich)),
        'N_ambiguous': int(np.sum(df_sat['truth_label'].values == 'ambiguous')),
        'poor_purity': poor['purity'],
        'poor_completeness': poor['completeness'],
        'poor_contamination': poor['contamination'],
        'poor_f1': poor['f1'],
        'poor_tp': poor['tp'],
        'poor_fp': poor['fp'],
        'poor_fn': poor['fn'],
        'poor_tn': poor['tn'],
        'rich_purity': rich['purity'],
        'rich_completeness': rich['completeness'],
        'rich_contamination': rich['contamination'],
        'rich_f1': rich['f1'],
        'rich_tp': rich['tp'],
        'rich_fp': rich['fp'],
        'rich_fn': rich['fn'],
        'rich_tn': rich['tn'],
    }

    if df_central is not None and len(df_central) > 0:
        cen = df_central.copy()
        cen = cen[np.isfinite(cen[KEYS['fdm']])].reset_index(drop=True)
        pred_cen_poor = (cen[KEYS['fdm']] < threshold)
        out['N_central'] = int(len(cen))
        out['central_false_positive_rate'] = float(np.nanmean(pred_cen_poor)) if len(cen) > 0 else np.nan
        out['central_false_positives'] = int(np.sum(pred_cen_poor))
    else:
        out['N_central'] = 0
        out['central_false_positive_rate'] = np.nan
        out['central_false_positives'] = np.nan

    return out


def evaluate_class_over_thresholds(class_name: str,
                                   df_sat: pd.DataFrame,
                                   df_central: Optional[pd.DataFrame],
                                   thresholds: np.ndarray = THRESHOLDS) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        row = evaluate_threshold(df_sat, float(t), df_central=df_central)
        row['sample_class'] = class_name
        rows.append(row)
    return pd.DataFrame(rows)


def build_threshold_table(curve_df: pd.DataFrame,
                          report_thresholds: List[float] = REPORT_THRESHOLDS) -> pd.DataFrame:
    rows = []
    for t in report_thresholds:
        sel = curve_df[np.isclose(curve_df['threshold'], t)].copy()
        if len(sel) == 0:
            continue
        rows.append(sel)
    if len(rows) == 0:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep_cols = [
        'sample_class', 'threshold',
        'N_total_sat', 'N_eval_sat', 'N_truth_poor', 'N_truth_rich', 'N_ambiguous',
        'poor_purity', 'poor_completeness', 'poor_contamination', 'poor_f1',
        'rich_purity', 'rich_completeness', 'rich_contamination', 'rich_f1',
        'central_false_positive_rate', 'central_false_positives', 'N_central',
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    return out[keep_cols].sort_values(['sample_class', 'threshold']).reset_index(drop=True)


def format_bounds(bounds: Dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in bounds.items()}])


def make_scatter_figure(data_by_class: Dict[str, Tuple[pd.DataFrame, Optional[pd.DataFrame]]],
                        outpath: str,
                        color_key: str = SCATTER_COLOR_KEY) -> None:
    nclass = len(data_by_class)
    fig, axes = plt.subplots(1, nclass, figsize=(6.2 * nclass, 5.2), squeeze=False)
    axes = axes[0]

    for ax, (class_name, (sat, cen)) in zip(axes, data_by_class.items()):
        if color_key == 'zentry' and KEYS.get('zentry') and KEYS['zentry'] in sat.columns:
            cvals = sat[KEYS['zentry']].values
            cbar_label = r'$z_{\rm entry}$'
        else:
            cvals = sat[KEYS['rmin']].values
            cbar_label = r'$(R/R_{200})_{\rm min}$'

        sc = ax.scatter(
            sat[KEYS['fdm']].values,
            sat[KEYS['dm_retained']].values,
            c=cvals,
            s=28,
            alpha=0.85,
            edgecolor='none'
        )

        if cen is not None and len(cen) > 0:
            ok = np.isfinite(cen[KEYS['fdm']]) & np.isfinite(cen[KEYS['dm_retained']])
            ax.scatter(
                cen.loc[ok, KEYS['fdm']].values,
                cen.loc[ok, KEYS['dm_retained']].values,
                marker='x',
                s=34,
                alpha=0.6,
                label='Centrals'
            )

        for t in SCATTER_THRESHOLDS:
            ax.axvline(t, ls='--', lw=1.2)

        ax.set_title(class_name)
        ax.set_xlabel(r'$f_{\rm DM,z=0}$')
        ax.set_ylabel(r'$M_{\rm DM}(z=0) / M_{\rm DM,max}$')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        if cen is not None and len(cen) > 0:
            ax.legend(frameon=False, loc='best')

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)

    fig.tight_layout()
    fig.savefig(outpath, dpi=250, bbox_inches='tight')
    plt.close(fig)


def make_curves_figure(curves_by_class: Dict[str, pd.DataFrame], outpath: str) -> None:
    classes = list(curves_by_class.keys())
    nclass = len(classes)
    fig, axes = plt.subplots(nclass, 2, figsize=(12.5, 4.2 * nclass), squeeze=False, sharex=True)

    for i, class_name in enumerate(classes):
        df = curves_by_class[class_name].sort_values('threshold')

        ax = axes[i, 0]
        ax.plot(df['threshold'], df['poor_purity'], lw=2.2, label='purity')
        ax.plot(df['threshold'], df['poor_completeness'], lw=2.2, label='completeness')
        ax.plot(df['threshold'], df['poor_contamination'], lw=1.7, ls='--', label='contamination')
        if 'central_false_positive_rate' in df.columns:
            ax.plot(df['threshold'], df['central_false_positive_rate'], lw=1.7, ls=':', label='central FPR')
        for t in REPORT_THRESHOLDS:
            ax.axvline(t, ls='--', lw=1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(class_name)
        ax.set_title('DM-poor metrics')
        ax.legend(frameon=False, fontsize=9)

        ax = axes[i, 1]
        ax.plot(df['threshold'], df['rich_purity'], lw=2.2, label='purity')
        ax.plot(df['threshold'], df['rich_completeness'], lw=2.2, label='completeness')
        ax.plot(df['threshold'], df['rich_contamination'], lw=1.7, ls='--', label='contamination')
        for t in REPORT_THRESHOLDS:
            ax.axvline(t, ls='--', lw=1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_title('DM-rich metrics')
        ax.legend(frameon=False, fontsize=9)

    for ax in axes[-1, :]:
        ax.set_xlabel(r'$f_{\rm DM,z=0}$ threshold')

    fig.tight_layout()
    fig.savefig(outpath, dpi=250, bbox_inches='tight')
    plt.close(fig)


def save_truth_labeled_catalog(df: pd.DataFrame, outpath: str) -> None:
    keep = [
        c for c in [
            KEYS.get('id'), KEYS['fdm'], KEYS['dm_retained'], KEYS['rmin'], KEYS.get('zentry'),
            'sample_class', 'sample_kind', 'truth_label', 'truth_votes_strong', 'truth_votes_weak'
        ] if c is not None and c in df.columns
    ]
    df[keep].to_csv(outpath, index=False)

#%%
def main() -> None:
    ensure_outdir(OUTDIR)

    all_curves = []
    all_bounds = []
    scatter_inputs = {}
    truth_catalogs = []

    for class_name, cfg in POP_CONFIG.items():
        print(f"\n=== {class_name} ===")
        sat, cen = load_parent_sample(class_name, cfg, dfName=DFNAME, Name=NAMECOL)
        sat = clean_satellite_df(sat)

        bounds = compute_truth_boundaries(sat)
        sat = assign_truth_labels(sat, bounds)

        if cen is not None:
            cen = clean_satellite_df(cen)

        # Save class-level truth-labeled catalogs
        truth_out = os.path.join(OUTDIR, f'{class_name}_truth_catalog.csv')
        save_truth_labeled_catalog(sat, truth_out)
        truth_catalogs.append(sat)

        # Save bounds used for truth construction
        bdf = format_bounds(bounds)
        bdf['sample_class'] = class_name
        all_bounds.append(bdf)
        bdf.to_csv(os.path.join(OUTDIR, f'{class_name}_truth_bounds.csv'), index=False)

        # Evaluate threshold sweep
        curve = evaluate_class_over_thresholds(class_name, sat, cen, thresholds=THRESHOLDS)
        curve.to_csv(os.path.join(OUTDIR, f'{class_name}_threshold_curve.csv'), index=False)
        all_curves.append(curve)

        # Small report table for 0.5, 0.6, 0.7, 0.8
        report = build_threshold_table(curve, REPORT_THRESHOLDS)
        report.to_csv(os.path.join(OUTDIR, f'{class_name}_threshold_report.csv'), index=False)
        print(report.to_string(index=False))

        scatter_inputs[class_name] = (sat, cen)

    # Combine outputs across all classes
    if len(all_curves) > 0:
        curves_all = pd.concat(all_curves, ignore_index=True)
        curves_all.to_csv(os.path.join(OUTDIR, 'ALL_threshold_curves.csv'), index=False)

        report_all = build_threshold_table(curves_all, REPORT_THRESHOLDS)
        report_all.to_csv(os.path.join(OUTDIR, 'ALL_threshold_report.csv'), index=False)

        curves_by_class = {k: curves_all[curves_all['sample_class'] == k].copy() for k in POP_CONFIG.keys()}
        make_curves_figure(curves_by_class, os.path.join(OUTDIR, 'fdm_purity_completeness_curves.png'))

    if len(all_bounds) > 0:
        pd.concat(all_bounds, ignore_index=True).to_csv(os.path.join(OUTDIR, 'ALL_truth_bounds.csv'), index=False)

    if len(scatter_inputs) > 0:
        make_scatter_figure(scatter_inputs, os.path.join(OUTDIR, 'fdm_vs_dmretained_scatter.png'))

    if len(truth_catalogs) > 0:
        pd.concat(truth_catalogs, ignore_index=True).to_csv(
            os.path.join(OUTDIR, 'ALL_truth_catalogs.csv'), index=False
        )

    print("\nDone. Outputs written to:")
    print(os.path.abspath(OUTDIR))


if __name__ == '__main__':
    main()

#%%
import pandas as pd
import matplotlib.pyplot as plt

base = "/home/abhner/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/TESTs/fdm_threshold_calibration"
curves_all = pd.read_csv(f"{base}/ALL_threshold_curves.csv")

curves_by_class = {
    k: curves_all[curves_all["sample_class"] == k].copy()
    for k in ["Normals", "CompactsMB", "CompactsSB"]
}

#%%
REPORT_THRESHOLDS = [0.50, 0.60, 0.70, 0.80]

def plot_fdm_purity_completeness_curves(curves_by_class,
                                        report_thresholds=REPORT_THRESHOLDS,
                                        figsize_per_row=(12.5, 4.2),
                                        sharex=True):
    classes = list(curves_by_class.keys())
    nclass = len(classes)

    fig, axes = plt.subplots(
        nclass, 2,
        figsize=(figsize_per_row[0], figsize_per_row[1] * nclass),
        squeeze=False,
        sharex=sharex
    )

    for i, class_name in enumerate(classes):
        df = curves_by_class[class_name].sort_values("threshold")

        ax = axes[i, 0]
        ax.plot(df["threshold"], df["poor_purity"], lw=2.2, label="purity")
        ax.plot(df["threshold"], df["poor_completeness"], lw=2.2, label="completeness")
        ax.plot(df["threshold"], df["poor_contamination"], lw=1.7, ls="--", label="contamination")
        if "central_false_positive_rate" in df.columns:
            ax.plot(df["threshold"], df["central_false_positive_rate"], lw=1.7, ls=":", label="central FPR")
        for t in report_thresholds:
            ax.axvline(t, ls="--", lw=1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(class_name)
        ax.set_title("DM-poor metrics")
        ax.legend(frameon=False, fontsize=9)

        ax = axes[i, 1]
        ax.plot(df["threshold"], df["rich_purity"], lw=2.2, label="purity")
        ax.plot(df["threshold"], df["rich_completeness"], lw=2.2, label="completeness")
        ax.plot(df["threshold"], df["rich_contamination"], lw=1.7, ls="--", label="contamination")
        for t in report_thresholds:
            ax.axvline(t, ls="--", lw=1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_title("DM-rich metrics")
        ax.legend(frameon=False, fontsize=9)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$f_{\rm DM,z=0}$ threshold")

    fig.tight_layout()
    return fig, axes

#%%
fig, axes = plot_fdm_purity_completeness_curves(curves_by_class)
plt.show()

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = "/home/abhner/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/TESTs/fdm_threshold_calibration"
df = pd.read_csv(f"{base}/ALL_threshold_curves.csv")

def plot_metric_heatmaps(df, classes=("Normals", "CompactsMB", "CompactsSB")):
    metrics_poor = ["poor_purity", "poor_completeness", "poor_contamination", "poor_f1"]
    metrics_rich = ["rich_purity", "rich_completeness", "rich_contamination", "rich_f1"]
    metric_labels = ["Purity", "Completeness", "Contamination", "F1"]

    thresholds = np.sort(df["threshold"].unique())
    nrows = len(classes)

    fig, axes = plt.subplots(
        nrows, 2,
        figsize=(10, 2.4*nrows),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    vmin, vmax = 0.0, 1.0

    for i, cls in enumerate(classes):
        dcls = df[df["sample_class"] == cls].sort_values("threshold")

        arr_poor = np.array([dcls[m].values for m in metrics_poor], dtype=float)
        arr_rich = np.array([dcls[m].values for m in metrics_rich], dtype=float)

        im0 = axes[i, 0].imshow(
            arr_poor,
            origin="lower",
            aspect="auto",
            vmin=vmin, vmax=vmax,
            extent=[thresholds.min(), thresholds.max(), -0.5, len(metric_labels)-0.5]
        )
        im1 = axes[i, 1].imshow(
            arr_rich,
            origin="lower",
            aspect="auto",
            vmin=vmin, vmax=vmax,
            extent=[thresholds.min(), thresholds.max(), -0.5, len(metric_labels)-0.5]
        )

        axes[i, 0].set_yticks(range(len(metric_labels)))
        axes[i, 0].set_yticklabels(metric_labels)
        axes[i, 0].set_ylabel(cls)

        axes[i, 0].set_title("DM-poor")
        axes[i, 1].set_title("DM-rich")

        for ax in axes[i]:
            ax.axvline(0.7, lw=2)
            for t in [0.5, 0.6, 0.8]:
                ax.axvline(t, lw=0.8, ls="--", alpha=0.7)

    axes[-1, 0].set_xlabel(r"$f_{\rm DM,z=0}$ threshold")
    axes[-1, 1].set_xlabel(r"$f_{\rm DM,z=0}$ threshold")

    fig.subplots_adjust(wspace=0.03, hspace=0.05, right=0.88)
    cax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
    fig.colorbar(im1, cax=cax, label="Metric value")

    return fig, axes


#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_fdm_vs_dmretained_glued(samples_dict,
                                 color_key="rOverR200Min",
                                 color_label=r"$(R/R_{200})_{\min}$",
                                 xkey="fDM_z0",
                                 ykey="DMz0_over_DMmax",
                                 class_order=("Normals", "CompactsMB", "CompactsSB"),
                                 bins=35):
    fig, axes = plt.subplots(
        len(class_order), 1,
        figsize=(6.2, 8.5),
        sharex=True,
        sharey=True,
        squeeze=False
    )
    axes = axes[:, 0]

    all_c = []
    for cls in class_order:
        d = samples_dict[cls]
        all_c.append(np.asarray(d[color_key], dtype=float))
    all_c = np.concatenate(all_c)
    vmin, vmax = np.nanmin(all_c), np.nanmax(all_c)

    mappable = None

    for i, cls in enumerate(class_order):
        d = samples_dict[cls].copy()

        x = np.asarray(d[xkey], dtype=float)
        y = np.asarray(d[ykey], dtype=float)
        c = np.asarray(d[color_key], dtype=float)

        # fundo 2D histogram
        axes[i].hist2d(x, y, bins=bins, cmin=1)

        # pontos com colormap
        sc = axes[i].scatter(
            x, y,
            c=c,
            s=18,
            vmin=vmin, vmax=vmax
        )
        mappable = sc

        axes[i].axvline(0.7, lw=2)
        axes[i].text(0.02, 0.92, cls, transform=axes[i].transAxes)

        axes[i].set_ylabel(r"$M_{\rm DM}(z=0)/M_{\rm DM,max}$")

    axes[-1].set_xlabel(r"$f_{\rm DM,z=0}$")

    fig.subplots_adjust(hspace=0.02, right=0.88)
    cax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
    fig.colorbar(mappable, cax=cax, label=color_label)

    return fig, axes

samples_dict = {
    "Normals": df_normals_sat,
    "CompactsMB": df_mb_sat,
    "CompactsSB": df_sb_sat,
}

#%%

def plot_purity_completeness_plane(df, classes=("Normals", "CompactsMB", "CompactsSB")):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

    for cls in classes:
        d = df[df["sample_class"] == cls].sort_values("threshold")

        sc0 = axes[0].scatter(
            d["poor_completeness"], d["poor_purity"],
            c=d["threshold"], s=70, label=cls
        )
        axes[0].plot(d["poor_completeness"], d["poor_purity"], lw=1.3)

        sc1 = axes[1].scatter(
            d["rich_completeness"], d["rich_purity"],
            c=d["threshold"], s=70, label=cls
        )
        axes[1].plot(d["rich_completeness"], d["rich_purity"], lw=1.3)

    axes[0].set_title("DM-poor")
    axes[1].set_title("DM-rich")

    for ax in axes:
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Completeness")
        ax.set_ylabel("Purity")
        ax.legend(frameon=False)

    cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label(r"$f_{\rm DM,z=0}$ threshold")

    fig.tight_layout()
    return fig, axes

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = "/home/abhner/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/TESTs/fdm_threshold_calibration"
df = pd.read_csv(f"{base}/ALL_threshold_curves.csv")

df.head()

#%%
def plot_fdm_purity_completeness_curves_clean(
    df,
    rows=("Normals", "CompactsMB", "CompactsSB"),
    columns=("DM-poor", "DM-rich"),
    cNum=4.8,
    lNum=3.0,
    threshold_ref=0.7,
    threshold_marks=(0.5, 0.6, 0.8),
):
    metric_map = {
        "DM-poor": [
            ("poor_purity", "Purity", "tab:blue", "-"),
            ("poor_completeness", "Completeness", "#00cc66", "-"),
            ("poor_contamination", "Contamination", "orange", "--"),
            ("poor_f1", "F1", "orangered", ":"),
        ],
        "DM-rich": [
            ("rich_purity", "Purity", "tab:blue", "-"),
            ("rich_completeness", "Completeness", "#00cc66", "-"),
            ("rich_contamination", "Contamination", "orange", "--"),
            ("rich_f1", "F1", "orangered", ":"),
        ],
    }

    plt.rcParams.update({"figure.figsize": (cNum * len(columns), lNum * len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex="col", sharey="row")

    drawstyle = "steps-mid"

    for i, row in enumerate(rows):
        d = df[df["sample_class"] == row].sort_values("threshold")
        x = d["threshold"].values

        for j, col in enumerate(columns):
            ax = axs[i, j]

            for t in threshold_marks:
                ax.axvline(t, color="deepskyblue", lw=1.1, ls="--", alpha=0.9)
            ax.axvline(threshold_ref, color="dodgerblue", lw=2.2)

            for name, lab, color, ls in metric_map[col]:
                ax.plot(
                    x, d[name].values,
                    color=color, ls=ls, lw=2.2,
                    drawstyle=drawstyle,
                    label=lab
                )

            ax.set_ylim(0, 1.05)
            ax.minorticks_on()
            ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=12)

            if i == 0:
                ax.set_title(col, fontsize=26, pad=10)

            if j > 0:
                ax.tick_params(labelleft=False)
            if i < len(rows)-1:
                ax.tick_params(labelbottom=False)

    axs[0, 0].legend(frameon=False, fontsize=11, loc="upper left")

    axs[-1, 0].set_xlabel(r"$f_{\rm DM,z=0}$ threshold", fontsize=18)
    axs[-1, 1].set_xlabel(r"$f_{\rm DM,z=0}$ threshold", fontsize=18)

    # rótulos externos das linhas
    y_positions = [0.83, 0.50, 0.18]
    for y, row in zip(y_positions, rows):
        fig.text(0.03, y, row, rotation=90, va="center", ha="center", fontsize=22)

    return fig, axs

#%%

base = "/home/abhner/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/TESTs/fdm_threshold_calibration"
df = pd.read_csv(f"{base}/ALL_threshold_curves.csv")

fig, axs = plot_fdm_purity_completeness_curves_clean(df)
plt.show()