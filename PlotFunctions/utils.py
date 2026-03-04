"""Small utilities shared by plotting functions."""

from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

import warnings

from style_registry import (markers, msize, colors, edgecolors)

from matplotlib.collections import LineCollection


logger = logging.getLogger(__name__)


def configure_matplotlib(style_path: str | None = None, *, ignore_warnings: bool = False) -> None:
    """Configure Matplotlib.

    Parameters
    ----------
    style_path
        Path to a .mplstyle file. If None, no style is applied.
    ignore_warnings
        If True, suppresses matplotlib warnings ONLY. Prefer leaving this False.
    """
    if style_path:
        try:
            plt.style.use(style_path)
            logger.info("Matplotlib style applied: %s", style_path)
        except OSError as e:
            raise OSError(f"Could not load matplotlib style: {style_path}") from e

    if ignore_warnings:
        import warnings
        warnings.filterwarnings("ignore", module="matplotlib")
        
def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def ensure_parent_dir(path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    


def _get_point_style_for_colorbar(names_l, msizet, linewidth):
    """Return (marker, edgecolor, size, lw) for COLORBAR scatter cases."""
    marker = markers(names_l + "Colorbar")
    edgecolor = edgecolors(names_l + "Colorbar")
    size = msizet * msize(names_l + "Colorbar")
    lw = 1.5 * linewidth
    return marker, edgecolor, size, lw


def _scatter_with_colorbar(
    ax,
    x, y,
    color_values,
    colorbar_key,
    names_l,
    cmap_name,
    alpha_scatter,
    linewidth,
    msizet,
    HIGHLIGHTPoints=False,
):
    """
    Apply all special COLORBAR rules and return the PathCollection (sc),
    plus any 'norm' that needs to be reused for the final colorbar.
    """

    # Default outputs
    norm_for_colorbar = None

    # base style
    marker, edgecolor, size, lw = _get_point_style_for_colorbar(
        names_l, msizet, linewidth
    )

    # Base cmap object
    cmap = plt.cm.get_cmap(cmap_name)

    # ---- 1) Group of keys with the same settings (your first `if COLORBAR[0] in [...]`) ----
    if colorbar_key in [
        "DecreaseBetweenGasStar_Over_starFinal",
        "SubhaloMassType4",
        "MassAboveAfter_over_In_Lost",
        "MassInAfterInfall_Lost",
    ]:
        sc = ax.scatter(
            x, y, c=color_values,
            edgecolor=edgecolor, alpha=alpha_scatter,
            lw=lw, marker=marker, s=7 * size,
            cmap=cmap,
        )
        return sc, norm_for_colorbar

    # ---- 2) LogNorm cases ----
    if colorbar_key == "sSFRRatio_ETOGAS_Before":
        norm_for_colorbar = mpl.colors.LogNorm(vmin=0.005, vmax=1.8)
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l),
                        alpha=alpha_scatter, lw=lw, marker=markers(names_l),
                        s=7 * (msizet * msize.get(names_l, msizet)),
                        cmap=cmap, norm=norm_for_colorbar)
        return sc, norm_for_colorbar

    if colorbar_key == "TimeInnerRegion":
        norm_for_colorbar = mpl.colors.LogNorm(vmin=0.005, vmax=13)
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l),
                        alpha=alpha_scatter, lw=lw, marker=markers(names_l),
                        s=7 * (msizet * msize.get(names_l, msizet)),
                        cmap=cmap, norm=norm_for_colorbar)
        return sc, norm_for_colorbar

    if colorbar_key == "RatioSFR":
        norm_for_colorbar = mpl.colors.LogNorm(vmin=0.005, vmax=1.1)
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l),
                        alpha=alpha_scatter, lw=lw, marker=markers(names_l),
                        s=7 * (msizet * msize.get(names_l, msizet)),
                        cmap=cmap, norm=norm_for_colorbar)
        return sc, norm_for_colorbar

    if colorbar_key == "rOrbMean_Entry_to_Gas":
        norm_for_colorbar = mpl.colors.LogNorm(vmin=0.05, vmax=0.7)
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, norm=norm_for_colorbar)
        return sc, norm_for_colorbar

    if colorbar_key in ["deltaInnersSFR_afterEntry_all", "deltaTrueInnersSFR_afterEntry_all"]:
        norm_for_colorbar = mpl.colors.LogNorm(vmin=0.05, vmax=1.2)
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, norm=norm_for_colorbar)
        return sc, norm_for_colorbar

    if colorbar_key == "MDM_Norm_Max_99":
        norm_for_colorbar = mpl.colors.LogNorm(vmin=0.005, vmax=0.2)
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=1.1 * linewidth, marker=marker, s=5 * size,
                        cmap=cmap, norm=norm_for_colorbar)
        return sc, norm_for_colorbar

    if colorbar_key == "z_At_FirstEntry":
        norm_for_colorbar = mpl.colors.LogNorm(vmin=0.2, vmax=3)
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors.get(names_l, None),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=size,
                        cmap=cmap, norm=norm_for_colorbar)
        return sc, norm_for_colorbar

    if colorbar_key == "deltaInnersSFR_afterEntry_all_EntryRh":
        norm_for_colorbar = mpl.colors.LogNorm(vmin=0.5, vmax=3)
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, norm=norm_for_colorbar)
        return sc, norm_for_colorbar

    # ---- 3) Fixed discrete cmap cases ----
    if colorbar_key == "MassInAfterInfall":
        cmap = plt.cm.get_cmap(cmap_name, 2)  # keep exactly your behavior
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap)
        return sc, norm_for_colorbar

    if colorbar_key == "U-r":
        cmap = plt.cm.get_cmap(cmap_name, 2)
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=-0.3, vmax=2.1)
        return sc, norm_for_colorbar

    # ---- 4) Simple vmin/vmax cases ----
    if colorbar_key == "SigmaIn":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=1.0 * linewidth, marker=marker, s=4 * size,
                        cmap=cmap, vmin=40, vmax=85)
        return sc, norm_for_colorbar

    if colorbar_key == "logMstar_Entry":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l),
                        alpha=alpha_scatter, lw=lw, marker=markers(names_l),
                        s=7 * (msizet * msize.get(names_l, msizet)),
                        cmap=cmap)
        return sc, norm_for_colorbar

    if colorbar_key == "sSFRTrueRatio_Entry_to_Nogas":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=0.8, vmax=1.7)
        return sc, norm_for_colorbar

    if colorbar_key == "SnapLastMerger_Before_Entry":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=0, vmax=14)
        return sc, norm_for_colorbar

    if colorbar_key == "TimeSinceQuenching":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=0, vmax=6)
        return sc, norm_for_colorbar

    if colorbar_key == "DeltasSFRRatio_Entry_to_Nogas":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=-2, vmax=3)
        return sc, norm_for_colorbar

    if colorbar_key == "deltaFirst_to_Final":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=0, vmax=10)
        return sc, norm_for_colorbar

    if colorbar_key == "MassStarIn_Over_Above_absolutevalue":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=0, vmax=2)
        return sc, norm_for_colorbar

    if colorbar_key == "GasjInflow_BeforeEntry":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=2, vmax=4)
        return sc, norm_for_colorbar

    if colorbar_key == "FracStarAfterEntry_Inner":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=size,
                        cmap=cmap, vmin=0.1, vmax=0.55)
        return sc, norm_for_colorbar

    if colorbar_key == "FracNew_Loss":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=size,
                        cmap=cmap, vmin=0, vmax=0.8)
        return sc, norm_for_colorbar

    if colorbar_key == "dSize_NoGas_Final":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=-0.006, vmax=0.01)
        return sc, norm_for_colorbar

    if colorbar_key == "DMFrac_99":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=0.3, vmax=0.9)
        return sc, norm_for_colorbar

    if colorbar_key == "sSFRRatio_Entry_to_Nogas":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=0.3, vmax=3.4)
        return sc, norm_for_colorbar

    if colorbar_key == "GasAbove_Entry_to_Nogas":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=-0.3, vmax=-0.05)
        return sc, norm_for_colorbar

    if colorbar_key == "sSFRInner_Entry_to_Nogas":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=-10.8, vmax=-9.1)
        return sc, norm_for_colorbar

    if colorbar_key == "dSize_Max_to_Nogas":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=-1, vmax=0.05)
        return sc, norm_for_colorbar

    if colorbar_key == "dSize_Entry_to_Max":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=-1.2, vmax=0.3)
        return sc, norm_for_colorbar

    if colorbar_key == "MassAboveAfterInfall_Lost":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=1.1 * linewidth, marker=marker, s=size,
                        cmap=cmap, vmin=-0.66, vmax=0.0)
 
        return sc, norm_for_colorbar

    if colorbar_key == "TimeLossGass":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=0, vmax=12)
        return sc, norm_for_colorbar

    if colorbar_key == "StarMass_GasLoss_Over_EntryToGas":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, vmin=0, vmax=1)
        return sc, norm_for_colorbar

    if colorbar_key == "logHalfRadstar_99":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=1.0 * linewidth, marker=marker, s=size,
                        cmap=cmap, vmin=0.035, vmax=0.85)
        return sc, norm_for_colorbar

    if colorbar_key == "logSUM_Mstar_merger_Corotate":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=size,
                        cmap=cmap, vmin=5, vmax=8.5)
        return sc, norm_for_colorbar

    if colorbar_key == "fEx_at_99":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=size,
                        cmap=cmap, vmin=0.002, vmax=0.1)
        return sc, norm_for_colorbar

    if colorbar_key == "M200Mean":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=size,
                        cmap=cmap, vmin=11.7, vmax=13.3)
        return sc, norm_for_colorbar

    if colorbar_key == "rOverR200Min":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=size,
                        cmap=cmap, vmin=0.002, vmax=0.2)
        return sc, norm_for_colorbar

    if colorbar_key == "Norbit_Entry_To_NoGas":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=size,
                        cmap=cmap)
        return sc, norm_for_colorbar

    if colorbar_key == "SnapLostGas":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=1.1 * linewidth, marker=marker, s=size,
                        cmap=cmap, vmin=0, vmax=14)
        return sc, norm_for_colorbar

    if colorbar_key == "deltaRatio_gastoend_entrytogas":
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=size,
                        cmap=cmap, vmin=0, vmax=5)
        return sc, norm_for_colorbar

    # ---- 5) BoundaryNorm / special norms ----
    if colorbar_key == "deltaInnersSFR_afterEntry":
        bounds = [0.9, 1.0, 1.3]
        norm_for_colorbar = mpl.colors.BoundaryNorm(bounds, cmap.N)
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, norm=norm_for_colorbar)
        return sc, norm_for_colorbar

    if colorbar_key == "DeltasSFR_Ratio":
        norm_for_colorbar = mpl.colors.BoundaryNorm([-1.2, -1, 0, 1, 2, 4], cmap.N)
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolor,
                        alpha=alpha_scatter, lw=lw, marker=marker, s=7 * size,
                        cmap=cmap, norm=norm_for_colorbar)
        return sc, norm_for_colorbar

    if colorbar_key == "NCorotateMergers":
        cmap = plt.cm.get_cmap("Blues", 4)
        color_values = color_values.astype(float)
        color_values[color_values == 0] = np.nan
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm_for_colorbar = mcolors.BoundaryNorm(bounds, cmap.N)
        sc = ax.scatter(x, y, c=color_values, edgecolor=edgecolors(names_l + "Colorbar"),
                        alpha=alpha_scatter, lw=lw, marker=marker, s=size,
                        cmap=cmap, norm=norm_for_colorbar)
        return sc, norm_for_colorbar

    # ---- 6) Your ThresholdNormalize cases (keep as-is, just moved here) ----
    if colorbar_key == "sSFRRatioPericenter":
        class ThresholdNormalize(mcolors.Normalize):
            def __init__(self, vmin=None, vmax=None, threshold=1.0, **kwargs):
                super().__init__(vmin, vmax, **kwargs)
                self.threshold = threshold

            def __call__(self, value, clip=None):
                value = np.asarray(value)
                out = super().__call__(value, clip)
                out[value > self.threshold] = 1.0
                return out

        norm_for_colorbar = ThresholdNormalize(vmin=0, vmax=2, threshold=1)
        color_values = color_values.astype(float)
        

        sc = ax.scatter(
            x, y, c=color_values,
            edgecolor=edgecolor, alpha=alpha_scatter,
            lw=1.2 * linewidth,
            marker=marker,
            s=4 * size,
            cmap=cmap, norm=norm_for_colorbar,
        )
        return sc, norm_for_colorbar

    if colorbar_key in ["logStarZ_99", "logStarZ_99_75dex"]:
        class ThresholdNormalize(mcolors.Normalize):
            def __init__(self, vmin=None, vmax=None, threshold=1.0, **kwargs):
                super().__init__(vmin, vmax, **kwargs)
                self.threshold = threshold

            def __call__(self, value, clip=None):
                value = np.asarray(value)
                out = super().__call__(value, clip)
                out[value > self.threshold] = 1.0
                return out

        if colorbar_key == "logStarZ_99":
            norm_for_colorbar = ThresholdNormalize(vmin=0, vmax=0.6, threshold=0.3)
        else:
            norm_for_colorbar = ThresholdNormalize(vmin=-0.75, vmax=0.6 - 0.75, threshold=0.3 - 0.75)

        sc = ax.scatter(
            x, y, c=color_values,
            edgecolor=edgecolor,
            alpha=alpha_scatter,
            lw=1.1 * linewidth if colorbar_key == "logStarZ_99" else 1.5 * linewidth,
            marker=marker,
            s=5 * size if colorbar_key == "logStarZ_99" else 7 * size,
            cmap=cmap, norm=norm_for_colorbar,
        )

        if (colorbar_key == "logStarZ_99") and HIGHLIGHTPoints:
            ax.scatter(8.994623, 0.088216, marker="*", color="blue", edgecolor="forestgreen", s=350)
            ax.scatter(9.157864, 0.321646, marker="*", color="black", edgecolor="black", s=50)
            ax.scatter(9.066031, 0.272490, marker="*", color="black", edgecolor="black", s=50)

        return sc, norm_for_colorbar

    # ---- 7) Fallback: your default branch (vmin=min, vmax=max) ----
    # NOTE: protect against all-NaN / empty arrays to avoid ValueError
    finite = np.isfinite(color_values)
    vmin = np.nanmin(color_values[finite]) if np.any(finite) else None
    vmax = np.nanmax(color_values[finite]) if np.any(finite) else None

    sc = ax.scatter(
        x, y, c=color_values,
        edgecolor=edgecolors.get(names_l + "Colorbar", None),
        alpha=alpha_scatter, lw=lw, marker=marker,
        s=size, cmap=cmap,
        vmin=vmin, vmax=vmax,
    )
    return sc, norm_for_colorbar
