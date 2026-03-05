"""
PlotFunctions
=============

Collection of plotting utilities used for the analysis of galaxy
properties in cosmological simulations and observational datasets.

These functions provide standardized plotting styles, color maps,
markers, and helper routines for generating figures used in scientific
analysis and publications.

Main features
-------------
- Consistent plotting styles for galaxy categories
- Standard color/marker/linestyle dictionaries
- Utilities for multi-panel and evolution plots
- Helper functions for statistical visualization

Author
------
Abhner P. de Almeida
abhner.almeida AT usp.br
University of São Paulo (USP)

Created for research on dwarf galaxies and galaxy evolution.
"""

from __future__ import annotations

import numpy as np
import os
import pandas as pd

np.seterr(divide="ignore")


import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Patch 
from matplotlib.lines import Line2D

from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FixedLocator
from matplotlib.ticker import FixedFormatter
from matplotlib.offsetbox import AnchoredText

from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from scipy import stats
from scipy.stats import spearmanr


from style_registry import (markers, msize, colors, edgecolors, 
                            titles, linesthicker, lines, capstyles,
                            scales, labelsequal, labels, texts)

from utils import colored_line, _scatter_with_colorbar

import sys
sys.path.append(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/analyzes")
sys.path.append(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/analyzes/GaryScripts")



# Project-specific dependencies
try:
    import TNGFunctions as TNG
    import ExtractTNG as ETNG
    import MATH

except Exception as e: 
    raise ImportError(
        "Missing project-specific modules (ExtractTNG, TNGFunctions, MATH).\n"
        "Original error: " + repr(e)
    ) from e


#Constants
Omegam0 = 0.3089
h = 0.6774

#Paths
SaveSubhaloPath = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory/'
SIMTNG = 'TNG50'
Nsim = '-1'
dfTime = pd.read_csv(os.getenv("HOME")+'/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/utils/SNAPS_TIME.csv')


def format_func_loglog(value, tick_number):
    '''
    change label in log plots
    Parameters
    ----------
    value : label values.
    tick_number : number of tickers
    Returns
    -------
    Requested label
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    if value == 0:
        return str(0)
    sign = value/abs(value)
    N = int(np.round(np.log10(abs(value))))

    if abs(N) < 2:
        string = 10**N
        if sign*string >= 1:
            return str(int(sign*string))
        else:
            return str(sign*string)
    elif abs(N) >= 2:
        N = N*sign
        label = ('$10^{%4.0f}$ ' % N)
        return label


def Legend(names, mult = 2, msizeMult= .6, linewidth = 1.5):
    '''
    make the legend
    Parameters
    ----------
    names : name for the legend. 
    Returns
    -------
    lines, labels, number of columns and fontsize multiplicative factor for the legend
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    custom_lines = []
    label = []
    for name in names: 
        
        if 'Scatter' in name or name == 'Bian et al. (2025)':
            name = name.replace('Scatter', '')
            
            BlackLine = False
            Empty = False

            if 'Legend' in name:
                name = name.replace('Legend', '')
                lw = lwe = 2
            
            elif 'BlackLine' in name:
                name = name.replace('BlackLine', '')
                BlackLine = True
                lwe = 0.8
                lw = 0
                
                if 'Empty' in name:
                    name = name.replace('Empty', '')
                    Empty = True
                    
            elif 'Empty' in name or 'Colorbar' in name:
                Empty = True
                lw = 0
                lwe = 1.5

                
            else:
                lw = lwe = 0
                
                Empty = False
                BlackLine = False
            if name == 'Bian et al. (2025)':
                custom_lines.append(Patch(facecolor='tab:red', alpha = 0.4))
            else:
                if 'Normal' in name:
                    msizeMult = 1.8
                    
                if 'SBC' in name or 'MBC' in name  or 'Diffuse' in name:
                    msizeMult = 0.8
                if 'LoseTheirGas' in name:
                    msizeMult = 0.8
                    
                if Empty:
                    custom_lines.append(
                    Line2D([0], [0], color='white', lw=lw, marker=markers(name),  markeredgewidth = 1.,
                           markersize = msizeMult*msize(name), markeredgecolor = edgecolors(name)))
                    
                elif BlackLine:
                    if Empty:
                        custom_lines.append(
                        Line2D([0], [0], color='white', lw=lw, marker=markers(name),  markeredgewidth = lwe,
                               markersize = msizeMult*msize(name), markeredgecolor = 'k'))
                    else:
                        custom_lines.append(
                        Line2D([0], [0], color=colors(name), lw=lw, marker=markers(name),  markeredgewidth = lwe,
                               markersize = msizeMult*msize(name), markeredgecolor = 'k'))
                else:
                    custom_lines.append(
                    Line2D([0], [0], color=colors(name), lw=lw, marker=markers(name),  markeredgewidth = lwe,
                           markersize = msizeMult*msize(name), markeredgecolor = edgecolors(name)))
            label.append(titles(name))

        elif name == 'None':
            custom_lines.append(Line2D([0], [0], lw=0))
            label.append('')
            
        elif 'IDsColumn' in name and 'RadType' in name:
            name = name.replace('IDsColumn', '')
            custom_lines.append(Line2D([0], [0], color=colors(name),
                                       ls= 'solid', lw=mult * 0.5* linesthicker(name), 
                                       dash_capstyle = capstyles(name)))
            label.append(titles(name))
        
        else:
            
            custom_lines.append(Line2D([0], [0], color=colors(name), ls=lines(name), 
                                       lw=mult * 0.5* linesthicker(name), dash_capstyle = capstyles(name)))
            label.append(titles(name))

    if len(names) < 4:
        ncol = 1
        mult = 0.6
    elif len(names) <= 8:
        ncol = 2
        mult = 0.6
    else:
        ncol = 3
        mult = 0.5
    return custom_lines, label, ncol, mult


def savefig(savepath, savefigname, TRANSPARENT = True, SIM = SIMTNG):
    '''
    save figures
    Parameters
    ----------
    savepath : save path. 
    savefigname : fig name.
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    pathBase =  os.getenv("HOME")+'/TNG_Analyzes/Figs/' + SIM + '/'
    try:
        plt.savefig(pathBase + savepath+'/'+savefigname +
                    '.pdf', bbox_inches='tight')

        plt.savefig(pathBase + savepath+'/'+'PNG'+'/'+savefigname +
                        '.png', bbox_inches='tight', transparent=TRANSPARENT, dpi=400)
        
    except:
        directories = savepath.split('/')
        directories.append('PNG')
        path = pathBase
        for name in directories:
            path = os.path.join(path, name)
            if not os.path.isdir(path):
                os.mkdir(path)
        plt.savefig(pathBase + savepath +  '/'+savefigname +
                    '.pdf', bbox_inches='tight')

       
        plt.savefig(path + '/'+savefigname +
                        '.png', bbox_inches='tight',  transparent=TRANSPARENT, dpi=400)
        
"""
PLOT FUNCTIONS
"""
def PlotMedianEvolution(
    # --- Dados / seleção do que plotar ---
    names,  columns,  rows,  Type="Evolution", Xparam=("Time",),

    # --- Modos / lógica do gráfico ---
    ColumnPlot=True,  lineparams=False, PhasingPlot=False,   LookBackTime=True,

    # --- Comparações / normalizações ---
    CompareToNormal=False, CompareToNormalLog=True, CompareToNormal_Name=False,  NormalizedExSitu=False, NormalRatio=False,

    # --- Eventos / cortes físicos ---
    Pericenter=False, EntryMedian=False, GasLim=False, Softening=False,

    # --- Layout do painel (grid) ---
    lNum=6,  cNum=6, SmallerScale=False, JustOneXlabel=False,

    # --- Limites e escalas ---
    yscale="linear",  xlimmin=None, xlimmax=None, ylimmin=None,
    ylimmax=None, XScaleSymlog=False, xPhaseLim=8, limaxis=False,

    # --- Texto / título / anotação ---
    title=False, Supertitle=False, Supertitle_Name="DM-rich",  Supertitle_y=0.99,  Text=None,
    xlabelintext=False, loctext=("best",),

    # --- Legenda ---
    legend=False, LegendNames=("None",), legpositions=None,
    loc=("best",), legendColumn=False,  # (parece pouco usado; pode ser deprecado)

    # --- Estilo ---
    GridMake=False,  Transparent=True, alphaShade=0.3, linewidth=1.1, framealpha=0.95, fontlabel=26,multtick=0.99,
    columnspacing=0.5, handlelength=2, handletextpad=0.4,  labelspacing=0.3,

    # --- IO / dependências ---
    savepath="PlotMedianEvolution", savefigname="fig", dfName="Sample",Name="Name", SampleName="SubfindID_99",

    # --- Estatística / reproducibilidade ---
    nboots=100, bins=10, seed=16040105,
):
    """
    Cleaner rewrite of my original PlotMedianEvolution.
    -------
    Core improvements:
    - Centralized indexing for ColumnPlot / lineparams paths
    - More explicit structure: validate -> fetch data -> make axes -> plot -> decorate.
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    """

    # -----------------------------
    # Helpers (local)
    # -----------------------------
    def _as_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        return [x]

    def _panel_get(arr, i, j):
        # arr is expected to be indexed [row][col] if ColumnPlot else [col][row]
        return arr[i][j] if ColumnPlot else arr[j][i]

    def _safe_array(seq):
        # convert list-like to numpy array without changing nested lists too aggressively
        return np.array([v for v in seq], dtype=float)

    def _finite_mask(a):
        a = np.asarray(a)
        return ~np.isnan(a)

    def _maybe_set_ylim(ax, i):
        if ylimmin is not None and ylimmax is not None:
            ax.set_ylim(ylimmin[i], ylimmax[i])

    def _maybe_set_xlim(ax, i):
        if xlimmin is not None and xlimmax is not None:
            ax.set_xlim(xlimmin[i], xlimmax[i])

    def _make_anchored_text(ax, s, loc_):
        if s is None:
            return
        Afont = {"color": "black", "size": fontlabel}
        ax.add_artist(AnchoredText(s, loc=loc_, prop=Afont))

    def _set_log_formatter(ax, axis="y"):
        if axis == "y":
            ax.yaxis.set_major_formatter(FuncFormatter(format_func_loglog))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(format_func_loglog))

    # -----------------------------
    # Validate / normalize inputs
    # -----------------------------
    np.random.seed(seed)

    names = _as_list(names)
    columns = _as_list(columns)
    rows = _as_list(rows)
    Xparam = _as_list(Xparam)

    if legpositions is None:
        legpositions = []
    LegendNames = _as_list(LegendNames)
    loc = _as_list(loc)
    loctext = _as_list(loctext)

    # -----------------------------
    # Load time array
    # -----------------------------
    dfTime = TNG.extractDF("SNAPS_TIME")
    time = np.asarray(dfTime.Age.values)  # assumed lookback time [Gyr] in your code

    # Used in some CoEvolution markers (legacy)
    snapsTime = np.array([88, 81, 64, 51, 37, 24])

    # -----------------------------
    # Optional: pericenter-related data
    # -----------------------------
    dataROverR200 = errROverR200 = None
    if Pericenter:
        dataROverR200, errROverR200 = TNG.makedataevolution(
            names, columns, ["r_over_R_Crit200"],
            SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
        )

    # -----------------------------
    # Fetch data (black-box calls)
    # -----------------------------
    data_bundle = {}

    if Type not in ("Evolution", "CoEvolution"):
        raise ValueError("Type must be 'Evolution' or 'CoEvolution'.")

    if Type == "Evolution":
        if lineparams:
            datasAll, dataserrAll = [], []
            datasPhaseAll, datasTimeAll, datasPhaseTimeAll = [], [], []

            if ColumnPlot:
                # rows define which list of parameters to plot as multiple lines
                for row in rows:
                    if PhasingPlot:
                        datas, dataserr, datasPhase, datasTime, datasPhaseTime = TNG.makedataevolution(
                            names, columns, row,
                            PhasingPlot=PhasingPlot,
                            SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
                        )
                        datasPhaseAll.append(datasPhase)
                        datasTimeAll.append(datasTime)
                        datasPhaseTimeAll.append(datasPhaseTime)
                    else:
                        datas, dataserr = TNG.makedataevolution(
                            names, columns, row,
                            PhasingPlot=PhasingPlot,
                            SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
                        )
                    datasAll.append(datas)
                    dataserrAll.append(dataserr)
            else:
                # columns define which list of parameters to plot as multiple lines
                for column in columns:
                    if PhasingPlot:
                        datas, dataserr, datasPhase, datasTime, datasPhaseTime = TNG.makedataevolution(
                            names, rows, column,
                            PhasingPlot=PhasingPlot,
                            SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
                        )
                        datasPhaseAll.append(datasPhase)          
                        datasTimeAll.append(datasTime)
                        datasPhaseTimeAll.append(datasPhaseTime)
                    else:
                        datas, dataserr = TNG.makedataevolution(
                            names, rows, column,
                            SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
                        )
                    datasAll.append(datas)
                    dataserrAll.append(dataserr)

            data_bundle["datasAll"] = datasAll
            data_bundle["dataserrAll"] = dataserrAll
            if PhasingPlot:
                data_bundle["datasPhaseAll"] = datasPhaseAll
                data_bundle["datasTimeAll"] = datasTimeAll
                data_bundle["datasPhaseTimeAll"] = datasPhaseTimeAll

        else:
            if ColumnPlot:
                if PhasingPlot:
                    datas, dataserr, datasPhase, datasTime, datasPhaseTime = TNG.makedataevolution(
                        names, columns, rows,
                        PhasingPlot=PhasingPlot,
                        SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
                    )
                else:
                    datas, dataserr = TNG.makedataevolution(
                        names, columns, rows,
                        SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
                    )
            else:
                if PhasingPlot:
                    datas, dataserr, datasPhase, datasTime, datasPhaseTime = TNG.makedataevolution(
                        names, rows, columns,
                        PhasingPlot=PhasingPlot,
                        SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
                    )
                else:
                    datas, dataserr = TNG.makedataevolution(
                        names, rows, columns,
                        SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
                    )
            data_bundle["datas"] = datas
            data_bundle["dataserr"] = dataserr
            if PhasingPlot:
                data_bundle["datasPhase"] = datasPhase
                data_bundle["datasTime"] = datasTime
                data_bundle["datasPhaseTime"] = datasPhaseTime

    else:  # Type == "CoEvolution"
        if lineparams:
            datasX, datasXerr = TNG.makedataevolution(
                names, columns, Xparam,
                SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
            )
            datasY, datasYerr = TNG.makedataevolution(
                names, columns, rows[0] if len(rows) else rows,
                SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
            )
        else:
            datasX, datasXerr = TNG.makedataevolution(
                names, columns, Xparam,
                SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
            )
            datasY, datasYerr = TNG.makedataevolution(
                names, columns, rows,
                SampleName=SampleName, dfName=dfName, Name=Name, nboots=nboots
            )

        data_bundle["datasX"] = datasX
        data_bundle["datasXerr"] = datasXerr
        data_bundle["datasY"] = datasY
        data_bundle["datasYerr"] = datasYerr

    # -----------------------------
    # Make axes grid
    # -----------------------------
    plt.rcParams.update({"figure.figsize": (cNum * len(columns), lNum * len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex="col", sharey="row")

    # Normalize axs to 2D array [nrows, ncols]
    if not isinstance(axs, np.ndarray):
        axs = np.array([[axs]])
    elif axs.ndim == 1:
        # either (ncols,) or (nrows,)
        if len(rows) == 1:
            axs = axs.reshape(1, -1)
        else:
            axs = axs.reshape(-1, 1)

    # -----------------------------
    # Plotting
    # -----------------------------
    plotLine_for_colorbar = None  # guard for CoEvolution colorbar

    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            ax = axs[i, j]

            # optional softening shading
            if Softening and isinstance(row, str) and ("Type4" in row):
                rSoftening = ETNG.Softening()
                ax.fill_between(np.flip(time), -1, np.log10(rSoftening), alpha=0.1, color="tab:red")

            # -------------------------
            # Decide what this panel plots
            # -------------------------
            if not lineparams:
                if Type == "Evolution":
                    if ColumnPlot:
                        param = row
                        panel_data = data_bundle["datas"][i][j]
                        panel_err = data_bundle["dataserr"][i][j]
                        if PhasingPlot:
                            panel_phase = data_bundle["datasPhase"][i][j]
                            panel_time_for_phase = data_bundle["datasTime"][i][j]
                    else:
                        param = column
                        panel_data = data_bundle["datas"][j][i]
                        panel_err = data_bundle["dataserr"][j][i]
                        if PhasingPlot:
                            panel_phase = data_bundle["datasPhase"][j][i]
                            panel_time_for_phase = data_bundle["datasTime"][j][i]

                else:  # CoEvolution
                    param = row
                    if ColumnPlot:
                        xparam = Xparam[i] if i < len(Xparam) else Xparam[0]
                        panel_x = data_bundle["datasX"][0][j]
                        panel_data = data_bundle["datasY"][i][j]
                        if Pericenter:
                            panel_r = dataROverR200[0][j]
                    else:
                        xparam = Xparam[i] if i < len(Xparam) else Xparam[0]
                        panel_x = data_bundle["datasX"][j][0]
                        panel_data = data_bundle["datasY"][i][0]
                        if Pericenter:
                            panel_r = dataROverR200[j][0]

                # -------------------------
                # Plot each population in this panel
                # -------------------------
                for l, values_seq in enumerate(panel_data):
                    # Fetch “normal” comparison if requested (black box)
                    if CompareToNormal:
                        if CompareToNormal_Name:
                            Y, Yerr = TNG.makedataevolution(
                                ["Normal"], [names[l]], [row],
                                SampleName=SampleName, dfName=dfName, nboots=nboots
                            )
                        else:
                            Y, Yerr = TNG.makedataevolution(
                                ["Normal"], [column], [row],
                                SampleName=SampleName, dfName=dfName, nboots=nboots
                            )
                        Y = _safe_array(Y[0][0][0])
                        Yerr = _safe_array(Yerr[0][0][0])

                    # x coordinate selection
                    values = _safe_array(values_seq)

                    if PhasingPlot:
                        xParam = np.asarray(panel_phase[l])
                        timeParam = np.asarray(panel_time_for_phase[l])
                    else:
                        xParam = time
                        timeParam = time

                    # pericenter markers need R/R200
                    if Pericenter:
                        ROverR200 = _safe_array(panel_r[l])
                        argInfall1 = np.argwhere(ROverR200 < 1).T[0]
                        argInfall2 = np.argwhere(ROverR200 < 2).T[0]

                    if Type == "Evolution":
                        err = _safe_array(panel_err[l])

                        # special-case NaNs
                        if param in ["sSFRCoreRatio"]:
                            values[values == 0] = np.nan

                        # Legacy: if phasing plot + “LoseTheir” gas case: cut after first NaN
                        if PhasingPlot and isinstance(row, str) and isinstance(column, str):
                            if ("GasMass" in row) and ("LoseTheir" in column):
                                arg_nan = np.argwhere(np.isnan(values)).T[0]
                                if len(arg_nan) > 0:
                                    values[arg_nan[0]:] = np.nan

                        # Compare-to-normal transforms
                        if CompareToNormal:
                            if not CompareToNormalLog:
                                # values / Y, error prop in linear space
                                err = np.sqrt((Yerr / Y) ** 2.0 + (err * values / (Y ** 2.0)) ** 2.0)
                                values = values / Y
                            else:
                                # interpret values as log10; compare in linear space
                                err = np.sqrt((10 ** Yerr / 10 ** Y) ** 2.0 + (err * 10 ** values / (10 ** Y) ** 2.0) ** 2.0)
                                values = (10 ** values) / (10 ** Y)

                        # NormalizedExSitu branch
                        if NormalizedExSitu:
                            Mass4, Mass4err = TNG.makedataevolution(
                                [names[l]], [column], ["SubhaloMassType4"],
                                SampleName=SampleName, dfName=dfName, nboots=nboots
                            )
                            Frac4, Frac4err = TNG.makedataevolution(
                                [names[l]], [column], ["MassExNormalizeAll"],
                                SampleName=SampleName, dfName=dfName, nboots=nboots
                            )

                            Mass4 = _safe_array(Mass4[0][0][0])
                            Mass4err = _safe_array(Mass4err[0][0][0])
                            Frac4 = _safe_array(Frac4[0][0][0])
                            Frac4err = _safe_array(Frac4err[0][0][0])

                            #normalize to z=0 stellar mass
                            values = (10 ** values) / (10 ** Mass4[0])
                            err = Frac4err

                        m = _finite_mask(values) & _finite_mask(xParam)
                        ax.plot(
                            xParam[m], values[m],
                            color=colors(names[l]),
                            ls=lines(names[l]),
                            lw=1.5 * linesthicker(names[l]),
                            dash_capstyle=capstyles(names[l]),
                        )
                        ax.fill_between(
                            xParam[m],
                            values[m] - err[m],
                            values[m] + err[m],
                            color=colors(names[l] + "Error"),
                            alpha=alphaShade,
                        )

                        # entry markers
                        if EntryMedian:
                            dfPop = TNG.extractPopulation(names[l] + column, dfName=dfName)
                            snap_first = int(np.nanmedian(dfPop.Snap_At_FirstEntry))
                            snap_final = int(np.nanmedian(dfPop.Snap_At_FinalEntry))
                            if not np.isnan(snap_first) and not np.isnan(snap_final):
                                ax.axvline(
                                    dfTime.Age.loc[dfTime.Snap == snap_first].values[0],
                                    ymax=0.15 - l * 0.015,
                                    ls="solid", color=colors(names[l]),
                                    lw=1.5 * linesthicker(names[l]),
                                )
                                ax.axvline(
                                    dfTime.Age.loc[dfTime.Snap == snap_final].values[0],
                                    ymax=0.15 - l * 0.015,
                                    ls="--", color=colors(names[l]),
                                    lw=1.5 * linesthicker(names[l]),
                                )

                        # pericenter marker
                        if Pericenter and len(argInfall1) > 0:
                            idx = argInfall1[-1]
                            if 0 <= idx < len(values) and 0 <= idx < len(xParam):
                                ax.scatter(
                                    xParam[idx], values[idx],
                                    color=colors(names[l]),
                                    lw=3 * linewidth, marker="x",
                                    edgecolors=colors(names[l]),
                                    s=120, alpha=0.9
                                )

                    else:
                        # CoEvolution: y vs x, time colored
                        x = _safe_array(panel_x[l])

                        # length-safe interpolation (fix)
                        n = len(values)
                        idx = np.arange(n)

                        mv = _finite_mask(values)
                        mx = _finite_mask(x)
                        if mv.sum() < 2 or mx.sum() < 2:
                            continue

                        f_y = interp1d(idx[mv], values[mv], fill_value="extrapolate")
                        f_x = interp1d(idx[mx], x[mx], fill_value="extrapolate")

                        plotLine = colored_line(
                            f_x(idx), f_y(idx), time[:n],
                            ax, linewidth=2, cmap="bwr_r"
                        )
                        plotLine_for_colorbar = plotLine

                        m = _finite_mask(values) & _finite_mask(x)
                        ax.plot(
                            x[m], values[m],
                            color=colors(names[l]),
                            ls=lines(names[l]),
                            lw=1.5 * linesthicker(names[l]),
                            dash_capstyle=capstyles(names[l]),
                            zorder=1
                        )

                        # start marker
                        ax.scatter(
                            x[0], f_y(0),
                            color="black", lw=2 * linewidth, marker="o",
                            edgecolors=colors(names[l]),
                            s=50, alpha=0.9, zorder=2
                        )

                        # pericenter markers
                        if Pericenter:
                            if len(argInfall2) > 0:
                                idx2 = argInfall2[-1]
                                if 0 <= idx2 < len(values) and 0 <= idx2 < len(x):
                                    ax.scatter(
                                        x[idx2], values[idx2],
                                        color=colors(names[l] + "Error"),
                                        lw=3 * linewidth, marker="x",
                                        edgecolors=colors(names[l] + "Error"),
                                        s=190, alpha=1.0, zorder=3
                                    )
                            if len(argInfall1) > 0:
                                idx1 = argInfall1[-1]
                                if 0 <= idx1 < len(values) and 0 <= idx1 < len(x):
                                    ax.scatter(
                                        x[idx1], values[idx1],
                                        color=colors(names[l] + "Error"),
                                        lw=3 * linewidth, marker="x",
                                        edgecolors=colors(names[l] + "Error"),
                                        s=190, alpha=1.0, zorder=3
                                    )

            # -------------------------
            # lineparams=True path
            # -------------------------
            else:
                # Determine which set defines the multiple lines per panel
                varParam = row if ColumnPlot else column
                varParam = _as_list(varParam)

                for k, paramname in enumerate(varParam):
                    if Type == "Evolution":
                        if ColumnPlot:
                            # datasAll: [i_row][k_param][j_col]
                            data = data_bundle["datasAll"][i][k][j]
                            dataerr = data_bundle["dataserrAll"][i][k][j]
                            if PhasingPlot:
                                dataphase = data_bundle["datasPhaseAll"][i][k][j]
                                datatime = data_bundle["datasTimeAll"][i][k][j]
                                dataphasetime = data_bundle["datasPhaseTimeAll"][i][k][j]
                        else:
                            # datasAll: [i_row][j_col][k_param]
                            data = data_bundle["datasAll"][i][j][k]
                            dataerr = data_bundle["dataserrAll"][i][j][k]
                            if PhasingPlot:
                                dataphase = data_bundle["datasPhaseAll"][i][j][k]
                                datatime = data_bundle["datasTimeAll"][i][j][k]
                                dataphasetime = data_bundle["datasPhaseTimeAll"][i][j][k]

                        param_for_formatting = paramname

                    else:
                        # CoEvolution with lineparams: x depends on Xparam (m) and y depends on param lines
                        xparam = Xparam[i] if i < len(Xparam) else Xparam[0]
                        if ColumnPlot:
                            dataX = data_bundle["datasX"][0][j]
                            data = data_bundle["datasY"][k][j]
                        else:
                            dataX = data_bundle["datasX"][i][j]
                            data = data_bundle["datasY"][k][j]
                        param_for_formatting = paramname

                    # compare-to-normal
                    if CompareToNormal:
                        Y, Yerr = TNG.makedataevolution(
                            ["Normal"], [column], [paramname],
                            SampleName=SampleName, dfName=dfName, nboots=nboots
                        )
                        Y = _safe_array(Y[0][0][0])
                        Yerr = _safe_array(Yerr[0][0][0])

                    # plot each population
                    for l, values_seq in enumerate(data):
                        values = _safe_array(values_seq)

                        if PhasingPlot and Type == "Evolution":
                            xParam = np.asarray(dataphase[l])
                            timeParam = np.asarray(datatime[l])
                            phaseParam = np.asarray(dataphasetime[l])
                        else:
                            xParam = time
                            timeParam = time
                            phaseParam = time

                        if Type == "Evolution":
                            err = _safe_array(dataerr[l])

                            # thresholds
                            if "sSFR" in str(paramname):
                                values[values < -13.5] = np.nan
                            elif "SFR" in str(paramname):
                                values[values < -4] = np.nan

                            if CompareToNormal:
                                err = np.sqrt((Yerr) ** 2.0 + (err) ** 2.0)
                                values = values - Y

                            # GasLim cutoff after SnapLostGas (fix boolean)
                            if GasLim and (("Gas" in str(paramname)) or ("SFR" in str(paramname)) or ("Type0" in str(paramname))):
                                dfPop = TNG.extractPopulation(names[l] + column, dfName=dfName)
                                med = np.nanmedian(dfPop.SnapLostGas)
                                if (not np.isnan(med)) and (med > 0):
                                    t_cut = dfTime.Age.loc[dfTime.Snap == int(med)].values[0]
                                    
                                    mask_after = timeParam > t_cut
                                    if mask_after.any():
                                        PhaseNonGas = phaseParam[mask_after][0]
                                        values[xParam > PhaseNonGas] = np.nan

                            m = _finite_mask(values) & _finite_mask(xParam)

                            ax.plot(
                                xParam[m], values[m],
                                color=colors(names[l]),
                                ls=lines(paramname),
                                lw=1.7 * linesthicker(paramname),
                                dash_capstyle=capstyles(paramname),
                            )
                            ax.fill_between(
                                xParam[m],
                                values[m] - err[m],
                                values[m] + err[m],
                                color=colors(names[l] + "Error"),
                                alpha=alphaShade,
                            )

                            # EntryMedian block: keep, but remove dead self-comparison
                            if EntryMedian:
                                dfPop = TNG.extractPopulation(names[l] + column, dfName=dfName)
                                snaps = np.asarray(dfPop.Snap_At_FirstEntry.values, dtype=float)
                                snaps = snaps[~np.isnan(snaps)]
                                if len(snaps) > 5:
                                    snap_med = int(np.nanmedian(snaps))
                                    ax.axvline(
                                        dfTime.Age.loc[dfTime.Snap == snap_med].values[0],
                                        ymax=0.15 - l * 0.015,
                                        ls="solid", color=colors(names[l]),
                                        lw=1.5 * linesthicker(names[l]),
                                    )
                                    ax.axvline(
                                        dfTime.Age.loc[dfTime.Snap == snap_med].values[0],
                                        ymax=0.15 - l * 0.015,
                                        ls="--", color=colors(names[l]),
                                        lw=1.5 * linesthicker(names[l]),
                                    )

                        else:
                            x = _safe_array(dataX[l])
                            m = _finite_mask(values) & _finite_mask(x)
                            ax.plot(
                                x[m], values[m],
                                color=colors(names[l]),
                                ls=lines(paramname),
                                lw=1.7 * linesthicker(paramname),
                                dash_capstyle=capstyles(paramname)
                            )
                            # Keep legacy snap markers but make safe for length
                            n = len(values)
                            idx_snaps = (n - 1) - snapsTime
                            idx_snaps = idx_snaps[(idx_snaps >= 0) & (idx_snaps < n)]
                            if len(idx_snaps) > 0:
                                ax.scatter(
                                    x[idx_snaps], values[idx_snaps],
                                    lw=2 * linewidth, marker="d",
                                    s=50, alpha=0.9
                                )
                            ax.scatter(
                                x[0], values[0],
                                color="black", lw=2 * linewidth, marker="o",
                                s=40, alpha=0.9
                            )

            # -------------------------
            # Common panel decorations
            # -------------------------
            if CompareToNormal:
                ax.axhline(y=0, color="gray", lw=1.5 * linewidth)

            if GridMake:
                ax.grid(GridMake, color="#9e9e9e", which="major", linewidth=0.6, alpha=0.3, linestyle=":")

            _maybe_set_ylim(ax, i)

            # yscale logic
            
            if lineparams:
                p_for_scale = param_for_formatting
            else:
                p_for_scale = param

            if not NormalizedExSitu:
                ax.set_yscale(scales(p_for_scale))
                if scales(p_for_scale) == "log":
                    _set_log_formatter(ax, axis="y")
            else:
                ax.set_yscale("log")
                _set_log_formatter(ax, axis="y")

            # special y-ticks (kept)
            if p_for_scale == "MassExNormalizeAll" and lineparams:
                ax.set_yticks([0.001, 0.005, 0.01, 0.02, 0.05])
                ax.set_yticklabels(["0.001", "0.005", "0.01", "0.02", "0.05"])
            elif p_for_scale == "MassExNormalizeAll" or NormalizedExSitu:
                ax.set_yticks([0.005, 0.01, 0.05])
                ax.set_yticklabels(["0.005", "0.01", "0.05"])

            if p_for_scale in ("MassExNormalize", "MassInNormalize"):
                ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.5, 1])
                ax.set_yticklabels(["0.01", "0.02", "0.05", "0.1", "0.5", "1"])

            if p_for_scale == "GroupNsubsFinalGroup":
                ax.set_yticks([20, 30, 40, 60])
                ax.set_yticklabels(["20", "30", "40", "60"])

            if p_for_scale == "StarMassNormalized":
                ax.set_yticks([0.1, 0.2, 0.5, 1])
                ax.set_yticklabels(["0.1", "0.2", "0.5", "1"])

            # Legends
            if legend:
                for legpos, LegendName in enumerate(LegendNames):
                    if legpos >= len(legpositions):
                        continue
                    if j == legpositions[legpos][0] and i == legpositions[legpos][1]:
                        custom_lines, label, ncol, mult = Legend(LegendName)
                        ax.legend(
                            custom_lines, label, ncol=ncol, loc=loc[legpos],
                            fontsize=0.88 * fontlabel, framealpha=framealpha,
                            columnspacing=columnspacing, handlelength=handlelength,
                            handletextpad=handletextpad, labelspacing=labelspacing
                        )

            # Y labels (left column)
            if j == 0:
                if xlabelintext:
                    if not CompareToNormal:
                        ax.set_ylabel(labelsequal.get(p_for_scale, p_for_scale), fontsize=fontlabel)
                    else:
                        ax.set_ylabel(labelsequal.get(p_for_scale, p_for_scale) + "$-$" + labelsequal.get(p_for_scale, p_for_scale) + "$_\\mathrm{Normals}$",
                                      fontsize=fontlabel)
                else:
                    if not CompareToNormal and not NormalizedExSitu:
                        if lineparams and len(_as_list(row if ColumnPlot else column)) > 1:
                            ax.set_ylabel(labelsequal.get(p_for_scale, p_for_scale), fontsize=fontlabel)
                        else:
                            ax.set_ylabel(labels.get(p_for_scale, p_for_scale), fontsize=fontlabel)
                    elif not CompareToNormal and NormalizedExSitu:
                        ax.set_ylabel(labels.get("MassExNormalizeAll", "MassExNormalizeAll"), fontsize=fontlabel)
                    else:
                        ax.set_ylabel(labels.get(p_for_scale, p_for_scale) + "$-$" + labels.get(p_for_scale, p_for_scale) + "$_\\mathrm{Normals}$", fontsize=fontlabel)

                ax.tick_params(axis="y", labelsize=multtick * fontlabel)

            # Rightmost column text
            if j == len(columns) - 1:
                if Text is not None and (p_for_scale not in ["SubhalosSFRInHalfRad", "SubhalosSFRwithinHalfandRad", "SubhalosSFRwithinRadandAll"]):
                    _make_anchored_text(ax, Text[i] if i < len(Text) else None, "lower left")

                if xlabelintext and (not limaxis) and len(rows) > 1:
                    _make_anchored_text(ax, texts.get(p_for_scale, p_for_scale), "upper right")

            # Top row titles + redshift axis for Evolution (non-phasing)
            if i == 0:
                if title:
                    ax.set_title(titles(title[j]), fontsize=1.1 * fontlabel)

                if Type == "Evolution" and (not PhasingPlot):
                    ax2 = ax.twiny()
                    ax2.grid(False)

                    if XScaleSymlog:
                        ax2.set_xlim(-0.5, 14.0)
                        ax2.set_xscale("symlog")
                        zlabels = np.array(["0", "0.5", "1", "2", "5", "20"])
                        zticks_Age = np.array([13.803, 8.587, 5.878, 3.285, 1.2, 0.0])
                    else:
                        ax2.set_xlim(-0.5, 14.5)
                        if (not JustOneXlabel) and (not SmallerScale):
                            zlabels = np.array(["0", "0.2", "0.5", "1", "2", "5", "20"])
                        else:
                            zlabels = np.array(["0", "0.2", "0.5", "1", "2", "5", "20" if j == 0 else ""])
                        zticks_Age = np.array([13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0.0])

                    ax2.xaxis.set_major_locator(FixedLocator(zticks_Age.tolist()))
                    ax2.xaxis.set_major_formatter(FixedFormatter(zlabels.tolist()))
                    ax2.set_xlabel(r"$z$", fontsize=fontlabel)
                    ax2.tick_params(labelsize=multtick * fontlabel)
                    ax2.tick_params(axis="x", which="minor", top=False)
                    ax2.minorticks_off()
                else:
                    ax.tick_params(axis="x", which="minor", top=False)

            # Bottom row x-axis formatting
            if i == len(rows) - 1:
                if Type == "Evolution":
                    if LookBackTime and (not PhasingPlot):
                        if JustOneXlabel:
                            if (j == 1):
                                ax.set_xlabel(r"$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$", fontsize=fontlabel)
                        else:
                            ax.set_xlabel(r"$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$", fontsize=fontlabel)

                        if XScaleSymlog:
                            ax.set_xscale("symlog")
                            ax.set_xlim(-0.5, 14.5)
                            ax.set_xticks([0, 1.97185714, 3.94371429, 5.91557143, 7.88742857, 9.85928571, 13.803])
                            ax.set_xticklabels(["14", "12", "10", "8", "6", "4", "0"])
                        else:
                            ax.set_xlim(-0.5, 14.5)
                            ax.set_xticks([0.0, 1.97185714, 3.94371429, 5.91557143, 7.88742857, 9.85928571, 11.83114286, 13.803])
                            if (not JustOneXlabel) and (not SmallerScale):
                                ax.set_xticklabels(["14", "12", "10", "8", "6", "4", "2", "0"])
                            else:
                                ax.set_xticklabels(["14", "12", "10", "8", "6", "4", "2", "0"] if j == 0 else ["", "12", "10", "8", "6", "4", "2", "0"])
                    elif not PhasingPlot:
                        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                        if JustOneXlabel:
                            if i == 1:
                                ax.set_xlabel(r"$t \, \, [\mathrm{Gyr}]$", fontsize=fontlabel)
                        else:
                            ax.set_xlabel(r"$t \, \, [\mathrm{Gyr}]$", fontsize=fontlabel)
                        ax.set_xticklabels(["0", "2", "4", "6", "8", "10", "12", "14"] if (not JustOneXlabel or j == 0) else ["", "2", "4", "6", "8", "10", "12", "14"])
                    else:
                        limXparam = int(xPhaseLim + 1)
                        postiveXticks = np.arange(limXparam)
                        postiveXLabels = np.array([str(int(v)) for v in postiveXticks])
                        postiveXticks = np.append([-1, -0.5], postiveXticks)
                        postiveXLabels = np.append(["", "E"], postiveXLabels)

                        ax.set_xlabel(r"$\phi_\mathrm{Orbital}$", fontsize=fontlabel)
                        ax.set_xticks(postiveXticks)
                        ax.set_xticklabels(postiveXLabels)
                        ax.set_xlim(-1, xPhaseLim + 0.5)

                    ax.tick_params(axis="x", labelsize=multtick * fontlabel)

                else:
                    # CoEvolution x formatting
                    ax.set_xscale(scales(xparam))
                    if scales(xparam) == "log":
                        _set_log_formatter(ax, axis="x")
                    _maybe_set_xlim(ax, i)
                    ax.set_xlabel(labels.get(xparam, xparam), fontsize=fontlabel)
                    ax.tick_params(axis="x", labelsize=multtick * fontlabel)

    # -----------------------------
    # CoEvolution colorbar (guarded)
    # -----------------------------
    if Type == "CoEvolution" and plotLine_for_colorbar is not None:
        cb = fig.colorbar(
            plotLine_for_colorbar,
            ax=axs.ravel().tolist(),
            ticks=[0.0, 1.97185714, 3.94371429, 5.91557143, 7.88742857, 9.85928571, 11.83114286, 13.803],
            pad=0.02, aspect=50
        )
        cb.ax.set_yticklabels(["14", "12", "10", "8", "6", "4", "2", "0"])
        cb.set_label("Lookback Time [Gyr]", fontsize=1.0 * fontlabel)
        cb.ax.tick_params(labelsize=multtick * fontlabel)

    if Supertitle:
        plt.suptitle(Supertitle_Name, fontsize=1.3 * fontlabel, y=Supertitle_y)

    savefig(savepath, savefigname, Transparent)
    return


def PlotHist(
    # --- Dados / seleção do que plotar ---
    names, columns, rows, Type="z0", snap=(99,), ColumnPlot=True,

    # --- Modo do histograma / KDE ---
    density=False, NormCount=False, bins="rice",

    # --- Estatísticas sobrepostas (linhas/áreas) ---
    mean=False, median=False, medianPlot=False, nboots=100,

    # --- Layout do painel (grid) ---
    lNum=6, cNum=6, GridMake=False, JustOneXlabel=False,

    # --- Limites e escalas ---
    xscale="linear", yscale="linear", xlimmin=None, xlimmax=None, ylimmin=None, ylimmax=None,
    toplim=1e3,limaixsy=False, liminvalue=(0,), limax=(1,),

    # --- Texto / títulos / lookback ---
    title=False, xlabelintext=False, LookBackTime=False, Supertitle=False,
    SupertitleName="", Supertitle_y=1.22,

    # --- Legenda ---
    legend=False, LegendNames=None, legpositions=None, loc="best",  legendColumn=False,

    # --- Estilo ---
    alphaShade=0.3, linewidth=1.8, fontlabel=24, framealpha=0.95,  columnspacing=0.5,
    handlelength=2, handletextpad=0.4,labelspacing=0.3,

    # --- IO / nomes de colunas no DF ---
    savepath="fig/PlotHist", savefigname="fig", TRANSPARENT=False,  dfName="Sample",
    SampleName="Samples", Name="Name",

    # --- Reprodutibilidade ---
    seed=16010504,
):
    """
    Plot histograms (or KDE) for a grid of parameters.
    -------
    - rows x columns define the subplot grid.
    - For each subplot, you plot distributions for each population in `names`.
    - If `density=True`, you draw KDE curves (PDF-like).
    - Otherwise, you draw histograms (optionally normalized via NormCount).
    - You may overlay mean/median vertical lines and optional shading.
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    """

    np.random.seed(seed)

    # ---------- Normalize inputs ----------
    def _as_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        return [x]

    names = _as_list(names)

    # Fix: treat "columns == 'Snap'" case robustly (since columns becomes list)
    columns_is_snap = (isinstance(columns, str) and columns == 'Snap')
    columns = _as_list(columns)
    rows = _as_list(rows)
    snap = _as_list(snap)

    if LegendNames is None:
        LegendNames = []
    else:
        LegendNames = _as_list(LegendNames)

    if legpositions is None:
        legpositions = []

    # ---------- Load time table (used only if param includes Snap) ----------
    dfTime = TNG.extractDF('SNAPS_TIME')

    # ---------- Get data (black box) ----------
    # Handle Snap case: if user wants columns='Snap', columns actually becomes snapshots
    if columns_is_snap or (len(columns) == 1 and columns[0] == 'Snap'):
        # overwrite columns with snap list (like your intention)
        columns = snap
        data_type = 'Snap'
    else:
        data_type = Type

    if ColumnPlot:
        datas = TNG.makedata(names, columns, rows, data_type, snap=snap, dfName=dfName, Name=Name, SampleName=SampleName)
    else:
        datas = TNG.makedata(names, rows, columns, data_type, snap=snap, dfName=dfName, Name=Name, SampleName=SampleName)

    # ---------- Figure / axes ----------
    plt.rcParams.update({'figure.figsize': (cNum * len(columns), lNum * len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')

    # normalize axs to 2D
    if not isinstance(axs, np.ndarray):
        axs = np.array([[axs]])
    elif axs.ndim == 1:
        axs = axs.reshape(1, -1) if len(rows) == 1 else axs.reshape(-1, 1)

    # ---------- Helpers ----------
    def _panel_data(i, j):
        """Return (titlename, param, data) for panel (i,j) with ColumnPlot switch."""
        if ColumnPlot:
            titlename = columns[j]
            param = rows[i]
            data = datas[i][j]
        else:
            titlename = rows[i]
            param = columns[j]
            data = datas[j][i]
        return titlename, param, data

    def _clean_values(v):
        """Remove inf and nan; return 1D array."""
        v = np.asarray(v, dtype=float)
        v = v[~np.isinf(v)]
        v = v[~np.isnan(v)]
        return v

    def _plot_kde(ax, v, name_key):
        if len(v) < 2:
            return
        kde = stats.gaussian_kde(v)
        xx = np.linspace(v.min(), v.max(), 1000)
        ax.plot(
            xx, kde(xx),
            color=colors(name_key),
            ls=lines(name_key),
            linewidth=linewidth,
            dash_capstyle=capstyles(name_key)
        )

    def _plot_hist(ax, v, name_key, param, binnumber):
        # special bins by param
        if param in ('rOverR200Min', 'rOverR200_99'):
            bins_edges = np.geomspace(0.03, 5, binnumber)
            ax.hist(v, bins=bins_edges, log=True, alpha=1, histtype='step',
                    color=colors(name_key), ls=lines(name_key),
                    density=density, linewidth=linewidth)
            return

        if param == 'r_over_R_Crit200':
            bins_edges = np.logspace(np.log10(0.1), np.log10(10), 20)
            ax.hist(v, bins=bins_edges, alpha=1, histtype='step',
                    color=colors(name_key), ls=lines(name_key),
                    density=density, linewidth=linewidth)
            return

        if param == 'GasFrac':
            bins_edges = np.logspace(np.log10(0.001), np.log10(0.5), 10)
            ax.hist(v, bins=bins_edges, alpha=1, histtype='step',
                    color=colors(name_key), ls=lines(name_key),
                    density=density, linewidth=linewidth)
            return

        if param == 'DMFrac':
            bins_edges = np.logspace(np.log10(0.15), np.log10(1), 10)
            ax.hist(v, bins=bins_edges, alpha=1, histtype='step',
                    color=colors(name_key), ls=lines(name_key),
                    density=density, linewidth=linewidth)
            return

        if param == 'StarFrac':
            bins_edges = np.logspace(np.log10(0.005), np.log10(1), 10)
            ax.hist(v, bins=bins_edges, alpha=1, histtype='step',
                    color=colors(name_key), ls=lines(name_key),
                    density=density, linewidth=linewidth)
            return

        # generic hist
        ax.hist(v, bins=binnumber, alpha=1, histtype='step',
                color=colors(name_key), ls=lines(name_key),
                density=density, linewidth=linewidth)

    def _overlay_stat(ax, v_raw, name_key, which='mean'):
        """Draw mean/median line and optional shading."""
        if len(v_raw) == 0:
            return
        if which == 'mean':
            center = np.nanmean(v_raw)
            ymax = 0.15
            ax.axvline(center, ymax=ymax, color=colors(name_key),
                       ls=lines(name_key), linewidth=2.3 * linewidth)
            if medianPlot:
                # choose ONE: bootstrap or std. Keep std as fallback
                try:
                    xerr = MATH.boostrap_func(v_raw, num_boots=nboots)
                    # if boostrap_func returns samples, convert to std
                    xerr = np.std(xerr) if np.ndim(xerr) > 0 else float(xerr)
                except Exception:
                    xerr = np.std(v_raw)
                ax.axvspan(center - xerr, center + xerr,
                           color=colors(name_key),
                           alpha=alphaShade)
        else:
            center = np.nanmedian(v_raw)
            ymax = 0.15
            ax.axvline(center, ymax=ymax, color=colors(name_key),
                       ls=lines(name_key), linewidth=2.3 * linewidth)
            if medianPlot:
                try:
                    xerr = MATH.boostrap_func(v_raw, num_boots=nboots)
                    xerr = np.std(xerr) if np.ndim(xerr) > 0 else float(xerr)
                except Exception:
                    xerr = np.std(v_raw)
                ax.axvspan(center - xerr, center + xerr,
                           color=colors(name_key + 'Error'),
                           alpha=alphaShade)

    # ---------- Main loop ----------
    for i, row in enumerate(rows):
        for j, column in enumerate(columns):
            ax = axs[i, j]
            titlename, param, panel = _panel_data(i, j)

            # plot each population
            for l, values_seq in enumerate(panel):
                values_raw = np.asarray([v for v in values_seq], dtype=float)

                # if Snap-based param, convert Snap->Age
                if isinstance(param, str) and ('Snap' in param):
                    # robust conversion; ignore snaps that aren't present
                    converted = []
                    for v in values_raw:
                        if np.isnan(v) or np.isinf(v):
                            continue
                        snap_int = int(v)
                        m = dfTime.Snap == snap_int
                        if m.any():
                            converted.append(dfTime.loc[m, 'Age'].values[0])
                    values_raw = np.asarray(converted, dtype=float)

                # keep original for mean/median printing/overlay, but clean for plotting
                v_clean = _clean_values(values_raw)
                if len(v_clean) == 0:
                    continue

                name_key = names[l]

                if density:
                    _plot_kde(ax, v_clean, name_key)
                else:
                    if NormCount:
                        hist, bin_edges = np.histogram(v_clean, bins=bins, density=False)
                        s = hist.sum()
                        if s > 0:
                            ax.step(bin_edges[:-1], hist / s,
                                    color=colors(name_key),
                                    ls=lines(name_key))
                    else:
                        if isinstance(bins, list):
                            binnumber = bins[i][l]
                        else:
                            binnumber = bins

                        # special handling: if "Above1" and MBC, turn zeros into nan then clean
                        if isinstance(param, str) and ('Above1' in param) and (name_key == 'MBC'):
                            v_clean = _clean_values(np.where(v_clean == 0, np.nan, v_clean))

                        _plot_hist(ax, v_clean, name_key, param, binnumber)

                # overlays
                if mean:
                    _overlay_stat(ax, v_clean, name_key, which='mean')
                if median or medianPlot:
                    _overlay_stat(ax, v_clean, name_key, which='median')

            # ---------- Panel formatting ----------
            if GridMake:
                ax.grid(GridMake, color='#9e9e9e', which="major", linewidth=0.6, alpha=0.3, linestyle=':')

            # yscale + limits
            ax.set_yscale(yscale)
            ax.tick_params(labelsize=0.99 * fontlabel)

            # IMPORTANT: default y-limits only for histogram counts (not KDE density)
            if not density:
                ax.set_ylim(bottom=0.5, top=toplim)

            # xscale special cases
            if param in ('rOverR200Min', 'rOverR200_99'):
                ax.set_xscale('log')
                ax.xaxis.set_major_formatter(FuncFormatter(format_func_loglog))
            elif (xscale == 'log') or (param == 'GasFrac'):
                ax.set_xscale(xscale)
                if ax.get_xscale() == 'log':
                    ax.xaxis.set_major_formatter(FuncFormatter(format_func_loglog))

            if yscale == 'log':
                ax.yaxis.set_major_formatter(FuncFormatter(format_func_loglog))

            # legend
            if legend and LegendNames:
                for legpos, LegendName in enumerate(LegendNames):
                    if legpos >= len(legpositions):
                        continue
                    if j == legpositions[legpos][0] and i == legpositions[legpos][1]:
                        custom_lines, label, ncol, mult = Legend(LegendName)
                        ax.legend(
                            custom_lines, label, ncol=ncol, loc=loc[legpos] if isinstance(loc, (list, tuple)) else loc,
                            fontsize=0.88 * fontlabel, framealpha=framealpha,
                            columnspacing=columnspacing, handlelength=handlelength,
                            handletextpad=handletextpad, labelspacing=labelspacing
                        )

            # explicit y limits if requested
            if limaixsy:
                ax.set_ylim(liminvalue[i], limax[i])
            if (ylimmin is not None) and (ylimmax is not None):
                ax.set_ylim(ylimmin[i], ylimmax[i])

            # y-label on left column
            if j == 0:
                if density:
                    ax.set_ylabel('Density', fontsize=fontlabel)
                else:
                    ax.set_ylabel('Normalized Counts' if NormCount else 'Counts', fontsize=fontlabel)
                ax.tick_params(axis='y', labelsize=0.99 * fontlabel)

            # x-label-in-text box on left column
            if j == 0 and xlabelintext:
                Afont = {'color': 'black', 'size': fontlabel}
                if not isinstance(xlabelintext, bool):
                    s = titles(xlabelintext[i])
                    anchored = AnchoredText(s, loc='upper left', prop=Afont, pad=0.3)
                    anchored.patch.set_facecolor('linen')
                    anchored.patch.set_edgecolor('black')
                    anchored.patch.set_alpha(0.5)
                    anchored.patch.set_boxstyle('round')
                else:
                    anchored = AnchoredText(texts.get(param, str(param)), loc='upper right', prop=Afont)
                ax.add_artist(anchored)

            # top row titles + z axis when param is time-like
            if i == 0:
                if columns_is_snap:
                    # titlename is a snap int
                    zval = dfTime.z.loc[dfTime.Snap == int(titlename)].values[0]
                    ax.set_title(r'$z = %.1f$' % zval, fontsize=1.1 * fontlabel)
                if title:
                    ax.set_title(titles(title[j]), fontsize=1.1 * fontlabel)

                lab = labels.get(param, 'None')
                time_like = ('Gyr' in lab) and ('Gyr^' not in lab) and ('_after_' not in str(param)) and ('Delta' not in lab)
                if time_like:
                    ax2 = ax.twiny()
                    ax2.grid(False)
                    ax2.set_xlim(-0.5, 14.5)

                    if len(columns) == 3:
                        zlabels = np.array(['0', '0.2', '0.5', '1', '2', '5'])
                    else:
                        zlabels = np.array(['0', '0.2', '0.5', '1', '2', '5', '20'])

                    zticks_Age = np.array([13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0.0])
                    ax2.xaxis.set_major_locator(FixedLocator(zticks_Age.tolist()))
                    ax2.xaxis.set_major_formatter(FixedFormatter(zlabels.tolist()))
                    ax2.set_xlabel(r"$z$", fontsize=fontlabel)
                    ax2.tick_params(labelsize=0.99 * fontlabel)
                    ax2.minorticks_off()

            # bottom row xlabels + lookback/time ticks
            if i == len(rows) - 1:
                if (xlimmin is not None) and (xlimmax is not None):
                    ax.set_xlim(xlimmin[j], xlimmax[j])
                    if xlimmin[j] == -0.05 and xlimmax[j] == 1.05:
                        ax.set_xticks([0, 0.5, 1])
                        ax.set_xticklabels(['0', '0.5', '1'])

                # One shared xlabel option
                if JustOneXlabel:
                    if j == 1:
                        fig.supxlabel(labelsequal.get(param, str(param)), fontsize=fontlabel, y=-0.05)
                    continue

                ax.set_xlabel(labels.get(param, str(param)), fontsize=fontlabel)
                ax.tick_params(axis='x', labelsize=0.99 * fontlabel)

                lab = labels.get(param, 'None')
                time_like = (('Gyr' in lab) and ('_after_' not in str(param)) and ('Delta' not in lab)) or ('Snap' in str(param))
                if time_like:
                    if LookBackTime:
                        ax.set_xlabel('Lookback  Time \n  [Gyr]', fontsize=fontlabel)
                        ax.set_xticks([0., 1.97185714, 3.94371429, 5.91557143, 7.88742857, 9.85928571, 11.83114286, 13.803])
                        if len(columns) == 3:
                            ax.set_xticks([1.97185714, 5.91557143, 9.85928571, 13.803])
                            ax.set_xticklabels(['12', '8', '4', '0'])
                        else:
                            ax.set_xticklabels(['14', '12', '10', '8', '6', '4', '2', '0'])
                    else:
                        ax.set_xlim(-0.9, 14.5)
                        ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                        if len(columns) == 3:
                            ax.set_xticks([0, 4, 8, 12])
                            ax.set_xticklabels(['0', '4', '8', '12'])
                        else:
                            ax.set_xticklabels(['', '2', '4', '6', '8', '10', '12', '14'])

    if Supertitle:
        plt.suptitle(SupertitleName, fontsize=1.3 * fontlabel, y=Supertitle_y)

    savefig(savepath, savefigname, TRANSPARENT)
    return


def PlotScatter(
    # --- Data / what to plot ---
    names, columns,  ParamX, ParamsY, Type="z0", snap=(99,), ColumnPlot=True,
    dfName="Sample", SampleName="Samples", Name="Name",

    # --- Extra layers ---
    All=None, COLORBAR=None, MarkerSizes=None, NoneEdgeColor=False,

    # --- Statistics ---
    medianBins=False, medianAll=False, medianDot=False, SpearManTest=False, SpearManTestAll=False,
    bins=10, quantile=0.95, q=0.95, HIGHLIGHTPoints=False,

    # --- Layout ---
    lNum=6, cNum=6, xscale = None, GridMake=False, InvertPlot=False, xlabelintext=False, title=False,

    # --- Helper lines ---
    EqualLine=False, EqualLineMin=None, EqualLineMax=None,

    # --- Limits ---
    xlimmin=None, xlimmax=None, ylimmin=None, ylimmax=None,

    # --- Style ---
    cmap="inferno", m="o", msizet=30, msizeMult=1, alphaScater=1.0, alphaShade=0.3, linewidth=1.2, fontlabel=26, framealpha=0.95,

    # --- Legend ---
    legend=False,LegendNames=None, legpositions=None, loc="best", columnspacing=0.5,
    handlelength=2, handletextpad=-0.5, labelspacing=0.3,

    # --- Colorbar ---
    ratioColorbar=None, mult=4.1,

    # --- IO ---
    savepath="fig/PlotScatter", savefigname="fig", TRANSPARENT=False,

    # --- Reproducibility ---
    seed=16010504,
):
    """
    Scatter plot grid for X–Y relations across columns (e.g., samples/snapshots) and multiple Y parameters.
    -------
    - keep the exact behavior,
    - move special rules out of the core loops into dedicated helpers,
    - reduce indexing bugs / duplicated logic,
    - make the function easier to maintain.
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    """

   
    np.random.seed(seed)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _as_list(x):
        return x if isinstance(x, (list, tuple, np.ndarray)) else [x]

    def _normalize_inputs(columns, ParamX, ParamsY):
        cols = _as_list(columns)
        Ys = _as_list(ParamsY)

        if isinstance(ParamX, (list, tuple, np.ndarray)):
            Xs = list(ParamX)
            label_general = True
        else:
            Xs = [ParamX] * len(Ys)
            label_general = False

        if len(Xs) != len(Ys):
            raise ValueError("ParamX must be a scalar or have the same length as ParamsY.")

        return cols, Xs, Ys, label_general

    def _load_data(names, columns, ParamsX, ParamsY):
        """
        Loads X and Y arrays using TNG.makedata with the same logic as the original code.
        Returns:
            panel_cols_for_data, dataX, dataY, dataColor, dataMarker
        """
        # Snap-special case
        if columns == ["Snap"]:
            cols_for_data = list(snap)

            dataX = TNG.makedata(
                names, cols_for_data, ParamsX, "Snap",
                snap=snap, SampleName=SampleName, dfName=dfName, Name=Name
            )
            dataY = TNG.makedata(
                names, cols_for_data, ParamsY, "Snap",
                snap=snap, SampleName=SampleName, dfName=dfName, Name=Name
            )

            dataColor = None
            dataMarker = None
            if COLORBAR is not None:
                dataColor = TNG.makedata(
                    names, cols_for_data, COLORBAR, "Snap",
                    snap=snap, SampleName=SampleName, dfName=dfName, Name=Name
                )
            return cols_for_data, dataX, dataY, dataColor, dataMarker

        # General case
        dataX = TNG.makedata(
            names, columns, ParamsX, Type,
            snap=snap, SampleName=SampleName, dfName=dfName, Name=Name
        )
        dataY = TNG.makedata(
            names, columns, ParamsY, Type,
            snap=snap, SampleName=SampleName, dfName=dfName, Name=Name
        )

        dataColor = None
        dataMarker = None
        if MarkerSizes is not None:
            dataMarker = TNG.makedata(
                names, columns, MarkerSizes, Type,
                snap=snap, SampleName=SampleName, dfName=dfName, Name=Name
            )
        if COLORBAR is not None:
            dataColor = TNG.makedata(
                names, columns, COLORBAR, Type,
                snap=snap, SampleName=SampleName, dfName=dfName, Name=Name
            )

        return columns, dataX, dataY, dataColor, dataMarker

    def _setup_axes(panel_columns, nrows):
        plt.rcParams.update({"figure.figsize": (cNum * len(panel_columns), lNum * nrows)})
        fig = plt.figure()
        gs = fig.add_gridspec(nrows, len(panel_columns), hspace=0, wspace=0)
        axs = gs.subplots(sharex="col", sharey="row")

        # Ensure 2D array shape
        if not isinstance(axs, (list, np.ndarray)):
            axs = [axs]
        if not isinstance(axs[0], np.ndarray):
            axs = np.array([axs])
            if len(panel_columns) == 1:
                axs = axs.T
        return fig, axs
    
    def _apply_special_xaxis_rules(ax, ParamX, ParamsY, yparam, ylimmin, fontlabel):
        """
        Special-case X axis formatting rules.
        Keep all one-off tick/label/line logic here to preserve behavior
        without cluttering the main plotting loop.
        """
    
        # --- Explicit tick sets for specific ParamX ---
        if ParamX == "DecreaseBeforeGas":
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])
    
        if ParamX == "Decrease_Entry_To_NoGas_Norm_Delta":
            ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
            ax.set_xticklabels(["-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2"])
    
        if "Snap" in ParamX:
            ax.set_xlim(-0.2, 14.2)
            ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
            ax.set_xticklabels(["0", "2", "4", "6", "8", "10", "12", "14"])
            
        if 'DMFrac_Birth' in yparam:
            ax.set_yticks([0.001, 0.01, 0.1, 0.5, 0.9, 0.99])
            ax.set_yticklabels(
                ['$10^{-3}$', '$10^{-2}$', '0.1', '0.5', '0.9', '0.99'])
    
        if ParamX == "MassIn_Infall_to_GasLost":
            ax.set_xticks([-0.15, 0, 0.25, 0.5, 0.75])
            ax.set_xticklabels(["-0.15", "0", "0.25", "0.50", "0.75"])
            
    
        # --- Smaller tick label size + custom ticks for a specific ParamX/Y combo ---
        if ("StarFrac" in ParamX) and ("GasFrac" in yparam) and (ylimmin != [0.001]):
            ax.tick_params(axis="y", labelsize=0.88 * fontlabel)
            ax.tick_params(axis="x", labelsize=0.88 * fontlabel)
    
            ax.set_yticks([0.02, 0.03, 0.04, 0.06, 0.08, 0.1])
            ax.set_yticklabels(["0.02", "0.03", "0.04", "0.06", "0.08", "0.1"])
            ax.set_xticks([0.004, 0.006, 0.01, 0.02, 0.03])
            ax.set_xticklabels(["0.004", "0.006", "0.01", "0.02", "0.03"])

    def _apply_special_background_rules(ax, ParamX, firstY, linewidth, fontlabel):
        """
        Place ALL your special-case quadrant fills / guide lines / custom ticks here.
        """
        # --- Special rules (subset copied from your original) ---

        if (ParamX == "MassIn_Infall_to_GasLost") and (ParamsY[0] == "MassAboveAfter_Infall_to_GasLost"):
            x = np.linspace(0, 1)
            y = -x
            ax.plot(x, y, color="darkorange", linestyle="dashed", lw=2)


            ax.axvline(0,color = 'black',linestyle='dashed',lw=2)
            ax.axhline(0,color = 'black',linestyle='dashed',lw=2)
            
            ax.fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
            ax.fill_between([-500, 0], 0, 500, alpha=0.2, color='tab:red')  # orange
            ax.fill_between([0, 500], 0, 500, alpha=0.2, color='tab:blue')  # red
            ax.text(-.145, -0.95, 'TS', fontsize = 0.98*fontlabel)
            ax.text(0.1, 0.02, 'SF', fontsize = 0.98*fontlabel)
            ax.text(0.15,-0.95,  'Interplay', fontsize = 0.98*fontlabel)
            
        elif (ParamX == 'Relative_logInnerZ_At_Entry' and  (ParamsY[0] == 'Relative_logZ_At_Entry')) :
            xfitline  = np.linspace(0 ,1, 100)
            axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth)
            
        elif (ParamX == 'Relative_Rhalf_MaxProfile_Minus_HalfRadstar_Entry' and  (ParamsY[0] == 'Relative_Rhalf_MinProfile_Minus_HalfRadstar_Entry')) :

            xfitline  = np.linspace(-6 ,2, 100)
            axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth)
            axs[i][j].fill_between(xfitline, -7, xfitline, alpha=0.2, color='tab:red')  # orange
            axs[i][j].text(0.2, -1.25, "TS", fontsize = 0.99*fontlabel)
            axs[i][j].fill_between(xfitline, xfitline, 1, alpha=0.2, color='tab:blue')  # orange
            axs[i][j].text(-1.8,-1., "SF", fontsize = 0.99*fontlabel)
            
        elif (ParamX == 'sSFRTrueInner_BeforeEntry' and  (ParamsY[0] == 'sSFRTrueInner_Entry_to_Nogas')) :

           x = np.linspace(-12,-8)
           axs[i][j].plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)
        
           xfitline  = np.linspace(-13 ,-7, 100)
           axs[i][j].fill_between(xfitline, -12, xfitline, alpha=0.2, color='tab:red')  # orange
           axs[i][j].text(-10,-10.55, "Inner $\overline{\mathrm{sSFR}}$ \n decrease", fontsize = 0.99*fontlabel)
           axs[i][j].fill_between(xfitline, xfitline,-8, alpha=0.2, color='tab:blue')  # orange
           axs[i][j].text(-10.9,-9.5, "Inner $\overline{\mathrm{sSFR}}$  \n increase", fontsize = 0.99*fontlabel)

     

        elif (ParamX == "Decrease_Entry_To_NoGas_Norm_Delta" and (firstY == "Decrease_NoGas_To_Final_Norm_Delta")):
            xfitline = np.linspace(-2, 0.4, 100)
            ax.fill_between(xfitline, -1, xfitline, alpha=0.2, color="tab:red")
            ax.fill_between(xfitline, xfitline, 0.25, alpha=0.2, color="tab:blue")
            ax.text(-0.6, -0.8, "Faster compaction \n after gas loss", fontsize=0.99 * fontlabel)
            ax.text(-0.80, -0.25, "Faster compaction  \n with gas ", fontsize=0.99 * fontlabel)
            ax.axvline(0, color="black", linestyle="dashed", lw=linewidth)
            ax.axhline(0, color="black", linestyle="dashed", lw=linewidth)
            ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
            ax.set_xticklabels(["-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2"])
            ax.set_yticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
            ax.set_yticklabels(["-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2"])

        elif (ParamX == "Rhalf_MaxProfile_Minus_HalfRadstar_Entry" and (firstY == "Rhalf_MinProfile_Minus_HalfRadstar_Entry")):
            xfitline = np.linspace(-6, 2, 100)
            ax.plot(xfitline, xfitline, ls="--", color="tab:blue", linewidth=linewidth)
            ax.fill_between(xfitline, -7, xfitline, alpha=0.2, color="tab:blue")
            ax.text(-2, -5, "TS", fontsize=0.99 * fontlabel)
            ax.fill_between(xfitline, xfitline, 1, alpha=0.2, color="tab:red")
            ax.text(-4.0, -1, "SF", fontsize=0.99 * fontlabel)
            
        elif  ('StarFrac' in ParamX  and (('GasFrac' in firstY) or ('DMFrac' in firstY)) )  :

            x = np.linspace(0, 1)
            ax.plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)

        if ParamX == "AgeBorn":
            x = np.arange(14)
            ax.plot(x, x, color="black", linestyle="dashed", lw=2)

    def _scatter_one(ax, x, y, name, color_values=None, marker_flags=None):
        """
        Scatter a single group (one 'name') in a single panel.
        """
        sc_local = None
        norm_local = None
    
        # Edgecolor logic (preserve original intent)
        if "BadFlag" in name:
            edcolor = "red"
        elif NoneEdgeColor:
            edcolor = None
        else:
            edcolor = "black"
    
        # 1) MarkerSizes mode
        if marker_flags is not None:
            Markers = marker_flags
    
            ax.scatter(x[Markers <= 1], y[Markers <= 1],
                       color=colors(name), edgecolor=edgecolors(name),
                       alpha=alphaScater, lw=linesthicker(name),
                       marker=markers(name), s=20)
    
            ax.scatter(x[Markers == 2], y[Markers == 2],
                       color=colors(name), edgecolor=edgecolors(name),
                       alpha=alphaScater, lw=linesthicker(name),
                       marker=markers(name), s=45)
    
            ax.scatter(x[Markers >= 3], y[Markers >= 3],
                       color=colors(name), edgecolor=edgecolors(name),
                       alpha=alphaScater, lw=linesthicker(name),
                       marker=markers(name), s=120)
    
            return None, None
    
        # 2) COLORBAR mode
        if color_values is not None:
            sc_local, norm_local = _scatter_with_colorbar(
                ax=ax,
                x=x, y=y,
                color_values=color_values,
                colorbar_key=COLORBAR[0],     # uses outer-scope COLORBAR
                names_l=name,
                cmap_name=cmap,              # uses outer-scope cmap variable
                alpha_scatter=alphaScater,
                linewidth=linewidth,
                msizet=msizet,
                HIGHLIGHTPoints=HIGHLIGHTPoints,
            )
            return sc_local, norm_local
    
        # 3) Normal mode
        ax.scatter(x, y,
                   color=colors(name),
                   edgecolor=edcolor,
                   alpha=alphaScater,
                   lw=linesthicker(name),
                   marker=markers(name),
                   s=msizet * msize(name))
        return None, None

    def _apply_post_panel_formatting(ax, yparam):
        if GridMake:
            ax.grid(GridMake, color="#9e9e9e", which="major", linewidth=0.6, alpha=0.3, linestyle=":")

        ax.set_yscale(scales(yparam))

        if scales(yparam) in ("log", "symlog"):
            ax.yaxis.set_major_formatter(FuncFormatter(format_func_loglog))

    def _add_colorbar(fig, axs, sc, norm=None):
        """
        Add a colorbar for the scatter plot.
        """
        if sc is None and norm is None:
            return None
    
        # If someone accidentally passed (sc, norm) as a tuple, unpack it.
        if isinstance(sc, tuple) and len(sc) == 2:
            sc, norm = sc
    
        # If still no mappable, nothing to do
        if sc is None and norm is None:
            return None
    
        cmap_obj = plt.cm.get_cmap(cmap)  # uses outer-scope `cmap` string
    
        # If a norm is given, prefer a ScalarMappable (works even if sc is None)
        if norm is not None:
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        else:
            mappable = sc  # PathCollection from ax.scatter
    
        # --- Now paste your original ticks/special cases logic ---
        if "Snap" in COLORBAR[0]:
            cb = fig.colorbar(
                mappable,
                ax=axs.ravel().tolist(),
                ticks=[0.0, 1.97185714, 3.94371429, 5.91557143, 7.88742857, 9.85928571, 11.83114286, 13.803],
                pad=0.02, aspect=30,
            )
            cb.ax.set_yticklabels(["14", "12", "10", "8", "6", "4", "2", "0"])
        
        else:
            if COLORBAR[0] == 'sSFRRatioPericenter':
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0, 0.25, 0.5, 0.75, 1,  2], pad=0.02, aspect=(ratioColorbar or 50))
                cb.ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1', '2'])
            elif COLORBAR[0] == 'logStarZ_99':
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0, 0.1, 0.2, 0.3, 0.7], pad=0.02, aspect=(ratioColorbar or 50))

                cb.ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.7'])
                
            else:
                cb = fig.colorbar(mappable, ax=axs.ravel().tolist(), pad=0.02, aspect=(ratioColorbar or 50))
    
        cb.set_label(labels.get(COLORBAR[0], COLORBAR[0]), fontsize=1.2 * fontlabel)
        cb.ax.tick_params(labelsize=0.99 * fontlabel)
        return cb
    # -----------------------------
    # Main
    # -----------------------------
    columns, ParamsX, ParamsY, label_general = _normalize_inputs(columns, ParamX, ParamsY)

    # Needed for z titles when using Snap
    dfTime = TNG.extractDF("SNAPS_TIME")

    panel_cols_for_data, dataX, dataY, dataColor, dataMarker = _load_data(names, columns, ParamsX, ParamsY)

    if len(snap) > 1:
        panel_columns = np.full(len(snap), "Snap")
    else:
        panel_columns = columns

    fig, axs = _setup_axes(panel_columns, nrows=len(ParamsY))

    sc_for_colorbar = None  # last valid scatter handle with colormap

    for i, yparam in enumerate(ParamsY):
        for j, colname in enumerate(panel_columns):
            ax = axs[i][j]

            # Special background rules (quadrants, guide lines, etc.)
            _apply_special_background_rules(ax, ParamX, ParamsY[0], linewidth, fontlabel)

            # Background "All" layer
            if All is not None:
                xAll = All[ParamX]
                yAll = All[yparam]
                ax.scatter(xAll, yAll, color=colors["All"], edgecolor=colors["All"], alpha=1.0, marker=".", s=10)

            # Optional Spearman accumulation
            if SpearManTestAll:
                XAllSMT = np.array([])
                YAllSMT = np.array([])
                CAllSMT = np.array([]) if COLORBAR is not None else None

            # Loop over each group in `names`
            for l in range(len(names)):
                name = names[l]

                # Preserve InvertPlot behavior (same as original)
                idx = l
                if InvertPlot and j == 1:
                    idx = len(names) - l - 1

                x = np.array(dataX[i][j][idx])
                y = np.array(dataY[i][j][idx])

                good = (~np.isnan(y)) & (~np.isinf(y))
                x_plot = x[good]
                y_plot = y[good]

                # Colorbar values for this group/panel
                cvals = None
                if dataColor is not None:
                    c_all = np.array(dataColor[i][j][idx])
                    cvals = c_all[good]

                # MarkerSizes flags for this group/panel
                mflags = None
                if dataMarker is not None:
                    # Bugfix: index per panel and per group (i, j, idx)
                    m_all = np.array(dataMarker[i][j][idx])
                    mflags = m_all[good]

                # Spearman test per group
                if SpearManTest and not SpearManTestAll:
                    corr, pval = spearmanr(
                        x_plot[(~np.isnan(x_plot)) & (~np.isinf(x_plot))],
                        y_plot[(~np.isnan(x_plot)) & (~np.isinf(x_plot))],
                    )
                    print("Name:", name, "corr:", corr, "p:", pval)

                # Scatter
                sc_local = _scatter_one(ax, x_plot, y_plot, name, color_values=cvals, marker_flags=mflags)
                if sc_local is not None:
                    sc_for_colorbar = sc_local

                # Medians / quantiles
                if medianBins:
                    xmean, ymed, yq_hi, yq_lo = MATH.split_quantiles(
                        x_plot, y_plot, total_bins=bins, quantile=quantile
                    )
                    ax.errorbar(
                        xmean, ymed,
                        yerr=(ymed - yq_lo, yq_hi - ymed),
                        ls="None", markeredgecolor="black", elinewidth=2, ms=10,
                        fmt="s", c=colors[name],
                    )

                elif medianDot:
                    ax.scatter(
                        np.nanmedian(x_plot), np.nanmedian(y_plot),
                        marker="*", edgecolor="black",
                        c=colors(name), s=450, lw=1.5,
                    )

                elif medianAll:
                    xmean, ymed, yq_hi, yq_lo = MATH.split_quantiles(
                        x_plot, y_plot, total_bins=bins
                    )
                    ax.plot(
                        xmean, ymed,
                        color=colors(name), ls=lines(name), linewidth=linewidth,
                    )
                    ax.fill_between(xmean, yq_lo, yq_hi, color=colors(name), alpha=alphaShade)

                # Accumulate Spearman arrays
                if SpearManTestAll:
                    XAllSMT = np.append(XAllSMT, x_plot)
                    YAllSMT = np.append(YAllSMT, y_plot)
                    if CAllSMT is not None and cvals is not None:
                        CAllSMT = np.append(CAllSMT, cvals)

            # Spearman test using all points in this panel
            if SpearManTestAll:
                cond = (
                    (~np.isnan(XAllSMT)) & (~np.isinf(XAllSMT))
                    & (~np.isnan(YAllSMT)) & (~np.isinf(YAllSMT))
                )
                corr, pval = spearmanr(XAllSMT[cond], YAllSMT[cond])
                print("Panel Spearman corr:", corr, "p:", pval)

            # Equal line if requested
            if EqualLine and (EqualLineMin is not None) and (EqualLineMax is not None):
                xx = np.linspace(EqualLineMin, EqualLineMax)
                ax.plot(xx, xx, ls="--", color="tab:blue", linewidth=linewidth)

            # Panel formatting
            _apply_post_panel_formatting(ax, yparam)

            # Y label on first column
            if j == 0:
                if label_general:
                    ax.set_ylabel(labelsequal.get(yparam, yparam), fontsize=1.2 * fontlabel)
                else:
                    ax.set_ylabel(labels.get(yparam, yparam), fontsize=1.2 * fontlabel)
                ax.tick_params(axis="y", labelsize=0.99 * fontlabel)

            # Title on first row
            if i == 0:
                if panel_columns[j] == "Snap":
                    ax.set_title(
                        r"$z = %.1f$" % dfTime.z.loc[dfTime.Snap == snap[j]].values[0],
                        fontsize=1.1 * fontlabel,
                    )
                if title:
                    ax.set_title(titles(title[j]), fontsize=1.1 * fontlabel)

            # X label on last row
            if i == len(ParamsY) - 1:
                if label_general:
                    ax.set_xlabel(labelsequal.get(ParamsX[j], ParamsX[j]), fontsize=1.2 * fontlabel)
                    ax.set_xscale(scales(ParamsX[j]))
                    if scales(ParamsX[j]) in ("log", "symlog"):
                        ax.xaxis.set_major_formatter(FuncFormatter(format_func_loglog))
                else:
                    ax.set_xlabel(labels.get(ParamX, ParamX), fontsize=1.2 * fontlabel)
                    if xscale != None:
                        ax.set_xscale(xscale)
                    else:
                        ax.set_xscale(scales(ParamX))
                    if xscale == None and scales(ParamX) in ("log", "symlog"):
                        ax.xaxis.set_major_formatter(FuncFormatter(format_func_loglog))
                        
                ax.tick_params(axis="x", labelsize=0.99 * fontlabel)
                
                _apply_special_xaxis_rules(
                                            ax=ax,
                                            ParamX=ParamX,
                                            ParamsY=ParamsY,
                                            yparam=yparam,     
                                            ylimmin=ylimmin,
                                            fontlabel=fontlabel,
                                        )

                if xlimmin is not None and xlimmax is not None:
                    ax.set_xlim(xlimmin[i], xlimmax[i])
                    
                if ylimmin is not None and ylimmax is not None:
                    ax.set_ylim(ylimmin[i], ylimmax[i])

                

            # Legend at specific panel positions
            if legend and LegendNames is not None and legpositions is not None:
                for legpos, LegendName in enumerate(LegendNames):
                    if j == legpositions[legpos][0] and i == legpositions[legpos][1]:
                        custom_lines, label, ncol, _mult = Legend(
                            LegendName, msizeMult=msizeMult, linewidth=linewidth
                        )
                        ax.legend(
                            custom_lines, label,
                            ncol=ncol, loc=loc[legpos],
                            fontsize=0.88 * fontlabel, framealpha=framealpha,
                            columnspacing=columnspacing,
                            handlelength=handlelength,
                            handletextpad=handletextpad,
                            labelspacing=labelspacing,
                        )

    # Global colorbar (preserve your original block here)
    _add_colorbar(fig, axs, sc_for_colorbar)

    savefig(savepath, savefigname, TRANSPARENT)
    return


def PlotID(
    # --- Grid definition / inputs (required) ---
    columns,  rows, IDs,

    # --- What to plot (data selection) ---
    Type: str = "Evolution", Xparam="Time", dfName: str = "Sample", SampleName: str = "Samples",
    SIM=SIMTNG, fmt: str = "csv", TreeHybridSubhalo: bool = False,

    # --- Panel mapping / layout logic ---
    ColumnPlot: bool = True, IDColumn: bool = False, title=False, xlabelintext: bool = False, limaxis: bool = False,

    # --- Optional overlays / annotations ---
    dataMarker=None, dataLine=None, sSFRMedian: bool = False,
    Softening: bool = False, Pericenter: bool = False, LookBackTime: bool = False, QuantileError: bool = True,    
    
    # --- Styling (axes/figure) ---
    yscale: str = "linear", GridMake: bool = False,alphaShade: float = 0.3, linewidth: float = 0.5, fontlabel: int = 24,

    # --- Limits ---
    ylimmin=None, ylimmax=None, xlimmin=None,  xlimmax=None,

    # --- Legend control ---
    legend: bool = False, LegendNames="None", legpositions=None,
    postext=("best",), loc="best", framealpha: float = 0.95, columnspacing: float = 0.5, handlelength: float = 2,
    handletextpad: float = 0.4, labelspacing: float = 0.3,

    # --- Figure size / export ---
    lNum: float = 6,  cNum: float = 6, savepath: str = "PlotID", savefigname: str = "fig", TRANSPARENT: bool = False,

    # --- Stochastic / computation controls ---
    nboots: int = 100, bins: int = 10, seed: int = 16010504,

    # --- Backward-compat / rarely used ---
    lineparams: bool = False,
):
    """
    Plot the evolution or co-evolution for selected subhalo IDs.
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    """

    # -----------------------------
    # Helpers
    # -----------------------------
    def _as_list(x):
        if isinstance(x, (list, np.ndarray)):
            return list(x)
        return [x]

    def _pad_to_length(arr, n):
        """Pad 1D array with NaNs to length n."""
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size >= n:
            return arr[:n]
        out = np.full(n, np.nan, dtype=float)
        out[: arr.size] = arr
        return out

    def _safe_series(df, key):
        """Return df[str(key)] as 1D float array; raise KeyError if missing."""
        s = df[str(key)].values
        s = np.asarray(s)
        if s.ndim > 1:
            s = s.T[0]
        return np.asarray(s, dtype=float).ravel()

    def _get_df_for_panel(row_param, col_param, argIDs, i, j):
        if ColumnPlot:
            dataY = TNG.makeDF(
                row_param,
                col_param,
                dfName=dfName,
                IDs=IDs[argIDs],
                TreeHybridSubhalo=TreeHybridSubhalo,
                SIM=SIM,
            )
            dataX = None
            if Type == "CoEvolution":
                dataX = TNG.makeDF(
                    Xparam[j],
                    dfName=dfName,
                    IDs=IDs[argIDs],
                    TreeHybridSubhalo=TreeHybridSubhalo,
                    SIM=SIM,
                )
        else:
            dataY = TNG.makeDF(
                col_param,
                row_param,
                dfName=dfName,
                IDs=IDs[argIDs],
                TreeHybridSubhalo=TreeHybridSubhalo,
                SIM=SIM,
            )
            dataX = None
            if Type == "CoEvolution":
                dataX = TNG.makeDF(
                    col_param,
                    Xparam[i],
                    dfName=dfName,
                    IDs=IDs[argIDs],
                    TreeHybridSubhalo=TreeHybridSubhalo,
                    SIM=SIM,
                )
        return dataY, dataX

    def _get_marker_df_for_panel(col_param, argIDs):
        """
        Return:
          - datamarkervalues
          - dataMarkervalues (major)
          - datamarkerTotvalues (total)
        depending on whether "Merger" in dataMarker.
        """
        if dataMarker is None:
            return None, None, None

        if "Merger" in str(dataMarker):
            datamarkerTotvalues = TNG.makeDF(
                col_param,
                "NumMergersTotal",
                dfName=dfName,
                IDs=IDs[argIDs],
                TreeHybridSubhalo=TreeHybridSubhalo,
                SIM=SIM,
            )
            dataMarkervalues = TNG.makeDF(
                col_param,
                "NumMajorMergersTotal",
                dfName=dfName,
                IDs=IDs[argIDs],
                TreeHybridSubhalo=TreeHybridSubhalo,
                SIM=SIM,
            )
            datamarkervalues = TNG.makeDF(
                col_param,
                "NumMinorMergersTotal",
                dfName=dfName,
                IDs=IDs[argIDs],
                TreeHybridSubhalo=TreeHybridSubhalo,
                SIM=SIM,
            )
            return datamarkervalues, dataMarkervalues, datamarkerTotvalues

        datamarkervalues = TNG.makeDF(
            col_param,
            dataMarker,
            dfName=dfName,
            IDs=IDs[argIDs],
            TreeHybridSubhalo=TreeHybridSubhalo,
            SIM=SIM,
        )
        return datamarkervalues, None, None

    def _compute_merger_deltas(minor_tot, major_tot, all_tot):
        minor = np.flip(minor_tot)
        major = np.flip(major_tot)
        allm = np.flip(all_tot)

        minor_delta = np.zeros_like(minor)
        major_delta = np.zeros_like(major)
        all_delta = np.zeros_like(allm)

        for k in range(1, minor.size):
            if np.isnan(minor[k]):
                minor_delta[k] = 0
            else:
                prev = minor[k - 1]
                minor_delta[k] = int(minor[k]) if np.isnan(prev) else int(minor[k]) - int(prev)

        for k in range(1, major.size):
            if np.isnan(major[k]):
                major_delta[k] = 0
            else:
                prev = major[k - 1]
                major_delta[k] = int(major[k]) if np.isnan(prev) else int(major[k]) - int(prev)

        for k in range(1, allm.size):
            if np.isnan(allm[k]):
                all_delta[k] = 0
            else:
                prev = allm[k - 1]
                all_delta[k] = int(allm[k]) if np.isnan(prev) else int(allm[k]) - int(prev)

        other_delta = all_delta - major_delta - minor_delta

        # Flip back to forward-time orientation
        return np.flip(minor_delta), np.flip(major_delta), np.flip(other_delta)

    def _add_top_z_axis(ax, row_param):
        lim = ax.get_xlim()
        ax2 = ax.twiny()
        ax2.grid(False)
        ax2.set_xlim(lim)

        is_young_mode = (row_param == "rToRNearYoung") or (savefigname == "Young")

        if is_young_mode:
            zlabels = ["0", "0.2"]
            zticks_Age = [13.803, 11.323]
        else:
            zlabels = ["0", "0.2", "0.5", "1", "2", "5", "20"]
            zticks_Age = [13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0.0]

        ax2.xaxis.set_major_locator(FixedLocator(zticks_Age))
        ax2.xaxis.set_major_formatter(FixedFormatter(zlabels))
        ax2.set_xlabel(r"$z$", fontsize=fontlabel)
        ax2.tick_params(labelsize=0.99 * fontlabel)

    def _format_axes(ax, i, j, row_param, col_param):
        """Panel styling + labels + legend + scale formatting."""
        if GridMake:
            ax.grid(GridMake, color="#9e9e9e", which="major", linewidth=0.6, alpha=0.3, linestyle=":")

        ax.tick_params(axis="y", labelsize=0.99 * fontlabel)
        ax.tick_params(axis="x", labelsize=0.99 * fontlabel)

        # y-limits
        if ylimmin is not None and ylimmax is not None:
            ax.set_ylim(ylimmin[i], ylimmax[i])

        # y-scale: preserve your behavior
        if ColumnPlot:
            yscale_use = scales(col_param)
        else:
            yscale_use = scales(row_param)

        ax.set_yscale(yscale_use)
        if yscale_use == "log":
            ax.yaxis.set_major_formatter(FuncFormatter(format_func_loglog))

        # legend
        if legend:
            for legpos, LegendName in enumerate(LegendNames):
                if (j == legpositions[legpos][0]) and (i == legpositions[legpos][1]):
                    custom_lines, label, ncol, mult = Legend(LegendName)
                    ax.legend(
                        custom_lines,
                        label,
                        ncol=ncol,
                        loc=loc[legpos],
                        fontsize=0.88 * fontlabel,
                        framealpha=framealpha,
                        columnspacing=columnspacing,
                        handlelength=handlelength,
                        handletextpad=handletextpad,
                        labelspacing=labelspacing,
                    )

        # left y-labels
        if j == 0:
            if xlabelintext:
                ax.set_ylabel(labelsequal.get(row_param, row_param), fontsize=fontlabel)
            else:
                if ColumnPlot:
                    ax.set_ylabel(labels.get(col_param, col_param), fontsize=fontlabel)
                else:
                    ax.set_ylabel(labels.get(row_param, row_param), fontsize=fontlabel)

        # in-panel text label at last column
        if j == len(columns) - 1 and xlabelintext and (not limaxis) and (len(rows) > 1):
            Afont = {"color": "black", "size": fontlabel}
            anchored_text = AnchoredText(texts.get(row_param, row_param), loc="upper right", prop=Afont)
            ax.add_artist(anchored_text)

        if xlabelintext and limaxis and (len(rows) > 1):
            Afont = {"color": "black", "size": fontlabel}
            anchored_text = AnchoredText(texts.get(row_param, row_param), loc="upper left", prop=Afont)
            ax.add_artist(anchored_text)

        # "title in first column" for ColumnPlot
        if (j == 0) and (len(rows) > 1) and title and ColumnPlot:
            Afont = {"color": "black", "size": fontlabel}
            anchored_text = AnchoredText(titles(title[i]), loc=postext[i], prop=Afont)
            ax.add_artist(anchored_text)

        # top titles for not-ColumnPlot mode
        if (i == 0) and title and (not ColumnPlot):
            ax.set_title(titles(title[j]), fontsize=1.1 * fontlabel)

    def _format_bottom_x_axis(ax, i, j, row_param, col_param, xparam_here):
        """Bottom row x-label formatting."""
        if i != len(rows) - 1:
            return

        if Type == "Evolution":
            ax.set_xlabel(r"$t \, \,  [\mathrm{Gyr}]$", fontsize=fontlabel)

            is_young_mode = (row_param == "rToRNearYoung") or (savefigname == "Young")

            if (xparam_here == "tsincebirth") or is_young_mode:
                # Preserve your ticks
                ax.set_xticks([10, 12, 14])
                ax.set_xticklabels(["10", "12", "14"])

                if xparam_here == "tsincebirth":
                    ax.set_xticks([0, 1, 2, 3, 4])
                    ax.set_xticklabels(["0", "1", "2", "3", "4"])
                    ax.set_xlabel(r"$t - t_\mathrm{birth} \, [\mathrm{Gyr}]$", fontsize=fontlabel)
                    ax.set_xlim(-0.09, 4.2)
            else:
                if LookBackTime:
                    ax.set_xlabel(r"$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$", fontsize=fontlabel)
                    ax.set_xticks([0.0, 1.97185714, 3.94371429, 5.91557143, 7.88742857, 9.85928571, 11.83114286, 13.803])
                    ax.set_xticklabels(["14", "12", "10", "8", "6", "4", "2", "0"])
                else:
                    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                    ax.set_xticklabels(["0", "2", "4", "6", "8", "10", "12", "14"])
                    ax.set_xlabel(r"$t \, \, [\mathrm{Gyr}]$", fontsize=fontlabel)

        elif Type == "CoEvolution":
            xscale_use = scales(xparam_here)
            ax.set_xscale(xscale_use)
            if xscale_use == "log":
                ax.xaxis.set_major_formatter(FuncFormatter(format_func_loglog))
            ax.set_xlabel(labels.get(xparam_here, xparam_here), fontsize=fontlabel)

    # -----------------------------
    # Begin function body
    # -----------------------------
    np.random.seed(seed)

    # Load time table
    time = np.asarray(dfTime.Age.values, dtype=float)

    snapsTime = np.array([88, 81, 64, 51, 37, 24], dtype=int)

    columns = _as_list(columns)
    rows = _as_list(rows)

    # Xparam can be str or list; we need indexable in your legacy logic
    if isinstance(Xparam, (list, np.ndarray)):
        Xparam_list = list(Xparam)
    else:
        # replicate across rows (Evolution uses Xparam[i])
        Xparam_list = [Xparam for _ in range(max(len(rows), len(columns)))]

    # Defensive check: IDs should be list-of-lists
    if not isinstance(IDs, (list, tuple)) or (len(IDs) == 0):
        raise ValueError("IDs must be a non-empty list of ID-lists, e.g. IDs=[ [id1,id2], [id3,...] , ... ].")

    # Create axes grid
    plt.rcParams.update({"figure.figsize": (cNum * len(columns), lNum * len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex="col", sharey="row")

    # Normalize axs shape to 2D array
    if not isinstance(axs, np.ndarray):
        axs = np.array([[axs]])
    elif axs.ndim == 1:
        # either one row or one column
        if len(rows) == 1:
            axs = axs.reshape(1, -1)
        else:
            axs = axs.reshape(-1, 1)

    # Optional: pericenter helper data
    r_over_R_Crit200 = None
    if Pericenter:
        r_over_R_Crit200 = TNG.extractDF("r_over_R_Crit200", SIM=SIM, fmt=fmt)

    # Loop panels
    for i, row_param in enumerate(rows):
        for j, col_param in enumerate(columns):
            ax = axs[i][j]

            # Decide which IDs list to use (preserve original behavior)
            argIDs = i if ColumnPlot else j
            if argIDs >= len(IDs):
                # safer than silent wrong indexing
                raise IndexError(
                    f"argIDs={argIDs} out of range for IDs (len={len(IDs)}). "
                    f"Check ColumnPlot and the shape of IDs."
                )

            # Build panel dataframes
            dataY, dataX = _get_df_for_panel(row_param, col_param, argIDs, i, j)

            # Optional line and marker dataframes
            datalinevalues = None
            if dataLine is not None:
                datalinevalues = TNG.makeDF(
                    col_param if ColumnPlot else col_param,
                    dataLine,
                    dfName=dfName,
                    IDs=IDs[argIDs],
                    TreeHybridSubhalo=TreeHybridSubhalo,
                    SIM=SIM,
                )

            datamarkervalues, dataMarkervalues, datamarkerTotvalues = _get_marker_df_for_panel(col_param, argIDs)

            # Softening curve (preserve condition)
            if Softening and (row_param == "SubhaloHalfmassRadType4"):
                rSoftening = ETNG.Softening()
                rSoftening = np.flip(np.asarray(rSoftening, dtype=float))
                ok = (~np.isinf(rSoftening)) & (~np.isnan(rSoftening))
                ax.plot(time[ok], np.log10(rSoftening[ok]), color="black", ls="solid", lw=2 * linewidth)

            # sSFR median band (preserve condition)
            if sSFRMedian and (row_param == "SubhalosSFRInHalfRad"):
                Y, Yerr = TNG.makedataevolution(
                    [""], ["Central"], ["SubhalosSFRInHalfRad"], SampleName=SampleName, dfName=dfName, nboots=nboots
                )
                Y = np.asarray([v for v in Y[0][0][0]], dtype=float)
                Yerr = np.asarray([v for v in Yerr[0][0][0]], dtype=float)
                ax.plot(time, Y, color="grey", ls="solid", lw=2 * linewidth)
                ax.fill_between(time, Y - 4 * Yerr, Y + 4 * Yerr, color="grey", alpha=0.5)

            # Iterate IDs in this panel
            for l, IDvalue in enumerate(IDs[argIDs]):

                # Robustly extract series for this ID
                try:
                    values_raw = _safe_series(dataY, IDvalue)
                except Exception:
                    # Missing ID column in DF -> skip safely
                    continue

                # Pad/truncate to time length (instead of hardcoded 100)
                values = _pad_to_length(values_raw, len(time))

                if Type == "Evolution":
                    if row_param == "r_over_R_Crit200_WithoutCorrection":
                        values[values == 0] = np.nan

                    argnotnan = ~np.isnan(values)
                    if np.sum(argnotnan) == 0:
                        continue

                    # x-axis for evolution
                    xparam_here = Xparam_list[i] if i < len(Xparam_list) else "Time"
                    if xparam_here == "tsincebirth":
                        TimeBirth = time[argnotnan] - time[argnotnan][-1]
                        ax.plot(TimeBirth, values[argnotnan], color=colors(str(l)),
                                ls=lines(str(l)), lw=linewidth)
                    else:
                        ax.plot(time[argnotnan], values[argnotnan], color=colors(str(l)),
                                ls=lines(str(l)), lw=linewidth)

                    # Pericenter markers
                    if Pericenter and (r_over_R_Crit200 is not None):
                        try:
                            rOveR200 = _pad_to_length(_safe_series(r_over_R_Crit200, IDvalue), len(time))
                            rOveR200[rOveR200 > 1] = np.nan
                            args = argrelextrema(rOveR200, np.less)[0]
                            for arg in args:
                                if np.isfinite(values[arg]):
                                    ax.scatter(time[arg], values[arg], color=colors(str(l)),
                                               marker="X", s=30, edgecolor="black")
                        except Exception:
                            pass

                    # dataLine highlighting
                    if datalinevalues is not None:
                        try:
                            linevalues = _pad_to_length(_safe_series(datalinevalues, IDvalue), len(time))
                            ok = (~np.isinf(linevalues)) & (~np.isnan(linevalues)) & (~np.isnan(values))
                            ax.plot(time[ok], values[ok], color=colors(str(l)),
                                    ls=lines(str(l)), lw=2 * linewidth)
                        except Exception:
                            pass

                    # Marker logic
                    if dataMarker is not None and datamarkervalues is not None:
                        try:
                            markervalues_raw = _pad_to_length(_safe_series(datamarkervalues, IDvalue), len(time))
                            if "Merger" in str(dataMarker):
                                minor_tot = markervalues_raw
                                major_tot = _pad_to_length(_safe_series(dataMarkervalues, IDvalue), len(time))
                                all_tot = _pad_to_length(_safe_series(datamarkerTotvalues, IDvalue), len(time))
                                minor_d, major_d, other_d = _compute_merger_deltas(minor_tot, major_tot, all_tot)

                                ax.scatter(time[major_d > 0], values[major_d > 0], color=colors(str(l)),
                                           lw=1.0, marker="o", edgecolors="black", s=250, alpha=0.7)
                                ax.scatter(time[minor_d > 0], values[minor_d > 0], color=colors(str(l)),
                                           lw=1.0, marker="s", edgecolors="black", s=100, alpha=0.7)
                                ax.scatter(time[other_d > 0], values[other_d > 0], color=colors(str(l)),
                                           lw=1.0, marker="s", edgecolors="black", s=100, alpha=0.7)
                            else:
                                ax.scatter(time[markervalues_raw > 0], values[markervalues_raw > 0],
                                           color=colors(str(l)),
                                           lw=1.0, marker="o", edgecolors="black", s=130, alpha=0.5)
                        except Exception:
                            pass

                elif Type == "CoEvolution":
                    if dataX is None:
                        continue

                    try:
                        x_raw = _safe_series(dataX, IDvalue)
                    except Exception:
                        continue

                    x = _pad_to_length(x_raw, len(time))
                    values = _pad_to_length(values, len(time))

                    colorSnap = np.array(["magenta", "blue", "cyan", "lime", "darkorange", "red"])
                    xparam_here = Xparam_list[i] if i < len(Xparam_list) else Xparam_list[0]

                    if xparam_here != "tsincebirth":
                        # Snap highlighting
                        idx = 99 - snapsTime
                        idx = idx[(idx >= 0) & (idx < len(x))]
                        ax.scatter(x[idx], values[idx], color=colorSnap[: len(idx)],
                                   lw=1.0, marker="d", edgecolors=colors(col_param),
                                   s=100, alpha=0.9)
                        ax.scatter(x[0], values[0], color="black", lw=1.0, marker="o",
                                   edgecolors=colors(col_param), s=70, alpha=0.9)

                    argnotnan = ~np.isnan(values) & ~np.isnan(x)
                    ax.plot(x[argnotnan], values[argnotnan], color=colors(str(l)),
                            ls=lines(col_param))

                    # dataLine highlighting in co-evolution
                    if datalinevalues is not None:
                        try:
                            linevalues = _pad_to_length(_safe_series(datalinevalues, IDvalue), len(time))
                            ok = (~np.isinf(linevalues)) & (~np.isnan(linevalues)) & (~np.isnan(values)) & (~np.isnan(x))
                            ax.plot(x[ok], values[ok], color=colors(str(l)),
                                    ls=lines(str(l)), lw=3.0)
                        except Exception:
                            pass

                    # Marker logic in co-evolution (kept close to Evolution behavior)
                    if dataMarker is not None and datamarkervalues is not None:
                        try:
                            markervalues_raw = _pad_to_length(_safe_series(datamarkervalues, IDvalue), len(time))
                            if "Merger" in str(dataMarker):
                                minor_tot = markervalues_raw
                                major_tot = _pad_to_length(_safe_series(dataMarkervalues, IDvalue), len(time))
                                all_tot = _pad_to_length(_safe_series(datamarkerTotvalues, IDvalue), len(time))
                                minor_d, major_d, other_d = _compute_merger_deltas(minor_tot, major_tot, all_tot)

                                ax.scatter(x[major_d > 0], values[major_d > 0], color=colors(str(l)),
                                           lw=1.0, marker="o", edgecolors="black", s=130, alpha=0.5)
                                ax.scatter(x[minor_d > 0], values[minor_d > 0], color=colors(str(l)),
                                           lw=1.0, marker="o", edgecolors="black", s=110, alpha=0.5)
                            else:
                                ax.scatter(x[markervalues_raw > 0], values[markervalues_raw > 0],
                                           color=colors(str(l)),
                                           lw=1.0, marker="o", edgecolors="black", s=110, alpha=0.5)
                        except Exception:
                            pass

            # Panel formatting
            _format_axes(ax, i, j, row_param, col_param)

            # Top z-axis (only for Evolution + not tsincebirth)
            if (i == 0) and (Type == "Evolution"):
                xparam_here = Xparam_list[i] if i < len(Xparam_list) else "Time"
                if xparam_here != "tsincebirth":
                    ax.tick_params(bottom=True, top=False)
                    _add_top_z_axis(ax, row_param)

            # Bottom axis formatting
            xparam_here = Xparam_list[i] if i < len(Xparam_list) else Xparam_list[0]
            _format_bottom_x_axis(ax, i, j, row_param, col_param, xparam_here)

            # Optional x-limits for co-evolution (preserve ability)
            if (Type == "CoEvolution") and (xlimmin is not None) and (xlimmax is not None):
                ax.set_xlim(xlimmin[i], xlimmax[i])

    # Save
    savefig(savepath, savefigname, TRANSPARENT=TRANSPARENT, SIM=SIM)
    return


def PlotIDsAllTogether(
    # --- Population definition (required) ---
    Names,  rows,
    # --- How to interpret `Names` / how to get IDs ---
    IDsNotNames: bool = False,  dfName: str = "Sample",  SampleName: str = "Samples", NameKey: str = "Name",
    # --- What to plot ---
    Type: str = "Evolution", Xparam: str = "Time",  PhasePlot: bool = False, xPhaseLim: float = 7,
    # --- Plot modes / overlays ---
    MedianPlot: bool = False, QuantileError: bool = True, Softening: bool = False, Pericenter: bool = False,
    LookBackTime: bool = False, InfallTime: bool = False, NoGas: bool = False, MaxSizeType: bool = False,
    # --- Styling / figure layout ---
    title=False, xlabelintext: bool = False,  lineparams: bool = False, ColumnPlot: bool = False,
    limaxis: bool = False,GridMake: bool = False, SmallerScale: bool = False,
    # --- Colormaps ---
    ColorMaps=None,
    # --- Axis limits ---
    ylimmax=None, ylimmin=None,
    # --- Size / typography ---
    lNum: float = 6, cNum: float = 6, linewidth: float = 0.5, fontlabel: int = 24, alphaShade: float = 0.3,
    # --- Legend (kept for API compatibility) ---
    legend: bool = False, LegendNames="None", loc: str = "best", postext=("best",), columnspacing: float = 0.5,
    handlelength: float = 2, handletextpad: float = 0.4, labelspacing: float = 0.3,
    # --- Statistics / reproducibility ---
    nboots: int = 100, bins: int = 10, seed: int = 16010504,
    # --- Output ---
    savepath: str = "fig/PlotIDsAllTogether", savefigname: str = "fig", TRANSPARENT: bool = False,
):
    """
    Plot the evolution (or phase evolution) of multiple IDs together, grouped by `Names`.
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    from matplotlib.ticker import FixedLocator, FixedFormatter, FuncFormatter

    # -----------------------------
    # Helpers
    # -----------------------------
    def _as_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        return [x]

    def _ensure_2d_axes(axs, nrows, ncols):
        """
        Always return axs as a (nrows, ncols) numpy array.
        Matplotlib returns:
        - Axes if nrows=ncols=1
        - 1D array if one of them is 1
        - 2D array otherwise
        """
        axs = np.asarray(axs, dtype=object)
        if axs.ndim == 0:
            axs = axs.reshape((1, 1))
        elif axs.ndim == 1:
            if nrows == 1 and ncols > 1:
                axs = axs.reshape((1, ncols))
            elif ncols == 1 and nrows > 1:
                axs = axs.reshape((nrows, 1))
            else:
                # fallback: try best reshape
                axs = axs.reshape((nrows, ncols))
        return axs

    def _nanmedian_safe(a, axis=None):
        try:
            return np.nanmedian(a, axis=axis)
        except Exception:
            return np.nan

    # -----------------------------
    # Seed + inputs normalization
    # -----------------------------
    np.random.seed(seed)
    Names = _as_list(Names)
    rows = _as_list(rows)

    if ColorMaps is None:
        ColorMaps = [plt.get_cmap("Reds")]
    else:
        ColorMaps = _as_list(ColorMaps)
        if len(ColorMaps) == 0:
            ColorMaps = [plt.get_cmap("Reds")]

    # If user provides 1 colormap for many Names, reuse it
    if len(ColorMaps) < len(Names):
        ColorMaps = (ColorMaps * (len(Names) // len(ColorMaps) + 1))[: len(Names)]

    # -----------------------------
    # Time array
    # -----------------------------
    dfTime = pd.read_csv(os.path.join(os.getenv("HOME"), "TNG_Analyzes/SubhaloHistory/SNAPS_TIME.csv"))
    time = dfTime.Age.values

    # -----------------------------
    # Figure + axes
    # -----------------------------
    plt.rcParams.update({"figure.figsize": (cNum * len(Names), lNum * len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(Names), hspace=0, wspace=0)
    axs = gs.subplots(sharex="col", sharey="row")
    axs = _ensure_2d_axes(axs, nrows=len(rows), ncols=len(Names))

    # -----------------------------
    # Common phase grid (if PhasePlot)
    # -----------------------------
    if PhasePlot:
        x_coarse = np.arange(-1.0, 9.0, 1.0)
        x_half = x_coarse + 0.5
        x_dense = np.linspace(-1.0, 9.0, 1000)
        x_phase_grid = np.unique(np.concatenate([x_coarse, x_half, x_dense]))
    else:
        x_phase_grid = None

    # -----------------------------
    # Main loop
    # -----------------------------
    for i, row in enumerate(rows):
        # Fetch the full DF for this row once (as in your original)
        df = TNG.extractDF(row)

        for j, Name in enumerate(Names):
            # Always try to get dfPopulation (original did this even when IDsNotNames=True)
            # because PhasePlot often needs metadata.
            dfPopulation = None
            try:
                dfPopulation = TNG.extractPopulation(Name, dfName=dfName, Name=NameKey)
            except Exception:
                dfPopulation = None

            # Determine IDs
            if not IDsNotNames:
                if dfPopulation is None:
                    continue
                IDs = dfPopulation["SubfindID_99"].values
            else:
                IDs = Name

            # Ensure iterable IDs (and avoid empty)
            try:
                IDs = np.array(list(IDs))
            except Exception:
                continue
            if IDs.size == 0:
                continue

            # Per-column colormap sampling (0..0.9 range)
            cmap = ColorMaps[j]
            colorsMap = cmap(np.linspace(0.0, 0.9, len(IDs)))

            # Optional: softening line (same condition as your original)
            if Softening and ("SubhaloHalfmassRadType4" in str(row)):
                try:
                    rSoftening = ETNG.Softening()
                    rSoftening = np.flip(rSoftening)
                    cond = (~np.isinf(rSoftening)) & (~np.isnan(rSoftening))
                    axs[i][j].plot(
                        time[cond],
                        np.log10(rSoftening[cond]),
                        color="black",
                        ls="solid",
                        lw=2 * linewidth,
                    )
                except Exception:
                    pass

            FinalValues = None
            xparam_ref = time  # default x-axis

            # -------------------------
            # Plot each ID
            # -------------------------
            for idindex, ID in enumerate(IDs):
                try:
                    values = np.array([v for v in df[str(ID)].values])
                except Exception:
                    continue

                # PhasePlot transformation
                if PhasePlot:
                    # Need phases
                    try:
                        phases = TNG.PhasingData(ID, dfPopulation)
                    except Exception:
                        phases = None

                    if not isinstance(phases, np.ndarray):
                        continue

                    # Keep only values with defined phases/values
                    cond_valid = (~np.isnan(values)) & (~np.isnan(phases))
                    phases = phases[cond_valid]
                    values = values[cond_valid]

                    if len(values) == 0:
                        continue

                    # Interpolate onto global grid (linear in y or in log10(y) depending on sign)
                    xparam = x_phase_grid.copy()

                    try:
                        if np.any(values < 0):
                            f = interp1d(phases, values, kind="linear", fill_value="extrapolate")
                            y_new = f(xparam)
                        else:
                            f = interp1d(phases, np.log10(values), kind="linear", fill_value="extrapolate")
                            y_new = 10 ** f(xparam)
                    except Exception:
                        continue

                    # Mask beyond last available phase (preserve your original intent)
                    y_new[xparam > np.nanmax(phases)] = np.nan

                    values = y_new
                    xparam_ref = xparam  # now x-axis is phase

                else:
                    xparam_ref = time

                # Special-case: first group truncation logic (preserved)
                if row == "r_over_R_Crit200_FirstGroup":
                    values = values.copy()
                    values[values == 0] = np.nan
                    argnan = np.argwhere(np.isnan(values)).T
                    if argnan.size > 0:
                        values[int(argnan[0][0]) :] = np.nan

                # Plot individual tracks when not MedianPlot
                if not MedianPlot:
                    cond = (~np.isnan(values)) & (~np.isinf(values))
                    axs[i][j].plot(
                        xparam_ref[cond],
                        values[cond],
                        color=colorsMap[idindex],
                        ls="solid",
                        lw=0.25 * linewidth,
                    )

                # Stack values to compute median lines later
                try:
                    FinalValues = values if FinalValues is None else np.vstack((FinalValues, values))
                except Exception:
                    continue

            # If nothing stacked, skip panel work
            if FinalValues is None:
                # still do axes cosmetics below
                pass
            else:
                # For non-median mode, compute median of stacked curves and plot it
                if not MedianPlot:
                    # Ensure shape (N_ids, N_time_or_phase)
                    try:
                        # Compute median curve robustly
                        y_med = np.array([_nanmedian_safe(FinalValues[:, k]) for k in range(FinalValues.shape[1])])
                    except Exception:
                        y_med = None

                    if y_med is not None:
                        try:
                            cmap_two = cmap([0.1, 0.999999999])
                            cond = (~np.isnan(y_med)) & (~np.isinf(y_med))
                            axs[i][j].plot(
                                xparam_ref[cond],
                                y_med[cond],
                                color=cmap_two[1],
                                ls="solid",
                                lw=1.5 * linewidth,
                            )
                        except Exception:
                            pass

                # MedianPlot mode: use makedataevolution + shading (preserve your logic)
                else:
                    try:
                        if PhasePlot:
                            Y, Yerr, xPhase, xTime = TNG.makedataevolution(
                                [Name], [""], [row],
                                SampleName=SampleName,
                                PhasingPlot=True,
                                dfName=dfName,
                                Name=NameKey,
                                nboots=nboots,
                            )
                            Y = np.array([v for v in Y[0][0][0]])
                            Yerr = np.array([v for v in Yerr[0][0][0]])
                            xparamMedian = np.array([v for v in xPhase[0][0][0]])
                        else:
                            Y, Yerr = TNG.makedataevolution(
                                [Name], [""], [row],
                                SampleName=SampleName,
                                PhasingPlot=False,
                                dfName=dfName,
                                Name=NameKey,
                                nboots=nboots,
                            )
                            Y = np.array([v for v in Y[0][0][0]])
                            Yerr = np.array([v for v in Yerr[0][0][0]])
                            xparamMedian = xparam_ref
                    except Exception:
                        Y = None
                        Yerr = None
                        xparamMedian = None

                    # Overlay gray individual curves with opacity tied to distance from median 
                    if (Y is not None) and (xparamMedian is not None):
                        if PhasePlot:
                            # In your original: you plot each curve flipped for PhasePlot.
                            for arrayValues in FinalValues:
                                try:
                                    cond = (~np.isnan(arrayValues)) & (~np.isinf(arrayValues))
                                    axs[i][j].plot(
                                        xparam_ref[cond],
                                        np.flip(arrayValues[cond]),
                                        color="gray",
                                        ls="solid",
                                        lw=0.25 * linewidth,
                                    )
                                except Exception:
                                    pass
                        else:
                            # deviation-based alpha
                            try:
                                deviation = np.abs(FinalValues - Y)
                                max_dev = np.nanpercentile(deviation, 90)
                                if not np.isfinite(max_dev) or max_dev == 0:
                                    max_dev = np.nanmax(deviation)
                                if not np.isfinite(max_dev) or max_dev == 0:
                                    max_dev = 1.0

                                normalized_dev = np.clip(deviation / max_dev, 0, 1)
                                alpha_values = 1 - normalized_dev
                                alpha_values[np.isnan(alpha_values)] = 0.0

                                base_alpha = 0.05 if ("Normal" in str(Name)) else 0.3

                                for idindex in range(len(IDs)):
                                    try:
                                        values = FinalValues[idindex, :].copy()
                                    except Exception:
                                        continue

                                    # sSFR clipping as in your original
                                    if "sSFR" in str(row):
                                        values[values < -13.5] = np.nan

                                    xt = xparamMedian.copy()
                                    cond = (~np.isnan(values)) & (~np.isinf(values)) & (~np.isnan(xt))
                                    xt = xt[cond]
                                    vv = values[cond]

                                    for t_i in range(len(xt) - 1):
                                        # alpha_values indexing: use original array index
                                        axs[i][j].plot(
                                            xt[t_i : t_i + 2],
                                            vv[t_i : t_i + 2],
                                            color="gray",
                                            alpha=float(alpha_values[idindex, np.where(cond)[0][t_i]]) * base_alpha,
                                            ls="solid",
                                            lw=0.52 * linewidth,
                                        )
                            except Exception:
                                pass

                        # Plot median and fill bands
                        Y_plot = Y.copy()
                        Yerr_plot = Yerr.copy()

                        if "sSFR" in str(row):
                            Yerr_plot[Y_plot < -3.5] = np.nan
                            Y_plot[Y_plot < -13.5] = np.nan

                        axs[i][j].plot(
                            xparamMedian[~np.isnan(Y_plot)],
                            Y_plot[~np.isnan(Y_plot)],
                            color=colors(Name),
                            ls="solid",
                            lw=1.5 * linewidth,
                        )

                        alpha_boost = 1.3 if ("Normal" in str(Name)) else 1.0
                        if "Normal" in str(Name):
                            Yerr_plot = Yerr_plot * 2

                        cond_band = (~np.isnan(Y_plot)) & (~np.isnan(Yerr_plot)) & (~np.isnan(xparamMedian))
                        axs[i][j].fill_between(
                            xparamMedian[cond_band],
                            (Y_plot - Yerr_plot)[cond_band],
                            (Y_plot + Yerr_plot)[cond_band],
                            color=colors(Name),
                            alpha=0.7 * alpha_boost,
                        )
                        axs[i][j].fill_between(
                            xparamMedian[cond_band],
                            (Y_plot - 3 * Yerr_plot)[cond_band],
                            (Y_plot + 3 * Yerr_plot)[cond_band],
                            color=colors(Name),
                            alpha=0.4 * alpha_boost,
                        )
                        

            # -------------------------
            # Panel cosmetics (grid, ticks, scales, labels)
            # -------------------------
            if GridMake:
                axs[i][j].grid(
                    GridMake,
                    color="#9e9e9e",
                    which="major",
                    linewidth=0.6,
                    alpha=0.3,
                    linestyle=":",
                )

            axs[i][j].tick_params(axis="y", labelsize=0.99 * fontlabel)
            axs[i][j].tick_params(axis="x", labelsize=0.99 * fontlabel)

            # y-limits
            if (ylimmin is not None) and (ylimmax is not None):
                try:
                    axs[i][j].set_ylim(ylimmin[i], ylimmax[i])
                except Exception:
                    pass

            # y-scale
            try:
                if scales(row) is not None:
                    axs[i][j].set_yscale(scales(row))
                if scales(row) == "log":
                    axs[i][j].yaxis.set_major_formatter(FuncFormatter(format_func_loglog))
            except Exception:
                pass

            # y-label (left column only)
            if j == 0:
                try:
                    
                    axs[i][j].set_ylabel(labelsequal.get(row, labels.get(row, row)), fontsize=fontlabel)
                except Exception:
                    pass

            # Titles + top z-axis (only first row) when not PhasePlot
            if i == 0:
                if title and (not ColumnPlot):
                    try:
                        # If title is list-like per column, use title[j]
                        if isinstance(title, (list, tuple, np.ndarray)):
                            ttl = title[j]
                        else:
                            ttl = title
                        axs[i][j].set_title(titles.get(ttl, ttl), fontsize=1.0 * fontlabel)
                    except Exception:
                        pass

                if not PhasePlot:
                    try:
                        axs[i][j].tick_params(bottom=True, top=False)
                        lim = axs[i][j].get_xlim()
                        ax2label = axs[i][j].twiny()
                        ax2label.grid(False)
                        ax2label.set_xlim(lim)

                        if (row == "rToRNearYoung") or (savefigname == "Young"):
                            zlabels = np.array(["0", "0.2"])
                            zticks_Age = np.array([13.803, 11.323])
                        else:
                            if SmallerScale:
                                # Preserve your logic: hide last label for j != 0
                                if j == 0:
                                    zlabels = np.array(["0", "0.2", "0.5", "1", "2", "5", "20"])
                                else:
                                    zlabels = np.array(["0", "0.2", "0.5", "1", "2", "5", ""])
                            else:
                                zlabels = np.array(["0", "0.2", "0.5", "1", "2", "5", "20"])
                            zticks_Age = np.array([13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0.0])

                        x_locator = FixedLocator(zticks_Age.tolist())
                        x_formatter = FixedFormatter(zlabels.tolist())
                        ax2label.xaxis.set_major_locator(x_locator)
                        ax2label.xaxis.set_major_formatter(x_formatter)
                        ax2label.set_xlabel(r"$z$", fontsize=fontlabel)
                        ax2label.tick_params(labelsize=0.85 * fontlabel)
                    except Exception:
                        pass

            # Bottom x-axis formatting (last row)
            if i == len(rows) - 1:
                if Type == "Evolution":
                    if (row == "rToRNearYoung") or (savefigname == "Young"):
                        axs[i][j].set_xlabel(r"$t \, \,  [\mathrm{Gyr}]$", fontsize=fontlabel)
                        axs[i][j].set_xticks([10, 12, 14])
                        axs[i][j].set_xticklabels(["10", "12", "14"])
                    else:
                        if LookBackTime and (not PhasePlot):
                            axs[i][j].set_xticks([0.0, 1.97185714, 3.94371429, 5.91557143,
                                                  7.88742857, 9.85928571, 11.83114286, 13.803])
                            if SmallerScale:
                                if j == 1:
                                    axs[i][j].set_xlabel(r"$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$", fontsize=fontlabel)
                                if j == 0:
                                    axs[i][j].set_xticklabels(["14", "12", "10", "8", "6", "4", "2", "0"])
                                else:
                                    axs[i][j].set_xticklabels(["", "12", "10", "8", "6", "4", "2", "0"])
                            else:
                                axs[i][j].set_xlabel(r"$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$", fontsize=fontlabel)
                                if j == 0:
                                    axs[i][j].set_xticklabels(["14", "12", "10", "8", "6", "4", "2", "0"])
                                else:
                                    axs[i][j].set_xticklabels(["", "12", "10", "8", "6", "4", "2", "0"])

                        elif PhasePlot:
                            limXparam = int(xPhaseLim + 1)
                            positive_ticks = np.arange(limXparam)
                            positive_labels = np.array([str(int(t)) for t in positive_ticks])

                            xticks = np.append([-1.0, -0.5], positive_ticks.astype(float))
                            xlabels = np.append(["", "E"], positive_labels)

                            axs[i][j].set_xlabel(r"$\phi_\mathrm{Orbital}$", fontsize=fontlabel)
                            axs[i][j].set_xticks(xticks)
                            axs[i][j].set_xticklabels(xlabels)
                            axs[i][j].set_xlim(-1, xPhaseLim + 0.5)

                        else:
                            axs[i][j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
                            axs[i][j].set_xlabel(r"$t \, \, [\mathrm{Gyr}]$", fontsize=fontlabel)
                            axs[i][j].set_xticklabels(["0", "2", "4", "6", "8", "10", "12", "14"])

    # -----------------------------
    # Save
    # -----------------------------
    savefig(savepath, savefigname, TRANSPARENT)

    return


def PlotProfile(
    # -------------------------
    # Required data inputs
    # -------------------------
    IDs, names, columns, rows, PartTypes,

    # -------------------------
    # What to plot / physical options
    # -------------------------
    ParamX: str = "rad", Condition: str = "All", cumulative: bool = False, norm: bool = False,
    Entry: bool = False, quantile: float = 0.95, rmaxlim: float = 50, nbins: int = 25,
    Nlim: int = 100, nboots: int = 100,

    # -------------------------
    # Labels / titles
    # -------------------------
    title = False,  xlabelintext: bool = False, Supertitle: bool = False,

    # -------------------------
    # Axis limits
    # -------------------------
    ylimmin=None, ylimmax=None,  xlimmin=None, xlimmax=None,

    # -------------------------
    # Plot appearance
    # -------------------------
    fontlabel: float = 24, linewidth: float = 1.2,  framealpha: float = 0.95, GridMake: bool = False,  line: bool = False,

    # -------------------------
    # Legend options
    # -------------------------
    legend: bool = False, LegendNames = None, legpositions=None, loc = "best",
    columnspacing: float = 0.7, handlelength: float = 2.0, handletextpad: float = 0.4, labelspacing: float = 0.3,

    # -------------------------
    # Figure size
    # -------------------------
    lNum: float = 6, cNum: float = 6,

    # -------------------------
    # External data / environment
    # -------------------------
    dfSample = None, dfName: str = "Sample", SampleName: str = "Samples",

    # -------------------------
    # Paths / saving / caching
    # -------------------------
    savepath: str = "fig/PlotProfile", savefigname: str = "fig", PATH: str = os.getenv("HOME", "") + "/TNG_Analyzes/SubhaloHistory",
    SIMTNG: str = "TNG50",  TRANSPARENT: bool = False,

    # -------------------------
    # Reproducibility / extras
    # -------------------------
    seed: int = 16010504, Softening: bool = False,
):
    """
    Plot radial profiles for multiple samples and snapshots.
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    """

    # -----------------------
    # Helpers (local)
    # -----------------------
    def _as_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        return [x]

    def _ensure_axs_2d(axs, nrows: int, ncols: int):
        # matplotlib can return scalar, 1d, or 2d depending on shape
        if not isinstance(axs, (list, np.ndarray)):
            axs = np.array([[axs]])
        axs = np.asarray(axs)
        if axs.ndim == 1:
            # If single row or single column
            if nrows == 1:
                axs = axs.reshape(1, -1)
            else:
                axs = axs.reshape(-1, 1)
        return axs

    def _safe_nanmedian(arr: np.ndarray) -> float:
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.nan
        return np.nanmedian(arr)

    def _filter_valid_xy(rad, y, yerr=None):
        rad = np.asarray(rad, dtype=float)
        y = np.asarray(y, dtype=float)
        if yerr is not None:
            yerr = np.asarray(yerr, dtype=float)

        m = np.isfinite(rad) & np.isfinite(y)
        rad = rad[m]
        y = y[m]
        if yerr is not None:
            yerr = yerr[m]
        return (rad, y, yerr) if yerr is not None else (rad, y)

    def _profile_cache_path(
        base: str,
        sim: str,
        condition: str,
        rowname: str,
        ptype: str,
        snap: int,
        sample_name: str,
    ):
        return (
            base + '/' + sim + '/Profiles/' +  condition + '/' + rowname + '/' + ptype + '/' + str(snap) + '/' + f"{sample_name}{condition}.csv"
        )

    def _read_cached_profile(path):
        
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path)
            rad = df["Rads"].values
            ymed = df["ymedians"].values
            yerr = df["yerrs"].values
            return rad, ymed, yerr
        except Exception:
            return None

    def _write_cached_profile(path, rad, ymed, yerr) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame({"Rads": rad, "ymedians": ymed, "yerrs": yerr})
        df.to_csv(path, index=False)

    def _compute_rmin_rmax(ptype: str, rads_linear: np.ndarray):
        med = _safe_nanmedian(rads_linear)
        if not np.isfinite(med) or med <= 0:
            # fallback, avoids zeros breaking geomspace
            return 0.1, min(rmaxlim, 10.0)

        if ptype == "PartType4":
            rmin = med / 5.0
            rmax = med * 150.0
        elif ptype in ("PartType0", "gas", "PartType1", "DM"):
            rmin = med / 300.0
            rmax = med * 7.0
        else:
            # default behavior if a new particle type appears
            rmin = med / 100.0
            rmax = med * 10.0

        # caps and floors (preserve your thresholds)
        if (not np.isfinite(rmax)) or (rmax == 0.0) or (rmax > rmaxlim):
            rmax = rmaxlim

        if ptype == "PartType0" and rmin < 0.07:
            rmin = 0.07
        if ptype == "PartType1" and rmin < 0.3:
            rmin = 0.3
        if ptype == "PartType4" and rmin < 0.1:
            rmin = 0.1

        # ensure valid
        if not np.isfinite(rmin) or rmin <= 0:
            rmin = 0.1
        if not np.isfinite(rmax) or rmax <= rmin:
            rmax = min(rmaxlim, rmin * 10.0)

        return rmin, rmax

    def _safe_interp(rad, y, npts: int = 25, kind_primary="cubic", kind_fallback="linear"):
        rad = np.asarray(rad, dtype=float)
        y = np.asarray(y, dtype=float)
        if rad.size < 3:
            return rad, y
        rmin = np.nanmin(rad)
        rmax = np.nanmax(rad)
        if not np.isfinite(rmin) or not np.isfinite(rmax) or rmin <= 0 or rmax <= rmin:
            return rad, y

        x = np.geomspace(rmin, rmax, npts)
        # interp can fail if rad is not strictly increasing
        order = np.argsort(rad)
        rad_sorted = rad[order]
        y_sorted = y[order]

        # remove duplicates in rad (interp1d requires strictly increasing for some modes)
        uniq, idx = np.unique(rad_sorted, return_index=True)
        rad_sorted = uniq
        y_sorted = y_sorted[idx]
        if rad_sorted.size < 3:
            return rad_sorted, y_sorted

        try:
            f = interp1d(rad_sorted, y_sorted, kind=kind_primary, fill_value="extrapolate")
            return x, f(x)
        except Exception:
            f = interp1d(rad_sorted, y_sorted, kind=kind_fallback, fill_value="extrapolate")
            return x, f(x)

    # -----------------------
    # Start
    # -----------------------
    np.random.seed(seed)

    columns = _as_list(columns)
    rows = _as_list(rows)
    names = _as_list(names)

    if len(rows) != len(PartTypes):
        raise ValueError(f"PartTypes must have same length as rows. Got {len(PartTypes)} vs {len(rows)}.")

    if len(names) != len(IDs):
        raise ValueError(f"names and IDs must have same length. Got {len(names)} vs {len(IDs)}.")

    # If any sample label contains 'Entry' we need dfSample (to get Snap_At_FirstEntry)
    needs_dfSample = any(("Entry" in str(nm)) for nm in names)
    if (Entry or needs_dfSample) and dfSample is None:
        raise ValueError("dfSample is required when Entry=True or when any sample name contains 'Entry'.")

    base_path = os.getenv("HOME")

    # Time / redshift table
    try:
        dfTime = TNG.extractDF("SNAPS_TIME")
    except Exception as e:
        raise RuntimeError("Could not load SNAPS_TIME via TNG.extractDF('SNAPS_TIME').") from e

    # Prepare axes
    plt.rcParams.update({"figure.figsize": (cNum * len(columns), lNum * len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(columns), hspace=0, wspace=0)
    axs = gs.subplots(sharex="col", sharey="row")
    axs = _ensure_axs_2d(axs, len(rows), len(columns))

    # Preload half-mass radii DF for stars/gas and gas mass
    dFHalfStar = TNG.extractDF("SubhaloHalfmassRadType4", PATH=PATH)
    dFHalfGasRad = TNG.extractDF("SubhaloHalfmassRadType0", PATH=PATH)
    dfGasMass = TNG.extractDF("SubhaloMassType0", PATH=PATH)

    # For Galactic condition use gas half-mass radius always
    dFHalfRadGas = None
    if "Galactic" in str(Condition):
        dFHalfRadGas = TNG.extractDF("SubhaloHalfmassRadType0", PATH=PATH)

    # -----------------------
    # Main loops
    # -----------------------
    for i, row in enumerate(rows):
        ptype = PartTypes[i]

        # choose half-radius DF for this row's particle type (preserve your logic)
        if ptype == "PartType4":
            dFHalfRad = TNG.extractDF("SubhaloHalfmassRadType4", PATH=PATH)
        elif ptype in ("PartType0", "gas"):
            dFHalfRad = TNG.extractDF("SubhaloHalfmassRadType0", PATH=PATH)
        elif ptype in ("PartType1", "DM"):
            dFHalfRad = TNG.extractDF("SubhaloHalfmassRadType1", PATH=PATH)
        else:
            # fallback: try to interpret 'PartTypeX' pattern, else default stars
            dFHalfRad = TNG.extractDF("SubhaloHalfmassRadType4", PATH=PATH)

        for l, ID_group in enumerate(IDs):
            sample_name = str(names[l])

            # Validate ID_group iterable
            try:
                _ = iter(ID_group)
            except TypeError as e:
                raise TypeError(
                    f"Each element of IDs must be an iterable of IDs. Problem at IDs[{l}] for name='{sample_name}'."
                ) from e

            for j, snap_in in enumerate(columns):
                # resolve snap value (int)
                snap = int(snap_in)

                # Titles
                if "Entry" in sample_name:
                    if j == 0:
                        axs[i][j].set_title(r"$z_\mathrm{entry}$", fontsize=1.1 * fontlabel)
                    elif j == 1:
                        snap = 99
                        zval = dfTime.z.loc[dfTime.Snap == snap].values
                        zlab = zval[0] if len(zval) else np.nan
                        axs[i][j].set_title(rf"$z = {zlab:.1f}$", fontsize=1.1 * fontlabel)
                else:
                    if i == 0:
                        zval = dfTime.z.loc[dfTime.Snap == snap].values
                        zlab = zval[0] if len(zval) else np.nan
                        axs[i][j].set_title(rf"$z = {zlab:.1f}$", fontsize=1.1 * fontlabel)

                # Collect half-mass radii (for optional vertical line)
                RadStars = np.array([], dtype=float)
                RadGas = np.array([], dtype=float)
                GasMass = np.array([], dtype=float)

                # Build Rads array used to set rmin/rmax (linear radii)
                Rads = np.array([], dtype=float)

                # Determine per-subhalo snap when Entry in name and first column
                for idValue in ID_group:
                    idValue_int = int(idValue)

                    snap_use = snap
                    if "Entry" in sample_name and j == 0:
                        # Snap at first entry
                        try:
                            snap_use = int(
                                dfSample.Snap_At_FirstEntry.loc[dfSample.SubfindID_99 == idValue_int].values[0]
                            )
                        except Exception:
                            snap_use = snap  # fallback

                    # Half radii / gas mass (for line/diagnostics)
                    try:
                        HalfRad_star = dFHalfStar[str(idValue_int)].loc[dFHalfStar.Snap == snap_use].values[0]
                    except Exception:
                        HalfRad_star = np.nan
                    try:
                        HalfRad_gas = dFHalfGasRad[str(idValue_int)].loc[dFHalfGasRad.Snap == snap_use].values[0]
                    except Exception:
                        HalfRad_gas = np.nan
                    try:
                        GasMassType = dfGasMass[str(idValue_int)].loc[dfGasMass.Snap == snap_use].values[0]
                    except Exception:
                        GasMassType = np.nan

                    RadStars = np.append(RadStars, HalfRad_star)
                    RadGas = np.append(RadGas, HalfRad_gas)
                    GasMass = np.append(GasMass, GasMassType)

                    # Rads used for rmin/rmax
                    if "Galactic" in str(Condition) and dFHalfRadGas is not None:
                        try:
                            HalfRadGas = dFHalfRadGas[str(idValue_int)].loc[dFHalfRadGas.Snap == snap_use].values[0]
                            Rads = np.append(Rads, 10 ** float(HalfRadGas))
                        except Exception:
                            Rads = np.append(Rads, np.nan)
                    else:
                        try:
                            HalfRad = dFHalfRad[str(idValue_int)].loc[dFHalfRad.Snap == snap_use].values[0]
                            Rads = np.append(Rads, 10 ** float(HalfRad))
                        except Exception:
                            # preserve your fallback value but in linear space
                            Rads = np.append(Rads, 10 ** 1.2)

                # If median radius is 0 or invalid, just add a dummy artist for legend consistency
                medR = _safe_nanmedian(Rads)
                if (not np.isfinite(medR)) or (medR <= 0):
                    axs[i][j].plot(
                        [np.nan],
                        [np.nan],
                        color=colors(sample_name),
                        ls=lines(sample_name),
                        lw=2.5 * linesthicker(sample_name),
                        dash_capstyle=capstyles(sample_name),
                    )
                    continue

                rmin, rmax = _compute_rmin_rmax(ptype, Rads)

                # -------------
                # Load profile
                # -------------
                rad = ymedian = yerr = None

                # Map special row names to cached directories
                special_map = {
                    "sSFR": None,
                    "GFM_Metallicity_Zodot": ("GFM_Metallicity", True),
                    "GradsSFR": None,
                    "joverR": None,
                    "DensityGasOverR2": None,
                    "DensityStarOverR2": None,
                }

                def _load_or_make(base_rowname: str):
                    cache_path = _profile_cache_path(
                        base=base_path,
                        sim=SIMTNG,
                        condition=Condition,
                        rowname=base_rowname,
                        ptype=ptype,
                        snap=snap,
                        sample_name=sample_name,
                    )
                    cached = _read_cached_profile(cache_path)
                    if cached is not None:
                        return cached

                    # compute on the fly
                    rad_local, ymed_local, yerr_local = TNG.make_profile(
                        ID_group,
                        snap,
                        base_rowname,
                        ptype,
                        rmin=rmin,
                        rmax=rmax,
                        nbins=nbins,
                        nboot=nboots,
                        Condition=Condition,
                        dfSample=dfSample,
                        Entry=Entry,
                    )
                    if isinstance(rad_local, float):
                        return None

                    _write_cached_profile(cache_path, rad_local, ymed_local, yerr_local)
                    return rad_local, ymed_local, yerr_local

                # Standard profiles
                if row not in special_map:
                    out = _load_or_make(row)
                    if out is None:
                        continue
                    rad, ymedian, yerr = out

                # sSFR = SFR / Mstellar
                elif row == "sSFR":
                    out_sfr = _load_or_make("SFR")
                    if out_sfr is None:
                        continue
                    radSFR, ySFR, eSFR = out_sfr

                    cache_ms = _profile_cache_path(base_path, SIMTNG, Condition, "Mstellar", "PartType4", snap, sample_name)
                    out_ms_cached = _read_cached_profile(cache_ms)
                    if out_ms_cached is None:
                        # Try compute it directly with PartType4
                        radM, yM, eM = TNG.make_profile(
                            ID_group, snap, "Mstellar", "PartType4",
                            rmin=rmin, rmax=rmax, nbins=nbins, nboot=nboots,
                            Condition=Condition, dfSample=dfSample, Entry=Entry
                        )
                        if isinstance(radM, float):
                            continue
                        _write_cached_profile(cache_ms, radM, yM, eM)
                        radMstellar, yMstellar, _ = radM, yM, eM
                    else:
                        radMstellar, yMstellar, _ = out_ms_cached

                    new_y = interp1d(radMstellar, yMstellar, kind="linear", fill_value="extrapolate")(radSFR)
                    rad = radSFR
                    ymedian = ySFR / new_y
                    yerr = eSFR / new_y

                # Metallicity in Z/Zsun
                elif row == "GFM_Metallicity_Zodot":
                    out = _load_or_make("GFM_Metallicity")
                    if out is None:
                        continue
                    rad, ymedian, yerr = out
                    ymedian = ymedian / 0.0127
                    yerr = yerr / 0.0127

                # Gradient of sSFR
                elif row == "GradsSFR":
                    out_sfr = _load_or_make("SFR")
                    if out_sfr is None:
                        continue
                    radSFR, ySFR, eSFR = out_sfr

                    cache_ms = _profile_cache_path(base_path, SIMTNG, Condition, "Mstellar", "PartType4", snap, sample_name)
                    out_ms_cached = _read_cached_profile(cache_ms)
                    if out_ms_cached is None:
                        radM, yM, eM = TNG.make_profile(
                            ID_group, snap, "Mstellar", "PartType4",
                            rmin=rmin, rmax=rmax, nbins=nbins, nboot=nboots,
                            Condition=Condition, dfSample=dfSample, Entry=Entry
                        )
                        if isinstance(radM, float):
                            continue
                        _write_cached_profile(cache_ms, radM, yM, eM)
                        radMstellar, yMstellar = radM, yM
                    else:
                        radMstellar, yMstellar, _ = out_ms_cached

                    new_y = interp1d(radMstellar, yMstellar, kind="linear", fill_value="extrapolate")(radSFR)
                    rad = radSFR
                    ymedian = (ySFR / new_y)
                    yerr = (eSFR / new_y)
                    # gradient
                    try:
                        ymedian = np.gradient(ymedian, rad)
                    except Exception:
                        pass

                # joverR = j / r
                elif row == "joverR":
                    out = _load_or_make("j")
                    if out is None:
                        continue
                    rad, ymedian, yerr = out
                    ymedian = ymedian / rad
                    yerr = yerr / rad

                # DensityGasOverR2 = DensityGas * r^2
                elif row == "DensityGasOverR2":
                    out = _load_or_make("DensityGas")
                    if out is None:
                        continue
                    rad, ymedian, yerr = out
                    ymedian = ymedian * rad**2
                    yerr = yerr * rad**2

                # DensityStarOverR2 = DensityStar * r^2
                elif row == "DensityStarOverR2":
                    out = _load_or_make("DensityStar")
                    if out is None:
                        continue
                    rad, ymedian, yerr = out
                    ymedian = ymedian * rad**2
                    yerr = yerr * rad**2

                # Filter invalid values
                rad, ymedian, yerr = _filter_valid_xy(rad, ymedian, yerr)

                # Cumulative and normalization
                if cumulative and row in ["Mstellar", "Mgas"]:
                    ymedian = np.cumsum(ymedian)
                    if ymedian.size == 0:
                        ymedian = np.full_like(rad, np.nan, dtype=float)
                        yerr = np.full_like(rad, np.nan, dtype=float)
                    else:
                        if norm and np.nanmax(ymedian) > 0:
                            yerr = yerr / np.nanmax(ymedian)
                            ymedian = ymedian / np.nanmax(ymedian)

                # Gas existence check
                if ptype == "PartType0":
                    try:
                        dfS = TNG.extractDF(dfName)
                        SnapCheck = dfS.loc[dfS.SubfindID_99.isin(ID_group), "SnapLostGas"].values
                        SnapCheck = SnapCheck.astype(float)
                        SnapCheck[SnapCheck < 0] = 99
                        if not (len(SnapCheck[SnapCheck >= snap]) > int(len(SnapCheck) / 2)):
                            continue
                    except Exception:
                        continue

                # Plot line: if too few points, plot directly; else interpolate for smooth curve
                if rad.size <= 2:
                    axs[i][j].plot(
                        rad,
                        ymedian,
                        color=colors(sample_name),
                        ls=lines(sample_name),
                        lw=3.5 * linesthicker(sample_name),
                        dash_capstyle=capstyles(sample_name),
                    )
                else:
                    x_s, y_s = _safe_interp(rad, ymedian, npts=25, kind_primary="cubic", kind_fallback="linear")
                    if "RadVelocity" not in row:
                        mpos = np.isfinite(y_s) & (y_s > 0)
                        axs[i][j].plot(
                            x_s[mpos],
                            y_s[mpos],
                            color=colors(sample_name),
                            ls=lines(sample_name),
                            lw=3.5 * linesthicker(sample_name),
                            dash_capstyle=capstyles(sample_name),
                        )
                    else:
                        axs[i][j].plot(
                            x_s,
                            y_s,
                            color=colors(sample_name),
                            ls=lines(sample_name),
                            lw=3.5 * linesthicker(sample_name),
                            dash_capstyle=capstyles(sample_name),
                        )

                    # vertical line at stellar half-mass radius median (preserve original)
                    if line and (i == len(rows) - 1):
                        med_star = _safe_nanmedian(RadStars)
                        if np.isfinite(med_star):
                            axs[i][j].axvline(
                                10 ** float(med_star),
                                ls="--",
                                color=colors(sample_name),
                                lw=1.1,
                            )

                # Softening shading (preserve)
                if Softening and ("DensityStar" in row):
                    try:
                        rSoftening = ETNG.Softening()
                        axs[i][j].axvspan(0, rSoftening[snap], facecolor="tab:red", alpha=0.1)
                    except Exception:
                        pass

                # -----------------
                # Panel formatting
                # -----------------
                if GridMake:
                    axs[i][j].grid(
                        GridMake,
                        color="#9e9e9e",
                        which="major",
                        linewidth=0.6,
                        alpha=0.3,
                        linestyle=":",
                    )

                if (ylimmin is not None) and (ylimmax is not None):
                    axs[i][j].set_ylim(ylimmin[i], ylimmax[i])

                if (xlimmin is not None) and (xlimmax is not None):
                    if hasattr(xlimmin, "__len__") and len(xlimmin) > 1:
                        axs[i][j].set_xlim(xlimmin[j], xlimmax[j])
                    else:
                        axs[i][j].set_xlim(xlimmin[0], xlimmax[0])

                # Legend block (preserve)
                if legend and (LegendNames is not None) and (legpositions is not None):
                    for legpos, LegendName in enumerate(LegendNames):
                        if j == legpositions[legpos][0] and i == legpositions[legpos][1]:
                            custom_lines, label, ncol, mult = Legend(LegendName, mult=5)
                            axs[i][j].legend(
                                custom_lines,
                                label,
                                ncol=ncol,
                                loc=loc[legpos] if isinstance(loc, (list, tuple, np.ndarray)) else loc,
                                fontsize=0.88 * fontlabel,
                                framealpha=framealpha,
                                columnspacing=columnspacing,
                                handlelength=handlelength,
                                handletextpad=handletextpad,
                                labelspacing=labelspacing,
                            )

                # Y axis labeling & scaling on first column
                if j == 0:
                    if cumulative and row in ["Mstellar", "Mgas"]:
                        if norm:
                            axs[i][j].set_yscale(scales(row + "Norm"))
                            if scales(row + "Norm") == "log":
                                axs[i][j].yaxis.set_major_formatter(FuncFormatter(format_func_loglog))
                            axs[i][j].set_ylabel(labels.get(row + "Norm"), fontsize=fontlabel)
                        else:
                            axs[i][j].set_yscale(scales(row + "Cum"))
                            if scales(row + "Cum") == "log":
                                axs[i][j].yaxis.set_major_formatter(FuncFormatter(format_func_loglog))
                            axs[i][j].set_ylabel(labels.get(row + "Cum"), fontsize=fontlabel)
                    else:
                        axs[i][j].set_yscale(scales(row))
                        if scales(row) == "log":
                            axs[i][j].yaxis.set_major_formatter(FuncFormatter(format_func_loglog))
                        if row in ["j", "RadVelocity", "joverR"]:
                            axs[i][j].set_ylabel(labels.get(row + ptype), fontsize=fontlabel)
                        else:
                            axs[i][j].set_ylabel(labels.get(row), fontsize=fontlabel)

                # In-panel text at last column
                if (j == len(columns) - 1) and xlabelintext:
                    Afont = {"color": "black", "size": fontlabel}
                    anchored_text = AnchoredText(texts.get(row), loc="upper right", prop=Afont)
                    axs[i][j].add_artist(anchored_text)

                # X scale & label on bottom row
                if i == (len(rows) - 1):
                    axs[i][j].set_xscale(scales(ParamX))
                    if scales(ParamX) == "log":
                        axs[i][j].xaxis.set_major_formatter(FuncFormatter(format_func_loglog))
                    axs[i][j].set_xlabel(labels.get(ParamX), fontsize=fontlabel)

                axs[i][j].tick_params(labelsize=0.99 * fontlabel)

    if Supertitle:
        plt.suptitle("Satellites", fontsize=1.3 * fontlabel, y=1.1)

    savefig(savepath, savefigname, TRANSPARENT)
    return



def PlotIDsColumns(IDs, rows, dataMarker=None, dataLine=None, SatelliteTime = False, 
                   PhasingPlot = False, ShowPop = False, ShowPopName = 'Normal', SnapTransition = False, 
                   SnapTransitionName = '',
                   title=False, xlabelintext=False, lineparams=False,  QuantileError=True, 
           alphaShade=0.3,  linewidth=0.5, fontlabel=24, nboots=100,  ColumnPlot=False, limaxis=False, 
           columnspacing = 0.5, handlelength = 2, handletextpad = 0.4, labelspacing = 0.3, LookBackTime = False, Pericenter = False, postext = ['best'],
           ylimmax = None, ylimmin = None, GridMake = False, CompareToNormal = False,
           lNum = 6, cNum = 6, InfallTime = False, NoGas = False, SmallerScale = False,
           Type='Evolution', Xparam='Time', savepath='fig/PlotIDColumns', savefigname='fig', dfName='Sample', SampleName='Samples', legend=False, LegendNames='None',  loc='best',
           bins=10, seed=16010504, TRANSPARENT = False, Softening = False, MaxSizeType = False):
    
    
    '''
    Plot teh evolution for random sample
    Parameters
    ----------
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    IDs: IDs for selected subhalos. 
    The rest is the same as the previous functions
    Returns
    -------
    Requested Evolution or Co-Evolution plot
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    np.random.seed(seed)

    dfTime = pd.read_csv(os.getenv("HOME")+"/TNG_Analyzes/SubhaloHistory/SNAPS_TIME.csv")
    Sample = TNG.extractPopulation(dfName, dfName = dfName)

    snapsTime = np.array([88, 81, 64, 51, 37, 24])
    # Verify NameParameters
    if type(IDs) is not list and type(IDs) is not np.ndarray:
        IDs = [IDs]

    if type(rows) is not list and type(rows) is not np.ndarray:
        rows = [rows]

    # Define axes(cNum*len(columns), lNum*len(rows))})
    plt.rcParams.update({'figure.figsize': (lNum*len(IDs), cNum*len(rows))})
    fig = plt.figure()
    gs = fig.add_gridspec(len(rows), len(IDs), hspace=0, wspace=0)
    axs = gs.subplots(sharex='col', sharey='row')
    
    if Pericenter:
        r_over_R_Crit200 = TNG.extractDF('r_over_R_Crit200')

   
    # Verify axs shape
    if type(axs) is not list and type(axs) is not np.ndarray:
        axs = [axs]
    if type(axs[0]) is not np.ndarray:
        axs = np.array([axs])
        if len(IDs) == 1:
            axs = axs.T

    time = dfTime.Age.values

    for i, row in enumerate(rows):
        if type(row) is not list and type(row) is not np.ndarray:
            row = [row]
        
        dfs = []
        Ys = []
        Yerrs = []
        for param in row:
            dfs.append(TNG.extractDF(param))
            if CompareToNormal:
                Y, Yerr = TNG.makedataevolution(['Normal'], [''], [param], SampleName=SampleName, dfName = dfName, nboots=nboots)
                Yerr = np.array([value for value in Yerr[0][0][0]])
                Y = np.array([value for value in Y[0][0][0]])
                Ys.append(Y)
                Yerrs.append(Yerr)
                
       
            
        if Type == 'CoEvolution':
            dfX = TNG.extractDF(Xparam[i]) 
        
        if dataLine is not None:
            datalinevalues = TNG.extractDF(dataLine) 

        if dataMarker is not None:
            if 'Merger' in dataMarker:
                datamarkerTotvalues = TNG.extractDF('NumMergersTotal') 
                dataMarkervalues =TNG.extractDF('NumMajorMergersTotal') 
                datamarkervalues = TNG.extractDF('NumMinorMergersTotal')               
            else:
                datamarkervalues = TNG.extractDF(dataMarker) 

        
        for j, ID in enumerate(IDs):
            
            if j == 0:
               
                if i > 0 and ('SubhaloHalfmassRadType0' in rows[i - 1][0] or  'StarMass_In_Rhpkpc' in rows[i - 1][0] ) and 'Mgas_Norm_Max' in row[0]:
                    None
                elif 'StarMass_In_Rhpkpc' in rows[i - 1][0] :
                    None
                
                elif legend and LegendNames !='None':
                        if len(LegendNames) <= i  :
                            None
                        else:
                            custom_lines, label, ncol, mult = Legend(LegendNames[i])
    
                            axs[i][j].legend(
                                   custom_lines, label, ncol=ncol, loc=loc, fontsize=0.88*fontlabel, 
                                  columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)
                            loc = 'best'
                    
                else:
                    
                    loc = 'best'   
                    if row == ['SubhaloStellarMass_in_Rhpkpc', 'SubhaloStellarMass_Above_Rhpkpc', 'SubhaloGasMass_in_Rhpkpc', 'SubhaloGasMass_Above_Rhpkpc']:
                        custom_lines, label, ncol, mult = Legend(['in_Rhpkpc', 'Above_'])
                    
                    elif row == ['SubhalosSFRInHalfRad', 'SubhalosSFRwithinHalfandRad']:

                        custom_lines, label, ncol, mult = Legend(['SubhalosSFRInHalfRad', 'SubhalosSFRwithinHalfandRad'])
                        loc = 'best'
                    elif len(row) > 1:
                        namesrow = [namerow for namerow in row]
                        for index, namerow in enumerate(namesrow):
                            namesrow[index] = namerow+'IDsColumn'
                        custom_lines, label, ncol, mult = Legend(namesrow)
                    
                    
                    if legend and not (row == ['r_over_R_Crit200_WithoutCorrection', 'r_over_R_Crit200'] or row == ['sSFR_In_TrueRhpkpc', 'sSFR_Above_TrueRhpkpc'] or row == ['SFR_In_Rhpkpc', 'SFR_Above_Rhpkpc'] or row == ['logStar_GFM_Metallicity_In_Rhpkpc', 'logStar_GFM_Metallicity_Above_Rhpkpc'] or row == ['sSFR_In_Rhpkpc', 'sSFR_Above_Rhpkpc'] ) and len(row) > 1: # or row == ['sSFR_In_Rhpkpc', 'sSFR_Above_Rhpkpc']
                        if row ==  ['Star_GFM_Metallicity_In_Rhpkpc', 'Star_GFM_Metallicity_Above_Rhpkpc']:
                            
                            None
                        else:
                            axs[i][j].legend(
                                   custom_lines, label, ncol=ncol, loc=loc, fontsize=0.88*fontlabel, 
                                  columnspacing = columnspacing, handlelength = handlelength, handletextpad = handletextpad, labelspacing = labelspacing)
                            loc = 'best'

            if Softening and 'SubhaloHalfmassRadType4' in row:
                rSoftening = ETNG.Softening()
                rSoftening = np.flip(rSoftening)
                axs[i][j].plot(time[(~np.isinf(rSoftening))], np.log10(rSoftening[(~np.isinf(rSoftening))]), 
                               color='black', ls='solid', lw=2*linewidth)
            
            for l, df in enumerate(dfs):
                #Y = Ys[l]
                #Yerr = Yerrs[l]
                values = np.array([value for value in df[str(ID)].values])
                if Type == 'Evolution':
                    if row[l] == 'r_over_R_Crit200_FirstGroup':
                        values[values == 0] = np.nan
                        arg = np.argwhere(np.isnan(values)).T[0]
                        values[arg[0]:] = np.nan
                    
                    if 'Type4' in row[l] or 'star' in row[l] and not 'HalfRad' in row[l]:
                        color = 'blue'
                        ls = 'solid'
                    elif 'Type0' in row[l] or ('gas' in row[l] and (not '_in_' in row[l] and not '_Above_' in row[l]) ):
                        color = 'green'
                        ls = 'solid'
                    elif 'Type1' in row[l] or 'DM' in row[l]:
                        color = 'purple'
                        ls = 'solid'
                    elif 'SubhalosSFRInHalfRad' in row[l]:
                        color = 'darkblue'
                        ls = 'solid'
                    elif 'SubhalosSFRwithinHalfandRad' in row[l]:
                        color = 'darkred'
                        ls = (0, (10, 8))
                    elif ('r_over_R_Crit200_FirstGroup' in row[l] ) or ('Group_M_Crit200' in row[l]):
                        color = 'red'
                        ls = 'dashed'
                    elif 'r_over_R_Crit200' in row[l]:
                        color = 'darkorange'
                        ls = 'solid'
                       
                    elif ('in_Rhpkpc' in row[l] or 'In_TrueRhpkpc' in row[l] or   'In_Rhpkpc' in row[l]) and not ('Inflow'  in row[l] or 'Outflow' in row[l] or 'Rhpkpc_entry'  in row[l]):
                        color = 'darkblue'
                        ls = 'solid'
                    elif ('Above_Rhpkpc' in row[l] or 'Above_TrueRhpkpc' in row[l]) and not ('Inflow'  in row[l] or 'Outflow' in row[l] ):
                        color = 'tab:blue'
                        ls =  (0, (10, 6))

                    else:
                        color = colors.get(row[l], 'black')
                        ls = lines.get(row[l], 'solid')
        
                    if CompareToNormal:
                        values[~np.isnan(values)] = (values[~np.isnan(values)] - Y[~np.isnan(values)]) / Yerr[~np.isnan(values)]
        
                    if PhasingPlot :
                        xparam = np.arange(-1, 9)
                        xparam = np.append(xparam, xparam+0.5)
                        xparam = np.append(xparam, np.linspace(-1, 9, 1000))
                        xparam = np.unique(xparam)
                        values = np.flip(values)
                        dfPopulation = TNG.extractPopulation(dfName, dfName = dfName)
                        
                        phases = TNG.PhasingData(ID, dfPopulation)
                        
                        if type(phases) != np.ndarray:
                            continue
                        phases = phases[(~np.isnan(values)) & (~np.isinf(values))]
                        values = values[(~np.isnan(values)) & (~np.isinf(values))]
                        if len(values) == 0:
                            continue
                        X_Y_Spline = interp1d(phases, values,kind="linear",fill_value="extrapolate")
                        values = X_Y_Spline(xparam)
                        if phases.max() < 8:
                            values[xparam > phases.max()] = np.nan
                        else:
                            values[xparam > 4] = np.nan

                    else:
                        xparam = time

                    if ShowPop:
                        Y, Yerr = TNG.makedataevolution([ShowPopName], [''], [row[l]], SampleName=SampleName, dfName = dfName, nboots=nboots)
                        Yerr = np.array([value for value in Yerr[0][0][0]])
                        Y = np.array([value for value in Y[0][0][0]])
                        if ('Gas' in row[l]  or 'Type0' in row[l]):
                            Y[:int(99 - 83)] = np.nan
                            Yerr[:int(99 - 83)] = np.nan
                            print(Y, Yerr)
                        #if ('Gas' in row[l] or 'SFR' in row[l] or 'Type0' in row[l]):
                            
                        #    dfPop = TNG.extractPopulation(ShowPopName, dfName = dfName)
                        #    if ~np.isnan(np.nanmedian(dfPop.SnapLostGas)) and np.nanmedian(dfPop.SnapLostGas) > 0:
                        #        Y[xparam > dfTime.Age.loc[dfTime.Snap == int(np.nanmedian(dfPop.SnapLostGas))].values[0]] = np.nan
                     
                        
                        axs[i][j].plot(xparam[~np.isnan(Y)], Y[~np.isnan(
                            Y)], color=colors.get(ShowPopName, 'black'), ls=ls, 
                            lw=1.*linesthicker.get(ShowPopName, linewidth), dash_capstyle = capstyles.get(ShowPopName, 'projecting'))
            

                        axs[i][j].fill_between(
                            xparam[~np.isnan(Y)], Y[~np.isnan(Y)] - Yerr[~np.isnan(Y)], 
                            Y[~np.isnan(Y)] + Yerr[~np.isnan(Y)], color=colors.get(ShowPopName+'Error', 'black'), ls=ls, alpha=alphaShade)
                         
                    axs[i][j].plot(xparam[~np.isnan(values)], values[~np.isnan(values)], color=color,  ls=ls, lw=linewidth)

                    if Pericenter :#and not row == 'r_over_R_Crit200':
                        snapFirstPeri = Sample['SnapFirstPeri'].loc[Sample.SubfindID == ID].values[0]
                        SnapSecondPeri = Sample['SnapSecondPeri'].loc[Sample.SubfindID == ID].values[0]
                        SnapThirdPeri = Sample['SnapThirdPeri'].loc[Sample.SubfindID == ID].values[0]
                        SnapFirstApo = Sample['SnapFirstApo'].loc[Sample.SubfindID == ID].values[0]
                        SnapSecondApo = Sample['SnapSecondApo'].loc[Sample.SubfindID == ID].values[0]
                        
                        if ~np.isnan(SnapThirdPeri):
                            Peris = np.array([99-int(snapFirstPeri), 99-int(SnapSecondPeri), 99-int(SnapThirdPeri)])
                        elif ~np.isnan(SnapSecondPeri):
                            Peris = np.array([99-int(snapFirstPeri), 99-int(SnapSecondPeri)])
                        elif ~np.isnan(snapFirstPeri):
                            Peris = np.array([99-int(snapFirstPeri)])
                            
                        if ~np.isnan(snapFirstPeri):
                            axs[i][j].scatter(time[Peris], values[Peris],color='red', marker = 'x', s = 30, edgecolor = 'black' )
                        
                        if ~np.isnan(SnapSecondApo):
                            Apos = np.array([99-int(SnapFirstApo), 99-int(SnapSecondApo)])
                        elif ~np.isnan(SnapFirstApo):
                            Apos = np.array([99-int(SnapFirstApo)])
                            
                        if ~np.isnan(SnapFirstApo):
                            axs[i][j].scatter(xparam[Apos], values[Apos],color='black', marker = 'x', s = 30, edgecolor = 'black' )

                    if InfallTime:
                        
                        infallsnap = Sample.loc[Sample.SubfindID_99 == ID, 'Snap_At_FirstEntry'].values[0]
                        infallsnap = float(infallsnap)
                        if ~np.isnan(infallsnap) and infallsnap > 0:
                            infallsnap = int(99-infallsnap)
                            axs[i][j].axvline(xparam[infallsnap], color='black', ls = (0, (10, 8)))
                            
                    if SnapTransition:
                        
                        infallsnap = Sample.loc[Sample.SubfindID_99 == ID, SnapTransitionName].values[0]
                        if ~np.isnan(infallsnap) and infallsnap > 0:
                            infallsnap = int(99-infallsnap)
                            axs[i][j].axvline(xparam[infallsnap], color='red', ls = (0, (10, 8)))

                    if SatelliteTime and 'Group_M_Crit200' in param:
                        
                        infallsnap = Sample.loc[Sample.SubfindID_99 == ID, 'SnapBecomeSatellite'].values[0]
                        if ~np.isnan(infallsnap) and infallsnap > 0:
                            axs[i][j].scatter(xparam[int(99-infallsnap)], values[int(99-infallsnap)], marker = '*', s = 220, color = 'red')

                    if NoGas:
                       infallsnap =  Sample.loc[Sample.SubfindID_99 == ID, 'SnapLostGas'].values[0]
                        
                       if  ~np.isnan(infallsnap) and infallsnap > 0:
                            infallsnap = int(99-infallsnap)
                            axs[i][j].axvspan(xparam[infallsnap], time[0], color='pink', alpha=0.5, lw=0)
                        
                    if MaxSizeType :
                         MaxSize = Sample['MaxSizeType4'].loc[Sample.SubfindID == ID].values[0]
                         axs[i][j].axhline(MaxSize)
                         
                    if dataLine is not None:
                        linevalues = np.array(
                            [value for value in datalinevalues[str(ID)].values])
                        if len(linevalues.shape) > 1:
                            linevalues = linevalues.T[0]
                            linevalues = np.array(
                                [value for value in linevalues])
                        axs[i][j].plot(xparam[(~np.isinf(linevalues)) & (~np.isnan(linevalues))], values[(~np.isinf(
                            linevalues)) & (~np.isnan(linevalues))], color=color, ls='solid', lw=2*linewidth)

                    if dataMarker is not None:
                        markervalues = np.array(
                            [value for value in datamarkervalues[str(ID)].values])
                        if len(markervalues.shape) > 1:
                            markervalues = markervalues.T[0]
                            markervalues = np.array(
                                [value for value in markervalues])

                        if 'Merger' in dataMarker:
                            SnapCorotateMerger = Sample.loc[Sample.SubfindID_99 == ID, 'SnapCorotateMergers'].values[0]
                            
                            mergerTot = np.array(
                                [value for value in datamarkerTotvalues[str(ID)].values])
                            if len(mergerTot.shape) > 1:
                                mergerTot = mergerTot.T[0]
                                mergerTot = np.array(
                                    [value for value in mergerTot])

                            MarkerTotvalues = np.array(
                                [value for value in datamarkerTotvalues[str(ID)].values])
                            if len(MarkerTotvalues.shape) > 1:
                                MarkerTotvalues = MarkerTotvalues.T[0]
                                MarkerTotvalues = np.array(
                                    [value for value in MarkerTotvalues])

                            mergernumber = np.array(
                                [value for value in datamarkervalues[str(ID)].values])
                            if len(mergernumber.shape) > 1:
                                mergernumber = mergernumber.T[0]
                                mergernumber = np.array(
                                    [value for value in mergernumber])

                            Mergernumber = np.array(
                                [value for value in dataMarkervalues[str(ID)].values])
                            if len(Mergernumber.shape) > 1:
                                Mergernumber = Mergernumber.T[0]
                                Mergernumber = np.array(
                                    [value for value in Mergernumber])

                            Markervalues = np.array(
                                [value for value in datamarkervalues[str(ID)].values])
                            if len(Markervalues.shape) > 1:
                                Markervalues = Markervalues.T[0]
                                Markervalues = np.array(
                                    [value for value in Markervalues])

                            mergernumber = np.flip(mergernumber)
                            Mergernumber = np.flip(Mergernumber)
                            mergerTot = np.flip(mergerTot)

                            for nmergerindex, nmerger in enumerate(mergernumber):

                                if nmergerindex == 0:
                                    markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            markervalues[nmergerindex] = int(
                                                nmerger) - int(mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(Mergernumber):

                                if nmergerindex == 0:
                                    Markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        Markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            Markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            Markervalues[nmergerindex] = int(
                                                nmerger) - int(Mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(mergerTot):

                                if nmergerindex == 0:
                                    MarkerTotvalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        MarkerTotvalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger) - int(mergerTot[nmergerindex - 1])
                            MarkerTotvalues = MarkerTotvalues - Markervalues - markervalues
                            Markervalues = np.flip(Markervalues)
                            markervalues = np.flip(markervalues)
                            MarkerTotvalues = np.flip(MarkerTotvalues)

                        axs[i][j].scatter(xparam[(Markervalues > 0)], values[(Markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=250, alpha=0.7)
                        axs[i][j].scatter(xparam[(markervalues > 0)], values[(markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='s',  edgecolors='black', s=100, alpha=0.7)
                        
                        if ~np.isnan(SnapCorotateMerger):
                            axs[i][j].scatter(time[(MarkerTotvalues > 0)], values[(MarkerTotvalues > 0)], 
                                              color=colors.get(str(l), 'black'), lw=1., marker='*',  edgecolors='black', s=300, alpha=0.7)

                elif Type == 'CoEvolution':
                    x = dfX[str(ID)].values
                    
                    if 'Type4' in row[l] or 'StarMass' in row[l]:
                        color = 'blue'
                    elif 'Type0' in row[l] or 'GasMass' in row[l]:
                        color = 'green'
                    elif 'Type1' in row[l] or 'DMMass' in row[l]:
                        color = 'purple'
                    else:
                        color = 'black'
                        
                    if len(x.shape) > 1:
                        x = np.array([value for value in x.T[0]])
                    else:
                        x = np.array([value for value in x])
                    colorSnap = np.array(
                        ['magenta', 'blue', 'cyan', 'lime', 'darkorange', 'red'])
                    if Xparam[i] != 'tsincebirth':
                        axs[i][j].scatter(x[99-snapsTime], values[99-snapsTime], color=colorSnap,
                                          lw=1., marker='d',  edgecolors=color, s=100, alpha=0.9)
                        axs[i][j].scatter(x[0], values[0], color='black', lw=1.,
                                          marker='o',  edgecolors=color, s=70, alpha=0.9)
                    argnotnan = ~np.isnan(values)
                    axs[i][j].plot(x[argnotnan], values[argnotnan], color=color, ls= 'solid')

                    if dataLine is not None:
                        linevalues = np.array(
                            [value for value in datalinevalues[str(ID)].values])
                        if len(linevalues.shape) > 1:
                            linevalues = linevalues.T[0]
                            linevalues = np.array(
                                [value for value in linevalues])
                        axs[i][j].plot(x[(~np.isinf(linevalues)) & (~np.isnan(linevalues))], values[(~np.isinf(linevalues)) & (
                            ~np.isnan(linevalues))], color=color, ls='solid', lw=3.)

                    if dataMarker is not None:
                        markervalues = np.array(
                            [value for value in datamarkervalues[str(ID)].values])
                        if len(markervalues.shape) > 1:
                            markervalues = markervalues.T[0]
                            markervalues = np.array(
                                [value for value in markervalues])

                        if 'Merger' in dataMarker:
                            mergerTot = np.array(
                                [value for value in datamarkerTotvalues[str(ID)].values])
                            if len(mergerTot.shape) > 1:
                                mergerTot = mergerTot.T[0]
                                mergerTot = np.array(
                                    [value for value in mergerTot])

                            MarkerTotvalues = np.array(
                                [value for value in datamarkerTotvalues[str(ID)].values])
                            if len(MarkerTotvalues.shape) > 1:
                                MarkerTotvalues = MarkerTotvalues.T[0]
                                MarkerTotvalues = np.array(
                                    [value for value in MarkerTotvalues])

                            mergernumber = np.array(
                                [value for value in datamarkervalues[str(ID)].values])
                            if len(mergernumber.shape) > 1:
                                mergernumber = mergernumber.T[0]
                                mergernumber = np.array(
                                    [value for value in mergernumber])

                            Mergernumber = np.array(
                                [value for value in dataMarkervalues[str(ID)].values])
                            if len(Mergernumber.shape) > 1:
                                Mergernumber = Mergernumber.T[0]
                                Mergernumber = np.array(
                                    [value for value in Mergernumber])

                            Markervalues = np.array(
                                [value for value in datamarkervalues[str(ID)].values])
                            if len(Markervalues.shape) > 1:
                                Markervalues = Markervalues.T[0]
                                Markervalues = np.array(
                                    [value for value in Markervalues])

                            mergernumber = np.flip(mergernumber)
                            Mergernumber = np.flip(Mergernumber)
                            mergerTot = np.flip(mergerTot)

                            for nmergerindex, nmerger in enumerate(mergernumber):

                                if nmergerindex == 0:
                                    markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            markervalues[nmergerindex] = int(
                                                nmerger) - int(mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(Mergernumber):

                                if nmergerindex == 0:
                                    Markervalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        Markervalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            Markervalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            Markervalues[nmergerindex] = int(
                                                nmerger) - int(Mergernumber[nmergerindex - 1])

                            for nmergerindex, nmerger in enumerate(mergerTot):

                                if nmergerindex == 0:
                                    MarkerTotvalues[nmergerindex] = 0
                                    continue
                                else:
                                    if np.isnan(nmerger):
                                        MarkerTotvalues[nmergerindex] = 0
                                    else:
                                        if np.isnan(mergernumber[nmergerindex - 1]):
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger)
                                        else:
                                            MarkerTotvalues[nmergerindex] = int(
                                                nmerger) - int(mergerTot[nmergerindex - 1])
                            MarkerTotvalues = MarkerTotvalues - Markervalues - markervalues
                            Markervalues = np.flip(Markervalues)
                            markervalues = np.flip(markervalues)
                            MarkerTotvalues = np.flip(MarkerTotvalues)

                        axs[i][j].scatter(x[(Markervalues > 0)], values[(Markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=130, alpha=0.5)
                        axs[i][j].scatter(x[(markervalues > 0)], values[(markervalues > 0)], color=colors.get(
                            str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=110, alpha=0.5)
                        #axs[i][j].scatter(x[(MarkerTotvalues > 0)], values[(MarkerTotvalues > 0)], color=colors.get(
                            #str(l), 'black'), lw=1., marker='o',  edgecolors='black', s=15, alpha=0.5)

            # Plot details

            if row[-1] == 'StarMassNormalized':
                axs[i][j].set_yticks([0.1, 0.2, 0.5, 1])
                axs[i][j].set_yticklabels(['0.1','0.2', '0.5', '1'])
                
            

            if GridMake:
                axs[i][j].grid(GridMake, color='#9e9e9e',  which="major", linewidth= 0.6,alpha= 0.3 , linestyle=':')
               
            axs[i][j].tick_params(axis='y', labelsize=0.99*fontlabel)
            axs[i][j].tick_params(axis='x', labelsize=0.99*fontlabel)
	    
            
            if ylimmin != None and ylimmax != None:
                axs[i][j].set_ylim(ylimmin[i], ylimmax[i])
            if scales.get(row[0]) != None :
                axs[i][j].set_yscale(scales.get(row[0]))
            if scales.get(row[0]) == 'log' :
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))
                
            if row[-1] == 'FracType1':
                axs[i][j].set_yticks([0.2, 0.3, 0.5, 1])
                axs[i][j].set_yticklabels(['0.2','0.3', '0.5', '1'])

            
            if j == 0:

                if len(row) > 1:
                    axs[i][j].set_ylabel(
                        labelsequal.get(row[0]), fontsize=fontlabel)

                else:
                    axs[i][j].set_ylabel(labels.get(row[0]), fontsize=fontlabel)

            if j == len(IDs) - 1:
                if xlabelintext and not limaxis and len(rows) > 1:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        texts.get(row), loc='upper right', prop=Afont)
                    axs[i][j].add_artist(anchored_text)

            if xlabelintext and limaxis and len(rows) > 1:
                Afont = {'color':  'black',
                         'size': fontlabel,
                         }
                anchored_text = AnchoredText(
                    texts[row], loc='upper left', prop=Afont)
                axs[i][j].add_artist(anchored_text)
                
            if j == 0:
                
                
                if title != None and ColumnPlot:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        titles.get(
                            title[i], title[i]), loc=postext[i], prop=Afont)
                    axs[i][j].add_artist(anchored_text)


            if i == 0:

                if title and not ColumnPlot:
                    axs[i][j].set_title(titles.get(
                        title[j], title[j]), fontsize=1*fontlabel)
                

                if Type == 'Evolution' and not PhasingPlot:

                    axs[i][j].tick_params(bottom=True, top=False)
                    lim = axs[i][j].get_xlim()
                    ax2label = axs[i][j].twiny() #secondary_xaxis('top', which='major')
                    ax2label.grid(False)
                    ax2label.set_xlim(lim)

                    if row == 'rToRNearYoung' or savefigname == 'Young':
                        zticks = np.array([0., 0.2])
                        zlabels = np.array(
                            ['0', '0.2'])
                        zticks_Age = np.array(
                            [13.803, 11.323])
                    elif not PhasingPlot:
                        zticks = np.array([0., 0.2, 0.5, 1., 2., 5., 20.])
                        if SmallerScale:

                            if j == 0:
                                zlabels = np.array(
                                    ['0', '0.2', '0.5', '1', '2', '5', '20'])
                            if j != 0:
                                zlabels = np.array(
                                    ['0', '0.2', '0.5', '1', '2', '5', ''])
                        else:
                            zlabels = np.array(
                                ['0', '0.2', '0.5', '1', '2', '5', '20'])
                        zticks_Age = np.array(
                            [13.803, 11.323, 8.587, 5.878, 3.285, 1.2, 0])


                    zticks = zticks.tolist()
                    zticks_Age = zticks_Age.tolist()

                    x_locator = FixedLocator(zticks_Age)
                    x_formatter = FixedFormatter(zlabels)
                    ax2label.xaxis.set_major_locator(x_locator)
                    ax2label.xaxis.set_major_formatter(x_formatter)
                    ax2label.set_xlabel(r"$z$", fontsize=fontlabel)
                    ax2label.tick_params(labelsize=0.85*fontlabel)
                    ax2label.tick_params(axis='x',  which='minor', top=False)


            if i == len(rows) - 1:
                
                if Type == 'Evolution':
                    
                    
                    if row == 'rToRNearYoung' or savefigname == 'Young':
                        axs[i][j].set_xlabel(r'$t \, \,  [\mathrm{Gyr}]$', fontsize=fontlabel)
                        axs[i][j].set_xticks([10, 12, 14])
                        axs[i][j].set_xticklabels(
                            ['10', '12', '14'])
                    elif not PhasingPlot:
                        if LookBackTime:
                            axs[i][j].set_xticks([0.  ,  1.97185714,  3.94371429,  5.91557143,  7.88742857, 9.85928571, 11.83114286, 13.803  ])
                            if SmallerScale:
                                fig.supxlabel(r'$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$', fontsize=fontlabel, y = 0.07)

                                if j == 0:
                                    axs[i][j].set_xticklabels(
                                    ['14', '12', '10', '8', '6', '4', '2', '0'])
                                if j != 0:
                                    axs[i][j].set_xticklabels(
                                    ['', '12', '10', '8', '6', '4', '2', '0'])
                            else:
                                axs[i][j].set_xlabel(r'$\mathrm{Lookback \; Time} \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)

                                axs[i][j].set_xticklabels(
                                    ['14', '12', '10', '8', '6', '4', '2', '0'])
                        else:
                            axs[i][j].set_xticks([0, 2, 4, 6, 8, 10, 12, 14])

                            axs[i][j].set_xlabel(r'$t \, \, [\mathrm{Gyr}]$', fontsize=fontlabel)
        
                            axs[i][j].set_xticklabels(
                                ['0', '2', '4', '6', '8', '10', '12', '14'])

                    elif PhasingPlot:
                        axs[i][j].set_xlabel(r'$\phi_\mathrm{Orbital}$', fontsize=fontlabel)
                        axs[i][j].set_xticks([-1, -0.5, 0, 1, 2, 3, 4, 5] )
                        axs[i][j].set_xticklabels(['', 'E', '0', '1', '2', '3', '4', '5'])
                        axs[i][j].set_xlim(-1, 5.5)
                        
                elif Type == 'CoEvolution':
                    axs[i][j].set_xscale(scales.get(Xparam[i], 'linear'))
                    if scales.get(Xparam[i]) == 'log':
                        axs[i][j].xaxis.set_major_formatter(
                            FuncFormatter(format_func_loglog))
                    axs[i][j].set_xlabel(labels.get(
                        Xparam[i], 'None'), fontsize=fontlabel)

    savefig(savepath, savefigname, TRANSPARENT)

    return



def MakeMedianAndIDs(Snaps, IDs, rmin, rmax, nbins, dfSample, PartType = 'PartType4', velPlot= False):
    yIDs  = np.array([])
    massIDs = np.array([])
    xIDs = np.array([])
    notIndex = np.array([])
    
    for l, ID in enumerate(IDs):
        if ID == 603556 or ID == 602133:
            continue
        snap = Snaps[l]
        if np.isnan(snap):
            continue
        snap = int(snap)
        #print('snap: ', snap)
        yrad, rad, mass = TNG.MakeDensityProfileMean(snap, ID, rmin, rmax, nbins, PartType = PartType, velPlot= velPlot)
        
        if len(yrad) == 1 or (ID == 603556 or ID == 602133):
            notIndex = np.append(notIndex, l)
            continue
        if l == 0 or len(yIDs ) == 0:
            yIDs  = np.append(yIDs , yrad)
            xIDs = np.append(xIDs, rad)
            massIDs = np.append(massIDs, mass)
    
        else:
            yIDs  = np.vstack((yIDs , yrad))
            massIDs = np.vstack((massIDs, mass))
            xIDs = np.vstack((xIDs, rad))
           
    
        Rvalues = xIDs.T
        Values = yIDs .T
        Masses = massIDs.T
    
    x = np.array([])
    y = np.array([])
    yerr = np.array([])
    mass = np.array([])

    if len(Values) > 0:
        if len(Values.shape) > 1:
            for k, value in enumerate(Values):
                x = np.append(x, np.nanmedian(Rvalues[k]))
                y = np.append(y, np.nanmedian(value))
                yerr = np.append(yerr, MATH.boostrap_func(value, func=np.nanmedian, num_boots=1000))
                mass = np.append(mass, np.nanmedian(Masses[k]))
        else:
            x = Rvalues
            y = Values
            yerr = np.zeros(len(y))
            mass = Masses
    
    else:
        x = np.nan
        y = np.nan
        yerr = np.nan
        mass = np.nan
            
    return x, y,yerr, mass, xIDs, yIDs, massIDs, notIndex

def MakeLines(j, ax,  yIDs, xIDs, IDs, notIndex, colors):
    k = 0
    for l, ID in enumerate(IDs):
        if l in notIndex:
            continue
        
        yvalues = yIDs[k]
        xvalues = xIDs[k]
        alpha = 0.3
        if j == 1 or j == 2:
            xvalues = xvalues[yvalues > 0] 
            yvalues = yvalues[yvalues > 0]*xvalues**2.
            if j == 2:
                try:
                    if yvalues[xvalues == 1.01871524] > 5e7:
                        alpha = 0
                except:
                    None
            ax.plot(xvalues, yvalues , 
                                 lw = 0.82,  alpha = alpha,  color = colors[k])
        else:
            ax.plot(xvalues , yvalues, 
                                     lw = 0.82,  alpha = alpha,  color = colors[k])

        k = k+ 1
        
        #y_p2 = np.percentile(yIDs, 25, axis=0)     # 2.5th percentile
    #y_p97 = np.percentile(yIDs, 75, axis=0)   # 97.5th percentile
    #if j == 0:
    #    axs[j][linplot].fill_between(xvalues[~np.isnan(y_p2)], y_p2[~np.isnan(y_p2)], y_p97[~np.isnan(y_p97)], color=ColorFill, alpha=0.2)  # 2σ equivalent
    #else:
    #    axs[j][linplot].fill_between(xvalues[~np.isnan(y_p2)] , y_p2[~np.isnan(y_p2)]  * xvalues[~np.isnan(y_p2)]**2., y_p97[~np.isnan(y_p97)]  * xvalues[~np.isnan(y_p97)]**2., color=ColorFill, alpha=0.2)  # 2σ equivalent

