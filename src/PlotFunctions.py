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
from matplotlib.gridspec import GridSpec

from matplotlib.patches import Circle
from matplotlib.legend_handler import HandlerPatch

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

plt.style.use(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/src/abhner.mplstyle")

#Paths
SaveSubhaloPath = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory/'

SIMTNG = 'TNG50'
Nsim = '-1'
MAIN_SAVE_FIG = os.getenv("HOME")+'/TNG_Analyzes/Figs/' + SIMTNG + '/'


dfTime = pd.read_csv(os.getenv("HOME")+'/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/utils/SNAPS_TIME.csv')



class HandlerCircle(HandlerPatch):
    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent, width, height, fontsize, trans
    ):
        radius = min(width, height) / 2.
        center = (xdescent + width / 2, ydescent + height / 2)

        circle = Circle(
            center,
            radius=radius,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            linewidth=orig_handle.get_linewidth(),
            linestyle=orig_handle.get_linestyle()
        )

        circle.set_transform(trans)
        return [circle]
    
    
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

            elif 'Selected' in name:
                lwe = 1.
                
            
                
                
            else:
                lw = lwe = 0
                
                Empty = False
                BlackLine = False
            if name == 'Bian et al. (2025)':
                custom_lines.append(Patch(facecolor='tab:red', alpha = 0.4))
            else:
                if 'Normal' in name or 'SubDiffuse' in name:
                    msizeMult = 1.9
                    
                    if 'Colorbar' in name:
                        msizeMult = 0.8
                        
                elif 'Diffuse' in name:
                    msizeMult = 1.1
                    
                elif 'SBC' in name or 'MBC' in name :
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
                    if 'LoseTheirGas' in name:
                        custom_lines.append(
                            Circle(
                                (0, 0),
                                radius=2,
                                facecolor='white',
                                edgecolor='k',
                                linewidth=lwe,
                                linestyle=lines(name)   # '-' ou '--'
                            )
                        )
                        
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
            
        elif 'Hist' in name:
            name = name.replace('Hist', '')
        
            if 'Empty' in name:
                name = name.replace('Empty', '')
                custom_lines.append(
                    Patch(
                        facecolor='none',
                        edgecolor=colors(name),
                        linewidth=mult * 0.5 * linesthicker(name)
                    )
                )
            else:
                custom_lines.append(
                    Patch(
                        facecolor=colors(name),
                        edgecolor=colors(name),
                        linewidth=mult * 0.5 * linesthicker(name)
                    )
                )
        
            label.append(titles(name))
        
        else:
            if '_Norm_Max' in name:
                mult = 0.7
                
            if 'CO' in name:
                custom_lines.append(Line2D([0], [0], color='black', ls=lines(name), 
                                           lw=mult * 0.5* linesthicker(name), dash_capstyle = capstyles(name)))
                  
            else:
                
                custom_lines.append(Line2D([0], [0], color=colors(name), ls=lines(name), 
                                           lw=mult * 0.5 * linesthicker(name), dash_capstyle = capstyles(name)))
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
    ColumnPlot=True,  lineparams=False, PhasingPlot=False,  PhasingMedianLine = False, LookBackTime=True,

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
                            print(row, column)

                            if ("GasMass" in row) and ("LoseTheir" in column):
                                print('Passou!')
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
                            lw=1.5 * linewidth,
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
                            if not 'Central' in names[l] + column:
                                snap_first = int(np.nanmedian(dfPop.Snap_At_FirstEntry))
                                if not np.isnan(snap_first):
                                   x_entry = dfTime.Age.loc[dfTime.Snap == snap_first].values[0]
                                   ax.axvline( x_entry,
                                               color=colors(names[l]),
                                               lw=0.8, ls = 'dashed',
                                               alpha=0.8, zorder = 0)
                                   # ax.plot(
                                   #      [x_entry, x_entry], [0.98, 1.03],
                                   #      transform=ax.get_xaxis_transform(),
                                   #      color=colors(names[l]),
                                   #      lw=2.6,
                                   #      alpha=1.,
                                   #      clip_on=False
                                   #  )
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
                            lw=1.5 * linewidth,
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
                            if PhasingMedianLine:
                                xParam = TNG.MedianPhasePopulation(
                                            names[l]+column,
                                            dfName=dfName,
                                            Name=Name
                                        )
                                xParam = np.asarray(xParam, dtype=float)

                                # Inverta xParam
                                xParam = np.flip(xParam)

                                # IMPORTANTE: se values está na mesma ordem de snapshot que xParam original,

                                eps = 1e-8

                                COND_Phase = np.zeros_like(xParam, dtype=bool)

                                last_kept = np.inf

                                for icond, x in enumerate(xParam):
                                    if not np.isfinite(x):
                                        continue

                                    if x < last_kept - eps:
                                        COND_Phase[icond] = True
                                        last_kept = x
                            else:
                                xParam = time
                                
                            timeParam = time
                            phaseParam = time

                        if Type == "Evolution":
                            err = _safe_array(dataerr[l])

                            # thresholds
                            if "sSFR" in str(paramname):
                                values[values <= -14] = np.nan
                            elif "SFRE" in str(paramname):
                                values[values <= -11.5] = np.nan
                            elif "SFR" in str(paramname) and not 'SFRE' in str(paramname):
                                values[values <= -4] = np.nan
                            # Legacy: if phasing plot + “LoseTheir” gas case: cut after first NaN
                            if PhasingPlot:

                                if ("GasMass" in row[0] and "GasMass" in row[1]):#and ("LoseTheir" in column):
                                    arg_inf = np.argwhere(np.isinf(values)).T[0]
                                    arg_inf = arg_inf[arg_inf > 5]
                                    if len(arg_inf) > 0:
                                        values[arg_inf[0]:] = np.nan

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
                            if PhasingMedianLine:

                                m = m & COND_Phase

                                m = m & np.isfinite(values)
                                print(xParam[m], values[m])

                                m = m & (xParam < 2) #CHECK
                                
                                
                            ax.plot(
                                xParam[m], values[m],
                                color=colors(names[l]),
                                ls=lines(paramname),
                                lw=1.5 * linewidth,
                                dash_capstyle=capstyles(paramname),
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
                                try:
                                    snap_first = int(np.nanmedian(dfPop.Snap_At_FirstEntry))
                                    if not np.isnan(snap_first):
                                       x_entry = dfTime.Age.loc[dfTime.Snap == snap_first].values[0]
                                       ax.axvline( x_entry,
                                                   color=colors(names[l]),
                                                   lw=0.8, ls = 'dashed',
                                                   alpha=0.8, zorder = 0)
                                       # ax.plot(
                                       #      [x_entry, x_entry], [0.98, 1.03],
                                       #      transform=ax.get_xaxis_transform(),
                                       #      color=colors(names[l]),
                                       #      lw=2.6,
                                       #      alpha=1.,
                                       #      clip_on=False
                                       #  )
                                except:
                                    None

                        else:
                            x = _safe_array(dataX[l])
                            m = _finite_mask(values) & _finite_mask(x)
                            ax.plot(
                                x[m], values[m],
                                color=colors(names[l]),
                                ls=lines(paramname),
                                lw=0.9 * linesthicker(paramname),
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
                
            if p_for_scale == "Frac_ExSitu":
                ax.set_yticks([0.01, 0.05, 0.1])
                ax.set_yticklabels(["0.01", "0.05", "0.1"])

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
                    _make_anchored_text(ax, Text[i] if i < len(Text) else None, "upper left")

                if xlabelintext and (not limaxis) and len(rows) > 1:
                    _make_anchored_text(ax, texts.get(p_for_scale, p_for_scale), "upper right")

            # Top row titles + redshift axis for Evolution (non-phasing)
            if i == 0:
                if title:
                    ax.set_title(titles(title[j]), fontsize=1.1 * fontlabel)

                if Type == "Evolution" and (not PhasingPlot and not PhasingMedianLine):
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
                    

                    if LookBackTime and (not PhasingPlot and not PhasingMedianLine):
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
                    elif not PhasingPlot and not PhasingMedianLine:
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

                        ax.set_xlabel(r"$\phi_\mathrm{orbital}$", fontsize=fontlabel)
                        ax.set_xticks(postiveXticks)
                        ax.set_xticklabels(postiveXLabels)
                        ax.set_xlim(-1, xPhaseLim)

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
    toplim=1e2,limaixsy=False, liminvalue=(0,), limax=(1,),

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
        
        
        if param == 'deltaSize_at_Entry':
            bins_edges = np.linspace(-1.5, 3.5, 11)
            ax.hist(v, bins=bins_edges, alpha=1, histtype='step',
                    color=colors(name_key), ls=lines(name_key),
                    density=density, linewidth=linewidth)
            return
        
        if param == 'U-r':
            bins_edges = np.linspace(-0.2, 2, 10)
            ax.hist(v, bins=bins_edges, alpha=1, histtype='step',
                    color=colors(name_key), ls=lines(name_key),
                    density=density, linewidth=linewidth)
            return
        
        if param == 'sSFRinHalfRadAfterz5':
            bins_edges = np.linspace(-11.8, -9.2, 9)
            ax.hist(v, bins=bins_edges, alpha=1, histtype='step',
                    color=colors(name_key), ls=lines(name_key),
                    density=density, linewidth=linewidth)
            return
        
        if 'logSUM_Mstar_merger' in param:
            bins_edges = np.linspace(4.8, 8.2, 7)
            ax.hist(v, bins=bins_edges, alpha=1, histtype='step',
                    color=colors(name_key), ls=lines(name_key),
                    density=density, linewidth=linewidth)
            return
        
        if 'logStarZ_99' in param or 'logZ' in param:
            bins_edges = np.linspace(-0.8, -0.45, 5)
            bins_edges = np.append(bins_edges, np.linspace(-0.45, 0.1, 8))
            bins_edges = np.unique(bins_edges)

            ax.axvline(-0.45, c = 'black', ls = '--')
            
            if 'BadFlag' in name_key:
                ax.hist(v, bins=bins_edges, alpha=1, histtype='step',
                    color=colors(name_key), ls=lines(name_key),
                    density=density, linewidth=linewidth)
            else:
                ax.hist(v, bins=bins_edges, alpha=1, histtype='stepfilled',
                    color='tab:green', ls=lines(name_key),
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
            ymax = 0.5
            ax.axvline(center, ymax=ymax, color=colors(name_key),
                       ls=lines(name_key), linewidth=1.5 * linewidth) # ymax=ymax,
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
            ymax = 0.5
            ax.axvline(center, ymax=ymax, color=colors(name_key),
                       ls='solid', linewidth=1.8 * linewidth, zorder = 5) 
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
            
            # if param == 'U-r':
            #     NormalSatellite = TNG.extractPopulation('NormalSatelliteDMpoor', dfName = 'PaperII')
            #     color_Normal = NormalSatellite['U-r'].values
            #     _overlay_stat(ax, color_Normal, 'Normal', which= 'median')

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
                    ax.set_title(titles(title[j]), fontsize=1.1 * fontlabel, y = 1.01)

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
                if 'sSFRinHalfRadAfterz5' in param:
                    ax.set_yticks([1, 10, 100])
                    ax.set_yticklabels(['1', '10', '100'])
                        
                # if 'logStarZ_99' in param or 'logZ_99' in param:
                #     ax.set_yticks([10, 20, 30, 40, 50, 60, 70])
                #     ax.set_yticklabels(['10', '20', '30', '40', '50', '60', '70'])

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
    medianBins=False, medianAll=False, medianDotStar = False, medianDot=False, SpearManTest=False, SpearManTestAll=False,
    bins=10, quantile=0.95, q=0.95, HIGHLIGHTPoints=False,

    # --- Layout ---
    lNum=6, cNum=6, xscale = None, yscales = None, GridMake=False, InvertPlot=False, xlabelintext=False, title=False,

    # --- Helper lines ---
    EqualLine=False, EqualLineMin=None, EqualLineMax=None,

    # --- Limits ---
    xlimmin=None, xlimmax=None, ylimmin=None, ylimmax=None,

    # --- Style ---
    cmap="inferno", m="o", msizet=30, msizetstar = 30, msizeMult=1, alphaScater=1.0, alphaShade=0.3, linewidth=1.2, fontlabel=26, framealpha=0.95,

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
            ax.plot(x, y, color="darkorange", linestyle="dashed", lw=linewidth)


            ax.axvline(0,color = 'black',linestyle='dashed',lw=linewidth)
            ax.axhline(0,color = 'black',linestyle='dashed',lw=linewidth)
            
            ax.fill_between([0, 500], -500, 0, alpha=0.2, color='tab:green')  # yellow
            ax.fill_between([-500, 0], -500, 0, alpha=0.2, color='tab:red')  # orange
            ax.fill_between([0, 500], 0, 500, alpha=0.2, color='tab:blue')  # red
            ax.text(-.145, -0.95, 'TS', fontsize = 0.98*fontlabel)
            ax.text(0.1, 0.02, 'SF', fontsize = 0.98*fontlabel)
            ax.text(0.15,-0.95,  'Interplay', fontsize = 0.98*fontlabel)
            
        elif (ParamX == 'Relative_logInnerZ_At_Entry' and  (ParamsY[0] == 'Relative_logZ_At_Entry')) :
            xfitline  = np.linspace(0 ,1, 100)
            axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth)
            axs[i][j].plot( xfitline, np.zeros(100), ls='--', color='k', linewidth=linewidth, zorder = 1)


        elif (ParamX[0] == 'RadEx' and  (ParamsY[0] == 'RadIn')) :
            xfitline  = np.linspace(0.7 , 25, 100)
            axs[i][j].plot( xfitline, xfitline, ls='--', color='gray', linewidth=linewidth)


        elif (ParamX == 'Relative_Rhalf_MaxProfile_Minus_HalfRadstar_Entry' and  (ParamsY[0] == 'Relative_Rhalf_MinProfile_Minus_HalfRadstar_Entry')) :

            xfitline  = np.linspace(-6 ,2, 100)
            ax.axvline(0,color = 'black',linestyle='dashed',lw=linewidth, zorder = 1)
            axs[i][j].plot( xfitline, xfitline, ls='--', color='tab:blue', linewidth=linewidth, zorder = 1)
            #axs[i][j].fill_between(xfitline, -7, xfitline, alpha=0.2, color='tab:red')  # orange
            axs[i][j].text(-1.15, -1.58, "Outer stellar profile \n evolution \n dominates", fontsize = 0.99*fontlabel)
            #axs[i][j].fill_between(xfitline, xfitline, 1, alpha=0.2, color='tab:blue')  # orange
            axs[i][j].text(-1.8,-0.75, "Inner stellar profile \n evolution \n dominates", fontsize = 0.99*fontlabel)
            
        elif (ParamX == 'sSFRTrueInner_BeforeEntry' and  (ParamsY[0] == 'sSFRTrueInner_Entry_to_Nogas')) :

           x = np.linspace(-12,-8)
           axs[i][j].plot( x, x, ls='--', color='tab:blue', linewidth=linewidth)
        
           xfitline  = np.linspace(-13 ,-7, 100)
           #axs[i][j].fill_between(xfitline, -12, xfitline, alpha=0.2, color='tab:red')  # orange
           axs[i][j].text(-10,-10.55, "Inner $\overline{\mathrm{sSFR}}$ \n decrease", fontsize = 0.99*fontlabel)
           #axs[i][j].fill_between(xfitline, xfitline,-8, alpha=0.2, color='tab:blue')  # orange
           axs[i][j].text(-10.9,-9.5, "Inner $\overline{\mathrm{sSFR}}$  \n increase", fontsize = 0.99*fontlabel)

     

        elif (ParamX == "Decrease_Entry_To_NoGas_Norm_Delta" and (firstY == "Decrease_NoGas_To_Final_Norm_Delta")):
            xfitline = np.linspace(-2, 0.4, 100)
            # ax.fill_between(xfitline, -1, xfitline, alpha=0.2, color="tab:red")
            # ax.fill_between(xfitline, xfitline, 0.25, alpha=0.2, color="tab:blue")
            ax.text(-0.6, -0.8, "Faster compaction \n after gas loss", fontsize=0.99 * fontlabel)
            ax.text(-0.80, -0.25, "Faster compaction  \n with gas ", fontsize=0.99 * fontlabel)
            ax.axvline(0, color="black", linestyle="dashed", lw=linewidth)
            ax.axhline(0, color="black", linestyle="dashed", lw=linewidth)
            ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
            ax.set_xticklabels(["-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2"])
            ax.set_yticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
            ax.set_yticklabels(["-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2"])
            
        elif (ParamX == "global_color_U_minus_r_1xRh" and (firstY == "ratio_color_U_minus_r_1xRh")):
            xfitline = np.linspace(-0.75, 2.5, 100)
            ax.fill_between(xfitline , -1.5, 0 , alpha=0.1, color="tab:blue")
            # ax.fill_between(xfitline, xfitline, 0.25, alpha=0.2, color="tab:blue")
            ax.axvline(0, color="black", linestyle="dashed", lw=linewidth)
            ax.axhline(0, color="black", linestyle="dashed", lw=linewidth)
           
        elif (ParamX == "Decrease_Entry_To_NoGas" and (firstY == "Decrease_NoGas_To_Final")):
            #xfitline = np.linspace(-2, 0.4, 100)
            # ax.fill_between(xfitline, -1, xfitline, alpha=0.2, color="tab:red")
            # ax.fill_between(xfitline, xfitline, 0.25, alpha=0.2, color="tab:blue")
            ax.text(-0.55, -1.3, "Larger compaction \n after gas loss", fontsize=0.99 * fontlabel)
            ax.text(-1.2, 0.2, "Larger compaction  \n with gas ", fontsize=0.99 * fontlabel)
            ax.axvline(0, color="black", linestyle="dashed", lw=linewidth, zorder = 1)
            ax.axhline(0, color="black", linestyle="dashed", lw=linewidth, zorder = 1)
            # ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
            # ax.set_xticklabels(["-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2"])
            # ax.set_yticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
            # ax.set_yticklabels(["-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2"])

        elif (ParamX == "Rhalf_MaxProfile_Minus_HalfRadstar_Entry" and (firstY == "Rhalf_MinProfile_Minus_HalfRadstar_Entry")):
            xfitline = np.linspace(-6, 2, 100)
            ax.plot(xfitline, xfitline, ls="--", color="tab:blue", linewidth=linewidth)
            ax.fill_between(xfitline, -7, xfitline, alpha=0.2, color="tab:blue")
            ax.text(-2, -5, "TS", fontsize=0.99 * fontlabel)
            ax.fill_between(xfitline, xfitline, 1, alpha=0.2, color="tab:red")
            ax.text(-4.0, -1, "SF", fontsize=0.99 * fontlabel)
            
        elif  'StarFrac' in ParamX  and 'GasFrac' in firstY  :

            x = np.linspace(0, 1)
            ax.plot( x, x, ls='--', color='gray', linewidth=linewidth, zorder = 0)
            
        elif  'StarFrac' in ParamX  and 'DMFrac' in firstY:

            x = np.linspace(0, 1)
            ax.plot( x, 1 - x, ls='dotted', color='gray', linewidth=linewidth, zorder = 0)
            
        elif  ('z_Birth' in ParamX  and ('DMFrac_Birth' in firstY) ) :

            ax.axhline( 0.8, ls='--', color='tab:red', linewidth=linewidth)

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
        if NoneEdgeColor:
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
                   lw=0.9,
                   marker=markers(name),
                   s=msizet * msize(name))
        return None, None

    def _apply_post_panel_formatting(ax, yparam, yscale = None):
        if GridMake:
            ax.grid(GridMake, color="#9e9e9e", which="major", linewidth=0.6, alpha=0.3, linestyle=":")

        if yscale != None:
            ax.set_yscale(yscale)
            
            if yscale in ("log", "symlog"):
                ax.yaxis.set_major_formatter(FuncFormatter(format_func_loglog))

        else:
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
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0,  0.5,  1, 1.5,  2], pad=0.02, aspect=(ratioColorbar or 50))
                cb.ax.set_yticklabels(['0',  '0.5', '1', '1.5', '2'])
            elif COLORBAR[0]  in ["logStarZ_99"]:
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0, 0.1, 0.2, 0.3, 0.7], pad=0.02, aspect=(ratioColorbar or 50))

                cb.ax.set_yticklabels(['0', '0.1', '0.2', '0.3', '0.7'])
                
            elif COLORBAR[0]  in ["logStarZ_99_75dex"]:
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[-0.7, -0.6, -0.5,  -0.4, -0.3, -0.2, -0.1, 0.], pad=0.02, aspect=(ratioColorbar or 50))

                cb.ax.set_yticklabels(['-0.7', '-0.6', '-0.5', '-0.4', '-0.3', '-0.2', '-0.1', '0'])
                
            elif COLORBAR[0]  in ["z_At_FirstEntry"]:
                cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), ticks=[0.0, 0.5, 1.0, 1.5], pad=0.02, aspect=(ratioColorbar or 50))

                cb.ax.set_yticklabels(['0', '0.5', '1.0', '1.5'])
                
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
            if i == 1 and yparam == 'DMFrac_99' and ParamX == 'StarFrac_99':
                _apply_special_background_rules(ax, ParamX, yparam, linewidth, fontlabel)

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
                        fmt="s", c=colors(name),
                    )

                elif medianDot:
                    if COLORBAR is not None :
                        if medianDotStar:
                            ax.scatter(
                                np.nanmedian(x_plot), np.nanmedian(y_plot),
                                marker='*', edgecolor='black', c =colors(name), 
                                s=30*msizetstar , lw=1.1, zorder = 20, alpha = 1.
                            )

                        else:
                            if COLORBAR[0] == 'last_look_BH':
                                ax.scatter(
                                    np.nanmedian(x_plot), np.nanmedian(y_plot),
                                    marker=markers(name+'Colorbar'), edgecolor='red', c = colors(name), 
                                    s=2*msizet * msize(name+'Colorbar'), lw=1.7, zorder = 20, alpha = 1.
                                )
                            else:
                                ax.scatter(
                                    np.nanmedian(x_plot), np.nanmedian(y_plot),
                                    marker=markers(name+'Colorbar'), edgecolor='red', facecolor = 'none', # c =colors(name), 
                                    s=1.5*msizet * msize(name+'Colorbar'), lw=1.7, zorder = 20, alpha = 1.
                                )
                    else:
                        ax.scatter(
                            np.nanmedian(x_plot), np.nanmedian(y_plot),
                            marker='*', edgecolor='black', c =colors(name), 
                            s=33*msizet , lw=1.3, zorder = 20, alpha = 1.
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
                if CAllSMT is not None and cvals is not None:
                    corr, pval = spearmanr(XAllSMT[cond], YAllSMT[cond])
                    print("Panel Spearman X and Y corr:", corr, "p:", pval)
                    cond = (
                        (~np.isnan(XAllSMT)) & (~np.isinf(XAllSMT))
                        & (~np.isnan(CAllSMT)) & (~np.isinf(CAllSMT))
                    )
                    corr, pval = spearmanr(XAllSMT[cond], CAllSMT[cond])
                    print("Panel Spearman X and Colorbar corr:", corr, "p:", pval)
                    cond = (
                        (~np.isnan(CAllSMT)) & (~np.isinf(CAllSMT))
                        & (~np.isnan(YAllSMT)) & (~np.isinf(YAllSMT))
                    )
                    corr, pval = spearmanr(CAllSMT[cond], YAllSMT[cond])
                    print("Panel Spearman Colorbar and Y corr:", corr, "p:", pval)
                    print('\n')
            # Equal line if requested
            if EqualLine and (EqualLineMin is not None) and (EqualLineMax is not None):
                xx = np.linspace(EqualLineMin, EqualLineMax)
                ax.plot(xx, xx, ls="--", color="tab:blue", linewidth=linewidth)

            # Panel formatting
            if yscales != None:
                yscale = yscales[i]
            else:
                yscale = None
            _apply_post_panel_formatting(ax, yparam, yscale = yscale)
            
            if ylimmin is not None and ylimmax is not None:
                ax.set_ylim(ylimmin[i], ylimmax[i])

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

            if j == len(panel_columns) - 1:
                if xlabelintext:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        texts.get(yparam), loc='upper right', prop=Afont)
                    ax.add_artist(anchored_text)
                    
            # X label on last row
            if i == len(ParamsY) - 1:
                if len(ParamsX) > 1 and label_general:
                    unique_count = len(set(ParamsX))
                    if unique_count == 1:
                        ax.set_xlabel(labelsequal.get(ParamsX[0], ParamsX[0]), fontsize=1.2 * fontlabel)
                        ax.set_xscale(scales(ParamsX[0]))
                        if scales(ParamsX[0]) in ("log", "symlog") and not ParamsX[0] == 'z_Birth':
                            ax.xaxis.set_major_formatter(FuncFormatter(format_func_loglog))
                    else:
                        ax.set_xlabel(labelsequal.get(ParamsX[j], ParamsX[j]), fontsize=1.2 * fontlabel)
                        if  ParamsX[j] == 'z_Birth':
                            ax.set_xscale('symlog',linthresh=0.05)
                            ax.set_xticks([ 0, 0.01, 0.1, 1, 10])
                            ax.set_xticklabels(["0", "$10^{-2}$", "0.1", "1", "10"])
                            
                        else:
                            ax.set_xscale(scales(ParamsX[j]))
                        if scales(ParamsX[j]) in ("log", "symlog") and not ParamsX[j] == 'z_Birth':
                            ax.xaxis.set_major_formatter(FuncFormatter(format_func_loglog))
                
                
                else:
                   
                    ax.set_xlabel(labels.get(ParamsX[0], ParamsX[0]), fontsize=1.2 * fontlabel)
                    if xscale != None:
                        ax.set_xscale(xscale)
                    else:
                        ax.set_xscale(scales(ParamsX[0]))
                        
                   
                    if len(ParamsX) > 1 and ParamsX[j] == 'z_Birth':
                        ax.set_xscale('symlog',linthresh=0.02)
                        ax.set_xticks([ 0, 0.01, 0.1, 1, 10])
                        ax.set_xticklabels(["0", "$10^{-2}$", "0.1", "1", "10"])
 
                    if  ParamsX == 'z_Birth':

                        ax.set_xscale('symlog',linthresh=0.02)
                        ax.set_xticks([ 0, 0.01, 0.1, 1, 10])
                        ax.set_xticklabels(["0", "$10^{-2}$", "0.1", "1", "10"])
                     
                    if len(ParamsX) > 1 and xscale == None and "log" in scales(ParamsX[0]) and not ParamsX[j] == 'z_Birth':
                        ax.xaxis.set_major_formatter(FuncFormatter(format_func_loglog))
                        
                    elif xscale == None and "log" in scales(ParamsX[0]):
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
                            handler_map={Circle: HandlerCircle()},
                        )

    # Global colorbar (preserve your original block here)
    _add_colorbar(fig, axs, sc_for_colorbar)

    savefig(savepath, savefigname, TRANSPARENT)
    return

def PlotScatterJoint(
    # --- Data / what to plot ---
    names,
    columns,
    ParamX,
    ParamsY,
    Type="z0",
    snap=(99,),
    dfName="PaperIII",
    SampleName="Samples",
    Name="Name",

    # --- Optional direct DataFrame mode ---
    DataFrame=None,
    columnFilter=None,

    # --- Marginal histograms ---
    hist=True,
    histDensity=True,
    sameHistBins=True,
    binsHist=22,

    # --- 2D density background ---
    density_background=False,
    densityMode="class",      # "class" or "all"
    density_bins=28,
    density_levels=(0.25, 0.50, 0.75),
    density_alpha=0.25,
    density_cmap="Greys",

    # --- Statistics ---
    medianBins=False,
    medianDot=False,
    bins=8,
    quantile=0.68,
    min_points_per_bin=5,

    # --- Extra guide lines ---
    guide_hlines=None,
    guide_vlines=None,
    EqualLine=False,

    # --- Layout ---
    figsize=(8.0, 7.2),
    xscale=None,
    yscales=None,
    GridMake=False,
    title=None,

    # --- Limits ---
    xlim=None,
    ylim=None,
    clipToLimits=True,

    # --- Cleaning ---
    removeNaN=True,
    removeInf=True,
    positiveForLog=True,

    # --- Style ---
    msizet=15, 
    alphaScater=1,
    linewidth=1.2,
    fontlabel=22,
    framealpha=0.95,

    # --- Legend ---
    legend=True,
    loc="best",

    # --- IO ---
    save=True,
    savepath="fig/PlotScatterJoint",
    savefigname="fig",
    TRANSPARENT=False,
    dpi=300,

    # --- Return ---
    returnData=False,

    # --- Reproducibility ---
    seed=16010504,
):
    """
    Scatter plot with marginal density histograms.

    This function is designed as a more specific companion to PlotScatter:
    it keeps the same input logic based on names, columns, ParamX, ParamsY,
    and TNG.makedata, but builds one joint scatter + marginal histogram
    figure for each Y parameter and each column.

    Main use cases:
    - Compare Normal vs Diffuse with very different sample sizes.
    - Use density-normalized marginal histograms.
    - Force identical histogram bins for all classes.
    - Add 2D density contours/background.
    - Preserve the same color/marker/label conventions used elsewhere.

    Returns
    -------
    results : dict
        Dictionary keyed by (yparam, column), containing:
        fig, axes, data, stats.
    """

    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Small robust helpers
    # ------------------------------------------------------------------
    def _as_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        return [x]

    def _normalize_inputs(ParamX, ParamsY):
        Ys = _as_list(ParamsY)

        if isinstance(ParamX, (list, tuple, np.ndarray)):
            Xs = list(ParamX)
            if len(Xs) == 1:
                Xs = Xs * len(Ys)
            elif len(Xs) != len(Ys):
                raise ValueError(
                    "ParamX must be scalar, length 1, or have the same length as ParamsY."
                )
        else:
            Xs = [ParamX] * len(Ys)

        return Xs, Ys

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_data_from_TNG(names, columns, ParamsX, ParamsY):
        """
        Uses the same expected TNG.makedata logic as PlotScatter.

        Expected output indexing:
            dataX[i][j][l]
            dataY[i][j][l]

        where:
            i = y parameter index
            j = column/panel index
            l = class/name index
        """

        cols = _as_list(columns)

        if cols == ["Snap"]:
            panel_columns = list(snap)
            dataX = TNG.makedata(
                names, panel_columns, ParamsX, "Snap",
                snap=snap,
                SampleName=SampleName,
                dfName=dfName,
                Name=Name,
            )
            dataY = TNG.makedata(
                names, panel_columns, ParamsY, "Snap",
                snap=snap,
                SampleName=SampleName,
                dfName=dfName,
                Name=Name,
            )
        else:
            panel_columns = cols
            dataX = TNG.makedata(
                names, panel_columns, ParamsX, Type,
                snap=snap,
                SampleName=SampleName,
                dfName=dfName,
                Name=Name,
            )
            dataY = TNG.makedata(
                names, panel_columns, ParamsY, Type,
                snap=snap,
                SampleName=SampleName,
                dfName=dfName,
                Name=Name,
            )

        return panel_columns, dataX, dataY

    def _load_data_from_dataframe(df, names, columns, xparam, yparam):
        """
        Optional fallback mode if you already have a DataFrame.
        This assumes:
            - class column is given by Name
            - xparam and yparam are columns in df
            - if columnFilter is provided, it filters the sample/environment.
        """

        out = []

        for cname in names:
            d = df.copy()

            if Name in d.columns:
                d = d[d[Name] == cname]

            if columnFilter is not None:
                for key, val in columnFilter.items():
                    if isinstance(val, (list, tuple, np.ndarray)):
                        d = d[d[key].isin(val)]
                    else:
                        d = d[d[key] == val]

            x = d[xparam].values
            y = d[yparam].values
            out.append((cname, x, y))

        return out

    # ------------------------------------------------------------------
    # Cleaning and binning
    # ------------------------------------------------------------------
    def _clean_xy(x, y, xscale_this="linear", yscale_this="linear"):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        good = np.ones(len(x), dtype=bool)

        if removeNaN:
            good &= ~np.isnan(x)
            good &= ~np.isnan(y)

        if removeInf:
            good &= np.isfinite(x)
            good &= np.isfinite(y)

        if positiveForLog:
            if xscale_this == "log":
                good &= x > 0
            if yscale_this == "log":
                good &= y > 0

        if xlim is not None and clipToLimits:
            good &= (x >= xlim[0]) & (x <= xlim[1])

        if ylim is not None and clipToLimits:
            good &= (y >= ylim[0]) & (y <= ylim[1])

        return x[good], y[good]

    def _auto_limits(values, user_lim=None, scale="linear", pad_frac=0.05):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if scale == "log":
            values = values[values > 0]

        if user_lim is not None:
            return user_lim

        if len(values) == 0:
            return None

        vmin = np.nanmin(values)
        vmax = np.nanmax(values)

        if vmin == vmax:
            if scale == "log":
                return (vmin / 1.5, vmax * 1.5)
            return (vmin - 0.5, vmax + 0.5)

        if scale == "log":
            logmin = np.log10(vmin)
            logmax = np.log10(vmax)
            pad = pad_frac * (logmax - logmin)
            return (10 ** (logmin - pad), 10 ** (logmax + pad))

        pad = pad_frac * (vmax - vmin)
        return (vmin - pad, vmax + pad)

    def _make_bins(values, bins, lim, scale="linear"):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if lim is None:
            lim = _auto_limits(values, None, scale)

        if lim is None:
            return np.linspace(0, 1, bins + 1)

        lo, hi = lim

        if scale == "log":
            lo = max(lo, np.nanmin(values[values > 0]))
            return np.logspace(np.log10(lo), np.log10(hi), bins + 1)

        return np.linspace(lo, hi, bins + 1)

    def _split_quantiles(x, y, edges, q=0.68):
        x = np.asarray(x)
        y = np.asarray(y)

        xmid, ymed, ylo, yhi = [], [], [], []

        qlo = 50.0 * (1.0 - q)
        qhi = 100.0 - qlo

        for k in range(len(edges) - 1):
            lo, hi = edges[k], edges[k + 1]
            cond = (x >= lo) & (x < hi)

            if np.sum(cond) < min_points_per_bin:
                continue

            yy = y[cond]
            xmid.append(0.5 * (lo + hi))
            ymed.append(np.nanmedian(yy))
            ylo.append(np.nanpercentile(yy, qlo))
            yhi.append(np.nanpercentile(yy, qhi))

        return np.asarray(xmid), np.asarray(ymed), np.asarray(ylo), np.asarray(yhi)

    def _make_stats(class_data):
        rows = []

        for cname, x, y in class_data:
            if len(x) == 0:
                rows.append({
                    "Name": cname,
                    "N": 0,
                    "median_x": np.nan,
                    "median_y": np.nan,
                    "p16_y": np.nan,
                    "p84_y": np.nan,
                })
                continue

            rows.append({
                "Name": cname,
                "N": len(x),
                "median_x": np.nanmedian(x),
                "median_y": np.nanmedian(y),
                "p16_y": np.nanpercentile(y, 16),
                "p84_y": np.nanpercentile(y, 84),
            })

        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except Exception:
            return rows

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _add_density(ax, class_data, xedges, yedges):
        if not density_background:
            return

        if densityMode == "all":
            allx = np.concatenate([d[1] for d in class_data if len(d[1]) > 0])
            ally = np.concatenate([d[2] for d in class_data if len(d[2]) > 0])

            if len(allx) < 5:
                return

            H, xe, ye = np.histogram2d(allx, ally, bins=[xedges, yedges])
            H = H.T.astype(float)

            if np.nanmax(H) <= 0:
                return

            H = H / np.nanmax(H)
            H[H <= 0] = np.nan

            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            X, Y = np.meshgrid(xc, yc)

            levels = np.array(density_levels)
            levels = levels[(levels > 0) & (levels < 1)]

            if len(levels) > 0:
                ax.contourf(
                    X, Y, H,
                    levels=np.r_[levels, 1.0],
                    cmap=density_cmap,
                    alpha=density_alpha,
                    zorder=0,
                )

        elif densityMode == "class":
            for cname, x, y in class_data:
                if len(x) < 8:
                    continue

                H, xe, ye = np.histogram2d(x, y, bins=[xedges, yedges])
                H = H.T.astype(float)

                if np.nanmax(H) <= 0:
                    continue

                H = H / np.nanmax(H)
                H[H <= 0] = np.nan

                xc = 0.5 * (xe[:-1] + xe[1:])
                yc = 0.5 * (ye[:-1] + ye[1:])
                X, Y = np.meshgrid(xc, yc)

                levels = np.array(density_levels)
                levels = levels[(levels > 0) & (levels < 1)]

                if len(levels) > 0:
                    ax.contour(
                        X, Y, H,
                        levels=levels,
                        colors=colors(cname),
                        linewidths=linewidth,
                        alpha=0.75,
                        zorder=2,
                    )

    def _plot_one_joint(xparam, yparam, col_label, class_data, i_y):
        xscale_this = scales(xparam)
        yscale_this = scales(yparam)

        allx = np.concatenate([d[1] for d in class_data if len(d[1]) > 0])
        ally = np.concatenate([d[2] for d in class_data if len(d[2]) > 0])

        local_xlim = _auto_limits(allx, xlim, xscale_this)
        local_ylim = _auto_limits(ally, ylim, yscale_this)

        if sameHistBins:
            xedges_hist = _make_bins(allx, binsHist, local_xlim, xscale_this)
            yedges_hist = _make_bins(ally, binsHist, local_ylim, yscale_this)
        else:
            xedges_hist = None
            yedges_hist = None

        xedges_density = _make_bins(allx, density_bins, local_xlim, xscale_this)
        yedges_density = _make_bins(ally, density_bins, local_ylim, yscale_this)

        fig = plt.figure(figsize=figsize)

        gs = GridSpec(
            2, 2,
            width_ratios=[4.0, 1.15],
            height_ratios=[1.15, 4.0],
            hspace=0.0,
            wspace=0.0,
        )

        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        ax_empty = fig.add_subplot(gs[0, 1])
        ax_empty.axis("off")

        axes = {
            "main": ax,
            "histx": ax_histx,
            "histy": ax_histy,
        }

        # Density behind points
        _add_density(ax, class_data, xedges_density, yedges_density)

        # Scatter and histograms
        for cname, x, y in class_data:
            if len(x) == 0:
                continue
            c = colors(cname)
            mk = markers(cname)
            
            if cname != 'SubDiffuse':

                ax.scatter(
                    x,
                    y,
                    color=c,
                    edgecolor=edgecolors(cname),
                    marker=mk,
                    s=msizet * msize(cname),
                    alpha=alphaScater,
                    lw=0.8,
                    label=cname,
                    zorder=3,
                )

            if hist:
                if sameHistBins:
                    xb = xedges_hist
                    yb = yedges_hist
                else:
                    xb = _make_bins(x, binsHist, local_xlim, xscale_this)
                    yb = _make_bins(y, binsHist, local_ylim, yscale_this)
            
                ax_histx.hist(
                    x,
                    bins=xb,
                    histtype="step",
                    density=histDensity,
                    color=c,
                    lw=linewidth,
                    ls=lines(cname),
                )
            
                ax_histy.hist(
                    y,
                    bins=yb,
                    histtype="step",
                    density=histDensity,
                    color=c,
                    lw=linewidth,
                    ls=lines(cname),
                    orientation="horizontal",
                )
            
                # Median lines in the marginal histograms
                if len(x) > 0:
                    ax_histx.axvline(
                        np.nanmedian(x),
                        color=c,
                        lw=linewidth,
                        ls="--",
                        alpha=0.9,
                        zorder=10,
                    )
            
                if len(y) > 0:
                    ax_histy.axhline(
                        np.nanmedian(y),
                        color=c,
                        lw=linewidth,
                        ls="--",
                        alpha=0.9,
                        zorder=10,
                    )
            if medianDot:
                ax.scatter(
                    np.nanmedian(x),
                    np.nanmedian(y),
                    marker="*",
                    s=5.0 * msizet,
                    color=c,
                    edgecolor="black",
                    lw=0.8,
                    zorder=10,
                )

            if medianBins:
                xmedges = _make_bins(allx, bins, local_xlim, xscale_this)
                xm, ym, ylo, yhi = _split_quantiles(
                    x, y, xmedges, q=quantile
                )

                if len(xm) > 0:
                    ax.plot(
                        xm,
                        ym,
                        color=c,
                        lw=linewidth,
                        marker=mk,
                        ms=6,
                        zorder=8,
                    )

                    ax.fill_between(
                        xm,
                        ylo,
                        yhi,
                        color=c,
                        alpha=0.18,
                        zorder=7,
                    )

        # Guide lines
        if guide_hlines is not None:
            for yy in _as_list(guide_hlines):
                ax.axhline(
                    yy,
                    color="black",
                    ls="--" if yy == 1.0 else ":",
                    lw=1.1,
                    alpha=0.75,
                    zorder=1,
                )

        if guide_vlines is not None:
            for xx in _as_list(guide_vlines):
                ax.axvline(
                    xx,
                    color="black",
                    ls=":",
                    lw=1.1,
                    alpha=0.75,
                    zorder=1,
                )

        if EqualLine:
            lo = max(local_xlim[0], local_ylim[0])
            hi = min(local_xlim[1], local_ylim[1])
            xx = np.linspace(lo, hi, 100)
            ax.plot(xx, xx, color="black", ls="--", lw=1.0, zorder=1)

        # Axis formatting
        ax.set_xlim(local_xlim)
        ax.set_ylim(local_ylim)

        ax.set_xscale(xscale_this)
        ax.set_yscale(yscale_this)

        ax_histx.set_xscale(xscale_this)
        ax_histy.set_yscale(yscale_this)

        if xscale_this in ["log", "symlog"]:
            ax.xaxis.set_major_formatter(FuncFormatter(format_func_loglog))


        if yscale_this in ["log", "symlog"]:
            ax.yaxis.set_major_formatter(FuncFormatter(format_func_loglog))

        ax.set_xlabel(labels.get(xparam, xparam), fontsize=fontlabel)
        ax.set_ylabel(labels.get(yparam, yparam), fontsize=fontlabel)

        if histDensity:
            ax_histx.set_ylabel("density", fontsize=0.75 * fontlabel)
            ax_histy.set_xlabel("density", fontsize=0.75 * fontlabel)
        else:
            ax_histx.set_ylabel("N", fontsize=0.75 * fontlabel)
            ax_histy.set_xlabel("N", fontsize=0.75 * fontlabel)

        ax.tick_params(axis="both", labelsize=0.85 * fontlabel)
        ax_histx.tick_params(axis="y", labelsize=0.70 * fontlabel)
        ax_histy.tick_params(axis="x", labelsize=0.70 * fontlabel)

        plt.setp(ax_histx.get_xticklabels(), visible=False)
        plt.setp(ax_histy.get_yticklabels(), visible=False)

        if GridMake:
            ax.grid(
                True,
                color="#9e9e9e",
                which="major",
                linewidth=0.6,
                alpha=0.3,
                linestyle=":",
            )

        if legend:
            ax.legend(
                loc=loc,
                fontsize=0.78 * fontlabel,
                framealpha=framealpha,
            )

        if title:
            if col_label is not None:
                ax_histx.set_title(
                    title,
                    fontsize=0.88 * fontlabel,
                )
            else:
                ax_histx.set_title(
                   title,
                    fontsize=0.88 * fontlabel,
                )

        # Cleaner marginal axes
        for marginal_ax in (ax_histx, ax_histy):
            for spine in marginal_ax.spines.values():
                spine.set_visible(True)
                spine.set_linestyle("-")
                spine.set_edgecolor("black")
                spine.set_linewidth(
                    ax.spines["left"].get_linewidth()
                )

        stats = _make_stats(class_data)

        return fig, axes, stats, {
            "xparam": xparam,
            "yparam": yparam,
            "column": col_label,
            "class_data": class_data,
        }

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    names = _as_list(names)
    columns = _as_list(columns)
    ParamsX, ParamsY = _normalize_inputs(ParamX, ParamsY)

    results = {}

    if DataFrame is None:
        panel_columns, dataX, dataY = _load_data_from_TNG(
            names, columns, ParamsX, ParamsY
        )

        for i, yparam in enumerate(ParamsY):
            xparam = ParamsX[i]
            xscale_this = scales(xparam)
            yscale_this = scales(yparam)

            for j, col_label in enumerate(panel_columns):
                class_data = []

                for l, cname in enumerate(names):
                    x = np.asarray(dataX[i][j][l], dtype=float)
                    y = np.asarray(dataY[i][j][l], dtype=float)

                    x, y = _clean_xy(
                        x,
                        y,
                        xscale_this=xscale_this,
                        yscale_this=yscale_this,
                    )

                    class_data.append((cname, x, y))

                fig, axes, stats, data_out = _plot_one_joint(
                    xparam=xparam,
                    yparam=yparam,
                    col_label=col_label,
                    class_data=class_data,
                    i_y=i,
                )

                key = (yparam, col_label)
                results[key] = {
                    "fig": fig,
                    "axes": axes,
                    "stats": stats,
                    "data": data_out,
                }

                if save:
                    os.makedirs(savepath, exist_ok=True)

                    if len(ParamsY) == 1 and len(panel_columns) == 1:
                        fname = f"{savefigname}.pdf"
                    else:
                        fname = (
                            f"{savefigname}.pdf"
                        )

                    fig.savefig(
                        os.path.join(MAIN_SAVE_FIG+savepath, fname),
                        bbox_inches="tight",
                        dpi=dpi,
                        transparent=TRANSPARENT,
                    )

    else:
        # DataFrame mode: useful if you already have PaperIII directly.
        # In this mode, columns is only used for naming unless columnFilter is used.
        for i, yparam in enumerate(ParamsY):
            xparam = ParamsX[i]
            xscale_this = scales(xparam)
            yscale_this = scales(yparam)

            class_data_raw = _load_data_from_dataframe(
                DataFrame,
                names,
                columns,
                xparam,
                yparam,
            )

            class_data = []
            for cname, x, y in class_data_raw:
                x, y = _clean_xy(
                    x,
                    y,
                    xscale_this=xscale_this,
                    yscale_this=yscale_this,
                )
                class_data.append((cname, x, y))

            col_label = columns[0] if len(columns) > 0 else None

            fig, axes, stats, data_out = _plot_one_joint(
                xparam=xparam,
                yparam=yparam,
                col_label=col_label,
                class_data=class_data,
                i_y=i,
            )

            key = (yparam, col_label)
            results[key] = {
                "fig": fig,
                "axes": axes,
                "stats": stats,
                "data": data_out,
            }

            savefig(savepath, fname, TRANSPARENT)

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
                                        values[values < -14] = np.nan

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
                            Y_plot[Y_plot < -14] = np.nan

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

                            axs[i][j].set_xlabel(r"$\phi_\mathrm{orbital}$", fontsize=fontlabel)
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

def PlotScatterColumn(
    # --- Data / what to plot ---
    names, columns, ParamX, ParamsY, Type="z0", snap=(99,), ColumnPlot=True,
    dfName="Sample", SampleName="Samples", Name="Name",

    # --- Extra layers ---
    All=None, COLORBAR=None, MarkerSizes=None, NoneEdgeColor=False,
    BackGroudnDensity=None,

    # --- Statistics ---
    medianBins=False, medianAll=False, medianDotStar=False,
    medianDot=False, SpearManTest=False, SpearManTestAll=False,
    bins=10, quantile=0.95, q=0.95, HIGHLIGHTPoints=False,

    # --- Layout ---
    lNum=6, cNum=6, xscale=None, yscales=None, GridMake=False,
    InvertPlot=False, xlabelintext=False, title=False,

    # --- Helper lines ---
    EqualLine=False, EqualLineMin=None, EqualLineMax=None,

    # --- Limits ---
    xlimmin=None, xlimmax=None, ylimmin=None, ylimmax=None,
    xlims=None, ylims=None,

    # --- Style ---
    cmap="inferno", m="o", msizet=30, msizetstar=30, msizeMult=1,
    alphaScater=1.0, alphaShade=0.3, linewidth=1.2, fontlabel=26,
    framealpha=0.95,

    # --- Legend ---
    legend=False, LegendNames=None, legpositions=None, loc="best",
    columnspacing=0.5, handlelength=2, handletextpad=-0.5,
    labelspacing=0.3,

    # --- Colorbar ---
    ratioColorbar=None, mult=4.1,

    # --- IO ---
    savepath="fig/PlotScatter", savefigname="fig", TRANSPARENT=False,

    # --- Reproducibility ---
    seed=16010504,
):
    """
    Scatter-plot grid for X--Y relations across samples/snapshots and
    multiple Y parameters.

    Panel orientation
    -----------------
    ColumnPlot=True (original orientation):
        rows    -> ParamsY
        columns -> columns / snapshots

    ColumnPlot=False (transposed orientation):
        rows    -> columns / snapshots
        columns -> ParamsY

    Notes
    -----
    The arrays returned by ``TNG.makedata`` are not transposed. They remain
    indexed as::

        data[y_parameter_index][column_index][population_index]

    Only the mapping between data indices and visual panel indices changes.

    ``BackGroudnDensity`` accepts either one pair ``[x_background,
    y_background]`` reused for every Y parameter, or one pair per Y
    parameter. The spelling is intentionally preserved for compatibility
    with existing calls.

    ``xlims`` and ``ylims`` may be one ``[min, max]`` pair or one pair per
    entry of ``columns`` / snapshot panel.

    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    """

    np.random.seed(seed)

    # ------------------------------------------------------------------
    # Helpers: normalization / indexing
    # ------------------------------------------------------------------
    def _as_list(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(value)
        return [value]

    def _normalize_inputs(columns_, ParamX_, ParamsY_):
        cols = _as_list(columns_)
        ys = _as_list(ParamsY_)

        if isinstance(ParamX_, (list, tuple, np.ndarray)):
            xs = list(ParamX_)
            label_general_ = True
        else:
            xs = [ParamX_] * len(ys)
            label_general_ = False

        if len(xs) != len(ys):
            raise ValueError(
                "ParamX must be a scalar or have the same length as ParamsY."
            )

        return cols, xs, ys, label_general_

    def _value_for_y(value, y_idx, name_="value"):
        """Return a scalar option or the option corresponding to ParamsY[y_idx]."""
        if value is None:
            return None

        if isinstance(value, str) or np.isscalar(value):
            return value

        values = list(value)
        if len(values) == 0:
            return None
        if len(values) == 1:
            return values[0]
        if y_idx >= len(values):
            raise IndexError(
                f"{name_} has {len(values)} values, but ParamsY requires "
                f"an entry at index {y_idx}."
            )
        return values[y_idx]

    def _value_for_index(value, idx, default=None):
        """Read a scalar or sequence option without indexing strings by character."""
        if value is None:
            return default
        if isinstance(value, str) or np.isscalar(value):
            return value

        values = list(value)
        if len(values) == 0:
            return default
        if len(values) == 1:
            return values[0]
        if idx < len(values):
            return values[idx]
        return default

    def _panel_indices(panel_i, panel_j):
        """Map a visual panel position to data indices."""
        if ColumnPlot:
            y_idx_ = panel_i
            column_idx_ = panel_j
        else:
            y_idx_ = panel_j
            column_idx_ = panel_i
        return y_idx_, column_idx_

    def _format_with(function_, key):
        """Use a project formatter when possible, otherwise return the raw key."""
        try:
            return function_(key)
        except Exception:
            return str(key)

    def _label_from(mapping, key):
        try:
            return mapping.get(key, key)
        except Exception:
            return str(key)

    def _single_value_equals(value, expected):
        if value is None:
            return False
        if isinstance(value, str) or np.isscalar(value):
            return value == expected
        values = list(value)
        return len(values) == 1 and values[0] == expected

    def _normalize_background_density(value, n_yparams):
        """
        Normalize the optional density variables.

        Accepted forms
        --------------
        [x_background, y_background]
            One pair reused for every Y parameter.

        [[x_background_0, y_background_0], ...]
            One pair for each entry of ParamsY.
        """
        if value is None:
            return None, None

        values = list(value)

        if (
            len(values) == 2
            and all(isinstance(item, str) for item in values)
        ):
            return (
                [values[0]] * n_yparams,
                [values[1]] * n_yparams,
            )

        if (
            len(values) == n_yparams
            and all(
                isinstance(item, (list, tuple, np.ndarray))
                and len(item) == 2
                for item in values
            )
        ):
            return (
                [item[0] for item in values],
                [item[1] for item in values],
            )

        raise ValueError(
            "BackGroudnDensity must be [x_background, y_background] "
            "or one [x_background, y_background] pair per ParamsY entry."
        )

    def _panel_limit_pair(value, column_idx, n_columns, name_):
        """
        Return one [min, max] pair for the current sample/snapshot panel.

        A single pair is reused for every panel. A list of pairs is indexed
        by the original sample/snapshot dimension, independently of whether
        the visual layout is transposed.
        """
        if value is None:
            return None

        values = list(value)

        if (
            len(values) == 2
            and all(np.isscalar(item) for item in values)
        ):
            pair = values
        elif (
            len(values) == n_columns
            and all(
                isinstance(item, (list, tuple, np.ndarray))
                and len(item) == 2
                for item in values
            )
        ):
            pair = values[column_idx]
        else:
            raise ValueError(
                f"{name_} must be [min, max] or contain one [min, max] "
                f"pair for each of the {n_columns} sample/snapshot panels."
            )

        lower, upper = float(pair[0]), float(pair[1])
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError(f"{name_} limits must be finite.")
        if lower >= upper:
            raise ValueError(
                f"{name_} lower limit must be smaller than its upper limit."
            )

        return lower, upper

    # ------------------------------------------------------------------
    # Helpers: data loading
    # ------------------------------------------------------------------
    def _load_data(names_, columns_, ParamsX_, ParamsY_):
        """
        Load the arrays while preserving the original TNG.makedata layout.

        Returns
        -------
        data_columns, dataX, dataY, dataColor, dataMarker
        """
        if columns_ == ["Snap"]:
            data_columns = list(snap)

            dataX_ = TNG.makedata(
                names_, data_columns, ParamsX_, "Snap",
                snap=snap, SampleName=SampleName, dfName=dfName, Name=Name,
            )
            dataY_ = TNG.makedata(
                names_, data_columns, ParamsY_, "Snap",
                snap=snap, SampleName=SampleName, dfName=dfName, Name=Name,
            )

            dataColor_ = None
            dataMarker_ = None

            if COLORBAR is not None:
                dataColor_ = TNG.makedata(
                    names_, data_columns, COLORBAR, "Snap",
                    snap=snap, SampleName=SampleName, dfName=dfName, Name=Name,
                )

            if MarkerSizes is not None:
                dataMarker_ = TNG.makedata(
                    names_, data_columns, MarkerSizes, "Snap",
                    snap=snap, SampleName=SampleName, dfName=dfName, Name=Name,
                )

            return data_columns, dataX_, dataY_, dataColor_, dataMarker_

        data_columns = list(columns_)

        dataX_ = TNG.makedata(
            names_, data_columns, ParamsX_, Type,
            snap=snap, SampleName=SampleName, dfName=dfName, Name=Name,
        )
        dataY_ = TNG.makedata(
            names_, data_columns, ParamsY_, Type,
            snap=snap, SampleName=SampleName, dfName=dfName, Name=Name,
        )

        dataColor_ = None
        dataMarker_ = None

        if MarkerSizes is not None:
            dataMarker_ = TNG.makedata(
                names_, data_columns, MarkerSizes, Type,
                snap=snap, SampleName=SampleName, dfName=dfName, Name=Name,
            )

        if COLORBAR is not None:
            dataColor_ = TNG.makedata(
                names_, data_columns, COLORBAR, Type,
                snap=snap, SampleName=SampleName, dfName=dfName, Name=Name,
            )

        return data_columns, dataX_, dataY_, dataColor_, dataMarker_

    def _load_background_density(
        names_, data_columns_, background_x_, background_y_
    ):
        """Load optional X/Y arrays used only by the KDE background."""
        if background_x_ is None or background_y_ is None:
            return None, None

        if columns == ["Snap"]:
            dataBackgroundX_ = TNG.makedata(
                names_, data_columns_, background_x_, "Snap",
                snap=snap, SampleName=SampleName, dfName=dfName, Name=Name,
            )
            dataBackgroundY_ = TNG.makedata(
                names_, data_columns_, background_y_, "Snap",
                snap=snap, SampleName=SampleName, dfName=dfName, Name=Name,
            )
        else:
            dataBackgroundX_ = TNG.makedata(
                names_, data_columns_, background_x_, Type,
                snap=snap, SampleName=SampleName, dfName=dfName, Name=Name,
            )
            dataBackgroundY_ = TNG.makedata(
                names_, data_columns_, background_y_, Type,
                snap=snap, SampleName=SampleName, dfName=dfName, Name=Name,
            )

        return dataBackgroundX_, dataBackgroundY_

    def _auxiliary_panel(data, y_idx, column_idx, auxiliary_name):
        """
        Select one COLORBAR/MarkerSizes panel.

        A single auxiliary parameter is reused for all ParamsY. When the
        auxiliary array has one parameter per ParamsY, y_idx is used.
        """
        if data is None:
            return None

        n_auxiliary_parameters = len(data)
        if n_auxiliary_parameters == 0:
            raise ValueError(f"{auxiliary_name} returned an empty data array.")

        auxiliary_idx = y_idx if n_auxiliary_parameters == len(ParamsY) else 0
        return data[auxiliary_idx][column_idx]

    # ------------------------------------------------------------------
    # Helpers: axes and labels
    # ------------------------------------------------------------------
    def _setup_axes(nrows, ncols):
        plt.rcParams.update({
            "figure.figsize": (cNum * ncols, lNum * nrows),
        })

        fig_ = plt.figure()
        gs = fig_.add_gridspec(nrows, ncols, hspace=0, wspace=0)

        # In the original orientation, each row contains one Y parameter,
        # so sharing Y within the row is appropriate. In the transposed
        # orientation, keep the Y axes independent so that every physical row
        # preserves its tick labels and ylabel, as in vertically stacked
        # panels such as Centrals / Sats loEnv / Sats hiEnv.
        share_y = "row" if ColumnPlot else False

        # When ParamX differs between ParamsY, sharing x in the original
        # orientation would incorrectly force distinct quantities to use the
        # same scale. In the transposed orientation each visual column has one
        # ParamsX, so sharing along columns is appropriate.
        if ColumnPlot and len(set(ParamsX)) > 1:
            share_x = False
        elif not ColumnPlot and xlims is not None:
            # In the transposed layout, each original sample/snapshot panel
            # occupies a different visual row. Explicit per-panel x limits
            # therefore cannot share one X axis down the visual column.
            share_x = False
        else:
            share_x = "col"

        axs_ = gs.subplots(
            sharex=share_x,
            sharey=share_y,
            squeeze=False,
        )
        return fig_, axs_

    def _panel_column_label(column_idx):
        """Label associated with the sample/snapshot dimension."""
        if panel_columns[column_idx] == "Snap":
            snap_value = data_columns[column_idx]
            values = dfTime.z.loc[dfTime.Snap == snap_value].values
            if len(values) > 0:
                return r"$z = %.1f$" % values[0]
            return f"Snap {snap_value}"

        if title:
            # ``title`` may be True (use the column name itself), a scalar
            # title key, or one title key per sample/snapshot.
            if isinstance(title, (bool, np.bool_)):
                title_key = panel_columns[column_idx]
            else:
                title_key = _value_for_index(
                    title,
                    column_idx,
                    panel_columns[column_idx],
                )
            return _format_with(titles, title_key)

        return _format_with(titles, panel_columns[column_idx])

    def _y_parameter_label(yparam):
        if label_general:
            return _label_from(labelsequal, yparam)
        return _label_from(labels, yparam)

    def _x_parameter_label(xparam):
        if label_general:
            return _label_from(labelsequal, xparam)
        return _label_from(labels, xparam)

    def _add_panel_title_text(ax, text_):
        """Place the sample/snapshot title inside the panel, upper left."""
        if text_ is None:
            return

        title_artist = AnchoredText(
            str(text_),
            loc="upper left",
            prop={"color": "black", "size": 0.92 * fontlabel},
            frameon=True,
            pad=0.25,
            borderpad=0.35,
        )
        title_artist.patch.set_facecolor("white")
        title_artist.patch.set_edgecolor("black")
        title_artist.patch.set_linewidth(0.8 * linewidth)
        title_artist.patch.set_alpha(framealpha)
        ax.add_artist(title_artist)

    # ------------------------------------------------------------------
    # Helpers: special plotting rules
    # ------------------------------------------------------------------
    def _apply_special_xaxis_rules(
        ax, ParamX_, yparam, ylimmin_for_y, fontlabel_
    ):
        """Apply the project-specific X/Y tick rules to one panel."""
        if ParamX_ == "DecreaseBeforeGas":
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xticklabels(["", "0.2", "0.4", "0.6", "0.8", "1.0"])

        if ParamX_ == "Decrease_Entry_To_NoGas_Norm_Delta":
            ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
            ax.set_xticklabels(["-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2"])
            
        if ParamX_ == "vrot_InSitu_component":
            ax.set_xticks([1, 5, 10, 20, 50])
            ax.set_xticklabels(["1", "5", "10", "20", "50"])

        if "Snap" in str(ParamX_):
            ax.set_xlim(-0.2, 14.2)
            ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
            ax.set_xticklabels(["0", "2", "4", "6", "8", "10", "12", "14"])

        if "DMFrac_Birth" in str(yparam):
            ax.set_yticks([0.001, 0.01, 0.1, 0.5, 0.9, 0.99])
            ax.set_yticklabels([
                "$10^{-3}$", "$10^{-2}$", "0.1", "0.5", "0.9", "0.99",
            ])

        if ParamX_ == "MassIn_Infall_to_GasLost":
            ax.set_xticks([-0.15, 0, 0.25, 0.5, 0.75])
            ax.set_xticklabels(["-0.15", "0", "0.25", "0.50", "0.75"])

        if (
            "StarFrac" in str(ParamX_)
            and "GasFrac" in str(yparam)
            and ylimmin_for_y != 0.001
        ):
            ax.tick_params(axis="y", labelsize=0.88 * fontlabel_)
            ax.tick_params(axis="x", labelsize=0.88 * fontlabel_)

            ax.set_yticks([0.02, 0.03, 0.04, 0.06, 0.08, 0.1])
            ax.set_yticklabels(["0.02", "0.03", "0.04", "0.06", "0.08", "0.1"])
            ax.set_xticks([0.004, 0.006, 0.01, 0.02, 0.03])
            ax.set_xticklabels(["0.004", "0.006", "0.01", "0.02", "0.03"])

    def _apply_special_background_rules(
        ax, ParamX_, yparam, linewidth_, fontlabel_
    ):
        """Apply guide lines, regions, and annotations to one panel."""
        
       
        if (
            ParamX_ == "MassIn_Infall_to_GasLost"
            and yparam == "MassAboveAfter_Infall_to_GasLost"
        ):
            x = np.linspace(0, 1)
            ax.plot(x, -x, color="darkorange", linestyle="dashed", lw=linewidth_)

            ax.axvline(0, color="black", linestyle="dashed", lw=linewidth_)
            ax.axhline(0, color="black", linestyle="dashed", lw=linewidth_)

            ax.fill_between([0, 500], -500, 0, alpha=0.2, color="tab:green")
            ax.fill_between([-500, 0], -500, 0, alpha=0.2, color="tab:red")
            ax.fill_between([0, 500], 0, 500, alpha=0.2, color="tab:blue")

            ax.text(-0.145, -0.95, "TS", fontsize=0.98 * fontlabel_)
            ax.text(0.1, 0.02, "SF", fontsize=0.98 * fontlabel_)
            ax.text(0.15, -0.95, "Interplay", fontsize=0.98 * fontlabel_)

        elif (
            ParamX_ == "Relative_logInnerZ_At_Entry"
            and yparam == "Relative_logZ_At_Entry"
        ):
            xfitline = np.linspace(0, 1, 100)
            ax.plot(
                xfitline, xfitline, ls="--", color="tab:blue",
                linewidth=linewidth_,
            )
            ax.plot(
                xfitline, np.zeros(100), ls="--", color="k",
                linewidth=linewidth_, zorder=1,
            )

        elif (
            ParamX_ == "Relative_Rhalf_MaxProfile_Minus_HalfRadstar_Entry"
            and yparam == "Relative_Rhalf_MinProfile_Minus_HalfRadstar_Entry"
        ):
            xfitline = np.linspace(-6, 2, 100)
            ax.axvline(
                0, color="black", linestyle="dashed", lw=linewidth_, zorder=1,
            )
            ax.plot(
                xfitline, xfitline, ls="--", color="tab:blue",
                linewidth=linewidth_, zorder=1,
            )
            ax.text(
                -1.15, -1.58,
                "Outer stellar profile \n evolution \n dominates",
                fontsize=0.99 * fontlabel_,
            )
            ax.text(
                -1.8, -0.75,
                "Inner stellar profile \n evolution \n dominates",
                fontsize=0.99 * fontlabel_,
            )

        elif (
            ParamX_ == "sSFRTrueInner_BeforeEntry"
            and yparam == "sSFRTrueInner_Entry_to_Nogas"
        ):
            x = np.linspace(-12, -8)
            ax.plot(x, x, ls="--", color="tab:blue", linewidth=linewidth_)

            ax.text(
                -10, -10.55,
                "Inner $\\overline{\\mathrm{sSFR}}$ \n decrease",
                fontsize=0.99 * fontlabel_,
            )
            ax.text(
                -10.9, -9.5,
                "Inner $\\overline{\\mathrm{sSFR}}$ \n increase",
                fontsize=0.99 * fontlabel_,
            )

        elif (
            ParamX_ == "Decrease_Entry_To_NoGas_Norm_Delta"
            and yparam == "Decrease_NoGas_To_Final_Norm_Delta"
        ):
            ax.text(
                -0.6, -0.8,
                "Faster compaction \n after gas loss",
                fontsize=0.99 * fontlabel_,
            )
            ax.text(
                -0.80, -0.25,
                "Faster compaction  \n with gas ",
                fontsize=0.99 * fontlabel_,
            )
            ax.axvline(0, color="black", linestyle="dashed", lw=linewidth_)
            ax.axhline(0, color="black", linestyle="dashed", lw=linewidth_)
            ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
            ax.set_xticklabels(["-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2"])
            ax.set_yticks([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2])
            ax.set_yticklabels(["-0.8", "-0.6", "-0.4", "-0.2", "0.0", "0.2"])

        elif (
            ParamX_ == "global_color_U_minus_r_1xRh"
            and yparam == "ratio_color_U_minus_r_1xRh"
        ):
            xfitline = np.linspace(-0.75, 2.5, 100)
            ax.fill_between(xfitline, -1.5, 0, alpha=0.1, color="tab:blue")
            ax.axvline(0, color="black", linestyle="dashed", lw=linewidth_)
            ax.axhline(0, color="black", linestyle="dashed", lw=linewidth_)

        elif (
            ParamX_ == "Decrease_Entry_To_NoGas"
            and yparam == "Decrease_NoGas_To_Final"
        ):
            ax.text(
                -0.55, -1.3,
                "Larger compaction \n after gas loss",
                fontsize=0.99 * fontlabel_,
            )
            ax.text(
                -1.2, 0.2,
                "Larger compaction  \n with gas ",
                fontsize=0.99 * fontlabel_,
            )
            ax.axvline(
                0, color="black", linestyle="dashed", lw=linewidth_, zorder=1,
            )
            ax.axhline(
                0, color="black", linestyle="dashed", lw=linewidth_, zorder=1,
            )

        elif (
            ParamX_ == "Rhalf_MaxProfile_Minus_HalfRadstar_Entry"
            and yparam == "Rhalf_MinProfile_Minus_HalfRadstar_Entry"
        ):
            xfitline = np.linspace(-6, 2, 100)
            ax.plot(
                xfitline, xfitline, ls="--", color="tab:blue",
                linewidth=linewidth_,
            )
            ax.fill_between(
                xfitline, -7, xfitline, alpha=0.2, color="tab:blue",
            )
            ax.text(-2, -5, "TS", fontsize=0.99 * fontlabel_)
            ax.fill_between(
                xfitline, xfitline, 1, alpha=0.2, color="tab:red",
            )
            ax.text(-4.0, -1, "SF", fontsize=0.99 * fontlabel_)

        elif "StarFrac" in str(ParamX_) and "GasFrac" in str(yparam):
            x = np.linspace(0, 1)
            ax.plot(
                x, x, ls="--", color="gray", linewidth=linewidth_, zorder=0,
            )
            
        elif "vrot_InSitu_component" in str(ParamX_) and "sigma_InSitu" in str(yparam):
            x = np.linspace(0, 100)
            ax.plot(
                x, x, ls="--", color="gray", linewidth=linewidth_, zorder=0,
            )

        elif "StarFrac" in str(ParamX_) and "DMFrac" in str(yparam):
            x = np.linspace(0, 1)
            ax.plot(
                x, 1 - x, ls="dotted", color="gray",
                linewidth=linewidth_, zorder=0,
            )

        elif "RadEx" in str(ParamX_) and "RadIn" in str(yparam):
            
            x = np.linspace(0, 50)
            ax.plot(
                x, x, ls="dotted", color="gray",
                linewidth=linewidth_, zorder=0,
            )
            
        elif "z_Birth" in str(ParamX_) and "DMFrac_Birth" in str(yparam):
            ax.axhline(0.8, ls="--", color="tab:red", linewidth=linewidth_)

        if ParamX_ == "AgeBorn":
            x = np.arange(14)
            ax.plot(x, x, color="black", linestyle="dashed", lw=2)

    # ------------------------------------------------------------------
    # Helpers: density / scatter / scales / colorbar
    # ------------------------------------------------------------------
    def _density_transform(values, scale_name):
        values = np.asarray(values, dtype=float)

        if scale_name == "log":
            return np.log10(values)

        # For linear and symlog axes, KDE is evaluated in the original
        # coordinates. This preserves all finite values for symlog panels.
        return values

    def _density_inverse(values, scale_name):
        if scale_name == "log":
            return 10.0 ** values
        return values

    def _density_bounds(values, scale_name, explicit_limits=None):
        """Bounds in KDE coordinates, using explicit panel limits if given."""
        if explicit_limits is not None:
            lower, upper = explicit_limits
            if scale_name == "log":
                if lower <= 0 or upper <= 0:
                    raise ValueError(
                        "Logarithmic density panels require positive limits."
                    )
                return np.log10(lower), np.log10(upper)
            return lower, upper

        transformed = _density_transform(values, scale_name)
        finite = transformed[np.isfinite(transformed)]
        if len(finite) == 0:
            return None

        lower, upper = np.nanpercentile(finite, [1.0, 99.0])
        width = upper - lower

        if not np.isfinite(width) or width <= 0:
            width = max(abs(lower), 1.0) * 0.10

        return lower - 0.08 * width, upper + 0.08 * width

    def _add_background_density(
        ax,
        x,
        y,
        color,
        xscale_name,
        yscale_name,
        xlim_pair=None,
        ylim_pair=None,
    ):
        """
        Draw a population KDE behind the foreground scatter.

        KDE is evaluated in log10 coordinates whenever the corresponding
        plotted axis is logarithmic. This prevents a density map on log axes
        from being biased by a grid uniform in linear coordinates.
        """
        from scipy.stats import gaussian_kde

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.shape != y.shape:
            raise ValueError(
                "BackGroudnDensity X and Y arrays must have the same shape."
            )

        good = np.isfinite(x) & np.isfinite(y)
        if xscale_name == "log":
            good &= x > 0
        if yscale_name == "log":
            good &= y > 0

        x = x[good]
        y = y[good]

        if len(x) < 10:
            return

        tx = _density_transform(x, xscale_name)
        ty = _density_transform(y, yscale_name)

        try:
            kde = gaussian_kde(np.vstack([tx, ty]))
        except Exception:
            # Singular covariance matrices can occur for very small or
            # nearly collinear samples. In that case the density layer is
            # skipped without affecting the scatter panel.
            return

        tx_bounds = _density_bounds(
            x,
            xscale_name,
            explicit_limits=xlim_pair,
        )
        ty_bounds = _density_bounds(
            y,
            yscale_name,
            explicit_limits=ylim_pair,
        )

        if tx_bounds is None or ty_bounds is None:
            return

        tx_grid = np.linspace(tx_bounds[0], tx_bounds[1], 160)
        ty_grid = np.linspace(ty_bounds[0], ty_bounds[1], 160)
        txx, tyy = np.meshgrid(tx_grid, ty_grid)

        density = kde(
            np.vstack([txx.ravel(), tyy.ravel()])
        ).reshape(txx.shape)

        if not np.isfinite(density).any():
            return

        density_max = np.nanmax(density)
        if not np.isfinite(density_max) or density_max <= 0:
            return

        density = density / density_max
        levels = np.linspace(0.20, 1.0, 5)

        xx = _density_inverse(txx, xscale_name)
        yy = _density_inverse(tyy, yscale_name)

        ax.contourf(
            xx,
            yy,
            density,
            levels=levels,
            colors=[color],
            alpha=alphaShade,
            antialiased=True,
            zorder=-10,
        )

        ax.contour(
            xx,
            yy,
            density,
            levels=levels,
            colors=[color],
            linewidths=0.5 * linewidth,
            alpha=min(1.0, alphaShade + 0.30),
            zorder=-9,
        )
        
        

    def _scatter_one(ax, x, y, name_, color_values=None, marker_flags=None):
        """Scatter one population in one panel."""
        if NoneEdgeColor:
            edcolor = None
        else:
            edcolor = "black"

        if marker_flags is not None:
            Markers = np.asarray(marker_flags)

            ax.scatter(
                x[Markers <= 1], y[Markers <= 1],
                color=colors(name_), edgecolor=edgecolors(name_),
                alpha=alphaScater, lw=linesthicker(name_),
                marker=markers(name_), s=20,
            )
            ax.scatter(
                x[Markers == 2], y[Markers == 2],
                color=colors(name_), edgecolor=edgecolors(name_),
                alpha=alphaScater, lw=linesthicker(name_),
                marker=markers(name_), s=45,
            )
            ax.scatter(
                x[Markers >= 3], y[Markers >= 3],
                color=colors(name_), edgecolor=edgecolors(name_),
                alpha=alphaScater, lw=linesthicker(name_),
                marker=markers(name_), s=120,
            )
            return None, None

        if color_values is not None:
            sc_local, norm_local = _scatter_with_colorbar(
                ax=ax,
                x=x,
                y=y,
                color_values=color_values,
                colorbar_key=COLORBAR[0],
                names_l=name_,
                cmap_name=cmap,
                alpha_scatter=alphaScater,
                linewidth=linewidth,
                msizet=msizet,
                HIGHLIGHTPoints=HIGHLIGHTPoints,
            )
            return sc_local, norm_local

        ax.scatter(
            x, y,
            color=colors(name_),
            edgecolor=edcolor,
            alpha=alphaScater,
            lw=0.9,
            marker=markers(name_),
            s=msizet * msize(name_),
        )
        return None, None

    def _apply_xscale(ax, xparam, y_idx):
        requested_scale = _value_for_y(xscale, y_idx, name_="xscale")
        current_scale = requested_scale if requested_scale is not None else scales(xparam)

        if xparam == "z_Birth":
            ax.set_xscale("symlog", linthresh=0.02)
            ax.set_xticks([0, 0.01, 0.1, 1, 10])
            ax.set_xticklabels(["0", "$10^{-2}$", "0.1", "1", "10"])
            return "symlog"

        ax.set_xscale(current_scale)
        if current_scale in ("log", "symlog"):
            ax.xaxis.set_major_formatter(FuncFormatter(format_func_loglog))
        return current_scale

    def _apply_post_panel_formatting(ax, yparam, y_idx):
        if GridMake:
            ax.grid(
                GridMake,
                color="#9e9e9e",
                which="major",
                linewidth=0.6,
                alpha=0.3,
                linestyle=":",
            )

        requested_yscale = _value_for_y(yscales, y_idx, name_="yscales")
        current_yscale = (
            requested_yscale if requested_yscale is not None else scales(yparam)
        )

        ax.set_yscale(current_yscale)


        # Force the project formatter on the Y axis
        ax.yaxis.set_major_formatter(
            FuncFormatter(format_func_loglog)
        )

    def _add_colorbar(fig_, axs_, sc, norm=None):
        if COLORBAR is None:
            return None
        if sc is None and norm is None:
            return None

        cmap_obj = plt.cm.get_cmap(cmap)

        if norm is not None:
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            mappable.set_array([])
        else:
            mappable = sc

        if "Snap" in str(COLORBAR[0]):
            cb = fig_.colorbar(
                mappable,
                ax=axs_.ravel().tolist(),
                ticks=[
                    0.0, 1.97185714, 3.94371429, 5.91557143,
                    7.88742857, 9.85928571, 11.83114286, 13.803,
                ],
                pad=0.02,
                aspect=30,
            )
            cb.ax.set_yticklabels(["14", "12", "10", "8", "6", "4", "2", "0"])

        elif COLORBAR[0] == "sSFRRatioPericenter":
            cb = fig_.colorbar(
                mappable,
                ax=axs_.ravel().tolist(),
                ticks=[0, 0.5, 1, 1.5, 2],
                pad=0.02,
                aspect=(ratioColorbar or 50),
            )
            cb.ax.set_yticklabels(["0", "0.5", "1", "1.5", "2"])

        elif COLORBAR[0] == "logStarZ_99":
            cb = fig_.colorbar(
                mappable,
                ax=axs_.ravel().tolist(),
                ticks=[0, 0.1, 0.2, 0.3, 0.7],
                pad=0.02,
                aspect=(ratioColorbar or 50),
            )
            cb.ax.set_yticklabels(["0", "0.1", "0.2", "0.3", "0.7"])

        elif COLORBAR[0] == "logStarZ_99_75dex":
            cb = fig_.colorbar(
                mappable,
                ax=axs_.ravel().tolist(),
                ticks=[-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0],
                pad=0.02,
                aspect=(ratioColorbar or 50),
            )
            cb.ax.set_yticklabels([
                "-0.7", "-0.6", "-0.5", "-0.4",
                "-0.3", "-0.2", "-0.1", "0",
            ])

        elif COLORBAR[0] == "z_At_FirstEntry":
            cb = fig_.colorbar(
                mappable,
                ax=axs_.ravel().tolist(),
                ticks=[0.0, 0.5, 1.0, 1.5],
                pad=0.02,
                aspect=(ratioColorbar or 50),
            )
            cb.ax.set_yticklabels(["0", "0.5", "1.0", "1.5"])

        else:
            cb = fig_.colorbar(
                mappable,
                ax=axs_.ravel().tolist(),
                pad=0.02,
                aspect=(ratioColorbar or 50),
            )

        cb.set_label(
            _label_from(labels, COLORBAR[0]),
            fontsize=1.2 * fontlabel,
        )
        cb.ax.tick_params(labelsize=0.99 * fontlabel)
        return cb

    # ------------------------------------------------------------------
    # Normalize user inputs
    # ------------------------------------------------------------------
    names = _as_list(names)
    columns, ParamsX, ParamsY, label_general = _normalize_inputs(
        columns, ParamX, ParamsY
    )

    BackgroundParamsX, BackgroundParamsY = _normalize_background_density(
        BackGroudnDensity,
        len(ParamsY),
    )

    if COLORBAR is not None:
        COLORBAR = _as_list(COLORBAR)
    if MarkerSizes is not None:
        MarkerSizes = _as_list(MarkerSizes)
    if LegendNames is not None:
        LegendNames = _as_list(LegendNames)

    dfTime = TNG.extractDF("SNAPS_TIME")

    data_columns, dataX, dataY, dataColor, dataMarker = _load_data(
        names, columns, ParamsX, ParamsY
    )

    dataBackgroundX, dataBackgroundY = _load_background_density(
        names,
        data_columns,
        BackgroundParamsX,
        BackgroundParamsY,
    )

    if columns == ["Snap"]:
        panel_columns = ["Snap"] * len(data_columns)
    else:
        panel_columns = list(columns)

    n_yparams = len(ParamsY)
    n_panel_columns = len(panel_columns)

    if ColumnPlot:
        nrows = n_yparams
        ncols = n_panel_columns
    else:
        nrows = n_panel_columns
        ncols = n_yparams

    fig, axs = _setup_axes(nrows=nrows, ncols=ncols)

    sc_for_colorbar = None
    norm_for_colorbar = None

    # ------------------------------------------------------------------
    # Main panel loop
    # ------------------------------------------------------------------
    for i in range(nrows):
        for j in range(ncols):
            y_idx, column_idx = _panel_indices(i, j)

            yparam = ParamsY[y_idx]
            xparam = ParamsX[y_idx]
            colname = panel_columns[column_idx]
            ax = axs[i, j]

            ylimmin_for_y = _value_for_y(ylimmin, y_idx, name_="ylimmin")
            ylimmax_for_y = _value_for_y(ylimmax, y_idx, name_="ylimmax")
            xlimmin_for_y = _value_for_y(xlimmin, y_idx, name_="xlimmin")
            xlimmax_for_y = _value_for_y(xlimmax, y_idx, name_="xlimmax")

            xlim_pair = _panel_limit_pair(
                xlims,
                column_idx,
                n_panel_columns,
                "xlims",
            )
            ylim_pair = _panel_limit_pair(
                ylims,
                column_idx,
                n_panel_columns,
                "ylims",
            )

            _apply_special_background_rules(
                ax, xparam, yparam, linewidth, fontlabel
            )

            # Background "All" layer
            if All is not None:
                xAll = np.asarray(All[xparam])
                yAll = np.asarray(All[yparam])
                good_all = np.isfinite(xAll) & np.isfinite(yAll)
                ax.scatter(
                    xAll[good_all],
                    yAll[good_all],
                    color=colors["All"],
                    edgecolor=colors["All"],
                    alpha=1.0,
                    marker=".",
                    s=10,
                )

            # Spearman accumulation for all populations in the panel
            if SpearManTestAll:
                XAllSMT = np.array([], dtype=float)
                YAllSMT = np.array([], dtype=float)
                CAllSMT = (
                    np.array([], dtype=float) if COLORBAR is not None else None
                )

            # ----------------------------------------------------------
            # Optional per-population background density
            # ----------------------------------------------------------
            if dataBackgroundX is not None and dataBackgroundY is not None:
                requested_xscale = _value_for_y(
                    xscale,
                    y_idx,
                    name_="xscale",
                )
                density_xscale = (
                    requested_xscale
                    if requested_xscale is not None
                    else scales(xparam)
                )

                requested_yscale = _value_for_y(
                    yscales,
                    y_idx,
                    name_="yscales",
                )
                density_yscale = (
                    requested_yscale
                    if requested_yscale is not None
                    else scales(yparam)
                )

                if xparam == "z_Birth":
                    density_xscale = "symlog"

                for l, name_ in enumerate(names):
                    idx = l
                    if InvertPlot and column_idx == 1:
                        idx = len(names) - l - 1

                    x_background = np.asarray(
                        dataBackgroundX[y_idx][column_idx][idx],
                        dtype=float,
                    )
                    y_background = np.asarray(
                        dataBackgroundY[y_idx][column_idx][idx],
                        dtype=float,
                    )

                    _add_background_density(
                        ax=ax,
                        x=x_background,
                        y=y_background,
                        color=colors(name_),
                        xscale_name=density_xscale,
                        yscale_name=density_yscale,
                        xlim_pair=xlim_pair,
                        ylim_pair=ylim_pair,
                    )
                    
                    if medianDot:
                        good_background = (
                            np.isfinite(x_background)
                            & np.isfinite(y_background)
                        )
        
                        if density_xscale == "log":
                            good_background &= x_background > 0
        
                        if density_yscale == "log":
                            good_background &= y_background > 0
        
                        if np.any(good_background):
                            ax.scatter(
                                np.nanmedian(x_background[good_background]),
                                np.nanmedian(y_background[good_background]),
                                marker="s",
                                edgecolor="black",
                                c=colors(name_),
                                s=16 * msizet,
                                lw=1.0,
                                zorder=19,
                                alpha=1.0,
                            )

            # ----------------------------------------------------------
            # Population loop
            # ----------------------------------------------------------
            for l, name_ in enumerate(names):
                idx = l
                if InvertPlot and column_idx == 1:
                    idx = len(names) - l - 1

                x = np.asarray(dataX[y_idx][column_idx][idx], dtype=float)
                y = np.asarray(dataY[y_idx][column_idx][idx], dtype=float)

                if x.shape != y.shape:
                    raise ValueError(
                        f"X and Y shapes differ in panel "
                        f"(yparam={yparam}, column={data_columns[column_idx]}, "
                        f"name={name_}): {x.shape} versus {y.shape}."
                    )

                good = np.isfinite(x) & np.isfinite(y)
                x_plot = x[good]
                y_plot = y[good]

                # Colorbar values
                cvals = None
                if dataColor is not None:
                    color_panel = _auxiliary_panel(
                        dataColor, y_idx, column_idx, "COLORBAR"
                    )
                    c_all = np.asarray(color_panel[idx], dtype=float)
                    if c_all.shape != x.shape:
                        raise ValueError(
                            f"COLORBAR shape differs from X/Y in panel "
                            f"(yparam={yparam}, column={data_columns[column_idx]}, "
                            f"name={name_}): {c_all.shape} versus {x.shape}."
                        )
                    cvals = c_all[good]

                # Marker-size flags
                mflags = None
                if dataMarker is not None:
                    marker_panel = _auxiliary_panel(
                        dataMarker, y_idx, column_idx, "MarkerSizes"
                    )
                    m_all = np.asarray(marker_panel[idx])
                    if m_all.shape != x.shape:
                        raise ValueError(
                            f"MarkerSizes shape differs from X/Y in panel "
                            f"(yparam={yparam}, column={data_columns[column_idx]}, "
                            f"name={name_}): {m_all.shape} versus {x.shape}."
                        )
                    mflags = m_all[good]

                # Per-population Spearman test
                if SpearManTest and not SpearManTestAll:
                    if len(x_plot) >= 2:
                        corr, pval = spearmanr(x_plot, y_plot)
                    else:
                        corr, pval = np.nan, np.nan
                    print("Name:", name_, "corr:", corr, "p:", pval)

                # Scatter
                sc_local, norm_local = _scatter_one(
                    ax,
                    x_plot,
                    y_plot,
                    name_,
                    color_values=cvals,
                    marker_flags=mflags,
                )
                if sc_local is not None or norm_local is not None:
                    sc_for_colorbar = sc_local
                    norm_for_colorbar = norm_local

                # Medians / quantiles
                if medianBins:
                    xmean, ymed, yq_hi, yq_lo = MATH.split_quantiles(
                        x_plot,
                        y_plot,
                        total_bins=bins,
                        quantile=quantile,
                    )
                    ax.errorbar(
                        xmean,
                        ymed,
                        yerr=(ymed - yq_lo, yq_hi - ymed),
                        ls="None",
                        markeredgecolor="black",
                        elinewidth=2,
                        ms=10,
                        fmt="s",
                        c=colors(name_),
                    )

                elif medianDot:
                    if COLORBAR is not None:
                        if medianDotStar:
                            ax.scatter(
                                np.nanmedian(x_plot),
                                np.nanmedian(y_plot),
                                marker="*",
                                edgecolor="black",
                                c=colors(name_),
                                s=30 * msizetstar,
                                lw=1.1,
                                zorder=20,
                                alpha=1.0,
                            )
                        elif COLORBAR[0] == "last_look_BH":
                            ax.scatter(
                                np.nanmedian(x_plot),
                                np.nanmedian(y_plot),
                                marker=markers(name_ + "Colorbar"),
                                edgecolor="red",
                                c=colors(name_),
                                s=2 * msizet * msize(name_ + "Colorbar"),
                                lw=1.7,
                                zorder=20,
                                alpha=1.0,
                            )
                        else:
                            ax.scatter(
                                np.nanmedian(x_plot),
                                np.nanmedian(y_plot),
                                marker=markers(name_ + "Colorbar"),
                                edgecolor="red",
                                facecolor="none",
                                s=1.5 * msizet * msize(name_ + "Colorbar"),
                                lw=1.7,
                                zorder=20,
                                alpha=1.0,
                            )
                    else:
                        ax.scatter(
                            np.nanmedian(x_plot),
                            np.nanmedian(y_plot),
                            marker="*",
                            edgecolor="black",
                            c=colors(name_),
                            s=33 * msizet,
                            lw=1.3,
                            zorder=20,
                            alpha=1.0,
                        )

                elif medianAll:
                    xmean, ymed, yq_hi, yq_lo = MATH.split_quantiles(
                        x_plot,
                        y_plot,
                        total_bins=bins,
                    )
                    ax.plot(
                        xmean,
                        ymed,
                        color=colors(name_),
                        ls=lines(name_),
                        linewidth=linewidth,
                    )
                    ax.fill_between(
                        xmean,
                        yq_lo,
                        yq_hi,
                        color=colors(name_),
                        alpha=alphaShade,
                    )

                # Accumulate all populations
                if SpearManTestAll:
                    XAllSMT = np.append(XAllSMT, x_plot)
                    YAllSMT = np.append(YAllSMT, y_plot)
                    if CAllSMT is not None and cvals is not None:
                        CAllSMT = np.append(CAllSMT, cvals)

            # ----------------------------------------------------------
            # Panel-level Spearman tests
            # ----------------------------------------------------------
            if SpearManTestAll:
                finite_xy = np.isfinite(XAllSMT) & np.isfinite(YAllSMT)
                if finite_xy.sum() >= 2:
                    corr, pval = spearmanr(
                        XAllSMT[finite_xy], YAllSMT[finite_xy]
                    )
                else:
                    corr, pval = np.nan, np.nan
                print("Panel Spearman X and Y corr:", corr, "p:", pval)

                if CAllSMT is not None and len(CAllSMT) == len(XAllSMT):
                    finite_xc = np.isfinite(XAllSMT) & np.isfinite(CAllSMT)
                    if finite_xc.sum() >= 2:
                        corr, pval = spearmanr(
                            XAllSMT[finite_xc], CAllSMT[finite_xc]
                        )
                    else:
                        corr, pval = np.nan, np.nan
                    print("Panel Spearman X and Colorbar corr:", corr, "p:", pval)

                    finite_cy = np.isfinite(CAllSMT) & np.isfinite(YAllSMT)
                    if finite_cy.sum() >= 2:
                        corr, pval = spearmanr(
                            CAllSMT[finite_cy], YAllSMT[finite_cy]
                        )
                    else:
                        corr, pval = np.nan, np.nan
                    print("Panel Spearman Colorbar and Y corr:", corr, "p:", pval)

                print("\n")

            # Equal line
            if (
                EqualLine
                and EqualLineMin is not None
                and EqualLineMax is not None
            ):
                xx = np.linspace(EqualLineMin, EqualLineMax)
                ax.plot(
                    xx, xx, ls="--", color="tab:blue", linewidth=linewidth
                )

            # ----------------------------------------------------------
            # Scales, limits, and project-specific ticks
            # ----------------------------------------------------------
           

            if ylimmin_for_y is not None and ylimmax_for_y is not None:
                ax.set_ylim(ylimmin_for_y, ylimmax_for_y)

            if xlimmin_for_y is not None and xlimmax_for_y is not None:
                ax.set_xlim(xlimmin_for_y, xlimmax_for_y)

            # Explicit panel-wise pairs take precedence over the older
            # xlimmin/xlimmax and ylimmin/ylimmax interfaces.
            if ylim_pair is not None:
                ax.set_ylim(*ylim_pair)

            if xlim_pair is not None:
                ax.set_xlim(*xlim_pair)

            
            
            _apply_post_panel_formatting(ax, yparam, y_idx)
            _apply_xscale(ax, xparam, y_idx)

            _apply_special_xaxis_rules(
                ax=ax,
                ParamX_=xparam,
                yparam=yparam,
                ylimmin_for_y=ylimmin_for_y,
                fontlabel_=fontlabel,
            )
            
            
            
            # ----------------------------------------------------------
            # Axis labels and in-panel sample/snapshot titles
            # ----------------------------------------------------------
            # Keep the numerical Y axis and its ylabel. In the transposed
            # layout this gives one ylabel for every stacked row, matching the
            # style of the reference figure.
            if j == 0:
                ax.set_ylabel(
                    _y_parameter_label(yparam),
                    fontsize=1.2 * fontlabel,
                )
                ax.tick_params(axis="y", labelsize=0.99 * fontlabel)
                
                ax.yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog)
                )
                
                if yparam == 'sigma_InSitu':
                    ax.set_yticks([20, 30, 40, 50, 60])
                    ax.set_yticklabels(['20', '30', '40', '50', '60'])
                

            # ``title`` identifies the sample/snapshot dimension. Instead of
            # using ax.set_title(), place it inside the relevant panel.
            if colname == "Snap" or title:
                if ColumnPlot:
                    # One title per visual column, placed in the top panel.
                    title_panel = i == 0
                else:
                    # One title per visual row, placed in its first panel.
                    title_panel = j == 0

                if title_panel:
                    _add_panel_title_text(
                        ax,
                        _panel_column_label(column_idx),
                    )

            # Preserve Y tick labels on the transposed stacked panels.
            if not ColumnPlot:
                ax.tick_params(axis="y", labelsize=0.99 * fontlabel)

            # One optional in-panel Y-parameter annotation per visual Y dimension.
            if xlabelintext:
                if ColumnPlot:
                    annotation_panel = j == ncols - 1
                else:
                    annotation_panel = i == 0

                if annotation_panel:
                    Afont = {
                        "color": "black",
                        "size": fontlabel,
                    }
                    anchored_text = AnchoredText(
                        _label_from(texts, yparam),
                        loc="upper right",
                        prop=Afont,
                    )
                    ax.add_artist(anchored_text)

            # X labels only on the last visual row.
            if i == nrows - 1:
                ax.set_xlabel(
                    _x_parameter_label(xparam),
                    fontsize=1.2 * fontlabel,
                )
                ax.tick_params(axis="x", labelsize=0.99 * fontlabel)
            elif not ColumnPlot:
                # With independent row-wise xlims, Matplotlib no longer
                # suppresses the upper tick labels automatically.
                ax.tick_params(axis="x", labelbottom=False)

            # ----------------------------------------------------------
            # Legends: positions remain physical (column, row)
            # ----------------------------------------------------------
            if legend and LegendNames is not None and legpositions is not None:
                for legpos, LegendName in enumerate(LegendNames):
                    if legpos >= len(legpositions):
                        continue

                    if ColumnPlot:
                        legend_column = legpositions[legpos][0]
                        legend_row = legpositions[legpos][1]
                    else:
                        # Keep the requested position attached to the same
                        # sample/snapshot panel after transposing the grid:
                        # [column, row] -> [row, column].
                        legend_row = legpositions[legpos][0]
                        legend_column = legpositions[legpos][1]

                    if j == legend_column and i == legend_row:
                        custom_lines, label, ncol, _mult = Legend(
                            LegendName,
                            msizeMult=msizeMult,
                            linewidth=linewidth,
                        )

                        ax.legend(
                            custom_lines,
                            label,
                            ncol=ncol,
                            loc=_value_for_index(loc, legpos, "best"),
                            fontsize=0.88 * fontlabel,
                            framealpha=framealpha,
                            columnspacing=columnspacing,
                            handlelength=handlelength,
                            handletextpad=handletextpad,
                            labelspacing=labelspacing,
                            handler_map={Circle: HandlerCircle()},
                        )

    # ------------------------------------------------------------------
    # Automatic layer legend for the density mode
    # ------------------------------------------------------------------
    if BackGroudnDensity is not None and legend:
        layer_handles = [
            mpl.patches.Patch(
                facecolor="0.5",
                edgecolor="0.5",
                alpha=alphaShade,
                label="all stars, density",
            ),
            mpl.lines.Line2D(
                [0],
                [0],
                color="black",
                marker=m,
                linestyle="none",
                markersize=0.75 * msizet,
                label="in-situ stars",
            ),
        ]

        if medianDot:
            layer_handles.append(
                mpl.lines.Line2D(
                    [0],
                    [0],
                    color="black",
                    marker="*",
                    markeredgecolor="black",
                    linestyle="none",
                    markersize=1.10 * msizet,
                    label="in-situ median",
                )
            )

        layer_ax = axs[1,0]
        previous_legend = layer_ax.get_legend()
        if previous_legend is not None:
            layer_ax.add_artist(previous_legend)

        layer_ax.legend(
            handles=layer_handles,
            ncol=1,
            loc="lower left",
            fontsize=0.88 * fontlabel,
            framealpha=framealpha,
            columnspacing=columnspacing,
            handlelength=handlelength,
            handletextpad=0.35,
            labelspacing=labelspacing,
        )

    # ------------------------------------------------------------------
    # Global colorbar and output
    # ------------------------------------------------------------------
    _add_colorbar(
        fig,
        axs,
        sc_for_colorbar,
        norm=norm_for_colorbar,
    )

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
                        color = colors(row[l])
                        ls = lines(row[l], 'solid')
        
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
                            Y)], color=colors(ShowPopName), ls=ls, 
                            lw=1.*linesthicker(ShowPopName), dash_capstyle = capstyles(ShowPopName))
            

                        axs[i][j].fill_between(
                            xparam[~np.isnan(Y)], Y[~np.isnan(Y)] - Yerr[~np.isnan(Y)], 
                            Y[~np.isnan(Y)] + Yerr[~np.isnan(Y)], color=colors(ShowPopName+'Error'), ls=ls, alpha=alphaShade)
                         
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

                        axs[i][j].scatter(xparam[(Markervalues > 0)], values[(Markervalues > 0)], color=colors(
                            str(l)), lw=1., marker='o',  edgecolors='black', s=250, alpha=0.7)
                        axs[i][j].scatter(xparam[(markervalues > 0)], values[(markervalues > 0)], color=colors(
                            str(l)), lw=1., marker='s',  edgecolors='black', s=100, alpha=0.7)
                        
                        if ~np.isnan(SnapCorotateMerger):
                            axs[i][j].scatter(time[(MarkerTotvalues > 0)], values[(MarkerTotvalues > 0)], 
                                              color=colors(str(l)), lw=1., marker='*',  edgecolors='black', s=300, alpha=0.7)

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

                        axs[i][j].scatter(x[(Markervalues > 0)], values[(Markervalues > 0)], color=colors(
                            str(l)), lw=1., marker='o',  edgecolors='black', s=130, alpha=0.5)
                        axs[i][j].scatter(x[(markervalues > 0)], values[(markervalues > 0)], color=colors(
                            str(l)), lw=1., marker='o',  edgecolors='black', s=110, alpha=0.5)
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
            if scales(row[0]) != None :
                axs[i][j].set_yscale(scales(row[0]))
            if scales(row[0]) == 'log' :
                axs[i][j].yaxis.set_major_formatter(
                    FuncFormatter(format_func_loglog))
                
            if row[-1] == 'FracType1':
                axs[i][j].set_yticks([0.2, 0.3, 0.5, 1])
                axs[i][j].set_yticklabels(['0.2','0.3', '0.5', '1'])

            
            if j == 0:

                if len(row) > 1:
                    axs[i][j].set_ylabel(
                        labelsequal.get(row[0], row[0]), fontsize=fontlabel)

                else:
                    axs[i][j].set_ylabel(labels.get(row[0], row[0]), fontsize=fontlabel)

            if j == len(IDs) - 1:
                if xlabelintext and not limaxis and len(rows) > 1:
                    Afont = {'color':  'black',
                             'size': fontlabel,
                             }
                    anchored_text = AnchoredText(
                        texts(row), loc='upper right', prop=Afont)
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
                        titles(
                            title[i]), loc=postext[i], prop=Afont)
                    axs[i][j].add_artist(anchored_text)


            if i == 0:

                if title and not ColumnPlot:
                    axs[i][j].set_title(titles(
                        title[j]), fontsize=1*fontlabel)
                

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
                        axs[i][j].set_xlabel(r'$\phi_\mathrm{orbital}$', fontsize=fontlabel)
                        axs[i][j].set_xticks([-1, -0.5, 0, 1, 2, 3, 4, 5] )
                        axs[i][j].set_xticklabels(['', 'E', '0', '1', '2', '3', '4', '5'])
                        axs[i][j].set_xlim(-1, 5.5)
                        
                elif Type == 'CoEvolution':
                    axs[i][j].set_xscale(scales(Xparam[i]))
                    if scales(Xparam[i]) == 'log':
                        axs[i][j].xaxis.set_major_formatter(
                            FuncFormatter(format_func_loglog))
                    axs[i][j].set_xlabel(labels(
                        Xparam[i]), fontsize=fontlabel)

    savefig(savepath, savefigname, TRANSPARENT)

    return

# def MakeMedianAndIDs(Snaps, IDs, rmin, rmax, nbins, dfSample,
#                      PartType='PartType4', velPlot=False, gasSF=False):

#     yIDs = []
#     massIDs = []
#     xIDs = []
#     notIndex = []

#     for l, ID in enumerate(IDs):
#         if ID in [603556, 602133]:
#             continue

#         snap = Snaps[l]
#         if np.isnan(snap):
#             continue
#         snap = int(snap)

#         yrad, rad, mass = TNG.MakeDensityProfileMean(
#             snap, ID, rmin, rmax, nbins,
#             PartType=PartType,
#             velPlot=velPlot,
#             gasSF=gasSF
#         )

#         # proteção contra retornos inválidos
#         if len(yrad) == 1 or (ID == 603556 or ID == 602133):
#             notIndex = np.append(notIndex, l)
#             continue
        
#         if (
#             yrad is None or rad is None or mass is None
#             or len(yrad) == 0
#             or len(rad) == 0
#             or len(mass) == 0
#             or len(yrad) != nbins
#             or len(rad) != nbins
#             or len(mass) != nbins
#         ):
#             notIndex.append(l)
#             continue

#         yIDs.append(yrad)
#         xIDs.append(rad)
#         massIDs.append(mass)

#     if len(yIDs) == 0:
#         return np.nan, np.nan, np.nan, np.nan, np.array([]), np.array([]), np.array([]), np.array(notIndex)

#     yIDs = np.array(yIDs)
#     xIDs = np.array(xIDs)
#     massIDs = np.array(massIDs)

#     Rvalues = xIDs.T
#     Values = yIDs.T
#     Masses = massIDs.T

#     x = np.array([])
#     y = np.array([])
#     yerr = np.array([])
#     mass = np.array([])

#     for k, value in enumerate(Values):
#         x = np.append(x, np.nanmedian(Rvalues[k]))
#         y = np.append(y, np.nanmedian(value))
#         yerr = np.append(yerr, MATH.boostrap_func(value, func=np.nanmedian, num_boots=1000))
#         mass = np.append(mass, np.nanmedian(Masses[k]))

#     return x, y, yerr, mass, xIDs, yIDs, massIDs, np.array(notIndex)


def MakeMedianAndIDs(Snaps, IDs, rmin, rmax, nbins, dfSample, PartType = 'PartType4', Cond = 'None', velPlot= False, gasSF = False):
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
        yrad, rad, mass = TNG.MakeDensityProfileMean(snap, ID, rmin, rmax, nbins, PartType = PartType, velPlot= velPlot, Cond = Cond, gasSF = True)
        
        if len(yrad) == 1 or (ID == 603556 or ID == 602133):
            notIndex = np.append(notIndex, l)
            continue
        if l == 0 or len(yIDs ) == 0:
            yIDs  = np.append(yIDs , yrad)
            xIDs = np.append(xIDs, rad)
            massIDs = np.append(massIDs, mass)
            lenPrevious = len(yrad)
    
        else:
            if len(yrad) == 0:
                yrad = np.zeros(lenPrevious)*np.nan
                mass = np.zeros(lenPrevious)*np.nan
                rad = np.zeros(lenPrevious)*np.nan

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