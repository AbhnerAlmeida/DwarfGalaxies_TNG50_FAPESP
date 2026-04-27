#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 10:59:25 2026

@author: abhner
"""

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import MATH

sys.path.append(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/analyzes")
sys.path.append(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/src")
sys.path.append(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/analyzes/GaryScripts")

import TNGFunctions as TNG
import ExtractTNG as ETNG
import PlotFunctions as plot

#%%

# --- Cosmological parameters (consistent with IllustrisTNG metadata) ---
Omegam0 = 0.3089
h = 0.6774

# --- Simulation tag used by your I/O helpers ---
SIMTNG = "TNG50"

# --- Base paths ---
SUBHALO_HISTORY_PATH = os.getenv("HOME")+"/TNG_Analyzes/SubhaloHistory"

# --- Common tables ---
SNAPS_TIME_PATH = SUBHALO_HISTORY_PATH + "/SNAPS_TIME.csv"
dfTime = pd.read_csv(SNAPS_TIME_PATH)

# --- Main data products (exported from your pipeline) ---
df_z0_Mstar_Range = TNG.extractDF("Sample", SIM=SIMTNG)
PaperII = TNG.extractDF("PaperII", SIM=SIMTNG)


#%%

def summarize_Sample_key(
    key,
    populations,
    dfName='Sample',
    Name='Name',
    round_digits=3,
    latex_style=False
):
    """
    Summarize one variable for several populations using:
    N, median, p16, p84, mean, std.

    Parameters
    ----------
    key : str
        Column name to summarize.
    populations : list of str
        Population names to extract with extractPopulation().
    dfName : str
        Dataframe identifier passed to extractPopulation().
    Name : str
        Name column passed to extractPopulation().
    round_digits : int
        Number of decimal places.
    latex_style : bool
        If True, adds a formatted 'median_p16_p84' column.

    Returns
    -------
    pd.DataFrame
    """
    rows = []

    for pop in populations:
        sample = TNG.extractPopulation(pop, dfName=dfName, Name=Name)

        # Special handling if needed
        if key == 'z_At_FinalEntry':
            values = sample.loc[sample[key] >= 0, key].values
        else:
            values = sample[key].values

        values = values[~np.isnan(values)]

        if len(values) == 0:
            row = {
                'population': pop,
                'N': 0,
                'median': np.nan,
                'p16': np.nan,
                'p84': np.nan,
                'mean': np.nan,
                'std': np.nan,
            }
        else:
            p16, med, p84 = np.percentile(values, [16, 50, 84])
            row = {
                'population': pop,
                'N': len(values),
                'median': np.round(med, round_digits),
                'p16': np.round(p16, round_digits),
                'p84': np.round(p84, round_digits),
                'mean': np.round(np.mean(values), round_digits),
                'std': np.round(np.std(values, ddof=1), round_digits) if len(values) > 1 else np.nan,
            }

            if latex_style:
                row['median_p16_p84'] = (
                    f"{row['median']:.{round_digits}f} "
                    f"[{row['p16']:.{round_digits}f}, {row['p84']:.{round_digits}f}]"
                )

        rows.append(row)

    return pd.DataFrame(rows)

#%%

pops_m200 = [
    'Normal_Satellite_DMrich',
    'MBC_Satellite_DMrich',
    'SBC_Satellite_DMrich',
    'Normal_Satellite_DMpoor',
    'MBC_Satellite_DMpoor',
    'SBC_Satellite_DMpoor',
]

df_m200_summary = summarize_Sample_key(
    'M200Mean',
    pops_m200,
    dfName='PaperII',
    round_digits=2,
    latex_style=True
)

print(df_m200_summary)

#%%

def build_table2_summary_for_key(key, mapping, dfName='PaperII', round_digits=2):
    """
    mapping example:
    {
        'DM-rich': {
            'Normals': 'Normal_Satellite_DMrich',
            'CompactsMB': 'MBC_Satellite_DMrich',
            'CompactsSB': 'SBC_Satellite_DMrich',
        },
        'DM-poor': {
            'Normals': 'Normal_Satellite_DMpoor',
            'CompactsMB': 'MBC_Satellite_DMpoor',
            'CompactsSB': 'SBC_Satellite_DMpoor',
        }
    }
    """
    out = {}

    for row_name, cols in mapping.items():
        out[row_name] = {}
        for col_name, pop_name in cols.items():
            df = summarize_Sample_key(
                key,
                [pop_name],
                dfName=dfName,
                round_digits=round_digits,
                latex_style=True
            )
            out[row_name][col_name] = df.iloc[0]['median_p16_p84']

    return pd.DataFrame(out).T

#%%
mapping = {
    'DM-rich': {
        'Normals': 'Normal_Satellite_DMrich',
        'CompactsMB': 'MBC_Satellite_DMrich',
        'CompactsSB': 'SBC_Satellite_DMrich',
    },
    'DM-poor': {
        'Normals': 'Normal_Satellite_DMpoor',
        'CompactsMB': 'MBC_Satellite_DMpoor',
        'CompactsSB': 'SBC_Satellite_DMpoor',
    }
}

table2_m200 = build_table2_summary_for_key('M200Mean', mapping, dfName='PaperII', round_digits=2)
print(table2_m200)


#%%

IDs_SBCSatelliteDMrich = np.array([   91,    156,    165,  63918,  96799, 167427, 198195, 198199,
       264900, 319740, 324134, 355736, 379807, 439101, 440412, 468596,
       489208, 499708, 513846, 531321, 570842, 573063, 574616, 592022,
       616015, 618581, 625186, 629892, 649988, 661261, 667085, 668241,
       671097, 684712, 692883, 694489, 697154, 724892, 725997, 770775,
       804308])

IDs_SBCSatelliteDMpoor = np.array([63990,  64002,  96853, 117357, 229992, 229996, 294887, 294895,
                                   358627, 422770, 428191, 500583, 516761, 602132, 602133, 603556])

IDs_MBCSatelliteDMrich = np.array([550,  63973,  96789, 117311, 220616, 229958, 264932, 289394,
       300918, 307497, 307502, 342468, 372756, 419621, 422762, 435755,
       445628, 450924, 487746, 488533, 489207, 571909, 579511, 590015,
       597142, 681818, 701373, 814011, 823293])

IDs_MBCSatelliteDMpoor = np.array([232,    261,    281,    300,    319,    333,  63993,  64081,
        64129,  96941, 117464, 144008, 144098, 167499, 185005, 185058,
       208883, 220626, 242863, 253897, 253905, 253965, 264911, 264972,
       275601, 282802, 282807, 289444, 307510, 319738, 319743, 377662,
       386293, 394628, 404834, 421566, 421567, 422763, 425726, 432119,
       457435, 467420, 482157, 502998, 530853, 536657, 545439, 549748,
       549750, 558069, 571075, 571910, 588180, 647769])

IDs_NormalSatelliteDMrich = np.array([333431, 677786, 143950,    203,    147, 775527, 198247, 117291,
       427216, 198231, 770462, 117299, 386274, 471252, 800643,  63957,
       594247, 465257, 253882, 763684, 574885, 837220, 143967, 184962,
          423, 508541, 472552,    214, 749432, 184980,  63939, 184966,
       435760, 414921,    181, 345881, 229952, 282799, 264901, 117426,
       184984, 117314,  64009, 184982, 294884, 229984, 208826, 494014,
       433290, 198237])

IDs_NormalSatelliteDMpoor = np.array([184957, 220633, 338455, 436937, 560083, 586424, 602131, 603005])

#%%
fontlabel = 22
nboot = 5000
nbins = 30
plt.rcParams.update({'figure.figsize': (4*3, 4*3)})
fig = plt.figure()
gs = fig.add_gridspec(3, 3, hspace=0, wspace=0)
axs = gs.subplots(sharex='col', sharey='row')
Names = [r'Normals', r'Compacts$_\mathrm{MB}$', r'Compacts$_\mathrm{SB}$']
rmin = 0.06
rmax = 35
for j, PartType in enumerate(['PartType0', 'PartType0', 'PartType4']):
    if j == 0:
        velPlot = True
        gasSF = False
    elif j ==1:
        velPlot = False
        gasSF = False
    else:
        velPlot = False
        gasSF = False
    for linplot, IDs in enumerate([IDs_NormalSatelliteDMrich, IDs_MBCSatelliteDMrich, IDs_SBCSatelliteDMrich]):
        
        dFHalfStar = TNG.extractDF('SubhaloHalfmassRadType4')
        dFHalfGasRad = TNG.extractDF('SubhaloHalfmassRadType0')
        
        
        #SNAP FIRST
        SnapsFirst = []
        for l, ID in enumerate(IDs):
            if ID == 603556 or ID == 602133:
                continue
            snapFirstEntry = PaperII.Snap_At_FirstEntry.loc[PaperII.SubfindID_99 == ID].values[0]
            if np.isnan(snapFirstEntry) or snapFirstEntry < 0:
                snapFirstEntry = 67
            snap = dfTime.loc[abs(dfTime.Age - dfTime.loc[dfTime.Snap == snapFirstEntry, 'Age'].values[0]) < 1.5, 'Snap'].values[-1]
            if snap < 17:
                snap = 17
            SnapsFirst.append(int(snap))
             
        xFirst, yFirst, yerrFirst, massFirst, xIDsFirst, yIDFirst, massIDsFirst, notIndex = plot.MakeMedianAndIDs(SnapsFirst, IDs, rmin, rmax, nbins, PaperII, PartType = PartType, velPlot= velPlot, gasSF = False)
      
        #First
        colors = plt.cm.Blues(np.linspace(0,1,len(yIDFirst)))
        #plot.MakeLines(j, axs[j][linplot], yIDFirst, xIDsFirst, IDs, notIndex, colors)
          
        #SNAP FIRST
        Snaps = []
        for l, ID in enumerate(IDs):
            if ID == 603556 or ID == 602133:
                continue
            snapFirstEntry = PaperII.Snap_At_FirstEntry.loc[PaperII.SubfindID_99 == ID].values[0]
            if np.isnan(snapFirstEntry) or snapFirstEntry < 0:
                snapFirstEntry = 67
            Snaps.append(int(snapFirstEntry))
             
        xSecond, ySecond, yerrSecond, massSecond, xIDsSecond, yIDSecond, massIDsSecond, notIndex = plot.MakeMedianAndIDs(Snaps, IDs, rmin, rmax, nbins, PaperII, PartType = PartType, velPlot= velPlot, gasSF = False)
      
        #Second
        colors = plt.cm.Greens(np.linspace(0,1,len(yIDSecond)))
        plot.MakeLines(j, axs[j][linplot], yIDSecond, xIDsSecond, IDs, notIndex, colors)
          
        #SNAP Final
        SnapsFinal = []
        for l, ID in enumerate(IDs):
            if ID == 603556 or ID == 602133:
                continue
            snapFirstEntry = PaperII.Snap_At_FirstEntry.loc[PaperII.SubfindID_99 == ID].values[0]
            if np.isnan(snapFirstEntry) or snapFirstEntry < 0:
                snapFirstEntry = 67
            if linplot == 1:
                snap = dfTime.loc[abs(dfTime.Age - dfTime.loc[dfTime.Snap == snapFirstEntry, 'Age'].values[0]) < 2, 'Snap'].values[0]
            else:
                snap = 99 #
            SnapsFinal.append(int(snap))
             
        xFinal, yFinal, yerrFinal, massFinal, xIDsFinal, yIDsFinal, massIDsFinal, notIndex = plot.MakeMedianAndIDs(SnapsFinal, IDs, rmin, rmax, nbins, PaperII, PartType = PartType, velPlot= velPlot, gasSF = False)
        
        colors = plt.cm.Reds(np.linspace(0,1,len(yIDsFinal)))
        #plot.MakeLines(j, axs[j][linplot], yIDsFinal, xIDsFinal, IDs, notIndex, colors)

    
        if j == 1 or j == 2:
            
            axs[j][linplot].fill_between(xFirst, (yFirst - yerrFirst)*xFirst**2, (yFirst + yerrFirst)*xFirst**2, 
                                        color='tab:red',  alpha=0.3) 
            axs[j][linplot].fill_between(xSecond, (ySecond - yerrSecond)*xSecond**2, (ySecond + yerrSecond)*xSecond**2, 
                                        color='tab:green',  alpha=0.3) 
            axs[j][linplot].fill_between(xFinal, (yFinal - yerrFinal)*xFinal**2, (yFinal + yerrFinal)*xFinal**2, 
                                        color='tab:blue',  alpha=0.3) 
            
            axs[j][linplot].plot(xFirst , yFirst*xFirst**2, color = 'red', lw = 1.5, label = r'$t_\mathrm{entry} - 2 \, \mathrm{[Gyr]}$')
   
            axs[j][linplot].plot(xSecond , ySecond*xSecond**2, color = 'green', lw = 1.5, label = r'$t_\mathrm{entry}$')
            
            axs[j][linplot].plot(xFinal , yFinal*xFinal**2, color = 'blue', lw = 1.5, label = r'$t_\mathrm{entry} + 2 \, \mathrm{[Gyr]}$')
          
            if j == 1:
                #SNAP FIRST
                SnapsFirst = []
                for l, ID in enumerate(IDs):
                    if ID == 603556 or ID == 602133:
                        continue
                    snapFirstEntry = PaperII.Snap_At_FirstEntry.loc[PaperII.SubfindID_99 == ID].values[0]
                    if np.isnan(snapFirstEntry) or snapFirstEntry < 0:
                        snapFirstEntry = 67
                    snap = dfTime.loc[abs(dfTime.Age - dfTime.loc[dfTime.Snap == snapFirstEntry, 'Age'].values[0]) < 1.5, 'Snap'].values[-1]
                    if snap < 17:
                        snap = 17
                    SnapsFirst.append(int(snap))
                     
                xFirst, yFirst, yerrFirst, massFirst, xIDsFirst, yIDFirst, massIDsFirst, notIndex = plot.MakeMedianAndIDs(SnapsFirst, IDs, rmin, rmax, nbins, PaperII, PartType = PartType, velPlot= velPlot, gasSF = True)
              
                
                #SNAP FIRST
                Snaps = []
                for l, ID in enumerate(IDs):
                    if ID == 603556 or ID == 602133:
                        continue
                    snapFirstEntry = PaperII.Snap_At_FirstEntry.loc[PaperII.SubfindID_99 == ID].values[0]
                    if np.isnan(snapFirstEntry) or snapFirstEntry < 0:
                        snapFirstEntry = 67
                    Snaps.append(int(snapFirstEntry))
                     
                xSecond, ySecond, yerrSecond, massSecond, xIDsSecond, yIDSecond, massIDsSecond, notIndex = plot.MakeMedianAndIDs(Snaps, IDs, rmin, rmax, nbins, PaperII, PartType = PartType, velPlot= velPlot, gasSF = True)
              
                #Second
                colors = plt.cm.Greens(np.linspace(0,1,len(yIDSecond)))
                #plot.MakeLines(j, axs[j][linplot], yIDSecond, xIDsSecond, IDs, notIndex, colors)
                  
                #SNAP Final
                SnapsFinal = []
                for l, ID in enumerate(IDs):
                    if ID == 603556 or ID == 602133:
                        continue
                    snapFirstEntry = PaperII.Snap_At_FirstEntry.loc[PaperII.SubfindID_99 == ID].values[0]
                    if np.isnan(snapFirstEntry) or snapFirstEntry < 0:
                        snapFirstEntry = 67
                    if linplot == 1:
                        snap = dfTime.loc[abs(dfTime.Age - dfTime.loc[dfTime.Snap == snapFirstEntry, 'Age'].values[0]) < 2, 'Snap'].values[0]
                    else:
                        snap = 99 #
                    SnapsFinal.append(int(snap))
                     
                xFinal, yFinal, yerrFinal, massFinal, xIDsFinal, yIDsFinal, massIDsFinal, notIndex = plot.MakeMedianAndIDs(SnapsFinal, IDs, rmin, rmax, nbins, PaperII, PartType = PartType, velPlot= velPlot, gasSF = True)
                
                    
                axs[j][linplot].plot(xFirst , yFirst*xFirst**2, ls = 'dashed', color = 'red', lw = 1.7, label = r'$t_\mathrm{entry} - 2 \, \mathrm{[Gyr]}$')
       
                axs[j][linplot].plot(xSecond , ySecond*xSecond**2,  ls = 'dashed', color = 'green', lw = 1.7, label = r'$t_\mathrm{entry}$')
                
                axs[j][linplot].plot(xFinal , yFinal*xFinal**2, ls = 'dashed',  color = 'blue', lw = 1.7, label = r'$t_\mathrm{entry} + 2 \, \mathrm{[Gyr]}$')
              
                
        else:
            axs[j][linplot].fill_between(xFirst, (yFirst - yerrFirst), (yFirst + yerrFirst), 
                                        color='tab:red',  alpha=0.3) 
            axs[j][linplot].fill_between(xSecond, (ySecond - yerrSecond), (ySecond + yerrSecond), 
                                        color='tab:green',  alpha=0.3) 
            axs[j][linplot].fill_between(xFinal, (yFinal - yerrFinal), (yFinal + yerrFinal), 
                                        color='tab:blue',  alpha=0.3) 
            
            
            axs[j][linplot].plot(xFirst , yFirst, color = 'red', lw = 1.5, label = r'$t_\mathrm{entry} - 2 \, \mathrm{[Gyr]}$')
            axs[j][linplot].plot(xSecond , ySecond, color = 'green', lw = 1.5, label = r'$t_\mathrm{entry}$')
            axs[j][linplot].plot(xFinal , yFinal, color = 'blue', lw = 1.5, label = r'$t_\mathrm{entry} + 2 \, \mathrm{[Gyr]}$')
        
        if j == 0:
            axs[j][linplot].set_title(Names[linplot], fontsize=1.1*fontlabel)
        
        if j != 0 :
            axs[j][linplot].set_yscale('log')

        axs[j][linplot].set_xscale('log')
        
        axs[j][linplot].set_xlim(0.2, 75)

        if j == 0:
            axs[j][linplot].set_ylim(-100, 10)
        elif j == 1:
            axs[j][linplot].set_ylim(5e5, 5e7)

        else:
            axs[j][linplot].set_ylim(5e5, 2e8)
        
        if j == 2:
            axs[j][linplot].tick_params(axis='x', labelsize=0.99*fontlabel)
    
        #axs[0][linplot].axhline(np.nansum(massEntry[massEntry > 0]) / 2., ls = '--', color = 'black')
    
    
        if linplot == 0:
            axs[j][linplot].tick_params(axis='y', labelsize=0.99*fontlabel)
            if j == 2:
                axs[j][linplot].set_ylabel(r'$\rho_{\star} (r) r^2  \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$', fontsize=fontlabel)
            elif j == 1:
                axs[j][linplot].set_ylabel(r'$\rho_{\mathrm{gas}} (r) r^2  \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$', fontsize=fontlabel)
            else:
                axs[j][linplot].set_ylabel( r'$v_\mathrm{r, \; gas} (r) \, \, [\mathrm{km \, s}^{-1}]$', fontsize=fontlabel)

        if j == 2:
            axs[j][linplot].set_xlabel(r'$r  \; [\mathrm{kpc}]$', fontsize=fontlabel)
        
        
        if j != 0:
            axs[j][linplot].yaxis.set_major_formatter(
                plot.FuncFormatter(plot.format_func_loglog))
        axs[j][linplot].xaxis.set_major_formatter(
            plot.FuncFormatter(plot.format_func_loglog))
        axs[0][0].legend(fontsize=0.9*fontlabel, framealpha = 0.4)
        
# ==========================================================
# LEGENDAS
# ==========================================================
time_handles = [
    Line2D([0], [0], color='red',   lw=1.5, ls='-', label=r'$t_\mathrm{entry} - 2 \, \mathrm{[Gyr]}$'),
    Line2D([0], [0], color='green', lw=1.5, ls='-', label=r'$t_\mathrm{entry}$'),
    Line2D([0], [0], color='blue',  lw=1.5, ls='-', label=r'$t_\mathrm{entry} + 2 \, \mathrm{[Gyr]}$')
]

style_handles = [
    Line2D([0], [0], color='black', lw=1.5, ls='-',  label='Total gas'),
    Line2D([0], [0], color='black', lw=1.2, ls='--', label='Star-forming gas')
]

leg1 = axs[0][0].legend(handles=time_handles, fontsize=0.9 * fontlabel, framealpha=0.4, loc='lower right')
axs[0][0].add_artist(leg1)

axs[1][2].legend(handles=style_handles, fontsize=0.75 * fontlabel, framealpha=0.4, loc='upper right')

plt.savefig(os.getenv("HOME")+'/TNG_Analyzes/Figs/ProfilesTest.pdf', bbox_inches='tight')    
#plt.ylim(- 0.99, 0.99)

#%%


#%%
fontlabel = 22
nboot = 5000
nbins = 30
plt.rcParams.update({'figure.figsize': (4*3, 4*3)})

fig = plt.figure()
gs = fig.add_gridspec(3, 3, hspace=0, wspace=0)
axs = gs.subplots(sharex='col', sharey='row')

Names = [r'Normals', r'Compacts$_\mathrm{MB}$', r'Compacts$_\mathrm{SB}$']
rmin = 0.06
rmax = 35


def plot_profile(ax, x, y, color, lw=1.5, ls='-', alpha=1.0, multiply_r2=False, label=None):
    """
    Plota apenas pontos válidos.
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    y = np.atleast_1d(np.array(y, dtype=float))

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if multiply_r2:
        mask &= (y > 0)

    if np.sum(mask) == 0:
        return

    xx = x[mask]
    yy = y[mask]

    if multiply_r2:
        yy = yy * xx**2

    ax.plot(xx, yy, color=color, lw=lw, ls=ls, alpha=alpha, label=label)


def fill_profile(ax, x, y, yerr, color, alpha=0.3, multiply_r2=False):
    """
    Fill_between apenas para pontos válidos.
    """
    x = np.atleast_1d(np.array(x, dtype=float))
    y = np.atleast_1d(np.array(y, dtype=float))
    yerr = np.atleast_1d(np.array(yerr, dtype=float))

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (x > 0)

    ylow = y - yerr
    yhigh = y + yerr

    if multiply_r2:
        mask &= (ylow > 0) & (yhigh > 0)
        if np.sum(mask) == 0:
            return
        ax.fill_between(
            x[mask],
            ylow[mask] * x[mask]**2,
            yhigh[mask] * x[mask]**2,
            color=color,
            alpha=alpha
        )
    else:
        if np.sum(mask) == 0:
            return
        ax.fill_between(
            x[mask],
            ylow[mask],
            yhigh[mask],
            color=color,
            alpha=alpha
        )


for j, PartType in enumerate(['PartType0', 'PartType0', 'PartType4']):

    if j == 0:
        velPlot = True
        gasSF = False
    elif j == 1:
        velPlot = False
        gasSF = False
    else:
        velPlot = False
        gasSF = False

    for linplot, IDs in enumerate([IDs_NormalSatelliteDMrich, IDs_MBCSatelliteDMrich, IDs_SBCSatelliteDMrich]):

        dFHalfStar = TNG.extractDF('SubhaloHalfmassRadType4')
        dFHalfGasRad = TNG.extractDF('SubhaloHalfmassRadType0')

        # -------------------------
        # SNAPS: t_entry - ~2 Gyr
        # -------------------------
        SnapsFirst = []
        for l, ID in enumerate(IDs):
            if ID == 603556 or ID == 602133:
                continue

            snapFirstEntry = PaperII.Snap_At_FirstEntry.loc[PaperII.SubfindID_99 == ID].values[0]
            if np.isnan(snapFirstEntry) or snapFirstEntry < 0:
                snapFirstEntry = 67

            snap = dfTime.loc[
                abs(dfTime.Age - dfTime.loc[dfTime.Snap == snapFirstEntry, 'Age'].values[0]) < 1.5,
                'Snap'
            ].values[-1]

            if snap < 17:
                snap = 17

            SnapsFirst.append(int(snap))

        xFirst, yFirst, yerrFirst, massFirst, xIDsFirst, yIDFirst, massIDsFirst, notIndex = \
            plot.MakeMedianAndIDs(
                SnapsFirst, IDs, rmin, rmax, nbins, PaperII,
                PartType=PartType, velPlot=velPlot, gasSF=False, Cond = 'None'
            )

        colors = plt.cm.Blues(np.linspace(0, 1, len(yIDFirst)))
        #plot.MakeLines(j, axs[j][linplot], yIDFirst, xIDsFirst, IDs, notIndex, colors)

        # -------------------------
        # SNAPS: t_entry
        # -------------------------
        Snaps = []
        for l, ID in enumerate(IDs):
            if ID == 603556 or ID == 602133:
                continue

            snapFirstEntry = PaperII.Snap_At_FirstEntry.loc[PaperII.SubfindID_99 == ID].values[0]
            if np.isnan(snapFirstEntry) or snapFirstEntry < 0:
                snapFirstEntry = 67

            Snaps.append(int(snapFirstEntry))

        xSecond, ySecond, yerrSecond, massSecond, xIDsSecond, yIDSecond, massIDsSecond, notIndex = \
            plot.MakeMedianAndIDs(
                Snaps, IDs, rmin, rmax, nbins, PaperII,
                PartType=PartType, velPlot=velPlot, gasSF=False, Cond = 'None'
            )

        colors = plt.cm.Greens(np.linspace(0, 1, len(yIDSecond)))
        plot.MakeLines(j, axs[j][linplot], yIDSecond, xIDsSecond, IDs, notIndex, colors)

        # -------------------------
        # SNAPS: t_entry + ~2 Gyr
        # -------------------------
        SnapsFinal = []
        for l, ID in enumerate(IDs):
            if ID == 603556 or ID == 602133:
                continue

            snapFirstEntry = PaperII.Snap_At_FirstEntry.loc[PaperII.SubfindID_99 == ID].values[0]
            if np.isnan(snapFirstEntry) or snapFirstEntry < 0:
                snapFirstEntry = 67

            if linplot == 1:
                snap = dfTime.loc[
                    abs(dfTime.Age - dfTime.loc[dfTime.Snap == snapFirstEntry, 'Age'].values[0]) < 2,
                    'Snap'
                ].values[0]
            else:
                snap = 99

            SnapsFinal.append(int(snap))

        xFinal, yFinal, yerrFinal, massFinal, xIDsFinal, yIDsFinal, massIDsFinal, notIndex = \
            plot.MakeMedianAndIDs(
                SnapsFinal, IDs, rmin, rmax, nbins, PaperII,
                PartType=PartType, velPlot=velPlot, gasSF=False, Cond = 'None'
            )

        colors = plt.cm.Reds(np.linspace(0, 1, len(yIDsFinal)))
        #plot.MakeLines(j, axs[j][linplot], yIDsFinal, xIDsFinal, IDs, notIndex, colors)

        # ==========================================================
        # PERFIS DE STAR-FORMING GAS APENAS PARA O PAINEL DO GÁS
        # ==========================================================
        if j == 1:
            xFirstSF, yFirstSF, yerrFirstSF, massFirstSF, xIDsFirstSF, yIDFirstSF, massIDsFirstSF, notIndexSF = \
                plot.MakeMedianAndIDs(
                    SnapsFirst, IDs, rmin, rmax, nbins, PaperII,
                    PartType='PartType0', velPlot=False, gasSF=True, Cond = 'SFgas'
                )

            xSecondSF, ySecondSF, yerrSecondSF, massSecondSF, xIDsSecondSF, yIDSecondSF, massIDsSecondSF, notIndexSF = \
                plot.MakeMedianAndIDs(
                    Snaps, IDs, rmin, rmax, nbins, PaperII,
                    PartType='PartType0', velPlot=False, gasSF=True, Cond = 'SFgas'
                )

            xFinalSF, yFinalSF, yerrFinalSF, massFinalSF, xIDsFinalSF, yIDsFinalSF, massIDsFinalSF, notIndexSF = \
                plot.MakeMedianAndIDs(
                    SnapsFinal, IDs, rmin, rmax, nbins, PaperII,
                    PartType='PartType0', velPlot=False, gasSF=True, Cond = 'SFgas'
                )

        # ==========================================================
        # PLOTS
        # ==========================================================
        if j == 1 or j == 2:
            # faixas do gás total / estrelas
            fill_profile(axs[j][linplot], xFirst,  yFirst,  yerrFirst,  color='tab:red',   alpha=0.3, multiply_r2=True)
            fill_profile(axs[j][linplot], xSecond, ySecond, yerrSecond, color='tab:green', alpha=0.3, multiply_r2=True)
            fill_profile(axs[j][linplot], xFinal,  yFinal,  yerrFinal,  color='tab:blue',  alpha=0.3, multiply_r2=True)

            # linhas sólidas: total gas / stars
            plot_profile(
                axs[j][linplot], xFirst, yFirst,
                color='red', lw=1.5, ls='-',
                multiply_r2=True,
                label=r'$t_\mathrm{entry} - 2 \, \mathrm{[Gyr]}$'
            )
            plot_profile(
                axs[j][linplot], xSecond, ySecond,
                color='green', lw=1.5, ls='-',
                multiply_r2=True,
                label=r'$t_\mathrm{entry}$'
            )
            plot_profile(
                axs[j][linplot], xFinal, yFinal,
                color='blue', lw=1.5, ls='-',
                multiply_r2=True,
                label=r'$t_\mathrm{entry} + 2 \, \mathrm{[Gyr]}$'
            )

            # linhas dashed: star-forming gas
            if j == 1:
                plot_profile(
                    axs[j][linplot], xFirstSF, yFirstSF,
                    color='red', lw=1.9, ls='--', alpha=1,
                    multiply_r2=True
                )
                plot_profile(
                    axs[j][linplot], xSecondSF, ySecondSF,
                    color='green', lw=1.9, ls='--', alpha=1,
                    multiply_r2=True
                )
                plot_profile(
                    axs[j][linplot], xFinalSF, yFinalSF,
                    color='blue', lw=1.9, ls='--', alpha=1,
                    multiply_r2=True
                )

        else:
            # painel de velocidade radial do gás
            fill_profile(axs[j][linplot], xFirst,  yFirst,  yerrFirst,  color='tab:red',   alpha=0.3, multiply_r2=False)
            fill_profile(axs[j][linplot], xSecond, ySecond, yerrSecond, color='tab:green', alpha=0.3, multiply_r2=False)
            fill_profile(axs[j][linplot], xFinal,  yFinal,  yerrFinal,  color='tab:blue',  alpha=0.3, multiply_r2=False)

            plot_profile(
                axs[j][linplot], xFirst, yFirst,
                color='red', lw=1.5, ls='-',
                multiply_r2=False,
                label=r'$t_\mathrm{entry} - 2 \, \mathrm{[Gyr]}$'
            )
            plot_profile(
                axs[j][linplot], xSecond, ySecond,
                color='green', lw=1.5, ls='-',
                multiply_r2=False,
                label=r'$t_\mathrm{entry}$'
            )
            plot_profile(
                axs[j][linplot], xFinal, yFinal,
                color='blue', lw=1.5, ls='-',
                multiply_r2=False,
                label=r'$t_\mathrm{entry} + 2 \, \mathrm{[Gyr]}$'
            )

        # ==========================================================
        # ESTILO DOS EIXOS
        # ==========================================================
        if j == 0:
            axs[j][linplot].set_title(Names[linplot], fontsize=1.1 * fontlabel)

        if j != 0:
            axs[j][linplot].set_yscale('log')

        axs[j][linplot].set_xscale('log')
        axs[j][linplot].set_xlim(0.2, 75)

        if j == 0:
            axs[j][linplot].set_ylim(-100, 10)
        elif j == 1:
            axs[j][linplot].set_ylim(5e5, 5e7)
        else:
            axs[j][linplot].set_ylim(5e5, 2e8)

        if j == 2:
            axs[j][linplot].tick_params(axis='x', labelsize=0.99 * fontlabel)

        if linplot == 0:
            axs[j][linplot].tick_params(axis='y', labelsize=0.99 * fontlabel)

            if j == 2:
                axs[j][linplot].set_ylabel(
                    r'$\rho_{\star} (r) r^2  \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$',
                    fontsize=fontlabel
                )
            elif j == 1:
                axs[j][linplot].set_ylabel(
                    r'$\rho_{\mathrm{gas}} (r) r^2  \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$',
                    fontsize=fontlabel
                )
            else:
                axs[j][linplot].set_ylabel(
                    r'$v_\mathrm{r, \; gas} (r) \, \, [\mathrm{km \, s}^{-1}]$',
                    fontsize=fontlabel
                )

        if j == 2:
            axs[j][linplot].set_xlabel(r'$r  \; [\mathrm{kpc}]$', fontsize=fontlabel)

        if j != 0:
            axs[j][linplot].yaxis.set_major_formatter(
                plot.FuncFormatter(plot.format_func_loglog)
            )

        axs[j][linplot].xaxis.set_major_formatter(
            plot.FuncFormatter(plot.format_func_loglog)
        )

# ==========================================================
# LEGENDAS
# ==========================================================
time_handles = [
    Line2D([0], [0], color='red',   lw=1.5, ls='-', label=r'$t_\mathrm{entry} - 2 \, \mathrm{[Gyr]}$'),
    Line2D([0], [0], color='green', lw=1.5, ls='-', label=r'$t_\mathrm{entry}$'),
    Line2D([0], [0], color='blue',  lw=1.5, ls='-', label=r'$t_\mathrm{entry} + 2 \, \mathrm{[Gyr]}$')
]

style_handles = [
    Line2D([0], [0], color='black', lw=1.5, ls='-',  label='Total gas'),
    Line2D([0], [0], color='black', lw=1.2, ls='--', label='Star-forming gas')
]

leg1 = axs[0][0].legend(handles=time_handles, fontsize=0.9 * fontlabel, framealpha=0.4, loc='lower right')
axs[0][0].add_artist(leg1)

axs[1][2].legend(handles=style_handles, fontsize=0.75 * fontlabel, framealpha=0.4, loc='upper right')

plt.savefig(os.getenv("HOME") + '/TNG_Analyzes/Figs/ProfilesTest.pdf', bbox_inches='tight')
# plt.ylim(- 0.99, 0.99)

#%%

import numpy as np
import pandas as pd

def add_unnormalized_decreases_from_snaps(PaperII, dfTime):
    """
    Reconstruct unnormalized cumulative relative size changes from the
    time-normalized quantities, using SnapLostGas and Snap_At_FirstEntry
    plus dfTime to convert snapshots into lookback times.

    Assumes:
      Decrease_Entry_To_NoGas_Norm_Delta = (ΔR/R)_entry_to_nogas / Δt_entry_to_nogas
      Decrease_NoGas_To_Final_Norm_Delta = (ΔR/R)_nogas_to_final / Δt_nogas_to_final

    Returns:
      copy of PaperII with new columns added.
    """

    df = PaperII.copy()
    dftime = dfTime.copy()

    # --- basic checks
    required_paper = [
        "Snap_At_FirstEntry",
        "SnapLostGas",
        "Decrease_Entry_To_NoGas_Norm_Delta",
        "Decrease_NoGas_To_Final_Norm_Delta",
    ]
    missing_paper = [c for c in required_paper if c not in df.columns]
    if missing_paper:
        raise KeyError(f"Missing columns in PaperII: {missing_paper}")

    required_time = ["Snap", "LBTime"]
    missing_time = [c for c in required_time if c not in dftime.columns]
    if missing_time:
        raise KeyError(f"Missing columns in dfTime: {missing_time}")

    # --- keep only what is needed and ensure unique Snap mapping
    dftime = dftime[["Snap", "LBTime"]].drop_duplicates(subset="Snap").copy()
    dftime["Snap"] = pd.to_numeric(dftime["Snap"], errors="coerce").astype("Int64")
    dftime["LBTime"] = pd.to_numeric(dftime["LBTime"], errors="coerce")

    # --- convert snap columns to integer-like type
    df["Snap_At_FirstEntry"] = pd.to_numeric(df["Snap_At_FirstEntry"], errors="coerce").astype("Int64")
    df["SnapLostGas"] = pd.to_numeric(df["SnapLostGas"], errors="coerce").astype("Int64")

    # --- map snaps to lookback times
    snap_to_lb = dftime.set_index("Snap")["LBTime"]

    df["LBT_FirstEntry_Gyr"] = df["Snap_At_FirstEntry"].map(snap_to_lb)
    df["LBT_LostGas_Gyr"] = df["SnapLostGas"].map(snap_to_lb)

    # z=0 snapshot in TNG is 99, with LBTime ~ 0
    lb_z0 = snap_to_lb.loc[99] if 99 in snap_to_lb.index else 0.0
    df["LBT_z0_Gyr"] = lb_z0

    # --- durations of the two phases
    # entry -> gas loss
    df["dt_Entry_To_NoGas_Gyr"] = df["LBT_FirstEntry_Gyr"] - df["LBT_LostGas_Gyr"]

    # gas loss -> final (z=0)
    df["dt_NoGas_To_Final_Gyr"] = df["LBT_LostGas_Gyr"] - df["LBT_z0_Gyr"]

    # --- recover unnormalized cumulative relative size changes
    xnorm = pd.to_numeric(df["Decrease_Entry_To_NoGas_Norm_Delta"], errors="coerce")
    ynorm = pd.to_numeric(df["Decrease_NoGas_To_Final_Norm_Delta"], errors="coerce")

    df["Decrease_Entry_To_NoGas"] = xnorm * df["dt_Entry_To_NoGas_Gyr"]
    df["Decrease_NoGas_To_Final"] = ynorm * df["dt_NoGas_To_Final_Gyr"]

    # --- invalidate problematic cases
    bad1 = (
        df["dt_Entry_To_NoGas_Gyr"].isna() |
        (df["dt_Entry_To_NoGas_Gyr"] <= 0)
    )
    bad2 = (
        df["dt_NoGas_To_Final_Gyr"].isna() |
        (df["dt_NoGas_To_Final_Gyr"] <= 0)
    )

    df.loc[bad1, "Decrease_Entry_To_NoGas"] = np.nan
    df.loc[bad2, "Decrease_NoGas_To_Final"] = np.nan

    return df

#%%
PaperII_new = add_unnormalized_decreases_from_snaps(PaperII, dfTime)

cols_show = [
    "SubfindID_99",
    "Snap_At_FirstEntry",
    "SnapLostGas",
    "LBT_FirstEntry_Gyr",
    "LBT_LostGas_Gyr",
    "dt_Entry_To_NoGas_Gyr",
    "dt_NoGas_To_Final_Gyr",
    "Decrease_Entry_To_NoGas_Norm_Delta",
    "Decrease_NoGas_To_Final_Norm_Delta",
    "Decrease_Entry_To_NoGas",
    "Decrease_NoGas_To_Final",
]

print(PaperII_new[cols_show].head(10))

#%%

MBCIDs = np.array([   232,    261,    281,    300,    319,    333,  63993,  64081,
        64129,  96941, 117464, 144008, 144098, 167499, 185005, 185058,
       208883, 220626, 242863, 253897, 253905, 253965, 264911, 264972,
       275601, 282802, 282807, 289444, 307510, 319738, 319743, 377662,
       386293, 394628, 404834, 421566, 421567, 422763, 425726, 432119,
       457435, 467420, 482157, 502998, 530853, 536657, 545439, 549748,
       549750, 558069, 571075, 571910, 588180, 647769])
SBCIDs = np.array([ 63990,  64002,  96853, 117357, 229992, 229996, 294887, 294895,
       358627, 422770, 428191, 500583, 516761, 602132, 602133, 603556])

#%%
fontlabel = 15
nboot = 5000
nbins = 25
plt.rcParams.update({'figure.figsize': (4*2, 3*2)})
fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
axs = gs.subplots(sharex='col', sharey='row')
Names = [r'Compacts$_\mathrm{MB}$', r'Compacts$_\mathrm{SB}$']
rmin = 0.1
rmax = 35
for linplot, IDs in enumerate([ MBCIDs, SBCIDs]):
    
    dFHalfStar = TNG.extractDF('SubhaloHalfmassRadType4')
    dFHalfGasRad = TNG.extractDF('SubhaloHalfmassRadType0')
    yIDsEntry = np.array([])
    massIDsEntry = np.array([])
    xIDsEntry = np.array([])
    notIndex = np.array([])
    for l, ID in enumerate(IDs):
        print(ID)
        if ID == 603556 or ID == 602133 :
            continue
        snap = PaperII.Snap_At_FirstEntry.loc[PaperII.SubfindID_99 == ID].values[0]
        if np.isnan(snap):
            continue
        snap = int(snap)
        print('snap: ', snap)
        yrad, rad, mass = TNG.MakeDensityProfileMean(snap, ID, rmin, rmax, nbins)
        
        if len(yrad) == 1 or (ID == 603556 or ID == 602133):
            notIndex = np.append(notIndex, l)
            continue
        if l == 0 or len(yIDsEntry) == 0:
            yIDsEntry = np.append(yIDsEntry, yrad)
            xIDsEntry = np.append(xIDsEntry, rad)
            massIDsEntry = np.append(massIDsEntry, mass)
    
        else:
            yIDsEntry = np.vstack((yIDsEntry, yrad))
            massIDsEntry = np.vstack((massIDsEntry, mass))
            xIDsEntry = np.vstack((xIDsEntry, rad))
           
    
        Rvalues = xIDsEntry.T
        Values = yIDsEntry.T
        Masses = massIDsEntry.T

    x = np.array([])
    y = np.array([])
    mass = np.array([])
    
    
    
    if len(Values) > 0:
        if len(Values.shape) > 1:
            for k, value in enumerate(Values):
                x = np.append(x, np.nanmedian(Rvalues[k]))
                y = np.append(y, np.nanmedian(value))
                mass = np.append(mass, np.nanmedian(Masses[k]))
        else:
            x = Rvalues
            y = Values
            mass = Masses

    else:
        x = np.nan
        y = np.nan
        mass = np.nan
            
    xEntry = x
    yEntry = y
    massEntry = mass
      
    yIDs = np.array([])
    xIDs = np.array([])
    massIDs = np.array([])


    for l, ID in enumerate(IDs):  
     
        snap = 99
        yrad, rad, mass = TNG.MakeDensityProfileMean(snap, ID, rmin, rmax, nbins)
        if len(yrad) == 1 or (ID == 603556 or ID == 602133):
            notIndex = np.append(notIndex, l)
            continue
        if l == 0 or len(yIDs) == 0:
            yIDs = np.append(yIDs, yrad)
            xIDs = np.append(xIDs, rad)
            massIDs = np.append(massIDs, mass)

    
        else:
            yIDs = np.vstack((yIDs, yrad))
            xIDs = np.vstack((xIDs, rad))
            massIDs = np.vstack((massIDs, mass))

        Rvalues = xIDs.T
        Values = yIDs.T
        Masses = massIDs.T

    x = np.array([])
    y = np.array([])
    mass = np.array([])
    
    
    
    if len(Values) > 0:
        if len(Values.shape) > 1:
            for k, value in enumerate(Values):
                x = np.append(x, np.nanmedian(Rvalues[k]))
                y = np.append(y, np.nanmedian(value))
                mass = np.append(mass, np.nanmedian(Masses[k]))
        else:
            x = Rvalues
            y = Values
            mass = Masses

    else:
        x = np.nan
        y = np.nan
        mass = np.nan
            
    x99 = x
    y99 = y
    mass99 = mass



    colors = plt.cm.OrRd(np.linspace(0,1,len(yIDsEntry)))
    k = 0
    for l, ID in enumerate(IDs):
        if l in notIndex:
            continue
        value = yIDsEntry[k]
        axs[0][linplot].plot(xIDsEntry[k][massIDsEntry[k] > 0] , np.nancumsum(massIDsEntry[k][massIDsEntry[k] > 0])  , 
                             lw = 0.55,  alpha = 0.2, color = colors[k])

        axs[1][linplot].plot(xIDsEntry[k][value > 0] , value[value > 0] * xIDsEntry[k][value > 0]**2., 
                             lw = 0.55,  alpha = 0.2,  color = colors[k])
        
        k = k+ 1

     
    colors = plt.cm.Purples(np.linspace(0,1,len(yIDs)))
    k = 0
    for l, ID in enumerate(IDs):
        if l in notIndex:
            continue
        value = yIDs[k]
        axs[0][linplot].plot(xIDs[k][massIDs[k] > 0] , np.nancumsum(massIDs[k][massIDs[k] > 0]) , 
                             lw = 0.55, alpha = 0.25,  color = colors[k])   
        axs[1][linplot].plot(xIDs[k][value > 0] , value[value > 0] * xIDs[k][value > 0]**2., 
                             lw = 0.55,alpha = 0.2,  color = colors[k])  
        
        k = k + 1

    axs[1][linplot].plot(xEntry , yEntry*xEntry**2, color = 'firebrick', lw = 2.1, label = r'$z_\mathrm{entry}$')
    axs[1][linplot].plot(x99 , y99*x99**2, color = 'midnightblue', lw = 2.1, label = r'$z = 0$')   
  
    axs[0][linplot].plot(xEntry[massEntry > 0] , np.nancumsum(massEntry[massEntry > 0]) , color = 'firebrick', lw = 2.1, label = r'$z_\mathrm{entry}$')
    axs[0][linplot].plot(x99[mass99 > 0] , np.nancumsum(mass99[mass99 > 0]), color = 'midnightblue', lw = 2.1, label = r'$z = 0$')     
  
    
    axs[0][linplot].set_title(Names[linplot], fontsize=1.1*fontlabel)
    axs[0][linplot].set_xscale('log')
    axs[0][linplot].set_yscale('log')
    axs[1][linplot].set_yscale('log')
    
    axs[0][linplot].set_xlim(0.2, 75)
    axs[1][linplot].set_xlim(0.2, 75)
    axs[0][linplot].set_ylim(2e6, 6e9)
    axs[1][linplot].set_ylim(2e4, 2e8)
    
    axs[1][linplot].tick_params(axis='x', labelsize=0.99*fontlabel)

    axs[0][linplot].axhline(np.nansum(massEntry[massEntry > 0]) / 2., ls = '--', color = 'firebrick')


    if linplot == 0:
        axs[1][linplot].tick_params(axis='y', labelsize=0.99*fontlabel)
        axs[0][linplot].set_ylabel(r'$M_\star (<r) \; [\mathrm{M}_\odot]$', fontsize=fontlabel)
        axs[1][linplot].set_ylabel(r'$\rho_{\star} (r) r^2  \; \, \, [\mathrm{M_\odot  \; kpc^{-1}}]$', fontsize=fontlabel)
    axs[1][linplot].set_xlabel(r'$r  \; [\mathrm{kpc}]$', fontsize=fontlabel)
    
    
    axs[1][linplot].yaxis.set_major_formatter(
        plot.FuncFormatter(plot.format_func_loglog))
    axs[0][linplot].yaxis.set_major_formatter(
        plot.FuncFormatter(plot.format_func_loglog))
    axs[1][linplot].xaxis.set_major_formatter(
        plot.FuncFormatter(plot.format_func_loglog))
    axs[0][0].legend(handlelength = 1.2 ,fontsize=0.9*fontlabel, framealpha = 0.4)

plt.savefig(os.getenv("HOME")+'/TNG_Analyzes/Figs/Profiles.pdf', bbox_inches='tight')    
#plt.ylim(- 0.99, 0.99)

#%%

fig = plt.figure()
plt.rcParams.update({'figure.figsize': (5, 5)})
gs = fig.add_gridspec(2, 1, hspace=0, wspace=0)
axs = gs.subplots(sharex='col', sharey='row')
cmap = plt.cm.get_cmap('rainbow_r')

Zsun =  0.0127 
SBCSatellite = TNG.extractPopulation('SBCSatellite')
MBCSatellite = TNG.extractPopulation('MBCSatellite')
NormalSatellite = TNG.extractPopulation('NormalSatellite')

XSBCMax = np.array([v for v in SBCSatellite.MaxStellarMassInRad[(SBCSatellite.MaxStellarMassInRad != SBCSatellite.logMstarRad_99)].values])
YSBC = np.log10( np.array([v for v in SBCSatellite.StarMetallicity_99[(SBCSatellite.MaxStellarMassInRad != SBCSatellite.logMstarRad_99)].values])/ Zsun  ) - 0.75
XSBC = np.array([v for v in SBCSatellite.logMstarRad_99[(SBCSatellite.MaxStellarMassInRad != SBCSatellite.logMstarRad_99)].values])
YSBCMax = np.log10(np.array([v for v in SBCSatellite.MaxStarMetallicity[(SBCSatellite.MaxStellarMassInRad != SBCSatellite.logMstarRad_99)].values]) / Zsun) - 0.75


XMBCMax = np.array([v for v in MBCSatellite.MaxStellarMassInRad[(MBCSatellite.MaxStellarMassInRad != MBCSatellite.logMstarRad_99)].values])
YMBC = np.log10(np.array([v for v in MBCSatellite.StarMetallicity_99[(MBCSatellite.MaxStellarMassInRad != MBCSatellite.logMstarRad_99)].values]) / Zsun) - 0.75
XMBC = np.array([v for v in MBCSatellite.logMstarRad_99[(MBCSatellite.MaxStellarMassInRad != MBCSatellite.logMstarRad_99)].values])
YMBCMax = np.log10(np.array([v for v in MBCSatellite.MaxStarMetallicity[(MBCSatellite.MaxStellarMassInRad != MBCSatellite.logMstarRad_99)].values]) / Zsun) - 0.75

XNormalMax = np.array([v for v in NormalSatellite.MaxStellarMassInRad[(NormalSatellite.MaxStellarMassInRad != NormalSatellite.logMstarRad_99)].values])
YNormal = np.log10(np.array([v for v in NormalSatellite.StarMetallicity_99[(NormalSatellite.MaxStellarMassInRad != NormalSatellite.logMstarRad_99)].values]) / Zsun) - 0.75
XNormal = np.array([v for v in NormalSatellite.logMstarRad_99[(NormalSatellite.MaxStellarMassInRad != NormalSatellite.logMstarRad_99)].values])
YNormalMax = np.log10(np.array([v for v in NormalSatellite.MaxStarMetallicity[(NormalSatellite.MaxStellarMassInRad != NormalSatellite.logMstarRad_99)].values]) / Zsun) - 0.75



for i, v in enumerate(XSBC):
    if abs(XSBC[i] - XSBCMax[i]) > 0.05:
        prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.35", color = 'forestgreen' ,
            shrinkA=0,shrinkB=0)
        #axs[1].arrow(XSBCMax[i], YSBCMax[i], XSBC[i] - XSBCMax[i] , YSBC[i] - YSBCMax[i] , color = 'forestgreen', shape='full', head_starts_at_zero=False, 
        #             lw = .2, width = 0.0002, head_width = 0.0025, head_length = 0.05)
        axs[1].annotate("", xy=(XSBC[i],YSBC[i]), xytext=(XSBCMax[i], YSBCMax[i]),  color = 'forestgreen',  arrowprops=prop, zorder=0)
        
        #YSBC[i] - YSBCMax[i] - 0.00
for i, v in enumerate(XMBC):
     if abs(XMBC[i] - XMBCMax[i]) > 0.05:
        prop = dict(arrowstyle="-|>,head_width=0.2,head_length=0.35", color = 'royalblue' ,
            shrinkA=0,shrinkB=0)
        #axs[0].arrow(XMBCMax[i], YMBCMax[i], XMBC[i] - XMBCMax[i] ,  YMBC[i] - YMBCMax[i], color = 'royalblue', shape='full', head_starts_at_zero=False, 
        #             lw = .2, width = 0.0002, head_width = 0.0025, head_length = 0.05) 
        axs[0].annotate("", xy=(XMBC[i],YMBC[i]), xytext=(XMBCMax[i], YMBCMax[i]),  color = 'royalblue',  arrowprops=prop, zorder=0)



# Define bins for x
num_bins = 10
bins = np.linspace(min(XNormal), max(XNormal), num_bins + 1)

# Calculate statistics for each bin
bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Center of each bin
median_y = []
percentile_25 = []
percentile_75 = []

for i in range(num_bins):
    bin_y = YNormal[(XNormal >= bins[i]) & (XNormal < bins[i + 1])]
    median_y.append(np.median(bin_y))
    percentile_25.append(np.percentile(bin_y, 25))
    percentile_75.append(np.percentile(bin_y, 75))

# Convert to numpy arrays
median_y = np.array(median_y)
percentile_25 = np.array(percentile_25)
percentile_75 = np.array(percentile_75)

# Median line
axs[1].plot(bin_centers, median_y, color='darkorange', label='Median', lw=2)

# Shaded region (IQR)
axs[1].fill_between(bin_centers, percentile_25, percentile_75, color='tab:orange', alpha=0.3)
# Median line
axs[0].plot(bin_centers, median_y, color='darkorange', label='Median', lw=2)

# Shaded region (IQR)
axs[0].fill_between(bin_centers, percentile_25, percentile_75, color='tab:orange', alpha=0.3)




sc = axs[1].scatter(XSBCMax,
            YSBCMax ,
            c=SBCSatellite.DMFrac_99[(SBCSatellite.MaxStellarMassInRad != SBCSatellite.logMstarRad_99)].values, 
            edgecolor='k', alpha=.4, 
            lw = 0.9,
            ls = 'dashed',

            marker='D', 
            s=3.1*8,
            cmap = cmap)

sc = axs[0].scatter(XMBCMax,
            YMBCMax ,
            c=MBCSatellite.DMFrac_99[(MBCSatellite.MaxStellarMassInRad != MBCSatellite.logMstarRad_99)].values, 
            edgecolor='k', alpha=.4, 
            lw = 0.9,
            ls = 'dashed',
            marker='o', 
            s=3.1*8,
            cmap = cmap)


sc = axs[1].scatter(XSBC,
            YSBC ,
            c=SBCSatellite.DMFrac_99[(SBCSatellite.MaxStellarMassInRad != SBCSatellite.logMstarRad_99)].values, 
            edgecolor='k', alpha=.8, 
            lw = 0.9,
            marker='D', 
            s=6*8,
            cmap = cmap)

sc = axs[0].scatter(XMBC,
            YMBC ,
            c=MBCSatellite.DMFrac_99[(MBCSatellite.MaxStellarMassInRad != MBCSatellite.logMstarRad_99)].values, 
            edgecolor='k', alpha=.8, 
            lw = 0.9,
            marker='o', 
            s=6*8,
            cmap = cmap)


cb = fig.colorbar(sc,  ax=axs.ravel().tolist(), pad=0.02,  aspect=60)
cb.set_label(r'$(M_\mathrm{DM}/ M)_{z = 0}$', fontsize = 11)
cb.ax.tick_params(labelsize=0.99*11)

plt.xlabel('$\log(M_\star/\mathrm{M}_\odot)$', fontsize = 11)
axs[0].set_ylabel(r'$\log( Z_\star / Z_\odot)$', fontsize = 11)
axs[1].set_ylabel(r'$\log( Z_\star / Z_\odot)$', fontsize = 11)
axs[0].set_ylim(np.log10(0.0075 / Zsun) - 0.75, np.log10(0.055 / Zsun) - 0.75)
axs[1].set_ylim(np.log10(0.0075 / Zsun) - 0.75, np.log10(0.055 / Zsun) - 0.75)
axs[1].tick_params(labelsize=0.99*11)
axs[0].tick_params(labelsize=0.99*11)


axs[0].text(9.5, -0.1 - 0.75,'Compacts$_\mathrm{MB}$', fontsize = 11,   color = 'royalblue')
axs[1].text(9.5, -0.1 - 0.75, 'Compacts$_\mathrm{SB}$', fontsize = 11,   color ='forestgreen')

plt.savefig(os.getenv('HOME')+'/TNG_Analyzes/Figs/TNG50/PaperII/PlotScatter/Zmetallicty.pdf', bbox_inches='tight',)


#%%

import matplotlib.patheffects as pe
fig = plt.figure()
plt.rcParams.update({'figure.figsize': (5, 5)})
gs = fig.add_gridspec(2, 1, hspace=0, wspace=0)
axs = gs.subplots(sharex='col', sharey='row')

Zsun =  0.0127 
SBCSatellite = TNG.extractPopulation('SBCSatellite', dfName = 'PaperII')
MBCSatellite = TNG.extractPopulation('MBCSatellite',  dfName = 'PaperII')
NormalSatellite = TNG.extractPopulation('NormalSatellite',  dfName = 'PaperII')

XSBCMax = np.array([v for v in SBCSatellite.MaxStellarMassInRad[(SBCSatellite.MaxStellarMassInRad != SBCSatellite.logMstarRad_99)].values])
YSBC = np.log10( np.array([v for v in SBCSatellite.StarMetallicity_99[(SBCSatellite.MaxStellarMassInRad != SBCSatellite.logMstarRad_99)].values])/ Zsun  ) - 0.75
XSBC = np.array([v for v in SBCSatellite.logMstarRad_99[(SBCSatellite.MaxStellarMassInRad != SBCSatellite.logMstarRad_99)].values])
YSBCMax = np.log10(np.array([v for v in SBCSatellite.MaxStarMetallicity[(SBCSatellite.MaxStellarMassInRad != SBCSatellite.logMstarRad_99)].values]) / Zsun) - 0.75
DMSBC = np.array([v for v in SBCSatellite.DMFrac_99[(SBCSatellite.MaxStellarMassInRad != SBCSatellite.logMstarRad_99)].values])


XMBCMax = np.array([v for v in MBCSatellite.MaxStellarMassInRad[(MBCSatellite.MaxStellarMassInRad != MBCSatellite.logMstarRad_99)].values])
YMBC = np.log10(np.array([v for v in MBCSatellite.StarMetallicity_99[(MBCSatellite.MaxStellarMassInRad != MBCSatellite.logMstarRad_99)].values]) / Zsun) - 0.75
XMBC = np.array([v for v in MBCSatellite.logMstarRad_99[(MBCSatellite.MaxStellarMassInRad != MBCSatellite.logMstarRad_99)].values])
YMBCMax = np.log10(np.array([v for v in MBCSatellite.MaxStarMetallicity[(MBCSatellite.MaxStellarMassInRad != MBCSatellite.logMstarRad_99)].values]) / Zsun) - 0.75
DMMBC = np.array([v for v in MBCSatellite.DMFrac_99[(MBCSatellite.MaxStellarMassInRad != MBCSatellite.logMstarRad_99)].values])

XNormalMax = np.array([v for v in NormalSatellite.MaxStellarMassInRad[(NormalSatellite.MaxStellarMassInRad != NormalSatellite.logMstarRad_99)].values])
YNormal = np.log10(np.array([v for v in NormalSatellite.StarMetallicity_99[(NormalSatellite.MaxStellarMassInRad != NormalSatellite.logMstarRad_99)].values]) / Zsun) - 0.75
XNormal = np.array([v for v in NormalSatellite.logMstarRad_99[(NormalSatellite.MaxStellarMassInRad != NormalSatellite.logMstarRad_99)].values])
YNormalMax = np.log10(np.array([v for v in NormalSatellite.MaxStarMetallicity[(NormalSatellite.MaxStellarMassInRad != NormalSatellite.logMstarRad_99)].values]) / Zsun) - 0.75

YNormal_50 = np.array([v for v in NormalSatellite.StarMetallicity_50.values]) - 0.75
XNormal_50 = np.array([v for v in NormalSatellite.logMstarRad_50.values])


def draw_median_arrow(ax, x0, y0, x1, y1, mask, color):
    if np.sum(mask) == 0:
        return

    x0m = np.median(x0[mask])
    y0m = np.median(y0[mask])
    x1m = np.median(x1[mask])
    y1m = np.median(y1[mask])

    
    prop = dict(
        arrowstyle="-|>,head_width=0.35,head_length=0.55",
        facecolor=color,
        edgecolor=color,
        lw=1.8,
        shrinkA=0,
        shrinkB=0,
        alpha=1.0,
        mutation_scale=16,
    )
    
    ann = ax.annotate(
        "",
        xy=(x1m, y1m),
        xytext=(x0m, y0m),
        arrowprops=prop,
        zorder=30,
    )
    
    ann.arrow_patch.set_path_effects([
        pe.Stroke(linewidth=3.2, foreground='white'),
        pe.Normal()
    ])
    ann.arrow_patch.set_zorder(30)

poor_sbc = DMSBC < 0.7
rich_sbc = DMSBC > 0.7
poor_mbc = DMMBC < 0.7
rich_mbc = DMMBC > 0.7

# ---------------------------
# Robust median tracks for Normals
# replace the current "Define bins for x" + "HIGHER REDSHIFT" blocks by this
# ---------------------------

def get_binned_median_track(x, y, bins, min_count=5, qlo=25, qhi=75):
    x = np.asarray(x)
    y = np.asarray(y)

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    idx = np.digitize(x, bins) - 1

    x_track = []
    y_med = []
    y_lo  = []
    y_hi  = []

    for i in range(len(bins) - 1):
        m = idx == i
        if np.sum(m) < min_count:
            continue

        xb = x[m]
        yb = y[m]

        x_track.append(np.median(xb))          # more robust than fixed bin center
        y_med.append(np.median(yb))
        y_lo.append(np.percentile(yb, qlo))
        y_hi.append(np.percentile(yb, qhi))

    return np.array(x_track), np.array(y_med), np.array(y_lo), np.array(y_hi)


# robust masks
mask_norm_99 = (
    (NormalSatellite.MaxStellarMassInRad != NormalSatellite.logMstarRad_99).values &
    np.isfinite(NormalSatellite.logMstarRad_99.values) &
    np.isfinite(NormalSatellite.StarMetallicity_99.values) &
    (NormalSatellite.StarMetallicity_99.values > 0)
)

mask_norm_50 = (
    np.isfinite(NormalSatellite.logMstarRad_50.values) &
    np.isfinite(NormalSatellite.StarMetallicity_50.values) 
)

# redefine Normal tracks consistently
XNormal = NormalSatellite.logMstarRad_99.values[mask_norm_99]
YNormal = np.log10(NormalSatellite.StarMetallicity_99.values[mask_norm_99] / Zsun) - 0.75

XNormal_50 = NormalSatellite.logMstarRad_50.values
YNormal_50 = NormalSatellite.StarMetallicity_50.values - 0.75

# same bins for z=0 and z~1
num_bins = 20
xmin = np.nanmin(np.concatenate([XNormal, XNormal_50]))
xmax = np.nanmax(np.concatenate([XNormal, XNormal_50]))
bins = np.linspace(8.3, 10.1, num_bins + 1)

xmed_0, ymed_0, ylo_0, yhi_0 = get_binned_median_track(
    XNormal, YNormal, bins, min_count=5, qlo=25, qhi=75
)

xmed_50, ymed_50, ylo_50, yhi_50 = get_binned_median_track(
    XNormal_50, YNormal_50, bins, min_count=5, qlo=25, qhi=75
)

# z = 0
axs[1].plot(xmed_0, ymed_0, color='darkorange', lw=1.4, label='Normal population at $z=0$')
axs[1].fill_between(xmed_0, ylo_0, yhi_0, color='tab:orange', alpha=0.25)

axs[0].plot(xmed_0, ymed_0, color='darkorange', lw=1.4, label='Normal population at $z=0$')
axs[0].fill_between(xmed_0, ylo_0, yhi_0, color='tab:orange', alpha=0.25)

# z ~ 1 (Max)
axs[1].plot(xmed_50, ymed_50, color='darkorange', ls='--', lw=1.4, label='Normal population at $z\\sim1$')
axs[0].plot(xmed_50, ymed_50, color='darkorange', ls='--', lw=1.4, label='Normal population at $z\\sim1$')

# optional: if you also want a lighter shaded region for z~1, uncomment below
axs[1].fill_between(xmed_50, ylo_50, yhi_50, color='tab:orange', alpha=0.10)
axs[0].fill_between(xmed_50, ylo_50, yhi_50, color='tab:orange', alpha=0.10)


# MAXIMUM (smaller / lighter markers)
axs[1].scatter(XSBCMax[poor_sbc],
            YSBCMax[poor_sbc],
            color='red',
            edgecolor='k', alpha=.4,
            lw = 0.9,
            marker='D',
            ls = 'solid',
            s=2.8*6.5)

axs[1].scatter(XSBCMax[rich_sbc],
            YSBCMax[rich_sbc],
            color='royalblue',
            edgecolor='k', alpha=.4,
            lw = 0.9,
            marker='D',
            ls = 'solid',
            s=2.8*6.5)

axs[0].scatter(XMBCMax[poor_mbc],
            YMBCMax[poor_mbc],
            color='red',
            edgecolor='k', alpha=.4,
            lw = 0.9,
            marker='o',
            ls = 'solid',
            s=2.8*6.5)

axs[0].scatter(XMBCMax[rich_mbc],
            YMBCMax[rich_mbc],
            color='royalblue',
            edgecolor='k', alpha=.4,
            lw = 0.9,
            marker='o',
            ls = 'solid',
            s=2.8*6.5)

# z = 0 (larger / darker markers)
axs[1].scatter(XSBC[poor_sbc],
            YSBC[poor_sbc],
            color='red',
            edgecolor='k', alpha=.8,
            lw = 0.9,
            marker='D',
            s=6*6.5)

axs[1].scatter(XSBC[rich_sbc],
            YSBC[rich_sbc],
            color='royalblue',
            edgecolor='k', alpha=.8,
            lw = 0.9,
            marker='D',
            s=6*6.5)

axs[0].scatter(XMBC[poor_mbc],
            YMBC[poor_mbc],
            color='red',
            edgecolor='k', alpha=.8,
            lw = 0.9,
            marker='o',
            s=6*6.5)

axs[0].scatter(XMBC[rich_mbc],
            YMBC[rich_mbc],
            color='royalblue',
            edgecolor='k', alpha=.8,
            lw = 0.9,
            marker='o',
            s=6*6.5)



# median arrows
# SB panel
if np.sum(poor_sbc) > 0:
    draw_median_arrow(axs[1], XSBCMax, YSBCMax, XSBC, YSBC, poor_sbc, 'forestgreen')
# if np.sum(rich_sbc) > 0:
#     draw_median_arrow(axs[1], XSBCMax, YSBCMax, XSBC, YSBC, rich_sbc, 'royalblue')

# MB panel
if np.sum(poor_mbc) > 0:
    draw_median_arrow(axs[0], XMBCMax, YMBCMax, XMBC, YMBC, poor_mbc, 'royalblue')
# if np.sum(rich_mbc) > 0:
#     draw_median_arrow(axs[0], XMBCMax, YMBCMax, XMBC, YMBC, rich_mbc, 'royalblue')



legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='k', markersize=6, label='$f_\mathrm{DM} < 0.7$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue', markeredgecolor='k', markersize=6, label='$f_\mathrm{DM} > 0.7$'),
    Line2D([0], [0], color='darkorange', lw=2, label='Normals at $z=0$'),
    Line2D([0], [0], color='k', lw=1.2, ls='--', label='$z = 1$'),
    Line2D([0], [0], color='k', lw=1.2, ls='-', label='$z=0$'),
]
axs[0].legend(handles=legend_handles, fontsize=8.5, loc='upper right', frameon=True)

plt.xlabel('$\log(M_\star/\mathrm{M}_\odot)$', fontsize = 11)
axs[0].set_ylabel(r'$\log( Z_\star / Z_\odot)$', fontsize = 11)
axs[1].set_ylabel(r'$\log( Z_\star / Z_\odot)$', fontsize = 11)
axs[0].set_ylim(np.log10(0.0075 / Zsun) - 0.75, np.log10(0.055 / Zsun) - 0.75)
axs[1].set_ylim(np.log10(0.0075 / Zsun) - 0.75, np.log10(0.055 / Zsun) - 0.75)
axs[1].tick_params(labelsize=0.99*11)
axs[0].tick_params(labelsize=0.99*11)

axs[0].set_xlim(8.3, 10.1)
axs[1].set_xlim(8.3, 10.1)
axs[0].text(9.5, -0.1 - 0.75,'Compacts$_\mathrm{MB}$', fontsize = 11,   color = 'royalblue')
axs[1].text(9.5, -0.1 - 0.75, 'Compacts$_\mathrm{SB}$', fontsize = 11,   color ='forestgreen')

plt.savefig(os.getenv('HOME')+'/TNG_Analyzes/Figs/TNG50/PaperII/PlotScatter/Zmetallicty.pdf', bbox_inches='tight',)

#%%

import numpy as np
import pandas as pd

def get_population_values(population, key, dfName='PaperII', Name='Name'):
    Sample = TNG.extractPopulation(population, dfName=dfName, Name=Name)
    values = np.asarray(Sample[key].values, dtype=float)
    values = values[np.isfinite(values)]

    if key == 'z_At_FinalEntry':
        values = values[values >= 0]

    return values


def compare_Sample(key, populations, dfName = 'PaperII', Name = 'Name', RankSums = False, Moodtest = False, KStest = False):

    for i, population in enumerate(populations):
        print(population[0], ' and ',population[1])
  
        if key == 'z_At_FinalEntry':
            Values1 = get_population_values(population[0], key, dfName=dfName, Name=Name)
            Values2 = get_population_values(population[1], key, dfName=dfName, Name=Name)


            MATH.TestPermutation(Values1[Values1 >= 0],Values2[Values2 >= 0], roundmedian = 3, RankSums = RankSums, Moodtest = Moodtest, KStest = KStest)
            
        else:
            Values1 = get_population_values(population[0], key, dfName=dfName, Name=Name)

            Values2 = get_population_values(population[1], key, dfName=dfName, Name=Name)
            
            MATH.TestPermutation(Values1,Values2, roundmedian = 3, RankSums = RankSums, Moodtest = Moodtest, KStest = KStest)
 

def bootstrap_statistic(values, func=np.median, n_boot=5000, random_state=42):
    """
    Bootstrap simples da estatística desejada.
    """
    rng = np.random.default_rng(random_state)
    n = len(values)

    if n == 0:
        return np.array([])

    boot_stats = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = func(sample)

    return boot_stats


def summarize_values(values, n_boot=5000, random_state=42):
    """
    Resume uma distribuição 1D.
    """
    values = np.asarray(values, dtype=float)

    n = len(values)
    if n == 0:
        return {
            'N': 0,
            'median': np.nan,
            'p16': np.nan,
            'p84': np.nan,
            'std': np.nan,
            'mad': np.nan,
            'boot_median_p16': np.nan,
            'boot_median_p84': np.nan,
            'boot_median_std': np.nan,
        }

    median = np.nanmedian(values)
    p16, p84 = np.nanpercentile(values, [16, 84])
    std = MATH.boostrap_func(values) 

    boot_medians = bootstrap_statistic(
        values,
        func=np.median,
        n_boot=n_boot,
        random_state=random_state
    )

    boot_median_p16, boot_median_p84 = np.percentile(boot_medians, [16, 84])

    return {
        'N': n,
        'median':  round(median, 3),
        'p16': round(p16, 3),
        'p84': round(p84, 3),
        'std': round(std, 3),
    }

def summarize_population_key(population, key, dfName='PaperII', Name='Name',
                             n_boot=5000, random_state=42):
    values = get_population_values(population, key, dfName=dfName, Name=Name)
    summary = summarize_values(values, n_boot=n_boot, random_state=random_state)
    summary['population'] = population
    summary['key'] = key
    return summary

def build_summary_table(key, populations, dfName='PaperII', Name='Name',
                        n_boot=5000, random_state=42):
    rows = []
    for population in populations:
        summary = summarize_population_key(
            population, key,
            dfName=dfName, Name=Name,
            n_boot=n_boot, random_state=random_state
        )
        rows.append(summary)

    df = pd.DataFrame(rows)

    # organiza colunas
    cols = [
        'population', 'key', 'N',
        'median',
        'std',  'p16', 'p84'
    ]
    return df[cols]

#%%
for param in ['deltaSize_at_Entry']:#,z_At_FirstEntry 'M200Mean', rOverR200Mean_New]' rOverR200Min:
    print(param)
    print('\n')
    print('fDM threshold 0.7')
    populations = [
        'MBC_DMpoor_Satellite',
        'MBC_DMrich_Satellite',
        'SBC_DMpoor_Satellite',
        'SBC_DMrich_Satellite',
    ]
    
    df_zentry = build_summary_table(param, populations)
    print(df_zentry)
    
    print('\n')
    print('fDM threshold 0.8')
    populations = [
        'MBC_DMpoor_08_Referee_Satellite',
        'MBC_DMrich_08_Referee_Satellite',
        'SBC_DMpoor_08_Referee_Satellite',
        'SBC_DMrich_08_Referee_Satellite',
    ]
    
    df_zentry = build_summary_table(param, populations)
    print(df_zentry)
    print('\n')
    
    print('fDM threshold 0.5')
    populations = [
        'MBC_DMpoor_05_Referee_Satellite',
        'MBC_DMrich_05_Referee_Satellite',
        'SBC_DMpoor_05_Referee_Satellite',
        'SBC_DMrich_05_Referee_Satellite',
    ]
    
    df_zentry = build_summary_table(param, populations)
    print(df_zentry)
    print('\n')
    
    compare_Sample(param, [
                               ['MBC_DMpoor_Satellite', 'MBC_DMrich_Satellite'],
                               ['SBC_DMpoor_Satellite', 'SBC_DMrich_Satellite']], dfName = 'PaperII')
    
    print('\n')

    compare_Sample(param, [
                               ['MBC_DMpoor_08_Referee_Satellite', 'MBC_DMrich_08_Referee_Satellite'],
                               ['SBC_DMpoor_08_Referee_Satellite', 'SBC_DMrich_08_Referee_Satellite']], dfName = 'PaperII')
    
    print('\n')

    compare_Sample(param, [
                               ['MBC_DMpoor_05_Referee_Satellite', 'MBC_DMrich_05_Referee_Satellite'],
                               ['SBC_DMpoor_05_Referee_Satellite', 'SBC_DMrich_05_Referee_Satellite']], dfName = 'PaperII')
    
    

#%%

import numpy as np
import pandas as pd

def finite_values(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]

def cliffs_delta(x, y):
    x = finite_values(x)
    y = finite_values(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan

    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return (gt - lt) / (len(x) * len(y))

def hodges_lehmann_shift(x, y):
    x = finite_values(x)
    y = finite_values(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    diffs = x[:, None] - y[None, :]
    return np.median(diffs)

def get_base_df(population, dfName='PaperII', Name='Name'):
    df = TNG.extractPopulation(population, dfName=dfName, Name=Name).copy()
    return df

def scan_thresholds_fig2(df, thresholds=np.arange(0.5, 0.86, 0.02)):
    rows = []

    # variáveis limpas
    df = df.copy()
    for col in ['DMFrac_99', 'MDM_Norm_Max_99', 'rOverR200Min']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[np.isfinite(df['DMFrac_99'])]
    df = df[np.isfinite(df['MDM_Norm_Max_99'])]
    df = df[np.isfinite(df['rOverR200Min'])]

    # usar log para a fração de halo retida
    df['log_MDM_Norm_Max_99'] = np.log10(df['MDM_Norm_Max_99'])

    for t in thresholds:
        poor = df[df['DMFrac_99'] < t]
        rich = df[df['DMFrac_99'] >= t]

        x1 = poor['log_MDM_Norm_Max_99'].values
        y1 = rich['log_MDM_Norm_Max_99'].values

        x2 = poor['rOverR200Min'].values
        y2 = rich['rOverR200Min'].values

        rows.append({
            'threshold': round(t, 3),
            'Npoor': len(poor),
            'Nrich': len(rich),

            'med_logMDM_poor': np.nanmedian(x1) if len(x1) else np.nan,
            'med_logMDM_rich': np.nanmedian(y1) if len(y1) else np.nan,
            'delta_logMDM': cliffs_delta(x1, y1),
            'HL_logMDM': hodges_lehmann_shift(x1, y1),

            'med_rperi_poor': np.nanmedian(x2) if len(x2) else np.nan,
            'med_rperi_rich': np.nanmedian(y2) if len(y2) else np.nan,
            'delta_rperi': cliffs_delta(x2, y2),
            'HL_rperi': hodges_lehmann_shift(x2, y2),
        })

    return pd.DataFrame(rows)

#%%

df = pd.concat([
    get_base_df('Normal_Satellite'),
    get_base_df('MBC_Satellite'),
    get_base_df('SBC_Satellite')
], ignore_index=True)

scan = scan_thresholds_fig2(df)
print(scan)

#%%
from scipy.stats import spearmanr
def fig2_spearman(df):
    df = df.copy()
    for col in ['DMFrac_99', 'MDM_Norm_Max_99', 'rOverR200Min']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[np.isfinite(df['DMFrac_99'])]
    df = df[np.isfinite(df['MDM_Norm_Max_99'])]
    df = df[np.isfinite(df['rOverR200Min'])]

    rho1, p1 = spearmanr(df['DMFrac_99'], np.log10(df['MDM_Norm_Max_99']))
    rho2, p2 = spearmanr(df['DMFrac_99'], df['rOverR200Min'])

    return {
        'rho_logMDM': rho1, 'p_logMDM': p1,
        'rho_rperi': rho2, 'p_rperi': p2
    }


#%%

SBCDMrichSatellite_IDs = np.array([513846, 570842, 694489,  91, 697154, 724892, 573063, 725997,
       629892, 684712, 692883, 319740, 198199, 592022, 439101, 379807, 531321, 440412, 625186, 499708, 324134,    165, 770775,
               63918, 804308, 355736, 671097, 198195,    156])

MBCDMrichSatellite_IDs = np.array([681818, 419621, 300918, 229958, 489207, 435755, 307497, 450924,
       579511, 422762, 372756,  63973, 117311,    550, 264932, 487746, 701373, 590015, 571909, 488533, 597142,  96789, 445628,
              220616, 307502, 342468, 289394, 823293, 814011])

NormalDMrichSatellite_IDs = np.array([368847, 208838, 368856, 313705,    178, 117426, 342457, 631791,
       368859, 229954, 424295,     45,  63915, 770462, 184962, 282811,  63933, 798393, 184982, 386281, 563474, 597935, 143950,
                  83, 598697,    266, 379810, 143913, 634802])

IDs = np.append(SBCDMrichSatellite_IDs, MBCDMrichSatellite_IDs)

IDs = np.append(IDs, NormalDMrichSatellite_IDs)
IDs = np.unique(IDs)
#%%
df_ssfr_fixed = EvolutionParticle_sSFR(
    IDs=IDs,
    dfSample=PaperII,
    aperture_mode="fixed_pkpc",
    apertures=[0.5, 2.5],
    save_particle_cache=True,
    use_particle_cache=True,
    verbose=True,
)