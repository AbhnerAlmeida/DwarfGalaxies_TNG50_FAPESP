#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:53:48 2026

@author: abhner
"""


#%%
import pandas as pd
import os
import sys
sys.path.append(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/analyzes/")
sys.path.append(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/src/")


import TNGFunctions as TNG

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic


plt.style.use(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/src/abhner.mplstyle")

MAINPATH = os.getenv("HOME")+'/TNG_Analyzes/TestMetallicity'

#%%

All = pd.read_pickle(os.getenv("HOME")+'/TNG_Analyzes/TestMetallicity/TNG50/DFs/All.pkl')

#%%
df_logMstar_Range = All.loc[All.logMstarRad_99.between(8.5, 10)]
#df_logMstar_logStar_Range = df_logMstar_Range.loc[df_logMstar_Range.logHalfRadstar_99 < 0.]
df_Final = df_logMstar_Range.loc[df_logMstar_Range.Flags == 1]

#%%

df = df_Final.copy()

#SBs
df['SubSample'] = np.nan
df.loc[df['logHalfRadstar_99'] <= -0.35, 'SubSample'] = 'CompactSB'

df_NotSB = df.loc[df.SubSample != 'CompactSB']

# seleção básica
mask = np.isfinite(df_NotSB['logMstarRad_99']) & np.isfinite(df_NotSB['logHalfRadstar_99'])
df_NotSB = df_NotSB.loc[mask].copy()

M_NotSB = df_NotSB['logMstarRad_99'].values
R_NotSB = df_NotSB['logHalfRadstar_99'].values
R = df['logHalfRadstar_99'].values
M = df['logMstarRad_99'].values

# bins em massa
nbins = 12
bins = np.linspace(M_NotSB.min(), M_NotSB.max(), nbins + 1)

median_R, edges, _ = binned_statistic(M_NotSB, R_NotSB, statistic='median', bins=bins)
bin_centers = 0.5 * (edges[:-1] + edges[1:])

valid = np.isfinite(median_R)

# tamanho esperado a partir da mediana
R_med = np.interp(M, bin_centers[valid], median_R[valid])

# residual da relação massa-tamanho
df['DeltaLogR_99'] = R - R_med

R_med = np.interp(M_NotSB, bin_centers[valid], median_R[valid])

df_NotSB['DeltaLogR_99'] = R_NotSB - R_med

# percentis
thr5 = np.nanpercentile(df_NotSB['DeltaLogR_99'], 5)
thr40 = np.nanpercentile(df_NotSB['DeltaLogR_99'], 40)
thr60 = np.nanpercentile(df_NotSB['DeltaLogR_99'], 60)

# inicializa

# compactas
df.loc[(df['DeltaLogR_99'] <= thr5) & (df.SubSample != 'CompactSB'), 'SubSample'] = 'Compact'

# normais
df.loc[(df['DeltaLogR_99'].between(thr40, thr60)) & (df.SubSample != 'CompactSB'), 'SubSample'] = 'Normal'

#%%

plt.scatter(df.logMstarRad_99,
            df.logHalfRadstar_99,
            s = 5,
            c='grey')


for label in ['CompactSB', 'Compact','Normal']:
    
    sel = df['SubSample'] == label
    
    plt.scatter(df.loc[sel,'logMstarRad_99'],
                df.loc[sel,'logHalfRadstar_99'],
                label=label)

plt.legend()
plt.xlabel(r'$\log M_\star$')
plt.ylabel(r'$\log R_{1/2}$')
plt.show()

#%%

df = df.loc[ (df.SubSample == 'CompactSB') |
                   (df.SubSample == 'Compact') |
                   (df.SubSample == 'Normal')].copy()

#%%
from scipy.stats import ks_2samp
def mass_distribution_match(df,
                            class_col='SubSample',
                            mass_col='logMstarRad_99',
                            classes=('CompactSB','Compact','Normal'),
                            p_threshold=0.2,
                            max_iter=10000,
                            random_state=42):

    rng = np.random.default_rng(random_state)

    df = df[df[class_col].isin(classes)].copy()

    # identificar população minoritária
    counts = df[class_col].value_counts()
    ref_class = counts.idxmin()

    print("Reference population:", ref_class)
    print(counts)

    df_ref = df[df[class_col] == ref_class].copy()
    N_target = len(df_ref)

    selected = [df_ref]

    mass_ref = df_ref[mass_col].values

    for cls in classes:
        if cls == ref_class:
            continue

        df_cls = df[df[class_col] == cls].copy()
        masses = df_cls[mass_col].values

        best_sample = None
        best_p = -1

        for _ in range(max_iter):

            idx = rng.choice(len(df_cls), size=N_target, replace=False)
            sample = masses[idx]

            stat, p = ks_2samp(mass_ref, sample)

            if p > best_p:
                best_p = p
                best_sample = idx

            if p >= p_threshold:
                break

        print(f"{cls}: best p-value =", round(best_p,3))

        selected.append(df_cls.iloc[best_sample])

    df_matched = np.concatenate([s.index.values for s in selected])

    return df.loc[df_matched].copy()

#%%
# df_matched = mass_distribution_match(
#     df,
#     class_col='SubSample',
#     mass_col='logMstarRad_99',
#     classes=('CompactSB', 'Compact', 'Normal'),
#     random_state=160401
# )

#df_matched.to_csv(MAINPATH + '/TNG50/DFs/df_matched.csv')

df_matched = TNG.extractDF('df_matched',  PATH=MAINPATH)

#%%

# df_GroupFirstSub = TNG.EvolutionDF('GroupFirstSub', df_matched.SubfindID_99.values, 
#                  PATH=os.getenv("HOME")+'/TNG_Analyzes/TestMetallicity', 
#                  SAVEFILE='DFs')

# df_SubfindID = TNG.EvolutionDF('SubfindID', df_matched.SubfindID_99.values, 
#                  PATH=os.getenv("HOME")+'/TNG_Analyzes/TestMetallicity', 
#                  SAVEFILE='DFs')

#%%

# for ID in df_matched.SubfindID_99.values:
#     ID_FirstGroup = df_GroupFirstSub[str(ID)].values[int(99 - 99)]
#     ID_SubfindID = df_SubfindID[str(ID)].values[int(99 - 99)]
#     IDs = df_SubfindID[str(ID)].values

#     argNotNan = np.argwhere(~np.isnan(IDs)).T[0]
    
#     try:
#         df_matched.loc[df_matched.SubfindID_99 == ID, 'SnapBirth'] = int(99 - argNotNan[-1])
#     except Exception as e:
#         print(e)

#     if ID_FirstGroup == ID_SubfindID:
#         df_matched.loc[df_matched.SubfindID_99 == ID, 'CentralSatellite'] = 'Central'
#     else:
#         df_matched.loc[df_matched.SubfindID_99 == ID, 'CentralSatellite'] = 'Satellite'
        
#%%

# df_Final = df_matched.loc[(df_matched.CentralSatellite == 'Central') & 
#                           (df_matched.SnapBirth <= 17.0)]

#%%

# df_Final_matched = mass_distribution_match(
#     df_Final,
#     class_col='SubSample',
#     mass_col='logMstarRad_99',
#     classes=('CompactSB', 'Compact', 'Normal'),
#     random_state=160401
# )

#df_Final_matched.to_csv(MAINPATH + '/TNG50/DFs/df_Final_matched.csv')

df_Final_matched = TNG.extractDF('df_Final_matched',  PATH=MAINPATH)

#%%
# dfGasMetallicity = TNG.EvolutionDF('SubhaloGasMetallicity', df_Final_matched.SubfindID_99.values, 
#                 PATH=os.getenv("HOME")+'/TNG_Analyzes/TestMetallicity', 
#                 SAVEFILE='DFs')

# dfGasMetallicity = TNG.EvolutionDF('SubhaloHalfmassRadType4', df_Final_matched.SubfindID_99.values, 
#                 PATH=os.getenv("HOME")+'/TNG_Analyzes/TestMetallicity', 
#                 SAVEFILE='DFs')

# dfGasMetallicity = TNG.EvolutionDF('SubhaloMassInRadType4', df_Final_matched.SubfindID_99.values, 
#                 PATH=os.getenv("HOME")+'/TNG_Analyzes/TestMetallicity', 
#                 SAVEFILE='DFs')

# dfHFraction = TNG.EvolutionDF('SubhaloGasMetalFractionsH', df_Final_matched.SubfindID_99.values, 
#                  PATH=os.getenv("HOME")+'/TNG_Analyzes/TestMetallicity', 
#                  SAVEFILE='DFs')

# dfOFraction = TNG.EvolutionDF('SubhaloGasMetalFractionsO', df_Final_matched.SubfindID_99.values, 
#                  PATH=os.getenv("HOME")+'/TNG_Analyzes/TestMetallicity', 
#                  SAVEFILE='DFs')


dfGasMetallicity = TNG.extractDF('SubhaloGasMetallicity',  PATH=MAINPATH)

dfRhalf = TNG.extractDF('SubhaloHalfmassRadType4',  PATH=MAINPATH)

dfMassStar = TNG.extractDF('SubhaloMassInRadType4',  PATH=MAINPATH)


#%%

main_snap = 78 # z= 0.3

for ID in df_Final_matched.SubfindID_99.values:
    zGas = dfGasMetallicity[str(ID)].values[int(99 - main_snap)]
    Rhalf = dfRhalf[str(ID)].values[int(99 - main_snap)]
    MassStar = dfMassStar[str(ID)].values[int(99 - main_snap)]
    
    df_Final_matched.loc[df_Final_matched.SubfindID_99 == ID, 'GasMetallicity_78'] = zGas
    df_Final_matched.loc[df_Final_matched.SubfindID_99 == ID, 'logHalfRadstar_78'] = Rhalf
    df_Final_matched.loc[df_Final_matched.SubfindID_99 == ID, 'logMstarRad_78'] = MassStar

#%%

# df_Final.to_csv(MAINPATH + '/TNG50/DFs/df_Final.csv')

#%%

import matplotlib.pyplot as plt

# =========================
# 1. Seleção básica
# =========================
cols = ['logMstarRad_78', 'logHalfRadstar_78', 'GasMetallicity_78',
        'logMstarRad_99', 'logHalfRadstar_99']

df = df_Final_matched[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

# cuidado: só funciona se GasMetallicity_78 > 0
df = df[df['GasMetallicity_78'] > 0].copy()

M78 = df['logMstarRad_78'].values
R78 = df['logHalfRadstar_78'].values
M99 = df['logMstarRad_99'].values
R99 = df['logHalfRadstar_99'].values
Z78 = np.log10(df['GasMetallicity_78'].values)

compactness = M78 - 2.0*R78
deltaR = R99 - R78

# =========================
# 2. MZR mediana e residual
# =========================
nbins = 12
bins = np.linspace(M78.min(), M78.max(), nbins + 1)

median_Z, edges, _ = binned_statistic(M78, Z78, statistic='median', bins=bins)
bin_centers = 0.5 * (edges[:-1] + edges[1:])

valid = np.isfinite(median_Z)
Z_mzr = np.interp(M78, bin_centers[valid], median_Z[valid])
deltaZ = Z78 - Z_mzr

# =========================
# 3. Figura 2x2
# =========================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax1, ax2, ax3, ax4 = axes.flatten()

# Painel 1: MZR colorida por compactness
sc1 = ax1.scatter(M78, Z78, c=compactness, alpha=0.75, cmap='plasma')
ax1.plot(bin_centers[valid], median_Z[valid], color='black', label='MZR mediana')
ax1.set_xlabel(r'$\log(M_\star/M_\odot)$')
ax1.set_ylabel(r'$\log(Z_{\rm gas})$')
ax1.set_title('z = 0.3: MZR colorida por compactness', fontsize = 14)
ax1.legend()
cbar1 = plt.colorbar(sc1, ax=ax1)
cbar1.set_label(r'$\log(M_\star/R_{1/2}^2)$')

# Painel 2: plano massa-tamanho colorido por metallicity
sc2 = ax2.scatter(M78, R78, c=Z78, alpha=0.75, cmap='viridis')
ax2.set_xlabel(r'$\log(M_\star/M_\odot)$')
ax2.set_ylabel(r'$\log(R_{1/2})$')
ax2.set_title('Plano massa–tamanho colorido por metallicity', fontsize = 14)
cbar2 = plt.colorbar(sc2, ax=ax2)
cbar2.set_label(r'$\log(Z_{\rm gas})$')

# Painel 3: residual da MZR vs compactness
ax3.scatter(compactness, deltaZ, alpha=0.75)
ax3.axhline(0, color='black', lw=1.5, ls='--')
ax3.set_xlabel(r'$\log(M_\star/R_{1/2}^2)$')
ax3.set_ylabel(r'$\Delta \log(Z_{\rm gas})$')
ax3.set_title('Offset da MZR vs compactness', fontsize = 14)

# Painel 4: compactness vs evolução de tamanho
ax4.scatter(compactness, deltaR,  alpha=0.75)
ax4.axhline(0, color='black', lw=1.5, ls='--')
ax4.set_xlabel(r'$\log(M_\star/R_{1/2}^2)$ at z=0.3')
ax4.set_ylabel(r'$\Delta \log(R_{1/2}) = \log R_{99} - \log R_{78}$')
ax4.set_title('Evolução estrutural até z = 0', fontsize = 14)

plt.tight_layout()
plt.show()


#%%

plt.scatter(df.logMstarRad_99,
            df.logHalfRadstar_99,
            s = 5,
            c='grey')


for label in ['CompactSB', 'Compact','Normal']:
    
    sel = df_Final_matched['SubSample'] == label
    
    plt.scatter(df_Final_matched.loc[sel,'logMstarRad_99'],
                df_Final_matched.loc[sel,'logHalfRadstar_99'],
                label=label)

plt.legend()
plt.xlabel(r'$\log M_\star$')
plt.ylabel(r'$\log R_{1/2}$')
plt.show()