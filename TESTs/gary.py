#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:54:04 2026

@author: abhner
"""

import numpy as np

#%%
PaperII = extractDF('PaperII')

#%%
np.random.seed(160401)


IDs_SBC_Centrals = np.random.choice(extractPopulation('SBC_Central', dfName = 'PaperII', SubfindID_99 = True), size = 20, replace = False)
IDs_MBC_Centrals = np.random.choice(extractPopulation('MBC_Central', dfName = 'PaperII', SubfindID_99 = True), size = 20, replace = False)
IDs_Normal_Centrals = np.random.choice(extractPopulation('Normal_Central', dfName = 'PaperII', SubfindID_99 = True), size = 20, replace = False)
IDs_SubDiffuse_Centrals = np.random.choice(extractPopulation('SubDiffuse_Central', dfName = 'PaperII', SubfindID_99 = True), size = 20, replace = False)
IDs_Diffuse_Centrals = np.random.choice(extractPopulation('Diffuse_Central', dfName = 'PaperII', SubfindID_99 = True), size = 20, replace = False)

IDs_SBC_SatelliteDMrichs = np.random.choice(extractPopulation('SBC_SatelliteDMrich', dfName = 'PaperII', SubfindID_99 = True), size = 20, replace = False)
IDs_MBC_SatelliteDMrichs = np.random.choice(extractPopulation('MBC_SatelliteDMrich', dfName = 'PaperII', SubfindID_99 = True), size = 20, replace = False)
IDs_Normal_SatelliteDMrichs = np.random.choice(extractPopulation('Normal_SatelliteDMrich', dfName = 'PaperII', SubfindID_99 = True), size = 20, replace = False)
IDs_SubDiffuse_SatelliteDMrichs = np.random.choice(extractPopulation('SubDiffuse_SatelliteNotInteract', dfName = 'PaperII', SubfindID_99 = True), size = 20, replace = False)
IDs_Diffuse_SatelliteDMrichs = np.random.choice(extractPopulation('Diffuse_SatelliteNotInteract', dfName = 'PaperII', SubfindID_99 = True), size = 18, replace = False)

IDs_SBC_SatelliteDMpoors = np.random.choice(extractPopulation('SBC_SatelliteDMpoor', dfName = 'PaperII', SubfindID_99 = True), size = 16, replace = False)
IDs_MBC_SatelliteDMpoors = np.random.choice(extractPopulation('MBC_SatelliteDMpoor', dfName = 'PaperII', SubfindID_99 = True), size = 20, replace = False)
IDs_Normal_SatelliteDMpoors = np.random.choice(extractPopulation('Normal_SatelliteDMpoor', dfName = 'PaperII', SubfindID_99 = True), size = 8, replace = False)
IDs_SubDiffuse_SatelliteDMpoors = np.random.choice(extractPopulation('SubDiffuse_SatelliteInteract', dfName = 'PaperII', SubfindID_99 = True), size = 20, replace = False)
IDs_Diffuse_SatelliteDMpoors = np.random.choice(extractPopulation('Diffuse_SatelliteInteract', dfName = 'PaperII', SubfindID_99 = True), size = 20, replace = False)

#%%


IDs =  np.concatenate([
    IDs_SBC_Centrals,
    IDs_MBC_Centrals,
    IDs_Normal_Centrals,
    IDs_SubDiffuse_Centrals,
    IDs_Diffuse_Centrals,
    IDs_SBC_SatelliteDMrichs,
    IDs_MBC_SatelliteDMrichs,
    IDs_Normal_SatelliteDMrichs,
    IDs_SubDiffuse_SatelliteDMrichs,
    IDs_Diffuse_SatelliteDMrichs,
    IDs_SBC_SatelliteDMpoors,
    IDs_MBC_SatelliteDMpoors,
    IDs_Normal_SatelliteDMpoors,
    IDs_SubDiffuse_SatelliteDMpoors,
    IDs_Diffuse_SatelliteDMpoors
])

IDs = np.unique(IDs)

#%%

df_io = EvolutionParticle_InnerOuter(
    IDs=IDs,
    dfSample=PaperII,
    Snaps = (50, 99),
    metrics=("ssfr", "colors"),
    color_pairs=(("U", "r"),),
    aperture_mode="fraction_reference_rh",
    apertures=(0.5, 1.0),
    reference_rh="final",
    radius_floor_pkpc=0.5,
    verbose = True,
)