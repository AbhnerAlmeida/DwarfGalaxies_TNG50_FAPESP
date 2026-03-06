"""
TNGFunctions
============

Utilities for analysing subhalo evolution in the IllustrisTNG simulations.

This module contains functions to extract merger tree histories,
compute derived physical quantities, and generate evolution
dataframes used in the analysis of dwarf galaxy evolution.

Author
------
Abhner P. de Almeida
abhner.almeida AT usp.br
University of São Paulo (USP)
"""

import h5py
import shutil
import os

import sys
sys.path.append(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/analyzes/GaryScripts")

import pandas as pd
import numpy as np
np.seterr(divide='ignore') # ignore divide by zero
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import MATH
import ExtractTNG as ETNG

import cosmoutils as cos
import warnings
from pytreegrav import Potential
from scipy.signal import find_peaks

from multiprocessing import Pool, Manager
import math


import ast # for ast.literal_eval



warnings.filterwarnings( "ignore")

# cosmological parameters
Omegam0 = 0.3089
h = 0.6774

#constants

kb = 1.380658e-16 # erg / K
mH = 1.67e-24 #g
mu = 0.58
Msun_to_g = 1.999999999E+33 
kpc_to_cm = 3.086e+21
kmc_to_cm = 100000
Gyr_to_s = 3.15e16 

G = 4.300917270038e-06 # kpc Msun^-1 km^2 s^-2

Xh = 0.76
gamma = 5./3.
mp = 1.673e-24 #g
kb = 1.380658e-16 # erg / K




#Paths
dfTime = pd.read_csv(os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory/SNAPS_TIME.csv')

component = { 
        '0': 'gas',
        '1': 'dm',
        '4': 'stars',
        '5': 'bhs',
}

def _as_id_list(ids):
    """Normalize IDs to a Python list[int]."""
    if isinstance(ids, (int, np.integer)):
        return [int(ids)]
    return [int(x) for x in np.atleast_1d(ids)]


def _ensure_dir(path):
    """Create directory if needed (including parents)."""
    if path is None or path == '':
        return
    os.makedirs(str(path), exist_ok=True)
    
def extractDF(Name, PATH=os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory' , 
              MERGER = False, SUBHALO = False, ACCRETED = False, EXsituINsitu = False, SIM = 'TNG50', fmt = 'csv'):
    '''
    Extract a specific dataframe
    Parameters
    ----------
    Name : file name. str
    PATH : path for the file. Default: './../DFs/'. str
    SIM : TNG simulation code. Default: SIMTNG. str
    fmt : format to save the dataframe. Default: csv. str
    Returns
    -------
    Requested dataframe
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    base = os.path.join(PATH, SIM)

    if SUBHALO:
        fpath = os.path.join(base, "Subhalos", f"{Name}.hdf5")
        df = h5py.File(str(fpath), "r")

    elif MERGER:
        fpath = os.path.join(base, "DFs", "Analysis", "Mergers", f"{Name}.{fmt}")
        df = pd.read_csv(fpath, index_col=0)

    elif ACCRETED:
        fpath = os.path.join(base, "DFs", "Analysis", "RecentAccretedParticle", f"{Name}.{fmt}")
        df = pd.read_csv(fpath, index_col=0)

    elif EXsituINsitu:
        fpath = os.path.join(base, "DFs", "Analysis", "StellarContent", f"{Name}.{fmt}")
        df = pd.read_csv(fpath, index_col=0)

    else:
        fpath = os.path.join(base, "DFs", f"{Name}.{fmt}")
        df = pd.read_csv(fpath, index_col=0)

    # Drop pandas' unnamed index columns if present
    if isinstance(df, pd.DataFrame):
        df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]

    return df

def getsuplementary(Name, PATH=os.getenv("HOME")+'/SIMS/TNG' , 
             SIM = 'TNG50-1'):
    '''
    Extract a specific suplementary catalog
    Parameters
    ----------
    Name : file name. str
    PATH : path for the file. Default: './../DFs/'. str
    SIM : TNG simulation code. Default: SIMTNG. str
    Returns
    -------
    Requested dataframe
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    fpath = os.path.join(PATH, SIM, "suplementary", f"{Name}.hdf5")
    return h5py.File(str(fpath), "r")

def getMock(Name, PATH=os.getenv("HOME")+'/SIMS/TNG' , 
             SIM = 'TNG50-1', CATALOG = 'sdss', SNAP = 'snapnum_099'):
    '''
    Extract a specific suplementary catalog
    Parameters
    ----------
    Name : file name. str
    PATH : path for the file. Default: './../DFs/'. str
    SIM : TNG simulation code. Default: SIMTNG. str
    Returns
    -------
    Requested dataframe
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    fpath = os.path.join(PATH, SIM, "MOCK", CATALOG, SNAP, f"{Name}.hdf5")
    return h5py.File(str(fpath), "r")

def initialDF(PATH = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory', verbose = False, SIM='TNG50', NSim = '-1',snapnum=99, fmt = 'csv'):
    """Extract all the subhalos and save in PATH
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    argument:
        PATH [string] path to save the DF
        simulation [string]
        snapnum [int] snapshot for the subhalos
    return:
    """

    if verbose:
        print('\n Taking DF for ALL subhalos ...')

    try:
        df_z0 = extractDF("All", SIM=SIM)
        if verbose:
            print('Luckily you already have DF for ALL subhalos ...')

    except Exception:
        if verbose:
            print('You don\'t have df for ALL subhalos, I will make for you ...')

        #Take the Mass and Size values from subhalos in TNG
        values = ETNG.getsubhalos(['SubhaloMassInRadType4', 'SubhaloHalfmassRadType4', 'SubhaloFlag'], 
                                                                        SIM=SIM+NSim,snapnum=snapnum)

        Mass_all = np.array([value for value in values[:, 0]])
        Rad_all = np.array([value for value in values[:, 1]])
        Flags = np.array([value for value in values[:, 2]])

                                                                        
    
        #Take only the stars Mass and Size, and also computes Sigma_eff
        Mass_star = Mass_all*1e10/h # Msun
        Rad_eff = (Rad_all/h) / (1+dfTime.z.loc[dfTime.Snap == snapnum].values[0])  # kpc, a = 1 (z = 0)
    
        Mass_star = np.log10(Mass_star)
        Rad_eff = np.log10(Rad_eff)
    
        ids = np.linspace(0, len(Flags)-1, len(Flags))
        ids = np.array([int(value) for value in ids])

        data = {'SubfindID_99': ids,
                'logMstarRad_99':Mass_star,  
                'logHalfRadstar_99': Rad_eff, 
                'Flags': Flags}
    
        df_z0 = pd.DataFrame(data)
        df_z0.to_pickle(PATH+'/'+SIM+'/DFs/'+'All.'+fmt)
            
    return df_z0

def DownloadSubhalo(ID, PATH=os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory' , SIM = 'TNG50', NSim = '-1', snapnum = 99): 
        '''
        Download subhalo evolution
        Parameters
        ----------
        ID : subhalo ID.
        Returns
        -------
        Requested subhalo
        -------
        Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
        '''
        cache_dir = os.path.join(PATH, SIM, "Subhalos")
        _ensure_dir(cache_dir)

        cached = cache_dir + '/' + f"{int(ID)}.hdf5"
        try:
            return h5py.File(str(cached), "r")
        except Exception:
            print(f"You don't have the file for {ID}")

        # Generate with ETNG (assumed to write to ~/SIMS/TNG/<SIM+NSim>/output/)
        f = ETNG.History(int(ID), sim=SIM + NSim, snapnum=snapnum)
        try:
            f.close()
        except Exception:
            pass

        etng_out = os.path.join(os.getenv("HOME"), "SIMS", "TNG", f"{SIM}{NSim}", "output")
        src = etng_out + '/' +f"sublink_mpb_{int(ID)}.hdf5"
        shutil.copyfile(src, cached)

        return h5py.File(str(cached), "r")
    
def CleanAllSubhalos(ID, PATH=os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory' , SIM = 'TNG50', NSim = '-1', snapnum = 99):
    """
    Fecha e remove todos os arquivos temporários criados.
    """
   
    
    candidates = [
        os.path.join(os.getenv("HOME"), "SIMS", "TNG", f"{SIM}{NSim}", "output", f"sublink_mpb_{int(ID)}.hdf5"),
        os.path.join(PATH, SIM, "Subhalos", f"{int(ID)}.hdf5"),
    ]

    for p in candidates:
        try:
            p.unlink()
        except Exception:
            pass

    print("All temporally subhalos are removed")


def ImportField(param, ID, SIM = 'TNG50', NSim = '-1'): 
        '''
        Import a param evolution of a given subhalo a specific dataframe
        Parameters
        ----------
        param: param to import.
        ID : subhalo ID.
        Returns
        -------
        Requested subhalo
        -------
        Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
        '''
        pathfile = os.getenv("HOME") + "/SIMS/TNG/" + SIM + NSim + "/output/" + 'sublink_mpb_'+str(ID)+'.hdf5'
        try:
                file = h5py.File(pathfile,'r') 
                Progenitor = file['FirstProgenitorID'][0]
        except Exception:
                pathfile = os.getenv("HOME") + "/SIMS/TNG/" + SIM + NSim + "/output/" + 'dic_99_'+str(ID)+'.csv'
                
                try:
                        file = pd.read_csv(pathfile, index_col = 0)
                except Exception:
                        file = ETNG.getsubhalo(ID, simulation=SIM + NSim)
                        
                        file = pd.DataFrame.from_dict(file)
                        file.to_csv(pathfile)


                Progenitor = file['related']['sublink_progenitor']
                
       
        if Progenitor is not None:
                
                file = DownloadSubhalo(ID)
                if 'Group' in param or 'over_R_Crit200' in param:
                        if 'over_R_Crit200' in param:
                                fileSubhalo = DownloadSubhalo(ID)

                        IDCens = file['GroupFirstSub'][:] 
                        IDCen = IDCens[0] 
                        subhaloCen =  DownloadSubhalo(IDCen)

                        if 'FirstGroup' in param:
                                try:
                                        IDFirstSubFirstGroup = extractDF('GroupFirstSubFirstGroup', SIM = SIM)
                                        IDCen = IDFirstSubFirstGroup[str(ID)][0]
                                        file =  DownloadSubhalo(IDCen)
                                        param = param.replace('FirstGroup', '')
                                except Exception:
                                        data = np.log10(subhaloCen['Group_M_Crit200'][:]*1e10/h)
                                        data = data[:int(99 - 33)] # start after z = 2

                                        Differ = data[:-1] - data[1:] #/ data[:-1]

                                        args = np.argwhere(Differ > 1.0).T[0]    



                                        if len(args) > 0:
                                                file = ETNG.History(int(IDCens[args[-1]]), snapnum = int(99 - args[-1]))
                                                #CenNewID = ETNG.getsubhaloid99(int(IDCens[args[-1]]),simulation=SIMTNG + '-1',snapnum=int(99 - args[-1]))
                                                CenNewID = file['GroupFirstSub'][0]
                                                file =  DownloadSubhalo(int(CenNewID))
                                        
                                        param = param.replace('FirstGroup', '')

                        elif 'FinalGroup' in param:
                                try:
                                        IDFirstSubFirstGroup = extractDF('GroupFirstSubFinalGroup', SIM = SIM)
                                        IDCen = IDFirstSubFirstGroup[str(ID)][0]
                                        file =  DownloadSubhalo(IDCen)
                                        param = param.replace('FinalGroup', '')
                                except Exception:
                                        data = np.log10(subhaloCen['Group_M_Crit200'][:]*1e10/h)
                                        data = data[:int(99 - 33)] # start after z = 2

                                        Differ = data[:-1] - data[1:] #/ data[:-1]

                                        args = np.argwhere(Differ > 1.0).T[0]    

                                        if len(args) > 0:
                                                print('ID: ', ID)
                                                #CenNewID = ETNG.getsubhaloid99(int(IDCens[args[0]]),simulation=SIMTNG + '-1',snapnum=int(99 - args[0]))
                                                file = ETNG.History(int(IDCens[args[0]]), snapnum = int(99 - args[0]))
                                                CenNewID = file['GroupFirstSub'][0]
                                                file =  DownloadSubhalo(int(CenNewID))
                                        
                                        param = param.replace('FinalGroup', '')
                              
                        if '_M_' in param or 'Mass' in param:
                                data = np.log10(file[param][:] * 1e10 / h) # Msun
                        elif '_R_' in param:
                                data = np.log10(file[param][:] / (1+dfTime.z.values[:len(file[param][:])]) / h) # kpc
                        elif 'Pos' in param[:-1]:
                                data = file[param[:-1]][:, int(param[-1]) - 1] / (1+dfTime.z.values[:len(file[param[:-1]][:, int(param[-1]) - 1])]) / h # kpc
                        elif 'CM' in param[:-1] :
                                data = file[param[:-1]][:, int(param[-1]) - 1] / (1+dfTime.z.values[:len(file[param[:-1]][:, int(param[-1]) - 1])]) / h # kpc
                        elif 'Vel' in param[:-1] :
                                data = file[param[:-1]][:, int(param[-1]) - 1] * (1+dfTime.z.values[:len(file[param[:-1]][:, int(param[-1]) - 1])])   # km / s
                        elif 'BHMdot ' in param[:-1]:
                                data = (file[param][:] * 1e10 / h) / (0.978 /h)   # Msun / Gyr

                        elif 'over_R_Crit200' in param :


                                pos = np.array([fileSubhalo['SubhaloPos'][:, 0], 
                                                fileSubhalo['SubhaloPos'][:, 1],
                                                fileSubhalo['SubhaloPos'][:, 2]]).T
  
                                posCen = np.array([file['SubhaloPos'][:, 0], 
                                                file['SubhaloPos'][:, 1],
                                                file['SubhaloPos'][:, 2]]).T
                   
                                
                                while len(pos) < 100:
                                        pos = np.append(pos,  [[np.nan, np.nan, np.nan]], axis=0)
                                while len(posCen) < 100:
                                        posCen = np.append(posCen,  [[np.nan, np.nan, np.nan]], axis=0)
                                

                                dr = ETNG.FixPeriodic(pos - posCen, sim=SIM + NSim) 
                                r = np.linalg.norm(dr, axis=1)
                                R200Cen = file['Group_R_Crit200'][:]
                                
                                while len(R200Cen) < 100:
                                        R200Cen = np.append(R200Cen, np.nan)
                                data = r / R200Cen

                        else:
                                data = file[param][:]

                elif 'Subhalo' in param:
                        #print(param, ID)
                        if 'Mass' in param[7:11]:
                                print(param)
                                if not 'Type' in param:
                                        data = np.log10(file[param][:] * 1e10 / h) # Msun
                                else:
                                        data = np.log10(file[param[:-1]][:, int(param[-1])] * 1e10 / h) # Msun
                        elif 'HalfmassRad' in param:
                                if not 'Type' in param:
                                        data = np.log10(file[param][:] / (1+dfTime.z.values[:len(file[param][:])]) / h) # kpc
                                else:
                                        data = np.log10(file[param[:-1]][:, int(param[-1])] / (1+dfTime.z.values[:len(file[param[:-1]][:])]) / h) # kpc
                        elif 'Pos' in param[:-1]:
                                data = file[param[:-1]][:, int(param[-1]) - 1] / (1+dfTime.z.values[:len(file[param[:-1]][:, int(param[-1]) - 1])]) / h # kpc
                        elif 'Spin' in param[:-1]:
                                data = file[param[:-1]][:, int(param[-1]) - 1] / h # kpc km / s
                        elif 'CM' in param[:-1]:
                                data = file[param[:-1]][:, int(param[-1]) - 1] / (1+dfTime.z.values[:len(file[param[:-1]][:, int(param[-1]) - 1])]) / h # kpc
                        elif 'Vel' in param[:-1] and not 'Disp' in param:
                                data = file[param[:-1]][:, int(param[-1]) - 1]  # km / s
                        elif 'BHMdot ' in param[:-1]:
                                data = (file[param][:] * 1e10 / h) / (0.978 /h)   # Msun / Gyr
                                
                        elif 'Metallicity' in param:
                            data = np.log10(file[param][:] /0.0127)   # Z0
                            
                        elif 'GasMetal' in param:
                            print(param)
                            if 'H' in param[-1]: 
                                data = np.log10(file['SubhaloGasMetalFractions'][:, 0] * file['SubhaloMassType'][:, 0] * 1e10 / h)   # H0
                            elif 'He' == param[-2:]: 
                                data = np.log10(file['SubhaloGasMetalFractions'][:, 1] * file['SubhaloMassType'][:, 0] * 1e10 / h)   # H0
                            
                            elif 'O' == param[-1:]: 
                                data = np.log10(file['SubhaloGasMetalFractions'][:, 4] * file['SubhaloMassType'][:, 0] * 1e10 / h)   # H0
                            
                            else: 
                                data = np.log10(np.nansum(file['SubhaloGasMetalFractions'][:, 2:] )* file['SubhaloMassType'][:, 0] * 1e10 / h )   # H0
                            
                        
                        elif 'StarMetal' in param:
                            if 'H' in param[-1]: 
                                data = np.log10(file['SubhaloStarMetalFractions'][:, 0] * file['SubhaloMassType'][:, 4] * 1e10 / h)   # H0
                            elif 'He' == param[-2:]: 
                                data = np.log10(file['SubhaloStarMetalFractions'][:, 1] * file['SubhaloMassType'][:, 4] * 1e10 / h)   # H0
                            
                            else: 
                                data = np.log10(np.nansum(file['SubhaloStarMetalFractions'][:, 2:] )* file['SubhaloMassType'][:, 4] * 1e10 / h )   # H0
                            
                        elif 'StellarPhotometrics' in param[:-1]:
                            if 'U' in param[-2:]:
                                data = file['SubhaloStellarPhotometrics'][:, 0]  
                            if 'B' in param[-2:]:
                                data = file['SubhaloStellarPhotometrics'][:, 1]  
                            if 'V' in param[-2:]:
                                data = file['SubhaloStellarPhotometrics'][:, 2]  
                            if 'K' in param[-2:]:
                                data = file['SubhaloStellarPhotometrics'][:, 3]  
                            if 'g' in param[-2:]:
                                data = file['SubhaloStellarPhotometrics'][:, 4]  
                            if 'r' in param[-2:]:
                                data = file['SubhaloStellarPhotometrics'][:, 5]  
                            if 'i' in param[-2:]:
                                data = file['SubhaloStellarPhotometrics'][:, 6]  
                            if 'z' in param[-2:]:
                                data = file['SubhaloStellarPhotometrics'][:, 7]  
                            

                        elif 'SFR' in param: 
                                if 'sSFR' in param:
                                        sfr = file[param.replace('s', '')][:]
                                        if 'in' in param:
                                                Mass_star = file['SubhaloMassIn'+param.split('in')[1]+'Type'][:, int(4)]  *1e10/h
                                        else:
                                                Mass_star = file['SubhaloMassType'][:, int(4)]*1e10/h
                                        data = np.array([])
                                        for idm, mass in enumerate(Mass_star):
                                                if mass == 0. :
                                                        data = np.append(data, 0.)
                                                elif np.isnan(mass):
                                                        data = np.append(data, np.isnan)
                                                else:
                                                        data = np.append(data,sfr[idm]/mass)
                                        data[data == 0] = -14
                                        if len(data) == len(data[data == -14]):
                                                data = data
                                        else:
                                                data[data != -14] = np.log10(data[data != -14])

                                else:
                                        data = file[param][:]
                                        data[data == 0] = -5
                                        data[data != -5] = np.log10(data[data != -5])

                        else:
                                data = file[param][:]
                
                else:
                        data = file[param][:]
                      
                        
        else:
                if 'Subhalo' in param:
                        if 'Mass' in param[7:11]:
                                if not 'Type' in param:
                                        if not 'InHalfRad' in param:
                                                data = np.array([np.log10(file['mass']['subhalo'] * 1e10 / h)]) # Msun
                                        elif 'InHalfRad' in param:
                                                data = np.array([np.log10(file['massinhalfrad']['subhalo'] * 1e10 / h)]) # Msun
                                        elif 'InRad' in param:
                                                data = np.array([np.log10(file['massinrad']['subhalo'] * 1e10 / h)]) # Msun
                                else:
                                        if not 'InHalfRad' in param:
                                                data = np.array([np.log10(file['mass_'+component.get(str(int(param[-1])))]['subhalo'] * 1e10 / h)]) # Msun
                                        elif 'InHalfRad' in param:
                                                data = np.array([np.log10(file['massinhalfrad_'+component.get(str(int(param[-1])))]['subhalo'] * 1e10 / h)]) # Msun
                                        elif 'InRad' in param:
                                                data = np.array([np.log10(file['massinrad_'+component.get(str(int(param[-1])))]['subhalo'] * 1e10 / h)]) # Msun
                        elif 'HalfmassRad' in param:
                                if not 'Type' in param:
                                        data = np.array([np.log10(file['halfmassrad']['subhalo'] / (1+dfTime.z.values[:1]) / h)]) # kpc
                                else:
                                        data = np.array([np.log10(file['halfmassrad_'+component.get(str(int(param[-1])))]['subhalo']/ (1+dfTime.z.values[:1]) / h)]) # kpc
                        elif 'Pos' in param[:-1]:
                                if param[-1] == '1':
                                        data = np.array([file['pos_x']['subhalo'] / (1+dfTime.z.values[:1]) / h]) # kpc

                                elif param[-1] == '2':
                                        data = np.array([file['pos_y']['subhalo'] / (1+dfTime.z.values[:1]) / h]) # kpc

                                elif param[-1] == '3':
                                        data = np.array([file['pos_z']['subhalo'] / (1+dfTime.z.values[:1]) / h]) # kpc
                            
                        elif 'Vel' in param[:-1]:
                                if param[-1] == '1':
                                        data = np.array([file['vel_x']['subhalo']])  # km / s

                                elif param[-1] == '2':
                                        data = np.array([file['vel_y']['subhalo']])  # km / s

                                elif param[-1] == '3':
                                        data = np.array([file['vel_z']['subhalo']])  # km / s
                                        
                        
                        elif 'SubhaloStellarPhotometrics' in param[:-1]:
                            if 'U' in param[-1:]:
                                data = np.array([file['stellarphotometrics_u']['subhalo'] ])
                            if 'B' in param[-1:]:
                                data = np.array([file['stellarphotometrics_b']['subhalo'] ])
                            if 'V' in param[-1:]:
                                data = np.array([file['stellarphotometrics_v']['subhalo'] ])
                            if 'K' in param[-1:]:
                                data = np.array([file['stellarphotometrics_k']['subhalo'] ])
                            if 'g' in param[-1:]:
                                data = np.array([file['stellarphotometrics_g']['subhalo'] ])
                            if 'r' in param[-1:]:
                                data = np.array([file['stellarphotometrics_r']['subhalo'] ])
                            if 'i' in param[-1:]:
                                data = np.array([file['stellarphotometrics_i']['subhalo'] ])
                            if 'z' in param[-1:]:
                                data = np.array([file['stellarphotometrics_z']['subhalo'] ])
                            
                        
                        elif 'SFR' in param:
                                if 'sSFR' in param:
                                        sfr = file['sfr']['subhalo']
                                        if not 'InHalfRad' in param:
                                                mass = file['mass_stars']['subhalo'] * 1e10 / h # Msun
                                        elif 'InHalfRad' in param:
                                                mass = file['massinhalfrad_stars']['subhalo'] * 1e10 / h # Ms
                                        elif 'InRad' in param:
                                                mass = file['massinrad_stars']['subhalo'] * 1e10 / h # Msun
                                        if sfr == 0 :
                                                data = np.array([-14])
                                        else:
                                                data = np.array([np.log(sfr / mass)])  # sSFR
                                else:
                                        sfr = file['sfr']['subhalo']
                                        if sfr == 0 :
                                                data = np.array([-14])
                                        else:
                                                data = np.array([np.log(sfr / mass)])  # sSFR

                
                elif 'SubfindID' in param:
                        data = np.array([file['id']['subhalo']])

                elif 'Group' in param:
                        group = ETNG.gethalo(file['grnr']['subhalo'], simulation=SIM + '-1')

                        if 'FirstGroup' in param:
                                param = param.replace('FirstGroup', '')

                        elif 'FinalGroup' in param:
                                param = param.replace('FinalGroup', '')

                        if 'over_R_Crit200' in param:
                                posSubHalo = pos = np.array([file['pos_x']['subhalo'], 
                                        file['pos_y']['subhalo'] ,
                                        file['pos_z']['subhalo'] ])
                        
                                posHalo = group['GroupPos'] 
                                dr = ETNG.FixPeriodic(posSubHalo - posHalo, sim=SIM + NSim)  / (1+dfTime.z.values[:1]) / h
                        
                        elif 'Nsub' in param:
                                file = DownloadSubhalo(ID)
                                data = np.array([file[param][0]])
                        else:
                                data = np.array([group[param]])
                

        CleanAllSubhalos(ID)
        return data
    


def EvolutionDF(param, IDs, PATH=os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory', 
                SAVEFILE='DFs', SIM='TNG50', fmt='csv', verbose=False, CheckNewDATA = True):
    '''
    Make dataframe for the evolution of a given param and subhalos
    
    Parameters
    ----------
    param : param to make the evolution. str
    IDs : array with the subhalos ID. array with int
    
    Returns
    -------
    Requested dataframe
    
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    
    if verbose:
        print('Making df for the EVOLUTION of '+param)
    
    file_path = f"{PATH}/{SIM}/{SAVEFILE}/{param}.{fmt}"
    
    # Check if file exists
    if os.path.exists(file_path):
        if verbose:
            print(f"Loading existing dataframe from {file_path}")
        df = pd.read_csv(file_path)
    else:
        if verbose:
            print("Creating new dataframe")
        snaps = np.array([int(i) for i in np.arange(100)])
        df = pd.DataFrame(data=np.zeros((100, len(IDs) + 1), dtype=object), 
                         columns=np.append('Snap', IDs))
        df[:] = np.nan
        df.Snap = np.flip(snaps)
    
    # Process each ID
    if CheckNewDATA:
        for i, ID in enumerate(IDs):
            ID_str = str(ID)
            
            # Check if this ID column exists and has valid data
            if ID_str in df.columns:
                ValidationArray = np.array([value for value in df[ID_str].values])
                nan_count = len(df[ID_str].values[np.isnan(ValidationArray)])
                
                # If we have enough valid data or it's 'Nsub' param, skip
                if nan_count <= 99 and not 'Nsub' in param:
                    if verbose:
                        print(f"ID {ID} already has sufficient data, skipping...")
                    continue
            
            # If we get here, we need to process this ID
            print(f'Processing Subhalo: {ID}')
            
            try:
                data = ImportField(param, ID)
            except Exception as e:
                print(f"Exception: {e}")
                data = np.array([])
            
            # Ensure data has length 100
            while len(data) < 100:
                data = np.append(data, np.nan)
            
            # Add or update the column
            df[ID_str] = data
            
            # Save after processing each ID to preserve progress
            if verbose:
                print(f'Saving progress after ID {ID}...')
                
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            try:
                df.to_csv(file_path)
            except Exception: 
                # Create directories if they don't exist
                path = PATH
                NewPATH = f"{SIM}/{SAVEFILE}/"
                directories = NewPATH.split('/')
                for name in directories:
                    path = os.path.join(path, name)
                    if not os.path.isdir(path):
                        os.mkdir(path)
                df.to_csv(file_path)
    
    return df

def EvolutionComposeDF(param, IDs, PATH = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory',  SAVEFILE = 'DFs', SIM = 'TNG50', fmt = 'csv', verbose = False):
    '''
    Make dataframe for the evolution of a given param and subhalos
    Parameters
    ----------
    param : param to make the evolution. str
    IDs : array with the subhalos ID. array with int
    Returns
    -------
    Requested dataframe
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    if verbose:
        print('Making df for the EVOLUTION of '+param)
        
    try:
        df = extractDF(param, SIM = SIM)
        NewData = False

    except Exception:
        if param == 'sSFRCoreRatio':
            dfsSFRInHalfRad = EvolutionDF('SubhalosSFRinHalfRad', IDs, SIM = SIM)
            dfsSFR = EvolutionDF('SubhalosSFR', IDs, SIM = SIM)

            df = dfsSFR.copy()
            for key in dfsSFR.keys():
                if key == 'Snap':
                    continue
                else:
                    sSFRInHalfRad = np.array([value for value in dfsSFRInHalfRad[key]])
                    sSFR = np.array([value for value in dfsSFR[key]])

                    Ratio = 10**sSFRInHalfRad / (10**sSFR - 10**sSFRInHalfRad)
                    Ratio[sSFR == -14] = np.nan
                    Ratio[sSFRInHalfRad == -14] = np.nan
                    Ratio[np.isnan(sSFRInHalfRad)] = np.nan
                    Ratio[np.isnan(sSFR)] = np.nan
                    Ratio[np.isinf(sSFRInHalfRad)] = np.nan
                    Ratio[np.isinf(sSFR)] = np.nan


                    df[key] = Ratio
                    
        elif param == 'sSFR_Outer':
            dfsSFRInHalfRad = EvolutionDF('SubhalosSFRinHalfRad', IDs, SIM = SIM)
            dfsSFR = EvolutionDF('SubhalosSFR', IDs, SIM = SIM)

            df = dfsSFR.copy()
            for key in dfsSFR.keys():
                if key == 'Snap':
                    continue
                else:
                    sSFRInHalfRad = np.array([value for value in dfsSFRInHalfRad[key]])
                    sSFR = np.array([value for value in dfsSFR[key]])

                    y = 10**sSFR - 10**sSFRInHalfRad
                    y = np.log10(y)
                    y[sSFR == -14] = -14
                    y[sSFRInHalfRad == -14] = -14
                    y[np.isnan(sSFRInHalfRad)] = np.nan
                    y[np.isnan(sSFR)] = np.nan
                    y[np.isinf(sSFRInHalfRad)] = np.nan
                    y[np.isinf(sSFR)] = np.nan


                    df[key] = y

        if 'Norm_Max' in param:
            if 'gas' in param:
                dfMass = EvolutionDF('SubhaloMassType0', IDs, SIM = SIM)
            elif 'DM' in param:
                dfMass = EvolutionDF('SubhaloMassType1', IDs, SIM = SIM)
            elif 'star' in param:
                dfMass = EvolutionDF('SubhaloMassType4', IDs, SIM = SIM)

            df = dfMass.copy()
            for key in dfMass.keys():
                if key == 'Snap':
                    continue
                else:
                    Mass = np.array([10**value for value in dfMass[key]])
                    MassMax = np.nanmax(Mass[:int(99 - 20)])

                    values = Mass / MassMax
                    values[np.isnan(Mass)] = np.nan
                    values[np.isinf(Mass)] = np.nan

                    df[key] = values
                    

        NewData = True

    if NewData:
        if verbose:
            print('Savefing...')

        try:
            df.to_csv(PATH+'/'+SIM+'/'+SAVEFILE+'/'+param+'.'+fmt)       
        except Exception: 
            path = PATH
            NewPATH = SIM+'/'+SAVEFILE+'/'
            directories = NewPATH.split('/')
            for name in directories:
                path = os.path.join(path, name)
                if not os.path.isdir(path):
                    os.mkdir(path)
            df.to_csv(PATH+'/'+SIM+'/'+SAVEFILE+'/'+param+'.'+fmt)       
    
    return df


def EvolutionParticle(Params, IDs, dfSample, SizeLim = 'Rhpkpc', slim = [0.5, 1.5, 4.5, 9., 12.], 
                      Snaps = [2, 3, 4,6, 8,11,17,21, 24, 25,33,36,37, 38, 40, 44, 46, 50, 51,59,63,  64, 67,72,78, 81, 84,88, 91,99],
                      Mult = [1, 2, 5], Update = False,  NearBirth = False,
                       NearEntryToGasLoss= False, EntrySnap = False,
                      PATH = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory',  
                      SAVEFILE = 'DFs', SIM = 'TNG50',  NSim = '-1', fmt = 'csv', verbose = False):
    '''
    Make dataframe for the evolution of a given param and subhalos
    Parameters
    ----------
    param : param to make the evolution. str
    IDs : array with the subhalos ID. array with int
    Returns
    -------
    Requested dataframe
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    SubfindID = extractDF('SubfindID')
    SubhaloHalfmassRadType0 = extractDF('SubhaloHalfmassRadType0')
    SubhaloHalfmassRadType4 = extractDF('SubhaloHalfmassRadType4')
    SubhaloHalfmassRadType1 = extractDF('SubhaloHalfmassRadType1')
    
    GroupFirstSub  = extractDF('GroupFirstSub')
    Group_M_Crit200  = extractDF('Group_M_Crit200')
    Group_R_Crit200  = extractDF('Group_R_Crit200')


    DFs = []
    
    if SizeLim == 'Rhpkpc' or SizeLim == 'RhpkpcDiffuse'  or SizeLim == 'RhOver2pkpc' or SizeLim == 'TrueRhpkpc':
        
        NewParams = []
        
        for Param in Params:
            #NewParams.append(Param)
            NewParams.append(Param+'_In_'+SizeLim)
            NewParams.append(Param+'_Above_'+SizeLim)

        Params = NewParams
        
    if SizeLim[2:] == 'XRhpkpcCurrent' or SizeLim == 'AbovestdRhpkpcCurrent':
        
        NewParams = []
        
        for Param in Params:
            NewParams.append(Param+'_Above_'+SizeLim)

        Params = NewParams
        
    
    if SizeLim == 'Rhpkpc_entry':
        
        NewParams = []
        
        for Param in Params:
            #NewParams.append(Param+'_In_'+SizeLim+'_minus200dex')
            #NewParams.append(Param+'_In_'+SizeLim+'_minus150dex')
            #NewParams.append(Param+'_In_'+SizeLim+'_minus100dex')
            #NewParams.append(Param+'_In_'+SizeLim+'_minus050dex')
            #NewParams.append(Param+'_In_'+SizeLim+'_minus025dex')
            NewParams.append(Param+'_In_'+SizeLim)
            #NewParams.append(Param+'_In_'+SizeLim+'_plus050dex')
            #NewParams.append(Param+'_In_'+SizeLim+'_plus025dex')
            #NewParams.append(Param+'_In_'+SizeLim+'_plus100dex')
            #NewParams.append(Param+'_minus050dex_r_'+SizeLim)
            #NewParams.append(Param+'_'+SizeLim+'_r_plus050dex')
            NewParams.append(Param+'_Above_'+SizeLim)


        
        Params = NewParams
        
    if SizeLim == 'RhpkpcInDex':
        
        NewParams = []
        
        for Param in Params:
            NewParams.append(Param+'_In_'+SizeLim+'_minus100dex')
            NewParams.append(Param+'_In_'+SizeLim+'_minus050dex')
            NewParams.append(Param+'_In_'+SizeLim+'_minus025dex')
            NewParams.append(Param+'_In_'+SizeLim)
            NewParams.append(Param+'_In_'+SizeLim+'_plus025dex')
            NewParams.append(Param+'_In_'+SizeLim+'_plus050dex')
            NewParams.append(Param+'_In_'+SizeLim+'_plus100dex')
        
        Params = NewParams
        
    if SizeLim == 'RhpkpcAboveDex':
        
        NewParams = []
        
        for Param in Params:
            NewParams.append(Param+'_Above_'+SizeLim+'_minus100dex')
            NewParams.append(Param+'_Above_'+SizeLim+'_minus050dex')
            NewParams.append(Param+'_Above_'+SizeLim+'_minus025dex')
            NewParams.append(Param+'_Above_'+SizeLim)
            NewParams.append(Param+'_Above_'+SizeLim+'_plus025dex')
            NewParams.append(Param+'_Above_'+SizeLim+'_plus050dex')
            NewParams.append(Param+'_Above_'+SizeLim+'_plus100dex')
        
        Params = NewParams
        
        
    if SizeLim == 'InRhpkpc':
        
        NewParams = []
        
        for Param in Params:
            NewParams.append(Param+'_In_Rhpkpc')
            NewParams.append(Param+'_In_2Rhpkpc')
            NewParams.append(Param+'_In_5Rhpkpc')
            NewParams.append(Param+'_In_10Rhpkpc')
            NewParams.append(Param+'_In_20Rhpkpc')

            Mult = [1, 2, 5, 10, 20]
       
       
        Params = NewParams
        
    if SizeLim == 'Inrpkpc':
        
        NewParams = []
        
        for Param in Params:
            #NewParams.append(Param+'_In_07pkpc')
            NewParams.append(Param+'_In_2pkpc')
            NewParams.append(Param+'_In_5pkpc')
            #NewParams.append(Param+'_In_12pkpc')
            #NewParams.append(Param+'_In_25pkpc')
            #NewParams.append(Param+'_In_50pkpc')
            #NewParams.append(Param+'_In_80pkpc')
            #NewParams.append(Param+'_In_100pkpc')

            Mult = [2, 5]
       
            
        Params = NewParams
        
    if SizeLim == 'MultRhpkpc':
        
        NewParams = []
        
        for Param in Params:
            NewParams.append(Param+'_In_'+SizeLim+'_plus000dex')
            NewParams.append(Param+'_In_'+SizeLim+'_plus015dex')
            NewParams.append(Param+'_In_'+SizeLim+'_plus025dex')
            NewParams.append(Param+'_In_'+SizeLim+'_plus050dex')
            NewParams.append(Param+'_In_'+SizeLim+'_plus075dex')

        
        Params = NewParams
    
    
    elif SizeLim == 'rpkpc':
        
        NewParams = []
        
        for Param in Params:
            for i, size in enumerate(slim):
                if i == 0:
                    NewParams.append(Param+'_In_'+str(size).replace('.', '')+SizeLim)
                elif i != len(slim):
                    NewParams.append(Param+'_'+str(slim[i -1]).replace('.', '')+'_r_'+str(size).replace('.', '')+SizeLim)
            NewParams.append(Param+'_Above_'+str(size).replace('.', '')+SizeLim)
        
        Params = NewParams
        
        
    dfTime.a = 1 / (1 + dfTime.z)

    
    for Param in Params:
        if verbose:
            print('Making df for the EVOLUTION of '+Param)
            
        try:
            df = extractDF(Param)
            keysSnaps = np.array([int(s) for s in df.keys().values[1:]])
            print('Making df')
            for ID in IDs:
                if not ID in keysSnaps:
                    df[str(ID)] = np.nan
            
            
        except Exception:
            print('NaN df')
            snaps= np.arange(100)
            df = pd.DataFrame(data=np.zeros((100,len(IDs) + 1), dtype=object), columns= np.append('Snap', IDs))
            df[:] = np.nan
            df.Snap = np.flip(snaps)   
                
        DFs.append(df)
        Snaps_Original = Snaps
    for j, ID in enumerate(IDs):
        print('ID: ',ID)

        if SizeLim == 'Rhpkpc'  or SizeLim == 'InRhpkpc' or SizeLim == 'TrueRhpkpc'  or  SizeLim == 'RhOver2pkpc' or SizeLim == 'RhpkpcDiffuse' :
            rh = 10**dfSample.loc[dfSample.SubfindID_99 == ID, 'logHalfRadstar_99'].values[0]
            if SizeLim == 'RhOver2pkpc':
                rh = 10**dfSample.loc[dfSample.SubfindID_99 == ID, 'logHalfRadstar_99'].values[0] / 2.
            
        elif SizeLim == 'MultRhpkpc' or SizeLim == 'RhpkpcInDex' or SizeLim == 'RhpkpcAboveDex':
            rh = dfSample.loc[dfSample.SubfindID_99 == ID, 'logHalfRadstar_99'].values[0]
            rHStarArray = np.array([2*10**v for v in SubhaloHalfmassRadType4[str(ID)].values])

        elif SizeLim == 'Rhpkpc_entry':
            
            rh = dfSample.loc[dfSample.SubfindID_99 == ID, 'Snap_At_FirstEntry'].values[0]
            rh = SubhaloHalfmassRadType4[str(ID)].values[int(99 - rh)]

        elif SizeLim == '2RhpkpcCurrent' or  SizeLim == 'AbovestdRhpkpcCurrent':
            rHStarArray = np.array([2*10**v for v in SubhaloHalfmassRadType4[str(ID)].values])
            rHDMArray = np.array([2*10**v for v in SubhaloHalfmassRadType1[str(ID)].values])
            
        elif SizeLim == 'RhpkpcCurrent' or  SizeLim == 'AbovestdRhpkpcCurrent':
            rHStarArray = np.array([10**v for v in SubhaloHalfmassRadType4[str(ID)].values])
            rHDMArray = np.array([10**v for v in SubhaloHalfmassRadType1[str(ID)].values])
            
        elif SizeLim[2:] == 'XRhpkpcCurrent':
            print('Times: '+str(int(SizeLim[:2])))
            rHStarArray = np.array([int(SizeLim[:2])*10**v for v in SubhaloHalfmassRadType4[str(ID)].values])
            rHDMArray = np.array([int(SizeLim[:2])*10**v for v in SubhaloHalfmassRadType1[str(ID)].values])
            
        elif 'Corotate' in Param or 'Counterrotate' in Param:
            rHStarArray = np.array([2*10**v for v in SubhaloHalfmassRadType4[str(ID)].values])

        else:
            rHStarArray = np.array([2*10**v for v in SubhaloHalfmassRadType4[str(ID)].values])

        Snaps = Snaps_Original
        if EntrySnap:
            if ID in np.array([  232,    261,    281,    300,    319,    333,  63990,  63993,
                    64002,  64081,  64129,  96853,  96941, 117357, 117464, 144008,
                   144098, 167499, 184957, 185005, 185058, 208883, 220626, 220633,
                   229992, 229996, 242863, 253897, 253905, 253965, 264911, 264972,
                   275601, 282802, 282807, 289444, 294887, 294895, 307510, 319738,
                   319743, 338455, 358627, 377662, 386293, 394628, 404834, 421566,
                   421567, 422763, 422770, 425726, 428191, 432119, 436937, 457435,
                   467420, 482157, 500583, 502998, 516761, 530853, 536657, 545439,
                   549748, 549750, 558069, 560083, 571075, 571910, 586424, 588180,
                   602131, 602132, 602133, 603005, 647769]):
                snap_at_entry = dfSample.loc[dfSample.SubfindID_99 == ID, 'Snap_At_FirstEntry'].values[0]
                snap_no_gas = dfSample.loc[dfSample.SubfindID_99 == ID, 'SnapLostGas'].values[0]
                if snap_at_entry > 3:
                    snap_at_entry = snap_at_entry -3
                if snap_no_gas == -1:
                    add_snaps = np.arange(snap_at_entry, 99)
                else:
                    add_snaps = np.arange(snap_at_entry, snap_no_gas)
                    
                Snaps = np.append(Snaps, add_snaps)
                Snaps = np.unique(Snaps)
                if snap_no_gas == -1:
                    snap_no_gas = 99
                Snaps = Snaps[Snaps <= snap_no_gas]
                Snaps = np.array([int(s) for s in Snaps])
                
        
        
        ParamUpdate = np.array([])
        for l, Param in enumerate(Params):
            testNan = np.array([v for v in DFs[l][str(ID)].values])
            if len(testNan[~ np.isnan(testNan)]) <= len(Snaps):
                ParamUpdate = np.append(ParamUpdate, Param)
        
        if len(ParamUpdate) == len(Param) and not Update:
            continue
        
        if NearBirth:
            IDsSnaps = np.array([subID for subID in SubfindID[str(ID)]])
            argBirth = np.argwhere(~np.isnan(IDsSnaps)).T[0][-1]
            
            SnapsAt0 = dfTime.Snap.loc[(dfTime.Age - dfTime.Age.loc[dfTime.Snap == int(99 - argBirth)].values[0] < 0.15) & (dfTime.Age - dfTime.Age.loc[dfTime.Snap == int(99 - argBirth)].values[0] > 0) &  (dfTime.Snap > int(99 - argBirth))].values
            SnapsAt1 = dfTime.Snap.loc[(dfTime.Age - dfTime.Age.loc[dfTime.Snap == int(99 - argBirth)].values[0] < 1.1) & (dfTime.Age - dfTime.Age.loc[dfTime.Snap == int(99 - argBirth)].values[0] > 0.9) &  (dfTime.Snap > int(99 - argBirth))].values
            #SnapsAt2 = dfTime.Snap.loc[(dfTime.Age - dfTime.Age.loc[dfTime.Snap == int(99 - argBirth)].values[0] < 2.25) & (dfTime.Age - dfTime.Age.loc[dfTime.Snap == int(99 - argBirth)].values[0] > 1.75) &  (dfTime.Snap > int(99 - argBirth))].values
            #SnapsAt4 = dfTime.Snap.loc[(dfTime.Age - dfTime.Age.loc[dfTime.Snap == int(99 - argBirth)].values[0] < 4.25) & (dfTime.Age - dfTime.Age.loc[dfTime.Snap == int(99 - argBirth)].values[0] > 3.75) &  (dfTime.Snap > int(99 - argBirth))].values
            Snaps = np.concatenate([[int(99 - argBirth)], SnapsAt0, SnapsAt1])#, SnapsAt2, SnapsAt4])
            Snaps = np.unique(Snaps)

        if NearEntryToGasLoss:
            snapAtentry = dfSample.loc[dfSample.SubfindID_99 == ID, 'SnapAtEntry_First'].values[0]
            
                            
            snap_no_gas = dfSample.loc[dfSample.SubfindID_99 == ID, 'SnapLostGas'].values[0]
            
            
            if snapAtentry > 5:
                snap_at_entry = snapAtentry -5
            if np.isnan(snapAtentry):
                snapAtentry = 90
            if snap_no_gas == -1 or np.isnan(snap_no_gas):
                snap_no_gas = 99
                add_snaps = np.arange(snap_at_entry, 99)
            else:
                if snap_no_gas > 10 + snap_at_entry:
                    add_snaps = np.linspace(snap_at_entry, snap_no_gas, 10)
                else:
                    add_snaps = np.arange(snap_at_entry, snap_no_gas)
                
            add_snaps = np.append(add_snaps, snapAtentry)
            add_snaps = np.append(add_snaps, snap_at_entry)
            add_snaps = np.append(add_snaps, snap_no_gas)

            SnapDeltaBefore = dfTime.Snap.loc[abs(dfTime.Age - dfTime.Age.loc[dfTime.Snap == snapAtentry].values[0]) < abs(dfTime.Age.loc[dfTime.Snap == snap_no_gas].values[0] - dfTime.Age.loc[dfTime.Snap == snapAtentry].values[0])]
            SnapDeltaBefore = SnapDeltaBefore[SnapDeltaBefore < snapAtentry]
            add_snaps = np.append(add_snaps, SnapDeltaBefore)

            Snaps = add_snaps
            Snaps = np.unique(Snaps)
            Snaps = np.array([int(s) for s in Snaps])
            Snaps = Snaps[Snaps < 100]
            Snaps = Snaps[Snaps > 0]

            
        for snap in Snaps: 
            
            if not Update and ~ np.isnan(DFs[l][str(ID)].values[99 - snap]):
                print('Not Update, snap: ', snap)
                continue
        
            
            
            print('snap: ', snap)
            
            try:
                f = extractParticles(ID, snaps = [snap])[0]
            except Exception:
                print('Failed to get Particle')
                continue
            
            scalefactorsqrt = np.sqrt(1. / (1+dfTime.z[int(99-snap)]))


            try:
                IDsGas_Current = f['PartType0']['ParticleIDs'][:]
                posGas = f['PartType0']['Coordinates'][:]  / (1+dfTime.z.values[dfTime.Snap == snap]) / h #kpc
                massGas = f['PartType0']['Masses'][:] * 1e10 / h
                velGas= f['PartType0']['Velocities'][:] * scalefactorsqrt
                
            except Exception:
                IDsGas_Current = np.array([0])
                posGas = np.array([[0, 0, 0]])
                massGas = np.array([0])
                velGas=  np.array([[0, 0, 0]])
                
            try:
                IDsDM_Current = f['PartType1']['ParticleIDs'][:]

                posDM = f['PartType1']['Coordinates'][:]  / (1+dfTime.z.values[dfTime.Snap == snap]) / h #kpc
                massDM = f['Header'].attrs['MassTable'][1]*np.ones(len(f['PartType1']['Coordinates'])) * 1e10 / h
                velDM =  f['PartType1']['Velocities'][:] * scalefactorsqrt

            except Exception:
                posDM = np.array([[0, 0, 0]])
                IDsDM_Current = massDM = np.array([0])
                velDM =  np.array([[0, 0, 0]])

            try:
                IDsStar_Current = f['PartType4']['ParticleIDs'][:]

                posStar = f['PartType4']['Coordinates'][:]  / (1+dfTime.z.values[dfTime.Snap == snap]) / h #kpc
                velStar= f['PartType4']['Velocities'][:] * scalefactorsqrt
                massStar = f['PartType4']['Masses'][:] * 1e10 / h
                ZStar = f['PartType4']['GFM_Metallicity'][:] /  0.0127
                AgeStar = ETNG.AgeUniverse(Omegam0,h,1/(f['PartType4']['GFM_StellarFormationTime'][:]) - 1)
            except Exception:
                posStar = np.array([[0, 0, 0]])
                IDsStar_Current = massStar = np.array([0])
                velStar =  np.array([[0, 0, 0]])
            
            
            if not (massStar[0] == 0 and len(massStar) == 1):
                Cen = np.array([MATH.weighted_median(posStar[:, 0], massStar), MATH.weighted_median(posStar[:, 1], massStar), MATH.weighted_median(posStar[:, 2], massStar)])
                VelBulk = np.array([MATH.weighted_median(velStar[:, 0], massStar), MATH.weighted_median(velStar[:, 1], massStar), MATH.weighted_median(velStar[:, 2], massStar)])
            elif not (massGas[0] == 0 and len(massGas) == 1):
                Cen = np.array([MATH.weighted_median(posGas[:, 0], massGas), MATH.weighted_median(posGas[:, 1], massGas), MATH.weighted_median(posGas[:, 2], massGas)])
                VelBulk = np.array([MATH.weighted_median(velGas[:, 0], massGas), MATH.weighted_median(velGas[:, 1], massGas), MATH.weighted_median(velGas[:, 2], massGas)])
            elif not (massDM[0] == 0 and len(massDM) == 1):
                Cen = np.array([MATH.weighted_median(posDM[:, 0], massDM), MATH.weighted_median(posDM[:, 1], massDM), MATH.weighted_median(posDM[:, 2], massDM)])
                VelBulk = np.array([MATH.weighted_median(velDM[:, 0], massDM), MATH.weighted_median(velDM[:, 1], massDM), MATH.weighted_median(velDM[:, 2], massDM)])


            posGas = ETNG.FixPeriodic(posGas - Cen , sim=SIM + NSim)
            posDM =  ETNG.FixPeriodic(posDM - Cen, sim=SIM + NSim)
            posStar =  ETNG.FixPeriodic(posStar - Cen, sim=SIM + NSim)
            
            velStar = velStar - VelBulk
            velGas = velGas - VelBulk
            velDM = velDM - VelBulk
            
        
            rGas = np.linalg.norm(posGas, axis=1)
            rStar = np.linalg.norm(posStar, axis=1)
            rDM = np.linalg.norm(posDM, axis=1)
            
            velRadStar = np.sum(posStar*velStar, axis = 1)/rStar
            velRadGas = np.sum(posGas*velGas, axis = 1)/rGas
            velRadDM = np.sum(posDM*velDM, axis = 1)/rDM
            
            Mass = np.append(massGas, massDM) #Msun
            Mass = np.append(Mass, massStar)
            
            Pos = np.vstack([posGas, posDM])
            Pos = np.vstack([Pos, posStar])
            
            Vel = np.vstack([velGas, velDM])
            Vel = np.vstack([Vel, velStar])
            
            R = np.append(rGas, rDM) # kpc
            R = np.append(R, rStar)
            
            V = np.append(velRadGas, velRadDM)
            V = np.append(V, velRadStar)
            
            J = Mass*np.linalg.norm(np.cross(Pos, Vel), axis = 1)
            JVec = Mass[:, np.newaxis]*np.cross(Pos, Vel)
            
            
            
            sizeCount = 0

            
            for l, Param in enumerate(Params):
                
                
                
                DFs[l].to_csv(PATH+'/'+SIM+'/'+SAVEFILE+'/'+Param+'.csv')
                
                
                if not Param in ParamUpdate:
                    print('ParamUpdate')

                    continue
                
                if 'SFR' in Param:
                    try:
                        SFR =  f['PartType0']['StarFormationRate'][:] 
                    except Exception:
                        SFR = np.array([0])
                Cond = None
                
                if SizeLim == 'Rhpkpc' :
                    if '_In_' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas <= 2*rh)
                            if 'sSFR' in Param:
                                CondStar = (rStar <= 2*rh)
                        elif 'Star' in Param:
                            Cond = (rStar <= rh)
                        elif 'DM' in Param:
                            Cond = (rDM <= 2*rh)
                        else:
                            Cond = (R <= rh)
                    elif 'Above' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas > 2*rh)
                            if 'sSFR' in Param:
                                CondStar = (rStar > 2*rh)
                        elif 'Star' in Param:
                            Cond = (rStar > rh)
                        elif 'DM' in Param:
                            Cond = (rDM > 2*rh)
                        else:
                            Cond = (R >rh)
                elif SizeLim == 'TrueRhpkpc' :
                    if '_In_' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas <= rh)
                            if 'sSFR' in Param:
                                CondStar = (rStar <= rh)
                            
                        elif 'Star' in Param:
                            Cond = (rStar <= rh)
                        elif 'DM' in Param:
                            Cond = (rDM <= rh)
                        else:
                            Cond = (R <= rh)
                    elif 'Above' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas > rh)
                            if 'sSFR' in Param:
                                CondStar = (rStar > rh)
                        elif 'Star' in Param:
                            Cond = (rStar > rh)
                        elif 'DM' in Param:
                            Cond = (rDM > rh)
                        else:
                            Cond = (R >rh)
                elif SizeLim == 'RhpkpcDiffuse' :
                    if '_In_' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas <= rh)
                            if 'sSFR' in Param:
                                CondStar = (rStar <= rh)
                        elif 'Star' in Param:
                            Cond = (rStar <= rh)
                        elif 'DM' in Param:
                            Cond = (rDM <= rh)
                        else:
                            Cond = (R <= rh)
                    elif 'Above' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas > rh)
                            if 'sSFR' in Param:
                                CondStar = (rStar > rh)
                        elif 'Star' in Param:
                            Cond = (rStar > rh)
                        elif 'DM' in Param:
                            Cond = (rDM > rh)
                        else:
                            Cond = (R >rh)     
                            
                elif SizeLim == 'RhOver2pkpc' :
                    if '_In_' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas <= rh)
                            if 'sSFR' in Param:
                                CondStar = (rStar <= rh)
                        elif 'Star' in Param:
                            Cond = (rStar <= rh)
                        elif 'DM' in Param:
                            Cond = (rDM <= rh)
                        else:
                            Cond = (R <= rh)
                    elif 'Above' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas > rh)
                            if 'sSFR' in Param:
                                CondStar = (rStar > rh)
                        elif 'Star' in Param:
                            Cond = (rStar > rh)
                        elif 'DM' in Param:
                            Cond = (rDM > rh)
                        else:
                            Cond = (R >rh)
                            
                            
                            
                elif  SizeLim == 'Rhpkpc_entry' or  SizeLim == 'RhpkpcInDex':
                     if '_In_' in Param and 'minus' in Param:
                         if 'Gas' in Param or 'SFR' in Param:
                             Cond = (rGas <= 10**(rh - int(Param[-6:-3]) / 100))
                             if 'sSFR' in Param:
                                 CondStar = (rStar <=  10**(rh - int(Param[-6:-3]) / 100))
                         elif 'Star' in Param:
                             Cond = (rStar <=   10**(rh - int(Param[-6:-3]) / 100))
                         elif 'DM' in Param:
                             Cond = (rDM <=  10**(rh - int(Param[-6:-3]) / 100))
                             
   
                         
                     elif '_In_' in Param and 'plus' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas <= 10**(rh + int(Param[-6:-3]) / 100))
                            if 'sSFR' in Param:
                                CondStar = (rStar <=  10**(rh + int(Param[-6:-3]) / 100))
                        elif 'Star' in Param:  
                            Cond = (rStar <= 10**(rh + int(Param[-6:-3]) / 100))
                        elif 'DM' in Param:
                            Cond = (rDM <=  10**(rh + int(Param[-6:-3]) / 100))
  
                     elif '_In_' in Param :
                       if 'Gas' in Param or 'SFR' in Param:
                           Cond = (rGas <= 10**(rh))
                           if 'sSFR' in Param:
                               CondStar = (rStar <=  10**(rh))
                       elif 'Star' in Param:
                           Cond = (rStar <=  10**(rh))
                       elif 'DM' in Param:
                           Cond = (rDM <=  10**(rh))
                         
                     elif '_Above_' in Param:
                         if 'Gas' in Param or 'SFR' in Param:
                             Cond = (rGas > 10**(rh))
                             if 'sSFR' in Param:
                                 CondStar = (rStar >  10**(rh))
                         elif 'Star' in Param:
                             Cond = (rStar >  10**(rh))
                         elif 'DM' in Param:
                             Cond = (rDM > 10**(rh))
                             
                     else:
                         if 'minus' in Param:
                             if 'Gas' in Param or 'SFR' in Param:
                                 Cond = (rGas >10**(rh - 0.5)) & (rGas <= 10**(rh))
                                 if 'sSFR' in Param:
                                     CondStar = (rStar >10**(rh - 0.5)) & (rStar <= 10**(rh))
                             elif 'Star' in Param:
                                 Cond = (rStar >10**(rh - 0.5)) & (rStar <= 10**(rh))
                             elif 'DM' in Param: 
                                 Cond = (rDM >10**(rh - 0.5)) & (rDM <= 10**(rh))
                             sizeCount = sizeCount + 1
                             
                         elif 'plus' in Param:
                            if 'Gas' in Param or 'SFR' in Param:
                                Cond = (rGas > 10**(rh) ) & (rGas <= 10**(rh + 0.5))
                                if 'sSFR' in Param:
                                    CondStar = (rStar > 10**(rh) ) & (rStar <= 10**(rh + 0.5))
                            elif 'Star' in Param:
                                Cond = (rStar > 10**(rh)) & (rStar <= 10**(rh + 0.5))
                            elif 'DM' in Param: 
                                Cond = (rDM > 10**(rh) ) & (rDM <= 10**(rh + 0.5))
                            sizeCount = sizeCount + 1
                  
                elif  SizeLim == 'RhpkpcAboveDex':
                     
                     if '_Above_' in Param and 'dex' in Param :
                       if 'Gas' in Param or 'SFR' in Param:
                           Cond = (rGas >= 10**(rh + int(Param[-6:-3])))
                           
                           if 'sSFR' in Param:
                               CondStar = (rStar  >= 10**(rh + int(Param[-6:-3])))
                       elif 'Star' in Param:
                           Cond = (rStar  >= 10**(rh + int(Param[-6:-3])))
                       elif 'DM' in Param:
                           Cond = (rDM  >= 10**(rh + int(Param[-6:-3])))
                       if 'Corotate' in Param or 'Counterrotate' in Param:
                           CondJ = (R <=  rHStarArray[99 - snap])
                         
                     else:
                      
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas >= 10**(rh))
                            if 'sSFR' in Param:
                                CondStar = (rStar  >= 10**(rh))
                        elif 'Star' in Param:
                            Cond = (rStar  >= 10**(rh))
                        elif 'DM' in Param:
                            Cond = (rDM  >= 10**(rh))
                        if 'Corotate' in Param or 'Counterrotate' in Param:
                            CondJ = (R <=  rHStarArray[99 - snap])
                            
                elif SizeLim == '2RhpkpcCurrent' or SizeLim[2:] == 'XRhpkpcCurrent' or SizeLim == 'RhpkpcCurrent':
                    if 'Gas' in Param or 'SFR' in Param:
                        Cond = (rGas <= rHStarArray[99 - snap])    
                        if 'jAngle_Gas_Star' in Param:
                            CondStar = (rStar <= rHStarArray[99 - snap])    
                            
                        if 'Corotate' in Param or 'Counterrotate' in Param:
                            CondJ = (R <=  rHStarArray[99 - snap])
                        
                    elif 'Star' in Param:
                        Cond = (rStar <= rHStarArray[99 - snap])
                        if 'Corotate' in Param or 'Counterrotate' in Param:
                            CondJ = (R <=   rHStarArray[99 - snap])
                    elif 'DM' in Param:
                        Cond = (rDM <= rHStarArray[99 - snap])
                        if 'Corotate' in Param or 'Counterrotate' in Param:
                            CondJ = (R <=   rHStarArray[99 - snap])
                   
                elif SizeLim == 'AbovestdRhpkpcCurrent':
                    if 'Gas' in Param or 'SFR' in Param:
                        RadGasMedian = MATH.weighted_median(rGas, massGas)
                        sigmaRadGasMedian = MATH.boostrap_func(rGas, func=np.nanmedian)
                        Cond = (rGas >= np.min([RadGasMedian + 3*np.std(rGas), RadGasMedian +  3*sigmaRadGasMedian]))    
                        if 'Corotate' in Param or 'Counterrotate' in Param:
                            CondJ = (R <= rHStarArray[99 - snap])
                    elif 'Star' in Param:
                        Cond = (rStar >= 3*np.std(rStar))    
                    elif 'DM' in Param:
                        Cond = (rDM >= 3*np.std(rDM))   
                        if 'Corotate' in Param or 'Counterrotate' in Param:
                            CondJ = (R <=   rHDMArray[99 - snap])

                        
                elif SizeLim == 'InRhpkpc':
                    if not Mult[sizeCount] == None:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas <= Mult[sizeCount]*rh)
                            if 'sSFR' in Param:  
                                CondStar = (rStar <= Mult[sizeCount]*rh)
                        elif 'Star' in Param:
                            Cond = (rStar <= Mult[sizeCount]*rh)
                        elif 'DM' in Param:
                            Cond = (rDM <= Mult[sizeCount]*rh)
                            
                        else:
                            Cond = (R <= Mult[sizeCount]*rh)
                            
                        sizeCount = sizeCount + 1
                        if sizeCount == len(Mult):
                            sizeCount = 0
                        
                    else:
                        Cond = None
                        
                elif SizeLim == 'Inrpkpc':
                    if not Mult[sizeCount] == None:
                        rh = Mult[sizeCount]
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas <= Mult[sizeCount])
                            if 'sSFR' in Param:
                                CondStar = (rStar <= Mult[sizeCount])
                        elif 'Star' in Param:
                            Cond = (rStar <= Mult[sizeCount])
                        elif 'DM' in Param:
                            Cond = (rDM <= Mult[sizeCount])
                            
                        else:
                            Cond = (R <= Mult[sizeCount])
                            
                        sizeCount = sizeCount + 1
                        if sizeCount == len(Mult):
                            sizeCount = 0
                        
                    else:
                        Cond = None
                    
                 
                elif SizeLim == 'MultRhpkpc':
                    if '_In_' in Param and 'minus' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas <= 10**(rh - int(Param[-6:-3]) / 100))
                            if 'sSFR' in Param:
                                CondStar = (rStar <=  10**(rh - int(Param[-6:-3]) / 100))
                        elif 'Star' in Param:
                            Cond = (rStar <=   10**(rh - int(Param[-6:-3]) / 100))
                        elif 'DM' in Param:
                            Cond = (rDM <=  10**(rh - int(Param[-6:3]) / 100))
                            
  
                        
                    elif '_In_' in Param and 'plus' in Param:
                       if 'Gas' in Param or 'SFR' in Param:
                           Cond = (rGas <= 10**(rh + int(Param[-6:-3]) / 100))
                           if 'sSFR' in Param:
                               CondStar = (rStar <=  10**(rh + int(Param[-5:-3]) / 100))
                       elif 'Star' in Param:
                           Cond = (rStar <= 10**(rh + int(Param[-6:-3]) / 100))
                       elif 'DM' in Param:
                           Cond = (rDM <=  10**(rh + int(Param[-6:-3]) / 100))
 
                    elif '_In_' in Param :
                      if 'Gas' in Param or 'SFR' in Param:
                          Cond = (rGas <= 10**(rh))
                          if 'sSFR' in Param:
                              CondStar = (rStar <=  10**(rh))
                      elif 'Star' in Param:
                          Cond = (rStar <=  10**(rh))
                      elif 'DM' in Param:
                          Cond = (rDM <=  10**(rh))
                        
                    elif '_Above_' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas <= 10**(rh + int(Param[-6:-3]) / 100))
                            if 'sSFR' in Param:
                                CondStar = (rStar <=  10**(rh + int(Param[-6:-3]) / 100))
                        elif 'Star' in Param:
                            Cond = (rStar <= 10**(rh + int(Param[-6:-3]) / 100))
                        elif 'DM' in Param:
                            Cond = (rDM <=  10**(rh + int(Param[-6:-3]) / 100))
                            
                    else:
                        if 'minus' in Param:
                            if 'Gas' in Param or 'SFR' in Param:
                                Cond = (rGas >10**(rh - 0.5)) & (rGas <= 10**(rh))
                                if 'sSFR' in Param:
                                    CondStar = (rStar >10**(rh - 0.5)) & (rStar <= 10**(rh))
                            elif 'Star' in Param:
                                Cond = (rStar >10**(rh - 0.5)) & (rStar <= 10**(rh))
                            elif 'DM' in Param: 
                                Cond = (rDM >10**(rh - 0.5)) & (rDM <= 10**(rh))
                            sizeCount = sizeCount + 1  
                            
                        elif 'plus' in Param:
                           if 'Gas' in Param or 'SFR' in Param:
                               Cond = (rGas > 10**(rh) ) & (rGas <= 10**(rh + 0.5))
                               if 'sSFR' in Param:
                                   CondStar = (rStar > 10**(rh) ) & (rStar <= 10**(rh + 0.5))
                                  
                           elif 'Star' in Param:
                               Cond = (rStar > 10**(rh)) & (rStar <= 10**(rh + 0.5))
                           elif 'DM' in Param: 
                               Cond = (rDM > 10**(rh) ) & (rDM <= 10**(rh + 0.5))
                           sizeCount = sizeCount + 1

                elif SizeLim == 'rpkpc':
                    
                    if '_In_' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas <= slim[0])
                            if 'sSFR' in Param:
                                CondStar = (rStar <= slim[0])
                            if 'jAngle_Gas_Star' in Param:
                                CondStar = (rStar <= slim[0])
                        elif 'Star' in Param and not 'jAngle_Gas_Star' in Param:
                            Cond = (rStar <= slim[0])
                        elif 'DM' in Param:
                            Cond = (rDM <= slim[0])
                        
                        else:
                            Cond = (R<= slim[0])
                            
                        if 'Corotate' in Param or 'Counterrotate' in Param:
                            CondJ = (R <=  slim[0])
                            
                        sizeCount = 1
                        
                    elif '_Above_' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas > slim[-1])
                            if 'sSFR' in Param:
                                CondStar = (rStar > slim[0])
                            if 'jAngle_Gas_Star' in Param:
                                CondStar = (rStar > slim[0])
                        elif 'Star' in Param and not 'jAngle_Gas_Star' in Param:
                            Cond = (rStar > slim[-1])
                        elif 'DM' in Param:
                            Cond = (rDM > slim[-1])
                            
                        else:
                            Cond = (R > slim[-1])
                        sizeCount = 0
                    else:
                        if 'Gas' in Param or 'SFR' in Param:
                            Cond = (rGas > slim[sizeCount - 1]) & (rGas <= slim[sizeCount])
                            if 'sSFR' in Param:
                                CondStar = (rStar > slim[sizeCount - 1]) & (rStar <= slim[sizeCount])
                            if 'jAngle_Gas_Star' in Param:
                                 CondStar = (rStar > slim[sizeCount - 1]) & (rStar <= slim[sizeCount])
                        elif 'Star' in Param and not 'jAngle_Gas_Star' in Param:
                            Cond = (rStar > slim[sizeCount - 1]) & (rStar <= slim[sizeCount])
                        elif 'DM' in Param: 
                            Cond = (rDM > slim[sizeCount - 1]) & (rDM <= slim[sizeCount])
                        else:
                            Cond = (R > slim[sizeCount - 1]) & (R <= slim[sizeCount])
                        sizeCount = sizeCount + 1
            
                    
                elif SizeLim == '200':
                    
                    #rhocrit200 = 200*2.77536627e11 * h**2. / (1e3)**3. # Msun / kpc
                    rhocrit200 = 200*MATH.critical_density(dfTime.loc[dfTime.Snap == snap, 'z'].values[0], H0=h*100, Omega_m=Omegam0, Omega_Lambda=1-Omegam0)

                    argsSort = np.argsort(R)
                    
                    Mass = Mass[argsSort]
                    R = R[argsSort]
                    V = V[argsSort]
                    J = J[argsSort]
                    JVec = JVec[argsSort]
                    MassCum = np.nancumsum(Mass)

                    
                    
                    rho = np.array([MassCum[i] / ((4./3.) * np.pi * R[i]**3.) for i in range(len(MassCum))])
                        
                    rholog10 = np.log10(rho)
                    Rlog10 = np.log10(R)
                    Mlog10 = np.log10(MassCum)
                   
                    
                    f_linear = interp1d(Rlog10, rholog10 - np.log10(rhocrit200), fill_value='extrapolate')
                    Rinterpolate = np.linspace(min(Rlog10), max(Rlog10), 100)
                    root = fsolve(f_linear , Rinterpolate)
                    root = root[ root > 0]
                    root = np.unique(root)
                    R200ForJ = 10**np.nanmedian(root)


                    f_linear = interp1d(Rlog10, Mlog10, fill_value='extrapolate')
                    M200ForJ = 10**f_linear(np.log10(R200ForJ))
                    
                    
                    if SubfindID[str(ID)].values[int(99 - snap)] == GroupFirstSub[str(ID)].values[int(99 - snap)]:
                        
                        valid = (~np.isnan(rho)) & (rho > 1e2)
                        R = R[valid]
                        rho = rho[valid]
                        MassCum = MassCum[valid]
                        
                        try:
                            R200ForJ = R[rho >= rhocrit200][-1]
                        except Exception:
                            Rlog10 = np.log10(R)
                            rholog10 = np.log10(rho)

                            f_linear_R = interp1d(rholog10,Rlog10, fill_value='extrapolate')
                            R200ForJ = 10**f_linear_R(np.log10(rhocrit200))
                            
                        try:
                            M200ForJ = MassCum[rho >= rhocrit200][-1]
                        except Exception:
                            Rlog10 = np.log10(R)
                            MassCumlog10 = np.log10(MassCum)

                            f_linear_M = interp1d(Rlog10,MassCumlog10, fill_value='extrapolate')
                            M200ForJ = 10**f_linear_M(np.log10(R200ForJ))
                        
                        
                        R200 = 10**Group_R_Crit200[str(ID)].values[int(99 - snap)]
                        M200 = 10**Group_M_Crit200[str(ID)].values[int(99 - snap)]

                    else:
                        R200 = R200ForJ
                        M200 = M200ForJ
                        

                else:
                    if 'Gas' in Param or 'SFR' in Param:
                        Cond = (rGas > 0)
                        if 'sSFR' in Param:
                            CondStar = (rStar > 0)
                        if 'jAngle_Gas_Star' in Param:
                            CondStar = (rStar > 0)
                        if 'Corotate' in Param or 'Counterrotate' in Param:
                            CondJ = (R <=  rHStarArray[99 - snap])
                    elif 'Star' in Param and not 'jAngle_Gas_Star' in Param:
                        Cond = (rStar > 0)
                        if 'Corotate' in Param or 'Counterrotate' in Param:
                            CondJ = (R <=  rHStarArray[99 - snap])
                    elif 'DM' in Param:
                        Cond = (rDM > 0)
                        if 'Corotate' in Param or 'Counterrotate' in Param:
                            CondJ = (R <=  rHStarArray[99 - snap])
                    else:
                        Cond = (R > 0)
                        
                if 'Accreted' in Param:
                    dfGasAccreted, dfStarAccreted, dfDMAccreted = ParticleParameters(f, ID, snap)
                    if 'Gas' in Param or 'SFR' in Param:
                        IDsCurrent = IDsGas_Current
                        Infalling = dfGasAccreted.loc[
                            ((dfGasAccreted.RadNormalized.between(1, 2.5)) & (dfGasAccreted.RadNormalized - 3 >= dfGasAccreted.VelRadOverSigma))
                            | ((dfGasAccreted.RadNormalized > 2.5) & (dfGasAccreted.VelRadOverSigma < -0.5))].copy()
                    elif 'Star' in Param:
                        IDsCurrent = IDsStar_Current
                        Infalling = dfStarAccreted.loc[
                            ((dfStarAccreted.RadNormalized.between(1, 2.5)) & (dfStarAccreted.RadNormalized - 3 >= dfStarAccreted.VelRadOverSigma))
                            | ((dfStarAccreted.RadNormalized > 2.5) & (dfStarAccreted.VelRadOverSigma < -0.5))].copy()
                    elif 'DM' in Param:
                        IDsCurrent = IDsDM_Current
                        Infalling = dfDMAccreted.loc[
                            ((dfDMAccreted.RadNormalized.between(1, 2.5)) & (dfDMAccreted.RadNormalized - 3 >= dfDMAccreted.VelRadOverSigma))
                            | ((dfDMAccreted.RadNormalized > 2.5) & (dfDMAccreted.VelRadOverSigma < -0.5))].copy()
                    
                    Cond = Cond &  [IDPartNow in Infalling.ParticleID.values for IDPartNow in IDsCurrent]
                    
                    
                    
                if 'ExSitu' in Param or 'InSitu' in Param:
                     try:
                         if verbose:
                             print('Computing Ex Situ')
                         as_Birth = np.array([tBirth for tBirth in f['PartType4']['GFM_StellarFormationTime'][:]])
                         IDsPart = np.array([idNum for idNum in f['PartType4']['ParticleIDs'][:]])
                         Tag = np.empty(len(IDsPart))
                         
                         SFSnap = as_Birth.copy()
                         for aindex, a_Birth in enumerate(as_Birth):
                             epsilon = 0.1
                             while len(dfTime.loc[abs(dfTime.a - a_Birth) < epsilon, 'Snap'].values) > 2:
                                 snapsBirth = dfTime.loc[abs(dfTime.a - a_Birth) < epsilon, 'Snap'].values
                                 epsilon = epsilon / 2.
                                 if len(dfTime.loc[abs(dfTime.a - a_Birth) < epsilon, 'Snap'].values) == 0:
                                     break
                             SFSnap[aindex] = int(snapsBirth[0])
                         
                         #BirthPos = np.array([t for t in f['PartType4']['BirthPos']])
                         SFSnapUnique = np.unique(SFSnap)
                         #Rads = as_Birth.copy()
                         #RadsReff = as_Birth.copy() 
                         
                         for snapUnique in SFSnapUnique:
                             
                             
                             snapUnique = int(snapUnique)
                             f_at_SnapBreak = extractParticles(ID, snaps=[int(snapUnique)])[0]
                             CondSnap = (SFSnap == snapUnique)


                             try:
                                 IDsParticle_at_SnapBreak = f_at_SnapBreak['PartType4']['ParticleIDs'][:]
                             except Exception:
                                 Tag[CondSnap] = 100 #100 -> ExSitu
                                 continue

                             
                             for idNum in IDsPart[CondSnap]:

                                 if (idNum in IDsParticle_at_SnapBreak):
                                     Tag[IDsPart == idNum] = -100

                                 if (not idNum in IDsParticle_at_SnapBreak):
                                     Tag[IDsPart == idNum] = 100

                       
                             
                         if 'ExSitu' in Param:
                            CondII = (Tag > 0)

                            CondIII = Cond & CondII
                            Cond = CondIII
                            
                         elif 'InSitu' in Param:
                            CondII = (Tag < 0)

                            CondIII = Cond & CondII
                            Cond = CondIII
                            
                     except Exception:
                        continue
                        
                if 'AgeStarAfterEntry' in Param:
                    SnapFirstEntry = dfSample.loc[dfSample.SubfindID_99 == ID, 'Snap_At_FirstEntry'].values[0]
                    
                    if np.isnan(SnapFirstEntry) or SnapFirstEntry < 0:
                        SnapFirstEntry = 99
                    AgeCond = ETNG.AgeUniverse(Omegam0,h,dfTime.loc[dfTime.Snap == SnapFirstEntry, 'z'].values[0])
                    CondAge = AgeStar > AgeCond
                    Cond = Cond & CondAge
                    
                if 'Gas' in Param or 'SFR' in Param:
                    if len(massGas[Cond]) < 10:
                        DFs[l][str(ID)][99 - snap] = - np.inf
                        continue
                    
                if 'Star' in Param :
                    
                    if 'jAngle_Gas_Star' in Param:
                        if len(massStar[CondStar]) < 10: 
                            DFs[l][str(ID)][99 - snap] = - np.inf
                            continue
                    elif len(massStar[Cond]) < 10 and not 'jAngle_Gas_Star' in Param:
                        DFs[l][str(ID)][99 - snap] = - np.inf
                        continue
                    
                 
                 
                if 'DM' in Param :
                    if len(massDM[Cond]) < 10:
                        DFs[l][str(ID)][99 - snap] = - np.inf
                        continue
  

                if 'Inflow' in Param:
                    if 'Gas' in Param or 'SFR' in Param:
                        
                        velGasRadMedian = MATH.weighted_median(velRadGas[Cond], massGas[Cond])
                        sigmavelGasRadMedian = MATH.boostrap_func(velRadGas[Cond], func=np.nanmedian)
                        Cond = Cond & (velRadGas <= np.min([velGasRadMedian - sigmavelGasRadMedian, -5]))

                    elif 'Star' in Param:
                        velStarRadMedian = MATH.weighted_median(velRadStar[Cond], massStar[Cond])
                        sigmavelStarRadMedian = MATH.boostrap_func(velRadStar[Cond], func=np.nanmedian)
                        Cond = Cond & (velRadStar <= np.min([velStarRadMedian - sigmavelStarRadMedian, -5]))
                    elif 'DM' in Param:
                        velDMRadMedian = MATH.weighted_median(velRadDM[Cond], massDM[Cond])
                        sigmavelDMRadMedian = MATH.boostrap_func(velRadDM[Cond], func=np.nanmedian)
                        Cond = Cond & (velRadDM <= np.min([velDMRadMedian - sigmavelDMRadMedian, -5]))
                elif 'Outflow' in Param:
                    if 'Gas' in Param or 'SFR' in Param:
                        velGasRadMedian = MATH.weighted_median(velRadGas[Cond], massGas[Cond])
                        sigmavelGasRadMedian = MATH.boostrap_func(velRadGas[Cond], func=np.nanmedian)
                        Cond = Cond & (velRadGas >= np.max([velGasRadMedian + sigmavelGasRadMedian, 5]))
                    elif 'Star' in Param:
                        velStarRadMedian = MATH.weighted_median(velRadStar[Cond], massStar[Cond])
                        sigmavelStarRadMedian = MATH.boostrap_func(velRadStar[Cond], func=np.nanmedian)
                        Cond = Cond & (velRadStar >= np.max([velStarRadMedian + sigmavelStarRadMedian, 5]))
                    elif 'DM' in Param:
                        velDMRadMedian = MATH.weighted_median(velRadDM[Cond], massDM[Cond])
                        sigmavelDMRadMedian = MATH.boostrap_func(velRadDM[Cond], func=np.nanmedian)
                        Cond = Cond & (velRadDM >= np.max([velDMRadMedian + sigmavelDMRadMedian, 5]))
                        
                if 'Corotate' in Param:
                     JMean = np.nansum(JVec[CondJ], axis = 0) / np.nansum(Mass[CondJ])
                     
                     if 'Gas' in Param:
                         j = np.cross(posGas, velGas)
                     elif 'Star' in Param:
                         j = np.cross(posStar, velStar)
                     elif 'DM' in Param:
                        j = np.cross(posDM, velDM)
                     angelsTheta = np.array([])
                     for jIndex in range(len(j)):
                         param = np.dot(JMean /  abs(np.linalg.norm(JMean)) , j[jIndex] / abs(np.linalg.norm(j[jIndex])) )
                   
                         angelsTheta = np.append(angelsTheta, np.degrees(np.arccos(param)))
                     Cond = Cond & (angelsTheta < 45)
                     
                elif 'Counterrotate' in Param:
                     JMean = np.nansum(JVec[CondJ], axis = 0) / np.nansum(Mass[CondJ])
                     
                     if 'Gas' in Param:
                         j = np.cross(posGas, velGas)
                     elif 'Star' in Param:
                         j = np.cross(posStar, velStar)
                     elif 'DM' in Param:
                        j = np.cross(posDM, velDM)
                     angelsTheta = np.array([])
                     for jIndex in range(len(j)):
                         param = np.dot(JMean /  abs(np.linalg.norm(JMean)) , j[jIndex] / abs(np.linalg.norm(j[jIndex])) )
                   
                         angelsTheta = np.append(angelsTheta, np.degrees(np.arccos(param)))
                     Cond = Cond & (angelsTheta > 135)


                if 'Gas' in Param or 'SFR' in Param:
                    if len(massGas[Cond]) < 10:
                        DFs[l][str(ID)][99 - snap] = - np.inf
                        continue
                    
                if 'Star' in Param:
                    

                    if 'jAngle_Gas_Star' in Param:
                        if len(massStar[CondStar]) < 10: 
                            DFs[l][str(ID)][99 - snap] = - np.inf
                            continue
                    elif len(massStar[Cond]) < 10 and not 'jAngle_Gas_Star' in Param:
                        DFs[l][str(ID)][99 - snap] = - np.inf
                        continue
                 
                if 'DM' in Param :
                    if len(massDM[Cond]) < 10:
                        DFs[l][str(ID)][99 - snap] = - np.inf
                        continue
                        
                if 'Outer' in Param:
                    

                    if 'Gas' in Param or 'SFR' in Param:
                        rGasMedian = MATH.weighted_median(rGas[Cond], massGas[Cond])
                        sigmarGasRadMedian = MATH.boostrap_func(rGas[Cond], func=np.nanmedian)
                        Cond = Cond & (rGas >= rGasMedian + 3*np.std(rGas[Cond]))

                    elif 'Star' in Param:
                        rStarMedian = MATH.weighted_median(rStar[Cond], massStar[Cond])
                        sigmarStarRadMedian = MATH.boostrap_func(rStar[Cond], func=np.nanmedian)
                              
                        Cond = Cond & (rStar >= rStarMedian + 3*np.std(rStar[Cond]))
                    elif 'DM' in Param:
                        rDMMedian = MATH.weighted_median(rDM[Cond], massDM[Cond])
                        sigmarDMRadMedian = MATH.boostrap_func(rDM[Cond], func=np.nanmedian)
                        Cond = Cond & (rDM >= rDMMedian  + 3*np.std(rDM[Cond]))
                        
                if 'Gas' in Param or 'SFR' in Param:
                    if len(massGas[Cond]) < 10:
                        DFs[l][str(ID)][99 - snap] = - np.inf
                        continue
                    
                if 'Star' in Param:
                    

                    if 'jAngle_Gas_Star' in Param:
                        if len(massStar[CondStar]) < 10: 
                            DFs[l][str(ID)][99 - snap] = - np.inf
                            continue
                    elif len(massStar[Cond]) < 10 and not 'jAngle_Gas_Star' in Param:
                        DFs[l][str(ID)][99 - snap] = - np.inf
                        continue
                 
                if 'DM' in Param :
                    if len(massDM[Cond]) < 10:
                        DFs[l][str(ID)][99 - snap] = - np.inf
                        continue
            
                if '200' in Param and not 'dex' in Param:
                    
                    
                    if 'M' in Param:
                        
                        
                        DFs[l][str(ID)][99 - snap] =  np.log10(M200ForJ)
                        
                        
                    elif 'V' in Param:
                        
                        DFs[l][str(ID)][99 - snap] =  np.nansum(Mass[MassCum < M200ForJ]*V[MassCum < M200ForJ]) /  np.nansum(Mass[MassCum < M200ForJ])

                    elif 'J' in Param:
                        
                        '''
                        if len(J[MassCum < M200ForJ]) == len(J):
                            Jlog10 = np.log10(J)
                            Mlog10 = np.log10(MassCum)
                            
                            Jlog10 = Jlog10[~np.isnan(Mlog10)]
                            Mlog10 = Mlog10[~np.isnan(Mlog10)]
                            
                            Mlog10 = Mlog10[~np.isnan(Jlog10)]
                            Jlog10 = Jlog10[~np.isnan(Jlog10)]
                            
                            Jlog10 = Jlog10[~np.isinf(Mlog10)]
                            Mlog10 = Mlog10[~np.isinf(Mlog10)]
                            
                            Mlog10 = Mlog10[~np.isinf(Jlog10)]
                            Jlog10 = Jlog10[~np.isinf(Jlog10)]

                            #f_linear_J = interp1d(Mlog10,Jlog10, fill_value='extrapolate')
                            # Fit the power law model
                            params, _ = curve_fit(MATH.power_law, Mlog10, Jlog10)
                            a, b = params
                            
                            
                            MCumInterpolate = np.linspace(min(Mlog10), np.log10(M200), 100)
                            JInterpolate =  MATH.power_law(MCumInterpolate, a, b)
                            
                            DFs[l][str(ID)][99 - snap] = np.nansum(JInterpolate[MCumInterpolate < M200]) /  M200  #np.linalg.norm(np.nansum(JVec[MassCum < M200], axis = 1))

                        else:
                        '''    
                        DFs[l][str(ID)][99 - snap] = np.log10(np.linalg.norm(np.nansum(JVec[MassCum < M200ForJ], axis = 0)))

                    elif 'R' in Param:
                        if np.log10(R200ForJ) < 0:
                            print(R200ForJ)
                        DFs[l][str(ID)][99 - snap] =  np.log10(R200ForJ)

                else:
                    if 'SFMass' in Param:
                        try:
                            SFR =  f['PartType0']['StarFormationRate'][:] 
                        except Exception:
                            SFR = np.array([0])
                            
                        Cond = Cond & (SFR > 0)
                        
                    #Compute df
                    if 'Gas' in Param or 'SFR' in Param:
                        if len(massGas[Cond]) < 10:
                            DFs[l][str(ID)][99 - snap] = - np.inf
                            continue
                        
                    if 'Star' in Param :
                        if 'jAngle_Gas_Star' in Param:
                            if len(massStar[CondStar]) < 10: 
                                DFs[l][str(ID)][99 - snap] = - np.inf
                                continue
                        elif len(massStar[Cond]) < 10 and not 'jAngle_Gas_Star' in Param:
                            DFs[l][str(ID)][99 - snap] = - np.inf
                            continue
                        
                     
                    if 'DM' in Param :
                        if len(massDM[Cond]) < 10:
                            DFs[l][str(ID)][99 - snap] = - np.inf
                            continue
                    
                    if 'Mass' in Param and not 'SFMass' in Param:
                        if 'Gas' in Param:
                            DFs[l][str(ID)][99 - snap] = np.log10(np.nansum(massGas[Cond]))
                        elif 'Star' in Param:
                            DFs[l][str(ID)][99 - snap] = np.log10(np.nansum(massStar[Cond]))
                        elif 'DM' in Param:
                            DFs[l][str(ID)][99 - snap] = np.log10(np.nansum(massDM[Cond]))
                        else:
                            DFs[l][str(ID)][99 - snap] =  np.log10(np.nansum(Mass[Cond]))
           
                        
                    elif 'rho' in Param and 'In' in Param:
                        if 'dex' in Param and 'minus' in Param:
                            V = (4.*np.pi/3.)*(10**(rh - int(Param[-6:-3]) / 100))**3.
                        elif 'dex' in Param and 'plus' in Param:
                            V = (4.*np.pi/3.)*(10**(rh + int(Param[-6:-3]) / 100))**3.
                        else:
                            V =  (4.*np.pi/3.)*(10**(rh))**3.
                        if 'Gas' in Param:
                            DFs[l][str(ID)][99 - snap] = np.log10(np.nansum(massGas[Cond]) / V)
                        elif 'Star' in Param:
                            DFs[l][str(ID)][99 - snap] = np.log10(np.nansum(massStar[Cond]) / V)
                        elif 'DM' in Param:
                            DFs[l][str(ID)][99 - snap] = np.log10(np.nansum(massDM[Cond]) / V)
                            
                    elif 'SFMass' in Param:
                        DFs[l][str(ID)][99 - snap] = np.log10(np.nansum(massGas[Cond]))
                        
                    
                    elif 'sSFR' in Param:
                        DFs[l][str(ID)][99 - snap] = np.log10(np.nansum(SFR[Cond])  / np.nansum(massStar[CondStar]))
                        
                    elif 'SFR' in Param and not 'sSFR' in Param:
                        DFs[l][str(ID)][99 - snap] = np.log10(np.nansum(SFR[Cond]))
                        
                    elif 'GFM_Metallicity' in Param:
                        if 'Star' in Param:
                            DFs[l][str(ID)][99 - snap] = np.nansum(massStar[Cond]*ZStar[Cond]) / np.nansum(massStar[Cond])
                        
                    elif 'Metallicity' in Param:
                        if 'Star' in Param:
                            DFs[l][str(ID)][99 - snap] = np.nansum(massStar[Cond]*ZStar[Cond]) / np.nansum(massStar[Cond])
                        
                        
                    elif 'j' in Param and not 'jAngle_Gas_Star' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            DFs[l][str(ID)][99 - snap] = np.nansum(massGas[Cond]*np.sqrt(np.nansum(np.cross(posGas[Cond], velGas[Cond])*np.cross(posGas[Cond], velGas[Cond]), axis = 1))) / np.nansum(massGas[Cond])
                        elif 'Star' in Param:
                            DFs[l][str(ID)][99 - snap] = np.nansum(massStar[Cond]*np.sqrt(np.nansum(np.cross(posStar[Cond], velStar[Cond])*np.cross(posStar[Cond], velStar[Cond]), axis = 1))) / np.nansum(massStar[Cond])
                        elif 'DM' in Param:
                            DFs[l][str(ID)][99 - snap] = np.nansum(massDM[Cond]*np.sqrt(np.nansum(np.cross(posDM[Cond], velDM[Cond])*np.cross(posDM[Cond], velDM[Cond]), axis = 1))) / np.nansum(massDM[Cond])
                        else:
                            DFs[l][str(ID)][99 - snap] =  np.linalg.norm(np.nansum(JVec[Cond], axis = 0) / np.nansum(Mass[Cond]))
                        
                    elif 'jAngle_Gas_Star' in Param:
                        GasAngMom = (np.nansum(massGas[Cond, np.newaxis]*np.cross(posGas[Cond], velGas[Cond]), axis = 0) / np.nansum(massGas[Cond]))
                        StarAngMom = (np.nansum(massStar[CondStar, np.newaxis]*np.cross(posStar[CondStar], velStar[CondStar]), axis = 0) / np.nansum(massStar[CondStar]))

                        DFs[l][str(ID)][99 - snap] = MATH.angle_between_vectors(GasAngMom, StarAngMom)
                       
                            
                    elif 'vrad' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            DFs[l][str(ID)][99 - snap] = MATH.weighted_median(velRadGas[Cond], massGas[Cond]) 
                        elif 'Star' in Param:
                            DFs[l][str(ID)][99 - snap] = MATH.weighted_median(velRadStar[Cond], massStar[Cond]) 
                        elif 'DM' in Param:
                            DFs[l][str(ID)][99 - snap] = MATH.weighted_median(velRadDM[Cond], massDM[Cond]) 
                
                
                    elif 'sigma' in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            sigmaVelGas = np.array([np.std(velGas[Cond, 0]), np.std(velGas[Cond, 1]), np.std(velGas[Cond, 2])])
                            DFs[l][str(ID)][99 - snap] = np.linalg.norm(sigmaVelGas) / np.sqrt(3)
                        elif 'Star' in Param:
                            sigmaVelStar = np.array([np.std(velStar[Cond, 0]), np.std(velStar[Cond, 1]), np.std(velStar[Cond, 2])])
                            DFs[l][str(ID)][99 - snap] =  np.linalg.norm(sigmaVelStar) / np.sqrt(3)
                        elif 'DM' in Param:
                            DFs[l][str(ID)][99 - snap] = np.nansum(massDM[Cond]*np.sqrt(np.nansum(np.cross(posDM[Cond], velDM[Cond])*np.cross(posDM[Cond], velDM[Cond]), axis = 1))) / np.nansum(massDM[Cond])
                
                    elif 'StarAge' in Param:
                        DFs[l][str(ID)][99 - snap] = MATH.weighted_median(AgeStar[Cond], massStar[Cond]) 

                        
                    elif 'GasT' in Param:
                        
                        try:
                            u = f['PartType0']['InternalEnergy'][:]
                            xe = f['PartType0']['ElectronAbundance'][:]
                        except Exception:
                            DFs[l][str(ID)][99 - snap] = np.nan
                            
                        Xh = 0.76
                        gamma = 5./3.
                        
                        mp = 1.673e-24 #g
                        
                        kb = 1.380658e-16 # erg / K
                        
                        mu = 4./ (1. + 3. *Xh + 4. *Xh * xe) * mp
                        T = (gamma - 1.) * (u / kb) * mu 
                        T = T * 1e10
                        
                        DFs[l][str(ID)][99 - snap] = MATH.weighted_median(T[Cond], massGas[Cond])
                        
                    
                DFs[l].to_csv(PATH+'/'+SIM+'/'+SAVEFILE+'/'+Param+'.csv')
                

    return
    

def makedataevolution(names, columns, row, PhasingPlot = False,
                     IDs=None, func=np.nanmedian, nboots=100, dfName='Sample', Name = 'Name',
                     SampleName = 'Name', SubfindIDkey = 'SubfindID_99',):
    '''
    make evolution data frame to plot
    Parameters
    ----------
    names : sample names. array with str
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    Type : plot type. Can be 'z0', 'Snap' or 'Sample'.
    func : can be 'np.mean', 'np.median' or analogues.
    The rest is the same as the previous functions
    Returns
    -------
    Requested data frame
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''
    data = []
    dataerr = []
    dataPhase = []
    dataTime = []
    dataPhaseTime = []
    
    MajorParameters = ['Ntot', 'NIntermediate', 'NMajor', 'NMinor',
                       'Dry', 'Wet', 'NDryMajorIntermediateMergers',
                       'NDryMinorMergers', 'NWetMajorIntermediateMergers',
                       'NwetMinorMergers','DryDry' , 'WetWet', 'DryWet',
                       'WetDry', 'ExSubhaloMassType0', 'ExSubhaloMassType4', 'ExSubhaloMassType1',
                       'ExSubhaloMassType5']
    

    for i, param in enumerate(row):

        dataSplit = []
        dataSpliterr = []
        dataSplitPhase = []
        dataSplitTime = []
        dataSplitPhaseTime = []
        for j, split in enumerate(columns):

            dataName = []
            dataNameerr = []
            dataNamePhase = []
            dataNameTime = []
            dataNamePhaseTime = []
            for k, name in enumerate(names):
                
                if name + split == 'TDGCentral': #Particular case
                    values = makeDF(
                        'SBCYoung', param, dfName=dfName,  Name = Name, SubfindIDkey = SubfindIDkey, IDs=[923816])
    
                elif name + split == 'TDGSatellite': #Particular case
                    values = makeDF(
                        'SBCYoung', param, dfName=dfName,  Name = Name,  SubfindIDkey = SubfindIDkey, IDs=[327])
                else:
                    population = name + split
                    values = makeDF(population, param,  Name = Name, dfName=dfName,  SubfindIDkey = SubfindIDkey, IDs=IDs)

                #print(population, param,dfName, SubfindIDkey)
                
                #print(values)
                if PhasingPlot:
                    yphase, y, yerr = makeMedianPhases(population, param, dfName = dfName,  Name = Name)
                    Timephase, TimeParam, Timeerr = makeMedianPhases(population, 'TimeID', dfName = dfName,  Name = Name)
                    dataNameTime.append(TimeParam)
                    dataNamePhaseTime.append(Timephase)
                    dataNamePhase.append(yphase)

                else:
                    y = []
                    yerr = []
                    for snap in np.arange(100):
                        if len(values) == 0:
                            ySnap = np.array([np.nan])
                        else:
                            ySnap = values.iloc[snap].values
                            ySnap = np.array([value for value in ySnap])
                           
        
                        if 'TDG' in name:
                            if len(ySnap) == 0:
                                y.append(np.nan)
                                yerr.append(np.nan)
                            else:
                                y.append(ySnap[0])
                                if np.isnan(ySnap[0]):
                                    yerr.append(np.nan)
                                else:
                                    yerr.append(0)
                        else:
                            
                            if len(ySnap[~np.isnan(ySnap)]) <= 5 and (not param in ['StarMassExSitu_Above_24rpkpc', 'logjProfile', 'SubhaloSpinInRadStar', 'SubhaloSpinStar', 'GFM_Metallicity_z0', 'deltaStarInAbove'] and not 'AngRatio' in param and not 'FinalAngPlusOrbit' in param):
                                y.append(np.nan)
                                yerr.append(np.nan)
                            elif len(ySnap[np.isnan(ySnap)]) > len(ySnap) / 2 and ((not param in ['MassTensorEigenVals', 'logjProfile', 'SubhaloSpinInRadStar', 'SubhaloSpinStar', 'GFM_Metallicity_z0', 'deltaStarInAbove'] and not param in MajorParameters) and not 'AngRatio' in param and not 'FinalAngPlusOrbit' in param) :
                                y.append(np.nan)
                                yerr.append(np.nan)
                            elif len(ySnap[~np.isnan(ySnap)]) <= 2 and ('AngRatio' in param or 'FinalAngPlusOrbit' in param):
                                y.append(np.nan)
                                yerr.append(np.nan)
                            else:
                                if 'Num' in param or 'Nsub' in param or param in ['Nminor', 'Nmajor', 'Nexcept', 'NDryMajorMergers', 'NDryMinorMergers', 'NWetMajorMergers', 'NwetMinorMergers' ] : #Because of integers 
                                    if 'Normal' in population:
                                        ySnap[ySnap > 300] = np.nan
                                    yerrSnap = MATH.boostrap_func(
                                        ySnap, func=np.nanmean, num_boots=nboots)
                                    y.append(np.nanmean(ySnap))
                                    yerr.append(yerrSnap)
                                elif param in ['SubhaloBHMass']:
                                    y.append(len(ySnap[ySnap > 0]) / len(ySnap))
                                    yerr.append(1/len(ySnap))
                                elif param in ['LhardXray']: #count only AGN
                                    y.append(len(ySnap[ySnap > 38]) / len(ySnap))
                                    yerr.append(1/len(ySnap))    
                                elif 'StellarMassExSitu' in param : #count only AGN
                                    if len(ySnap[np.isinf(ySnap)]) > len(ySnap) / 2:
                                        y.append(np.nan)
                                        yerr.append(np.nan)
                                    else:
                                        ySnap[np.isinf(ySnap)] = np.nan
                                        yerrSnap = MATH.boostrap_func(
                                            ySnap, func=func, num_boots=nboots)
                                        y.append(func(ySnap))
                                        yerr.append(0.45*yerrSnap)  
                                elif 'SigmasSFRRatio' in param:
                                    yerrSnap = MATH.boostrap_func(
                                        ySnap, func=func, num_boots=nboots)
                                    y.append(func(ySnap))
                                    yerr.append(0.7*yerrSnap)
                                elif 'ExMassType' in param or 'Profile' in param:
                                    yerrSnap = MATH.boostrap_func(
                                        ySnap, func=func, num_boots=nboots)
                                    y.append(func(ySnap))
                                    yerr.append(0.7*yerrSnap)
                                elif param in ['MergerTotalRate', 'MinorMergerTotalRate', 'MajorMergerTotalRate']: #Also due integers
                                    
                                
                                    yerrSnap = MATH.boostrap_func(
                                        ySnap, func=np.nanmean, num_boots=nboots)
                                    y.append(np.nanmean(ySnap))
                                    yerr.append(yerrSnap)
                                
                                    
                                else:
                                    yerrSnap = MATH.boostrap_func(
                                        ySnap, func=func, num_boots=nboots)
                                    y.append(func(ySnap))
                                    yerr.append(yerrSnap)

         
                dataName.append(y)
                dataNameerr.append(yerr)
                

            dataSplit.append(dataName)
            dataSpliterr.append(dataNameerr)
            if PhasingPlot:
                dataSplitPhase.append(dataNamePhase)
                dataSplitTime.append(dataNameTime)
                dataSplitPhaseTime.append(dataNamePhaseTime)
            
        data.append(dataSplit)
        dataerr.append(dataSpliterr)
        if PhasingPlot:
                dataPhase.append(dataSplitPhase)
                dataTime.append(dataSplitTime)
                dataPhaseTime.append(dataSplitPhaseTime)


    if PhasingPlot:
        return data, dataerr, dataPhase, dataTime, dataPhaseTime

    else:
        return data, dataerr

def makedata(names, columns, row, Type, snap=[99], dfName='Sample', Name = 'Name', SampleName='Name', SIM = 'TNG50', SubfindIDkey = 'SubfindID_99'):

    '''
    make data frame to plot
    Parameters
    ----------
    names : sample names. array with str
    columns : specific set in the sample / or different param to plot in each column. array with str
    rows : specific set in the sample / or different param to plot in each row. array with str
    Type : plot type. Can be 'z0', 'Snap' or 'Sample'.
    The rest is the same as the previous functions
    Returns
    -------
    Reuested data frame
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    data = []
    

    if Type == 'Snap':
        if len(columns) == 1:
            nameSec = columns[0]
        else:
            nameSec = ''
        columns = snap
        
    for i, param in enumerate(row):
        dataSplit = []

        for j, split in enumerate(columns):
            dataName = []
            for k, name in enumerate(names):
                if Type == 'z0':

                    values = extractPopulation(name + split, dfName=dfName, Name = Name, SIM = SIM)

                    if 'Snap' in param:
                        valuesSnap = np.array([v for v in values[param].values])
                        valuesSnap[np.isinf(valuesSnap)] = 99
                        valuesSnap[valuesSnap == -1] = 99

                        valuesSnap[~np.isnan(valuesSnap)] = np.array([dfTime.Age.loc[dfTime.Snap == int(s)].values[0] for s in valuesSnap[~np.isnan(valuesSnap)]])
                        dataName.append(valuesSnap)

                    else:
                        dataName.append(values[param].values)

                elif Type == 'Snap':
                    values = makeDF(name + nameSec, param, dfName=dfName,  Name = Name, SubfindIDkey = SubfindIDkey, SIM = SIM)

                    dataName.append(values.iloc[99 - split].values)
                elif Type == 'Sample':
                    try:

                        values = makeDF(
                            name + split, param, dfName=dfName,  Name = Name, SubfindIDkey = SubfindIDkey, SIM = SIM)
             
                        dataName.append(values.iloc[int(99 - snap[0])].values)

                    except Exception:
                        values = extractPopulation(
                            name + split, dfName=dfName, Name = Name,  SIM = SIM)
                        dataName.append(values[param].values)

                    
            dataSplit.append(dataName)

        data.append(dataSplit)

    return data

def makeDF(population, param, dfName = 'Sample', Name = 'Name', IDs=None,  TreeHybridSubhalo = False, SubfindIDkey = 'SubfindID_99', PATH = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory',  SIM = 'TNG50', verbose=False):
    '''
    Make the parameter evolution dataframe for a specific sample
    Parameters
    ----------
    sample : sample name. str
    case : parameter name. str
    IDs : for random cases the subhalo IDs. Default: None, can be a int or array with int
    PATH : path for the file. Default: './../DFs/'. str
    dfName : for the complete dataframe sample to select the subhalos. Default: None. str
    SampleName : the sample name if using a specific dataframe sample. Default: Sample. str
 
    Returns
    -------
    Requested dataframe
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    if verbose:
        print('\n Taking df for the EVOLUTION of '+str(param)+' ...')

    #verify the parameter df
    try: 
        dfEvolution = extractDF(param, PATH=PATH)
        
    except Exception:
        if verbose:
            print('You don\'t have the DF for this case')


 
    if verbose:
        print('Restricting  df for the '+param+' SAMPLE ...')

    #take subhalo for the sample
    dSample = extractPopulation(
                population, PATH=PATH, dfName=dfName, Name=Name)

    
    #restricting dfEvolution only for subhalos in dfSample
    
    if verbose:
        print('Take the '+param+' EVOLUTION for the '+population+' SAMPLE  ...')
        
    if not TreeHybridSubhalo:

        if type(IDs) == int or IDs is None:
            keys = dSample[SubfindIDkey]
            try:
                df = dfEvolution[keys.astype(str)].copy()
            except Exception:
                df = pd.DataFrame()
                for key in keys:
                    try:
                        df[str(key)] = dfEvolution[str(key)].copy()
                    except Exception:
                        continue
        else:
            keys = IDs
            try:
                df = dfEvolution[keys.astype(str)].copy()
            except Exception:
                df = pd.DataFrame()
                for key in keys:
                    try:
                        df[str(key)] = dfEvolution[str(key)].copy()
                    except Exception:
                        continue
                    
    else:
        df = build_columns_by_id(IDs, param)
                
    return df


def build_columns_by_id(IDs, param, target_len=None):
    """
    Monta um DataFrame com uma coluna por ID:
      - Se param == 'SubhaloHalfmassRadType4':
          usa df['halfmassrad_stars'] (ckpc/h) -> kpc físicos -> log10
      - Se param == 'SubhalosSFRInHalfRad':
          usa df['sfrinhalfrad'] e df['massinrad_stars'] para log10(sSFR)

    Parâmetros:
        IDs: iterável de IDs
        param: 'SubhaloHalfmassRadType4' ou 'SubhalosSFRInHalfRad'
        target_len: opcional, força truncar/alinhar ao tamanho desejado

    Retorna:
        pd.DataFrame com colunas nomeadas por str(ID)
    """
    out_cols = []

    dfTime = extractDF('SNAPS_TIME')


    for ID in IDs:
        df = ReadTreeHybrid(subhaloz0=ID)

        z = dfTime['z'].values
        
        # Ajustar comprimento, se solicitado
        if param == 'SubhaloHalfmassRadType4':
            # halfmassrad_stars em ckpc/h -> kpc físicos
            if 'halfmassrad_stars' not in df.columns:
                raise KeyError("Coluna 'halfmassrad_stars' não encontrada no df.")
            val = df['halfmassrad_stars'].to_numpy(dtype=float)
            while len(val) < 100:
                val = np.append(val, np.nan)
            # Conversão: ckpc/h -> kpc (físicos)
            # kpc = (ckpc/h) / (1+z) * ???  (ckpc/h dividido por (1+z) e por h)
            # -> kpc = val / (1.0 + z) / h
            kpc = val / (1.0 + z) / h

            # Proteger contra valores <= 0 antes do log10
            kpc = np.where(kpc > 0, kpc, np.nan)
            logr = np.log10(kpc)
            while len(logr) < 100:
                logr = np.append(logr, np.nan)
            series = pd.Series(logr, name=str(ID))
        
        elif param == 'r_over_R_Crit200':
            # precisa ter Group_R_Crit200, GroupPos, pos_x,pos_y,pos_z
            if 'Group_R_Crit200' not in df.columns:
                raise KeyError("Coluna 'Group_R_Crit200' não encontrada no df.")
            if 'GroupPos' in df.columns:
                grp = np.vstack(df['GroupPos'].to_numpy())
            else:
                grp = np.column_stack([df['GroupPos_x'].to_numpy(),
                                       df['GroupPos_y'].to_numpy(),
                                       df['GroupPos_z'].to_numpy()])
        
            sub = np.column_stack([df['pos_x'].to_numpy(),
                                   df['pos_y'].to_numpy(),
                                   df['pos_z'].to_numpy()])
        
            # vetor relativo
            rel = sub - grp
            r = np.linalg.norm(rel, axis=1)  # ckpc/h
            R200 = df['Group_R_Crit200'].to_numpy(dtype=float)  # ckpc/h
        
            # razão r/R200
            val = r / R200
        
            # garantir comprimento fixo 100
            while len(val) < 100:
                val = np.append(val, np.nan)
        
            # log10 se você quiser (ou deixe linear)
            logr = np.log10(np.where(val > 0, val, np.nan))
            while len(logr) < 100:
                logr = np.append(logr, np.nan)
        
            series = pd.Series(logr, name=str(ID))

        elif param == 'SubhalosSFRInHalfRad':
            # sSFR = SFR / M*
            # log10(sSFR) = log10(SFR) - log10(M*)
            # massinrad_stars em 1e10 Msun/h -> Msun
            if 'sfrinhalfrad' not in df.columns:
                raise KeyError("Coluna 'sfrinhalfrad' não encontrada no df.")
            if 'massinrad_stars' not in df.columns:
                raise KeyError("Coluna 'massinrad_stars' não encontrada no df.")

            sfr = df['sfrinhalfrad'].to_numpy(dtype=float)          # Msun/yr
            mstar = df['massinrad_stars'].to_numpy(dtype=float)     # em 1e10 Msun/h

            # Converter massa para Msun
            mstar_Msun = mstar * 1e10 / h

            # Proteger contra não-positivos
            sfr = np.where(sfr > 0, sfr, np.nan)
            mstar_Msun = np.where(mstar_Msun > 0, mstar_Msun, np.nan)

            logsSFR = np.log10(sfr) - np.log10(mstar_Msun)  # log10(sSFR/yr^-1)
            
            while len(logsSFR) < 100:
                logsSFR= np.append(logsSFR, np.nan)
            series = pd.Series(logsSFR, name=str(ID))

            # OBS: se você realmente quiser usar (sfr / log10(M*)) como no rascunho,
            # substitua a linha acima por:
            # Mass_log10 = np.log10(mstar * 1e10 / h)
            # ratio = sfr / Mass_log10
            # series = pd.Series(np.log10(np.where(ratio > 0, ratio, np.nan)),
            #                    index=idx, name=str(ID))
        else:
            raise ValueError(f"param '{param}' não suportado.")

        out_cols.append(series)

    # Concatenar colunas por índice (outer para garantir alinhamento)
    result = pd.concat(out_cols, axis=1)

    # Se o índice for numérico (snap), ordenar ajuda
    try:
        result = result.sort_index()
    except Exception:
        pass

    return result

def ReadTreeHybrid(subhaloz0=None, json_file=None, base_path="/home/abhner/SIMS/TNG/TNG50-1/TreeHybrid"):
    """
    Read TreeHybrid from list of dicts on disk
    
    Arguments:
        subhaloz0: z=0 subhalo ID (default None)
        json_file: json-like file of list of dicts from TreeHybrid (default None)
        base_path: path to folder containing TreeHybrid json files
                   (default "/home/abhner/SIMS/TNG/TNG50-1/TreeHybrid")
    Returns:
        df of TreeHybrid
    """
    
    # JSON file
    if (json_file is None) and (subhaloz0 is not None):
        json_file = f"TreeHybrid_{subhaloz0}.json"
    
    # caminho completo
    json_path = os.path.join(base_path, json_file)
    
    # open and read JSON file
    dicts = []
    with open(json_path, 'r') as f:
        for line in f:
            d = ast.literal_eval(line.strip())
            dicts.append(d)

    # convert list of dicts to df
    return pd.DataFrame(dicts)

def extractPopulation(sample, PATH=os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory', SubfindID_99 = False, SIM = 'TNG50', dfName = 'Sample', Name = 'Name'):
    '''
    Extract a subhalo sample
    Parameters
    ----------
    sample : sample name. str
    case : parameter name. str
    PATH : path for the file. Default: './../DFs/'. str
    dfName : for the complete dataframe sample to select the subhalos. Default: None. str
    SampleName : the sample name if using a specific dataframe sample. Default: Sample. str
 
    Returns
    -------
    Requested dataframe
    -------
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    #Using the dataframe with all the subhalos
    
    df = extractDF(dfName, PATH=PATH , SIM = SIM)

    # Restricting the sample if sample is not 'All'
    if sample != 'All' and sample != '':
        for name in ['SubDiffuse', 'Diffuse',  'Normal', 'SBC', 'MBC', 'SubCompact', 'Compact']:
            if name in sample :
                df = df.loc[df[Name] == name]
                break
        for name in ['Satellite', 'Central']:
            if name in sample:
                #print('CentralSatellite', name)
                df = df.loc[df.CentralSatellite == name]
                
        for name in ['MassBin01', 'MassBin02', 'MassBin03', 'MassBin04']:
            if name in sample:
                #print('MassBin', name)

                df = df.loc[df.MassBin == name]
         
        for name in ['EntryToNoGas', 'NoGasToFinal']:
            if name in sample:
                #print('MassBin', name)

                df = df.loc[df.FasterCompaction == name]
        for name in ['DMrich', 'DMpoor']:
            if name in sample:
                #print('DMFracStatus', name)

                df = df.loc[df.DMFracStatus == name]
                
        for name in ['NoneMajorMerger', 'MoreOneMajorMerger']:
            if name in sample:

                df = df.loc[df.NoneMajorMerger == name]
        for name in ['InterplayRegion', 'TSRegion', 'SFRegion']:
            if name in sample:
                print('DMpoorMechanism', name)

                df = df.loc[df.DMpoorMechanism == name]

        for name in ['WithoutBH', 'WithBH']:
            if name in sample:
                print('BHHostStatus', name)

                df = df.loc[df.BHHostStatus == name]
                
        for name in ['MassiveHost', 'LowerMassHost']:
            if name in sample:
                print('Host', name)

                df = df.loc[df.Host == name]
                
        for name in ['None', 'EqualOne', 'MoreThanOne']:
            if name in sample and not 'Merger' in sample:
                print('Host', name)

                df = df.loc[df.NumMajorMergerStatus == name]
                
        for name in ['DMFracLower', 'DMFracHigher']:
            if name in sample:
                print('DMFracStatusPaperIII', name)

                df = df.loc[df.DMFracStatusPaperIII == name]

        if dfName in ['PaperI', 'PaperII', 'Sample', 'PaperIII']:
            for name in ['LowerThan13half', 'Higher13half']:
    
                if name in sample:
                    print('M200_Status', name)
                    df = df.loc[df.M200_Status == name]
    
            for name in ['LowerThan13', 'Higher13']:
    
                if name in sample and not 'half' in sample:
                    print('M200_Mean_Status', name)

                    df = df.loc[df.M200_Mean_Status == name]
        else:
            for name in ['LowerThan12','12_to_13', 'Higher13']:
    
                if name in sample and not 'half' in sample:
                    df = df.loc[df.M200_Mean_Status == name]
                
        for name in ['PreProcessingGalaxy', 'NotProcessing']:
            if name in sample:
                print('PreProcessingGalaxy', name)

                df = df.loc[(df.PreProcessingGalaxy == name)]

        for name in ['MetalUltraRich', 'MetalRich']:
            if name in sample:
                print('Zstar_Status', name)

                df = df.loc[(df.Zstar_Status == name)]
         
        for name in ['EMNew', 'ESNew']:
            if name in sample:
                print('SatelliteEnvironmentNew', name)

                df = df.loc[(df.SatelliteEnvironmentNew == name)]
         
        for name in ['NotInteract', 'Interact', 'Exclude']:
            if name in sample:
                
                if name == 'Interact' and not 'Not' in sample:
                    df = df.loc[df.SatelliteEnvironment == name]
                elif name == 'NotInteract':
                    df = df.loc[df.SatelliteEnvironment == name]
                elif not 'Interact' in sample:
                    df = df.loc[df.SatelliteEnvironment == name]
                    
        for name in ['LowerMajor', 'HigherMajor']:
            if name in sample:
                print('MajorMergerStatus', name)

                df = df.loc[df.MajorMergerStatus == name]
                
        for name in ['SpinHigher', 'SpinLower']:
            if name in sample:
                print('l200Value', name)

                df = df.loc[df.l200Value == name]
        for name in ['l200Higher', 'l200Lower']:
            if name in sample:
                print('LAMBDAValue', name)

                df = df.loc[df.LAMBDAValue == name]
           
        for name in ['HigherExSitu', 'LowerExSitu']:
            if name in sample:
                print('ExStatusFrac', name)

                df = df.loc[df.ExStatusFrac == name]
                
        if 'LoseTheirGas' in sample:
            if 'Dont' in sample:
                df = df.loc[df.GasStatus == 'DontLoseTheirGas']

            else:
                df = df.loc[df.GasStatus == 'LoseTheirGas']
        
    if  'l200DATA' in sample:
        df = df.loc[(df.l200DATA == 'l200DATA')]

    if 'NumPericenter' in sample:
            for numpericenters in ['4', '5_7', '8']:
                if 'NumPericenter'+str(numpericenters) in sample:
                    df = df.loc[(df.NumPericenter == 'NumPericenter'+str(numpericenters))]
        
    if  'BadFlag' in sample:
        df = df.loc[(df.Flags == 0)]
        
    
    else:
        df = df.loc[(df.Flags == 1)]

        if  'GMMIndex' in sample:
            df = df.loc[(df.GMMIndex == 0)]
            
        
        if  not 'WithoutProgenitor' in sample  and dfName in ['PaperI', 'PaperII', 'Sample', 'PaperIII']:
            #print('Flags', name)
            df = df.loc[(df.ProgenitorStatus == 'WithProgenitor')]
        elif  dfName in ['PaperI', 'PaperII', 'Sample', 'PaperIII']:
            df = df.loc[(df.ProgenitorStatus == 'WithoutProgenitor')]

        if not 'NotSelected'in sample  and dfName in ['PaperI', 'PaperII', 'Sample', 'PaperIII']:
            #print('Flags', name)
            df = df.loc[(df.SelectedStatus == 'Selected')]
        elif  dfName in ['PaperI', 'PaperII', 'Sample', 'PaperIII']:
            df = df.loc[(df.SelectedStatus == 'NotSelected')]
    
        
        OldInSample = True
    
        for name in ['BornYoung', 'BornIntermediate']:
            if name in sample:
                df = df.loc[df.StatusBorn == name]
                OldInSample = False
        if OldInSample and dfName in ['PaperI', 'PaperII', 'Sample', 'PaperIII']:
            df = df.loc[df.StatusBorn == 'BornOld']

    if not SubfindID_99:
        return df
    
    elif SubfindID_99:
          return df.SubfindID_99.values
    

def make_profile(ID99, snap, Ycase, PartType, dfSample = None, Entry = False, Histogram = False, nbins = 50, z = 0, rmin = 0.1, rmax = 300, 
                 Condition = None, Subhalo = None, log_spacing=True, bootstrap=True, nboot=10, Median = False,
                 PATH = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory', SIMTNG = 'TNG50'):
    """
    Compute a density profile with logarithmic spacing bins and an optional minimum radius value using the specified positions and masses.
    
    Parameters
    ----------
    pos : ndarray of shape (n, 3)
        Array of 3D positions.
    mass : ndarray of shape (n,)
        Array of masses corresponding to the positions.
    nbins : int
        Number of bins to use.
    log_spacing : bool, optional
        Whether to use logarithmic spacing between bins (default is True).
    bootstrap : bool, optional
        Whether to compute bootstrap errors (default is True).
    nboot : int, optional
        Number of bootstrap samples to use if computing errors (default is 1000).
    
    Returns
    -------
    rad : ndarray of shape (nbins,)
        Array of bin radii.
    yrad : ndarray of shape (nbins,)
        Array of bin densities.
    yrad_err : ndarray of shape (nbins,), optional
        Array of bootstrap errors for the bin densities (only returned if bootstrap is True).
    """
    
    yIDs = np.array([])
    xIDs = np.array([])
    
    factor =  1. / h / (1+dfTime.z[int(99-snap)])
    scalefactorsqrt = np.sqrt(1. / (1+dfTime.z[int(99-snap)]))
    a = 1. / (1+dfTime.z[int(99-snap)])
    


    dFHalfStar = extractDF('SubhaloHalfmassRadType4', PATH = PATH)
    dFHalfGasRad = extractDF('SubhaloHalfmassRadType0', PATH = PATH)
    
    yHist = np.array([])
    rHist = np.array([])
    
    for l, ID in enumerate(ID99):
        if Entry and snap != 99:
            snap = int(dfSample.Snap_at_Entry.loc[dfSample.SubfindID_99 == ID].values[0])
            factor =  1. / h / (1+dfTime.z[int(99-snap)])
            scalefactorsqrt = np.sqrt(1. / (1+dfTime.z[int(99-snap)]))
            a = 1. / (1+dfTime.z[int(99-snap)])
            print('snap: ', snap)

        HalfRad = dFHalfStar[str(ID)].loc[dFHalfStar.Snap == snap].values[0]
        HalfGasRad = dFHalfGasRad[str(ID)].loc[dFHalfGasRad.Snap == snap].values[0]
        print('ID: ', ID)
        try:
            file = extractParticles(ID, snaps = [snap])[0]
        except Exception:
            continue
        
        try:
            pos = posStar = file['PartType4']['Coordinates'][:] * factor
            mass = massStar = file['PartType4']['Masses'][:] * 1e10 / h
            vel = velStar = file['PartType4']['Velocities'][:] * scalefactorsqrt
            
            Cen = np.array([MATH.weighted_median(posStar[:, 0], massStar), MATH.weighted_median(posStar[:, 1], massStar), MATH.weighted_median(posStar[:, 2], massStar)])
            Vmean = np.array([MATH.weighted_median(velStar[:, 0], massStar), MATH.weighted_median(velStar[:, 1], massStar), MATH.weighted_median(velStar[:, 2], massStar)])
            CenFind = True
            
        except Exception:
            posStar = velStar =  np.array([0, 0, 0])
            massStar = np.array([0])  
            CenFind = False
            
        try:
            posGas = file['PartType0']['Coordinates'][:] * factor
            massGas = file['PartType0']['Masses'][:] * 1e10 / h
            velGas = file['PartType0']['Velocities'][:] * scalefactorsqrt  
            
            if not CenFind:
                pos = posGas
                mass = massGas
                vel = velGas
                Cen = np.array([MATH.weighted_median(posGas[:, 0], massGas), MATH.weighted_median(posGas[:, 1], massGas), MATH.weighted_median(posGas[:, 2], massGas)])
                Vmean = np.array([MATH.weighted_median(velGas[:, 0], massGas), MATH.weighted_median(velGas[:, 1], massGas), MATH.weighted_median(velGas[:, 2], massGas)])
                CenFind = True
                
        except Exception:
            posGas = velGas = np.array([0, 0, 0])
            massGas = np.array([0])
        
        try:
            posDM = file['PartType1']['Coordinates'][:] * factor
            massDM = file['Header'].attrs['MassTable'][1]*np.ones(len(file['PartType1']['Coordinates'])) * 1e10 / h
            velDM = file['PartType1']['Velocities'][:] * scalefactorsqrt
            
            if not CenFind:
                pos = posDM
                mass = massDM
                vel = velDM
                Cen = np.array([MATH.weighted_median(posDM[:, 0], massDM), MATH.weighted_median(posDM[:, 1], massDM), MATH.weighted_median(posDM[:, 2], massDM)])
                Vmean = np.array([MATH.weighted_median(velDM[:, 0], massDM), MATH.weighted_median(velDM[:, 1], massDM), MATH.weighted_median(velDM[:, 2], massDM)])
                CenFind = True
           
        except Exception:
            posDM = velDM = np.array([0, 0, 0])
            massDM = np.array([0])
            
    
        if not PartType in file.keys():
            print('Doesn\'t have ', PartType,' at snap: ', snap)
            continue
        
        if PartType != 'PartType1':
            pos = file[PartType]['Coordinates'][:]  / (1+dfTime.z.values[dfTime.Snap == snap]) / h #kpc
            vel = file[PartType]['Velocities'][:]   * scalefactorsqrt
            mass = file[PartType]['Masses'][:] * 1e10 / h
        else:
            pos = file[PartType]['Coordinates'][:]  / (1+dfTime.z.values[dfTime.Snap == snap]) / h #kpc
            vel = file[PartType]['Velocities'][:]   * scalefactorsqrt
            mass = file['Header'].attrs['MassTable'][1]*np.ones(len(file[PartType]['Coordinates'])) * 1e10 / h
          

        pos = pos - Cen
        vel = vel - Vmean
        # Compute distances from the origin
        r = np.linalg.norm(pos, axis=1)
        #R = np.linalg.norm(pos[:, 0:2], axis=1)
        #Hr = cos.Eofz(Omegam0, 1.-Omegam0, z) * h * 100 * r /1e3
        velrad = np.sum(pos*vel, axis = 1)/r # + Hr
    
        #Define y-
        if Ycase in ['Density', 'Mass', 'DensityStar', 'MassesStar', 'DensityGas', 'MassesGas', 'DensityDM', 'MassesDM']:
            y = mass
            
        
        elif Ycase == 'T':
            
            u = file['PartType0']['InternalEnergy'][:]
            xe = file['PartType0']['ElectronAbundance'][:]
            
            
            mu = 4./ (1. + 3. * Xh + 4. * Xh * xe) * mp
            T = (gamma - 1.) * (u / kb) * mu 
            y = T * 1e10
            
        elif Ycase == 'RadVelocity':
            Hr = cos.Eofz(Omegam0, 1.-Omegam0, z) * h * 100 * r /1e3
            y = np.sum(pos*vel, axis = 1)/r  + Hr
        
        elif Ycase == 'AngMomentum':
            y = mass*np.sqrt(np.sum(np.cross(pos, vel)*np.cross(pos, vel), axis = 1))
          
        elif Ycase == 'j':
            y = np.sqrt(np.sum(np.cross(pos, vel)*np.cross(pos, vel), axis = 1))
            
        elif Ycase == 'PhiVelocity':
            costheta = np.clip(pos[:, 2]/r,-1,1)
            theta = np.arccos(costheta)
            sintheta = np.clip(np.sin(theta),-1,1)
            sinphi = np.clip(pos[:, 1] / (r*sintheta),-1,1)
            cosphi = np.clip(pos[:, 0] / (r*sintheta),-1,1)
            y = np.sum(vel * np.array([-sinphi, cosphi, np.zeros(pos.shape[0])]).T, axis=1)
            
        elif Ycase == 'StellarFeedback':
            y = file['PartType0']['StarFormationRate'][:]
            Zmet = file['PartType0']['GFM_Metallicity'][:]
            fZ = 1. + 3./ ( 1. + (Zmet/0.002)**2.)
            y = 3.41e41 * y * fZ
            
        elif Ycase == 'n': 
            y = file['PartType0']['Density'][:] * 6.8e-29 * 1e3 / (1e2)**3 # g/cm
            
            mH = 1.67e-24 
            mu = 0.58
            y = 1e10*y/(mu*mH)
            
        elif Ycase == 'PhiOverRad':
            costheta = np.clip(pos[:, 2]/r,-1,1)
            theta = np.arccos(costheta)
            sintheta = np.clip(np.sin(theta),-1,1)
            sinphi = np.clip(pos[:, 1] / (r*sintheta),-1,1)
            cosphi = np.clip(pos[:, 0] / (r*sintheta),-1,1)
            y =  np.abs(np.sum(vel * np.array([-sinphi, cosphi, np.zeros(pos.shape[0])]).T, axis=1))
            yR = np.abs(np.sum(pos*vel, axis = 1)/r)
         
        elif Ycase == 'zOverRad':
            yR= np.abs( r )
            y = np.abs( pos[:, 2] )
            
        elif Ycase == 'z':
            y = np.abs(pos[:, 2])
            
        elif Ycase == 'PhiOverDisp':
            costheta = np.clip(pos[:, 2]/r,-1,1)
            theta = np.arccos(costheta)
            sintheta = np.clip(np.sin(theta),-1,1)
            sinphi = np.clip(pos[:, 1] / (r*sintheta),-1,1)
            cosphi = np.clip(pos[:, 0] / (r*sintheta),-1,1)
            
            VelocitiesMean = np.mean(vel, axis = 0)
    
            sigma = np.sqrt((vel - VelocitiesMean)**2)
            sigma = np.sqrt(np.sum(sigma*sigma, axis = 1))

            #y = np.abs( np.sum(vel * np.array([-sinphi, cosphi, np.zeros(pos.shape[0])]).T, axis=1) / yden)
            y =  np.abs(np.sum(vel * np.array([-sinphi, cosphi, np.zeros(pos.shape[0])]).T, axis=1))
            yR = np.abs(sigma)
                
        elif Ycase == 'Sigma':
            costheta = np.clip(pos[:, 2]/r,-1,1)
            theta = np.arccos(costheta)
            sintheta = np.clip(np.sin(theta),-1,1)
            sinphi = np.clip(pos[:, 1] / (r*sintheta),-1,1)
            cosphi = np.clip(pos[:, 0] / (r*sintheta),-1,1)
            
            VelocitiesMean = np.mean(vel, axis = 0)
    
            sigma = np.sqrt((vel - VelocitiesMean)**2)
            sigma = np.sqrt(np.sum(sigma*sigma, axis = 1))
            
            y = sigma
          
        elif Ycase == 'tcool2':

            Msun_to_g_over_kpc_to_cm_3 =  6.8e-32
            rho = file["PartType0"]["Density"][:] *( (1e10 / h) / (factor / h)**3) * Msun_to_g_over_kpc_to_cm_3 # g / cm^3
            Hfrac = file['PartType0']['GFM_Metals'][:, 0]
            Lambda = file["PartType0"]["GFM_CoolingRate"][:] # erg cm^3 / s 
            ratefact = Hfrac**2 / mH * ( rho / mH) # 1/(g*cm^3)
            
            coolrate = Lambda * ratefact # erg cm^3/s * (1/g/cm^3) = erg/s/g (i.e. specific rate)
            uenergy = file["PartType0"]["InternalEnergy"][:] * kmc_to_cm**2.
            
            y = uenergy / (-1.0*coolrate) 
            y = y / Gyr_to_s
            
            # if lambda_net is positive set t_cool=nan (i.e. actual net heating, perhaps from the background)
            w = np.where(Lambda >= 0.0)

     
            y[w] = np.nan

            
        elif Ycase == 'tcool':
            

            u = file['PartType0']['InternalEnergy'][:]

            xe = file['PartType0']['ElectronAbundance'][:]
            
            
            mu = 4./ (1. + 3. *Xh + 4. *Xh * xe) * mp
            T = (gamma - 1.) * (u / kb) * mu  * 1e10
            
            Msun_to_g_over_kpc_to_cm_3 =  6.8e-32
            rho = file["PartType0"]["Density"][:] *( (1e10 / h) / (factor / h)**3) * Msun_to_g_over_kpc_to_cm_3 # g / cm^3

            
            Lambda = file["PartType0"]["GFM_CoolingRate"][:] # erg cm^3 / s 
            uenergy = file["PartType0"]["InternalEnergy"][:] * kmc_to_cm**2. # km^2 / s^2
            
            y = 1.37 * mH *(uenergy * mH) / ( - rho * Lambda ) 
            
            y = np.log10(1.37) + 2*np.log10(mH) + np.log10(uenergy) - (np.log10(rho) + np.log10(abs(Lambda)))
            y = 10**y / Gyr_to_s
            w = np.where(Lambda >= 0.0)
     
            y[w] = np.nan
 
            
        elif Ycase == 'PhiOverCirc':
            costheta = np.clip(pos[:, 2]/r,-1,1)
            theta = np.arccos(costheta)
            sintheta = np.clip(np.sin(theta),-1,1)
            sinphi = np.clip(pos[:, 1] / (r*sintheta),-1,1)
            cosphi = np.clip(pos[:, 0] / (r*sintheta),-1,1)
            y = np.sum(vel * np.array([-sinphi, cosphi, np.zeros(pos.shape[0])]).T, axis=1) 
            
            massStar = file['PartType4']['Masses'][:] * 1e10 / h
            posDM = file['PartType1']['Coordinates'][:] * factor
            massDM = file['Header'].attrs['MassTable'][1]*np.ones(len(file['PartType1']['Coordinates'])) * 1e10 / h
            posGas = file['PartType0']['Coordinates'][:] * factor
      
            posDM = posDM - Cen
            rdm = np.linalg.norm(posDM, axis=1)
            posGas = posGas - Cen
            rgas = np.linalg.norm(posGas, axis=1)
            posStar = posStar - Cen
            rstar = np.linalg.norm(posStar, axis=1)
            
        elif Ycase == 'sSFR':
            y = file['PartType0']['StarFormationRate'][:]
            massStar = file['PartType4']['Masses'][:] * 1e10 / h
            posStar = file['PartType4']['Coordinates'][:] * factor
            posStar = posStar - Cen
            rstar = np.linalg.norm(posStar, axis=1)
            
            
        elif Ycase == 'sSFRE':
            y = file['PartType0']['StarFormationRate'][:]
            massGasE = file['PartType0']['NeutralHydrogenAbundance'][:] * file['PartType0']['Masses'][:] * 1e10 / h
            posGasE = file['PartType0']['Coordinates'][:] * factor
            posGasE = posGasE - Cen
            rGasE = np.linalg.norm(posGasE, axis=1)
                
        elif Ycase == 'Mstellar':
            y =  file['PartType4']['Masses'][:] * 1e10 / h
            
        elif Ycase == 'Mgas':
            y =  file['PartType0']['Masses'][:] * 1e10 / h
            
        elif Ycase == 'Potential':
            y =  file['PartType0']['Potential'][:] / a

        
        elif Ycase == 'Mdm':
            y =  file['Header'].attrs['MassTable'][1]*np.ones(len(file['PartType1']['Coordinates'])) * 1e10 / h
        
        elif Ycase == 'MassesDM':
            y =  file['Header'].attrs['MassTable'][1]*np.ones(len(file['PartType1']['Coordinates'])) * 1e10 / h
            
        elif Ycase == 'SFR' or Ycase == 'SFE':
            y = file['PartType0']['StarFormationRate'][:]
            
        elif Ycase == 'Age':
            mass = file['PartType4']['Masses'][:] * 1e10 / h
            a = file['PartType4']['GFM_StellarFormationTime'][:]        
            pos = file['PartType4']['Coordinates'][:] * factor
            pos = pos - Cen
            r = np.linalg.norm(pos, axis=1)
            aInvert = 1/a
            z = aInvert - 1
            zNow = 1/file['Header'].attrs['Time'] - 1
            y = ETNG.AgeUniverse(Omegam0,h,zNow) - ETNG.AgeUniverse(Omegam0,h,z)
            
        elif Ycase == 'EnergyDissipation':
            y = file[PartType][Ycase][:] * (1e10 / a) / (factor) 
            y = y *( 2e33 / 3.086e21 ) * 1e5 
        
        elif Ycase == 'MagneticField':
            y = file[PartType][Ycase][:] * (h / a**2) *2.6e-6
            y =  np.log10(np.linalg.norm(y, axis=1))
        
        elif Ycase == 'StarOverGas':
            y = file['PartType0']['Masses'][:] * 1e10 / h     
            massStar = file['PartType4']['Masses'][:] * 1e10 / h
            pos = file['PartType0']['Coordinates'][:] * factor
            pos = pos - Cen
            r = np.linalg.norm(pos, axis=1)
            posStar = file['PartType4']['Coordinates'][:] * factor
            posStar = posStar - Cen
            rstar = np.linalg.norm(posStar, axis=1)
            
        elif Ycase == 'GFM_MetalsTagged_SNIa':
            y = file['PartType0']['GFM_MetalsTagged'][:, 0]
           
        elif Ycase == 'GFM_MetalsTagged_SNII':
            y = file['PartType0']['GFM_MetalsTagged'][:, 1]
        
        else:
            y = file[PartType][Ycase][:]
            
        # Initialize bin edges and densities
        if log_spacing:
            if np.min(r) <= rmin:
                minvalue = rmin
            else:
                minvalue = np.min(r)
            if np.max(r) >= rmax:
                maxvalue = rmax
            else:
                maxvalue = np.max(r)
                
            if minvalue == 0:
                minvalue = 0.08
            if maxvalue == 0:
                maxvalue = 10
                
            bin_edges = np.geomspace(minvalue, maxvalue, nbins+1)
    
        else:
            bin_edges = np.linspace(np.min(r), np.max(r), nbins+1)
            
           
        if 'GasVelInflow' in Condition and PartType == 'PartType0':
            Hr = cos.Eofz(Omegam0, 1.-Omegam0, z) * h * 100 * r /1e3
            velrad = np.sum(pos*vel, axis = 1)/r  + Hr
        
            args = np.argwhere(velrad < -50*scalefactorsqrt).T[0]
            args = np.argwhere(velrad < np.nanquantile(velrad[velrad < 0], 0.05)).T[0]

            y = y[args]
            mass = mass[args]
            r = r[args]
            
        
        elif 'GasVelOutflow' in Condition and PartType == 'PartType0':
            Hr = cos.Eofz(Omegam0, 1.-Omegam0, z) * h * 100 * r /1e3
            velrad = np.sum(pos*vel, axis = 1)/r  + Hr
            args = np.argwhere(velrad > 50*scalefactorsqrt).T[0]

            if np.nanquantile(velrad, 0.95) < 0:
                args = np.argwhere(velrad >  np.nanquantile(velrad[velrad > 0], 0.95)).T[0]
            
            else:
                args = np.argwhere(velrad >  np.nanquantile(velrad, 0.95)).T[0]
                
            y = y[args]
            mass = mass[args]
            r = r[args]
     
 
        rad = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        yrad = np.zeros(nbins)
        yerr = np.zeros(nbins)
        
        
        # Iterate over bins and compute density
        argGalactic = np.argwhere( r <= 2*10**HalfRad ).T[0]
        argGalactic = np.argwhere( r <= 2*10**HalfRad ).T[0]
        argICGM = np.argwhere((r > 2*10**HalfRad) & ( r <= 10**HalfGasRad)).T[0]
        argOCGM = np.argwhere( r > 10**HalfGasRad ).T[0]
        argEOCGM = np.argwhere( r > 2*10**HalfGasRad ).T[0]
        argGasHalf = np.argwhere( r < 10**HalfGasRad ).T[0]
        
        argLower2kpc = np.argwhere( r <= 5 ).T[0]
        argAbove2kpc = np.argwhere( r > 5 ).T[0]

        
        if 'ICGM' in Condition:
            y = y[argICGM]
            mass = mass[argICGM]
            r = r[argICGM]
            
        elif 'OCGM' in Condition:
            y = y[argOCGM]
            mass = mass[argOCGM]
            r = r[argOCGM]
            
        elif 'ECGM' in Condition:

            y = y[argEOCGM]
            mass = mass[argEOCGM]
            r = r[argEOCGM]    
            
        elif 'Galactic' in Condition:

            y = y[argGalactic]
            mass = mass[argGalactic]
            r = r[argGalactic]  
            
        elif 'GasHalf' in Condition:

            y = y[argGasHalf]
            mass = mass[argGasHalf]
            r = r[argGasHalf]
            
        elif 'Lower2kpc' in Condition:
            y = y[argLower2kpc]
            mass = mass[argLower2kpc]
            r = r[argLower2kpc]
            
        elif 'Above2kpc' in Condition:
            y = y[argAbove2kpc]
            mass = mass[argAbove2kpc]
            r = r[argAbove2kpc]
            
         
            
        
        if Histogram:
            yHist = np.append(yHist, y)
            rHist = np.append(rHist, r)
            continue
        
        for i in range(nbins):
            in_bin = (r >= bin_edges[i]) & (r < bin_edges[i+1])
            mass_in_bin = mass[in_bin]
            y_in_bin = y[in_bin]
            volume = 4.0/3.0 * np.pi * (bin_edges[i+1]**3 - bin_edges[i]**3)
            if len(in_bin) < 50 and not 'Mass' in Ycase:
                
                yrad[i] = np.nan
                yerr[i] = np.nan
            if 'Density' in Ycase:
                yrad[i] = np.sum(y_in_bin) /  volume
                
                array = y_in_bin /  volume
                ypatternOriginal = np.median(array)
    
                yerr[i] = ResampleBoot(array[~np.isnan(array)], y_in_bin, ypatternOriginal, nboot )
                
            elif Ycase == 'SFE':
                yrad[i] = np.sum(y_in_bin) /  np.sum(mass_in_bin)
                
                array = y_in_bin /  np.sum(mass_in_bin)
                ypatternOriginal = np.median(array)
    
                yerr[i] = ResampleBoot(array[~np.isnan(array)], y_in_bin, ypatternOriginal, nboot )
                    
                
            elif Ycase in ['Mass', 'SFR', 'Mstellar', 'Mgas', 'Mdm']:
                yrad[i] = np.sum(y_in_bin)
                
                array = y_in_bin 
                ypatternOriginal = np.median(array)
    
                yerr[i] = ResampleBoot(array[~np.isnan(array)], y_in_bin, ypatternOriginal, nboot )
                
            elif Ycase in ['RadVelocity', 'PhiVelocity','Age', 'T', 'n', 'z', 'j', 'Potential', 'EnergyDissipation', 'Sigma']:
                dens = np.sum(mass_in_bin) /  volume
                dens_in_bin = dens*mass_in_bin /  np.sum(mass_in_bin)
                yrad[i] = np.sum(y_in_bin*mass_in_bin) / np.sum(mass_in_bin)
                
                
                array = y_in_bin*mass_in_bin / np.sum(mass_in_bin)
                ypatternOriginal = np.median(array)
    
                yerr[i] = ResampleBoot(array[~np.isnan(array)], y_in_bin, ypatternOriginal, nboot )
                
            elif Ycase in ['PhiOverRad']:
                dens = np.sum(mass_in_bin) /  volume
                dens_in_bin = dens*mass_in_bin /  np.sum(mass_in_bin)
                yR_in_bin = yR[in_bin]
                yR_median = np.sum(yR_in_bin*dens_in_bin) / dens
                yrad[i] = (np.sum(y_in_bin*dens_in_bin) / dens) / yR_median
                array = (y_in_bin*dens_in_bin / dens) / (yR_in_bin*dens_in_bin / dens)
                ypatternOriginal = np.median(array)
    
                yerr[i] = ResampleBoot(array[~np.isnan(array)], y_in_bin, ypatternOriginal, nboot )
            
            elif Ycase in ['PhiOverDisp']:
                dens = np.sum(mass_in_bin) /  volume
                dens_in_bin = dens*mass_in_bin /  np.sum(mass_in_bin)
                yR_in_bin = yR[in_bin]
                yR_median = np.sum(yR_in_bin*dens_in_bin) / dens
                yrad[i] = (np.sum(y_in_bin*dens_in_bin) / dens) / yR_median
                array = (y_in_bin*dens_in_bin / dens) / (yR_in_bin*dens_in_bin / dens)
                ypatternOriginal = np.median(array)
    
                yerr[i] = ResampleBoot(array[~np.isnan(array)], y_in_bin, ypatternOriginal, nboot )
                
            elif Ycase in ['zOverRad']:
                dens = np.sum(mass_in_bin) /  volume
                dens_in_bin = dens*mass_in_bin /  np.sum(mass_in_bin)
                yR_in_bin = yR[in_bin]
                yR_median = np.sum(yR_in_bin*dens_in_bin) / np.sum(mass_in_bin)
                radBin = (bin_edges[i] + bin_edges[i+1]) / 2.
                yrad[i] = (np.sum(y_in_bin*dens_in_bin) / np.sum(dens_in_bin)) / radBin
                array = (y_in_bin*dens_in_bin / np.sum(dens_in_bin)) / radBin
                ypatternOriginal = np.median(array)
    
                yerr[i] = ResampleBoot(array[~np.isnan(array)], y_in_bin, ypatternOriginal, nboot )
                
            elif Ycase == 'PhiOverCirc':
                dens = np.sum(mass_in_bin) /  volume
                dens_in_bin = dens*mass_in_bin /  np.sum(mass_in_bin)
                VelPhi = np.sum(y_in_bin*dens_in_bin) / dens
                radBin = (bin_edges[i] + bin_edges[i+1]) / 2.
                
                in_bin_STAR = (rstar <= radBin)
                massSTAR_in_bin = np.sum(massStar[in_bin_STAR])
                in_bin_DM = (rdm  <= radBin)
                massDM_in_bin = np.sum(massDM[in_bin_DM])
                in_bin_Gas = (rgas  <= radBin)
                massGas_in_bin = np.sum(massGas[in_bin_Gas])
                MassTot = np.sum(massGas_in_bin+massSTAR_in_bin+massDM_in_bin)
                
                GkpcMsun = 4.3009e-6
                
                Vcirc = np.sqrt(GkpcMsun * MassTot / radBin)
                yrad[i] = np.abs(VelPhi / Vcirc)
                
                array = y_in_bin*dens_in_bin / dens / Vcirc
                ypatternOriginal = np.median(array)
                y_in_bin = np.abs(y[in_bin] / Vcirc)
                yerr[i] = ResampleBoot(array[~np.isnan(array)], y_in_bin, ypatternOriginal, nboot )
                
                
            elif Ycase == 'sSFR':
                in_bin_STAR = (rstar >= bin_edges[i]) & (rstar < bin_edges[i+1])
                massSTAR_in_bin = massStar[in_bin_STAR]
                yrad[i] = np.sum(y_in_bin)  / np.sum(massSTAR_in_bin)
                
                array = y_in_bin  / np.sum(massSTAR_in_bin)
                ypatternOriginal = np.median(array)
    
                yerr[i] = ResampleBoot(array[~np.isnan(array)], y_in_bin, ypatternOriginal, nboot )
                
            elif Ycase == 'sSFRE':
                in_bin_Gas = (rGasE >= bin_edges[i]) & (rGasE < bin_edges[i+1])
                massGas_in_bin = massGasE[in_bin_Gas]
                yrad[i] = np.sum(y_in_bin)  / np.sum(massGas_in_bin)
                
                array = y_in_bin  / np.sum(massGas_in_bin)
                ypatternOriginal = np.median(array)
    
                yerr[i] = ResampleBoot(array[~np.isnan(array)], y_in_bin, ypatternOriginal, nboot )
                    
                
            elif Ycase == 'StarOverGas':
                in_bin_STAR = (rstar >= bin_edges[i]) & (rstar < bin_edges[i+1])
                massSTAR_in_bin = massStar[in_bin_STAR]
                yrad[i] = np.sum(massSTAR_in_bin) / np.sum(y_in_bin)
                
                array = np.sum(massSTAR_in_bin) / y_in_bin
                ypatternOriginal = np.median(array)
    
                yerr[i] = ResampleBoot(array[~np.isnan(array)], y_in_bin, ypatternOriginal, nboot )
                
            elif Ycase != 'Mass':
                dens = np.sum(mass_in_bin) /  volume
                dens_in_bin = dens*mass_in_bin /  np.sum(mass_in_bin)
                if len(y_in_bin) > 0 and len(y_in_bin[~np.isinf(y_in_bin)]) > 0:
                    y_in_bin[np.isinf(y_in_bin)] = 5*max(y_in_bin[~np.isinf(y_in_bin)])
                if len(y_in_bin) > 0 and len(y_in_bin[~np.isinf(y_in_bin)]) == 0:
                    y_in_bin[np.isinf(y_in_bin)] = 0
                
                yrad[i] = np.sum(y_in_bin*dens_in_bin) / dens
                array = y_in_bin*dens_in_bin / dens
                ypatternOriginal = np.median(array)
                
           

            if not 'Velocity' in Ycase:
                if yrad[i] < 1e-12:
                    yrad[i] = np.nan
                    yerr[i] = np.nan
                    

        if Ycase == 'Mass':
            yrad = np.cumsum(yrad)
            
        if l == 0 or len(yIDs) == 0:
            yIDs = np.append(yIDs, yrad)
            xIDs = np.append(xIDs, rad)


        else:
            yIDs = np.vstack((yIDs, yrad))
            xIDs = np.vstack((xIDs, rad))
           
    if Histogram:
        return yHist, rHist
    Rvalues = xIDs.T
    Values = yIDs.T

    x = np.array([])
    y = np.array([])
    y_err = np.array([])
    
    

    if len(Values) > 0:
        if len(Values.shape) > 1:
            for k, value in enumerate(Values):
                x = np.append(x, np.nanmedian(Rvalues[k]))
                y = np.append(y, np.nanmedian(value))
                y_errpartial = MATH.boostrap_func(value, func=np.nanmedian, num_boots= nboot)
                if abs(y_errpartial) > abs(np.nanmedian(value)):
                    y_err = np.append(y_err, np.nanmedian(y_err))
                elif type(y_errpartial) == np.float64:
                    y_err = np.append(y_err,y_errpartial) 
                else:
                    y_err = np.append(y_err, np.nanmedian(y_err))
        else:
            x = Rvalues
            y = Values
            y_err = 0*Values
            

    else:
        x = np.nan
        y = np.nan
        y_err = np.nan
        
    return x, y, y_err

def extractParticles(SubfindID99, snaps = [99], IDatSnap = False,
                 PATH = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory',
                 SIMS = 'TNG50-1', SIMTNG = 'TNG50', HOME = os.getenv("HOME")+'/' , NEW = False):
    
    path = PATH
    PATH = PATH + '/' + SIMTNG
    file = []
    paths = []
    dfSubfindID = extractDF('SubfindID', PATH = path)
    
    for snap in snaps:
        #print('ID: '+str(SubfindID99)+' snap: '+str(snap))
        
        if not IDatSnap:
            SubfindID = dfSubfindID[str(SubfindID99)].iloc[99 - snap]
            if np.isnan(SubfindID):
                continue
            SubfindID = int(SubfindID)
        else:
            SubfindID = int(SubfindID99)

        hdf5_path = HOME+'SIMS/TNG/'+SIMS+'/snapshots/'+str(snap)+'/subhalos/'+str(SubfindID)+'/cutout_'+SIMS+'_'+str(snap)+'_'+str(SubfindID)+'.hdf5'
        
        if not NEW:
            try:
                
                file.append(h5py.File(hdf5_path, 'r'))
                if IDatSnap:
                    return h5py.File(hdf5_path, 'r')
        
            except Exception:
                ETNG.saveParticles(SubfindID, snapnum=snap)
    
                if IDatSnap:
                    return h5py.File(hdf5_path, 'r')
                
                file.append(h5py.File(hdf5_path, 'r'))
        elif NEW:
            
            os.remove(hdf5_path)
            
            ETNG.saveParticles(SubfindID, snapnum=snap)

            if IDatSnap:
                return h5py.File(hdf5_path, 'r')
            
            file.append(h5py.File(hdf5_path, 'r'))
            
        try:
            if not IDatSnap:
                
                shutil.copyfile(hdf5_path, PATH+'/Particles/'+str(SubfindID99)+'/'+str(snap)+'.hdf5')
                

        except Exception:
            path = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory/'+SIMTNG
            for name in ['Particles', str(SubfindID99)]:
                path = os.path.join(path, name)
                if not os.path.isdir(path):
                    os.mkdir(path)
            if not IDatSnap:
                shutil.copyfile(hdf5_path, PATH+'/Particles/'+str(SubfindID99)+'/'+str(snap)+'.hdf5')
            
        paths.append(hdf5_path)
    
    return file

def corrected(SubfindID99, snaps = [99], PATH = os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory',  SIMTNG = 'TNG50'):
    

    dfSubhalo = extractDF(str(SubfindID99), PATH=PATH, SUBHALO = True)
    dfSubhalo = ETNG.ConvertHDF52df(dfSubhalo)

    SubhaloHalfMassRadType4 = extractDF('SubhaloHalfmassRadType4', PATH = PATH)
    SubhaloHalfMassRadType0 = extractDF('SubhaloHalfmassRadType0', PATH = PATH)
    dfTime = extractDF('SNAPS_TIME', PATH = PATH)
    
    
    for snap in snaps:
        #### SUBHALO
        factor =  1. / h / (1+dfTime.z[int(99-snap)])
        scalefactorsqrt = np.sqrt(1. / (1+dfTime.z[int(99-snap)]))
        
        try:
            Rad = 2.*10**SubhaloHalfMassRadType0[str(SubfindID99)].loc[SubhaloHalfMassRadType0.Snap == snap].values[0]

            if Rad == 0:
                Rad = 2.*10**SubhaloHalfMassRadType4[str(SubfindID99)].loc[SubhaloHalfMassRadType4.Snap == snap].values[0]
            
        except Exception:
            continue

        
        #HDF5
        file = extractParticles(SubfindID99, snaps = [snap])[0]  

        file.close()
        
        file = h5py.File(PATH+'/'+SIMTNG+'/Particles/'+str(SubfindID99)+'/'+str(snap)+'.hdf5', 'r+')
        file["Header"].attrs["NumPart_Total"] = np.array(file["Header"].attrs["NumPart_ThisFile"])
        
        BoxSize = file['Header'].attrs['BoxSize'] * factor / h
        
        if not 'Config' in file.keys():
            file.create_group("Config")
            file["/Config"].attrs["VORONOI"] = 1
            
        try:
            posStar = file['PartType4']['Coordinates'][:] * factor
            massStar = file['PartType4']['Masses'][:] * 1e10 / h
            velStar = file['PartType4']['Velocities'][:] * scalefactorsqrt
            
            Cen = np.array([MATH.weighted_median(posStar[:, 0], massStar), MATH.weighted_median(posStar[:, 1], massStar), MATH.weighted_median(posStar[:, 2], massStar)])
            VelMean = np.array([MATH.weighted_median(velStar[:, 0], massStar), MATH.weighted_median(velStar[:, 1], massStar), MATH.weighted_median(velStar[:, 2], massStar)])
            CenFind = True
            
        except Exception:
            posStar = velStar =  np.array([0, 0, 0])
            massStar = np.array([0])  
            CenFind = False
            
        try:
            posGas = file['PartType0']['Coordinates'][:] * factor
            massGas = file['PartType0']['Masses'][:] * 1e10 / h
            velGas = file['PartType0']['Velocities'][:] * scalefactorsqrt  
            
            if not CenFind:
                Cen = np.array([MATH.weighted_median(posGas[:, 0], massGas), MATH.weighted_median(posGas[:, 1], massGas), MATH.weighted_median(posGas[:, 2], massGas)])
                VelMean = np.array([MATH.weighted_median(velGas[:, 0], massGas), MATH.weighted_median(velGas[:, 1], massGas), MATH.weighted_median(velGas[:, 2], massGas)])
                CenFind = True
                
        except Exception:
            posGas = velGas = np.array([0, 0, 0])
            massGas = np.array([0])
            
            
            
        try:
            posDM = file['PartType1']['Coordinates'][:] * factor
            massDM = file['Header'].attrs['MassTable'][1]*np.ones(len(file['PartType1']['Coordinates'])) * 1e10 / h
            velDM = file['PartType1']['Velocities'][:] * scalefactorsqrt
            
            if not CenFind:
                Cen = np.array([MATH.weighted_median(posDM[:, 0], massDM), MATH.weighted_median(posDM[:, 1], massDM), MATH.weighted_median(posDM[:, 2], massDM)])
                VelMean = np.array([MATH.weighted_median(velDM[:, 0], massDM), MATH.weighted_median(velDM[:, 1], massDM), MATH.weighted_median(velDM[:, 2], massDM)])
                CenFind = True
           
        except Exception:
            posDM = velDM = np.array([0, 0, 0])
            massDM = np.array([0])
            
        try:
            posBH = file['PartType5']['Coordinates'][:] * factor
            velBH = file['PartType5']['Velocities'][:] * scalefactorsqrt
           
        except Exception:
            posBH = velBH = np.array([0, 0, 0])
            
        
        ##  
        posStar = FixPeriodic_kpc(posStar - Cen, snap)
        posGas = FixPeriodic_kpc(posGas - Cen, snap)
        posDM = FixPeriodic_kpc(posDM - Cen, snap)
        posBH = FixPeriodic_kpc(posBH - Cen, snap)

        velStar = FixPeriodic_kpc(velStar - VelMean, snap)
        velGas = FixPeriodic_kpc(velGas - VelMean, snap)
        velDM = FixPeriodic_kpc(velDM - VelMean, snap)
        velBH = FixPeriodic_kpc(velBH - VelMean, snap)
        
           
        for key in file.keys():
            #####
            # Coordinates and Velocities
            #####
    
                
            if not 'PartType' in key:
                continue
            else:
                if 'PartType0' in key:
                    pos = posGas
                    vel = velGas
                elif 'PartType1' in key:
                    pos = posDM
                    vel = velDM
                elif 'PartType4' in key:
                    pos = posStar
                    vel = velStar
                elif 'PartType5' in key:
                    pos = posBH
                    vel = velBH    
        
            if key == 'PartType0':
                if 'CenterOfMass' in file[key].keys():
                    posCen = np.transpose(np.array([file[key]['CenterOfMass'][:,0] * factor   / h, 
                                                file[key]['CenterOfMass'][:,1] * factor   / h, 
                                                file[key]['CenterOfMass'][:,2] * factor   / h]))
                    posCen = posCen - Cen
                else:
                    posCen = pos
            else:
                posCen = pos
                
            
       
            if 'PartType0' in file.keys():
                #ang_mom = np.sum(np.cross(PosGas, Masses[:, np.newaxis] * VelGas), axis=0)
                RadMass = np.linalg.norm(posGas, axis = 1)
                MassInRad = massGas[RadMass < Rad]
                ang_mom = np.sum(np.cross(posGas[RadMass < Rad], MassInRad[:, np.newaxis] * velGas[RadMass < Rad]), axis=0)
            else:
                #ang_mom = np.sum(np.cross(PosGas, Masses[:, np.newaxis] * VelGas), axis=0)
                try:
                    RadMass = np.linalg.norm(posStar, axis = 1)
                    MassInRad = massStar[RadMass < Rad]
                    ang_mom = np.sum(np.cross(posStar[RadMass < Rad], MassInRad[:, np.newaxis] * velStar[RadMass < Rad]), axis=0)


                except Exception:
                    RadMass = np.linalg.norm(posDM, axis = 1)
                    MassInRad = massDM[RadMass < Rad]
                    ang_mom = np.sum(np.cross(posDM[RadMass < Rad], MassInRad[:, np.newaxis] * velDM[RadMass < Rad]), axis=0)
  
           
            cen = np.transpose(np.array([Cen[0] , 
                                         Cen[1] , 
                                         Cen[2] ])) * np.ones((len(pos), 3))
            
          
            #print('pos, ', pos)
            POS, POSC = MATH.CartesiantToRotated(pos,posCen,ang_mom)
            CEN, VEL = MATH.CartesiantToRotated(cen,vel,ang_mom)
            #print('posAfter, ', POS)

            if key == 'PartType0':
                test = np.sum(np.cross(POS, massGas[:, np.newaxis] * VEL), axis=0)
                test = test / np.linalg.norm(test)
                
                while BoxSize  < 10*max( abs(POS.max()), abs(POS.min())):

                    file['Header'].attrs['BoxSize'] = 2*BoxSize * h / factor
                    BoxSize = file['Header'].attrs['BoxSize'] * factor / h
                #print(file['Header'].attrs['BoxSize'])

            file[key]['Coordinates'][:,0] = POS[:, 0] 
            file[key]['Coordinates'][:,1] = POS[:, 1]
            file[key]['Coordinates'][:,2] = POS[:, 2]
            
            
            file[key]['Velocities'][:,0] = VEL[:, 0]
            file[key]['Velocities'][:,1] = VEL[:, 1] 
            file[key]['Velocities'][:,2] = VEL[:, 2] 

            if key == 'PartType0':
                if 'CenterOfMass' in file[key].keys():
                    file[key]['CenterOfMass'][:,0] = POSC[:, 0] 
                    file[key]['CenterOfMass'][:,1] = POSC[:, 1] 
                    file[key]['CenterOfMass'][:,2] = POSC[:, 2] 
    
        #file.flush()

    return file


def ExtractParticlesParameters(param, f, snap, scalefactorsqrt, PartType = 'PartType4'):
    
    
    if param == 'Coordinates':
        try:
            return f[PartType]['Coordinates'][:] / (1+dfTime.z.values[dfTime.Snap == snap]) / h 
        except Exception:
            return np.array([[0,0,0]])
    elif param == 'Velocities':
        try:
            return f[PartType]['Velocities'][:]  * scalefactorsqrt
        except Exception:
            return np.array([[0,0,0]])
    elif param == 'Masses':
        if PartType == 'PartType1':
            try:
                return f['Header'].attrs['MassTable'][1]*np.ones(len(f['PartType1']['Coordinates'])) * 1e10 / h
            except Exception:
                return np.array([0])
        else:
            try:
                return f[PartType]['Masses'][:]  * 1e10 / h
            except Exception:
                return np.array([0])

def FixPeriodic_kpc(dx, snap, sim='TNG50-1'):
    """
    Handle periodic boundary conditions
    Arguments:
        dx: difference in positions (in kpc/h)
        sim: simulation (default "TNG50-1")

    Returns: dx corrrected for periodic box (in kpc/h)
    Author: Abhner P de Almeida, modified from Gary Mamon (gam AAT iap.fr)
    """
    if sim=='TNG300-1':
    	L = 205000.0 / (1+dfTime.z.values[dfTime.Snap == snap]) / h 
    elif sim=='TNG50-1':
    	L = 35000.0 / (1+dfTime.z.values[dfTime.Snap == snap]) / h 
    else:
    	L = ETNG.BoxSize(sim) # BoxSize is in kpc/h
    # L = getsim(sim)['boxsize']
    dx = np.where(dx>L/2,dx-L,dx)
    dx = np.where(dx<-L/2,dx+L,dx)
    return dx


def ExtractExSituParticles(IDs, snaps = [99], UpdateParams = True):
    
    
    dfTime['a'] = 1/ (1. + dfTime.z)
    
    for snap in snaps:

        for ID in IDs:
            print('ID: ', ID)
            try:
                df = extractDF(str(snap)+'/'+str(ID)+'_StellarContent', EXsituINsitu = True)
                
                if UpdateParams: 
                    f = extractParticles(ID, snaps = [snap])[0]
                    scalefactorsqrt = np.sqrt(1. / (1+dfTime.z[int(99-snap)]))
                    
                    ###
                    posStar = ExtractParticlesParameters('Coordinates', f, snap, scalefactorsqrt, PartType = 'PartType4')
                    velStar = ExtractParticlesParameters('Velocities', f, snap, scalefactorsqrt, PartType = 'PartType4')
                    massStar = ExtractParticlesParameters('Masses', f, snap, scalefactorsqrt, PartType = 'PartType4')
                    
                    posDM = ExtractParticlesParameters('Coordinates', f, snap, scalefactorsqrt, PartType = 'PartType1')
                    velDM = ExtractParticlesParameters('Velocities', f, snap, scalefactorsqrt, PartType = 'PartType1')
                    massDM = ExtractParticlesParameters('Masses', f, snap, scalefactorsqrt, PartType = 'PartType1')
                    
                    posGas = ExtractParticlesParameters('Coordinates', f, snap, scalefactorsqrt, PartType = 'PartType0')
                    velGas = ExtractParticlesParameters('Velocities', f, snap, scalefactorsqrt, PartType = 'PartType0')
                    massGas = ExtractParticlesParameters('Masses', f, snap, scalefactorsqrt, PartType = 'PartType0')
                    
                    posBH = ExtractParticlesParameters('Coordinates', f, snap, scalefactorsqrt, PartType = 'PartType5')
                    velBH = ExtractParticlesParameters('Velocities', f, snap, scalefactorsqrt, PartType = 'PartType5')
                    massBH = ExtractParticlesParameters('Masses', f, snap, scalefactorsqrt, PartType = 'PartType5')

                    ###
                
                    Cen = np.array([MATH.weighted_median(posStar[:, 0], massStar), MATH.weighted_median(posStar[:, 1], massStar), MATH.weighted_median(posStar[:, 2], massStar)])
                    VelBulk = np.array([MATH.weighted_median(velStar[:, 0], massStar), MATH.weighted_median(velStar[:, 1], massStar), MATH.weighted_median(velStar[:, 2], massStar)])
                
                
                    ###
                    posStar = FixPeriodic_kpc(posStar - Cen, snap)
                    velStar = FixPeriodic_kpc(velStar - VelBulk, snap)
                    rStar = np.linalg.norm(posStar, axis=1)
                    velRadStar = np.sum(posStar*velStar, axis = 1)/rStar
                    Jstar = np.cross(posStar, velStar, axis = 1)
                    
                    posDM = FixPeriodic_kpc(posDM - Cen, snap)
                    velDM = FixPeriodic_kpc(velDM - VelBulk, snap)
                    
                    posGas = FixPeriodic_kpc(posGas - Cen, snap)
                    velGas = FixPeriodic_kpc(velGas - VelBulk, snap)
                    
                    posBH = FixPeriodic_kpc(posBH - Cen, snap)
                    velBH = FixPeriodic_kpc(velBH - VelBulk, snap)
                    
                    
                    ###
                    posParticles = posStar
                    velParticles = velStar
                    massParticles = massStar
                    
                    
                    if not (len(posGas) == 1 and massGas[0] == 0):
                        posParticles = np.vstack([posParticles, posGas])
                        velParticles = np.vstack([velParticles, velGas])
                        massParticles = np.concatenate((massParticles, massGas))
                        
                    if not (len(posBH) == 1 and massBH[0] == 0):
                        
                        posParticles = np.vstack([posParticles, posBH])
                        velParticles = np.vstack([velParticles, velBH])
                        massParticles = np.concatenate((massParticles,  massBH))
                        
                    if not (len(posDM) == 1 and massDM[0] == 0):
                        
                        posParticles = np.vstack([posParticles, posDM])
                        velParticles = np.vstack([velParticles, velDM])
                        massParticles = np.concatenate((massParticles, massDM))

                    
                    phiParticles = Potential(posParticles,massParticles, G = G)

                    
                    ###
                    
                    df['Potential'] = phiParticles[:len(massStar)]
                    
                    df.to_csv('/home/abhner/TNG_Analyzes/SubhaloHistory/TNG50/DFs/'+str(ID)+'_StellarContent.csv')
                    
            except Exception:
                
                f = extractParticles(ID, snaps = [snap])[0]
                try:
                    as_Birth = np.array([a for a in f['PartType4']['GFM_StellarFormationTime'][:]])
                except Exception:
                    continue
                IDsPart = np.array([idNum for idNum in f['PartType4']['ParticleIDs'][:]])
            
                df = pd.DataFrame(data=np.zeros((len(IDsPart),6), dtype=object), columns= ['IDParticle', 'Snap_at_Birth', 'BirthTag', 'r', 'vrad', 'mass'])
                df[:] = np.nan
                df.IDParticle = IDsPart
                df.BirthTag = 'None'
            
                scalefactorsqrt = np.sqrt(1. / (1+dfTime.z[int(99-snap)]))
            
                ###
                posStar = ExtractParticlesParameters('Coordinates', f, snap, scalefactorsqrt, PartType = 'PartType4')
                velStar = ExtractParticlesParameters('Velocities', f, snap, scalefactorsqrt, PartType = 'PartType4')
                massStar = ExtractParticlesParameters('Masses', f, snap, scalefactorsqrt, PartType = 'PartType4')
                
                posDM = ExtractParticlesParameters('Coordinates', f, snap, scalefactorsqrt, PartType = 'PartType1')
                velDM = ExtractParticlesParameters('Velocities', f, snap, scalefactorsqrt, PartType = 'PartType1')
                massDM = ExtractParticlesParameters('Masses', f, snap, scalefactorsqrt, PartType = 'PartType1')
                
                posGas = ExtractParticlesParameters('Coordinates', f, snap, scalefactorsqrt, PartType = 'PartType0')
                velGas = ExtractParticlesParameters('Velocities', f, snap, scalefactorsqrt, PartType = 'PartType0')
                massGas = ExtractParticlesParameters('Masses', f, snap, scalefactorsqrt, PartType = 'PartType0')
                
                posBH = ExtractParticlesParameters('Coordinates', f, snap, scalefactorsqrt, PartType = 'PartType5')
                velBH = ExtractParticlesParameters('Velocities', f, snap, scalefactorsqrt, PartType = 'PartType5')
                massBH = ExtractParticlesParameters('Masses', f, snap, scalefactorsqrt, PartType = 'PartType5')

                ###
                
                Cen = np.array([MATH.weighted_median(posStar[:, 0], massStar), MATH.weighted_median(posStar[:, 1], massStar), MATH.weighted_median(posStar[:, 2], massStar)])
                VelBulk = np.array([MATH.weighted_median(velStar[:, 0], massStar), MATH.weighted_median(velStar[:, 1], massStar), MATH.weighted_median(velStar[:, 2], massStar)])
            
            
                ###
                posStar = FixPeriodic_kpc(posStar - Cen, snap)
                velStar = FixPeriodic_kpc(velStar - VelBulk, snap)
                rStar = np.linalg.norm(posStar, axis=1)
                velRadStar = np.sum(posStar*velStar, axis = 1)/rStar
                Jstar = np.cross(posStar, velStar, axis = 1)
                
                posDM = FixPeriodic_kpc(posDM - Cen, snap)
                velDM = FixPeriodic_kpc(velDM - VelBulk, snap)
                
                posGas = FixPeriodic_kpc(posGas - Cen, snap)
                velGas = FixPeriodic_kpc(velGas - VelBulk, snap)
                
                posBH = FixPeriodic_kpc(posBH - Cen, snap)
                velBH = FixPeriodic_kpc(velBH - VelBulk, snap)
                
                ###
                posParticles = posStar
                velParticles = velStar
                massParticles = massStar
                
                
                if not (len(posGas) == 1 and massGas[0] == 0):
                    posParticles = np.vstack([posParticles, posGas])
                    velParticles = np.vstack([velParticles, velGas])
                    massParticles = np.concatenate((massParticles, massGas))
                    
                if not (len(posBH) == 1 and massBH[0] == 0):
                    
                    posParticles = np.vstack([posParticles, posBH])
                    velParticles = np.vstack([velParticles, velBH])
                    massParticles = np.concatenate((massParticles,  massBH))
                    
                if not (len(posDM) == 1 and massDM[0] == 0):
                    
                    posParticles = np.vstack([posParticles, posDM])
                    velParticles = np.vstack([velParticles, velDM])
                    massParticles = np.concatenate((massParticles, massDM))

                
                phiParticles = Potential(posParticles,massParticles, G = G)

                
                ###
                
                df['r'] = rStar
                df['vrad'] = velRadStar
                df['VelX'] = velStar[:, 0]
                df['VelY'] = velStar[:, 1]
                df['VelZ'] = velStar[:, 2]
                df['jX'] = Jstar[:, 0]
                df['jY'] = Jstar[:, 1]
                df['jZ'] = Jstar[:, 2]
                df['PosX'] = posStar[:, 0]
                df['PosY'] = posStar[:, 1]
                df['PosZ'] = posStar[:, 2]
                df['mass'] = massStar
                df['Potential'] = phiParticles[:len(massStar)]
    
            
            
                for l, idNum in enumerate(IDsPart):
            
                    epsilon = 0.1
                    while len(dfTime.loc[abs(dfTime.a - as_Birth[l]) < epsilon, 'Snap'].values) > 2:
                        Snaps = dfTime.loc[abs(dfTime.a - as_Birth[l]) < epsilon, 'Snap'].values
                        epsilon = epsilon / 2.
                        if len(dfTime.loc[abs(dfTime.a - as_Birth[l]) < epsilon, 'Snap'].values) == 0:
                            break
                        
                    if Snaps[0] > snap:
                        df.loc[df.IDParticle == idNum, 'Snap_at_Birth'] = snap

                    else:
                        df.loc[df.IDParticle == idNum, 'Snap_at_Birth'] = Snaps[0]
                    
            
                for snapBreak in np.arange(100):
                    if snapBreak > snap:
                        continue
                    print('Snap: ', snapBreak)
                    if len(df.loc[df.Snap_at_Birth == snapBreak, 'IDParticle'].values) > 0:
                        try:
                            f_at_SnapBreak = extractParticles(ID, snaps=[snapBreak])[0]
                            IDsParticle_at_SnapBreak = f_at_SnapBreak['PartType4']['ParticleIDs'][:]
    
                        except Exception:
                            continue
                        
            
            
                        for idNum in df.loc[df.Snap_at_Birth == snapBreak, 'IDParticle'].values:
            
                            if (idNum in IDsParticle_at_SnapBreak) and (df.loc[df.IDParticle == idNum, 'BirthTag'].values[0] == 'None'):
                                df.loc[df.IDParticle == idNum, 'BirthTag'] = 'InSitu'         
            
                            if (not idNum in IDsParticle_at_SnapBreak) and (df.loc[df.IDParticle == idNum, 'BirthTag'].values[0] == 'None'):
                                df.loc[df.IDParticle == idNum, 'BirthTag'] = 'ExSitu'  
                        
            df.to_csv('/home/abhner/TNG_Analyzes/SubhaloHistory/TNG50/DFs/Analysis/StellarContent/'+str(snap)+'/'+str(ID)+'_StellarContent.csv')
    return 
                        
def ExtractRecentAccretedParticles(IDs, PartType = 'PartType0', snap = 99):
    
    
    dfTime['a'] = 1/ (1. + dfTime.z)

    for ID in IDs:
        print('ID: ', ID)
        try:
            df = extractDF(str(ID)+'_RecentAccretedContent')
        except Exception:
            f_Current = extractParticles(ID, snaps = [snap])[0]
            f_Previous = extractParticles(ID, snaps = [int(snap - 1)])[0]

            IDsPart_Current = np.array([idNum for idNum in f_Current[PartType]['ParticleIDs'][:]])
            IDsPart_Previous = np.array([idNum for idNum in f_Previous[PartType]['ParticleIDs'][:]])

            df = pd.DataFrame(data=np.zeros((len(IDsPart_Current),6), dtype=object), columns= ['IDParticle', 'Snap_at_Birth', 'BirthTag', 'r', 'vrad', 'mass'])
            df[:] = np.nan
            df.IDParticle = IDsPart_Current
            df.BirthTag = 'None'
        
            scalefactorsqrt = np.sqrt(1. / (1+dfTime.z[int(99-99)]))
        
            pos = f_Current['PartType0']['Coordinates'][:]  / (1+dfTime.z.values[dfTime.Snap == 99]) / h #kpc
            vel = f_Current['PartType0']['Velocities'][:] * scalefactorsqrt
            mass = f_Current['PartType0']['Masses'][:] * 1e10 / h
        
        
        
            Cen = np.array([MATH.weighted_median(pos[:, 0], mass), MATH.weighted_median(pos[:, 1], mass), MATH.weighted_median(pos[:, 2], mass)])
            VelBulk = np.array([MATH.weighted_median(vel[:, 0], mass), MATH.weighted_median(vel[:, 1], mass), MATH.weighted_median(vel[:, 2], mass)])
        
        
            pos = pos - Cen
            vel = vel - VelBulk
            r = np.linalg.norm(pos, axis=1)
            velRad = np.sum(pos*vel, axis = 1)/r
            
            df['r'] = r
            df['vrad'] = velRad
            df['VelX'] = vel[:, 0]
            df['VelY'] = vel[:, 1]
            df['VelZ'] = vel[:, 2]
            df['PosX'] = pos[:, 0]
            df['PosY'] = pos[:, 1]
            df['PosZ'] = pos[:, 2]
            df['mass'] = mass
        
        
            rParticleMedian = MATH.weighted_median(r, mass)
            
            for l, idNum in enumerate(IDsPart_Current):
                
                if (idNum in IDsPart_Previous) and (df.loc[df.IDParticle == idNum, 'BirthTag'].values[0] == 'None'):
                    df.loc[df.IDParticle == idNum, 'BirthTag'] = 'InSitu'         

                elif (not idNum in IDsPart_Previous) and (df.loc[df.IDParticle == idNum, 'r'].values[0] > rParticleMedian) :
                    df.loc[df.IDParticle == idNum, 'BirthTag'] = 'ExSitu_Outer'  
                    
                else:
                    df.loc[df.IDParticle == idNum, 'BirthTag'] = 'ExSitu_Inner'  
                        
            df.to_csv('/home/abhner/TNG_Analyzes/SubhaloHistory/TNG50/DFs/Analysis/RecentAccretedParticle/'+str(ID)+'_'+str(snap)+'_StellarContent.csv')
    return          
        

def ResampleBoot(array, y_in_bin, ypatternOriginal, nboot ):

    if len(array) > 5:
        
        n_size = int(len(array)*0.5)
        medians = np.zeros(nboot)
        
        for j in range(nboot):     
            bootstrap_sample = np.random.choice(array, size=n_size, replace=True) #resample(array, n_samples = n_size)
            medians[j] = np.median(bootstrap_sample)

        lower_bound = np.percentile(medians, 2.5)
        upper_bound = np.percentile(medians, 97.5)
        bootstrap_quantiles = (upper_bound - lower_bound) / 2
        
        original_median = np.median(array)
        bootstrap_median = np.median(medians)
        bootstrap_error = bootstrap_median - original_median
        
        bootstrap_std = np.std(medians)
        
        value = max(bootstrap_error, bootstrap_std, bootstrap_quantiles)

        
    else:
        value = ypatternOriginal
        
    return value


def ReadingSIM(subhaloFindID_99, SIM = 'TNG50-1', baseURL = 'http://www.tng-project.org/api/', 
               MAINPATH = '/home/abhner/TNG_Analyzes/SubhaloHistory/', FINALPATH = 'DFs', SIMPATH = 'TNG50'):
    
    PATH = MAINPATH + SIMPATH + '/' + FINALPATH
    
    warnings.filterwarnings("ignore")

    try:
        Main = extractDF('/IDMain/' + str(subhaloFindID_99), MERGER = True)
        Merger = extractDF('/IDMergers/'+str(subhaloFindID_99), MERGER = True)     
        Flag = Merger['Flag']
        return Main, Merger

    except Exception:
        print('ID: ', str(subhaloFindID_99))
        url = baseURL + SIM + '/snapshots/' + str(99) + '/subhalos/' + str(subhaloFindID_99)
        MergersDATA = ETNG.get(url+'/sublink/simple.json')
        Mergers = pd.DataFrame(MergersDATA['Mergers'], columns=['Snap', 'SubFindID'])
        Main = pd.DataFrame(MergersDATA['Main'], columns=['Snap', 'SubFindID'])
        
        Main['SubhaloMassType0'] = np.nan
        Main['SubhaloMassType1'] = np.nan
        Main['SubhaloMassType4'] = np.nan
        Main['SubhaloMassType5'] = np.nan
        Main['SubhaloMass'] = np.nan
        Main['SubhaloPos0'] = np.nan
        Main['SubhaloPos1'] = np.nan
        Main['SubhaloPos3'] = np.nan
        Main['SubhaloVel0'] = np.nan
        Main['SubhaloVel1'] = np.nan
        Main['SubhaloVel2'] = np.nan
        Main['SubhaloSpin0'] = np.nan
        Main['SubhaloSpin1'] = np.nan
        Main['SubhaloSpin2'] = np.nan
        Main['GasFrac'] = np.nan
        
        Mergers['SubhaloMassType0'] = np.nan
        Mergers['SubhaloMassType1'] = np.nan
        Mergers['SubhaloMassType4'] = np.nan
        Mergers['SubhaloMassType5'] = np.nan
        Mergers['SubhaloMass'] = np.nan
        Mergers['SubhaloPos0'] = np.nan
        Mergers['SubhaloPos1'] = np.nan
        Mergers['SubhaloPos3'] = np.nan
        Mergers['SubhaloVel0'] = np.nan
        Mergers['SubhaloVel1'] = np.nan
        Mergers['SubhaloVel2'] = np.nan
        Mergers['SubhaloSpin0'] = np.nan
        Mergers['SubhaloSpin1'] = np.nan
        Mergers['SubhaloSpin2'] = np.nan
        Mergers['GasFrac'] = np.nan
        Mergers['Flag'] = np.nan
        Mergers['Born_Ago'] = np.nan
    
        for snap in Main.Snap:
            if Main.SubFindID.loc[Main.Snap == snap].values[0] < 0:
                MGas = np.nan
                MDM = np.nan
                MStar = np.nan
                MBH = np.nan
                Mtot = np.nan
                GasFrac = np.nan
            else:
                ParametersMerger = ETNG.getsubhalo(Main.SubFindID.loc[Main.Snap == snap].values[0], snapnum = snap,
                                                   parameter=['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z' , 'spin_x', 'spin_y', 'spin_z' , 
                                                              'subhaloflag', 'prog_snap', 
                                                              'mass_gas','mass_dm','mass_stars','mass_bhs'])
                
                MGas = ParametersMerger[11]
                MDM = ParametersMerger[12]
                MStar = ParametersMerger[13]
                MBH = ParametersMerger[14]
                Mtot = MDM+MGas+MStar+MBH
                GasFrac = MGas / Mtot
                
            
            Main['SubhaloMassType0'].loc[Main.Snap == snap] = MGas
            Main['SubhaloMassType1'].loc[Main.Snap == snap] = MDM
            Main['SubhaloMassType4'].loc[Main.Snap == snap] = MStar
            Main['SubhaloMassType5'].loc[Main.Snap == snap] = MBH
            Main['SubhaloMass'].loc[Main.Snap == snap] = Mtot
            Main['SubhaloPos0'].loc[Main.Snap == snap]  = ParametersMerger[0]
            Main['SubhaloPos1'].loc[Main.Snap == snap]  = ParametersMerger[1]
            Main['SubhaloPos3'].loc[Main.Snap == snap]  = ParametersMerger[2]
            Main['SubhaloVel0'].loc[Main.Snap == snap]  = ParametersMerger[3]
            Main['SubhaloVel1'].loc[Main.Snap == snap]  = ParametersMerger[4]
            Main['SubhaloVel2'].loc[Main.Snap == snap]  = ParametersMerger[5]
            Main['SubhaloSpin0'].loc[Main.Snap == snap]  = ParametersMerger[6]
            Main['SubhaloSpin1'].loc[Main.Snap == snap]  = ParametersMerger[7]
            Main['SubhaloSpin2'].loc[Main.Snap == snap]  = ParametersMerger[8]
            Main['GasFrac'].loc[Main.Snap == snap] = GasFrac
            
        for j in range(len(Mergers.Snap)):
            ParametersMerger = ETNG.getsubhalo(Mergers.SubFindID.iloc[j], snapnum = Mergers.Snap.iloc[j], 
                                               parameter=['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z' , 'spin_x', 'spin_y', 'spin_z' , 
                                                          'subhaloflag', 'prog_snap', 
                                                          'mass_gas','mass_dm','mass_stars','mass_bhs'])
            Mergers['Flag'].iloc[j] = ParametersMerger[9]
            
            urlMerger = baseURL + SIM + '/snapshots/' + str(99) + '/subhalos/' + str(subhaloFindID_99)
            MergerHistory = ETNG.get(urlMerger+'/sublink/simple.json')
            
            Mergers['Born_Ago'].iloc[j] = Mergers.Snap.iloc[j] - MergerHistory['Main'][-1][0]
                    
            MGas = ParametersMerger[11]
            MDM = ParametersMerger[12]
            MStar = ParametersMerger[13]
            MBH = ParametersMerger[14]
            Mtot = MDM+MGas+MStar+MBH
            GasFrac = MGas / Mtot
            
            Mergers['SubhaloMassType0'].iloc[j] = MGas
            Mergers['SubhaloMassType1'].iloc[j] = MDM
            Mergers['SubhaloMassType4'].iloc[j] = MStar
            Mergers['SubhaloMassType5'].iloc[j] = MBH
            Mergers['SubhaloMass'].iloc[j] = Mtot
            Mergers['SubhaloPos0'].iloc[j] = ParametersMerger[0]
            Mergers['SubhaloPos1'].iloc[j] = ParametersMerger[1]
            Mergers['SubhaloPos3'].iloc[j] = ParametersMerger[2]
            Mergers['SubhaloVel0'].iloc[j] = ParametersMerger[3]
            Mergers['SubhaloVel1'].iloc[j] = ParametersMerger[4]
            Mergers['SubhaloVel2'].iloc[j] = ParametersMerger[5]
            Mergers['SubhaloSpin0'].iloc[j] = ParametersMerger[6]
            Mergers['SubhaloSpin1'].iloc[j] = ParametersMerger[7]
            Mergers['SubhaloSpin2'].iloc[j] = ParametersMerger[8]
            Mergers['GasFrac'].iloc[j] = GasFrac
            
            
            
        try:
            Main.to_csv(PATH+'/Analysis/Mergers/IDMain/'+str(subhaloFindID_99)+'.csv')
            Mergers.to_csv(PATH+'/Analysis/Mergers/IDMergers/'+str(subhaloFindID_99)+'.csv')
        except Exception:
            path = PATH
            for name in ['Analysis', 'Mergers', 'IDMain']:
                path = os.path.join(path, name)
                if not os.path.isdir(path):
                    os.mkdir(path)
            Main.to_csv(PATH+'/Analysis/Mergers/IDMain/'+str(subhaloFindID_99)+'.csv')
    
            path = PATH
            for name in ['Analysis', 'Mergers', 'IDMergers']:
                path = os.path.join(path, name)
                if not os.path.isdir(path):
                    os.mkdir(path)
            Mergers.to_csv(PATH+'/Analysis/Mergers/IDMergers/'+str(subhaloFindID_99)+'.csv')

              
    return Main, Mergers
    
def CountMergers(SIM = 'TNG50-1', baseURL = 'http://www.tng-project.org/api/',
                       MAINPATH = '/home/abhner/TNG_Analyzes/SubhaloHistory/', FINALPATH = 'DFs', SIMPATH = 'TNG50'):
            
    PATH = MAINPATH + SIMPATH + '/' + FINALPATH
    
    #SET MERGER LIST
    SubhaloHalfmassRadType4 = extractDF('SubhaloHalfmassRadType4')
    df = SubhaloHalfmassRadType4.copy()
    snaps =np.array([ int(i) for i in np.arange(100)])

    df[:] = np.nan
    df.Snap = np.flip(snaps)

    Ntot = df.copy()
    NMinor = df.copy()
    NMajor = df.copy()  
    NIntermediate = df.copy()
    NInverse = df.copy()

            
    Dry = df.copy()      
    Wet = df.copy()
        
    NDryMajorIntermediateMergers = df.copy()       
    NDryMinorMergers = df.copy()  
    NWetMajorIntermediateMergers = df.copy()   
    NwetMinorMergers = df.copy()
        
    DryDry = df.copy()
    WetWet = df.copy()
            
    DryWet = df.copy()
    WetDry = df.copy()
    
    ExSubhaloMassType0 = df.copy()
    ExSubhaloMassType4 = df.copy() 
    ExSubhaloMassType1 = df.copy()     
    ExSubhaloMassType5 = df.copy()

    with warnings.catch_warnings():
    
    
        for subhaloFindID_99 in df.keys().values[1:]:
            
            try:
                Main = extractDF('/IDMain/'+str(subhaloFindID_99), MERGER = True)
                Merger = extractDF('/IDMergers/'+str(subhaloFindID_99), MERGER = True)
                
            except Exception:
                #try:
                #    Main, Merger = ReadingSIM(subhaloFindID_99)
                #except Exception:
                continue

            GasMassEx = 0
            StarMassEx = 0
            DMMassEx = 0
            BHMassEx = 0
            
            countNintermediate = 0
            countNmajor = 0
            countNminor = 0
            countWet = 0
            countDry = 0
            countWetMajorIntermediate = 0
            countDryMajorIntermediate = 0
            countWetMinor = 0
            countDryMinor = 0
            countWetWet = 0
            countDryDry = 0
            countWetDry = 0
            countDryWet = 0
            countInverse = 0
            
            print('ID: ', subhaloFindID_99)
            for snap in snaps:
                
                MassMain = Main.SubhaloMass.loc[Main.Snap == snap].values * 1e10 / h
                GasMassMain = Main.SubhaloMassType0.loc[Main.Snap == snap].values  * 1e10 / h
                StarMassMain = Main.SubhaloMassType4.loc[Main.Snap == snap].values  * 1e10 / h


                MassMerger = Merger.SubhaloMass.loc[Merger.Snap == snap].values  * 1e10 / h
                GasMassMerger = Merger.SubhaloMassType0.loc[Merger.Snap == snap].values * 1e10 / h
                StarMassMerger = Merger.SubhaloMassType4.loc[Merger.Snap == snap].values  * 1e10 / h
                BHMassMerger = Merger.SubhaloMassType5.loc[Merger.Snap == snap].values  * 1e10 / h
                DMMassMerger = Merger.SubhaloMassType1.loc[Merger.Snap == snap].values  * 1e10 / h
                try:
                    FlagMerger = Merger['Flag'].loc[Merger.Snap == snap].values 
                except Exception:
                    break
                SnapAgoMerger = Merger['Born_Ago'].loc[Merger.Snap == snap].values 

                Ntotvalue = len(Merger.Snap.loc[Merger.Snap == snap].values)

                if Ntotvalue > 0:
                    
                   
                    for j, massmerger in enumerate(MassMerger):
                        
                        if FlagMerger[j] != 1:
                            continue
                        
                        if massmerger < 10**7.2:
                            continue
                        
                        if SnapAgoMerger[j] <= 2:
                            continue
                        
                        GasMassEx = GasMassEx + GasMassMerger[j]
                        StarMassEx = StarMassEx + StarMassMerger[j]
                        DMMassEx = DMMassEx + DMMassMerger[j]
                        BHMassEx = BHMassEx + BHMassMerger[j]
                        
                        if (StarMassMerger[j] / StarMassMain) > 1:
                            countInverse  += 1
                            mu = MassMain / massmerger
                            mustar = StarMassMain / StarMassMerger[j]
                            fGasMain = (GasMassMerger[j] /  massmerger)
                            fgasMerger = (GasMassMain / MassMain)
                        else:
                            mu = (massmerger / MassMain)
                            
                            mustar = StarMassMerger[j] / StarMassMain

                            fgasMerger = (GasMassMerger[j] /  massmerger)
                            fGasMain = (GasMassMain / MassMain)
                            
                        if  mustar >= 1./10. and mustar < 1./4.:
                            countNintermediate += 1
                        elif mustar >= 1./4.:
                            countNmajor += 1
                        elif mustar < 1./10.:
                            countNminor += 1
                      
       
                        if (GasMassMerger[j] + GasMassMain ) / (massmerger + MassMain) < 0.1:
                            countDry += 1
                        elif (GasMassMerger[j] + GasMassMain ) / (massmerger + MassMain) >= 0.1:
                            countWet += 1
                            
                        if (GasMassMerger[j] + GasMassMain ) / (massmerger + MassMain) < 0.1 and mustar >= 1./10.:
                            countDryMajorIntermediate += 1
                        elif (GasMassMerger[j] + GasMassMain ) / (massmerger + MassMain) >= 0.1 and mustar >= 1./10.:
                            countWetMajorIntermediate += 1
                            
                        if (GasMassMerger[j] + GasMassMain ) / (massmerger + MassMain) < 0.1 and mustar < 1./10 :
                            countDryMinor += 1
                        elif (GasMassMerger[j] + GasMassMain ) / (massmerger + MassMain) >= 0.1 and mustar < 1./10 :
                            countWetMinor += 1
                            
                        if fgasMerger < 0.1 and fGasMain < 0.1:
                            countDryDry += 1
                        elif fgasMerger >= 0.1 and fGasMain >= 0.1:
                            countWetWet += 1
                        elif fgasMerger >= 0.1 and fGasMain < 0.1:
                            countDryWet += 1
                        elif fgasMerger < 0.1 and fGasMain >= 0.1:
                            countWetDry += 1
                        
                        
                
                Ntot[str(subhaloFindID_99)].iloc[99 - snap] = countNminor + countNintermediate + countNmajor
                
                NMinor[str(subhaloFindID_99)].iloc[99 - snap] = countNminor
                
                NMajor[str(subhaloFindID_99)].iloc[99 - snap] = countNmajor
                    
                NIntermediate[str(subhaloFindID_99)].iloc[99 - snap] = countNintermediate
                        
                NInverse[str(subhaloFindID_99)].iloc[99 - snap] = countInverse
                Dry[str(subhaloFindID_99)].iloc[99 - snap] = countDry
                    
                Wet[str(subhaloFindID_99)].iloc[99 - snap] = countWet
                    
                NDryMajorIntermediateMergers[str(subhaloFindID_99)].iloc[99 - snap] = countDryMajorIntermediate
                    
                NDryMinorMergers[str(subhaloFindID_99)].iloc[99 - snap] = countDryMinor
                 
                NWetMajorIntermediateMergers[str(subhaloFindID_99)].iloc[99 - snap] = countWetMajorIntermediate
                    
                NwetMinorMergers[str(subhaloFindID_99)].iloc[99 - snap] = countWetMinor
                    
                DryDry[str(subhaloFindID_99)].iloc[99 - snap] = countDryDry
                    
                WetWet[str(subhaloFindID_99)].iloc[99 - snap] = countWetWet
                        
                DryWet[str(subhaloFindID_99)].iloc[99 - snap] = countDryWet
                    
                WetDry[str(subhaloFindID_99)].iloc[99 - snap] = countWetDry
                
                ExSubhaloMassType0[str(subhaloFindID_99)].iloc[99 - snap] = np.log10(  GasMassEx  )
                
                ExSubhaloMassType4[str(subhaloFindID_99)].iloc[99 - snap] = np.log10( StarMassEx )
                
                ExSubhaloMassType1[str(subhaloFindID_99)].iloc[99 - snap] = np.log10( DMMassEx)
                    
                ExSubhaloMassType5[str(subhaloFindID_99)].iloc[99 - snap] = np.log10( BHMassEx )
                
            

        Ntot.to_csv(PATH+'/Ntot.csv')
        NIntermediate.to_csv(PATH+'/NIntermediate.csv')
        NMajor.to_csv(PATH+'/NMajor.csv')
        NMinor.to_csv(PATH+'/NMinor.csv')
        Dry.to_csv(PATH+'/Dry.csv')
        Wet.to_csv(PATH+'/Wet.csv')      
        NDryMajorIntermediateMergers.to_csv(PATH+'/NDryMajorIntermediateMergers.csv')
        NDryMinorMergers.to_csv(PATH+'/NDryMinorMergers.csv')
        NWetMajorIntermediateMergers.to_csv(PATH+'/NWetMajorIntermediateMergers.csv')
        NwetMinorMergers.to_csv(PATH+'/NwetMinorMergers.csv')
        DryDry.to_csv(PATH+'/DryDry.csv')
        WetWet.to_csv(PATH+'/WetWet.csv')       
        DryWet.to_csv(PATH+'/DryWet.csv')
        WetDry.to_csv(PATH+'/WetDry.csv')
        ExSubhaloMassType0.to_csv(PATH+'/ExSubhaloMassType0.csv')
        ExSubhaloMassType4.to_csv(PATH+'/ExSubhaloMassType4.csv')
        ExSubhaloMassType1.to_csv(PATH+'/ExSubhaloMassType1.csv')
        ExSubhaloMassType5.to_csv(PATH+'/ExSubhaloMassType5.csv')
      

    return 


def MakeMergerAnalysis(SIM = 'TNG50-1', baseURL = 'http://www.tng-project.org/api/',
                       MAINPATH = '/home/abhner/TNG_Analyzes/SubhaloHistory/', FINALPATH = 'DFs', SIMPATH = 'TNG50'):
            
    PATH = MAINPATH + SIMPATH + '/' + FINALPATH
    
    #SET MERGER LIST
    

    SubhaloHalfmassRadType4 = extractDF('SubhaloHalfmassRadType4')

    with warnings.catch_warnings():
    
    
        for subhaloFindID_99 in SubhaloHalfmassRadType4.keys().values[1:]:
            
            try:
                df = extractDF(str(subhaloFindID_99)+'s', MERGER = True)
                
            except Exception:
                try:
                    Main = extractDF('IDMain/'+str(subhaloFindID_99), MERGER = True)
                    Merger = extractDF('IDMergers/'+str(subhaloFindID_99), MERGER = True)
                except Exception:
                    continue
                
                df = Merger[['Snap',  'SubFindID']].copy()
                
                df[['Type',  'K', 'U',  'E', 'StarMass', 'GasMass',
                    'AngMomOrb', 'AngMomMax', 'FinalAngMom', 'AngMomRatio', 'AngMomRatioPlusOrbit',
                    'theta', 'thetaOrbit', 'thetaSpin', 'thetaApproach', 'thetaFinalPlusOrbit', 
                    'mu', 'mustar', 'FracGasBaryon', 'FracGasTot']] = np.nan
                print('ID: ', subhaloFindID_99)

                for snap in np.arange(100):
                    
                    Ntotvalue = len(Merger.Snap.loc[Merger.Snap == snap].values)
                    
                    if Ntotvalue == 0:
                        continue
                    try:
                        MassMain = Main.SubhaloMass.loc[Main.Snap == snap].values[0] * 1e10 / h
                    except Exception:
                        continue
                    GasMassMain = Main.SubhaloMassType0.loc[Main.Snap == snap].values[0]  * 1e10 / h
                    StarMassMain = Main.SubhaloMassType4.loc[Main.Snap == snap].values[0]  * 1e10 / h
    

                    x = Main.SubhaloPos0.loc[Main.Snap == snap].values  / (1+dfTime.z.loc[dfTime.Snap == snap].values[0]) / h # kpc
                    y = Main.SubhaloPos1.loc[Main.Snap == snap].values / (1+dfTime.z.loc[dfTime.Snap == snap].values[0]) / h # kpc
                    z = Main.SubhaloPos3.loc[Main.Snap == snap].values / (1+dfTime.z.loc[dfTime.Snap == snap].values[0]) / h # kpc
                    vx = Main.SubhaloVel0.loc[Main.Snap == snap].values
                    vy = Main.SubhaloVel1.loc[Main.Snap == snap].values
                    vz = Main.SubhaloVel2.loc[Main.Snap == snap].values
                    jx = Main.SubhaloSpin0.loc[Main.Snap == snap].values   / h
                    jy = Main.SubhaloSpin1.loc[Main.Snap == snap].values   / h
                    jz = Main.SubhaloSpin2.loc[Main.Snap == snap].values   / h
    
                    V = np.array([vx, vy, vz]).T[0]
                    J = np.array([jx, jy, jz]).T[0]
    
                    IDsMerger = Merger.SubFindID.loc[Merger.Snap == snap].values  
                    MassMerger = Merger.SubhaloMass.loc[Merger.Snap == snap].values  * 1e10 / h
                    GasMassMerger = Merger.SubhaloMassType0.loc[Merger.Snap == snap].values * 1e10 / h
        
                    StarMassMerger = Merger.SubhaloMassType4.loc[Merger.Snap == snap].values  * 1e10 / h
                    
                    xMerger = Merger.SubhaloPos0.loc[Merger.Snap == snap].values / (1+dfTime.z.loc[dfTime.Snap == snap].values[0]) / h # kpc
                    yMerger = Merger.SubhaloPos1.loc[Merger.Snap == snap].values / (1+dfTime.z.loc[dfTime.Snap == snap].values[0]) / h # kpc
                    zMerger = Merger.SubhaloPos3.loc[Merger.Snap == snap].values / (1+dfTime.z.loc[dfTime.Snap == snap].values[0]) / h # kpc
                    vxMerger = Merger.SubhaloVel0.loc[Merger.Snap == snap].values
                    vyMerger = Merger.SubhaloVel1.loc[Merger.Snap == snap].values
                    vzMerger = Merger.SubhaloVel2.loc[Merger.Snap == snap].values 
                    jxMerger  = Merger.SubhaloSpin0.loc[Merger.Snap == snap].values  / h
                    jyMerger  = Merger.SubhaloSpin1.loc[Merger.Snap == snap].values  / h
                    jzMerger  = Merger.SubhaloSpin2.loc[Merger.Snap == snap].values / h
                    
                    try:
                        FlagMerger = Merger['Flag'].loc[Merger.Snap == snap].values 
                    except Exception:
                        FlagMerger = np.zeros(len(MassMerger))
                        FlagMerger[:] = 1
                    try:
                        SnapAgoMerger = Merger['Born_Ago'].loc[Merger.Snap == snap].values 
                    except Exception:
                        SnapAgoMerger = np.zeros(len(MassMerger))
                        SnapAgoMerger[:] =  3
                    for i, massmerger in enumerate(MassMerger):
                        
                        
                        if FlagMerger[i] != 1:
                            continue
                        
                        if massmerger < 10**7.2:
                            continue
                        
                        if SnapAgoMerger[i] <= 2:
                            continue
                        
                        IDj = IDsMerger[i]
                        
                        R = np.array([x, y, z]).T[0]
                        r = np.array([xMerger[i], yMerger[i], zMerger[i]])
                        V = np.array([vx, vy, vz]).T[0]
                        v = np.array([vxMerger[i], vyMerger[i], vzMerger[i]])
                        j = np.array([jxMerger[i], jyMerger[i], jzMerger[i]])
                        
                        mu = massmerger / MassMain
                        
                        
                        
                        if mu <=1 :
                            r12 =  r - R
                            V12 =  v - V
                            
                            r12, V12 = MATH.CartesiantToRotated(r12, V12, J)
                            j, JNew = MATH.CartesiantToRotated(j, J, J)

                            
                            angMom = np.cross(r12, V12)
                            angMom2 = j  + angMom
                            
                            FinalAng = (j * massmerger + JNew*MassMain) / (MassMain + massmerger)
                            
                            FinalAngPlusOrbit = (angMom2 * massmerger + JNew*MassMain ) / (MassMain + massmerger)

                      
                            param = np.dot(j / abs(np.linalg.norm(j)) , JNew / abs(np.linalg.norm(JNew)))
                            
                            thetaSpin = np.degrees(np.arccos(param))
                            
                            param = np.dot(np.cross(r12, V12) / abs(np.linalg.norm(np.cross(r12, V12))) , JNew / abs(np.linalg.norm(JNew)))

                            thetaOrbit = np.degrees(np.arccos(param))
                            
                            
                            
                            param = np.dot(FinalAng /  abs(np.linalg.norm(FinalAng)) , JNew / abs(np.linalg.norm(JNew)))

                            thetaFinal = np.degrees(np.arccos(param))
                            
                            param = np.dot(FinalAngPlusOrbit /  abs(np.linalg.norm(FinalAngPlusOrbit)) , JNew / abs(np.linalg.norm(JNew)))

                            thetaFinalPlusOrbit = np.degrees(np.arccos(param))
                            
                            K = (massmerger*np.linalg.norm(V12)**2.)  / 2. 
                            U = (-1)* G * massmerger * MassMain / np.linalg.norm(r12)            
                            
                            df.loc[df.SubFindID == int(IDj), 'K'] = np.log10(K)
                            df.loc[df.SubFindID == int(IDj), 'U'] = np.log10(U)
      
                            df.loc[df.SubFindID == int(IDj), 'E'] = np.log10(K - U)
                            
                            df.loc[df.SubFindID == int(IDj), 'AngMomMax'] = np.linalg.norm(angMom) / (np.linalg.norm(r12)*np.linalg.norm(V12))
                            df.loc[df.SubFindID == int(IDj), 'massmerger'] =  np.log10(massmerger)
                            df.loc[df.SubFindID == int(IDj), 'MassMain'] =  np.log10(MassMain)
                            
                            df.loc[df.SubFindID == int(IDj), 'AngMom'] =  np.linalg.norm(j)
                            df.loc[df.SubFindID == int(IDj), 'AngMomOrbit'] =  np.linalg.norm(angMom)
                            df.loc[df.SubFindID == int(IDj), 'AngMomPlusOrbit'] =  np.linalg.norm(angMom2)
                            df.loc[df.SubFindID == int(IDj), 'jPlusOrbit'] =  np.linalg.norm(angMom + j)
                            df.loc[df.SubFindID == int(IDj), 'AngMomPlusOrbitMax'] = np.linalg.norm(angMom2) / (np.linalg.norm(r12)*np.linalg.norm(V12))
                            df.loc[df.SubFindID == int(IDj), 'J'] =  np.linalg.norm(J)

                            
                            df.loc[df.SubFindID == int(IDj), 'AngMomOrb'] = np.linalg.norm(angMom)
            
                            df.loc[df.SubFindID == int(IDj), 'FinalAngMom'] = np.linalg.norm(FinalAng)
                            df.loc[df.SubFindID == int(IDj), 'FinalAngPlusOrbit'] = np.linalg.norm(FinalAngPlusOrbit)
                            
                            

                            df.loc[df.SubFindID == int(IDj), 'AngMomRatio'] = np.linalg.norm(angMom2* massmerger) / np.linalg.norm(JNew*MassMain )
                            df.loc[df.SubFindID == int(IDj), 'AngMomMax'] = np.linalg.norm(angMom) / (np.linalg.norm(r12)*np.linalg.norm(V12))
                            df.loc[df.SubFindID == int(IDj), 'AngMomRatioPlusOrbit'] = np.linalg.norm(FinalAngPlusOrbit * ( massmerger + MassMain )) / np.linalg.norm(JNew*MassMain)

                            df.loc[df.SubFindID == int(IDj), 'thetaFinal'] = thetaFinal
                            df.loc[df.SubFindID == int(IDj), 'thetaFinalPlusOrbit'] = thetaFinalPlusOrbit
                            
                            param = np.dot(r12 /  abs(np.linalg.norm(r12)) , V12 / abs(np.linalg.norm(V12)))
                            thetaApproach = np.degrees(np.arccos(param))
                            
                            df.loc[df.SubFindID == int(IDj), 'thetaApproach'] = thetaApproach

                            param = np.dot((angMom + j) /  abs(np.linalg.norm(angMom + j)) , JNew / abs(np.linalg.norm(JNew)))
                            thetaPlusOrbit = np.degrees(np.arccos(param))
                            df.loc[df.SubFindID == int(IDj), 'thetaPlusOrbit'] = thetaPlusOrbit

                            df.loc[df.SubFindID == int(IDj), 'thetaOrbit'] = thetaOrbit
                            df.loc[df.SubFindID == int(IDj), 'thetaSpin'] = thetaSpin

    
                            

                            df.loc[df.SubFindID == int(IDj), 'mu'] = mu
                            if StarMassMain > StarMassMerger[i]:
                                mustar = StarMassMerger[i] /StarMassMain
                                df.loc[df.SubFindID == int(IDj), 'mustar'] = StarMassMerger[i] /StarMassMain
                            else:
                                mustar = StarMassMain/StarMassMerger[i] 
                                df.loc[df.SubFindID == int(IDj), 'mustar'] = StarMassMain/StarMassMerger[i] 
                            
                            if mustar >= 1./10. and mustar < 1./4.:
                                df.loc[df.SubFindID == int(IDj), 'Type'] = 'Intermediate'
                            elif mustar >= 1./4.:
                                df.loc[df.SubFindID == int(IDj), 'Type'] = 'Major'
                            elif mustar < 1./10.:
                                df.loc[df.SubFindID == int(IDj), 'Type'] = 'Minor'
                                
                            df.loc[df.SubFindID == int(IDj), 'StarMass'] = np.log10(StarMassMerger[i])
                            
                            df.loc[df.SubFindID == int(IDj), 'GasMass'] = np.log10(GasMassMerger[i])

    
                            fgas =  (GasMassMerger[i] + GasMassMain ) / ( GasMassMerger[i] + GasMassMain + StarMassMain + StarMassMerger[i] )
                            df.loc[df.SubFindID == int(IDj), 'FracGasBaryon'] = fgas
                            
                            fgas =  (GasMassMerger[i] + GasMassMain ) / ( massmerger + MassMain)
                            df.loc[df.SubFindID == int(IDj), 'FracGasTot'] = fgas
                            
                            fstar =  (StarMassMerger[i] + StarMassMain ) / ( GasMassMerger[i] + GasMassMain + StarMassMain + StarMassMerger[i] )
                            df.loc[df.SubFindID == int(IDj), 'FracStarBaryon'] = fstar
                            
                            fstar =  (StarMassMerger[i] + StarMassMain ) / ( massmerger + MassMain)
                            df.loc[df.SubFindID == int(IDj), 'FracStarTot'] = fstar
                            
                        else:
                            r12 =  R - r
                            V12 =  V - v
                            
                            r12, V12 = MATH.CartesiantToRotated(r12, V12, j)
                            j, Jnew = MATH.CartesiantToRotated(j, J, j)
                            
                            mu = MassMain / massmerger
                            
                            
                            angMom = np.cross(r12, V12)
                            angMom2 = Jnew  + angMom


                            FinalAng = (j * massmerger + Jnew*MassMain) / (MassMain + massmerger)
                            
                            FinalAngPlusOrbit = (j * massmerger + angMom2*MassMain) / (MassMain + massmerger)

                           
                            param = np.dot(Jnew / abs(np.linalg.norm(Jnew)) , j / abs(np.linalg.norm(j)))

                            thetaSpin = np.degrees(np.arccos(param)) 
                            
                            param = np.dot(np.cross(r12, V12) / abs(np.linalg.norm(np.cross(r12, V12))) , j / abs(np.linalg.norm(j)))

                            thetaOrbit = np.degrees(np.arccos(param))

                            param = np.dot(FinalAng /  abs(np.linalg.norm(FinalAng)) , j / abs(np.linalg.norm(j)))

                            thetaFinal = np.degrees(np.arccos(param))
                            
                            param = np.dot(FinalAngPlusOrbit /  abs(np.linalg.norm(FinalAngPlusOrbit)) , j / abs(np.linalg.norm(j)))

                            thetaFinalPlusOrbit = np.degrees(np.arccos(param))
                            
                            K = (MassMain*np.linalg.norm(V)**2. )  / 2 
                            U = G * massmerger * MassMain / np.linalg.norm(r12)
          
                            
                            df.loc[df.SubFindID == int(IDj), 'K'] = np.log10(K)
                            df.loc[df.SubFindID == int(IDj), 'U'] = np.log10(U)
      
                            df.loc[df.SubFindID == int(IDj), 'E'] = np.log10(K - U)
                            
                            
                            df.loc[df.SubFindID == int(IDj), 'massmerger'] =  np.log10(massmerger)
                            df.loc[df.SubFindID == int(IDj), 'MassMain'] =  np.log10(MassMain)
                            
                            df.loc[df.SubFindID == int(IDj), 'AngMom'] =  np.linalg.norm(j)
                            df.loc[df.SubFindID == int(IDj), 'AngMomOrbit'] =  np.linalg.norm(angMom)
                            df.loc[df.SubFindID == int(IDj), 'jPlusOrbit'] =  np.linalg.norm(angMom + j)
                            df.loc[df.SubFindID == int(IDj), 'AngMomPlusOrbit'] =  np.linalg.norm(angMom2)
                            df.loc[df.SubFindID == int(IDj), 'AngMomPlusOrbitMax'] = np.linalg.norm(angMom2) / (np.linalg.norm(r12)*np.linalg.norm(V12))
                            df.loc[df.SubFindID == int(IDj), 'J'] =  np.linalg.norm(J)

                            
                            df.loc[df.SubFindID == int(IDj), 'AngMomOrb'] = np.linalg.norm(angMom)

            
                            df.loc[df.SubFindID == int(IDj), 'FinalAngMom'] = np.linalg.norm(FinalAng)
                            df.loc[df.SubFindID == int(IDj), 'FinalAngPlusOrbit'] = np.linalg.norm(FinalAngPlusOrbit) 
                            

                            df.loc[df.SubFindID == int(IDj), 'AngMomRatio'] = np.linalg.norm(angMom2*MassMain) / np.linalg.norm(j*massmerger) 
                            df.loc[df.SubFindID == int(IDj), 'AngMomMax'] =  np.linalg.norm(angMom) / (np.linalg.norm(r12)*np.linalg.norm(V12))
                            
                            param = np.dot(r12 /  abs(np.linalg.norm(r12)) , V12 / abs(np.linalg.norm(V12)))
                            thetaApproach = np.degrees(np.arccos(param))
                            df.loc[df.SubFindID == int(IDj), 'thetaApproach'] = thetaApproach
                            
                            param = np.dot((angMom + Jnew) /  abs(np.linalg.norm(angMom + Jnew)) , j / abs(np.linalg.norm(j)))
                            thetaPlusOrbit = np.degrees(np.arccos(param))
                            df.loc[df.SubFindID == int(IDj), 'thetaPlusOrbit'] = thetaPlusOrbit
                            
                            df.loc[df.SubFindID == int(IDj), 'AngMomRatioPlusOrbit'] = np.linalg.norm(FinalAngPlusOrbit*(MassMain + massmerger)) / np.linalg.norm(j*massmerger)


                            df.loc[df.SubFindID == int(IDj), 'thetaFinal'] = thetaFinal
                            df.loc[df.SubFindID == int(IDj), 'thetaFinalPlusOrbit'] = thetaFinalPlusOrbit
                            

                            df.loc[df.SubFindID == int(IDj), 'thetaOrbit'] = thetaOrbit
                            df.loc[df.SubFindID == int(IDj), 'thetaSpin'] = thetaSpin

    
                            df.loc[df.SubFindID == int(IDj), 'mu'] = mu
                   
                                
                            if StarMassMain > StarMassMerger[i]:
                                mustar = StarMassMerger[i] /StarMassMain
                                df.loc[df.SubFindID == int(IDj), 'mustar'] = StarMassMerger[i] /StarMassMain
                            else:
                                mustar = StarMassMain/StarMassMerger[i] 
                                df.loc[df.SubFindID == int(IDj), 'mustar'] = StarMassMain/StarMassMerger[i] 
                            
                            if mustar >= 1./10. and mustar < 1./4.:
                                df.loc[df.SubFindID == int(IDj), 'Type'] = 'Intermediate'
                            elif mustar >= 1./4.:
                                df.loc[df.SubFindID == int(IDj), 'Type'] = 'Major'
                            elif mustar < 1./10.:
                                df.loc[df.SubFindID == int(IDj), 'Type'] = 'Minor'
                                

                            df.loc[df.SubFindID == int(IDj), 'StarMass'] = np.log10(StarMassMerger[i])
                            
                            df.loc[df.SubFindID == int(IDj), 'GasMass'] = np.log10(GasMassMerger[i])
    
                            fgas =  (GasMassMerger[i] + GasMassMain ) / ( GasMassMerger[i] + GasMassMain + StarMassMain + StarMassMerger[i] )
                            df.loc[df.SubFindID == int(IDj), 'FracGasBaryon'] = fgas
                            
                            fgas =  (GasMassMerger[i] + GasMassMain ) / ( massmerger + MassMain)
                            df.loc[df.SubFindID == int(IDj), 'FracGasTot'] = fgas
                            
                            fstar =  (StarMassMerger[i] + StarMassMain ) / ( GasMassMerger[i] + GasMassMain + StarMassMain + StarMassMerger[i] )
                            df.loc[df.SubFindID == int(IDj), 'FracStarBaryon'] = fstar
                            
                            fstar =  (StarMassMerger[i] + StarMassMain ) / ( massmerger + MassMain)
                            df.loc[df.SubFindID == int(IDj), 'FracStarTot'] = fstar
                            
                            
                try:
                    df.to_csv(PATH+'/Analysis/Mergers/'+str(subhaloFindID_99)+'.csv')
                except Exception:
                    path = PATH
                    for name in ['Analysis', 'Mergers']:
                        path = os.path.join(path, name)
                        if not os.path.isdir(path):
                            os.mkdir(path)
                    df.to_csv(PATH+'/Analysis/Mergers/'+str(subhaloFindID_99)+'.csv')

    return 



def compare_Sample_key(key, populations, dfName = 'Sample', Name = 'Name', RankSums = False, Moodtest = False, KStest = False):

    for i, population in enumerate(populations):
        print(population[0], ' and ',population[1])
        Sample1 = extractPopulation(population[0], dfName = dfName, Name = Name)
        Sample2 = extractPopulation(population[1], dfName = dfName, Name = Name)

        try:

            if key == 'z_At_FinalEntry':
                MATH.TestPermutation(Sample1.loc[Sample1.z_At_FinalEntry >= 0, key].values,Sample2.loc[Sample2.z_At_FinalEntry >= 0, key].values, roundmedian = 3, RankSums = RankSums, Moodtest = Moodtest, KStest = KStest)
                
            else:
                Values1 = Sample1[key].values[~np.isnan(Sample1[key].values)]
                Values2 = Sample2[key].values[~np.isnan(Sample2[key].values)]
                
                MATH.TestPermutation(Values1,Values2, roundmedian = 3, RankSums = RankSums, Moodtest = Moodtest, KStest = KStest)
        except Exception:
            df1 = makeDF(population[0], key,  Name = Name)
            df2 = makeDF(population[1], key,  Name = Name)

            MATH.TestPermutation(df1.iloc[0].values,df2.iloc[0].values, roundmedian = 3) 

def compare_Parameters(Population, parameters, dfName = 'Sample', roundmedian = 3):

    print(Population)
    Sample = extractPopulation(Population, dfName = dfName)

    for i, parameter in enumerate(parameters):
        print(parameter[0], ' and ',parameter[1])
        MATH.TestPermutation(Sample[parameter[0]].values,Sample[parameter[1]].values, roundmedian = roundmedian)
       
            
def Ratio_Sample_key(key, populations, log10 = True, Name = 'Name'):

    print(key)
    for i, population in enumerate(populations):
        print(population[0], ' / ',population[1])
        Sample1 = extractPopulation(population[0], Name)
        Sample2 = extractPopulation(population[1], Name)

        if log10:
            print('Ratio: ', round(10**np.nanmedian(Sample1[key].values) / 10**np.nanmedian(Sample2[key].values), 4))
        else:
            print('Ratio: ', round(np.nanmedian(Sample1[key].values) / np.nanmedian(Sample2[key].values), 4))


def MakeDensityProfileMean(snap, ID, rmin, rmax, nbins, PartType = 'PartType4', velPlot = False, MomAng = False, Cond = 'None'):

    
    factor =  1. / h / (1+dfTime.z[int(99-snap)])
    scalefactorsqrt = np.sqrt(1. / (1+dfTime.z[int(99-snap)]))
    
    dFHalfStar = extractDF('SubhaloHalfmassRadType4')
    dFHalfGasRad = extractDF('SubhaloHalfmassRadType0')
    
    HalfRad = dFHalfStar[str(ID)].loc[dFHalfStar.Snap == snap].values[0]
    HalfGasRad = dFHalfGasRad[str(ID)].loc[dFHalfGasRad.Snap == snap].values[0]
    #print('ID: ', ID)

    if ID == 319738 and snap != 99:
        file = h5py.File('/home/abhner/Downloads/cutout_212960.hdf5', 'r')
    else:
        try:
            file = extractParticles(ID, snaps = [snap])[0]
        except Exception:
            return [0], [np.nan], [np.nan]

    try:
        pos = posStar = file['PartType4']['Coordinates'][:] * factor
        mass = massStar = file['PartType4']['Masses'][:] * 1e10 / h
        vel = velStar = file['PartType4']['Velocities'][:] * scalefactorsqrt
        
        JStar = massStar[:, np.newaxis]*(np.cross(posStar, velStar))

        Cen = np.array([MATH.weighted_median(posStar[:, 0], massStar), MATH.weighted_median(posStar[:, 1], massStar), MATH.weighted_median(posStar[:, 2], massStar)])
        Vmean = np.array([MATH.weighted_median(velStar[:, 0], massStar), MATH.weighted_median(velStar[:, 1], massStar), MATH.weighted_median(velStar[:, 2], massStar)])
        CenFind = True

    except Exception:
        posStar = velStar =  np.array([0, 0, 0])
        massStar = np.array([0]) 
        JStar = massStar[:, np.newaxis]*(np.cross(posStar, velStar))

        CenFind = False

    try:
        posGas = file['PartType0']['Coordinates'][:] * factor
        massGas = file['PartType0']['Masses'][:] * 1e10 / h
        velGas = file['PartType0']['Velocities'][:] * scalefactorsqrt  

        JGas = massGas[:, np.newaxis]*(np.cross(posGas, velGas))

        if not CenFind:
            pos = posGas
            mass = massGas
            vel = velGas
            Cen = np.array([MATH.weighted_median(posGas[:, 0], massGas), MATH.weighted_median(posGas[:, 1], massGas), MATH.weighted_median(posGas[:, 2], massGas)])
            Vmean = np.array([MATH.weighted_median(velGas[:, 0], massGas), MATH.weighted_median(velGas[:, 1], massGas), MATH.weighted_median(velGas[:, 2], massGas)])
            CenFind = True

    except Exception:
        posGas = velGas = np.array([0, 0, 0])
        massGas = np.array([0])
        JGas = massGas[:, np.newaxis]*(np.cross(posGas, velGas))


    try:
        posDM = file['PartType1']['Coordinates'][:] * factor
        massDM = file['Header'].attrs['MassTable'][1]*np.ones(len(file['PartType1']['Coordinates'])) * 1e10 / h
        velDM = file['PartType1']['Velocities'][:] * scalefactorsqrt

        JDM = massDM[:, np.newaxis]*(np.cross(posDM, velDM))

        if not CenFind:
            pos = posDM
            mass = massDM
            vel = velDM
            Cen = np.array([MATH.weighted_median(posDM[:, 0], massDM), MATH.weighted_median(posDM[:, 1], massDM), MATH.weighted_median(posDM[:, 2], massDM)])
            Vmean = np.array([MATH.weighted_median(velDM[:, 0], massDM), MATH.weighted_median(velDM[:, 1], massDM), MATH.weighted_median(velDM[:, 2], massDM)])
            CenFind = True

    except Exception:
        posDM = velDM = np.array([0, 0, 0])
        massDM = np.array([0])
        JDM = massDM[:, np.newaxis]*(np.cross(posDM, velDM))

        
    

    if not PartType in file.keys():
        print('Doesn\'t have ', PartType,' at snap: ', snap)
        return [0], [np.nan], [np.nan]

  
    pos = file[PartType]['Coordinates'][:]  / (1+dfTime.z.values[dfTime.Snap == snap]) / h #kpc
    vel = file[PartType]['Velocities'][:]   * scalefactorsqrt
    mass = file['Header'].attrs['MassTable'][1]*np.ones(len(file[PartType]['Coordinates'])) * 1e10 / h
    
    
  
    pos = pos - Cen
    vel = vel - Vmean
    # Compute distances from the origin
    r = np.linalg.norm(pos, axis=1)
    #R = np.linalg.norm(pos[:, 0:2], axis=1)
    #Hr = cos.Eofz(Omegam0, 1.-Omegam0, z) * h * 100 * r /1e3
    velrad = np.sum(pos*vel, axis = 1)/r # + Hr
    jang = np.sqrt(np.sum(np.cross(pos, vel)*np.cross(pos, vel), axis = 1))
    if PartType == 'PartType1':
        y = file['Header'].attrs['MassTable'][1]*np.ones(len(file['PartType1']['Coordinates'])) * 1e10 / h
    else:
        y =  file[PartType]['Masses'][:] * 1e10 / h
    
        
    if Cond == 'AngMom':
                
        JMean = (np.nansum(JDM, axis = 0) + np.nansum(JStar, axis = 0) + np.nansum(JGas, axis = 0)) / (np.nansum(massGas) + np.nansum(massStar) + np.nansum(massDM))
        
        j = np.cross(pos, vel)
        angelsTheta = np.array([])
        for jIndex in range(len(j)):
            param = np.dot(JMean /  abs(np.linalg.norm(JMean)) , j[jIndex] / abs(np.linalg.norm(j[jIndex])) )
            angelsTheta = np.append(angelsTheta, np.degrees(np.arccos(param)))
        Cond = (angelsTheta < 45)
        
    
        pos = pos[Cond]
        vel = vel[Cond]
        mass = mass[Cond]
        r = r[Cond]
        velrad = velrad[Cond]
        jang = jang[Cond]
        y = y[Cond]
  
    
    if len(r) == 0:
        return y, r, mass

    if np.min(r) <= rmin:
        minvalue = rmin
    else:
        minvalue = np.min(r)
    if np.max(r) >= rmax:
        maxvalue = rmax
    else:
        maxvalue = np.max(r)
    
    if minvalue == 0:
        minvalue = rmin
    if maxvalue == 0:
        maxvalue = rmax
    bin_edges = np.geomspace(minvalue, maxvalue, nbins +1)
    
    rad = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    yrad = np.zeros(nbins )
    massrad = np.zeros(nbins )

    for i in range(nbins ):
        in_bin = (r >= bin_edges[i]) & (r < bin_edges[i+1])
        mass_in_bin = mass[in_bin]
        y_in_bin = y[in_bin]
        velrad_in_bin = velrad[in_bin]
        jang_in_bin = jang[in_bin]
        volume = 4.0/3.0 * np.pi * (bin_edges[i+1]**3 - bin_edges[i]**3)
        if len(in_bin) < 50:
            yrad[i] = np.nan
            massrad[i] = np.nan

        else:
            if velPlot:
                yrad[i] = np.nansum(y_in_bin*velrad_in_bin) / np.nansum(y_in_bin)
            elif MomAng:
                yrad[i] = np.nansum(y_in_bin*jang_in_bin) / np.nansum(y_in_bin)

            else:
                yrad[i] = np.nansum(y_in_bin) /  volume
                massrad[i] = np.nansum(y_in_bin) 
        #volumeHalfRad = 4.0/3.0 * np.pi * (10**HalfRad)**3.
    #yRadHalf = yrad[rad < 10**HalfRad]
    y = yrad #/ yRadHalf[-1]
    r = rad #/ 10**HalfRad
    mass = massrad
    return y, r, mass


def MakeScatterParams(df_Param, df_Sample, ParamName, Mean_At_Time = [1, 2, 5, 8], AtBirth = True, AtEnd = True):
    
    for ID in df_Sample.SubfindID_99.values:
        try:
            Param = np.array([v for v in df_Param[str(ID)].values])
        except Exception:
            continue
        SnapAtBirth = df_Sample.loc[df_Sample.SubfindID_99 == ID, 'SnapBorn'].values[0]
        SnapAtBirth = int(SnapAtBirth)
        AgeBirth = dfTime.loc[dfTime.Snap ==  SnapAtBirth, 'Age'].values[0]
        if AtBirth:
            df_Sample.loc[df_Sample.SubfindID_99 == ID, ParamName+'_at_Birth'] = Param[99 - SnapAtBirth]
            snaps = np.array([])
            i = 1
            while len(snaps) < 3:
                snaps = dfTime.loc[dfTime.Age.between(AgeBirth, AgeBirth+i*0.05), 'Snap'].values
                i = i + 1
                
            snaps = np.array([int(99 - s) for s in snaps])
            df_Sample.loc[df_Sample.SubfindID_99 == ID, ParamName+'_at_Birth_Mean'] = np.nanmean(Param[99 - SnapAtBirth])

        if AtEnd:
            df_Sample.loc[df_Sample.SubfindID_99 == ID, ParamName+'_at_99'] = Param[99 - 99]
            
        for dt in Mean_At_Time:
            snaps = np.array([])
            i = 1
            while len(snaps) < 3:
                snaps = dfTime.loc[dfTime.Age.between(AgeBirth+dt-i*0.05, AgeBirth+dt+i*0.05), 'Snap'].values
                i = i + 1
                
            snaps = np.array([int(99 - s) for s in snaps])
            df_Sample.loc[df_Sample.SubfindID_99 == ID, ParamName+'_at_'+str(dt)] = np.nanmean(Param[snaps])

    return
    
def PhasingData(ID, dfStudy):
    r_over_R_Crit200FinalGroup = extractDF('r_over_R_Crit200_FirstGroup')
    time = np.flip(dfTime['Age'].values)
    Snap_At_FirstEntry  = dfStudy.loc[dfStudy['SubfindID_99'] == ID, 'Snap_At_FirstEntry'].values[0]
    
    checkIfPreprocessing = False #dfStudy.loc[dfStudy['SubfindID_99'] == ID, 'PreProcessingGalaxy'].values[0] == 'PreProcessingGalaxy'
    if checkIfPreprocessing or np.isnan(Snap_At_FirstEntry):
        return None
    else:
        rOverR200 = np.flip(np.array([v for v in r_over_R_Crit200FinalGroup[str(ID)].values]))
        rOverR200[:int(Snap_At_FirstEntry)] = np.nan
        pericenters, _ = find_peaks(-rOverR200)
        apocenters, _ = find_peaks(rOverR200)

        phase_array = np.zeros_like(time)
        
        #BEFORE ENTRY
        start = 0
        end = int(Snap_At_FirstEntry)

        normalized_phase_time = (time[start:end] - time[start:end][0]) / (time[start:end][-1] - time[start:end][0])
        phase_array[start:end] = (-0.5+1)*normalized_phase_time - 1 # -1 -> Snap = 0 , -0.5 -> Entry
        
        #AFTER ENTRY
        current_phase = -0.5
        for i, pericenter in enumerate(pericenters):
            start = end
            end = pericenter
            normalized_phase_time = (time[start:end] - time[start:end][0]) / (time[start:end][-1] - time[start:end][0])
            phase_array[start:end] = 0.5*normalized_phase_time+current_phase
            current_phase = current_phase + 0.5

            #Apocenter
            if i < len(apocenters):
                start = end
                end = apocenters[i]
                if apocenters[i] < start:
                    continue
                normalized_phase_time = (time[start:end] - time[start:end][0]) / (time[start:end][-1] - time[start:end][0])
                phase_array[start:end] = 0.5*normalized_phase_time+current_phase
                current_phase = current_phase + 0.5


        start = end
        normalized_phase_time = (time[start:] - time[start:][0]) / (time[start:][-1] - time[start:][0])
        phase_array[start:] = 0.5*normalized_phase_time+current_phase
        
        phases = phase_array 
    
    return phases
        
def MedianPhases(AllValues, Allphases, func = np.nanmedian, nboots = 100):
    FinalValue = np.array([])
    Finalphase = np.array([])
    FinalError = np.array([])
    try:
        #Phases = np.unique(np.append(np.append(np.linspace(-1, np.nanmax(Allphases), 100), np.arange(-1, np.nanmax(Allphases))),  np.arange(-1, np.nanmax(Allphases)) + 0.5))
        #print(AllValues[0])
        for phase in Allphases[0]:
            #print(phase, Allphases[Allphase == phase])
        
        
            Values = AllValues[Allphases == phase]
            Values = Values[~np.isnan(Values)]
            #print(phase, len(Values))
            if len(Values) < 5:
                Finalphase = np.append(Finalphase, np.nan)
                FinalValue = np.append(FinalValue, np.nan)
                FinalError = np.append(FinalError, np.nan)

            else:
                Finalphase = np.append(Finalphase, phase)
                FinalValue = np.append(FinalValue, np.nanmedian(Values))
                
                error = MATH.boostrap_func(
                                    Values, func=func, num_boots=nboots)
                FinalError = np.append(FinalError, error)

        Finalphase = Finalphase[~np.isnan(FinalValue)]
        FinalError = FinalError[~np.isnan(FinalValue)]
        FinalValue = FinalValue[~np.isnan(FinalValue)]

        return Finalphase, FinalValue, FinalError
    
    except Exception:
        Allphases = np.zeros(Allphases)
        Allphases[Allphases == 0 ] = np.nan
        return  Allphases,Allphases, Allphases


def makeMedianPhases(Study, param,  dfName = 'df_z0_Mstar_Range', Name = 'Name' , N = 1000):
    
    dfParam = extractDF(param)
    dfStudy = extractPopulation(Study, dfName = dfName, Name = Name)
    #try:
    X_ = np.arange(-1, 9)
    X_ = np.append(X_, X_+0.5)
    X_ = np.append(X_, np.linspace(-1, 9, N))
    X_ = np.unique(X_)
    N = len(X_)
    for i, ID in enumerate(dfStudy['SubfindID_99'].values):
        
        try:
            Values = np.flip(np.array([v for v in dfParam[str(ID)].values]))
        except Exception:
            Values = np.zeros(100)
            Values = np.nan
        checkIfPreprocessing = False #dfStudy.loc[dfStudy['SubfindID_99'] == ID, 'PreProcessingGalaxy'].values[0] == 'PreProcessingGalaxy'
        Snap_At_FirstEntry  = dfStudy.loc[dfStudy['SubfindID_99'] == ID, 'Snap_At_FirstEntry'].values[0]

        if checkIfPreprocessing or np.isnan(Snap_At_FirstEntry):
            Values = np.zeros(N)
            Values[Values == 0] = np.nan
            phases = Values
            if i == 0:
                AllValues = Values
                Allphases = phases
            else:
                AllValues = np.vstack((AllValues, Values))
                Allphases = np.vstack((Allphases, phases))
        
        else:

            phases = PhasingData(ID, dfStudy)

            phases = phases[(~np.isnan(Values)) ]
            if 'sSFR' in param:
                Values[Values < -14] = -14
            Values = Values[(~np.isnan(Values)) ]
            if len(Values) == 0:
                Values = np.zeros(N)
                Values[Values == 0] = np.nan
                phases = Values
                if i == 0:
                    AllValues = Values
                    Allphases = phases
                else:
                    AllValues = np.vstack((AllValues, Values))
                    Allphases = np.vstack((Allphases, phases))

            else:

                
                #X_ = np.append(X_, np.linspace(10.1, 30, 200))
                if len(Values[Values < 0]) > 0:
                    X_Y_Spline = interp1d(phases, Values,kind="linear",fill_value="extrapolate")
                    Values = X_Y_Spline(X_)

                else:
                    X_Y_Spline = interp1d(phases, np.log10(Values),kind="linear",fill_value="extrapolate")
                    Values = 10**X_Y_Spline(X_)
                Values[X_ > phases.max()] = np.nan
                phases = X_

                if i == 0:
                    AllValues = Values
                    Allphases = phases
                else:
                    AllValues = np.vstack((AllValues, Values))
                    Allphases = np.vstack((Allphases, phases))
    #print(AllValues, Allphases)
    Finalphase, FinalValue, FinalError = MedianPhases(AllValues, Allphases)
    
    return Finalphase, FinalValue, FinalError
    #except Exception:
    #    Finalphase = np.zeros(100)
    #    Finalphase[Finalphase == 0] = np.nan
    #    FinalValue = Finalphase
    #    FinalError = Finalphase

    #    return Finalphase, FinalValue, FinalError


def ParticleParameters(f, ID, snap):
    
    dFHalfStar = extractDF('SubhaloHalfmassRadType4')
    dFHalfGas = extractDF('SubhaloHalfmassRadType0')
    dFHalfDM = extractDF('SubhaloHalfmassRadType1')

    scalefactorsqrt = np.sqrt(1. / (1+dfTime.z[int(99-snap)]))

    try:
        IDsGas = f['PartType0']['ParticleIDs'][:]
        posGas = f['PartType0']['Coordinates'][:]  / (1+dfTime.z.values[dfTime.Snap == snap]) / h #kpc
        massGas = f['PartType0']['Masses'][:] * 1e10 / h
        velGas= f['PartType0']['Velocities'][:] * scalefactorsqrt
        
    except Exception:
        IDsGas = np.array([0])
        posGas = np.array([[0, 0, 0]])
        massGas = np.array([0])
        velGas=  np.array([[0, 0, 0]])
        
    try:
        IDsDM = f['PartType1']['ParticleIDs'][:]
        posDM = f['PartType1']['Coordinates'][:]  / (1+dfTime.z.values[dfTime.Snap == snap]) / h #kpc
        massDM = f['Header'].attrs['MassTable'][1]*np.ones(len(f['PartType1']['Coordinates'])) * 1e10 / h
        velDM =  f['PartType1']['Velocities'][:] * scalefactorsqrt

    except Exception:
        posDM = np.array([[0, 0, 0]])
        IDsDM = massDM = np.array([0])
        velDM =  np.array([[0, 0, 0]])

    try:
        IDsStar = f['PartType4']['ParticleIDs'][:]

        posStar = f['PartType4']['Coordinates'][:]  / (1+dfTime.z.values[dfTime.Snap == snap]) / h #kpc
        velStar= f['PartType4']['Velocities'][:] * scalefactorsqrt
        massStar = f['PartType4']['Masses'][:] * 1e10 / h
        ZStar = f['PartType4']['GFM_Metallicity'][:] /  0.0127

    except Exception:
        posStar = np.array([[0, 0, 0]])
        IDsStar = massStar = np.array([0])
        velStar =  np.array([[0, 0, 0]])
    
    
    if not (massStar[0] == 0 and len(massStar) == 1):
        Cen = np.array([MATH.weighted_median(posStar[:, 0], massStar), MATH.weighted_median(posStar[:, 1], massStar), MATH.weighted_median(posStar[:, 2], massStar)])
        VelBulk = np.array([MATH.weighted_median(velStar[:, 0], massStar), MATH.weighted_median(velStar[:, 1], massStar), MATH.weighted_median(velStar[:, 2], massStar)])
    elif not (massGas[0] == 0 and len(massGas) == 1):
        Cen = np.array([MATH.weighted_median(posGas[:, 0], massGas), MATH.weighted_median(posGas[:, 1], massGas), MATH.weighted_median(posGas[:, 2], massGas)])
        VelBulk = np.array([MATH.weighted_median(velGas[:, 0], massGas), MATH.weighted_median(velGas[:, 1], massGas), MATH.weighted_median(velGas[:, 2], massGas)])
    elif not (massDM[0] == 0 and len(massDM) == 1):
        Cen = np.array([MATH.weighted_median(posDM[:, 0], massDM), MATH.weighted_median(posDM[:, 1], massDM), MATH.weighted_median(posDM[:, 2], massDM)])
        VelBulk = np.array([MATH.weighted_median(velDM[:, 0], massDM), MATH.weighted_median(velDM[:, 1], massDM), MATH.weighted_median(velDM[:, 2], massDM)])


    posGas = posGas - Cen
    posDM = posDM - Cen
    posStar = posStar - Cen
    
    velStar = velStar - VelBulk
    velGas = velGas - VelBulk
    velDM = velDM - VelBulk
    

    rGas = np.linalg.norm(posGas, axis=1) / 10**dFHalfGas[str(ID)].values[99 - snap]
    rStar = np.linalg.norm(posStar, axis=1) / 10**dFHalfStar[str(ID)].values[99 - snap]
    rDM = np.linalg.norm(posDM, axis=1) / 10**dFHalfDM[str(ID)].values[99 - snap]
    
    
    # Compute the velocity dispersion for each axis
    sigma_x = np.std(velStar[:, 0])  # Dispersion along x
    sigma_y = np.std(velStar[:, 1])  # Dispersion along y
    sigma_z = np.std(velStar[:, 2])  # Dispersion along z
    
    # Compute the 3D velocity dispersion
    sigma3DStar = np.sqrt(sigma_x**2 + sigma_y**2 + sigma_z**2)

    # Compute the velocity dispersion for each axis
    sigma_x = np.std(velDM[:, 0])  # Dispersion along x
    sigma_y = np.std(velDM[:, 1])  # Dispersion along y
    sigma_z = np.std(velDM[:, 2])  # Dispersion along z
    
    # Compute the 3D velocity dispersion
    sigma3DDM = np.sqrt(sigma_x**2 + sigma_y**2 + sigma_z**2)

    # Compute the velocity dispersion for each axis
    sigma_x = np.std(velGas[:, 0])  # Dispersion along x
    sigma_y = np.std(velGas[:, 1])  # Dispersion along y
    sigma_z = np.std(velGas[:, 2])  # Dispersion along z
    
    # Compute the 3D velocity dispersion
    sigma3DGas = np.sqrt(sigma_x**2 + sigma_y**2 + sigma_z**2)

    
    velRadStar = (np.sum(posStar*velStar, axis = 1)/rStar) / sigma3DStar
    velRadGas = (np.sum(posGas*velGas, axis = 1)/rGas)  / sigma3DGas
    velRadDM = (np.sum(posDM*velDM, axis = 1)/rDM)  / sigma3DDM
    

    JStar = massStar*np.linalg.norm(np.cross(posStar, velStar), axis = 1)
    JGas = massGas*np.linalg.norm(np.cross(posGas, velGas), axis = 1)
    JDM = massDM*np.linalg.norm(np.cross(posDM, velDM), axis = 1)
    
    Pos = {'posGas': posGas,  
           'posStar': posStar,  
           'posDM': posDM}
    
    
    Vel = {'velGas': velGas,  
           'velStar': velStar,  
           'velDM': velDM}
    
    
    dfGas = pd.DataFrame(np.array([IDsGas, massGas, rGas, velRadGas, JGas]).T, columns = ['ParticleID', 'mass',  'RadNormalized', 'VelRadOverSigma', 'J'])
    dfStar= pd.DataFrame(np.array([IDsStar, massStar,  rStar, velRadStar, JStar]).T, columns = ['ParticleID', 'mass', 'RadNormalized', 'VelRadOverSigma', 'J'])
    dfDM = pd.DataFrame(np.array([IDsDM, massDM,  rDM, velRadDM, JDM]).T, columns = ['ParticleID', 'mass', 'RadNormalized', 'VelRadOverSigma', 'J'])
    
    return dfGas, dfStar, dfDM, Pos, Vel


def get_evolution_accreted_particles(Params, IDs, dfSample, rterm=2, 
                      Snaps=[4, 8, 17, 33, 50, 81, 99],
                      Update=False, NearBirth=False,
                      PATH=os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory',  
                      SAVEFILE='DFs', SIM='TNG50', NSim='-1', verbose=0):
    '''
    Make dataframe for the evolution of given parameters and subhalos using particle data
    
    Parameters
    ----------
    Params : list of str
        Parameters to compute evolution for
    IDs : list of int  
        Array with the subhalos IDs to analyze
    dfSample : pandas.DataFrame
        Sample dataframe with subhalo information
    rterm : float, optional
        Radial term multiplier for filtering
    Snaps : list of int, optional
        Snapshots to analyze
    Update : bool, optional  
        Whether to update existing values in dataframes
    NearBirth : bool, optional
        Whether to focus on snapshots near subhalo birth
    PATH : str, optional
        Base path for saving results
    SAVEFILE : str, optional
        Subdirectory for saving files  
    SIM : str, optional
        Simulation name
    NSim : str, optional
        Simulation number
    verbose : bool, optional
        Whether to print detailed progress information
    
    Returns
    -------
    None
        Saves results to CSV files in the specified directory
    
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    # Load required subhalo data
    SubfindID = extractDF('SubfindID')
    
    # Initialize dataframes for each parameter
    DFs = []
    dfTime.a = 1 / (1 + dfTime.z)  # Convert redshift to scale factor

    if verbose == 0:
        print(f"Initializing analysis for {len(Params)} parameters and {len(IDs)} subhalos")
        print(f"Parameters: {Params}")
        print(f"Snapshots: {Snaps}")
    
    # Create or load dataframes for each parameter
    for Param in Params:
        if verbose > 0:
            print(f'Initializing dataframe for parameter: {Param}')
            
        try:
            # Try to load existing dataframe
            df = extractDF(Param)
            
            # Add missing IDs as new columns with NaN values
            for ID in IDs:
                if str(ID) not in df.columns:
                    if verbose:
                        print(f'  Adding new subhalo ID: {ID}')
                    df[str(ID)] = np.nan
                    
        except Exception:
            # Create new dataframe if loading fails
            if verbose > 0:
                print(f'  Creating new dataframe for {Param}')
            snaps = np.arange(100)
            df = pd.DataFrame(data=np.zeros((100, len(IDs) + 1), dtype=object), 
                            columns=np.append('Snap', IDs))
            df[:] = np.nan
            df.Snap = np.flip(snaps)   
            
        DFs.append(df)
    
    Snaps_Original = Snaps.copy()  # Keep original snapshot list
    
    # Process each subhalo ID
    for j, ID in enumerate(IDs):
        if verbose == 0:
            print(f'\n{"="*50}')
            print(f'Processing subhalo {ID} ({j+1}/{len(IDs)})')
            print(f'{"="*50}')
        
        Snaps = Snaps_Original.copy()  # Reset to original snapshots
        
        # Determine which parameters need updating for this ID
        ParamUpdate = []
        for l, Param in enumerate(Params):
            testNan = np.array([v for v in DFs[l][str(ID)].values])
            valid_values = testNan[~np.isnan(testNan)]
            if len(valid_values) <= len(Snaps) or Update:
                ParamUpdate.append(Param)
        
        if  verbose > 0:
            print(f'Parameters to update: {ParamUpdate}')
        
        # Skip if no parameters need updating and Update flag is False
        if len(ParamUpdate) == 0 and not Update:
            if verbose > 0:
                print(f'No parameters need updating for ID {ID}, skipping...')
            continue
        
        # Adjust snapshots if focusing on birth period
        if NearBirth:
            if verbose > 0:
                print('Selecting snapshots near subhalo birth...')
            IDsSnaps = np.array([subID for subID in SubfindID[str(ID)]])
            argBirth = np.argwhere(~np.isnan(IDsSnaps)).T[0][-1]
            birth_snap = int(99 - argBirth)
            
            if verbose > 0:
                print(f'Birth snapshot: {birth_snap}')
            
            # Find snapshots at different age intervals after birth
            birth_age = dfTime.Age.loc[dfTime.Snap == birth_snap].values[0]
            
            SnapsAt0 = dfTime.Snap.loc[
                (dfTime.Age - birth_age < 0.15) & 
                (dfTime.Age - birth_age > 0) & 
                (dfTime.Snap > birth_snap)
            ].values
            
            SnapsAt1 = dfTime.Snap.loc[
                (dfTime.Age - birth_age < 1.1) & 
                (dfTime.Age - birth_age > 0.9) & 
                (dfTime.Snap > birth_snap)
            ].values
            
            SnapsAt2 = dfTime.Snap.loc[
                 (dfTime.Age - birth_age < 2.1) & 
                 (dfTime.Age - birth_age > 1.8) & 
                 (dfTime.Snap > birth_snap)
            ].values
            
            # SnapsAt3 = dfTime.Snap.loc[
            #     (dfTime.Age - birth_age < 3.50) & 
            #     (dfTime.Age - birth_age > 2.25) & 
            #     (dfTime.Snap > birth_snap)
            # ].values

            Snaps = np.concatenate([[birth_snap], SnapsAt0, SnapsAt1, SnapsAt2])#, SnapsAt3])
            Snaps = np.unique(Snaps)
            
            if verbose > 0:
                print(f'Selected {len(Snaps)} snapshots near birth: {Snaps}')

        # Process each snapshot
        for snap_idx, snap in enumerate(Snaps):
            if verbose == 0:
                print(f'\n  Processing snapshot {snap} ({snap_idx+1}/{len(Snaps)})')
            
            # Skip if not updating and value already exists
            if not Update and any(~np.isnan(DFs[l][str(ID)].values[99 - snap]) for l in range(len(Params))):
                if verbose > 0:
                    print(f'    Value exists and Update=False, skipping...')
                continue

            # Load particle data
            try:
                if verbose > 0:
                    print(f'    Loading particle data...')
                f = extractParticles(ID, snaps=[snap])[0]
            except Exception as e:
                print(f'    Failed to get particles for ID {ID}, snap {snap}: {e}')
                # Mark all parameters as failed for this snapshot
                for l in range(len(Params)):
                    DFs[l].loc[99 - snap, str(ID)] = -np.inf
                continue

            # Extract particle parameters
            if verbose > 0:
                print(f'    Extracting particle parameters...')
            dfGas, dfStar, dfDM , Pos_dic, Vel_dic = ParticleParameters(f, ID, snap)
            
            # Combine particle data for total calculations
            if verbose > 0:
                print(f'    Combining particle data...')
            IDsGas_Current = dfGas.ParticleID.values
            IDsDM_Current = dfDM.ParticleID.values
            IDsStar_Current = dfStar.ParticleID.values
            
            # Combine masses, positions, velocities, and radial distances
            Mass = np.concatenate([dfGas.mass.values, dfDM.mass.values, dfStar.mass.values])
            Pos = np.vstack([Pos_dic.get('posGas'), Pos_dic.get('posDM'), Pos_dic.get('posStar')])
            Vel = np.vstack([Vel_dic.get('velGas'), Vel_dic.get('velDM'), Vel_dic.get('velStar')])
            R = np.concatenate([dfGas.RadNormalized.values, dfDM.RadNormalized.values, dfStar.RadNormalized.values])
            
            # Calculate angular momentum vectors
            JVec = Mass[:, np.newaxis] * np.cross(Pos, Vel)
            
            # Process each parameter
            for l, Param in enumerate(Params):
                if Param not in ParamUpdate:
                    if verbose > 0:
                        print(f'    Skipping {Param} (no update needed)')
                    continue
                    
                if verbose > 0:
                    print(f'    Computing parameter: {Param}')
                
                # Save progress periodically
                if snap_idx % 5 == 0:  # Save every 5 snapshots
                    DFs[l].to_csv(PATH+'/'+SIM+'/'+SAVEFILE+'/'+Param+'.csv')
                
                # Initialize conditions based on particle type
                Cond = None
                CondStar = None
                CondJ = None
                
                # Set base conditions based on parameter type
                if 'Gas' in Param or 'SFR' in Param:
                    Cond = (dfGas.RadNormalized.values > 0)
                    if 'jAngle_Gas_Star' in Param:
                        CondStar = (dfStar.RadNormalized.values > 0)
                    if 'rotate' in Param:
                        CondJ = (R <= rterm)
                        
                elif 'Star' in Param and 'jAngle_Gas_Star' not in Param:
                    Cond = (dfStar.RadNormalized.values > 0)
                    if 'rotate' in Param:
                        CondJ = (R <= rterm)
                        
                elif 'DM' in Param:
                    Cond = (dfDM.RadNormalized.values > 0)
                    if 'rotate' in Param:
                        CondJ = (R <= rterm)
                else:
                    Cond = (R > 0)
                
                # Apply accretion criteria if parameter requires it
                if 'Accreted' in Param:
                    if verbose > 0:
                        print(f'      Applying accretion criteria...')
                    
                    if 'Gas' in Param or 'SFR' in Param:
                        IDsCurrent = IDsGas_Current
                        Infalling = dfGas.loc[
                            ((dfGas.RadNormalized.between(1, 2.5)) & (dfGas.RadNormalized - 3 >= dfGas.VelRadOverSigma)) |
                            ((dfGas.RadNormalized > 2.5) & (dfGas.VelRadOverSigma < -0.5))
                        ].copy()
                        
                    elif 'Star' in Param:
                        IDsCurrent = IDsStar_Current
                        Infalling = dfStar.loc[
                            ((dfStar.RadNormalized.between(1, 2.5)) & (dfStar.RadNormalized - 3 >= dfStar.VelRadOverSigma)) |
                            ((dfStar.RadNormalized > 2.5) & (dfStar.VelRadOverSigma < -0.5))
                        ].copy()
                        
                    elif 'DM' in Param:
                        IDsCurrent = IDsDM_Current
                        Infalling = dfDM.loc[
                            ((dfDM.RadNormalized.between(1, 2.5)) & (dfDM.RadNormalized - 3 >= dfDM.VelRadOverSigma)) |
                            ((dfDM.RadNormalized > 2.5) & (dfDM.VelRadOverSigma < -0.5))
                        ].copy()
                    
                    # Update condition to include only accreting particles
                    infalling_ids = set(Infalling.ParticleID.values)
                    Cond = Cond & [ID_part in infalling_ids for ID_part in IDsCurrent]
                    
                    if verbose > 0:
                        print(f'      Found {len(infalling_ids)} accreting particles')
                
                # Apply ex-situ/in-situ criteria for star particles
                if ('ExSitu' in Param or 'InSitu' in Param) and 'Star' in Param:
                    try:
                        if verbose > 0:
                            print(f'      Computing formation history (ex-situ/in-situ)...')
                        
                        as_Birth = np.array([tBirth for tBirth in f['PartType4']['GFM_StellarFormationTime'][:]])
                        IDsPart = np.array([idNum for idNum in f['PartType4']['ParticleIDs'][:]])
                        Tag = np.empty(len(IDsPart))
                        
                        # Map formation times to snapshots
                        SFSnap = as_Birth.copy()
                        for aindex, a_Birth in enumerate(as_Birth):
                            epsilon = 0.1
                            while len(dfTime.loc[abs(dfTime.a - a_Birth) < epsilon, 'Snap'].values) > 2:
                                snapsBirth = dfTime.loc[abs(dfTime.a - a_Birth) < epsilon, 'Snap'].values
                                epsilon = epsilon / 2.
                                if len(dfTime.loc[abs(dfTime.a - a_Birth) < epsilon, 'Snap'].values) == 0:
                                    break
                            SFSnap[aindex] = int(snapsBirth[0])
                        
                        SFSnapUnique = np.unique(SFSnap)
                        
                        # Classify particles as in-situ or ex-situ
                        for snapUnique in SFSnapUnique:
                            snapUnique = int(snapUnique)
                            try:
                                f_at_SnapBreak = extractParticles(ID, snaps=[snapUnique])[0]
                                CondSnap = (SFSnap == snapUnique)
                                
                                IDsParticle_at_SnapBreak = f_at_SnapBreak['PartType4']['ParticleIDs'][:]
                                
                                for idNum in IDsPart[CondSnap]:
                                    if idNum in IDsParticle_at_SnapBreak:
                                        Tag[IDsPart == idNum] = -100  # In-situ
                                    else:
                                        Tag[IDsPart == idNum] = 100   # Ex-situ
                                        
                            except Exception:
                                Tag[CondSnap] = 100  # Default to ex-situ if loading fails
                                continue
                        
                        # Apply the appropriate tag filter
                        if 'ExSitu' in Param:
                            CondII = (Tag > 0)
                            Cond = Cond & CondII
                            if  verbose > 0:
                                print(f'      Selected {np.sum(CondII)} ex-situ particles')
                        elif 'InSitu' in Param:
                            CondII = (Tag < 0)
                            Cond = Cond & CondII
                            if verbose > 0:
                                print(f'      Selected {np.sum(CondII)} in-situ particles')
                                
                    except Exception as e:
                        print(f'      Error in formation history computation: {e}')
                        continue
                
                # Validate particle counts before computation
                if 'Gas' in Param:
                    valid_count = np.sum(Cond)
                    if valid_count < 10:
                        if verbose > 0:
                            print(f'      Insufficient gas particles ({valid_count}), skipping...')
                        DFs[l].loc[99 - snap, str(ID)] = -np.inf
                        continue
                        
                if 'Star' in Param:
                    if 'jAngle_Gas_Star' in Param:
                        valid_count = np.sum(CondStar)
                        if valid_count < 10:
                            if verbose > 0:
                                print(f'      Insufficient star particles ({valid_count}) for angle calculation, skipping...')
                            DFs[l].loc[99 - snap, str(ID)] = -np.inf
                            continue
                    else:
                        valid_count = np.sum(Cond)
                        if valid_count < 10:
                            if verbose > 0:
                                print(f'      Insufficient star particles ({valid_count}), skipping...')
                            DFs[l].loc[99 - snap, str(ID)] = -np.inf
                            continue
                            
                if 'DM' in Param:
                    valid_count = np.sum(Cond)
                    if valid_count < 10:
                        if verbose > 0:
                            print(f'      Insufficient DM particles ({valid_count}), skipping...')
                        DFs[l].loc[99 - snap, str(ID)] = -np.inf
                        continue
                
                # Apply rotation criteria (corotating/counterrotating)
                if 'rotate' in Param and CondJ is not None:
                    if verbose > 0:
                        print(f'      Applying rotation criteria...')
                    
                    JMean = np.nansum(JVec[CondJ], axis=0) / np.nansum(Mass[CondJ])
                    
                    if 'Gas' in Param:
                        
                        
                        j = np.cross(Pos_dic.get('posGas'), Vel_dic.get('velGas'))
                    elif 'Star' in Param:
                        j = np.cross(Pos_dic.get('posStar'), Vel_dic.get('velStar'))
                    elif 'DM' in Param:
                        j = np.cross(Pos_dic.get('posDM'), Vel_dic.get('velDM'))
                    
                    # Calculate angles between individual angular momenta and mean
                    j_norm = j / np.linalg.norm(j, axis=1, keepdims=True)
                    JMean_norm = JMean / np.linalg.norm(JMean)
                    dots = np.dot(j_norm, JMean_norm)
                    angelsTheta = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
                    
                    if 'Corotate' in Param:
                        Cond = Cond & (angelsTheta < 45)
                        if verbose > 0:
                            print(f'      Selected {np.sum(angelsTheta < 45)} corotating particles')
                    elif 'Counterrotate' in Param:
                        Cond = Cond & (angelsTheta > 135)
                        if verbose > 0:
                            print(f'      Selected {np.sum(angelsTheta > 135)} counterrotating particles')
                
                # Final validation before computation
                if 'Gas' in Param:
                    valid_count = np.sum(Cond)
                    if valid_count < 10:
                        if verbose > 0:
                            print(f'      Insufficient gas particles after filtering ({valid_count}), skipping...')
                        DFs[l].loc[99 - snap, str(ID)] = -np.inf
                        continue
                        
                if 'Star' in Param:
                    if 'jAngle_Gas_Star' in Param:
                        valid_count = np.sum(CondStar)
                        if valid_count < 10:
                            if verbose > 0:
                                print(f'      Insufficient star particles after filtering ({valid_count}), skipping...')
                            DFs[l].loc[99 - snap, str(ID)] = -np.inf
                            continue
                    else:
                        valid_count = np.sum(Cond)
                        if valid_count < 10:
                            if verbose > 0:
                                print(f'      Insufficient star particles after filtering ({valid_count}), skipping...')
                            DFs[l].loc[99 - snap, str(ID)] = -np.inf
                            continue
                            
                if 'DM' in Param:
                    valid_count = np.sum(Cond)
                    if valid_count < 10:
                        if verbose > 0:
                            print(f'      Insufficient DM particles after filtering ({valid_count}), skipping...')
                        DFs[l].loc[99 - snap, str(ID)] = -np.inf
                        continue
                
                # Compute the actual parameter value
                if verbose > 0:
                    print(f'      Computing value for {Param}...')
                
                try:
                    if 'Mass' in Param and 'SFMass' not in Param:
                        if 'Gas' in Param:
                            total_mass = np.nansum(dfGas.mass.values[Cond])
                            DFs[l].loc[99 - snap, str(ID)] = np.log10(total_mass) if total_mass > 0 else -np.inf
                        elif 'Star' in Param:
                            total_mass = np.nansum(dfStar.mass.values[Cond])
                            DFs[l].loc[99 - snap, str(ID)] = np.log10(total_mass) if total_mass > 0 else -np.inf
                        elif 'DM' in Param:
                            total_mass = np.nansum(dfDM.mass.values[Cond])
                            DFs[l].loc[99 - snap, str(ID)] = np.log10(total_mass) if total_mass > 0 else -np.inf
                        else:
                            total_mass = np.nansum(Mass[Cond])
                            DFs[l].loc[99 - snap, str(ID)] = np.log10(total_mass) if total_mass > 0 else -np.inf
                    
                    elif 'j' in Param and 'jAngle_Gas_Star' not in Param:
                        if 'Gas' in Param or 'SFR' in Param:
                            j_magnitudes = np.linalg.norm(np.cross(Pos_dic.get('posGas')[Cond], Vel_dic.get('velGas')[Cond]), axis=1)
                            weighted_j = np.nansum(dfGas.mass.values[Cond] * j_magnitudes)
                            total_mass = np.nansum(dfGas.mass.values[Cond])
                            DFs[l].loc[99 - snap, str(ID)] = weighted_j / total_mass if total_mass > 0 else -np.inf
                        elif 'Star' in Param:
                            j_magnitudes = np.linalg.norm(np.cross(Pos_dic.get('posStar')[Cond], Vel_dic.get('velStar')[Cond]), axis=1)
                            weighted_j = np.nansum(dfStar.mass.values[Cond] * j_magnitudes)
                            total_mass = np.nansum(dfStar.mass.values[Cond])
                            DFs[l].loc[99 - snap, str(ID)] = weighted_j / total_mass if total_mass > 0 else -np.inf
                        elif 'DM' in Param:
                            j_magnitudes = np.linalg.norm(np.cross(Pos_dic.get('posDM')[Cond], Vel_dic.get('velDM')[Cond]), axis=1)
                            weighted_j = np.nansum(dfDM.mass.values[Cond] * j_magnitudes)
                            total_mass = np.nansum(dfDM.mass.values[Cond])
                            DFs[l].loc[99 - snap, str(ID)] = weighted_j / total_mass if total_mass > 0 else -np.inf
                        else:
                            J_total = np.nansum(JVec[Cond], axis=0)
                            total_mass = np.nansum(Mass[Cond])
                            DFs[l].loc[99 - snap, str(ID)] = np.linalg.norm(J_total / total_mass) if total_mass > 0 else -np.inf
                    
                    elif 'jAngle_Gas_Star' in Param:
                        GasAngMom = np.nansum(dfGas.mass.values[Cond, np.newaxis] * np.cross(Pos_dic.get('posGas')[Cond], Vel_dic.get('velGas')[Cond]), axis=0)
                        GasAngMom /= np.nansum(dfGas.mass.values[Cond])
                        
                        StarAngMom = np.nansum(dfStar.mass.values[CondStar, np.newaxis] * np.cross(Pos_dic.get('posStar')[CondStar], Vel_dic.get('velStar')[CondStar]), axis=0)
                        StarAngMom /= np.nansum(dfStar.mass.values[CondStar])
                        
                        angle = MATH.angle_between_vectors(GasAngMom, StarAngMom)
                        DFs[l].loc[99 - snap, str(ID)] = angle
                    
                    if verbose > 0:
                        value = DFs[l].loc[99 - snap, str(ID)]
                        print(f'      Computed value: {value}')
                        
                except Exception as e:
                    print(f'      Error computing parameter {Param}: {e}')
                    DFs[l].loc[99 - snap, str(ID)] = -np.inf
                
                # Save after each parameter computation for this snapshot
                DFs[l].to_csv(PATH+'/'+SIM+'/'+SAVEFILE+'/'+Param+'.csv')
    
    if verbose > 0:
        print(f'\n{"="*50}')
        print("Analysis completed successfully!")
        print(f"{'='*50}")
    
    return



# ----------------


def safe_save_dataframe(df, path, lock):
    """Thread-safe dataframe saving"""
    with lock:
        df.to_csv(path)

def save_individual_id_results(ID, DFs_local, Params, PATH, SIM, SAVEFILE, lock, verbose=0):
    """
    Save results for individual ID to separate files
    """
    # Create directory for individual ID results if it doesn't exist
    individual_dir = f"{PATH}/{SIM}/{SAVEFILE}/individual_ids"
    os.makedirs(individual_dir, exist_ok=True)
    
    for l, Param in enumerate(Params):
        # Extract just the data for this ID
        id_data = DFs_local[l][[str(ID)]].copy()
        
        # Save individual ID data
        individual_file = f"{individual_dir}/{Param}_ID_{ID}.csv"
        safe_save_dataframe(id_data, individual_file, lock)
        
        if verbose > 1:
            print(f"Saved individual results for ID {ID}, parameter {Param}")
    
    # Also save a marker file to indicate this ID is complete
    marker_file = f"{individual_dir}/completed_ID_{ID}.txt"
    with open(marker_file, 'w') as f:
        f.write(f"Completed processing ID {ID} at {pd.Timestamp.now()}\n")

def load_individual_id_results(IDs, Params, PATH, SIM, SAVEFILE, verbose=0):
    """
    Load individual ID results and combine into main DataFrames
    """
    individual_dir = f"{PATH}/{SIM}/{SAVEFILE}/individual_ids"
    
    if not os.path.exists(individual_dir):
        if verbose > 0:
            print("No individual ID results directory found")
        return None
    
    # Initialize empty DataFrames structure
    combined_DFs = []
    for Param in Params:
        snaps = np.arange(100)
        df = pd.DataFrame(data=np.zeros((100, len(IDs) + 1), dtype=object), 
                        columns=np.append('Snap', IDs))
        df[:] = np.nan
        df.Snap = np.flip(snaps)
        combined_DFs.append(df)
    
    # Load data from individual files
    loaded_count = 0
    for ID in IDs:
        marker_file = f"{individual_dir}/completed_ID_{ID}.txt"
        if os.path.exists(marker_file):
            # This ID has been processed, load its data
            for l, Param in enumerate(Params):
                individual_file = f"{individual_dir}/{Param}_ID_{ID}.csv"
                if os.path.exists(individual_file):
                    try:
                        id_data = pd.read_csv(individual_file, index_col=0)
                        combined_DFs[l][str(ID)] = id_data[str(ID)]
                        loaded_count += 1
                    except Exception as e:
                        if verbose > 0:
                            print(f"Error loading {individual_file}: {e}")
    
    if verbose > 0:
        print(f"Loaded data for {loaded_count} ID-parameter combinations from individual files")
    
    return combined_DFs

def process_single_subhalo(args):
    """
    Process a single subhalo ID (worker function for parallel processing)
    """
    ID, shared_data = args
    
    # Extract shared data
    Params = shared_data['Params']
    Snaps_Original = shared_data['Snaps_Original']
    Update = shared_data['Update']
    NearBirth = shared_data['NearBirth']
    dfTime = shared_data['dfTime']
    rterm = shared_data['rterm']
    verbose = shared_data['verbose']
    PATH = shared_data['PATH']
    SIM = shared_data['SIM']
    SAVEFILE = shared_data['SAVEFILE']
    SubfindID = shared_data['SubfindID']
    DFs_template = shared_data['DFs_template']
    lock = shared_data['lock']
    
    # Create local copies of DataFrames for this process
    DFs_local = []
    for l in range(len(Params)):
        DFs_local.append(DFs_template[l].copy())
    
    updates = []  # Store all updates for this subhalo
    
    if verbose:
        print(f'Processing subhalo {ID}')
    
    # Check if this ID is already completed
    individual_dir = f"{PATH}/{SIM}/{SAVEFILE}/individual_ids"
    marker_file = f"{individual_dir}/completed_ID_{ID}.txt"
    if os.path.exists(marker_file) and not Update:
        if verbose > 0:
            print(f'ID {ID}: Already completed, skipping...')
        return ID, DFs_local, updates
    
    Snaps = Snaps_Original.copy()
    
    # Determine which parameters need updating for this ID
    ParamUpdate = []
    for l, Param in enumerate(Params):
        testNan = np.array([v for v in DFs_template[l][str(ID)].values])
        valid_values = testNan[~np.isnan(testNan)]
        if len(valid_values) <= len(Snaps) or Update:
            ParamUpdate.append(Param)
    
    if verbose > 0:
        print(f'ID {ID}: Parameters to update: {ParamUpdate}')
    
    # Skip if no parameters need updating and Update flag is False
    if len(ParamUpdate) == 0 and not Update:
        if verbose > 0:
            print(f'ID {ID}: No parameters need updating, skipping...')
        return ID, DFs_local, updates
    
    # Adjust snapshots if focusing on birth period
    if NearBirth:
        if verbose > 0:
            print(f'ID {ID}: Selecting snapshots near subhalo birth...')
        IDsSnaps = np.array([subID for subID in SubfindID[str(ID)]])
        argBirth = np.argwhere(~np.isnan(IDsSnaps)).T[0][-1]
        birth_snap = int(99 - argBirth)
        
        if verbose > 0:
            print(f'ID {ID}: Birth snapshot: {birth_snap}')
        
        # Find snapshots at different age intervals after birth
        birth_age = dfTime.Age.loc[dfTime.Snap == birth_snap].values[0]
        
        SnapsAt0 = dfTime.Snap.loc[
            (dfTime.Age - birth_age < 0.15) & 
            (dfTime.Age - birth_age > 0) & 
            (dfTime.Snap > birth_snap)
        ].values
        
        SnapsAt1 = dfTime.Snap.loc[
            (dfTime.Age - birth_age < 1.1) & 
            (dfTime.Age - birth_age > 0.9) & 
            (dfTime.Snap > birth_snap)
        ].values
        
        SnapsAt2 = dfTime.Snap.loc[
            (dfTime.Age - birth_age < 2.25) & 
            (dfTime.Age - birth_age > 1.75) & 
            (dfTime.Snap > birth_snap)
        ].values
        
        SnapsAt3 = dfTime.Snap.loc[
            (dfTime.Age - birth_age < 3.50) & 
            (dfTime.Age - birth_age > 2.25) & 
            (dfTime.Snap > birth_snap)
        ].values

        Snaps = np.concatenate([[birth_snap], SnapsAt0, SnapsAt1, SnapsAt2, SnapsAt3])
        Snaps = np.unique(Snaps)
        
        if verbose > 0:
            print(f'ID {ID}: Selected {len(Snaps)} snapshots near birth: {Snaps}')

    # Process each snapshot
    for snap_idx, snap in enumerate(Snaps):
        if verbose == 0:
            print(f'ID {ID}: Processing snapshot {snap} ({snap_idx+1}/{len(Snaps)})')
        
        # Skip if not updating and value already exists
        skip_snapshot = False
        if not Update:
            for l in range(len(Params)):
                if Param in ParamUpdate and not np.isnan(DFs_template[l].loc[99 - snap, str(ID)]):
                    skip_snapshot = True
                    break
            if skip_snapshot:
                if verbose:
                    print(f'ID {ID}: Value exists and Update=False, skipping snapshot {snap}...')
                continue

        # Load particle data
        try:
            if verbose > 0:
                print(f'ID {ID}: Loading particle data...')
            f = extractParticles(ID, snaps=[snap])[0]
        except Exception as e:
            print(f'ID {ID}: Failed to get particles for snap {snap}: {e}')
            # Mark all parameters as failed for this snapshot
            for l in range(len(Params)):
                if Params[l] in ParamUpdate:
                    DFs_local[l].loc[99 - snap, str(ID)] = -np.inf
                    updates.append((l, 99 - snap, str(ID), -np.inf))
            continue

        # Extract particle parameters
        if verbose > 0:
            print(f'ID {ID}: Extracting particle parameters...')
        dfGas, dfStar, dfDM, Pos_dic, Vel_dic = ParticleParameters(f, ID, snap)
        
        # Combine particle data for total calculations
        if verbose > 0:
            print(f'ID {ID}: Combining particle data...')
        IDsGas_Current = dfGas.ParticleID.values
        IDsDM_Current = dfDM.ParticleID.values
        IDsStar_Current = dfStar.ParticleID.values
        
        # Combine masses, positions, velocities, and radial distances
        Mass = np.concatenate([dfGas.mass.values, dfDM.mass.values, dfStar.mass.values])
        Pos = np.vstack([Pos_dic.get('posGas'), Pos_dic.get('posDM'), Pos_dic.get('posStar')])
        Vel = np.vstack([Vel_dic.get('velGas'), Vel_dic.get('velDM'), Vel_dic.get('velStar')])
        R = np.concatenate([dfGas.RadNormalized.values, dfDM.RadNormalized.values, dfStar.RadNormalized.values])
        
        # Calculate angular momentum vectors
        JVec = Mass[:, np.newaxis] * np.cross(Pos, Vel)
        
        # Process each parameter
        for l, Param in enumerate(Params):
            if Param not in ParamUpdate:
                if verbose > 0:
                    print(f'ID {ID}: Skipping {Param} (no update needed)')
                continue
                
            if verbose > 0:
                print(f'ID {ID}: Computing parameter: {Param}')
            
            # Initialize conditions based on particle type
            Cond = None
            CondStar = None
            CondJ = None
            
            # Set base conditions based on parameter type
            if 'Gas' in Param or 'SFR' in Param:
                Cond = (dfGas.RadNormalized.values > 0)
                if 'jAngle_Gas_Star' in Param:
                    CondStar = (dfStar.RadNormalized.values > 0)
                if 'rotate' in Param:
                    CondJ = (R <= rterm)
                    
            elif 'Star' in Param and 'jAngle_Gas_Star' not in Param:
                Cond = (dfStar.RadNormalized.values > 0)
                if 'rotate' in Param:
                    CondJ = (R <= rterm)
                    
            elif 'DM' in Param:
                Cond = (dfDM.RadNormalized.values > 0)
                if 'rotate' in Param:
                    CondJ = (R <= rterm)
            else:
                Cond = (R > 0)
            
            # Apply accretion criteria if parameter requires it
            if 'Accreted' in Param:
                if verbose > 0:
                    print(f'ID {ID}: Applying accretion criteria...')
                
                if 'Gas' in Param or 'SFR' in Param:
                    IDsCurrent = IDsGas_Current
                    Infalling = dfGas.loc[
                        ((dfGas.RadNormalized.between(1, 2.5)) & (dfGas.RadNormalized - 3 >= dfGas.VelRadOverSigma)) |
                        ((dfGas.RadNormalized > 2.5) & (dfGas.VelRadOverSigma < -0.5))
                    ].copy()
                    
                elif 'Star' in Param:
                    IDsCurrent = IDsStar_Current
                    Infalling = dfStar.loc[
                        ((dfStar.RadNormalized.between(1, 2.5)) & (dfStar.RadNormalized - 3 >= dfStar.VelRadOverSigma)) |
                        ((dfStar.RadNormalized > 2.5) & (dfStar.VelRadOverSigma < -0.5))
                    ].copy()
                    
                elif 'DM' in Param:
                    IDsCurrent = IDsDM_Current
                    Infalling = dfDM.loc[
                        ((dfDM.RadNormalized.between(1, 2.5)) & (dfDM.RadNormalized - 3 >= dfDM.VelRadOverSigma)) |
                        ((dfDM.RadNormalized > 2.5) & (dfDM.VelRadOverSigma < -0.5))
                    ].copy()
                
                # Update condition to include only accreting particles
                infalling_ids = set(Infalling.ParticleID.values)
                Cond = Cond & [ID_part in infalling_ids for ID_part in IDsCurrent]
                
                if verbose > 0:
                    print(f'ID {ID}: Found {len(infalling_ids)} accreting particles')
            
            # Apply ex-situ/in-situ criteria for star particles
            if ('ExSitu' in Param or 'InSitu' in Param) and 'Star' in Param:
                try:
                    if verbose > 0:
                        print(f'ID {ID}: Computing formation history (ex-situ/in-situ)...')
                    
                    as_Birth = np.array([tBirth for tBirth in f['PartType4']['GFM_StellarFormationTime'][:]])
                    IDsPart = np.array([idNum for idNum in f['PartType4']['ParticleIDs'][:]])
                    Tag = np.empty(len(IDsPart))
                    
                    # Map formation times to snapshots
                    SFSnap = as_Birth.copy()
                    for aindex, a_Birth in enumerate(as_Birth):
                        epsilon = 0.1
                        while len(dfTime.loc[abs(dfTime.a - a_Birth) < epsilon, 'Snap'].values) > 2:
                            snapsBirth = dfTime.loc[abs(dfTime.a - a_Birth) < epsilon, 'Snap'].values
                            epsilon = epsilon / 2.
                            if len(dfTime.loc[abs(dfTime.a - a_Birth) < epsilon, 'Snap'].values) == 0:
                                break
                        SFSnap[aindex] = int(snapsBirth[0])
                    
                    SFSnapUnique = np.unique(SFSnap)
                    
                    # Classify particles as in-situ or ex-situ
                    for snapUnique in SFSnapUnique:
                        snapUnique = int(snapUnique)
                        try:
                            f_at_SnapBreak = extractParticles(ID, snaps=[snapUnique])[0]
                            CondSnap = (SFSnap == snapUnique)
                            
                            IDsParticle_at_SnapBreak = f_at_SnapBreak['PartType4']['ParticleIDs'][:]
                            
                            for idNum in IDsPart[CondSnap]:
                                if idNum in IDsParticle_at_SnapBreak:
                                    Tag[IDsPart == idNum] = -100  # In-situ
                                else:
                                    Tag[IDsPart == idNum] = 100   # Ex-situ
                                    
                        except Exception:
                            Tag[CondSnap] = 100  # Default to ex-situ if loading fails
                            continue
                    
                    # Apply the appropriate tag filter
                    if 'ExSitu' in Param:
                        CondII = (Tag > 0)
                        Cond = Cond & CondII
                        if verbose > 0:
                            print(f'ID {ID}: Selected {np.sum(CondII)} ex-situ particles')
                    elif 'InSitu' in Param:
                        CondII = (Tag < 0)
                        Cond = Cond & CondII
                        if verbose > 0:
                            print(f'ID {ID}: Selected {np.sum(CondII)} in-situ particles')
                            
                except Exception as e:
                    print(f'ID {ID}: Error in formation history computation: {e}')
                    continue
            
            # Validate particle counts before computation
            if 'Gas' in Param:
                valid_count = np.sum(Cond)
                if valid_count < 10:
                    if verbose > 0:
                        print(f'ID {ID}: Insufficient gas particles ({valid_count}), skipping...')
                    DFs_local[l].loc[99 - snap, str(ID)] = -np.inf
                    updates.append((l, 99 - snap, str(ID), -np.inf))
                    continue
                    
            if 'Star' in Param:
                if 'jAngle_Gas_Star' in Param:
                    valid_count = np.sum(CondStar)
                    if valid_count < 10:
                        if verbose > 0:
                            print(f'ID {ID}: Insufficient star particles ({valid_count}) for angle calculation, skipping...')
                        DFs_local[l].loc[99 - snap, str(ID)] = -np.inf
                        updates.append((l, 99 - snap, str(ID), -np.inf))
                        continue
                else:
                    valid_count = np.sum(Cond)
                    if valid_count < 10:
                        if verbose > 0:
                            print(f'ID {ID}: Insufficient star particles ({valid_count}), skipping...')
                        DFs_local[l].loc[99 - snap, str(ID)] = -np.inf
                        updates.append((l, 99 - snap, str(ID), -np.inf))
                        continue
                        
            if 'DM' in Param:
                valid_count = np.sum(Cond)
                if valid_count < 10:
                    if verbose > 0:
                        print(f'ID {ID}: Insufficient DM particles ({valid_count}), skipping...')
                    DFs_local[l].loc[99 - snap, str(ID)] = -np.inf
                    updates.append((l, 99 - snap, str(ID), -np.inf))
                    continue
            
            # Apply rotation criteria (corotating/counterrotating)
            if 'rotate' in Param and CondJ is not None:
                if verbose > 0:
                    print(f'ID {ID}: Applying rotation criteria...')
                
                JMean = np.nansum(JVec[CondJ], axis=0) / np.nansum(Mass[CondJ])
                
                if 'Gas' in Param:
                    j = np.cross(Pos_dic.get('posGas'), Vel_dic.get('velGas'))
                elif 'Star' in Param:
                    j = np.cross(Pos_dic.get('posStar'), Vel_dic.get('velStar'))
                elif 'DM' in Param:
                    j = np.cross(Pos_dic.get('posDM'), Vel_dic.get('velDM'))
                
                # Calculate angles between individual angular momenta and mean
                j_norm = j / np.linalg.norm(j, axis=1, keepdims=True)
                JMean_norm = JMean / np.linalg.norm(JMean)
                dots = np.dot(j_norm, JMean_norm)
                angelsTheta = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
                
                if 'Corotate' in Param:
                    Cond = Cond & (angelsTheta < 45)
                    if verbose > 0:
                        print(f'ID {ID}: Selected {np.sum(angelsTheta < 45)} corotating particles')
                elif 'Counterrotate' in Param:
                    Cond = Cond & (angelsTheta > 135)
                    if verbose > 0:
                        print(f'ID {ID}: Selected {np.sum(angelsTheta > 135)} counterrotating particles')
            
            # Final validation before computation
            if 'Gas' in Param:
                valid_count = np.sum(Cond)
                if valid_count < 10:
                    if verbose > 0:
                        print(f'ID {ID}: Insufficient gas particles after filtering ({valid_count}), skipping...')
                    DFs_local[l].loc[99 - snap, str(ID)] = -np.inf
                    updates.append((l, 99 - snap, str(ID), -np.inf))
                    continue
                    
            if 'Star' in Param:
                if 'jAngle_Gas_Star' in Param:
                    valid_count = np.sum(CondStar)
                    if valid_count < 10:
                        if verbose > 0:
                            print(f'ID {ID}: Insufficient star particles after filtering ({valid_count}), skipping...')
                        DFs_local[l].loc[99 - snap, str(ID)] = -np.inf
                        updates.append((l, 99 - snap, str(ID), -np.inf))
                        continue
                else:
                    valid_count = np.sum(Cond)
                    if valid_count < 10:
                        if verbose > 0:
                            print(f'ID {ID}: Insufficient star particles after filtering ({valid_count}), skipping...')
                        DFs_local[l].loc[99 - snap, str(ID)] = -np.inf
                        updates.append((l, 99 - snap, str(ID), -np.inf))
                        continue
                        
            if 'DM' in Param:
                valid_count = np.sum(Cond)
                if valid_count < 10:
                    if verbose > 0:
                        print(f'ID {ID}: Insufficient DM particles after filtering ({valid_count}), skipping...')
                    DFs_local[l].loc[99 - snap, str(ID)] = -np.inf
                    updates.append((l, 99 - snap, str(ID), -np.inf))
                    continue
            
            # Compute the actual parameter value
            if verbose > 0:
                print(f'ID {ID}: Computing value for {Param}...')
            
            try:
                if 'Mass' in Param and 'SFMass' not in Param:
                    if 'Gas' in Param:
                        total_mass = np.nansum(dfGas.mass.values[Cond])
                        value = np.log10(total_mass) if total_mass > 0 else -np.inf
                    elif 'Star' in Param:
                        total_mass = np.nansum(dfStar.mass.values[Cond])
                        value = np.log10(total_mass) if total_mass > 0 else -np.inf
                    elif 'DM' in Param:
                        total_mass = np.nansum(dfDM.mass.values[Cond])
                        value = np.log10(total_mass) if total_mass > 0 else -np.inf
                    else:
                        total_mass = np.nansum(Mass[Cond])
                        value = np.log10(total_mass) if total_mass > 0 else -np.inf
                
                elif 'j' in Param and 'jAngle_Gas_Star' not in Param:
                    if 'Gas' in Param or 'SFR' in Param:
                        j_magnitudes = np.linalg.norm(np.cross(Pos_dic.get('posGas')[Cond], Vel_dic.get('velGas')[Cond]), axis=1)
                        weighted_j = np.nansum(dfGas.mass.values[Cond] * j_magnitudes)
                        total_mass = np.nansum(dfGas.mass.values[Cond])
                        value = weighted_j / total_mass if total_mass > 0 else -np.inf
                    elif 'Star' in Param:
                        j_magnitudes = np.linalg.norm(np.cross(Pos_dic.get('posStar')[Cond], Vel_dic.get('velStar')[Cond]), axis=1)
                        weighted_j = np.nansum(dfStar.mass.values[Cond] * j_magnitudes)
                        total_mass = np.nansum(dfStar.mass.values[Cond])
                        value = weighted_j / total_mass if total_mass > 0 else -np.inf
                    elif 'DM' in Param:
                        j_magnitudes = np.linalg.norm(np.cross(Pos_dic.get('posDM')[Cond], Vel_dic.get('velDM')[Cond]), axis=1)
                        weighted_j = np.nansum(dfDM.mass.values[Cond] * j_magnitudes)
                        total_mass = np.nansum(dfDM.mass.values[Cond])
                        value = weighted_j / total_mass if total_mass > 0 else -np.inf
                    else:
                        J_total = np.nansum(JVec[Cond], axis=0)
                        total_mass = np.nansum(Mass[Cond])
                        value = np.linalg.norm(J_total / total_mass) if total_mass > 0 else -np.inf
                
                elif 'jAngle_Gas_Star' in Param:
                    GasAngMom = np.nansum(dfGas.mass.values[Cond, np.newaxis] * np.cross(Pos_dic.get('posGas')[Cond], Vel_dic.get('velGas')[Cond]), axis=0)
                    GasAngMom /= np.nansum(dfGas.mass.values[Cond])
                    
                    StarAngMom = np.nansum(dfStar.mass.values[CondStar, np.newaxis] * np.cross(Pos_dic.get('posStar')[CondStar], Vel_dic.get('velStar')[CondStar]), axis=0)
                    StarAngMom /= np.nansum(dfStar.mass.values[CondStar])
                    
                    # Simple angle calculation between vectors
                    dot_product = np.dot(GasAngMom, StarAngMom)
                    norm_product = np.linalg.norm(GasAngMom) * np.linalg.norm(StarAngMom)
                    angle = math.degrees(math.acos(np.clip(dot_product / norm_product, -1.0, 1.0)))
                    value = angle
                
                if verbose > 0:
                    print(f'ID {ID}: Computed value: {value}')
                    
                # Update local DataFrame and record the update
                DFs_local[l].loc[99 - snap, str(ID)] = value
                updates.append((l, 99 - snap, str(ID), value))
                    
            except Exception as e:
                print(f'ID {ID}: Error computing parameter {Param}: {e}')
                DFs_local[l].loc[99 - snap, str(ID)] = -np.inf
                updates.append((l, 99 - snap, str(ID), -np.inf))
    
    # Save individual ID results immediately after processing
    save_individual_id_results(ID, DFs_local, Params, PATH, SIM, SAVEFILE, lock, verbose)
    
    return ID, DFs_local, updates

def get_evolution_accreted_particles_parallel(Params, IDs, dfSample, rterm=2, 
                      Snaps=[4, 8, 17, 33, 50, 81, 99],
                      Update=False, NearBirth=False,
                      PATH=os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory',  
                      SAVEFILE='DFs', SIM='TNG50', NSim='-1', verbose=0,
                      Resume=True):
    '''
    Make dataframe for the evolution of given parameters and subhalos using particle data (parallel version)
    
    Parameters
    ----------
    Params : list of str
        Parameters to compute evolution for
    IDs : list of int  
        Array with the subhalos IDs to analyze
    dfSample : pandas.DataFrame
        Sample dataframe with subhalo information
    rterm : float, optional
        Radial term multiplier for filtering
    Snaps : list of int, optional
        Snapshots to analyze
    Update : bool, optional  
        Whether to update existing values in dataframes
    NearBirth : bool, optional
        Whether to focus on snapshots near subhalo birth
    PATH : str, optional
        Base path for saving results
    SAVEFILE : str, optional
        Subdirectory for saving files  
    SIM : str, optional
        Simulation name
    NSim : str, optional
        Simulation number
    verbose : bool, optional
        Whether to print detailed progress information
    Resume : bool, optional
        Whether to resume from previously completed IDs
    
    Returns
    -------
    None
        Saves results to CSV files in the specified directory
    
    Author: Abhner P. de Almeida (abhner.almeida AAT usp.br)
    '''

    # Load required subhalo data
    SubfindID = extractDF('SubfindID')
    
    # Initialize dataframes for each parameter
    DFs = []
    dfTime.a = 1 / (1 + dfTime.z)  # Convert redshift to scale factor

    if verbose == 0:
        print(f"Initializing analysis for {len(Params)} parameters and {len(IDs)} subhalos")
        print(f"Parameters: {Params}")
        print(f"Snapshots: {Snaps}")
    
    # Try to load existing individual results if Resume is True
    if Resume:
        existing_data = load_individual_id_results(IDs, Params, PATH, SIM, SAVEFILE, verbose)
        if existing_data is not None:
            DFs = existing_data
            if verbose > 0:
                print("Resumed from existing individual ID results")
        else:
            # Create new dataframes if no existing results
            for Param in Params:
                if verbose > 0:
                    print(f'Initializing dataframe for parameter: {Param}')
                    
                try:
                    # Try to load existing dataframe
                    df = extractDF(Param)
                    
                    # Add missing IDs as new columns with NaN values
                    for ID in IDs:
                        if str(ID) not in df.columns:
                            if verbose:
                                print(f'  Adding new subhalo ID: {ID}')
                            df[str(ID)] = np.nan
                            
                except Exception:
                    # Create new dataframe if loading fails
                    if verbose > 0:
                        print(f'  Creating new dataframe for {Param}')
                    snaps = np.arange(100)
                    df = pd.DataFrame(data=np.zeros((100, len(IDs) + 1), dtype=object), 
                                    columns=np.append('Snap', IDs))
                    df[:] = np.nan
                    df.Snap = np.flip(snaps)   
                    
                DFs.append(df)
    else:
        # Create new dataframes
        for Param in Params:
            if verbose > 0:
                print(f'Initializing dataframe for parameter: {Param}')
                
            try:
                # Try to load existing dataframe
                df = extractDF(Param)
                
                # Add missing IDs as new columns with NaN values
                for ID in IDs:
                    if str(ID) not in df.columns:
                        if verbose:
                            print(f'  Adding new subhalo ID: {ID}')
                        df[str(ID)] = np.nan
                        
            except Exception:
                # Create new dataframe if loading fails
                if verbose > 0:
                    print(f'  Creating new dataframe for {Param}')
                snaps = np.arange(100)
                df = pd.DataFrame(data=np.zeros((100, len(IDs) + 1), dtype=object), 
                                columns=np.append('Snap', IDs))
                df[:] = np.nan
                df.Snap = np.flip(snaps)   
                
            DFs.append(df)
    
    Snaps_Original = Snaps.copy()  # Keep original snapshot list
    
    # Prepare shared data for multiprocessing
    manager = Manager()
    lock = manager.Lock()
    
    shared_data = {
        'Params': Params,
        'Snaps_Original': Snaps_Original,
        'Update': Update,
        'NearBirth': NearBirth,
        'dfTime': dfTime,
        'rterm': rterm,
        'verbose': verbose,
        'PATH': PATH,
        'SIM': SIM,
        'SAVEFILE': SAVEFILE,
        'SubfindID': SubfindID,
        'DFs_template': DFs,
        'lock': lock
    }

    # Filter IDs to process (skip already completed ones if Resume=True)
    IDs_to_process = IDs.copy()
    if Resume:
        individual_dir = f"{PATH}/{SIM}/{SAVEFILE}/individual_ids"
        completed_ids = []
        for ID in IDs:
            marker_file = f"{individual_dir}/completed_ID_{ID}.txt"
            if os.path.exists(marker_file) and not Update:
                completed_ids.append(ID)
        
        IDs_to_process = [ID for ID in IDs if ID not in completed_ids]
        
        if verbose > 0:
            print(f"Resuming: {len(completed_ids)} IDs already completed, {len(IDs_to_process)} IDs to process")
    
    if not IDs_to_process:
        if verbose > 0:
            print("All IDs already completed. Use Update=True to force re-computation.")
        return

    # Prepare arguments for parallel processing
    args = [(ID, shared_data) for ID in IDs_to_process]
    
    if verbose > 0:
        print(f"Starting parallel processing with {len(IDs_to_process)} subhalos")
    
    # Use multiprocessing Pool
    with Pool() as pool:
        results = pool.map(process_single_subhalo, args)
    
    # Combine results from all processes
    if verbose > 0:
        print("Combining results from all processes...")
    
    for ID, DFs_local, updates in results:
        # Apply all updates to the final DataFrames
        for l, snap_idx, id_str, value in updates:
            DFs[l].loc[snap_idx, id_str] = value
        
        # Save main DataFrames after each subhalo (optional, since individual files are already saved)
        for l in range(len(Params)):
            safe_save_dataframe(
                DFs[l], 
                f"{PATH}/{SIM}/{SAVEFILE}/{Params[l]}.csv",
                lock
            )
        
        if verbose > 0:
            print(f"Completed and saved results for subhalo {ID}")

    # Final save of main DataFrames
    for l, Param in enumerate(Params):
        safe_save_dataframe(
            DFs[l], 
            f"{PATH}/{SIM}/{SAVEFILE}/{Param}.csv",
            lock
        )
    
    if verbose > 0:
        print(f'\n{"="*50}')
        print("Analysis completed successfully!")
        print(f"{'='*50}")
    
    return

# --- Constantes: G em (kpc * (km/s)^2) / Msun ---
G_KPC = 4.30091e-6

def _rho_crit_kpc3_from_Hz(Hz_km_s_Mpc: float) -> float:
    """rho_c = 3 H^2 / (8 pi G); H em km/s/kpc -> rho em Msun/kpc^3"""
    Hz_km_s_kpc = Hz_km_s_Mpc / 1e3  # 1 Mpc = 1000 kpc
    return 3.0 * Hz_km_s_kpc**2 / (8.0 * np.pi * G_KPC)

def _Hz_km_s_Mpc(z: float, H0=70.0, Omega_m=0.3, Omega_Lambda=None):
    if Omega_Lambda is None:
        Omega_Lambda = 1.0 - Omega_m
    return H0 * np.sqrt(Omega_m*(1+z)**3 + Omega_Lambda)

def _weighted_median(x, w):
    x = np.asarray(x); w = np.asarray(w)
    s = np.argsort(x)
    x, w = x[s], w[s]
    c = np.cumsum(w) - 0.5*w
    return x[np.searchsorted(c, 0.5*np.sum(w))]

def _solve_r200_kpc(R_kpc_sorted, Mcum_Msun_sorted, rho_c_kpc3, overdensity=200.0):
    """Resolve M(<R) = (4/3)*pi*Δ*rho_c*R^3 por bisseção com PCHIP."""
    valid = np.isfinite(R_kpc_sorted) & np.isfinite(Mcum_Msun_sorted)
    R = np.asarray(R_kpc_sorted[valid]); M = np.asarray(Mcum_Msun_sorted[valid])
    positive = R > 0
    R, M = R[positive], M[positive]
    if len(R) < 5:
        return np.nan, np.nan
    idx = np.argsort(R)
    R, M = R[idx], M[idx]
    R, uniq_idx = np.unique(R, return_index=True)
    M = M[uniq_idx]
    if len(R) < 5:
        return np.nan, np.nan

    M_of_R = chipInterpolator(R, M, extrapolate=False)

    def F(Rval):
        Mlhs = M_of_R(Rval)
        Mrhs = (4.0/3.0)*np.pi*overdensity*rho_c_kpc3*(Rval**3)
        return Mlhs - Mrhs

    a, b = R[0], R[-1]
    fa, fb = F(a), F(b)
    if not (np.isfinite(fa) and np.isfinite(fb) and fa*fb <= 0):
        for _ in range(8):
            b *= 1.5
            fb = F(b)
            if np.isfinite(fa) and np.isfinite(fb) and fa*fb <= 0:
                break
        else:
            return np.nan, np.nan

    for _ in range(60):
        c = 0.5*(a+b)
        fc = F(c)
        if not np.isfinite(fc):
            b = c
            continue
        if abs(fc) < 1e-8:
            a = b = c
            break
        if fa*fc <= 0:
            b = c; fb = fc
        else:
            a = c; fa = fc

    R200 = 0.5*(a+b)
    M200 = (4.0/3.0)*np.pi*200.0*rho_c_kpc3*(R200**3)
    return R200, M200

def _center_and_bulk(pos, vel, mass):
    cx = _weighted_median(pos[:,0], mass)
    cy = _weighted_median(pos[:,1], mass)
    cz = _weighted_median(pos[:,2], mass)
    vx = _weighted_median(vel[:,0], mass)
    vy = _weighted_median(vel[:,1], mass)
    vz = _weighted_median(vel[:,2], mass)
    return np.array([cx,cy,cz]), np.array([vx,vy,vz])

def Evolution200(
    IDs,
    Snaps,
    *,
    extractParticles,         # callable: (ID, snaps=[snap]) -> [file]
    dfTime,                   # DataFrame com colunas 'Snap' e 'z'
    h, Omegam0,               # cosmologia
    ETNG=None,                # opcional: ETNG.FixPeriodic
    sim_name='TNG50-1',
    use_group_catalog_for_centrals=True,
    extractDF=None,           # callable para pegar Group_* e SubfindID/GroupFirstSub
    # IMPORTANTES: Group_* já em log10(kpc) e log10(Msun)
    group_keys=('Group_R_Crit200','Group_M_Crit200','SubfindID','GroupFirstSub'),
    return_log10=True,        # aplica log10 em R200 e M200 nos DataFrames
    verbose=False
):
    """
    Computa R200, M200, V200 e J200 (log10 |J|) por (ID, snap).
    - R200 em kpc (log10 se return_log10=True)
    - M200 em Msun (log10 se return_log10=True)
    - V200 em km/s (sempre linear)
    - J200 em log10(|J|) (consistente com seu fluxo)
    """
    IDs = list(IDs)
    Snaps = np.array(Snaps, dtype=int)

    # prepara DFs
    idx = pd.Index(Snaps, name='Snap')
    cols = pd.Index([str(i) for i in IDs], name='ID')
    df_R200 = pd.DataFrame(np.nan, index=idx, columns=cols)
    df_M200 = pd.DataFrame(np.nan, index=idx, columns=cols)
    df_V200 = pd.DataFrame(np.nan, index=idx, columns=cols)
    df_J200 = pd.DataFrame(np.nan, index=idx, columns=cols)

    # catálogo (em log10)
    if use_group_catalog_for_centrals and extractDF is not None:
        Group_R_log10 = extractDF(group_keys[0])  # log10(kpc)
        Group_M_log10 = extractDF(group_keys[1])  # log10(Msun)
        SubfindID_df = extractDF(group_keys[2])
        GroupFirstSub_df = extractDF(group_keys[3])
    else:
        Group_R_log10 = Group_M_log10 = SubfindID_df = GroupFirstSub_df = None

    for ID in IDs:
        sid = str(ID)
        if verbose: print(f'ID {ID}')
        for snap in Snaps:
            if verbose: print(f'  snap {snap}')
            try:
                f = extractParticles(ID, snaps=[snap])[0]
            except Exception as e:
                if verbose: print(f'    extractParticles falhou: {e}')
                continue

            # z e fatores
            z = float(dfTime.loc[dfTime.Snap == snap, 'z'].values[0])
            a = 1.0/(1.0+z); sqrt_a = np.sqrt(a)

            # GAS
            try:
                posGas = f['PartType0']['Coordinates'][:] / (1+z) / h
                velGas = f['PartType0']['Velocities'][:] * sqrt_a
                massGas = f['PartType0']['Masses'][:] * 1e10 / h
            except Exception:
                posGas = np.zeros((0,3)); velGas = np.zeros((0,3)); massGas = np.zeros(0)

            # DM
            try:
                posDM = f['PartType1']['Coordinates'][:] / (1+z) / h
                velDM = f['PartType1']['Velocities'][:] * sqrt_a
                mDM = f['Header'].attrs['MassTable'][1] * 1e10 / h
                massDM = np.full(posDM.shape[0], mDM)
            except Exception:
                posDM = np.zeros((0,3)); velDM = np.zeros((0,3)); massDM = np.zeros(0)

            # STARS
            try:
                posStar = f['PartType4']['Coordinates'][:] / (1+z) / h
                velStar = f['PartType4']['Velocities'][:] * sqrt_a
                massStar = f['PartType4']['Masses'][:] * 1e10 / h
            except Exception:
                posStar = np.zeros((0,3)); velStar = np.zeros((0,3)); massStar = np.zeros(0)

            # centróide/bulk
            if massStar.size >= 10:
                Cen, VB = _center_and_bulk(posStar, velStar, massStar)
            elif massGas.size >= 10:
                Cen, VB = _center_and_bulk(posGas, velGas, massGas)
            elif massDM.size >= 10:
                Cen, VB = _center_and_bulk(posDM, velDM, massDM)
            else:
                continue

            if ETNG is not None and hasattr(ETNG, 'FixPeriodic'):
                posGas = ETNG.FixPeriodic(posGas - Cen, sim=sim_name)
                posDM  = ETNG.FixPeriodic(posDM  - Cen, sim=sim_name)
                posStar= ETNG.FixPeriodic(posStar- Cen, sim=sim_name)
            else:
                posGas -= Cen; posDM -= Cen; posStar -= Cen

            velGas -= VB; velDM -= VB; velStar -= VB

            Mass = np.concatenate([massGas, massDM, massStar])
            Pos  = np.vstack([posGas, posDM, posStar]) if Mass.size>0 else np.zeros((0,3))
            Vel  = np.vstack([velGas, velDM, velStar]) if Mass.size>0 else np.zeros((0,3))
            if Mass.size < 10:
                continue

            R = np.linalg.norm(Pos, axis=1)
            order = np.argsort(R)
            R_sorted = R[order]
            Mcum = np.cumsum(Mass[order])

            Hz = _Hz_km_s_Mpc(z, H0=100*h, Omega_m=Omegam0)
            rho_c_kpc3 = _rho_crit_kpc3_from_Hz(Hz)

            # R200/M200 por solver
            R200_kpc, M200_Msun = _solve_r200_kpc(R_sorted, Mcum, rho_c_kpc3, overdensity=200.0)

            # opcional: usar catálogo para CENTRAIS (catalogo JÁ em log10)
            if use_group_catalog_for_centrals and Group_R_log10 is not None:
                try:
                    idx_cat = int(99 - snap)  # convenção dos seus DFs
                    is_central = (SubfindID_df[sid].values[idx_cat] == GroupFirstSub_df[sid].values[idx_cat])
                except Exception:
                    is_central = False

                if is_central:
                    try:
                        R200_kpc_cat = 10**Group_R_log10[sid].values[idx_cat]  # log10(kpc) -> kpc
                        M200_Msun_cat = 10**Group_M_log10[sid].values[idx_cat] # log10(Msun) -> Msun
                        # priorize catálogo se válido
                        if np.isfinite(R200_kpc_cat) and R200_kpc_cat > 0 and np.isfinite(M200_Msun_cat):
                            R200_kpc, M200_Msun = R200_kpc_cat, M200_Msun_cat
                    except Exception:
                        pass

            # V200: velocidade circular em R200
            if np.isfinite(R200_kpc) and R200_kpc > 0 and np.isfinite(M200_Msun):
                V200 = np.sqrt(G_KPC * M200_Msun / R200_kpc)
                inside = (R_sorted < R200_kpc)
                Jvec = Mass[:,None]*np.cross(Pos, Vel)
                J200_vec = np.nansum(Jvec[order][inside], axis=0)
                J200_log = np.log10(np.linalg.norm(J200_vec)) if np.isfinite(np.linalg.norm(J200_vec)) and np.linalg.norm(J200_vec)>0 else np.nan
            else:
                V200 = np.nan
                J200_log = np.nan

            col = sid
            df_R200.at[snap, col] = np.log10(R200_kpc) if (return_log10 and np.isfinite(R200_kpc) and R200_kpc>0) else R200_kpc
            df_M200.at[snap, col] = np.log10(M200_Msun) if (return_log10 and np.isfinite(M200_Msun) and M200_Msun>0) else M200_Msun
            df_V200.at[snap, col] = V200                         # sempre linear (km/s)
            df_J200.at[snap, col] = J200_log                     # log10(|J|)

    return {
        'R200': df_R200.sort_index(),
        'M200': df_M200.sort_index(),
        'V200': df_V200.sort_index(),
        'J200': df_J200.sort_index()
    }
