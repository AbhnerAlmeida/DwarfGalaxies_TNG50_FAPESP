import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mplcol # for log colorbars
import os
import sys
import h5py
import requests
import weightedstats as ws
import mpltern # ternary diagrams
from statsmodels.stats.weightstats import DescrStatsW
from scipy.spatial import distance
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from astropy.table import Table  # for FITS handling

#import illustris_python as il
from collections.abc import Iterable # for isinstance ... to check if param2 is scalar or array
home = os.getenv("HOME")
home_dir = home + '/'
tng_dir = home_dir + 'SIMS/TNG/'
import illustris_python as il
from profcl import *
import mathutils as mmu
import graphutils as ggu
    
# plt.style.use('bmh')
# plt.style.use('/Users/gam/.matplotlib/stylelib/gam-dark.mplstyle')

baseUrl = 'http://www.tng-project.org/api/'

token = "caaddaa4f3de0daa0e043058306cf52d"

# Make sure you have a TNGTOKEN environment variable set
if 'token' not in globals():
    token = os.getenv("TNGTOKEN")
    if token is None:
        raise ValueError("Need to set environment variable TNGTOKEN and relaunch Jupyter...")
headers = {"api-key":token}
# print("headers=",headers)

data_root_dir = home_dir + "SIMS/TNG/"
if os.path.isdir(data_root_dir) == False:
    answer = input("Directory data_root_dir is not present on your system: Create (y/n)? ")
    if answer in ['y','Y']:
        os.mkdir(data_root_dir)
    else:
        root_dir = input("Enter root directory for TNG simulation data: ")
        data_root_dir = home_dir + root_dir
        
    
basePath50 = data_root_dir + 'TNG50-1/output/'
basePath100 = data_root_dir + 'TNG100-1/output/'

# print("past header...")

# cosmological parameters
Omegam0 = 0.3089
h = 0.6774
lh = np.log10(h)

# dictionaries for conversions

dict_param = {
    'grnr':'SubhaloGrNr',
    'id':'SubfindID',
    'm200':'Group_M_Crit200',
    'm500':'Group_M_Crit500',
    'mass':'SubhaloMass',
    'massrh':'SubhaloMassInHalfRad',
    'massrhtype':'SubhaloMassInHalfRadType',
    'masstype':'SubhaloMassType',
    'masses':'SubhaloMassType',
    'mass2rh':'SubhaloMassInRad',
    'mass2rhtype':'SubhaloMassInRadType',
    'masses2rh':'SubhaloMassInRadType',
    'mdot':'SubhaloBHMdot',
    'r200':'Group_R_Crit200',
    'r500':'Group_R_Crit500',
    'sfr':'SubhaloSFR',
    'sfr2rh':'SubhaloSFRinRad',
    'rh':'SubhaloHalfmassRad',
    'rhtype':'SubhaloHalfmassRadType'
    }

dict_labels = {
    'Group_M_Crit200':'\log(M_{200,\mathrm{c}}/\mathrm{M}_\odot)',
    'Group_M_Crit500':'\log(M_{500,\mathrm{c}}/\mathrm{M}_\odot)',
    'SubhaloMass'    :'\log(M_\mathrm{tot}/\mathrm{M}_\odot)',
    'SubhaloMassType'    :'\log(M/\mathrm{M}_\odot)',
    'SubhaloMassInHalfRadType'    :'\log[M(r_\mathrm{half}^\mathrm{stars})/\mathrm{M}_\odot]',
    'SubhaloMassInRadType'    :'\log[M(2r_\mathrm{half}^\mathrm{stars})/\mathrm{M}_\odot]',
    'SubhaloBHMdot'    :'\log(\dot M_\mathrm{BH})\ (\mathrm{M}_\odot)/\mathrm{yr})',
    'Group_R_Crit200' : '\log(R_{200,\mathrm{c}}/\mathrm{kpc})',
    'Group_R_Crit500' : '\log(R_{500,\mathrm{c}}/\mathrm{kpc})',
    'SubhaloSFR' : '\log(\mathrm{SFR})\ (\mathrm{M}_\odot/\mathrm{yr})',
    'SubhaloSFRinRad' : '\log[\mathrm{SFR}(2r_\mathrm{half}^\mathrm{stars})]\ (\mathrm{M}_\odot/\mathrm{yr})',
    'SubhaloHalfmassRad' : 'stellar half mass radius (kpc)',
    'SubhaloHalfmassRadType' : 'half mass radius (kpc)',
    'SubhaloGrNr' : 'group ID'
    }
dict_types = {
    0: 'gas',
    1: 'dm',
    4: 'stars',
    5: 'bh'}

def df2fits(df,file=None,data_dir=None,sim='TNG50-1',verbose=0):
    if data_dir is None:
        data_dir = tng_dir + sim + '/'
    if verbose > 0:
        print("converting dataframe to astropy table...")
    t = Table.from_pandas(df)
    if file.find('.fits') == -1:
        file = file + '.fits'
    if verbose > 0:
        print("writing astropy table to FITS file",data_dir + file,"...")
    try:
        t.write(data_dir + file)
    except OSError:
        ans = input("Do you want to overwrite the existing file? (y/n): ")
        if ans == 'y':
            t.write(data_dir + file,overwrite=True)

def ReadFITS(file,data_dir,df=True,verbose=0):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    file = data_dir + file
    if file[-5:] != '.fits':
        file = file + '.fits'
    if verbose>0:
        print("reading",file,"...")
    t = Table.read(file)
    if verbose > 0:
        print(t)
        if df:
            print("converting to Pandas dataframe...")

    if not df:
        return t
    df = t.to_pandas()
    if ('ra_gal' in df.columns) & ('dec_gal' in df.columns):
        df.rename(columns={'ra_gal':'RA','dec_gal':'Dec'},inplace=True)
    if verbose > 0:
        print(df.columns)
        print(len(df),"galaxies")
    return df

def get(path, params=None, verbose=0, save_dir=None):
    # make HTTP GET request to path
    
    if verbose > 0:
        print("extracting ...")
    r = requests.get(path, params=params, headers=headers)
    
    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if verbose > 0:
        print("get: r.headers=",r.headers)
        print("path = ",path)
        print('"ls -ltr | tail -3" gives:')
        os.system("ls -ltr | tail -3")
    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically

    if 'content-disposition' in r.headers:
        filename = r.headers['content-disposition'].split("filename=")[1]
        if save_dir is not None:
            filename = save_dir + '/' + filename
            filename = filename.replace('//','/')
        if verbose > 0:
            print("filename = ",filename)
        with open(filename, 'wb') as f:
            if verbose > 0:
                print("writing in", filename,"...")
            f.write(r.content)

        return filename # return the filename string
    if verbose > 0:
        print("returning full request")
    return r

def getsims():
    """Extract list of Illustris (and TNG) simulations
    Author: Gary Mamon (gam AAT iap.fr)
    """
    r = get(baseUrl)
    sims = [sim['name'] for sim in r['simulations']]
    return sims

def getsim(simulation="TNG50-1"):
    """Extract simulation info
    Author: Gary Mamon (gam AAT iap.fr using Illustrus-TNG recipes)
    argument: simulation (default "TNG50-1") [string]
    returns info [dict]
    """
    r = get(baseUrl)
    sims = [sim['name'] for sim in r['simulations']]
    isim = sims.index(simulation)
    return get(r['simulations'][isim]['url'])

def getTree(subhalo,snap,sim='TNG50-1',tree_method='sublink_mpb',
                data_dir=None,forceExtract=False,
                verbose=0,
                params=['SnapNum','SubfindID','FirstSubhaloInFOFGroupID',
                        'DescendantID','FirstProgenitorID']):
    """Extract tree for subhalo
    Arguments:
            subhalo: SubhaloID
            snap: SnapNum
            sim: simulation (default: 'TNG50-1')
            tree_method: one of 'sublink', 'sublink_mpb' (default), 'sublink_mdb'
            data_dir: data directory (None for hierarchy under dataroot_dir by default)
            forceExtract: force extraction from TNG database (default False)
            
    Returns:
        dictionary
        
    Author: Gary Mamon (gam AAT iap.fr)
            """
    if data_dir is None:
        data_dir = dataroot_dir + sim + '/output/'
    elif data_dir == '.':
        data_dir = os.getcwd() + '/'
    
    # check if data on disk
    file = tree_method + '_' + str(subhalo) + '.hdf5'
    filePath = data_dir + file
    if verbose > 0:
        print("filepath=",filePath)
    if forceExtract or not os.path.isfile(filePath):
        if verbose > 0:
            print("extracting from TNG database...")
        # subhalo info
        sub = getsubhalo(subhalo,snapnum=snap,simulation=sim,
                         save_dir=dataroot_dir+sim+'/')
        filePath = get(sub['trees'][tree_method],save_dir=data_dir,verbose=verbose)
        filePath = filePath.replace('//','/')
        if verbose > 0:
            print("output saved on",file)
    if verbose > 0:
        print("filePath for read = ", filePath)
    f = h5py.File(filePath,'r')
    return f
    # if 'SnapNum' in params:
    #     snaps = f['SnapNum'][:]
    # if 'SubfindID' in params:
    #     subhalos = f['SubfindID'][:]
    # if 'FirstSubhaloInFOFGroupID' in params:
    #     centrals = f['FirstSubhaloInFOFGroupID'][:]
    # if 'DescendantID' in params:
    #     descendants = f['DescendantID'][:]
    # if 'FirstProgenitorID' in params:
    #     firstprogs = f['FirstProgenitorID'][:]
        
    
# def CleanTree(subhalo,snap,sim='TNG50-1',tree_method='sublink_mpb',
#                 data_dir=None,forceExtract=False):
#     f = getTree(subhalo,snap=snap,sim=sim,tree_method=tree_method,
#                 data_dir=data_dir,forceExtract=forceExtract)
    
def zofsnap(snap,sim="TNG50-1"):
    """redshift of given snapshot number
    author: Gary Mamon (gam AAT iap.fr)"""
    sim_dict = getsim(sim)
    snap_dict = get(sim_dict['snapshots'])
    return snap_dict[snap]['redshift']
    
def zofsnaps(snaps,sim="TNG50-1"):
    """redshift(s) of given snapnum or array of snapnums
    author: Gary Mamon, gam AAT iap.fr
    """
    snaps_z_all = snapofz('all',simulation=sim)
    snaps_z_all.reverse()
    snaps_z_all = np.array(snaps_z_all)
    # print("snaps_z_all=",snaps_z_all)
    z = snaps_z_all[np.isin(snaps_z_all[:,0],snaps)][:,1]
    return z

def getsnapofz(z,simulation="TNG50-1",):
    """Extract snapnum info
    arguments:
        redshift
        simulation (default "TNG50-1") [string]
    returns info [dict]
    Author: Gary Mamon (gam AAT iap.fr using Illustris-TNG recipes)
    """
    s = getsim(simulation)
    url = s['snapshots'] + "z=" + str(z) + "/"
    snap = get(url)
    return snap

def snapofz(z='all',simulation="TNG50-1",show_z=False):
    """Extract snapnum of given redshift for given simulation
    arguments:
        redshift ('all' for all snapnums and corresponding redshifts)
        simulation (default "TNG50-1") [string]
        show_z: if True also return the redshift
    Author: Gary Mamon (gam AAT iap.fr using Illustris-TNG recipes)
    """
    if z == 'all':
        sim = get(getsim(simulation)['snapshots'])
        snap_and_z = [(s["number"],s["redshift"]) for s in sim]
        return snap_and_z
    elif show_z:
        snap = getsnapofz(z,simulation=simulation)
        return snap['number'],snap['redshift']
    else:
        return getsnapofz(z,simulation=simulation)['number']


def ztall(simulation='TNG50-1'):
    """returns table snap,z,t,tlook"""
    tabsnapz = np.array(snapofz())
    z = tabsnapz[:,1]
    t = AgeUniverse(Omegam0, h, z)
    tlook = t.max()-t
    return np.transpose([tabsnapz[:,0],z,t,tlook])

def tlookofsnap(snaps,sim='TNG50-1'):
   """returns lookback times for list of snaps"""
   tab = ztall(sim)
   # a = 1/(1+tab[:,1])
   snapsall = tab[:,0]
   f = interp1d(snapsall,tab[:,-1],kind='cubic',bounds_error=False)
   return f(snaps)
   
def tlookofa(a,sim='TNG50-1',interp_kind='linear'):
    """returns lookback time for given scale factor
    arguments: 
        a: array of scale factors
        sim: simulation (default 'TNG50-1')
        interp_kind: interpolation order ('linear' or 'cubic'), default 'cubic'
    returns array of lookback times
    author: Gary Mamon (gam AAT iap.fr) """
    tab = ztall(sim)
    z = tab[:,1]
    aofsnaps = 1/(1+z)
    tlook = tab[:,-1]
    f = interp1d(aofsnaps,tlook,kind=interp_kind)
    return f(a)

def gethalo(haloid,subhalo=False,simulation='TNG50-1',snapnum=99,parameter=None):
    """Extract halo parameters at given snapnum for given haloID or subhaloID (if subhalo=True)
    Author: Gary Mamon (gam AAT iap.fr)"""
    if subhalo:
        # extract haloID
        haloid = getsubhalo(haloid,simulation=simulation,snapnum=snapnum,parameter='grnr')

    url = 'http://www.tng-project.org/api/' + simulation + '/snapshots/' \
            + str(snapnum) + '/halos/' + str(haloid) + '/info.json'
    if parameter is None:
        return get(url)
    else:
        return get(url)[parameter]
    
def getsubhalo(subhaloid,simulation='TNG50-1',snapnum=99,snapmax=99,parameter=None,
               fromTree=False,treeMethod='sublink_mpb',datafileprefix=None,
               save_dir=None,verbose=0):
    """Extract subhalo parameters at given snapnum
    return dictionary of parameters and values
    Author: Gary Mamon (gam AAT iap.fr)"""
    if fromTree:
        if verbose > 0:
            print("in fromTree block")
        if save_dir is None:
            save_dir = os.getenv("HOME") + "/SIMS/TNG/" + simulation + "/output/"
        if verbose > 0:
            print("getsubhalo: save_dir = ",save_dir)
        if datafileprefix is None:
            # extract subhalo URL 
            if verbose > 0:
                print("extracting subhalo URL...")
            if snapnum < snapmax:
                subhaloid = getsubhaloid99(subhaloid,simulation=simulation,snapnum=snapnum,verbose=verbose)
                if verbose > 0:
                    print("subhaloid99 = "), subhaloid
            sub = getsubhalo(subhaloid,simulation=simulation,snapnum=snapmax,
                             save_dir=save_dir,verbose=verbose)
            progurl = sub['related']['sublink_progenitor']
            if progurl is None:
                print("no progenitor for subhalo", subhaloid)
                return
            
            # extract tree of main progenitor
            if verbose > 0:
                print("extracting tree of main progenitor...")
            datafile = get( sub['trees'][treeMethod] )
            if verbose > 0:
                print("datafile = ", datafile)
        elif datafileprefix in [0,'a',"auto"]:
            datafile = "sublink_mpb_" + str(subhaloid) + ".hdf5"
        # elif datafileprefix = 'groups99':
        #     datafile = 'groups_099/'
        else:
            datafile = datafileprefix + ".hdf5" 
        # extract values
        if verbose > 0:
            print("extracting parameters from tree in file " + datafile + "...")
        try:
            f = h5py.File(datafile,'r')
        except:
            raise FileNotFoundError(datafile)
        fsnapnum = f.loc[f['snapnum']==snapnum]
        if parameter == None:
            return fsnapnum
        elif type(parameter) == list:
            return list(map(fsnapnum).get,parameter)
        else:
            return fsnapnum[parameter]
    else:
        url = 'http://www.tng-project.org/api/' + simulation + '/snapshots/' + str(snapnum) + '/subhalos/' + str(subhaloid)
        if verbose > 0:
            print("in regular block: url =",url)
        if parameter is None:
            return get(url)
        elif parameter in ["ssfr","sSFR"]:
            return get(url)['sfrinrad']/(1e10*get(url)['massinrad_stars'])
        elif type(parameter) == list:
            return list(map(get(url).get,parameter))
        else:
            return get(url)[parameter]

def getsubhalos(params,  parammins=None,parammaxs=None,snapnum=99, SIM = 'TNG50-1',
                verbose=0):
                
               
    basePath=home + "/SIMS/TNG/"+SIM+"/output"   
    paramsDownload = []
    for param in params:
           if  ('Type' in param) or ('Photometrics' in param):
           	paramsDownload.append(param[:-1])
           else:
           	paramsDownload.append(param)
    subhalos = il.groupcat.loadSubhalos(basePath=basePath,snapNum=snapnum,fields=paramsDownload)
    # restrict to range of parameters
    
    if len(params) > 1:
        count = subhalos['count']
        i = np.arange(count).astype(int)
        # inew = np.zeros(len(params))
        for j, param in enumerate(params):
            print("param=",param)
            paramName = paramsDownload[j]
            paramall = subhalos[paramName]
            # must do below np.where instead ...
            if ((parammins is not None) & (parammaxs is not None)):
                inew = i[(paramall >= parammins[j]) & (paramall <= parammaxs[j])]
            else:
                inew = i
            # take intersection of inew[j]s
            if j == 0:
                i2 = inew
            else:
                i2 = np.intersect1d(i2,inew,assume_unique=True)
            print("j len(i2)=",j,len(i2))
        values = np.zeros((len(params),len(i2)))
        for j, param in enumerate(params):
            print('param now =',param)
            paramName = paramsDownload[j]
            if ('Type' in param) or ('Photometrics' in param): # all types
                values[j,:] = subhalos[paramName][i2][:,int(param[-1])]
            else:
                values[j,:] = subhalos[paramName][i2]
    elif ((parammins is not None) & (parammaxs is not None)):
        values = subhalos[(subhalos>=parammins[0]) & (subhalos <= parammaxs[0])]
    else:
        values = subhalos
    if verbose > 0:
        print(values)
    return values.T

def getMPB(subhaloid,snapnum=99,sim='TNG50-1',verbose=0):
    """Get main progenitor branch
    Arguments:sub
        subhaloid: subhalo Subfind_ID (at snapshot = snapnum)
        snapnum: snapshot number [default 99]
        sim: simulation [default TNG50-1]
    Returns: dictionary-like h5py._hl.files.File 
    
    Author: Gary Mamon (gam AAT iap.fr)
    """
    
    filearg =  sim + "/snapshots/" + str(snapnum) + "/subhalos/" + str(subhaloid) + "/sublink/"
    file = dataroot_dir + filearg + "sublink_mpb_"+ str(subhaloid) + ".hdf5"
    if verbose > 0:
        print("attempting to read from file=",file)
    save_dir = dataroot_dir + sim + "/snapshots/" + str(snapnum) + "/subhalos/" + str(subhaloid) + "/sublink/"
    ok = 1
    try:       
        os.chdir(save_dir)
    except:
        if verbose > 0:
            print("cannot change directory to ", save_dir)
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            # directory already exists
            pass
        except:
            print("cannot mkdir", save_dir)
            ok = 0
        if ok == 1:
            if verbose > 1:
                print('changing cirectory to ',save_dir)
            try:
                os.chdir(save_dir)
            except:
                if verbose > 0:
                    print("failed to chdir to ", save_dir)
                ok = 0
    if ok == 1:
        try:
            f = h5py.File(file, 'r')
        except:
            if verbose > 0:
                print("cannot read file", file)
            ok = 0
    if ok == 0:
        url = baseUrl + filearg + "mpb.hdf5"
        if verbose > 0:
            print("reading from TNG database",url," (saving to ",os.getcwd() + ") ...")
        mpb = get(url)
        f = h5py.File(mpb,'r')
            
    return f

def getMDB(subhaloid,snapnum=50,sim='TNG50-1',extract=False,verbose=0):
    """Get main descendant branch
    Arguments:
        subhaloid: subhalo Subfind_ID (at snapshot = snapnum)
        snapnum: snapshot number [default 50]
        sim: simulation [default TNG50-1]
    Returns: dictionary-like h5py._hl.files.File 
    
    Author: Gary Mamon (gam AAT iap.fr)
    """
        
    filearg =  sim + "/snapshots/" + str(snapnum) + "/subhalos/" + str(subhaloid) + "/sublink/"
    file = dataroot_dir + filearg + "sublink_mdb_"+ str(subhaloid) + ".hdf5"
    if (verbose > 0) & (not extract):
        print("attempting to read from file=",file)
    save_dir = dataroot_dir + sim + "/snapshots/" + str(snapnum) + "/subhalos/" + str(subhaloid) + "/sublink/"
    ok = 1
    try:       
        os.chdir(save_dir)
    except:
        if verbose > 0:
            print("cannot change directory to ", save_dir)
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            # directory already exists
            pass
        except:
            print("cannot mkdir", save_dir)
            ok = 0
        if ok == 1:
            if verbose > 1:
                print('changing cirectory to ',save_dir)
            try:
                os.chdir(save_dir)
            except:
                if verbose > 0:
                    print("failed to chdir to ", save_dir)
                ok = 0
    if (ok == 1 ) & (not extract):
        try:
            f = h5py.File(file, 'r')
        except:
            if verbose > 0:
                print("cannot read file", file)
            ok = 0
    if (ok == 0) or (extract):
        url = baseUrl + filearg + "mdb.hdf5"
        if verbose > 0:
            print("reading from TNG database " + url + " (saving to ",os.getcwd() + ") ...")
        mdb = get(url)
        f = h5py.File(mdb,'r')
            
    return f

# constants
# simvals = getsim()
# Omegam0 = simvals['omega_0'] 
# h = simvals['hubble']

def Softening(sim='TNG50-1',zeps_table=False):
    """Softening scales for stars and dark matter at all redshifts
    arguments:
        sim: simulation (default 'TNG50-1')
        zeps_table: return (snap,z,t,tlook,epsilon) table instead of  epsilons
                    (default False)
    returns:
        softening scales [physical kpc, array] if zeps_table is False
        2D array of (redshifts,softening scales [physical kpc]) otherwise
        
    author: Gary Mamon (gam AAT iap.fr)
        """
    sim_values = getsim(sim)
    h = sim_values['hubble']
    rsoft_comov = sim_values['softening_stars_comoving']
    
    # redshifts
    tab = ztall(sim)
    z = tab[:,1]
   
    # softening scales: constant comoving for z > 1, then constant physical
    rsoft = rsoft_comov/h*np.where(z > 1, 1/(1+z), 0.5)
    if zeps_table:
        snaps = tab[:,0]
        t = tab[:,2]
        tlook = tab[:,3]
        return np.transpose([snaps,z,t,tlook,rsoft])
    else:
        return rsoft

def getStellarCircularities(sim='TNG50-1',snapnum=99,df=True,verbose=0):
    """Get supplemntary table c: Stellar Circularities"""
    data_dir = tng_dir + sim + '/'
    filePath = data_dir + 'stellar_circs.hdf5'
    if verbose > 0:
        print("Extracting stellar circularities from",filePath)
    if os.path.isfile(filePath):
        snapnum_str = 'Snapshot_' + str(snapnum)
        f = h5py.File(filePath)[snapnum_str]
        if df:
            return ConvertHDF52df(f)
        else:
            return f
    else:
        raise ValueError(filePath + " is not a file")

def getKinematicDecomposition(sim='TNG50-1',df=True,verbose=0):
    """Get supplemntary table m: Kinematic decompositions"""
    data_dir = tng_dir + sim + '/'
    print("data_dir = ",data_dir)
    filePath = data_dir + 'kinematic_decomposition_099.hdf5'
    if verbose > 0:
        print("Extracting kinematic decompositions from",filePath)
    if os.path.isfile(filePath):
        f = h5py.File(filePath)
        if df:
            return ConvertHDF52df(f)
        else:
            return f
    else:
        raise ValueError(filePath + " is not a file")
        
def getMWM31(df=True,newcols=True,verbose=0):
    """Get supplemntary table 1: TNG50 Milky Way+Andromeda-like galaxies"""
    sim = 'TNG50-1'
    data_dir = tng_dir + sim + '/'
    filePath = data_dir + 'mwm31s_hostcatalog.hdf5'
    if verbose > 0:
        print("Extracting MW/M31 catalog from",filePath)
    if os.path.isfile(filePath):
        f = h5py.File(filePath)
        if df:
            df = ConvertHDF52df(f)
            if newcols:
                df['lM_200'] = np.log10(df.HaloMass_M200c)
                df['lM_stars'] = np.log10(df.StellarMass_all)
                df['lM_BH'] = np.log10(df.SMBH_Mass)
            return df
        else:
            if newcols:
                f['lM_200'] = np.log10(df.HaloMass_M200c)
                f['lM_stars'] = np.log10(df.StellarMass_all)
                f['lM_BH'] = np.log10(df.SMBH_Mass)
            return f
    else:
        raise ValueError(filePath + " is not a file")
        
def getMergerHistory(sim='TNG50-1',snapnum=99,df=True,verbose=0):
    """Get supplemntary table y: Merger history"""
    data_dir = tng_dir + sim + '/'
    filePath = data_dir + 'MergerHistory_%03d'% snapnum + '.hdf5'
    if verbose > 0:
        print("Extracting merger histories from",filePath)
    if os.path.isfile(filePath):
        f = h5py.File(filePath)
        if df:
            return ConvertHDF52df(f)
        else:
            return f
    else:
        raise ValueError(filePath + " is not a file")
        
def getHaloStructure(sim='TNG50-1',snapnum=99,df=True,verbose=0):
    """Get supplemntary table q: Halo structure"""
    data_dir = tng_dir + sim + '/'
    filePath = data_dir + 'halo_structure_%03d'% snapnum + '.hdf5'
    if verbose > 0:
        print("Extracting halo structure from",filePath)
    if os.path.isfile(filePath):
        f = h5py.File(filePath)
        if df:
            return ConvertHDF52df(f,avoid_keys=['Header'])
        else:
            return f
    else:
        raise ValueError(filePath + " is not a file")

def Type2Str(Type):
    if Type == 0:
        name = "gas"
    elif Type == 1:
        name = "dm"
    elif Type == 4:
        name = "stars"
    elif Type == 5:
        name = "bh"
    else:
        raise ValueError("cannot recognize Type=",Type)
    return name
    
def ConvertParam(abbrev):
    try:
        param = dict_param[abbrev.lower()]
    except:
        param = abbrev
    return param

def saveParticles(subhalo,sim="TNG50-1",snapnum=99,cutout_request=None,save_dir=None,
                  verbose=0):
    """ Download and save particle cutout for one subhalo. """
    baseUrl = 'http://www.tng-project.org/api/'

    sub_prog_url = baseUrl+sim+"/snapshots/"+str(snapnum)+"/subhalos/"+str(subhalo)+"/"
    sub_prog_url = sub_prog_url + 'cutout.hdf5'
    if verbose > 0:
        print("sub_prog_url = ", sub_prog_url)
        
    cutout = get(sub_prog_url)

    '''
    if cutout_request is None:
        cutout_request = {'gas':'Coordinates,Masses, GFM_MetalsTagged',
                          'dm':'Coordinates',
                          'stars':'Coordinates,Masses'}

    if verbose > 0:
        print("cutout_request = ", cutout_request)
        
    # extract particle data from TNG web site
    cutout = get(sub_prog_url, cutout_request)
    '''
    if verbose > 0:
        print("type(cutout)=",type(cutout))
        print("cutout=",cutout)

    # move to desired directory
    if save_dir is None:
        save_dir = \
            tng_dir + sim + "/snapshots/" + str(snapnum) + "/subhalos/" + str(subhalo) 
    try:
        os.makedirs(save_dir)
    except:
        # directory already exists
        pass
    final_filename = save_dir + "/cutout_" + sim + "_" + str(snapnum) \
                        + "_" + str(subhalo) + ".hdf5"
    if verbose > 0:
        print("filename is now", final_filename)
    os.rename(cutout, final_filename)
    
def particles(subhalo,sim='TNG50-1',snapnum=99,PartType=None,
              cutout_request=None,params=None,
              save_dir=None,extract=False,df=False,retry=0,verbose=0):
    """
    Particle data

    Arguments:
    subhalo : subhaloID
    sim : simulation (default ("TNG50-1")
    snapnum : snapshot number (default 99)
    cutout_request : requested particle properties, 
            e.g. 
            {'stars':'Coordinates,Velocities,Masses,GFM_StellarFormationTime',
             'gas':'Coordinates,Velocities,Masses,Density,InternalEnergy',
             'dm':'Coordinates'}
        
    PartType : particle type number (default 4 for stars)
    params : parameters to study (default "Coordinates") for single PartType
    save_dir : directory where particle data is stored (default ".")
    extract : extract from TNG server? (default False)
    df: convert output to dataframe 
    verbose : verbosity (default 0)

    Returns dict of particle data (limited to one PartType THSI IS A BUG!)

    Author: Gary Mamon (gam AAT iap.fr), inspired by TNG guidelines
    with help from Houda Haidar
    """
    if (retry == 1) & (verbose > 0):
        print("retrying ...")
    if save_dir is None:
        save_dir = \
            home + "/SIMS/TNG/" + sim + "/snapshots/" + str(snapnum) + "/subhalos/" + str(subhalo) 
    try:
        os.makedirs(save_dir)
    except:
        # directory already exists
        pass
    if extract: 
        if (cutout_request is None) & (PartType is not None):
            # extract specific particle parameter values and save to disk
            paramsString = ','.join(params)
            if verbose > 0:
                print("paramsString=",paramsString)
            cutout_request = {Type2Str(PartType):paramsString}
        if verbose > 0:
            print("extracting for subhalo",subhalo,"with cutout_request=",
                  cutout_request)
        saveParticles(subhalo,sim=sim,snapnum=snapnum,
                      cutout_request=cutout_request,
                      save_dir=save_dir,verbose=verbose)

    # read file of particle parameter values
    filename = save_dir + "/cutout_" + sim + "_" + str(snapnum) + "_" + str(subhalo) + ".hdf5"
    if verbose > 0:
        print("extracting data from", filename,"...")
    try:
        f = h5py.File(filename,'r')
    except:
        print("cannot open file",filename,"..., extracting from TNG database...")
        # extract from TNG database
        return particles(subhalo,sim=sim,snapnum=snapnum,PartType=PartType,
                         cutout_request=cutout_request,params=params,
                  save_dir=save_dir,extract=True,retry=1,verbose=verbose)
        # saveParticles(subhalo,sim=sim,snapnum=snapnum,request=params)
    if verbose > 0:
        print("f.keys()=",f.keys())
    if params is None:
        return f

    groupname= "PartType" + str(PartType)
    if groupname not in f:
        print("groupname",groupname,"not in f")
        print("f['Header'].keys() = ",f['Header'].keys())
        return {}
    data = {}
    if isinstance(params,str):
        if params in f[groupname]:
            data = f[groupname][params]
        else:
            print("parameter",params,"not found in",groupname)
            return {}      
    else:
        for i, param in enumerate(params):
            if param in f[groupname]:
                data[param] = f[groupname][param][:]
            else:
                print("parameter",param,"not found in",groupname + 
                      ', re-extracting...')
                # print("choices are",f[groupname].keys())
                p = particles(subhalo,sim=sim,snapnum=snapnum,PartType=PartType,
                          params=params, save_dir=save_dir,extract=True,df=df,
                             retry=retry,verbose=verbose)
                return p
    if df:
        if 'Coordinates' in data.keys():
            data['x'] = data['Coordinates'][:,0]
            data['y'] = data['Coordinates'][:,1]
            data['z'] = data['Coordinates'][:,2]
            del data['Coordinates']
        if 'Velocities' in data.keys():
            data['vx'] = data['Velocities'][:,0]
            data['vy'] = data['Velocities'][:,1]
            data['vz'] = data['Velocities'][:,2]   
            del data['Velocities']
        return pd.DataFrame.from_dict(data)
    else:
        return data
    f.close()

def ConvertDict(dict,df=False,verbose=0):
    """convert dict to new dict (or pandas dataframe) so that all 2D data become 1D"""
    dictNew = {}
    for k,key in enumerate(dict):
        # skip non-arrays (e.g. count in TNG)
        if not isinstance(dict[key],np.ndarray):
            if (k==0) & (verbose > 0):
                print("not np.array...")
            continue
        if isinstance(dict[key][0],np.ndarray):
            if (k==0) & (verbose > 0):
                print("np.array...")
            keylen2 = len(dict[key][0])
            for i in range(keylen2):
                key2 = key + str(i)
                dictNew[key2] = dict[key][:,i]
        else:
            if verbose > 0:
                print("standard")
            dictNew[key] = dict[key]
    if verbose > 0:
        print("ConvertDict: df = ",df)
    if df:
        return pd.DataFrame.from_dict(dictNew)
    else:
        return dictNew
    
def ConvertHDF52df(f,avoid_keys=[None],verbose=0):
    """convert HDF5 type to dataframe
    Arguments:
        f: h5py._hl.files.File type
        avoid_keys: keys to avoid
    Returns: dataframe (with 2d entries with suffixes 0, 1, etc.)
    Author: Gary Mamon (gam AAT iap.fr) 
      using stack overflow contribution from NeStack
      (https://stackoverflow.com/questions/71388502/read-hdf5-file-created-with-h5py-using-pandas)
      """
     # initialize dict
    dictionary = {}
    
    # loop over dict keys
    for key in f.keys():
        if key in avoid_keys:
            continue
        if verbose>0:
            print(key)
        ds_arr = f[key][()]   # returns as a numpy array
        dictionary[key] = ds_arr # appends the array in the dict under the key
        
    # convert dict to dataframe handling 2D entries
    df = ConvertDict(dictionary,df=True,verbose=verbose)
    return df

def getsubhaloid99(subhalo,simulation="TNG50-1",snapnum=None,verbose=0):
    if simulation.find("TNG") != 0:
        raise ValueError("cannot run getsubhaloid99 for non-TNG simulation... use getsubhaloid_z0")
    if verbose > 0:
        print("getsubhaloid99: snapnum=",snapnum,"subhalo=",subhalo)
    if snapnum is None:
        raise ValueError("snapnum must be given (integer from 0 to 98)")
    elif snapnum == 99:
        return subhalo
    elif snapnum < 99:
        subhalodesc = getsubhalo(subhalo,simulation=simulation,snapnum=snapnum
                                 ,parameter="desc_sfid")
        if subhalodesc==-1:
            print("subhalo",subhalo,"at snapnum",snapnum,"has no descendant")
            return -1
        else:
            return getsubhaloid99(subhalodesc,simulation=simulation,snapnum=snapnum+1,
                                 verbose=verbose)

def getsubhaloid_z0(subhalo,simulation="TNG50-1",snapnum=None,verbose=0):
    """get z=0 subhaloid
    arguments:
        subhalo: subhalo ID
        simulation
        snapnum    
    returns subhalo id at final snapnum
    author: Gary Mamon (gam AAT iap.fr)
        """
    snapnumz0 = snapofz(0,simulation=simulation)
    if verbose > 0:
        print("snapnum=",snapnum,"subhalo=",subhalo)
    if snapnum is None:
        raise ValueError("snapnum must be given (integer from 0 to 98)")
    elif snapnum == snapnumz0:
        return subhalo
    elif snapnum < snapnumz0:
        subhalodesc = getsubhalo(subhalo,simulation=simulation,snapnum=snapnum
                                 ,parameter="desc_sfid")
        if subhalodesc==-1:
            print("subhalo",subhalo,"at snapnum",snapnum,"has no descendant")
            return -1
        else:
            return getsubhaloid_z0(subhalodesc,simulation=simulation,snapnum=snapnum+1,
                                 verbose=verbose)

def SubhaloCenswGroupParams(paramsSubhalos=None,paramsHalos=None,sim="TNG50-1",
                            snapnum=99,verbose=0):
    """Produce dataframe with selected parameters from both subhalo centrals and Groups

    Parameters:
        paramsSubhalos: list of Subhalo table keys (None for all)
        paramsHalos: list of Group table keys (None for all)
        sim: TNG simulation (default: "TNG50-1")
        snapnum: snap number (default 99)
        verbose: verbosity (default 0 for no special output)

    Returns: merged dataframe

    Author: Gary Mamon (with help grom Houda Haidar)
    """
    basePath = home + "/SIMS/TNG/" + sim + "/output"
    
    # subhalos
    print("paramsSubhalos is first",paramsSubhalos)
    if 'SubhaloGrNr' not in paramsSubhalos:
        paramsSubhalos.append('SubhaloGrNr')
    if verbose > 0:
        print("subhalos...")
        print("paramsSubhalos=",paramsSubhalos)
    subs = il.groupcat.loadSubhalos(basePath=basePath,snapNum=snapnum,fields=paramsSubhalos)
    if verbose > 0:
        print("converting to df...")
    dfSubs = ConvertDict(subs,df=True)
    
    # groups
    print("paramsHalos is first",paramsHalos)
    if 'GroupFirstSub' not in paramsHalos:
        paramsHalos.append('GroupFirstSub')
    if verbose > 0:
        print("groups...")
        print("paramsHalos=",paramsHalos)
    groups = il.groupcat.loadHalos(basePath=basePath,snapNum=snapnum,fields=paramsHalos)
    if verbose > 0:
        print("converting to df...")
    dfGroups = ConvertDict(groups,df=True)
    dfGroupsClean = dfGroups.loc[dfGroups.GroupFirstSub >= 0]
    # restrict subhalos to the centrals in dfGroups
    dfSubs_tmp = dfSubs.iloc[dfGroupsClean.GroupFirstSub]
    
    # merge
    if verbose > 0:
        print("merging...")
    dfm = pd.merge(dfSubs_tmp,dfGroupsClean,how='inner',left_on='SubhaloGrNr',right_index=True)
    return dfm

def indices(dict,param,valuemin,valuemax):
    """Convert dict to dataframe    
    returns indices as numpy array
    Author: Gary Mamon (gam AAT iap.fr) and Yuankang Liu (yuankang.liu AAT gmail.com)"""
    df = ConvertDict(dict,df=True)
    return df.index[(df[param]>=valuemin) & (df[param]<=valuemax)].values

def BoxSize(sim='TNG50-1'):
    """
    box size of simulation (Mpc/h)
    argument: simulation (default: 'TNG50-1')
    author: Gary Mamon (gam AAT iap.fr)
    """
    return getsim(sim)['boxsize']

def FixPeriodic(dx,sim='TNG50-1'):
    """
    Handle periodic boundary conditions
    Arguments:
        dx: difference in positions (in ckpc/h)
        sim: simulation (default "TNG50-1")

    Returns: dx corrrected for periodic box (in ckpc/h)
    Author: Gary Mamon (gam AAT iap.fr)
    """
    if sim=='TNG300-1':
    	L = 205000.0
    elif sim == 'TNG50-1':
        L = 35000.0
    else:
    	L = BoxSize(sim) # BoxSize is in kpc/h
    # L = getsim(sim)['boxsize']
    dx = np.where(dx>L/2,dx-L,dx)
    dx = np.where(dx<-L/2,dx+L,dx)
    return dx

def merge_groups_subhalos(df_subs,df_groups,sim='TNG50-1',snapnum=99):
    """merge subhalos and groups

        arguments:
            df_subs: dataframe of subhalos (containing 'SubhaloGrNr')
            df_groups: dataframe of groups 
                (containing 'GroupFirstSub' to identify centrals)
            sim: simulation (default: 'TNG50-1')
            snapnum: snapshot number (default: 99)

        returns merged dataframe with extra columns:
            cen_id: subhaloID of central
            is_central: True if subhalo is central, else False
            lM_200: log_10(M_200_crit/M_Sun)
    
        author: Gary Mamon (gam AAT iap.fr)
            """
    df_groups['group_id'] = np.arange(len(df_groups)).astype(int)
    df_merge = df_subs.merge(df_groups,how='inner',left_on='SubhaloGrNr',right_on='group_id')
    df_merge['cen_id'] = df_merge['GroupFirstSub']
    df_merge['is_central'] = np.where(df_merge.cen_id==df_merge.SubhaloID,True,False)
    df_merge['lM_200'] = 10 - lh + np.log10(df_merge.Group_M_Crit200)
    return df_merge

def merge_groups_subhalos_from_scratch(sim='TNG50-1',snapnum=99,lmmin=7,
                                       params=None,verbose=0):
    # # extract interesting parameters for all subhalos
    # params=['SubhaloMass','SubhaloMassType','SubhaloMassInRadType',
    #         'SubhaloPos','SubhaloVel','SubhaloHalfmassRadType',
    #         'SubhaloGrNr','SubhaloFlag','SubhaloGasMetallicity','SubhaloLenType',
    #         'SubhaloSFR','SubhaloSFRinRad','SubhaloSpin','SubhaloStarMetallicity',
    #         'SubhaloStellarPhotometrics','SubhaloVmax','SubhaloVmaxRad']
    
    z = zofsnap(snapnum,sim=sim)
    a = 1/(1+z)
    if verbose > 0:
        print("a=",a)
    
    if verbose > 0:
        print("reading subhalos...")
    basePath = data_root_dir + sim + '/output/'
    subhalos = il.groupcat.loadSubhalos(basePath=basePath,snapNum=99,
                                        fields=params)
    
    if verbose > 0:
        print("converting to subhalo dataframe...")
    df = ConvertDict(subhalos,df=True)
    
    # remove low-stellar-mass galaxies (first save SubhaloID and build lM_stars)
    if verbose > 0:
        print("filtering out low stellar mass subhalos...")
    df['SubhaloID'] = np.arange(len(df)).astype(int)
    df['lM_stars'] = 10 - lh + np.log10(df.SubhaloMassType4)
    df = df.loc[df.lM_stars > lmmin]
    
    if verbose > 0:
        print("new column names...")
    df['x'] = df.SubhaloPos0
    df['y'] = df.SubhaloPos1
    df['z'] = df.SubhaloPos2
    df['v_x'] = df.SubhaloVel0
    df['v_y'] = df.SubhaloVel1
    df['v_z'] = df.SubhaloVel2
    df['lM_gas'] = 10 - lh + np.log10(df.SubhaloMassType0)
    df['lM_DM'] = 10 - lh + np.log10(df.SubhaloMassType1)
    df['lM_stars'] = 10 - lh + np.log10(df.SubhaloMassType4)
    df['lM_BH'] = 10 - lh + np.log10(df.SubhaloMassType5)
    df['lM_tot'] = 10 - lh + np.log10(df.SubhaloMass)
    df['lM_gas_rh'] = 10 - lh + np.log10(df.SubhaloMassInHalfRadType0)
    df['lM_DM_rh'] = 10 - lh + np.log10(df.SubhaloMassInHalfRadType1)
    df['lM_stars_rh'] = 10 - lh + np.log10(df.SubhaloMassInHalfRadType4)
    df['lM_gas_2rh'] = 10 - lh + np.log10(df.SubhaloMassInRadType0)
    df['lM_DM_2rh'] = 10 - lh + np.log10(df.SubhaloMassInRadType1)
    df['lM_stars_2rh'] = 10 - lh + np.log10(df.SubhaloMassInRadType4)
    df['lM_tot_2rh'] = 10 - lh + np.log10(df.SubhaloMassInRad)
    df['lSFR'] = np.log10(df.SubhaloSFR)
    df['lSFR_rh'] = np.log10(df.SubhaloSFRinHalfRad)
    df['lSFR_2rh'] = np.log10(df.SubhaloSFRinRad)
    df['lsSFR'] = df['lSFR'] - df['lM_stars']
    df['lsSFR_rh'] = df['lSFR_rh'] - df['lM_stars_rh']
    df['lsSFR_2rh'] = df['lSFR_2rh'] - df['lM_stars_2rh']
    df['lM_BH'] = 10 - lh + np.log10(df.SubhaloMassType5)
    df['rh_gas'] = df.SubhaloHalfmassRadType0/h
    df['rh_DM'] = df.SubhaloHalfmassRadType1/h
    df['rh_stars'] = df.SubhaloHalfmassRadType4/h
    df['Mg'] = df.SubhaloStellarPhotometrics4
    df['Mr'] = df.SubhaloStellarPhotometrics5
    df['Mi'] = df.SubhaloStellarPhotometrics6
    
    if verbose >= 2:
        print(df[['SubhaloID','SubhaloMassType4','lM_stars']].loc[1000:1020])
    # drop transformed columns
    if verbose > 0:
        print("dropping pre-transformed columns...")
    df.drop(columns=['SubhaloSFR','SubhaloSFRinHalfRad','SubhaloSFRinRad']
            ,inplace=True)
    for j in [0,1,2]:
        df.drop(columns=['SubhaloPos' + str(j)],inplace=True)
        df.drop(columns=['SubhaloVel' + str(j)],inplace=True)
    for j in [0,1,4,5]:
        df.drop(columns=['SubhaloMassType' + str(j)],inplace=True)
        df.drop(columns=['SubhaloMassInHalfRadType' + str(j)],inplace=True)
        df.drop(columns=['SubhaloMassInRadType' + str(j)],inplace=True)
        df.drop(columns=['SubhaloHalfmassRadType' + str(j)],inplace=True)
    for j in [4,5,6]:
        df.drop(columns=['SubhaloStellarPhotometrics' + str(j)],inplace=True)
    
    # extract groups
    if verbose > 0:
        print("reading groups...")
    groups = il.groupcat.loadHalos(basePath=basePath50,snapNum=99)
    
    if verbose > 0:
        print("converting to groups dataframe...")
    df_groups = ConvertDict(groups,df=True)
    
    if verbose > 0:
        print("new column names...")
    df_groups['group_id'] = np.arange(len(df_groups)).astype(int)
    df_groups['lM_200'] = 10 - lh + np.log10(df_groups.Group_M_Crit200)
    df_groups['r_200'] = df_groups.Group_R_Crit200/h
    df_groups['x_group'] = df_groups.GroupPos0
    df_groups['y_group'] = df_groups.GroupPos1
    df_groups['z_group'] = df_groups.GroupPos2
    df_groups['v_x_group'] = df_groups.GroupVel0 / a
    df_groups['v_y_group'] = df_groups.GroupVel1 / a
    df_groups['v_z_group'] = df_groups.GroupVel2 / a
    
    # drop transformed columns
    if verbose > 0:
        print("dropping pre-transformed columns...")
    df_groups.drop(columns=['Group_R_Crit200','Group_M_Crit200',
                            'Group_R_TopHat200','Group_M_TopHat200',
                            'GroupPos0','GroupPos1','GroupPos2',
                            'GroupVel0','GroupVel1','GroupVel2'],
                   inplace=True)

    # merge groups and selected subhalos
    if verbose > 0:
        print("merging groups and selected subhalos...")
    df_merge = df.merge(df_groups,how='inner',
                        left_on='SubhaloGrNr',right_on='group_id')
    
    # centrals
    df_merge['cen_id'] = df_merge['GroupFirstSub']
    df_merge['is_central'] = np.where(df_merge.cen_id==df_merge.SubhaloID,True,False)
    df_merge.drop(columns=['GroupFirstSub'],inplace=True)
        
    # write to disk in FITS
    df2fits(df_merge,sim=sim,file='subhalos_groups',verbose=verbose)
    
def componentFractions_from_logMs(df,suf='_2rh'):
    if suf is None:
        suf = ''
    M_gas = 10**df['lM_gas' + suf].values
    M_DM = 10**df['lM_DM' + suf].values
    M_stars = 10**df['lM_stars' + suf].values
    M_BH = 10**df['lM_BH'].values
    M_tot = M_gas + M_DM + M_stars + M_BH
    return np.array([M_gas,M_DM,M_stars,M_BH])/M_tot
    
def RadialCoordinate(pos,posRef,sim='TNG50-1',verbose=0):
    """Radial coordinate, correcting for periodic box
    Author: Gary Mamon (gam AAT iap.fr)"""
    pos = np.array(pos)
    if pos.shape[-1] != 3:
        raise ValueError("pos must have a shape of 3 or N,3")
    if len(posRef) != 3:
         raise ValueError("posRef must have a length of 3")       
    dpos = []
    for i in range(3):
        if pos.ndim == 2:
            pos_tmp = pos[:,i]
        else:
            pos_tmp = pos[i]
        dpos_raw = pos_tmp - posRef[i]
        # correct for possible system at edge of box (periodic boundary conditions)
        dpos.append(FixPeriodic(dpos_raw,sim=sim))
    dpos = np.array(dpos)
    return np.sqrt(dpos[0]*dpos[0] + dpos[1]*dpos[1] + dpos[2]*dpos[2])

def VelSubhaloInGroup(velSub,velGroup,sim='TNG50-1',snap=99,verbose=0):
    """Relative peculiar velocity of subhalo in group frame
    Author: Gary Mamon (gam AAT iap.fr)"""
    z = zofsnap(snap,sim=sim)
    a = 1/(1+z)
    return np.array(velSub) - np.array(velGroup)/a

def VelPartInSubhalo(velPart,velSub,sim='TNG50-1',snap=99):
    """Relative peculiar velocity of particle in subhalo frame
    Author: Gary Mamon (gam AAT iap.fr)"""
    z = zofsnap(snap,sim=sim)
    a = 1/(1+z)
    return np.sqrt(a)*np.array(velPart) - np.array(velSub)
    
def tickInterval(deltax,numticks=5):
    """Tick interval given tick limits
    Author: Gary Mamon, gam AAT iap.fr"""
    n = [1,2,3,4,5]
    logn = np.log10(n)
    dx = deltax/numticks
    logdx = np.log10(dx)
    logdx_mantissa = logdx - np.floor(logdx)
    logdx_exponent = logdx - logdx_mantissa
    absdiff = np.abs(logn - logdx_mantissa)
    return n[np.argmin(absdiff)]*10**logdx_exponent

def CleanTicks(xmin,xmax,dx=None,numticks=5):
    """Return optimal ticks given min and max values
    author: Gary Mamon, gam AAT iap.fr """
    if dx is None:
        dx = tickInterval(xmax-xmin,numticks=numticks)
    ticks = dx*np.arange(np.floor(xmin/dx),np.floor(xmax/dx)+1)
    ticks = ticks[(ticks>= xmin) & (ticks <= xmax)]
    return ticks

def VelocityField(subhaloid,sim='TNG50-1',snap=99,roverrhalf=2,rshell=0,droverrhalf=0.2,
                  drshell=0,partType=0,axis='xy',
                  save_dir_root='SIMS/TNG',savefig=None,
                  alpha=1,title='auto',verbose=0):
    """

    Parameters
    ----------
    subhaloid: TYPE
        subhalo id
    sim : TYPE, optional
        simulation. The default is 'TNG50-1'.
    snap : TYPE, optional
        snapshot number. The default is 99.
    roverrhalf : TYPE, optional
        r_shell / r_half. The default is 2. (0 to use rshell)
    rshell : TYPE, optional
        radius of shell (in ckpc/h). The default is 0. (then uses roverrhalf)
    droverrhalf : TYPE, optional
        thickness of shell (in half-mass radii). The default is 0.2.
    drrshell : TYPE, optional
        thickness of shell (in ckpc/h). The default is 0. (then uses roverrhalf)
    partType : TYPE, optional
        particle type (0 for gas, 1 for dark matter, 4 for stars). The default is 0.
    axis: 
        sky axis, default: 'xy'
    savefig:
        'auto': automatic filename
        otherwise filename,
        default None
    alpha:
        opacity for plot (default 1)
    title:
        plot title (default false)
            'auto': automatic title
            otherwise title
            default None
    verbose:
        verbosity: 0 = no debugging output

    Returns
    -------
    plot

    """
    # global df, parts
    # Particles
    if verbose > 0:
        print("Extracting particles...")
    save_dir = os.getenv("HOME") + '/' + save_dir_root + '/' + sim
    parts = particles(subhaloid,sim,snapnum=snap,params=['Coordinates','Velocities','Masses'],
                  PartType=partType,save_dir=save_dir,verbose=0,extract=True)
    
    # Subhalo
    if verbose > 0:
        print("Extracting subhalo...")
    sub = getsubhalo(subhaloid,simulation=sim,snapnum=snap)
    sub['Pos'] = np.array([sub['pos_x'],sub['pos_y'],sub['pos_z']])
    tmpvel = [sub['vel_x'],sub['vel_y'],sub['vel_z']]
    sub['Vel'] = [sub['vel_x'],sub['vel_y'],sub['vel_z']]
    
    # Group
    if verbose > 0:
        print("Extracting group...")
    group_id = sub['grnr']
    group = gethalo(group_id,simulation=sim,snapnum=snap)
    
    # bulk velocity in Group
    vsub = sub['Vel']
    vgroup = group['GroupVel']
    sub['Vrel'] = VelSubhaloInGroup(sub['Vel'],group['GroupVel'],sim=sim,verbose=verbose)
    
    # radii 
    parts['r'] = RadialCoordinate(np.array(parts['Coordinates']),np.array(sub['Pos']),sim=sim,
                                  verbose=verbose)
        
    # relative velocities
    parts['Vrel'] = VelPartInSubhalo(parts['Velocities'],vsub,sim=sim,snap=snap)
        
    # convert to dataframe for easier selection
    df = ConvertDict(parts,df=True)
    if verbose > 1:
        print("df.keys = ",df.keys())
    
    # half-mass radius of partType
    if roverrhalf > 0:
        # mass weighted median
        rhalf = ws.weighted_median(df.r,df.Masses)
        rshell = rhalf * roverrhalf
        drshell = droverrhalf * rhalf
        
    # select shell particles
    dfshell = df.loc[np.abs(df.r-rshell) < drshell]
    
    # statistics
    dsw = DescrStatsW(df.r.values,weights=df.Masses.values)
    
    # velocity statistics
    # vrelmean = np.zeros(3)
    # for i in range(3):
    #     vrelmean[i] = np.average(dfshell.Vrel.values,weights=dfshell.Masses.values)
    vrelmean = np.average(dfshell.Vrel.values,axis=1,weights=dfshell.Masses.values)
    cos_vrel_vbulk = 1 - distance(vrelmean,np.array(sub['Vrel']))
    rsubinGroup = RadialCoordinate(sub['Pos'],group['GroupPos'])
    cos_ringroup_vrel = 1 - distance(vrelmean,rsubinGroup)
    vrel_r = np.sum(np.array(parts['Coordinates']),np.array(parts['Vrel'])) / parts['r']
    sigma_v_shell = np.std(parts['Vrel'],axis=1)
    vr_over_sigma = vrel_r/sigma_v_shell
    
    if verbose > 0:
        print("plotting...")
    if title == 'auto':
        title = sim + ' snap ' + str(snap) + ' subhalo ' + str(subhaloid) + ' gas'
    plotVelField(df,dfshell,sim=sim,snap=snap,alpha=alpha,title=title,
                 savefig=savefig)
    return cos_vrel_vbulk, vr_over_sigma, cos_ringroup_vrel
    
def plotVelField(df,dfshell,sim="TNG50-1",snap=99,axis='xy',alpha=1,title=None,
                 savefig=None):
    # plot velocity field
    if axis == 'xy':
        x = df.Coordinates0 / 1000
        y = df.Coordinates1 / 1000
        vx = df.Vrel0
        vy = df.Vrel1
        xs = dfshell.Coordinates0 / 1000
        ys = dfshell.Coordinates1 / 1000
        vxs = dfshell.Vrel0
        vys = dfshell.Vrel1
    elif axis == 'yz':
        x = df.Coordinates1 / 1000
        y = df.Coordinates2 / 1000
        vx = df.Vrel1
        vy = df.Vrel2
        xs = dfshell.Coordinates1 / 1000
        ys = dfshell.Coordinates2 / 1000
        vxs = dfshell.Vrel1
        vys = dfshell.Vrel2
    elif axis == 'xz':
        x = df.Coordinates0 / 1000
        y = df.Coordinates2 / 1000
        vx = df.Vrel0
        vy = df.Vrel2
        xs = dfshell.Coordinates0 / 1000
        ys = dfshell.Coordinates2 / 1000
        vxs = dfshell.Vrel0
        vys = dfshell.Vrel2
    else:
        raise ValueError("Cannot recognize axis = " + axis)
    xlabel = '$' + axis[0] + '$'
    ylabel = '$' + axis[1] + '$'
    # plt.figure(figsize=[5,5])
    plt.quiver(x,y,vx,vy,color='gray')
    plt.quiver(xs,ys,vxs,vys,color='r',alpha=alpha)
    plt.xlabel(xlabel + ' (cMpc/$h$)')
    plt.ylabel(ylabel + ' (cMpc/$h$)')
    ax = plt.gca()
    ax.set_aspect(1)
    ax.ticklabel_format(useOffset=False)
    partTypes = ['gas','dark matter', None, None, 'stars']
    if title is not None:
        plt.title(title)
    # plt.axis('scaled')
    if savefig == 'auto':
        plt.savefig(sim + '_' + str(snap) + '_' + str(subhalo) + '_vfield_' + axis + '.pdf')
    elif savefig is not None:
        plt.savefig(savefig + '.pdf')
        


def PlotHistory(y,x,subhaloid=None,sim=None,snapmin=None,snapmax=99,
                param=None,param2=None,yscale=None,
                labels=None,colors=None,ylabel=None,marker=None,legend=None,
                savefig=False,verbose=0):
    """Plot TNG history
    Author: Gary Mamon, gam AAT iap.fr"""
    snaps = range(snapmin,snapmax+1)
    snaps_z_all = np.array(snapofz('all'))
    z = snaps_z_all[np.isin(snaps_z_all[:,0],snaps)][:,1]
    t = AgeUniverse(Omegam0,h,z)
    print("len(t)=",len(t))
    print("len(y)=",len(y))
    
    fig = plt.figure()
    
    # first horizontal axis: times (age Universe)
    ax = fig.add_subplot(111)
    # ax.plot(t,y,marker='o')
    if legend:
        if param2 is not None:
            if isinstance(param2,Iterable): # param2 is a list or tuple or array
                for i, par2 in enumerate(param2):
                    ax.plot(t,y[:,i],label=labels[par2],
                             color=colors[par2],marker=marker)
            else: # param2 is a scalar
                par2 = param2
                ax.plot(t,y[:],color=colors[par2],marker=marker)
                legend = False
        else: # param2 is not given
            for i in range(len(labels)):
                if ((param.find("Type") > 0) & (i in [2,3])):
                    continue
                ax.plot(t,y[:,i],label=labels[i],
                         color=colors[i],marker=marker)
        # check again legend, which may have been turned off in the block above
        if legend:
            if ((len(labels) <= 4) or (param.find("Type") > 0)):
                fsize = 16
            else:
                fsize = int(32/np.sqrt(len(labels)))
            print("len(labels)=",len(labels))
            print("legend fontsize = ",fsize)
#             if param == "SubhaloStellarPhotometrics":
#                 print("small legend?")
#                 plt.legend(fontsize="small")
#             else:
            plt.legend(fontsize=fsize)
    elif param in ["xy","xz","yz"]:
        ax.plot(x,y,marker=marker)
        ax.scatter([x[0]],[y[0]],marker='o',c='k',s=150)
        ax.text(x[0],y[0],"99",c='orange',fontsize=10,
                 horizontalalignment='center',verticalalignment='center')
    else: # Type is not in parameter name
        ax.plot(t,y,marker=marker)
        if param in ["mass","Mass"]:
            plt.legend()
    if ylabel:
        ax.set_ylabel(ylabel)
    if verbose > 0:
        print("yscale=",yscale)
    ax.set_yscale(yscale)
    ax.grid(True)
    if subhaloid:
        plt.title(sim + "  subhalo " + str(subhaloid), fontsize=18)
    
    # 2nd horiontal axis (top of box): redshifts
    ax2 = ax.twiny()
    zticks = CleanTicks(0,np.max(z))
    zticks = np.array([0,0.2,0.5,1,2,5])
    if verbose > 0:
        print("zticks=",zticks)
    tticks = AgeUniverse(Omegam0,h,zticks)
    zticks = zticks.tolist()
    tticks = tticks.tolist()
    zlabels = ['%.1f' % z for z in zticks]
    axlims = ax.get_xlim()
    ax2.set_xlim(axlims)
    ax2.set_xticks(tticks)
    ax2.set_xticklabels(zlabels)
    ax2.set_xlabel('redshift')
    ax.set_xlabel('time (Gyr)')
    # plt.grid(None)
    # ax.grid(True)
    ax2.minorticks_off()
    ax2.grid(None)
    
    # 3rd horizonatal axis: snapnums
    ax3 = ax.twiny()
    # Add some extra space for the second axis at the bottom
    fig.subplots_adjust(bottom=0.2)
    # Move twinned axis ticks and label from top to bottom
    ax3.xaxis.set_ticks_position("bottom")
    ax3.xaxis.set_label_position("bottom")
    # Offset the twin axis below the host
    ax3.spines["bottom"].set_position(("axes", -0.25))
    snaps4labels = CleanTicks(snapmin,snapmax)
    if 99 not in snaps4labels:
        snaps4labels = np.append(snaps4labels,99)
    zsnaps = snaps_z_all[np.isin(snaps_z_all[:,0],snaps4labels)][:,1]
    tticks2 = AgeUniverse(Omegam0,h,np.array(zsnaps))
    snaplabels = ['%d' % s for s in snaps4labels]
    ax3.set_xlim(axlims)
    ax3.set_xticks(tticks2)
    ax3.set_xticklabels(snaplabels)
    ax3.set_xlabel('snapnum')
    ax3.grid(None)
    ax3.minorticks_off()
    
    # finalize plot
    if savefig:
        plt.tight_layout()
        plt.savefig(savefig + ".pdf")
    else:
        plt.show()
        
    return t
    
def MMP(subhaloid,snapmax=99,snapmin=0,sim='TNG50-1',data_root_dir=None,datafileprefix=None,halo=False,verbose=0):
    if data_root_dir is None:
        data_root_dir = os.getenv("HOME") + "/SIMS/TNG/"
        # answer = input("Enter root directory for TNG simulations: [" + data_root_dir + "]")
        # if answer != '':
        #     data_root_dir = answer
    treeMethod='sublink_mbp'
    datadir = data_root_dir + sim + "/output/"
    if verbose > 0:
        print("History: datadir=",datadir)
    if datafileprefix is None:
        if halo:
            # extract halo URL
            if verbose> 0:
                print("extracting halo URL...")
        else:
            # extract subhalo URL 
            if verbose > 0:
                print("extracting subhalo URL...")
            sub = getsubhalo(subhaloid,simulation=sim,snapnum=snapmax,
                             save_dir=datadir,verbose=verbose)
            if verbose > 1:
                print ("sub=",sub)
            progurl = sub['related']['sublink_progenitor']
            if progurl is None:
                print("no progenitor for subhalo", subhaloid)
                return
            # extract tree of main progenitor
            datafile = get(sub['trees'][treeMethod],save_dir=datadir)
    elif datafileprefix in [0,'a',"auto"]:
        datafile = datadir + "sublink_mpb_" + str(subhaloid) + ".hdf5"
    else:
        datafile = datadir + datafileprefix + ".hdf5" 
    if verbose > 0:
        print("cwd=",os.getcwd())
        os.system("ls -l " + datafile)
    
    # extract values
    if verbose > 0:
        print("extracting parameters from tree in file " + data_root_dir + datafile + "...")
    try:
        f = h5py.File(datafile,'r')
        if verbose > 0:
            print("done")
        
    except:
        # recursive relaunch with datafileprefix=None to force read from TNG database
        print("file ", datafile, " not found on disk, trying TNG database...")
        MMP(subhaloid,snapmax=snapmax,snapmin=snapmin,sim=sim,
            data_root_dir=data_root_dir,datafileprefix=None,halo=halo,verbose=verbose)

def HistoriesToDisk(subhaloid,sim='TNG50-1',treeMethod='sublink_mpb',
                    snapmax=99,verbose=0):
    """Save subhalo histories to disk
    returns nothing
    """
    data_dir = os.getenv("HOME") + "/SIMS/TNG/" + sim + '/output/'
    sub = getsubhalo(subhaloid,simulation=sim,snapnum=snapmax,
                      save_dir=data_dir,verbose=verbose)
    if verbose >= 2:
        print ("after getsubhalo: sub=",sub)
    progurl = sub['related']['sublink_progenitor']
    if progurl is None:
        print("no progenitor for subhalo", subhaloid)
        return
    # extract tree of main progenitor
    if verbose > 0:
        print("extracting tree of main progenitor with method",treeMethod)
        print("... and saving tree to", data_dir)
    datafile = get(sub['trees'][treeMethod],save_dir=data_dir)
    if verbose > 0:
        print("HistoriesToDisk: datafile=",datafile)

def History(subhaloid,param=None,param2=None,sim='TNG50-1',treeMethod='sublink_mpb',
                snapmin=None,snapmax=99,snapnum=99,
                snapminPlot=None,
                extract=True,datafileprefix='auto',plot=True,
                xaxis='time', topxaxis=None,shadex=None,
                xlabel='snapnum',ylabel=None,yscale='log',ylims=None,relative=True,
                h=0.6774,
                data_root_dir=None,halo=False,yCen=0.01,cen=False,
                flag=None,labelsize=None,
                legend=True,legendsize=None,legendloc='best',abbrevLegend=False,
                savefig=False,usetex=True,marker=None,peris=None,verbose=0):
    """Extract and optionally plot evolution of subhalo parameter
    arguments:
        subhaloid
        param=param, where param is one of the subhalo attributes, or shortcuts:
            'pos': x, y, z evolution
            'vel': v_x, v_y, v_z evolution
            'posrel': x, y, z evolution relative to main prog of final central
            'velrel': v_x, v_y, v_z evolution relative to main prog of final central
            'mass': total mass of subhalo
            'masstype': gas, DM, stars and BH mass (total for subhalo)
            'mass2rhtype': gas, DM, stars and BH mass within 2_rh
            'SFR': SFR within 2 r_h
            'sSFR': sSFR = SFR/Mass_Stars, both within 2 r_h
            'fgas': gas fraction (in 2 r_h,stars)
            'gass/stars': gas/stars (in 2 r_h,stars)
            'sBHMdot': specific black hole accretion rate = Mdot_BH/M_BH
            'r' or 'sep': distance to main progenitor of final central
            'r_r200': both distance to main prog of final central and R_200c of that central
            'r/r200': R/R_200c relative to the main progenitor of the final central
            'masstype_r':  like masstype and 'r_r200'
            'mass2rhtype_r':  like masstype2rh and 'r_r200'
            'masstype_r/r200':  like masstype and 'r/r200'
            'mass2rhtype_r200':  like masstype2rh and 'r/r200'
            'masstype_r_rh': like masstype and 'r' and SubhaloRadType
        param2=num, where num is the type for Mass (e.g. 4 for Stars)
        snapmax: highest (latest, i.e. base) snapnum (default 99)
        snapmin: lowest (earliest) snapnum (default None last plus one)
        snapnum: reference snapnum for subhaloid (default snapmax)
        extract: True if extract from TNG
        datafileprefix: data file prefix (0 or "auto" for sublink_mpb_[subhaloid], None to extract (default))
        halo: if True, assume subhaloID is really haloID [default False]
        yCen: radial position of center in plots of r/r_vir [default 0.01]
        cen: True if central (to avoid infinite recursion), else False [default False]
        plot: do-plot? [boolean]
        xaxis: time x-axis: 'time', 'tlook', 'z'
        topxaxis: time upper x-axis: 'time', 'tlook', 'z'
        shadex: [min, max, color] (default None)
        xlabel: x-label
        ylabel: y-label
        yscale: scale of y axis
        ylims: [ymin,ymax]
        flag: 0 (bad) or 1 (good) for title
        savefig: output file of figure
        usetex: use TeX fonts?
        marker: marker (default None)
        peris: mark pericenters by vertical lines (None, 'vert-lines', 'pos')
    Author: Gary Mamon (gam AAT iap.fr)"""
    
    # dimensionless Hubble constant
    if h is None: 
        h = 0.6774
    
    if param == 'r/rvir':
        param = 'r/r200'
        
    if data_root_dir is None:
        data_root_dir = os.getenv("HOME") + "/SIMS/TNG/"
        # answer = input("Enter root directory for TNG simulations: [" + data_root_dir + "]")
        # if answer != '':
        #     data_root_dir = answer
    datadir = data_root_dir + sim + "/output/"
    if verbose > 0:
        print("History: datadir=",datadir)
    
    need_ax2 = False
    if extract: # extract from TNG web site
        if halo:
            # extract halo URL
            if verbose> 0:
                print("extracting halo URL...")
        else:
            # extract subhalo URL 
            if verbose > 0:
                print("extracting subhalo URL...")
            if snapnum < snapmax:
                subhaloid = getsubhaloid99(subhaloid,simulation=sim,snapnum=snapnum,verbose=verbose)
                if verbose > 0:
                    print("at snapnum=99, subhaloid = ",subhaloid)
            datafile = datadir + 'sublink_mpb_' + str(subhaloid) + '.hdf5'
            try:
                f = h5py.File(datafile,'r')
            except:
                HistoriesToDisk(subhaloid,sim=sim,verbose=verbose)

            if verbose > 0:
                print("datafile=",datafile)
                print("now reading from disk at",datafile),
                os.system("ls -l " + datafile)
            
            # now read from disk
            f = h5py.File(datafile,'r')

            # extract  history of z=0 central if needed
            if param != None:
                if (('_r' in param) or (param in ['r','r_over_R_Crit200','r_over_R_Crit500', 'r/r200','r/r500'])) & (not cen):
                    need_ax2 = True
                    subhaloCen = f['GroupFirstSub'][0]
                    HistoriesToDisk(subhaloCen,sim=sim,verbose=verbose)
                    datafileCen = datadir + 'sublink_mpb_' + str(subhaloCen) + '.hdf5'
                    fCen = h5py.File(datafileCen,'r')
                
    else: # read from disk
        if datafileprefix in [0,'a',"auto"]:
            datafile = datadir + "sublink_mpb_" + str(subhaloid) + ".hdf5"
        else:
            datafile = datadir + datafileprefix + ".hdf5" 
        if verbose > 0:
            print("reading data from",datafile,"...")
            print("cwd=",os.getcwd())
            os.system("ls -l " + datafile)
        
        # extract from disk
        if verbose > 0:
            print("extracting parameters from tree in file " + datafile + "...")
        try:
            f = h5py.File(datafile,'r')
            if verbose > 0:
                print("done")
                if verbose >= 2:
                    print("type(f)=",type(f))
                    print("f.keys=",f.keys())
        except:
            if verbose >= 1:
                print("file ", datafile,
                  " not found on disk, trying TNG database...")
            HistoriesToDisk(subhaloid,sim=sim,verbose=verbose)
            f = h5py.File(datafile,'r')
            
        # extract  history of z=0 central if needed
        fCen = None
        if param != None:
          if (('_r' in param) or \
            (param in ['r','r_over_R_Crit200','r_over_R_Crit500',
                     'r/r200','r/r500'])) & (not cen):
              need_ax2 = True

        if need_ax2:
             subhaloCen = f['GroupFirstSub'][0]
             # HistoriesToDisk(subCen,sim=sim,verbose=verbose)
             datafileCen = datadir + 'sublink_mpb_' + str(subhaloCen) + '.hdf5'
             
             try:
                 fCen = h5py.File(datafileCen,'r')
             except:
                 if verbose >= 1: 
                     print("file ", datafileCen, 
                       " not found on disk, trying TNG database...")
                 HistoriesToDisk(subhaloCen,sim=sim,verbose=verbose)
                 fCen = h5py.File(datafileCen,'r')
        
    if param is None:
            return f

    # extract values for z=0 central if needed

    if need_ax2:
        # subhaloID of z=0 central
        subhaloid_Cen = f['GroupFirstSub'][0]
        if verbose > 0:
            print("subhaloCen=",subhaloid_Cen)

        # restrict Cen history to length of Sat history
        posCen = fCen['SubhaloPos'][:]
        posCen = posCen[0:len(f['SubhaloPos']),:]
        R200Cen = fCen['Group_R_Crit200'][:]
        R200Cen = R200Cen[0:len(f['Group_R_Crit200'])]
        SnapCen = fCen['SnapNum']
        
    snapnums = f['SnapNum'][:]
    x = snapnums.copy()
    z = zofsnaps(x,sim=sim)
#     if snapmin is None:
#         if verbose > 0:
#             print("setting snapmin to ",snapnum[-1]-1)
#         snapmin = snapnum[-1]-1
    if verbose >= 2:
        print("first 2 SubhaloID = ",f['SubhaloID'][0:2])
        print("first 2 SubfindID = ",f['SubfindID'][0:2])

    # special treatment for specific paramters
    substr = "sSFR"
    isSFR = param.index(substr) if substr in param else -1
    # param = ConvertParam(param)
    y2 = None
    if verbose > 0:
        print("param is now", param)
    if (param[0:8] == 'masstype') or (param[0:11] == 'mass2rhtype'):
        if param[0:8] == 'masstype':
            mass = f['SubhaloMassType']
            flag_2rh = False
        else:
            mass = f['SubhaloMassInRadType']
            flag_2h = True
        y = np.log10(1e10 * mass[:,:] / h)
        # omit BHs if never there
        if y[:,5].max() < -10:
            if verbose > 0:
                print("max lM_BH=",y[:,5].max())
            # y = y[:,0:3]
            param2 = (1,0,4)
        else:
            param2 = (1,0,4,5)
        if verbose > 0:
            print("param2=",param2)
            print("shape y=",y.shape)
        yscale = 'linear'
        if ylabel is None:
            if usetex:
                if flag_2rh:
                    ylabel = "\log\,M(2\,r_{\star,\mathrm{1/2})}\ [\mathrm{M}_\odot]"
                else:
                    ylabel = "\log(M/\mathrm{M}_\odot)"
            else:
                if flag_2rh:
                    ylabel = "log mass (2 R_e) [M_sun]"
                else:
                    ylabel = "log mass [M_sun]"
                
        # case with r/r_200 plot superposed
        if (param[-7:] == '_r/r200') or (param[-2:] == '_r') or (param[-5:] == '_r_rh'):
            # align f and fCen to the same SnapNums
            # iGood 
            if verbose > 0:
                print("in '_r/r200, _r' block...")
            SnapSat = f['SnapNum']
            if len(SnapCen) != len(SnapSat):
                # itersect snap numbers
                SnapCommon = np.intersect1d(SnapSat,SnapCen)
                condSatGood = np.in1d(SnapSat,SnapCommon)
                condCenGood = np.in1d(SnapCen,SnapCommon)
                # SnapsConcat = np.concatenate((SnapSat, SnapCen))
                # SnapUnion = np.arange(SnapsConcat.min(),SnapsConcat.max()).astype(int)                
                # condSatGood = np.in1d(SnapSat,SnapCen)
                # condCenGood = np.in1d(SnapCen,SnapSat)
                if verbose >= 1:
                    print("mismatch Sat and Cen:")
                    print("len condSatGood=",len(condSatGood))
                #     print("bad iSat:")
                #     print("bad iCen")
            else:
                condSatGood = np.ones(len(SnapSat)).astype(bool)
                condCenGood = np.ones(len(SnapSat)).astype(bool)
            pos = f['SubhaloPos'][condSatGood,:]
            posCen = fCen['SubhaloPos'][condCenGood,:]
            y2 = np.zeros(len(pos[:]))
            for j in range(3): # loop over cartesian axes
                # dr = FixPeriodic(pos[:,j]-posGroup[:,j],sim)
                dr = FixPeriodic(pos[:,j]-posCen[:,j],sim)
                y2 = y2 + dr*dr
                if '00' in param:
                    ii = param.find('00')
                    Delta = param[ii-1:ii+2]
                else:
                    Delta = '200'  
            y2 = np.sqrt(y2) / (1+z) # physical separations
            paramRef = "Group_R_Crit" + Delta
            yRef = fCen[paramRef][condCenGood] / (1+z) # physical virial radius
            if param[-7:] == '_r/r200':
                y2 = y2 / yRef
            # assign non-zero value (yCen) if central
            y2 = np.where(y2 < yCen,yCen,y2)
            if verbose >= 2:
                print("pos[:,:] = ",pos[:,:]) 
                print("len y2 = ",len(y2))
    elif param in ["sBHmdot"]: # special treatment for specific black hole mass growth rate
        # use 2 R_e aperture
        mdotBH = np.asarray(f['SubhaloBHMdot'])
        massBH= np.asarray(f['SubhaloBHMass'])
        sBHMdot = mdotBH/massBH
        y = np.log10(sBHMdot)
        yscale = 'linear'
        if ylabel is None:
            if usetex:
                ylabel = "\log\,[\dot M_\mathrm{BH}/M_\mathrm{BH}]\ (\mathrm{0.978 Gyr}^{-1})"
            else:
                ylabel = "log [Mdot_BH/M_BH] [1/0.978 Gyr]"
    elif param in ["fgas"]: # special treatment for fgas 
        # use 2 R_e aperture
        massAll = f['SubhaloMassInRadType']
        massStars = massAll[:,4]*1e10
        massGas = massAll[:,0]*1e10
        y = np.log10(massGas/(massGas+massStars)) # independent of h
        yscale = 'linear'
        if ylabel is None:
            if usetex:
                ylabel = "\log\,(M_\mathrm{gas}/(M_\mathrm{stars}+M_\mathrm{gas}))"
            else:
                ylabel = "log M_gas/M_stars"
    elif param in ["gas/stars"]: # special treatment for gas/stars 
        # use 2 R_e aperture
        massAll = f['SubhaloMassInRadType']
        massStars = massAll[:,4]*1e10
        massGas = massAll[:,0]*1e10
        y = np.log10(massGas/massStars) # independent of h
        yscale = 'linear'
        if ylabel is None:
            if usetex:
                ylabel = "\log\,(M_\mathrm{gas}/M_\mathrm{stars})"
            else:
                ylabel = "log M_gas/M_stars"
    elif param in ["sSFR","ssfr","SSFR"]: # special treatment for sSFR 
        # use 2 R_e aperture
        sfr = f['SubhaloSFRinRad']
        massStars = f['SubhaloMassInRadType']
        massStars = massStars[:,4]*1e10
        y = np.log10(sfr/massStars)
        yscale = 'linear'
        if ylabel is None:
            if usetex:
                ylabel = "\log\,[\mathrm{sSFR}\,(2\,R_\mathrm{e})]\ (\mathrm{yr}^{-1})"
            else:
                ylabel = "log sSFR (2 R_e) [1/yr]"
    elif param in ["sfr",'SFR']: # special treatment for SFR like param name
        # use 2 R_e aperture
        sfr = f['SubhaloSFRinRad']
        y = np.log10(sfr[:,])
        yscale = 'linear'
        if ylabel is None:
            if usetex:
                ylabel = "\log\,[\mathrm{SFR}\,(2\,R_\mathrm{e})]\ (\mathrm{M_\odot\,yr}^{-1})"
            else:
                ylabel = "log SFR (2 R_e) [M_sun/yr]"
    elif param == "SubhaloMassType":
        mass = f['SubhaloMassType']
        y = np.log10(1e10 * mass[:,:] / h)
        yscale = 'linear'
        param2 = (1,0,4,5)
        if ylabel is None:
            if usetex:
                ylabel = "$\log\,M\ [\mathrm{M}_\odot]$"
            else:
                ylabel = "log mass [M_sun]"
    elif param in ["Group_M_Crit200","Group_R_Crit200",
                  "Group_M_Crit500","Group_R_Crit500","m200","M200"]:
        ytmp = f[param]
        if param[0:7] == 'Group_M':
            y = np.log10(1e10 * ytmp[:] / h)
            yscale = 'linear'
        elif param[0:7] == 'Group_R':
            y = ytmp[:] / h  / (1+z)
        else:
            y = ytmp
        if ylabel is None:
            ylabel = param.replace('Crit200','\mathrm{200,c}')
            ylabel = ylabel.replace("Crit500","\mathrm{500,c}")
            if param[0:7] == 'Group_M':
                ylabel = ylabel + ' (\mathrm{M}_\odot)'
            elif param[0:7] == 'Group_R':
                ylabel = ylabel + ' \mathrm{(kpc)}'
            ylabel = ylabel.replace("Group_","")
    elif param in ["r_over_R_Crit200","r_over_R_Crit500","r/r200","r/r500"]:
        if '/' in param:
            param = "r_over_R_Crit" + param[-3:]
        
        # align f and fCen to the same SnapNums
        # iGood 
        SnapSat = f['SnapNum']
        if len(SnapCen) != len(SnapSat):
            iSatGood = np.in1d(SnapSat,SnapCen)
            iCenGood = np.in1d(SnapCen,SnapSat)
            if verbose >= 1:
                print("mismatch Sat and Cen:")
                print("bad iSat:")
                print("bad iCen")
        else:
            iSatGood = np.ones(len(SnapSat)).astype(bool)
            iCenGood = np.ones(len(SnapSat)).astype(bool)
        pos = f['SubhaloPos'][iSatGood,:]
        posCen = fCen['SubhaloPos'][iCenGood,:]
        y = np.zeros(len(pos[:]))
        for j in range(3): # loop over cartesian axes
            # dr = FixPeriodic(pos[:,j]-posGroup[:,j],sim)
            dr = FixPeriodic(pos[:,j]-posCen[:,j],sim)
            y = y + dr*dr
            Delta = param[-3:]    
        y = np.sqrt(y)
        paramRef = "Group_R_Crit" + Delta
        yRef = fCen[paramRef][iCenGood]
        y = y / yRef
        y = np.where(y < yCen,yCen,y)
        if verbose >= 2:
            print("pos[:,:] = ",pos[:,:]) 
            # print("posGroup[:,:]=",posGroup[:,:])
            print("yRef = ", yRef[:])
        # r^2 = dx^2 + dy^2 + dz^2

        # r / R_vir (independent of h)

        yscale = 'log'
        ysuf = param[7:].replace('Crit200','\mathrm{200,c}')
        ysuf = ysuf.replace("Crit500","\mathrm{500,c}")
        if ylabel is None:
            ylabel = "$r/" + ysuf + "$"
    elif param in ['r','sep','r_r200']:
        pos = f['SubhaloPos']
        # posGroup = f['GroupPos']
        y = np.zeros(len(pos[:]))
        if verbose > 0:
            print("pos[:,:] = ",pos[:,:]) 
            # print("posGroup[:,:]=",posGroup[:,:])
            print("x = ",x)
        # r^2 = dx^2 + dy^2 + dz^2
        for j in range(3): # loop over cartesian axes
            # dr = FixPeriodic(pos[:,j]-posGroup[:,j],sim)
            dr = FixPeriodic(pos[:,j]-posCen[:,j],sim)
            y = y + dr*dr
        if param == 'r_r200':
            y1 = np.sqrt(y)
            y2 = R200Cen
            y = np.transpose([y1,y2])
            if ylabel is None:
                ylabel = '$\mathrm{radial\ distance,\ r_{200c}\ (kpc)}$'
        else:
            y = np.sqrt(y)
            if ylabel is None:
                ylabel = '$\mathrm{radial\ distance\ (kpc)}$'
    elif param == 'SurfaceMassDensity':
        massStarsrhalf = 1e10/h*f['SubhaloMassInHalfRadType'][:,4]
        massGasrhalf = 1e10/h*f['SubhaloMassInHalfRadType'][:,0]
        rhalf = f['SubhaloHalfmassRadType'][:,4] / (1+z)
        surfdensStars = massStarsrhalf/(np.pi*rhalf**2)
        surfdensGas = massGasrhalf/(np.pi*rhalf**2)
        y = np.zeros((len(surfdensStars),2))
        y[:,0] = surfdensStars
        y[:,1] = surfdensGas
        if ylabel is None:
            ylabel = '$\mathrm{' + param + "\ (M_\odot/(kpc^2))}$"
    elif param[-3:] in ["Pos","pos"]:
        pos = f['SubhaloPos']
        if relative:
            y = pos[:]-pos[0,:]
            if ylabel is None:
                ylabel = "$\mathrm{" + param + " - " + param + "}$[99] (kpc)"
        else:
            y = pos[:]
        # handle periodic boundary conditions
        y = FixPeriodic(y,sim) / h
        yscale = 'linear'
    elif param[-3:] in ["PosRel","posrel"]:
        # show positions relative to group
        pos = f['SubhaloPos']
        # posGroup = f['GroupPos']
        # y = pos - posGroup
        y = pos - posCen
        # watch for crossing of box boundaries
        y = FixPeriodic(y, sim) / h
        yscale = 'linear'
        if ylabel is None:
            ylabel = "$\mathrm{" + param + " - " + param + "_\mathrm{group}}$ (kpc)"
    elif param[-3:] in ["Vel","vel"]:
        vels = f['SubhaloVel']
        y = vels[:]
        yscale = 'linear'
        if ylabel is None:
            ylabel = '$\mathrm{' + param + "\ (km\ s^{-1})}$"
    elif param[-6:] in ["VelRel","velrel"]:
        vels = f['SubhaloVel']
        velGroup = f['GroupVel'] * (1+z[:,None])
        y = vels[:] - velGroup[:]
        yscale = 'linear'
        if ylabel is None:
            ylabel = "$v - v_\mathrm{group}\ (\mathrm{km\ s^{-1})}$"
    elif param in ["xy","xz","yz"]:
        # trajectory with origin at final position
        xaxis = None
        pos = f['SubhaloPos']
        print("type(pos)=",type(pos))
        print("type(pos[:,0])=",type(pos[:,0]))
        if relative:
            dpos = (pos - pos[0])
        else:
            dpos = np.zeros((len(pos),3))
            for i in range(3):
                dpos[:,i] = pos[:,i]/1000 # in Mpc/h
        if param == "xy":
            x = dpos[:,0]
            y = dpos[:,1]
        elif param == "xz":
            x = dpos[:,0]
            y = dpos[:,2]
        else:
            x = dpos[:,1]
            y = dpos[:,2]
        x = FixPeriodic(x,sim) / h # kpc
        y = FixPeriodic(y,sim) / h # kpc
        if verbose > 0:
            print("pos=",pos[:])
            print("dpos=",dpos)
            print("x = ",x)
        if relative:
            xlabel = '$' + param[0] + "$ (kpc)"
            if ylabel is None:
                ylabel = '$' + param[1] + "$ (kpc)"
        else:
            xlabel = '$' + param[0] + "$ (Mpc)"
            if ylabel is None:
                ylabel = '$' + param[1] + "$ (Mpc)"
        yscale = 'linear'
    elif param in ["EJ","EL","Energy-AngMom"]: # energy and abular momentum both per unit mass
        pos = f['SubhaloPos']
        posGroup = f['GroupPos']
        rvir = f['Group_R_Crit200']
        G = 43
        Delta = 200
        H = h*cu.Eofz(Omegam0,1-Omegam0,z)
        vvir = 100 * np.sqrt(Delta/2) * H * rvir
        r = np.zeros(len(pos[:]))
        # r^2 = dx^2 + dy^2 + dz^2
        for j in range(3): # loop over cartesian axes
            dr = FixPeriodic(pos[:,j]-posGroup[:,j],sim)
            r = r + dr*dr
        r = np.sqrt(r)
        pot = vvir * cu.phiNFWhat(r/rvir)
    elif isSFR > 0: # sSFR trick
        sfrvarname = param.replace("sSFR","SFR")
        massvarname = param.replace("sSFR","Mass") + "Type"
        massvarname = massvarname.replace("in","In")
        if verbose > 0:
            print("massvarname=",massvarname)
        massStars = f[massvarname]
        massStars = massStars[:,4]*1e10
        y = f[sfrvarname]/massStars # independent of h
        if ylabel is None:
            if usetex:
                ylabel = "sSFR\ (2\,R_\mathrm{e}) [yr^{-1}]"
            else:
                ylabel = "sSFR (2 R_e) [1/yr]"           
    else: # use given param
        y = f[param][:]
        if ylabel is None:
            ylabel = '$' + dict_labels[param] + '$'
    
    if param2 is not None: # particle type if given
        if verbose > 0:
            print("line 2081: param2=",param2,"shape y=",y.shape)
        y = y[:,param2]
        
    # pericenters
    if peris is not None:
        [tperis,rperis],[tapos,rapos] = PeriApoCenters(subhaloid,sim=sim,
                                                           f=f,fCen=fCen,
                                                           verbose=verbose)            
            
    if ("Mass" in param) & (param != "SurfaceMassDensity"):
        # y = 1e10*y / h
        usetex = False
        if ylabel is None:
            ylabel = param + " ($\mathrm{M}_\odot$)"
    if verbose > 0:
        print("snapnums=",snapnums)
        print("x=",x)
        if verbose >= 2:
            print("y = ",y)
        print("ylabel=",ylabel)
        print("plot=",plot)
        print("param=",param)

    # plot
    if not plot:
        return y

    print("plotting...")
    # restrict to chosen snapnums
    if snapmin is None:
        snapmin = snapnums[-1]-1
    cond = ((snapnums >= snapmin) & (snapnums <= snapmax))
    if verbose >= 1:
        print("snapnums_min snapnums_max snapmin snapmax=",
              snapnums.min(),snapnums.max(),snapmin,snapmax)
        print("before snap filter: len x y cond ", len(x), len(y), len(cond))
    x = x[cond]
    y = y[cond]
    if y2 is not None:
        if verbose > 0:
            print("len y2 cond=",len(y2),len(cond))
        if (param[-7:] != '_r/r200') & (param[-2:] != '_r'):
            y2 = y2[condSatGood]

    tabztall = ztall()
    tab = np.flip(tabztall,axis=0) # flip to have decreasing time, as is y 
    tabsnaps = tab[:,0].astype(int)
    
    if xaxis in ['time','tlook','z']:
        if xaxis == 'time':
            tabindex = 2
            xlabel = 'age of the Universe (Gyr)'
        elif xaxis == 'tlook':
            tabindex = 3
            xlabel = 'lookback time (Gyr)'
        elif xaxis == 'z':
            tabindex = 1
            xlabel = 'redshift'
        if param not in ['xy','xz','yz']:
            cond = np.in1d(tabsnaps,x)
            tabx = tab[:,tabindex]
            x = tabx[cond]
    elif param not in ['xy','xz','yz']:
        xlabel = 'SnapNum'

    # plt.figure(figsize=[5,5])
    fig, ax = plt.subplots(figsize=[5,5])
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    if yscale == 'log':
        ypositive = y[y>0]
        if len(ypositive) == 0:
            raise ValueError("no positive quantities to plot in log scale!")
        if verbose > 0:
            print("len y = ",len(y),"len(ypositive)=",len(ypositive))
        ymin = np.min(ypositive)
        ymin4plot = 10**(np.floor(np.log10(ymin)))
        ymax = np.max(y[y<1e36])
        ymax4plot = 10**(np.ceil(np.log10(ymax)))
#         y2 = np.where(y<=0,ymin4plot,y)
        plt.ylim(0.5*ymin4plot,ymax4plot)
    # else:
        # y2 = y
    if ylabel is None:
        ylabel = param
        usetex = False
        
    # print("before colors: ylabel=",ylabel)
    # limit types to Dark matter, gas, and stars
    if verbose > 0:
        print("param = ",param)
        print("xlab y lab=",xlabel,ylabel)

    if param in ["SubhaloPos","SubhaloVel","SubhaloSpin","GroupPos","GroupVel",
                 "VelRel"]:
        colors = ('r','g','b')
        labels = ('$x$','$y$','$z$')
    elif param == "SubhaloStellarPhotometrics":
        colors = ('purple','b','darkgreen','gold','g','r','salmon','orange')
        labels = ('$U$','$B$','$V$','$K$','$g$','$r$','$i$','$z$')
    elif param.lower().find('type') > 0:
        colors = ('g','purple','brown','orange','b','k')
        if abbrevLegend:
            labels = ('gas','DM','type 2','tracers','stars','BH')
        else:
            labels = ('gas','dark matter','type 2','tracers','stars','black holes')
    elif param.find("MetalFractions") > 0:
        colors = ('purple','magenta','brown','orange','g','b','cyan','navy','r','k')
        labels = ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe', 'total')
    elif param == 'SurfaceMassDensity':
        colors = ('b','g')
        labels = ('stars','gas')
        print("setting colors: ylabel=",ylabel)
    elif param == 'r_r200':
        colors = ('darkorange','royalblue')
        labels = ('r (kpc)','$r_\mathrm{200c}$ (kpc)')
    else:
        colors = None
        legend = False
    if snapmax-snapmin>12:
        marker=None
        
    # ax = plt.gca()
    if colors is not None:
        if verbose > 0:
            print("in colors block")
        if param2 is not None:
            if isinstance(param2,Iterable): # param2 is a list or tuple or array
                for i, par2 in enumerate(param2):
                    ax.plot(x,y[:,i],label=labels[par2],
                             color=colors[par2],marker=marker)
            else: # param2 is a scalar
                par2 = param2
                ax.plot(x,y[:],color=colors[par2],marker=marker)
                legend = False
        else: # param2 is not given
            for i in range(len(labels)):
                if ((param.find("Type") > 0) & (i in [2,3])):
                    continue
                if verbose > 0:
                    print("[x]Type...")
                ax.plot(x,y[:,i],label=labels[i],
                         color=colors[i],marker=marker)

    elif param in ["xy","xz","yz"]:
        ax.plot(x,y,marker=marker)
        ax.scatter([x[0]],[y[0]],marker='o',c='k',s=150)
        ax.text(x[0],y[0],"99",c='orange',fontsize=10,
                 horizontalalignment='center',verticalalignment='center')
    elif param in ["r_over_R_Crit200","r_over_R_Crit500","r/r200","r/r500"]:
        ax.plot(x,y,'k')
        ax.scatter(x,y,s=20,c='darkorange')
    else: # Type is not in parameter name
        ax.plot(x,y,marker=marker)
        # if param in ["mass","Mass"]:
        #     ax.legend()
            
    if (param[-7:]=='_r/r200') or (param[-2:]=='_r') or (param[-5:]=='_r_rh'): 
        # add r or r/r200 on right axis
        if verbose > 0:
            print("r/r200 on right axis...")
            if verbose >= 2:
                print("len x x[condGoodSat] y2=",len(x),len(y2[condSatGood]),
                      len(y2))
        ax2 = ax.twinx()
        ax2.set_zorder(10)
    
        ax2.plot(x[condSatGood],y2,ls='--',color='darkorange')
        if param[-2:] == '_r':
            ax2.plot(x[condSatGood],y2,ls='--',color='darkorange',
                     label='$r$')    
            ax2.plot(x[condSatGood],yRef,ls=':',color='red',
                     label='$r_{200}$')
        else:
            ax2.plot(x[condSatGood],y2,ls='--',color='darkorange')        
        ax2.set_yscale('log')
        if param[-7:] == '_r/r200':
            ax2.set_ylabel('$r/r_{200}$',color='darkorange') 
            ax2.set_ylim(0.03,5)
            y2ticks = np.array([0.05,0.1,0.2,0.5,1,2,5])
        elif param[-3:] == '_rh':
            ax2.set_ylabel('$r$, $r_{200}$, $100\,r_\mathrm{h}$ (kpc)',color='k') 
            ax2.set_ylim(10,3000)
            y2ticks = np.array([10,20,50,100,200,500,1000,2000])  
        else:
            ax2.set_ylabel('$r$, $r_{200}$ (kpc)',color='k') 
            ax2.set_ylim(10,1000)
            y2ticks = np.array([10,20,50,100,200,500,1000])             
        # y2ticks = y2ticks[(y2ticks>y2.min()) & (y2ticks < y2.max())]
        if verbose > 0:
            print("y2ticks=",y2ticks)
        y2ticklabels = [str(yt) for yt in y2ticks]
        ax2.set_yticks(y2ticks)
        ax2.set_yticklabels(y2ticklabels)
        ax2.tick_params(axis='y',labelcolor='k')
        
        # horizontal line for r_200
        if param[-7:] == '_r/r200':
            xlims = ax2.set_xlim()
            xx = np.linspace(xlims[0],xlims[1],101)
            ax2.plot(xx,0*xx+1,color='darkorange',lw=1,ls='--')
            
        # add half-mass radii
        if param[-3:] == '_rh':
            colors = ['g','b']
            for ik, k in enumerate([0,4]):
                y = f['SubhaloHalfmassRadType'][:,k] / h
                ax2.plot(x[condSatGood],100*y,color=colors[ik],ls='--')
        # ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax2.grid(None)
        
    # add vertical lines for t peri
    if (peris is not None) & ('time' in xlabel):
        ylims = ax.set_ylim()
        if peris == 'vert-lines':
            ax.vlines(tperis,ylims[0],ylims[1],colors='darkorange',lw=1,ls='--')
        elif peris == 'pos':
            axright = ax.twinx()
            rpticks = np.array([0.1,1,10,100,1000,10000])
            rpminallow = 0.8*rperis.min()
            rpmaxallow = 1.2*rperis.max()
            rpticks = rpticks[(rpticks>rpminallow) & (rpticks < rpmaxallow)]
            rpticklabels = [str(tick) for tick in rpticks]
            if verbose > 0:
                print("rperis=",rperis)
                print("rpticks=",rpticks)
            _yticks = ylims[0] + (ylims[1]-ylims[0]) \
                        * np.log10(rpticks/rpminallow) \
                        / np.log10(rpmaxallow/rpminallow)
            yperis = ylims[0] + (ylims[1]-ylims[0]) \
                        * np.log10(rperis/rpminallow) \
                        / np.log10(rpmaxallow/rpminallow)
            axright.set_ylim(ylims)
            axright.set_yticks(_yticks)
            axright.set_yticklabels(rpticklabels)
            axright.set_ylabel('$r_\mathrm{peri}$ (kpc/$h$)')
            axright.set_yscale('log')
            ax.scatter(tperis,yperis,marker='v',c='brown')
            
    # check again legend, which may have been turned off in the block above
    if legend:
        if legendsize is not None:
            fsize = legendsize
        elif ((len(labels) <= 4) or (param.find("Type") > 0)):
            fsize = 16
        else:
            fsize = int(32/np.sqrt(len(labels)))

#             if param == "SubhaloStellarPhotometrics":
#                 print("small legend?")
#                 ax.legend(fontsize="small")
#             else:
        if ('type' in param) & (param[-2:] == '_r'):
            legendloc = 'lower left'
        ax.legend(fontsize=fsize,loc=legendloc)
        if param[-2:] == '_r':
            ax2.legend(fontsize=fsize,loc='lower right')
    if usetex: # assume LaTeX labels
        # check that the label has no $ signs
        # if ((xlabel[0] != '$') & (xlabel[-1] != '$')):
        #     xlabel = '$\mathrm{' + xlabel + '}$'
        if ((ylabel[0] != '$') & (ylabel[-1] != '$')):
            ylabel = '$' + ylabel + '$'
    if verbose > 0:
        print("ylabel=",ylabel)
    if labelsize is None:
        labelsize = 18
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.set_yscale(yscale)
    if snapmax is None:
        snapmax = 99
    if ylims is not None:
        ax.set_ylim(ylims)    
    if param not in ["xy","xz","yz"]:
        if xaxis not in ['time','tlook','z']:
            ax.set_xlim(snapmin,snapmax+1)
        elif xaxis in ['tlook','z']:
            ax.invert_xaxis()
        if snapminPlot is not None:
            ax.set_xlim(tabztall[snapminPlot,tabindex],tabztall[snapmax,tabindex])
        if shadex is not None:
            xx = np.linspace(shadex[0],shadex[1],51)
            ylims = ax.get_ylim()
            ax.fill_between(xx,ylims[0],ylims[1],color=shadex[-1],zorder=0)
        if topxaxis is not None:
           if (topxaxis == 'z') & (xaxis != 'z'):
               # 2nd horiontal axis (top of box): redshifts
               ax2 = ax.twiny()
               # zticks = CleanTicks(0,np.max(z))
               zticks = np.array([0,0.2,0.5,1,2,5])
               if verbose >= 2:
                   print("before top z ticks: snapminPlot=",snapminPlot)
                   print("tabztall=\n",tabztall)
               zticks = zticks[zticks <= tabztall[snapminPlot,1]]
               if verbose > 0:
                   print("zticks=",zticks)
               if xaxis == 'time':
                   tticks = AgeUniverse(Omegam0,h,zticks)
                   tticks = tticks.tolist()
               else:
                   tticks = AgeUniverse(Omegam0,h,0) \
                            - AgeUniverse(Omegam0,h,zticks)
                   tticks = tticks.tolist()
               zticks = zticks.tolist()
               zlabels = ['%.1f' % z for z in zticks]
               axlims = ax.get_xlim()
               ax2.set_xlim(axlims)
               ax2.set_xticks(tticks)
               ax2.set_xticklabels(zlabels)
               ax2.set_xlabel('redshift',fontsize=labelsize)
               ax2.minorticks_off()
               ax2.grid(None)
    elif param == "SubhaloStellarPhotometrics":
        if np.max(y) > -1:
            ygood = y[(y>-30) & (y<-1)]
            ax.set_ylim(np.min(ygood)-0.5,np.max(ygood)+0.5)
#     ax.tight_layout(w_pad=0.5, h_pad=1)
#     ax.grid()
    title = sim + "  subhalo " + str(subhaloid)
    if flag is not None:
        if flag == 1:
            title = title + ' (good flag)'
        elif flag == 0:
            title = title + ' (bad flag)'
        else:
            raise ValueError('Cannot understand flag = '+str(flag))
    plt.title(title, fontsize=18)
    if savefig is not False: # save figure to PDF file
        if snapmin > 1:
            snapsuf = str(snapmin) + "to" + str(snapmax)
        else:
            snapsuf = ""
        if savefig in [0,'auto']: # automatic file prefix
            simshort = sim.replace('-1', '')
            if param in ["xy","xz","yz"]:
                savefig = param + "_" + simshort + '_' + str(subhaloid) \
                            + "_" + snapsuf
            elif xaxis in ['time','tlook','z']:
                savefig = param + "vs" + xaxis + "_" + simshort + '_' \
                    + str(subhaloid) + "_" + snapsuf
            else:
                savefig = param + "vsSnapnum_" + simshort + '_' \
                    + str(subhaloid) + "_" + snapsuf
            savefig = savefig.replace('/','_')
            savefig = savefig.replace('_.','.')
        print("plotting into file",savefig + ".pdf")
#         ax.savefig(savefig + ".pdf",bbox_inches='tight',pad_inches=0.25)
#        savefigGAM(savefig + "_GAM.pdf")
        plt.tight_layout()
        plt.savefig(savefig + ".pdf",bbox_inches='tight')
    if param in ["xy","xz","yz"]: 
        return x,y
    elif y2 is not None:
        if param[-2:] == '_r':
            return y, y2, yRef
        else:
            return y, y2
    else:
        return y
    
def EnergyAngmomHistory(subhaloid,sim='TNG50-1',snapnum=99,snapmin=0,snapmax=99,
                        treeMethod='sublink_mpb',data_root_dir=None,datafileprefix=None, 
                        verbose=0):
    if data_root_dir is None:
        data_root_dir = os.getenv("HOME") + "/SIMS/TNG/"
        # answer = input("Enter root directory for TNG simulations: [" + data_root_dir + "]")
        # if answer != '':
        #     data_root_dir = answer
    datadir = data_root_dir + sim + "/output/"
    if verbose > 0:
        print("History: datadir=",datadir)
    if datafileprefix is None:
        # extract subhalo URL 
        if verbose > 0:
            print("extracting subhalo URL...")
        if snapnum < snapmax:
            subhaloid = getsubhaloid99(subhaloid,simulation=sim,snapnum=snapnum,verbose=verbose)
            if verbose > 0:
                print("at snapnum=99, subhaloid = ",subhaloid)
        if verbose > 0:
            print("getsubhalo...")
        sub = getsubhalo(subhaloid,simulation=sim,snapnum=snapmax,
                         save_dir=datadir,verbose=verbose)
        if verbose > 0:
            print ("History: sub=",sub)
        progurl = sub['related']['sublink_progenitor']
        if progurl is None:
            print("no progenitor for subhalo", subhaloid)
            return
        # extract tree of main progenitor
        if verbose > 0:
            print("extracting tree of main progenitor with method",treeMethod)
            print("... and saving tree to", datadir)
        datafile = get(sub['trees'][treeMethod],save_dir=datadir)
    elif datafileprefix in [0,'a',"auto"]:
        datafile = datadir + "sublink_mpb_" + str(subhaloid) + ".hdf5"
    else:
        datafile = datadir + datafileprefix + ".hdf5" 
    if verbose > 0:
        print("cwd=",os.getcwd())
        os.system("ls -l " + datafile)
    
    # extract values
    if verbose > 0:
        print("extracting parameters from tree in file " + data_root_dir + datafile + "...")
    try:
        f = h5py.File(datafile,'r')
    except:
        raise ValueError("cannot open file " + datafile)

    pos = f['SubhaloPos']
    posGroup = f['GroupPos']
    rvir = f['Group_R_Crit200']
    vels = f['SubhaloVel']
    velsGroup = f['GroupVel']
    snapnums = f['SnapNum']
    # print("snapnums from f = ", snapnums[:])
    # print("f.keys = ",f.keys())
    Delta = 200
    # snapnums = np.arange(snapmin,snapmax+1)
    z = zofsnap(snapnums)
    # print("z = ",z)
    hofz = h * Eofz(Omegam0,1-Omegam0,z)
    # print("hofz=",hofz)
    vvir = 100 * np.sqrt(Delta/2) * (hofz/1000) * rvir # in km/s
    r = np.zeros(len(pos[:]))
    dpos = np.zeros((len(pos),3))
    dvels = np.zeros((len(vels),3))
    # r^2 = dx^2 + dy^2 + dz^2
    r = separation(pos,posGroup,sim=sim)
    r_over_rvir = r/rvir
    
    # potential (assuming c=4 NFW for now)
    c = 6
    phihat0 = -1/(np.log(1+c)-c/(1+c))
    # print("phihat0=",phihat0)
    phihat = phihat0 * np.log(1+c*r_over_rvir)/r_over_rvir
    # print("phihat=",phihat)
    # print("rvir = ",rvir[:])
    # print("vvir = ",vvir)
    pot = vvir**2 * phihat
    # print("min max pot = ",np.min(pot),np.max(pot))
    # print("pot = ",pot)
    # kinetic term
    # vsq = vels[:,0]*vels[:,0] + vels[:,1]*vels[:,1] + vels[:,2]*vels[:,2]
    vsq = np.sum(dvels[:,:]*dvels[:,:],axis=1)
    # energy per unit mass
    # print("kin = ",0.5*vsq)
    E = 0.5*vsq + pot # in (km/s)^2
    
    # angular momentum

    # print("r=",r)
    vr = np.sum(dpos[:,:]*dvels[:,:],axis=1)/r
    # print("dpos=",dpos[:,:])
    # print("dvels=",dvels[:,:])
    # print("vsq=",vsq)
    # print("vr = ",vr)
    vt = np.sqrt(vsq-vr*vr)
    # print("vt=",vt)
    J = r*vt   # in kpc*(km/s)
    
    # print("type(snapnums) = ",type(snapnums))
    # print("snapnums=",snapnums,"min max =",snapnums.min(),snapnums.max())
    # print("rvir=",rvir)
    # print("snapmin=",snapmin,"snapmax=",snapmax)
    # snapnums = np.arange(0,100)
    cond = ((snapnums[:] >= snapmin) & (snapnums[:] <= snapmax))
    snaps = snapnums[cond]
    E = E[cond]
    J = J[cond]
    plt.scatter(snaps,-E,label='-E (km/s)$^2$')
    plt.scatter(snaps,J,label='J (kpc$\,$km/s)')
    plt.yscale('log')
    plt.legend()
    
    return E, J, dpos, dvels

def FitMainSequence(subhalodict,lssfrThreshold=-12,maxlMass=10,plot=False):
    lmass,lsfr = 10+np.log10(subhalodict['SubhaloMassInRadType'][:,4]),np.log10(subhalodict['SubhaloSFRinRad'])
    lssfr = lsfr - lmass
    lmbins = np.linspace(8,12,9)
    dlm = lmbins[1]-lmbins[0]
    lmbins = lmbins + dlm/2 # so that first bin is from 8 to 8.5
    lssfrMed = np.zeros(len(lmbins))
    lssfrMean = np.zeros(len(lmbins))
    # sfrThreshold = 10**lsfr_Threshold
    for i, lmb in enumerate(lmbins):
        # select by mass and above SFR threshold 
        cond = ((np.abs(lmass-lmb)<dlm/2) & (lssfr > lssfrThreshold))
        lmasstmp,lssfrtmp = lmass[cond],lssfr[cond]
        lssfrMed[i] = np.median(lssfrtmp)
        lssfrMean[i] = lssfrtmp.mean()
    lssfrMode = 3*lssfrMed - 2*lssfrMean
    
    # fit line
    
    coeffsMedian = np.polyfit(lmbins[lmbins<maxlMass],lssfrMed[lmbins<maxlMass],1)
    coeffsMode = np.polyfit(lmbins[lmbins<maxlMass],lssfrMode[lmbins<maxlMass],1)
    
    if plot:
        plt.scatter(lmass,lssfr,s=1)
        plt.scatter(lmbins,lssfrMed,label='median')
        plt.scatter(lmbins,lssfrMode,label='mode')
        plt.plot(lmbins,lmbins*coeffsMedian[0]+coeffsMedian[1],label='median')
        plt.plot(lmbins,lmbins*coeffsMode[0]+coeffsMode[1],label='mode')
        plt.xlim(8,12)
        plt.ylim(-13)
        plt.xlabel('$\log\,(M_\star/\mathrm{M}_\odot)$')
        plt.ylabel('$\log\,(\mathrm{sSFR}/\mathrm{yr}^{-1})$')
        plt.legend()
        plt.savefig(plot + '.pdf')
        
    return coeffsMedian, coeffsMode

def FitMainSequence2(subhalodict,lssfrThreshold=-12,maxlMass=10,minlMass=8,plot=False):
    lmass,lsfr = (10+np.log10(subhalodict['SubhaloMassInRadType'][:,4]),
                    np.log10(subhalodict['SubhaloSFRinRad']))
    lssfr = lsfr - lmass
    lmbins = np.linspace(8,12,9)
    dlm = lmbins[1]-lmbins[0]
    lmbins = lmbins + dlm/2 # so that first bin is from 8 to 8.5
    lssfrMed = np.zeros(len(lmbins))
    lssfrMean = np.zeros(len(lmbins))
    # sfrThreshold = 10**lsfr_Threshold
    for i, lmb in enumerate(lmbins):
        # select by mass and above SFR threshold 
        cond = ((np.abs(lmass-lmb)<dlm/2) & (lssfr > lssfrThreshold))
        lmasstmp,lssfrtmp = lmass[cond],lssfr[cond]
        lssfrMed[i] = np.median(lssfrtmp)
        lssfrMean[i] = lssfrtmp.mean()
    lssfrMode = 3*lssfrMed - 2*lssfrMean
    
    # fit line
    
    coeffsMedian = np.polyfit(lmbins[lmbins<maxlMass],lssfrMed[lmbins<maxlMass],1)
    coeffsMode = np.polyfit(lmbins[lmbins<maxlMass],lssfrMode[lmbins<maxlMass],1)

    # iterate up to 3 times
    
    for j in range(3):
        print("iteration",j+1)
        # residuals

        dlssfr = lssfr - (coeffsMode[0]*lmass + coeffsMode[1])
        lmbins2 = np.linspace(8,maxlMass,int(1+(maxlMass-8)*2))
        sigResiduals = np.zeros(len(lmbins2))
        for i, lmb in enumerate(lmbins2):
            # select by mass 
            cond = ((np.abs(lmass-lmb)<dlm/2) & (dlssfr > -10))
            dlssfrtmp = dlssfr[cond]
            sigResiduals[i] = np.std(dlssfrtmp)

        # fit line to the standard deviation of the residuals 

        print("lmbins2 = ",lmbins2)
        print("sigResiduals=",sigResiduals)
        plt.scatter(lmbins2,sigResiduals)
        coeffsSTD = np.polyfit(lmbins2,sigResiduals,1)

        # remove points that are beyond 3 sigma

#         cond = np.abs(lssfr - lssfrMode) < 3*(coeffsSTD[0]*lmass+coeffsSTD[1])

#         lmass = lmass[cond]
#         lssfr = lssfr[cond]
        lssfrMed = np.zeros(len(lmbins))
        lssfrMean = np.zeros(len(lmbins))
        # sfrThreshold = 10**lsfr_Threshold
        for i, lmb in enumerate(lmbins):
            # select by mass and above SFR threshold 
            cond = ((np.abs(lmass-lmb)<dlm/2) & (np.abs(lssfr - lssfrMode[i]) < 3*(coeffsSTD[0]*lmass+coeffsSTD[1])))
            lmasstmp,lssfrtmp = lmass[cond],lssfr[cond]
            print("lmass=",lmb,"N=",len(lmasstmp))
            lssfrMed[i] = np.median(lssfrtmp)
            lssfrMean[i] = lssfrtmp.mean()
        lssfrMode = 3*lssfrMed - 2*lssfrMean

        # fit line

        coeffsMedian = np.polyfit(lmbins[lmbins<maxlMass],lssfrMed[lmbins<maxlMass],1)
        coeffsMode = np.polyfit(lmbins[lmbins<maxlMass],lssfrMode[lmbins<maxlMass],1)
        print("Mode: slope=",coeffsMode[0],"value at 10=",coeffsMode[0]*10+coeffsMode[1])
        
    if plot:
        print("plotting...")
        plt.scatter(lmass,lssfr,s=1)
        plt.scatter(lmbins,lssfrMed,label='median')
        plt.scatter(lmbins,lssfrMode,label='mode')
        plt.plot(lmbins,lmbins*coeffsMedian[0]+coeffsMedian[1],label='median')
        plt.plot(lmbins,lmbins*coeffsMode[0]+coeffsMode[1],label='mode')
        plt.xlim(8,12)
        plt.ylim(-13)
        plt.xlabel('$\log\,(M_\star/\mathrm{M}_\odot)$')
        plt.ylabel('$\log\,(\mathrm{sSFR}/\mathrm{yr}^{-1})$')
        plt.legend()
        if plot == 'auto':
            plt.show()
        else:
            plt.savefig(plot + '.pdf')        
    return coeffsMode

def partTemperature(f):
    """
    temperature given internal energy from U = kT/[(gamma-1)mu m_p]
    where mu is mean particle mass in proton masses (code uses grams)
    
    argument: h5py file object
    source: https://www.tng-project.org/data/docs/faq/
    """
    u           = f['PartType0']['InternalEnergy'][:]    #  Internal Energy
    # VelDisp     = f['PartType0']['SubfindVelDisp'][:]
    X_e         = f['PartType0']['ElectronAbundance'][:] # electron abundance
    X_H         = 0.76             # Hydrogen mass fraction
    gamma       = 5.0/3.0          # adiabatic index
    kB          = 1.3807e-16       # Boltzmann constant in CGS units  [cm^2 g s^-2 K^-1]
    kB_kev      = 8.6173324e-8 
    mp          = 1.6726e-24       # proton mass  [g]
    mu          = (4*mp) / (1 + 3*X_H + 4*X_H*X_e) # mean particle mass
    temperature = (gamma-1)* (u/kB)* mu * 1e10
    return temperature

def testParticles():
    save_dir = home_dir + "/NOSAVE/SIMS/TNG/"
    # saveParticles(300907,save_dir=save_dir,request={'stars':'Masses,GFM_InitialMass,GFM_StellarFormationTime'})
    # toto = particles(300907,PartType=4,params="Masses",verbose=1)
    data = particles(300906,PartType=4,params=["Masses","GFM_InitialMass","GFM_StellarFormationTime"],verbose=1,
                     extract=True,save_dir=save_dir)
    print("type data[0] = ",type(data[0]))
    plt.scatter(10+np.log10(data[0]),10+np.log10(data[1]),s=1)
    lm = np.linspace(4, 5.5, 21)
    plt.plot(lm,lm,c='r')
    plt.xlabel('particle mass ($\mathrm{M}_\odot$)')
    plt.ylabel('particle initial mass ($\mathrm{M}_\odot$)')

def H0inverseinGyr(h):
    """inverse Hubble consytant in Gyr
    arg: h = H_0/(100km/s/Mpc)
    author: Gary Mamon"""
    yearindays = 365.25
    dayinseconds = 86400
    yearinseconds = yearindays * dayinseconds
    Mpc = 3.0857e24 # in cm
    return 1e-9/yearinseconds / (h*100*1e5/Mpc)

def Eofz(Omegam,Omegal,z,Omegar=0):
    """Dimensionless Hubble constant H(z)/H_0
    args:
        Omegam: z=0 density parameter
        Omegal: z=0 dark energy parameter
        z:      redshift
        Omegar: z=0 radiaton density parameter
    author: Gary Mamon"""
    # need to generalize for z as either scalar or numpy array
    return np.sqrt(Omegam*(1+z)**3 + Omegal + (1-Omegam-Omegal)*(1+z)**2 + Omegar*(1+z)**4)

def Omegam(Omegam0,Omega_lambda0,z,Omegar0=0):
    """Density parameter for given redshift
    author: Gary Mamon"""
    return Omegam0*(1+z)**3 / Eofz(Omegam0,Omega_lambda0,z,Omegar0) ** 2

def H0tflat(Omegam0,z):
    """Dimensionless time in flat Universe
    args:
        Omegam0:     Omega_matter,0
        z:          redshift
    source: Carroll, Press & Turner 92, ARAA, eq. (17)
    author Gary Mamon"""
     # need to generalize for z as either scalar or numpy array
    Om = Omegam(Omegam0,1-Omegam0,z)
    return 2/3*np.arcsinh(np.sqrt((1-Om)/Om)) / np.sqrt(1-Om)/Eofz(Omegam0,1-Omegam0,z)

def AgeUniverse(Omegam0,h,z=0):
    """Age of Universe for given redshift
    args:
        Omegam0: density parameter at z=0
        h:       H_0/(100 km/s/Mpc)
        z:       redshift
    author: Gary Mamon"""
    return H0tflat(Omegam0,z) * H0inverseinGyr(h)

def get_z_t(snapmin,snapmax=99,sim='TNG50-1'):
    # snaps = range(snapmin,snapmax+1)
    snap_z = snapofz('all')
    z = np.array([sz[1] for sz in snap_z if sz[0] >= snapmin and sz[0] <= snapmax])
    t = AgeUniverse(Omegam0, h, z)
    return z, t

def get_sats(group_id_zmin,snap_min=50,snap_max=99,sim='TNG50-1',basePath=None,
             MinLogSatMass=None,
             halos_zmin=None,halos_z=None,verbose=0):
    """
    Satellites of group that is not merged into more massive group later on
    Arguments:
        group_id_zmin : group_id at z=z_min
        snap_min: snapnum at z = z_max (starting time, default 50 [z=1])
        snap_max: snapnum at z = z_min (ending time, default 99 [z=0])
        sim : simulation (default: 'TNG50-1')
        MinLogSatMass : min log stellar mass at start redshift 
            (default None, auto 7.5 and 8.5 for TNG50 and TNG100)
        halos_zmin : Illustris dictionary of all halos at z=0 (default None)
        halos_z : Illustris dictionary of all halos at start redshift (default None)
        verbose : verbosity (default 0)

    Returns:
        satellite IDs at starting redshift
        
    Authors:
        Gary Mamon & Houda Haidar
    """
    
    # initialization
    if MinLogSatMass is None:
        if sim == "TNG50-1":
            MinLogSatMass = 7.5
        elif sim == "TNG100-1":
            MinLogSatMass = 8.5
    if basePath is None:
        basePath = os.getenv("HOME") + "/SIMS/TNG/" + sim + "/output"
        
    # all centrals of z=z_min groups
    if halos_zmin is None:
        print("extracting halos...")
        halos_zmin = il.groupcat.loadHalos(basePath=basePath, snapNum=99)
    centrals = halos_zmin['GroupFirstSub']
    
    # central of selected z=z_min group
    subcen_zmin = centrals[group_id_zmin]
    if verbose > 0:
        print("Histories for central subhalo at z_min: ",subcen_zmin)
        
    # most massive progenitors at z
    ff = History(subcen_zmin,sim=sim,verbose=verbose)
    # mmps = ff['SubfindID'][:]
    mmps = ff['SubhaloID'][:]
    groups = ff['SubhaloGrNr'][:]
    # mmps = History(subcen_zmin,param='SubfindID',plot=False)
    # groups = History(subcen_zmin,param='SubhaloGrNr',plot=False)
    snapnums = np.linspace(99,99-len(groups)+1,len(groups))
    subcen_z = mmps[snapnums==snap_min]
    group_z = groups[snapnums==snap_min]
    if halos_z is None:
        print("re-extacting halos...")
        halos_z = il.groupcat.loadHalos(basePath=basePath, snapNum=snap_min)
        print("done")
    GroupFirstSub_z = halos_z['GroupFirstSub'][group_z]
    if verbose > 0:
        print("subcen_z GroupFirstSub_z type = ",subcen_z,GroupFirstSub_z,type(subcen_z))
    
    # satellites of same group at z
    GroupNsubs_z = halos_z['GroupNsubs'][group_z]
    if GroupNsubs_z == 1:
        if verbose > -1:
            print("no satellites!")
        return np.empty(1)
    
    satellites = np.arange(GroupFirstSub_z+1, GroupFirstSub_z+GroupNsubs_z)
    
    # stellar masses and flags
    if verbose > 0:
        print("basePath=",basePath)
    tab = il.groupcat.loadSubhalos(basePath=basePath,snapNum=snap_min,
                                   fields=['SubhaloMassInRadType','SubhaloFlag'])
    sat_masses_z = tab['SubhaloMassInRadType'][satellites][:,4]
    sat_flags_z = tab['SubhaloFlag'][satellites]
    
    # filter on stellar masses and flags
    cond = (sat_flags_z == 1) & (sat_masses_z*1e10 > 10**MinLogSatMass)
    return satellites[cond]

def TraceBack(group_id_snap_max,snap_max=99,snap_start=50,simulation='TNG50-1',
              groups_snap_max=None,groups_snap_start=None,extract=False,verbose=0):
    
    # 1) All groups at snapnum=snap_max and snapnum=snap_start (slow, do once)
    basePath = home + "/SIMS/TNG/" + simulation + "/output"
    if groups_snap_max is None:
        if verbose > 0:
            print("extracting halos from disk for snapnum=",snap_max,"...")
        groups_snap_max = il.groupcat.loadHalos(basePath=basePath, snapNum=snap_max)

    if groups_snap_start is None:
        if verbose > 0:
            print("extracting halos from disk for snapnum=",snap_start,"...")
        groups_snap_start = il.groupcat.loadHalos(basePath=basePath, snapNum=snap_start)
    
    # 2) All central subhalos at that snapnum
    centrals = groups_snap_max['GroupFirstSub']
    
    # 3) central of selected z=z_min group
    subcen_snap_max = centrals[group_id_snap_max]
    
    # 4) most massive progenitor of that central to z (using sublink_mpb) —>  f
    if verbose > 0:
        print("subcen_snap_max=",subcen_snap_max)
    f = getMPB(subcen_snap_max,sim=simulation,verbose=verbose)
    groups = f['SubhaloGrNr'][:]
    snaps = f['SnapNum'][:]
    group_snap_start = groups[snaps==snap_start][0]
    if verbose > 0:
        print("group_snap_start=",group_snap_start)
    group_dict = gethalo(group_snap_start,snapnum=snap_start,simulation=simulation)
    subcen_snap_start = group_dict['GroupFirstSub']
    Nsubs_snap_start = group_dict['GroupNsubs']
    if Nsubs_snap_start == 1:
        print("no satellites!")
        return -1
    subsats_snap_start = np.arange(subcen_snap_start+1, subcen_snap_start+ Nsubs_snap_start)
    
    # 5) trace forward history of descendants (and possible other parameters)
    f = []
    for i, subsat in enumerate(subsats_snap_start):
        f.append(getMDB(subsat,snapnum=snap_start,sim=simulation,verbose=verbose))
    return f, groups_snap_max, groups_snap_start

    # 6) NEED TO FIND WHEN subhalo IS MERGED INTO MORE MASSIVE ONE
    # USING DesctructionNext FORWARD IN THE TREE
    
def MainProg(subhalo,snap=99,sim='TNG50-1',method='url',wSnap=False,verbose=0):
    """Main progenitor subhalo
    author: Gary Mamon (gam AAT iap.fr) & Houda Haidar
    """
    if method == "url":
        url = baseUrl + sim + "/snapshots/" + str(snap) + "/subhalos/" + str(subhalo) + "/"
        f = get(url)
        if wSnap:
            return f['prog_sfid'],f['prog_snap']
        else:
            return f['prog_sfid']
    # elif method == "getMPB":
    #     f = getMPB(subhalo,snap,sim=sim,verbose=verbose)
    #     return f['SubfindID'][1]
    else:
        raise ValueError("cannot understand method=" + method)
    
def MainDesc(subhalo,snap=50,sim='TNG50-1',method='url',wSnap=False,verbose=0):
    """Main descendant subhalo
    author: Gary Mamon (gam AAT iap.fr) & Houda Haidar
    """
    if method == "url":
        url = baseUrl + sim + "/snapshots/" + str(snap) + "/subhalos/" + str(subhalo) + "/"
        f = get(url)
        if wSnap:
            return f['desc_sfid'],f['desc_snap']
        else:
            return f['desc_sfid']  
    # elif method == "getMDB":
    #     f = getMDB(subhalo,snap,sim=sim,verbose=verbose)
    #     return f['SubfindID'][1]
    else:
        raise ValueError("cannot understand method=" + method)
        
def DestructionNext(subhalo,snap=50,sim='TNG50-1',verbose=0):
    """Will subhalo be merged into more massive one on next step?
    returns boolean
    author: Gary Mamon (gam AAT iap.fr) & Houda Haidar"""
    # test that main progenitor of main descendant is not subhalo itself
    cond = MainProg(MainDesc(subhalo,snap=snap,sim=sim),snap=snap+1,sim=sim) != subhalo
    return cond

def GroupMergeNext(subhalo,snap=50,sim='TNG50-1',verbose=0):
    """Will halo be merged into more massive one on next step?
    returns boolean
    author: Gary Mamon (gam AAT iap.fr) & Houda Haidar"""
    cen = Central(subhalo,snap=snap,sim=sim)
    cond = MainProg(MainDesc(cen,snap=snap,sim=sim),snap=snap+1,sim=sim) != cen
    return cond

def Central(subhalo,snap=50,sim='TNG50-1'):
    return gethalo(subhalo,subhalo=True,snapnum=snap)['GroupFirstSub']

def SatCenGroupHistory(subhalo,snap=99,sim='TNG50-1',plot=False,verbose=0):
    if verbose > 0:
        print('extracting satellite history...')
    fSat = History(subhalo)
    GrSat = fSat['SubhaloGrNr'][:]
    if verbose > 0:
        print('extracting z=0 central ...')
    cen = Central(subhalo,snap=99,sim=sim)
    if verbose > 0:
        print('extracting central history...')
    fCen = History(cen)
    GrCen = fCen['SubhaloGrNr'][:]
    if verbose > 0:
        print("plotting...")
    snapnum = fSat['SnapNum']
    # earliest snapnum of agreement
    snapsAgree = snapnum[GrCen==GrSat]
    
    if plot:
        plt.plot(fSat['SnapNum'][:],GrSat,label='satellite')
        plt.plot(fCen['SnapNum'][:],GrCen,label='central')
        plt.legend()
        plt.yscale('symlog')
        plt.xlabel('snapnum')
        plt.ylabel('group number')
        title = sim + ' subhalo ' + str(subhalo) \
            + ' groups match from snapnum=%d'%snapsAgree.min()
        plt.title(title,fontsize=13)
    return snapsAgree

def DestructionNextHouda(subhalo,snap=50,sim='TNG50-1',verbose=0):
    """Will subhalo be merged into more massive one on next step?
    returns boolean
        False : means yes, it will merge
        True  : means no, it won't merge
    author: Houda Haidar"""
    # test that main progenitor of main descendant is not subhalo itself
    desc, desc_snap = MainDesc(subhalo,snap=snap,sim=sim,wSnap=True)
    if verbose > 1:
        print("desc,",desc,"desc_snap",desc_snap)
    #print("descendant")
    #print(desc, desc_snap)
    prog, prog_snap = MainProg(desc,snap=snap+1,sim=sim,wSnap=True)
    #print("progenitor")
    #print(prog, prog_snap)
    cond = (prog != subhalo)
    if verbose > 1:
        print("(prog(desc) != subhalo)?= ",cond)
    return desc, desc_snap, cond

def LastDescBeforeMerge(subhalo,snap=50,sim='TNG50-1',verbose=0):
    merge = False
    while snap < 99:
        if verbose > 0:
            print("snap subhalo = ", snap,subhalo)
        subhalo_old = subhalo
        snap_old = snap
        subhalo, snap, cond = DestructionNextHouda(subhalo,snap,sim=sim)
        if verbose > 1:
            print("now snap subhalo = ", snap,subhalo)
        if cond:
            merge = True
            break
        if verbose > 1:
            print("no merger, snap is now",snap)
    if merge:
        return snap_old,subhalo_old
    else:
        return 100,0

def FollowSubhaloForward(subhalo,snap=50,snapmax=99,sim='TNG50-1',
                         params = ['mass_gas','mass_dm','mass_stars','mass_bhs',
                                   'sfr',
                                   'massinrad_gas','massinrad_dm',
                                   'massinrad_stars','massinrad_bhs',
                                   'sfrinrad','pos_x','pos_y','pos_z',
                                   'vel_x','vel_y','vel_z','grnr'],
                         verbose=0,tab=[]):
    """follow subhalo until galaxy is merged by more massive one 
    or if host group is merged into more massive one
    
    arguments: 
        subhalo
        snapnum
        snapmax
        sim
        parameters
        verbose (0 for quiet)
        tab (should be [] at start)
        
    returns dataframe with snapnum and subhalo as first two columns
    
    authors: Gary Mamon & Houda Haidar"""
    
    properties = getsubhalo(subhalo,simulation=sim,snapnum=snap,parameter=params)
    properties = [snap,subhalo] + properties
    tab.append(properties)   
    print(properties)
 
    paramswExtra = ['snapnum','subhalo'] + params
    if snap == snapmax:
        return pd.DataFrame(tab,columns=paramswExtra)
    if ((not GroupMergeNext(subhalo,snap=snap,sim=sim))
        & (not DestructionNext(subhalo,snap=snap,sim=sim))
        & (snap < snapmax)):
        FollowSubhaloForward(MainDesc(subhalo,snap=snap,sim=sim),
                             snap=snap+1,
                             snapmax=snapmax,sim=sim,params=params,tab=tab)
    return pd.DataFrame(tab,columns=paramswExtra)
    
# now need to extract histories from snap_first to last time before subhalo destruction

def SubhaloScaleRadius(subhalo,sim='TNG50-1',snap=99):
    return getsubhalo(subhalo,sim,snap,parameter='vmaxrad')/2.16258

def Density_vir_NFW(x,c):
    """Dimensionless NFW 3D density (for testing) in virial units
    arguments:
        r/r_vir
        c (concentration) = r_vir/r_s
    returns: 
        rho(r) / (M_v/r_v^3)
    """
    
    # author: Gary Mamon
    
    denom = 4* np.pi * (np.log(1+c) - c/(1+c)) * x*(x+1/c)**2
    return 1/denom

def Boundaries(pos,halfWidth,sim="TNG50-1"):
    posmin = FixPeriodic(pos-halfWidth)
    posmax = FixPeriodic(pos+halfWidth)
    return posmin, posmax

def GasMap(sub_sat,snap,sim='TNG100-1',subpastsatmax=10,halfwidthoverrhalf=5,
           nxbins=40,Nmaxvfield=500,h=0.6774,extract=True,plotprefix=None,
           satOnly=True,vfcolor='cyan',alpha=0.4,verbose=0):
    """
    Prepare gas map with velocity field, without plotting
    
    Arguments:
     sub_sat: subhaloID of test satellite
     snap: snapshot number
     sim: TNG simulation (default: 'TNG100-1')
     subpastsatmax: max(subhaloID)-sub_sat (default: 10)
     halfwidthoverrhalf: box half-width in units of half-mass radius of all particles
                          (default: 5)
     nxbins: number of bins in one dimension in map (default: 40)
     Nmaxvfield: max number of velocity field arrows in plot (default: 500)
     h: dimensionless Hubble constant at z=0 (default: 0.6774)
     extract: [boolean] (default: True) 
         True: force extraction of particles
         False: read particle data from disk
     plotprefix: prefix of file (default None)
     verbose: verbosity (default: 0)

    Returns:
        subs (subhalos considered) [N]
        pos_rel (positions of gas particles relative to subhalo) [N,3]
        vel_rel (velocities of gas particles relative to subhalo) [N,3]
        mass (masses of gas particles) [N]
        pos_group_rel (position of group relative to subhalo) [3]
        vel_group_rel (velocity of group relative to subhalo) [3]
        vel_gas_bulk (bulk velocity of all gas particles relative to subhalo) [3]
    """
    # redshift parameters for velocity conversions
    z = zofsnap(snap,sim)
    a = 1/(1+z)
    asqrt = np.sqrt(a)
    
    # satellite
    params_sat = getsubhalo(sub_sat,simulation=sim,snapnum=snap)
    pos_sat = np.array([params_sat['pos_x'],params_sat['pos_y'],params_sat['pos_z']])
    vel_sat = np.array([params_sat['vel_x'],params_sat['vel_y'],params_sat['vel_z']])
    
    # half-width
    rhalf = params_sat['halfmassrad'] / h
    halfwidth = halfwidthoverrhalf * rhalf
    
    # velocity dispersion
    sigmav = params_sat['veldisp']
    
    # group
    groupID = params_sat['grnr']
    params_group = gethalo(groupID,simulation=sim,snapnum=snap)
    pos_group = params_group['GroupPos']
    vel_group = [v/a for v in params_group['GroupVel']]      # km/s
    # M200c = 1e10/h*params_group['Group_M_Crit200'] # M_sun
    Nsubs = params_group['GroupNsubs']
    print("Nsubs = ", Nsubs)
    
    # central
    sub_cen = params_group['GroupFirstSub']
    params_cen = getsubhalo(sub_cen,simulation=sim,snapnum=snap)
    pos_cen = np.array([params_cen['pos_x'],params_cen['pos_y'],params_cen['pos_z']])
    vel_cen = np.array([params_cen['vel_x'],params_cen['vel_y'],params_cen['vel_z']])
    
    # gas particles of all subhalos within sub_sat + subpastsatmax
    # . initialize
    pos_rel_all = np.empty(shape=(0,3))
    vel_rel_all = np.empty(shape=(0,3))
    mass_all = np.empty(shape=0)
    subs_all = np.empty(shape=0)
    vel_gas_bulk = np.zeros(3)
    # loop over subhalos from central to final
    for sub in range(sub_cen,sub_cen+Nsubs):
        if sub == sub_cen:
            print("\ncentral")
        elif sub == sub_sat:
            print("\ntest satellite")
        print("sub", sub,"...")
        
        # stop after final desired subhalo
        if sub > sub_sat + subpastsatmax:
            break
        params_sub =  getsubhalo(sub_sat,simulation=sim,snapnum=snap)
        pos_sub = np.array([params_sub['pos_x'],params_sub['pos_y'],params_sub['pos_z']])
        # vel_sub = np.array([params_sub['vel_x'],params_sub['vel_y'],params_sub['vel_z']])
    
        # filter subhalos to be in cube
        pos_rel_sub = FixPeriodic(pos_sub-pos_sat,sim) / h # comoving kpc
        if np.any(np.abs(pos_rel_sub) > halfwidth):
            continue
        
        # extract particles
        if verbose > 0:
            print("particles ...")
        parts = particles(sub,sim=sim,snapnum=snap,PartType=0,extract=extract)
        if len(parts.keys()) == 0:
            print("empty key, moving to next sub...")
            continue
        elif verbose > 0:
            print("found", len(parts['Masses']),"gas particles")
        pos_parts = parts['Coordinates']
        vel_parts = [asqrt * v for v in parts['Velocities']] # comoving km/s
        mass_parts = parts['Masses']
        
        # save subhalos of particles
        sub_parts = 0*mass_parts + sub
        subs_all = np.append(subs_all,sub_parts)
        if verbose > 0:
            print("past pos vel mass...")
        
        # bulk velocity of test satellite gas particles
        if sub == sub_sat:
            # vel_gas_bulk = np.sum(mass_parts[:,None]*vel_parts,axis=0) \
            #     / np.sum(mass_parts)
            # print("vel_bulk_sat-gas-1=",vel_gas_bulk)
            vel_gas_bulk = np.average(vel_parts,axis=0,weights=mass_parts) \
                - vel_sat
            print("vel_bulk_sat-gas-2=",vel_gas_bulk)

        # relative positions and velocities in subhalo frame
        pos_rel = FixPeriodic(pos_parts-pos_sat,sim) / h # comoving kpc
        pos_rel_all = np.append(pos_rel_all,pos_rel)
        pos_rel_all.reshape((-1,3))
        vel_rel = vel_parts - vel_sat
        vel_rel_all = np.append(vel_rel_all,vel_rel)
        vel_rel_all.reshape((-1,3))
        mass_all = np.append(mass_all,mass_parts)
        if verbose > 1:
            print("shapes pos vel mass = ", 
                  pos_rel_all.shape,vel_rel_all.shape,mass_all.shape)
    
    if verbose > 0:
        print("out of subhalo loop")
    pos_rel_all2 = pos_rel_all.reshape(-1,3)
    vel_rel_all2 = vel_rel_all.reshape(-1,3)
    # bulk gas velocity of full cube
    # vel_gas_bulk = np.sum(mass_all[:,None]*vel_rel_all2,axis=0) / np.sum(mass_all)
    
    # 	. one for opposite of direction to central (from FixPeriodic(pos_cen–pos_sat,sim))
    pos_cen_rel = FixPeriodic(pos_cen-pos_sat) / h # comoving kpc
    
    # 	. one for opposite to direction of group motion in test subhalo frame (from vel_cen–vel_sat), with length in same units as those of particle velocity field
    vel_group_rel = vel_group - vel_sat
    
    # filter to be in box width
    cond = (np.abs(pos_rel_all2[:,0])<halfwidth) \
            & (np.abs(pos_rel_all2[:,1])<halfwidth) \
            & (np.abs(pos_rel_all2[:,2])<halfwidth)
    posf = pos_rel_all2[cond]
    velf = vel_rel_all2[cond]
    massf = mass_all[cond]
    subsf = subs_all[cond]
        
    if plotprefix is not None:
        for axes in [[0,1],[0,2],[1,2]]:
            print("\nplotting axes",axes)
            # np.random.seed(123)
            plotGasMap(subsf,posf,velf,massf,pos_cen_rel,vel_group_rel,vel_gas_bulk,
                   sigmav,
                   rhalf,halfwidth,nxbins=nxbins,Nmaxvfield=Nmaxvfield,
                   satOnly=satOnly,vfcolor=vfcolor,
                   alpha=alpha,axes=axes,
                   verbose=verbose,prefix=plotprefix)
    return [subsf, posf, velf, massf, pos_cen_rel, vel_group_rel, vel_gas_bulk, 
            sigmav, rhalf, halfwidth]

def plotGasMap(subsf,posf,velf,massf,pos_cen_rel,vel_group_rel,vel_gas_bulk,
               sigmav,rhalf,halfwidth,nxbins=40,Nmaxvfield=500,
               satOnly=True,vfcolor='cyan',alpha=0.4,
               axes=[0,1], verbose=0,
               prefix=None):    

    # filter to limit number of vectors
    np.random.seed(123)
    if len(posf) > Nmaxvfield:
        print("len pos (filtered) = ", len(posf))
        i = range(len(posf))
        iGood = np.random.choice(i,size=Nmaxvfield,replace=False)
        posf2 = posf[iGood]
        velf2 = velf[iGood]
        massf2 = massf[iGood]
        subsf2 = subsf[iGood]
    else:
        posf2 = posf
        velf2 = velf
        massf2 = massf
        subsf2 = subsf
    
    # figure
    plt.figure(figsize=[6,6])
    ax = plt.gca()
    
    # * Build surface densities of gas in grid inside cube viewed along viewing axis
    # * Gas map
    ax0 = axes[0]
    ax1 = axes[1]
    print("ax0 =",ax0,"ax1 =",ax1)
    plt.hist2d(posf[:,ax0],posf[:,ax1],weights=np.log10(massf),bins=[nxbins,nxbins],
               cmap='pink')
    print("past hist2d")
    
    # * Velocity field
    print("velocity field for",len(posf2),"points")
    
    if not satOnly:
        ## not in sat or central
        print("v-field other...")
        i = np.arange(len(posf2)).astype(int)
        iOther= np.argwhere((subsf2 != sub_sat) & (subsf2 > subsf[0]))
        print("len(iOther)=",len(iOther))
        if len(iOther) > 0:
            if alpha == 0:
                alphaOther = alphaGasMap(len(iOther))
            else:
                alphaOther = alpha    
            plt.quiver(posf2[iOther,ax0],posf2[iOther,ax1],
                       velf2[iOther,ax0],velf2[iOther,ax1],
                       scale=50,scale_units='xy',
                       color='purple',alpha=alphaOther,label='other particles')
        
        # > central
        print("v-field center...")
        iCen = np.argwhere(subsf2 == subsf[0])
        print("len(iCen)=",len(iCen))
        if len(iCen) > 0:
            if alpha == 0:
                alphaCen = alphaGasMap(len(iCen))
            else:
                alphaCen = alpha
            print("max v0 v1 = ",velf2[iCen,ax0].max(),velf2[iCen,ax1].max())
            print("median v0 v1 = ",np.median(velf2[iCen,ax0]),np.median(velf[iCen,ax1]))
            print("median v = ",np.median(np.sqrt(velf2[iCen,ax0]**2+velf[iCen,ax1]**2)))
            plt.quiver(posf2[iCen,ax0],posf2[iCen,ax1],
                       velf2[iCen,ax0],velf2[iCen,ax1],
                       scale=50,scale_units='xy',
                       color='gray',alpha=alphaCen,label='central particles')
    
    # > test satellite
    print("v-field test satellite...")
    iSat= np.argwhere(subsf2 == sub_sat)
    print("len(iSat)=",len(iSat))
    if len(iSat) > 0:
        if alpha == 0:
            alphaSat = alphaGasMap(len(iSat))
        else:
            alphaSat = alpha
        print("max v0 v1 = ",velf2[iSat,ax0].max(),velf2[iSat,ax1].max())
        print("median v0 v1 = ",np.median(velf2[iSat,ax0]),np.median(velf[iSat,ax1]))
        print("median v = ",np.median(np.sqrt(velf2[iSat,ax0]**2+velf[iSat,ax1]**2)))
        if satOnly:
            plt.quiver(posf2[iSat,ax0],posf2[iSat,ax1],
                       velf2[iSat,ax0],velf2[iSat,ax1],
                       color=vfcolor,alpha=alphaSat,label='satellite gas velocities')
        else:
            plt.quiver(posf2[iSat,ax0],posf2[iSat,ax1],
                   velf2[iSat,ax0],velf2[iSat,ax1],
                   scale=50,scale_units='xy',
                   color='royalblue',alpha=alphaSat,label='satellite gas velocities')
        v = np.transpose([velf2[iSat,ax0].flatten(),velf2[iSat,ax1].flatten()])
        m = massf2[iSat].flatten()
        print("new bulk sat v_proj gas of selected = ",np.average(v,axis=0,weights=m))
    print("past particle v-fields")
    
    # superpose 2 vectors
    pos1 = np.array([0,0,0])
    pos3 = np.array([pos1,pos1])
    print("pos3=",pos3)
    # velocity in units of sigma_v/5
    vel3 = 25/sigmav * \
        np.array([vel_group_rel[0:3],vel_gas_bulk[0:3]])
    print("vel3=",vel3)
    colors = ['g','b']
    labels = ['group velocity (ram pressure)',
              'satellite gas bulk velocity']
    print("2nd quiver")
    for i in range(2):
        print("i ax0 ax1=",i,ax0,ax1)
        plt.quiver(pos3[i,ax0],pos3[i,ax1],
                   vel3[i,ax0],vel3[i,ax1],
                   color=colors[i],
                   label=labels[i])
        
    # radial arrow
    plt.quiver(pos1[ax0],pos1[ax1],-pos_cen_rel[ax0],-pos_cen_rel[ax1],
               color='r',label='direction away from central (AGN-cen)')
    print("done")
    
    # add circles
    print("circles...")
    for i, n in enumerate([1,2,halfwidthoverrhalf]):
        rad = n*rhalf
        print("n rad = ",n,rad)
        circle = plt.Circle((0, 0), radius=rad, color='k', fill=False, ls='-')
        ax.add_artist(circle)
        
    # * plot limits –5 r_1/2, 5_r_1/2, so that  at center of test subhalo
    plt.xlim([-halfwidth,halfwidth])
    plt.ylim([-halfwidth,halfwidth])
    axlabels = ['x','y','z']
    plt.xlabel('$' + axlabels[ax0] + '\ \mathrm{(kpc)}$')
    plt.ylabel('$' + axlabels[ax1] + '\ \mathrm{(kpc)}$')
    plt.grid()
    plt.legend(loc='upper left',fontsize=12,title=r'{\bf velocities in satellite frame}',
               title_fontsize=12)
    plt.title(sim + '$\ \ \ \mathrm{snapnum}\ $' + str(snap) + '$\ \ \ \mathrm{subhalo}\ $' + str(sub_sat), fontsize=14)
    if prefix is None:
        return
    if prefix == 'auto':
        file = "GasMap_" + str(sub_sat) + '_' + axlabels[ax0] + axlabels[ax1] + ".pdf"
    else:
        file = prefix + '_' + axlabels[ax0] + axlabels[ax1] + ".pdf"
    print("printing to",file)
    plt.savefig(file)
    
def alphaGasMap(N):
    return min(1,np.sqrt(400/N))

def FollowForward(subhaloid,param=None,param2=None,sim='TNG50-1',treeMethod='sublink_mdb',
                snapmin=None,snapmax=99,snapnum=99,datafileprefix=None,plot=True,
                xlabel='snapnum',ylabel=None,yscale='log',ylims=None,relative=True,
                data_root_dir=None,halo=False,
                verbose=0,savefig=False,usetex=True,marker=None):
    url = 'http://www.tng-project.org/api/' + sim + '/snapshots/' + str(snapnum) + '/subhalos/' \
        + str(subhaloid)
    sub = get(url) # get json response of subhalo properties
    if verbose >= 1:
        print("starting: sub=",sub)
    if param is None:
        param = sub.keys()
        param2 = param
        plot = False
    else:
        param2 = ['snap','id'] + param
    # prepare dict to hold result arrays

    r = {}
    for par in param2:
        r[par] = []
    
    i = 0
    while (sub['desc_sfid'] != -1) & (sub['snap'] <= snapmax):
        if verbose >= 1:
            if i == 0:
                print("sub=",sub)
            i += 1
            print("snap=",sub['snap'],"sub['desc_sfid']=",sub['desc_sfid'])
        for par in param2:
            r[par].append(sub[par])
        # request the full subhalo details of the descendant by following the sublink URL
        sub = get(sub['related']['sublink_descendant'])
    
    if not plot:
        return r, sub

    if 'ype' in param:
        for partType in ['gas','dm','stars','bhs']:
            mass_logmsun = np.log10( np.array(r['mass_'+partType])*1e10/0.704)
            plt.plot(r['snap'],mass_logmsun,label=partType)
    # elif ('r' in param) & ('')
    for par in param:
        plt.plot(r['snap'],r[par],label=par)
    plt.xlabel(xlabel)
    if ylabel is None:
        ylabel = param
    plt.ylabel(ylabel)
    if isinstance(param,list):
        plt.legend(loc='lower right')
    plt.show()

    return r

# # * For test subhalo:

def HistoryTree(subhalo,snapnum,sim='TNG50-1',param='Group_M_Crit200',
                param2=None,
                tree_method='sublink_mpb',log=False,xlog=False,
                forceExtract=False,plot=False,
                savefig=False,plotsuffix='pdf',verbose=0):
    f = getTree(subhalo,snap=snapnum,sim=sim,tree_method=tree_method,
                forceExtract=forceExtract,verbose=verbose)
    snaps = f['SnapNum'][:]
    cens = f['SubhaloID'][:] == f['FirstSubhaloInFOFGroupID'][:]
    print("sum cens = ",sum(cens))
    colors = np.array(['grey','m','purple','b','c','g','darkorange','r','brown','k'])
    colorsall = colors[snaps%10]
    if len(snaps) < 150:
        size = 20
    else:
        size = gu.Markersize(len(snaps))
    sizes0 = 0*snaps + size
    if sum(cens) < len(snaps):
        sizes = np.where(cens,2*sizes0,0.25*sizes0)
    else:
        sizes = sizes0
    print("sizes0: min max = ",sizes0.min(),sizes0.max())
    print("sizes: min max = ",sizes.min(),sizes.max())
    print("sizes unique = ",np.unique(sizes))
    params = f[param][:]
    if param2 is not None:
        param2s = f[param2][:]
        plt.scatter(param2s,params,s=sizes,c=colorsall)
        plt.xlabel(param2)
    else:
        # plt.plot(snaps,params)
        print("max snap=",snaps.max())
        plt.scatter(snaps,params,s=sizes,c=colorsall)
        if plot:
            plt.plot(snaps,params)
        plt.xlabel('snapnum')
        plt.xscale('linear')
    plt.ylabel(param)
    if log:
        plt.yscale('log')
    if xlog:
        plt.xscale('log')
    plt.title(tree_method + ': ' + sim + ' subhalo ' + str(subhalo) 
              + ' @ snap=' + str(snapnum))
    if savefig:
        fileprefix = str(subhalo) + '@' + str(snapnum) + '_' + tree_method
        if param2 is not None:
            fileprefix = fileprefix + '_' + param + '_' + param2
        else:
            fileprefix = fileprefix + '_' + param
        if plotsuffix == 'png':
            plt.savefig(fileprefix + '.png',dpi=300)
        else:
            plt.savefig(fileprefix + '.pdf')

def LocalBranches(f,subhalo,snap):
    """ Return local rbanch(es) from HDF5 type 
    Arguments: 
        f: h5py._hl.files.File type
    Returns: dataframe (with 2d entries with suffixes 0, 1, etc.)
    Author: Gary Mamon (gam AAT iap.fr)
    """
    df = ConvertHDF52df(f)
    dfsub = df.loc[(df.SubfindID==subhalo) & (df.SnapNum == snap)]
    return dfsub

def MDB(f,subhalo,snap):
    """Main Descendant Branch
    Arguments:
        f: h5py._hl.files.File type
        subhalo: subhalo (SubFindID)
        snap: SnapNum
    Returns:
        dataframe for Main Descendant Branch
    Author: Gary Mamon (gam AAT iap.fr)
        """
        
    df = ConvertHDF52df(f)
    dfsub = df.loc[(df.SubfindID==subhalo) & (df.SnapNum == snap)]
    if len(dfsub) > 1:
        raise ValueError("found " + str(len(dfsub)) + " cases for SubFindID="
                         + str(subhalo) + " and SnapNum=" + str(snap))
    desc = 0
    # snaps = [dfsub.SnapNum.values[0][0]]
    # subfinds = [dfsub.SubfindID.values[0][0]]
    subhalos = [dfsub.SubhaloID.values[0]]
    # masses = [dfsub.SubhaloMass.values[0][0]]
    # groupmasses = [dfsub.Group_M_Crit200.values[0][0]]
    while desc > -1:
        desc = dfsub.DescendantID.values[0]
        dfsub = df.loc[df.SubhaloID==desc]
        subhalos.append(desc)
        # subfinds.append(dfsub.SubfindID.values[0][0])
        # snaps.append(dfsub.SnapNum.values[0][0])
        # masses.append(dfsub.SubhaloMass.values[0][0])
        # groupmasses.append(dfsub.Group_M_Crit200.values[0][0])
    subhalos = np.array(subhalos)
    # subfinds = np.array(subfinds)
    # snaps = np.array(snaps)
    # masses = np.array(masses)
    # groupmasses = np.array(groupmasses)
    # dfmdb['SnapNum'] = snaps
    # dfmdb['SubhaloID'] = subhalos    
    # dfmdb['SubfindID'] = subfinds
    # dfmdb['SubhaloMasses'] = masses
    dfmdb = df.loc[df.SubhaloID.isin(subhalos)]
    return dfmdb

def ForwardHistory(subhalo,snapmin,sim="TNG50-1",param=None,yCen=0.01,
                   forceExtract=False,ylog=False,
                   verbose=0):
    f = getTree(subhalo,snapmin,tree_method="sublink_mdb",
                forceExtract=forceExtract,verbose=verbose)
    df = MDB(f,subhalo,snapmin)
    if param is None:
        raise ValueError("must specify param=")
    if len(df) > 136:
        raise ValueError("len=" + str(len(df)) + " too high")
    yplot = False
    ylabel = None
    yscale = None
    y = None
    legend = False
    # plt.scatter(df.SnapNum,df[param])
    if ('Type' in param) & ('ass' in param):
        print("type...")
        if '2Re' in param:
            param = 'SubhaloMassInRadType'
        else:
            param = 'SubhaloMassType'
        y = np.zeros((len(df),4))
        types = [0,1,4,5]
        names = ['gas','dark matter','stars','black holes']
        colors = ['g','purple','b','k']
        for i in range(len(types)):
            param2 = param + str(i)
            y[:,i] = 10 + np.log10(df[param2]/h)
            plt.plot(df.SnapNum,y[:,i],marker='o',c=colors[i],
                     label=names[i])
            legend = True
        yplot = True
        if param == 'SubhaloMassInRadType':
            ylabel = "$\log\,M (2\,r_\mathrm{1/2}) [\mathrm{M}_\odot]$"
        else:
            ylabel = "$\log\,M (\mathrm{M}_\odot)$"
    elif 'ass' in param:
        y = np.log10(10+df[param].values/h)
        if param == 'SubhaloMassInRad':
            ylabel = "$\log\,M\ (2\,r_\mathrm{1/2}) [\mathrm{M}_\odot]$"
        else:
            ylabel = "$\log\,M\ (\mathrm{M}_\odot)$"
    elif (param[0:12]=='Group_M_Crit') or (param=='M200') or (param=='M500'):
        param = 'Group_M_Crit' + param[-3:]
        y = 10 + np.log10(df[param].values/h)
        ylabel = param[6:].replace('Crit200','\mathrm{200,c}')
        ylabel = ylabel.replace("Crit500","\mathrm{500,c}")
        ylabel = '$' + ylabel + '$'
    elif param in ["r_over_R_Crit200","r_over_R_Crit500","r/r200","r/r500"]:
       if '/' in param:
           param = "r_over_R_Crit" + param[-3:]
       paramRef = "Group_" + param[7:]
       yRef = df[paramRef].values
       xGal, yGal, zGal = df['SubhaloPos0'], df['SubhaloPos1'], df['SubhaloPos2']
       xGrp, yGrp, zGrp = df['GroupPos0'], df['GroupPos1'], df['GroupPos2']
       y = np.zeros(len(xGal))
       dx = FixPeriodic(xGal-xGrp,sim=sim)
       dy = FixPeriodic(yGal-yGrp,sim=sim)
       dz = FixPeriodic(zGal-zGrp,sim=sim)
       r = np.sqrt(dx*dx+dy*dy+dz*dz)

       # r / R_vir (independent of h)
       y = r / yRef
       y = np.where(y < yCen,yCen,y) 
       yscale = 'log'
       ysuf = param[7:].replace('Crit200','\mathrm{200,c}')
       ysuf = ysuf.replace("Crit500","\mathrm{500,c}")
       ylabel = "$r/" + ysuf + "$"
    else:
       y = df[param].values
    if not yplot:
        plt.plot(df.SnapNum,y,marker='o')
    plt.xlabel('snapnum')
    if ylabel is None:
        plt.ylabel(param)
    else:
        plt.ylabel(ylabel)
    if ylog:
        plt.yscale('log')
    elif yscale is not None:
        plt.yscale(yscale)
    if legend:
        plt.legend()
    
    # if ('Mass' in param) or ('Rad' in param) or ('M_Group' in param) or ('R_Group' in param):
    #     plt.yscale('log')
    plt.title(sim + ': subhalo ' + str(subhalo) + ' (snap=' + str(snapmin) + ')')
    return y

def SurfaceDensityClass(df,sim='TNG50-1',lMstarsMax=13,lMstarsMin=7,qmax=0.9,
                        goodflag=True,coeffsCOMPACT = [-12.37,1.33],
                        plot=False,
                        plotresid=False):
    df['lMstars2Re'] = 10 + np.log10(df.SubhaloMassInRadType4/h)
    MstarsRe = df.SubhaloMassInHalfRadType4/h
    df = df.loc[df.lMstars2Re.between(lMstarsMin,lMstarsMax)]
    df['lsurfdens'] = np.log10(MstarsRe/(np.pi*df.SubhaloHalfmassRadType4**2))    
    if goodflag:
        df = df.loc[df.SubhaloFlag]
    coeffsCOMPACT = np.array(coeffsCOMPACT)
        
    # iterative fit of Surface density main sequence after removing the COMPACTs
    
    q = np.array([qmax])
    
    # select non-COMPACT
    df2 = df.loc[df.lsurfdens<mmu.buildpolyn(df.lMstars2Re,coeffsCOMPACT)]
    
    # iterate
    npasses = 2
    coeffs = np.zeros((2,npasses+1))
    for i in range(npasses):
        coeffs[:,i] = mmu.fitpolyn(df2.lMstars2Re.values, df2.lsurfdens.values, 1)
        print("pass coeffs = ",i,coeffs[:,i])
        residuals = df2.lsurfdens.values - mmu.buildpolyn(df2.lMstars2Re.values, coeffs[:,i])
        if plotresid:
            plt.scatter(df2.lMstars2Re.values,residuals,s=5)
            plt.xlabel('$\log(M_\star/\mathrm{M}_\odot)$')
            plt.ylabel('residuals')
            plt.title('pass' + str(i))
            plt.show()
        residualsq = np.quantile(residuals, q)
        df2 = df2.loc[residuals < residualsq[0]]

    # final fit        
    coeffs[:,i+1] = mmu.fitpolyn(df2.lMstars2Re.values, df2.lsurfdens.values, 1)
    print("final coeffs=",coeffs[:,i+1])
    # classes
    q = np.array([1-qmax,qmax])
    residuals = df.lsurfdens.values - mmu.buildpolyn(df.lMstars2Re.values, coeffs[:,i+1])
    if plotresid:
        plt.scatter(df.lMstars2Re.values,residuals,s=5)
        plt.xlabel('$\log(M_\star/\mathrm{M}_\odot)$')
        plt.ylabel('residuals')
        plt.title('final pass')
        plt.show()
    residualsq = np.quantile(residuals, q)
    lsurfdensminCOMPACT = mmu.buildpolyn(df.lMstars2Re,coeffsCOMPACT)
    cond = [residuals <= residualsq[0],
            (residuals > residualsq[0]) & (df.lsurfdens<lsurfdensminCOMPACT),
            df.lsurfdens>=lsurfdensminCOMPACT]
    choices = ['DIFFUSE','NORMAL','COMPACT']
    classes = np.select(cond,choices)
    colors = ['g','grey','b']
    sizes = [3,10,20]
    lMbins = np.linspace(lMstarsMin,lMstarsMax,11)
    colors2 = ['r','g','b','k']
    if plot:
        for i, ch in enumerate(choices):
            df3 = df.loc[classes == ch]
            plt.scatter(df3.lMstars2Re,df3.lsurfdens,s=sizes[i],c=colors[i],
                        label=choices[i])
        for i in range(npasses+1):
            plt.plot(lMbins,mmu.buildpolyn(lMbins, coeffs[:,i]),c=colors2[i])
        plt.plot(lMbins,mmu.buildpolyn(lMbins,coeffsCOMPACT),c='purple',lw=5)
        plt.ylabel('log surface density ($\mathrm{M_\odot}/\mathrm{kpc}^2)$')
        plt.xlabel('$\log(M_\star/\mathrm{M}_\odot)$')
        plt.legend()
        if goodflag:
            simflag = sim + ' good flag'
        else:
            simflag = sim + ' all flags'
        plt.title(simflag + ': ' + str(lMstarsMin) 
                  + '$ < \log(M_\star/\mathrm{M}_\odot) < $' + str(lMstarsMax))
    return classes
        
def loglen(x):
    return np.log(len(x))


def GasStarDMFracs(df,sim='TNG50-1',fontsize=20,savefig=True,markersize=10,
                   norm=None,
                   points='scatter',gridsize=10,func=len,lmmin=None,lmmax=None,
                   color='b',logMasses=False,title='auto',printfracs=False):
    mGas = df.Mass_gas_2Re_99
    mDM = df.Mass_DM_2Re_99
    mStars = df.Mass_star_2Re_99
    if logMasses:
        mGas = 10**mGas
        mDM = 10**mDM
        mStars = 10**mStars
        idvar = 'id_sub_99'
    else:
        idvar = 'SubhaloID'
    mTot = mGas + mStars + mDM
    fDM = mDM/mTot
    fGas = mGas/mTot
    fStars = mStars/mTot
    fStars2 = np.where(fStars==1,0.99,fStars)
    fGas2 = np.where(fStars==1,0.01,fGas)
    fDM2 = np.where(fStars==1,0.01,fDM)
    fStars = pd.Series(fStars2)
    fGas = pd.Series(fGas2)
    fDM = pd.Series(fDM2)
    if printfracs:
        tab = np.transpose([df[idvar].values,fStars.values,fGas.values,fDM.values])
        print("subhalo fStars fGas fDM=",tab)
    # print("stats fDM=",fDM.describe())
    ax = plt.subplot(projection="ternary")
    position = 'tick1'
    if norm == 'log':
        norm = mplcol.LogNorm()
    if points == 'scatter':
        ax.scatter(fStars,fGas,fDM,s=markersize,color=color)
    elif points == 'tribin':
        ax.tribin(fStars,fGas,fDM,gridsize=gridsize,
                  color=color,cmap='Greys',reduce_C_function=func,
                  edgecolors='r')
    elif points == 'hexbin':
        ax.hexbin(fStars,fGas,fDM,gridsize=gridsize,norm=norm,
                  color=color,cmap='Greys',reduce_C_function=func,
                  edgecolors='face')
    ax.set_tlabel('$\leftarrow$ Star fraction',fontsize=fontsize)
    ax.set_llabel('$\leftarrow$ Gas fraction',fontsize=fontsize)
    ax.set_rlabel('Dark matter fraction $\\rightarrow$',fontsize=fontsize)
    ax.taxis.set_label_position(position)
    ax.laxis.set_label_position(position)
    ax.raxis.set_label_position(position)
    if title == 'auto':
        ax.set_title(sim,fontsize=24)
    elif title is not None:
        ax.set_title(title,fontsize=24)
    if savefig:
        plt.savefig('fracs_ternary_lM' + str(lmmin) + str(lmmax) + sim.replace('-1','') + '.pdf')
    
def GasStarDMFracsTernary(df,sim='TNG50-1',fontsize=20,savefig=True,markersize=10,
                  logMasses=False, norm=None,
                   points='scatter',gridsize=20,func=len,lmmin=None,lmmax=None,
                   color='b',edgecolor='gray',title='auto',printfracs=False):
    mGas = df.Mass_gas_2Re_99
    mDM = df.Mass_DM_2Re_99
    mStars = df.Mass_star_2Re_99
    if logMasses:
        mGas = 10**mGas
        mDM = 10**mDM
        mStars = 10**mStars
        idvar = 'id_sub_99'
    else:
        idvar = 'SubhaloID'
    mTot = mGas + mStars + mDM
    fDM = mDM/mTot
    fGas = mGas/mTot
    fStars = mStars/mTot
    fStars2 = np.where(fStars==1,0.99,fStars)
    fGas2 = np.where(fStars==1,0.01,fGas)
    fDM2 = np.where(fStars==1,0.01,fDM)
    fStars = pd.Series(fStars2)
    fGas = pd.Series(fGas2)
    fDM = pd.Series(fDM2)
    if printfracs:
        tab = np.transpose([df[idvar].values,fStars.values,fGas.values,fDM.values])
        print("subhalo fStars fGas fDM=",tab)
    # print("stats fDM=",fDM.describe())
    if savefig:
        prefix = 'fracs_ternary_lM' + str(lmmin) + str(lmmax) + sim.replace('-1','')
    else:
        prefix= None
    ggu.Ternary(fStars,fGas,fDM,
            labeltop='Star',
            labelleft='Gas',
            labelright='Dark matter',gridsize=gridsize,norm=norm,
            points=points,ms=markersize,color=color,edgecolor=edgecolor,
            title=title,saveprefix=prefix)

def PeriApoCenters(subhalo,sim='TNG50-1',f=None,fCen=None,subhalo_cen=None,
                   plot=False,verbose=0):

    # subhalo history and positions
    if f is None:
        if verbose > 0:
            print("history...")
        f = History(subhalo,extract=True,plot=False)
    pos = f['SubhaloPos']   
    snaps = f['SnapNum']
    
    # subhalo of z=0 central
    subhalo_cen = f['GroupFirstSub'][0]
    if verbose > 0:
        print("subhalo_cen=",subhalo_cen)
        if fCen is None:
            print("fCen is None")
        else:
            print("fCen is NOT None")
            
    # central history and positions
    if (fCen is None) & (subhalo_cen is not None):
        if verbose > 0:
            print("history of central...")
        fCen =  History(subhalo_cen,extract=True,plot=False)

    if fCen is None:
        raise ValueError("Missing fCen!")
    posCen = fCen['SubhaloPos']
    snapsCen = fCen['SnapNum']
    if verbose > 0:
        print("len snaps snapsCen =",len(snaps),len(snapsCen))

    # limit central histories to snapnums of subhalo histories    
    snapsGood = np.intersect1d(snaps,snapsCen)
    cond = np.isin(snaps,snapsGood)
    cond2 = np.isin(snapsCen,snapsGood)
    snaps = snaps[cond]
    pos = pos[cond]
    snapsCen = snapsCen[cond2]
    posCen = posCen[cond2]
    dr = separation(pos, posCen, sim=sim)
    if verbose > 0:
        print("len snapsCen is now",len(snapsCen))
        print("len pos posCen = ",len(pos),len(posCen))
        
    # snapshots and lookback times
    tabztall = np.flip(ztall(),axis=0)
    tlook_all = tabztall[:,-1]
    snaps_all = tabztall[:,0]
    cond3 = np.isin(snaps_all,snapsGood)
    tlook = tlook_all[cond3]
    tlookfine = np.linspace(tlook[0],tlook[-1],1001)
    
    # fit pos vs time 6 times (3 for sub and 3 for Cen)
    if verbose > 0:
        print("interpolating...")
    posfine = np.zeros((len(tlookfine),3))
    posCenfine = np.zeros((len(tlookfine),3))
    # handle box crossings
    shift_pos = FixPeriodic(pos-pos[0],sim=sim)
    shift_posCen = FixPeriodic(posCen-posCen[0],sim=sim)
    for j in range(3):
        fsub = interp1d(tlook,shift_pos[:,j])
        posfine[:,j] = FixPeriodic(pos[0,j] + fsub(tlookfine), sim=sim)
        fCen = interp1d(tlook,shift_posCen[:,j])
        posCenfine[:,j] = FixPeriodic(posCen[0,j] + fCen(tlookfine), sim=sim)
        
    if verbose >= 2:
        plt.figure()
        colors = ['r','g','b']
        for j in range(3):
            plt.scatter(tlook,shift_pos[:,j],c=colors[j],label='sub',marker='^')
            plt.scatter(tlook,shift_posCen[:,j],c=colors[j],label='cen',marker='o')
        plt.legend()
        plt.xlabel('lookback time')
        plt.ylabel('position relative to last')
        plt.show()
        
    # separations
    drfine = separation(posfine, posCenfine, sim=sim)
    if verbose >= 2:
        for i, tl in enumerate(tlook):
            if (tl < 10.51) or (tl > 10.68):
                continue
            print("i tlook=",i,tl)
            for j in range(3):
                print(j,pos[i,j],posCen[i,j])
        i = np.argwhere(drfine > 1300).astype(int)
        print("i = ", i)
        for _i in i:
            print("_i=",_i)
            for _ii in range(_i[0]-1,_i[0]+2):
                print(_ii, tlookfine[_ii], drfine[_ii])
                for j in range(3):
                    print(j,posfine[_ii,j], posCenfine[_ii,j])

    # extract pericenters
    iperi = argrelextrema(drfine,np.less)
    iapo = argrelextrema(drfine,np.greater)
    tperi = tlookfine[iperi]
    rperi = drfine[iperi]
    tapo = tlookfine[iapo]
    rapo = drfine[iapo]
    if plot:
        plt.scatter(tlook,dr,s=5,c='k')
        plt.plot(tlookfine,drfine,'g-',lw=0.5)
        plt.scatter(tperi,rperi,s=50,facecolors='none',edgecolors='b',marker='v')
        plt.scatter(tapo,rapo,s=50,facecolors='none', edgecolors='r',marker='^')
        plt.xlabel('lookback time (Gyr)')
        plt.ylabel('separation (kpc/h)')
        plt.yscale('log')
        plt.savefig('periapo_test_' + str(subhalo) + '.pdf')
    return [tperi,rperi],[tapo,rapo]

def PeriApoCenters_onseps(subhalo,sim='TNG50-1',f=None,fCen=None,subhalo_cen=None,
                   plot=False,verbose=0):

    # subhalo history and positions
    if f is None:
        if verbose > 0:
            print("history...")
        f = History(subhalo,extract=True,plot=False)
    pos = f['SubhaloPos']   
    snaps = f['SnapNum']
    
    # subhalo of z=0 central
    subhalo_cen = f['GroupFirstSub'][0]
    if verbose > 0:
        print("subhalo_cen=",subhalo_cen)
        if fCen is None:
            print("fCen is None")
        else:
            print("fCen is NOT None!")
            
    # central history and positions
    if (fCen is None) & (subhalo_cen is not None):
        if verbose > 0:
            print("history of central...")
        fCen =  History(subhalo_cen,extract=True,plot=False)

    if fCen is None:
        raise ValueError("Missing fCen!")
    posCen = fCen['SubhaloPos']
    snapsCen = fCen['SnapNum']
    if verbose > 0:
        print("len snaps snapsCen =",len(snaps),len(snapsCen))

    # limit central histories to snapnums of subhalo histories    
    snapsGood = np.intersect1d(snaps,snapsCen)
    cond = np.isin(snaps,snapsGood)
    cond2 = np.isin(snapsCen,snapsGood)
    snaps = snaps[cond]
    pos = pos[cond]
    snapsCen = snapsCen[cond2]
    posCen = posCen[cond2]
    if verbose > 0:
        print("len snapsCen is now",len(snapsCen))
        print("len pos posCen = ",len(pos),len(posCen))
        
    # separations
    dpos = np.zeros((len(pos),3))
    for j in range(3): # loop over cartesian axes
        dpos_raw = pos[:,j]-posCen[:,j]
        dpos[:,j] = FixPeriodic(dpos_raw,sim=sim)
        if verbose >= 3:
            print("\nj=",j)
            print("pos posCen=",np.transpose([pos[:,j],posCen[:,j],dpos[:,j]]))

    # interpolate separations
    tabztall = np.flip(ztall(),axis=0)
    tlook_all = tabztall[:,-1]
    snaps_all = tabztall[:,0]
    cond3 = np.isin(snaps_all,snapsGood)
    tlook = tlook_all[cond3]
    if verbose >= 2:
        print(np.transpose([snaps_all, snapsCen]))
    tlookfine = np.linspace(tlook[0],tlook[-1],1001)
    dr = np.sqrt(np.sum(np.square(dpos),axis=1))
    dposfine = np.zeros((len(tlookfine),3))
    for j in range(3):
        f = interp1d(tlook,dpos[:,j],kind='cubic')
        dposfine[:,j] = f(tlookfine)
    if verbose >= 2:
        for j in range(3):
            plt.scatter(tlook,dpos[:,j],s=5,c='k')
            plt.plot(tlookfine,dposfine[:,j],'g-',lw=0.5)
        plt.show()
    drfine = np.sqrt(np.sum(np.square(dposfine),axis=1))

    # extract pericenters
    iperi = argrelextrema(drfine,np.less)
    iapo = argrelextrema(drfine,np.greater)
    tperi = tlookfine[iperi]
    rperi = drfine[iperi]
    tapo = tlookfine[iapo]
    rapo = drfine[iapo]
    if plot:
        plt.scatter(tlook,dr,s=5,c='k')
        plt.plot(tlookfine,drfine,'g-',lw=0.5)
        plt.scatter(tperi,rperi,s=50,facecolors='none',edgecolors='b',marker='v')
        plt.scatter(tapo,rapo,s=50,facecolors='none', edgecolors='r',marker='^')
        plt.xlabel('lookback time (Gyr)')
        plt.ylabel('separation (kpc/h)')
        plt.yscale('log')
        plt.savefig('periapo_' + str(subhalo) + '.pdf')
    return [tperi,rperi],[tapo,rapo]

def separation(pos1,pos2,sim='TNG50-1'):
    dpos = np.zeros((len(pos1),3))
    for j in range(3):
        dpos[:,j] = FixPeriodic(pos1[:,j] - pos2[:,j],sim=sim)
    return np.sqrt(np.sum(np.square(dpos),axis=1))

def selectMWs(sim='TNG50-1',lmmin=10.4,lmmax=11.2,
              covera_max=0.45,covera_inner=False,
              visual='only',centrals_only=True,verbose=0):
    """select Milky Way galaxies
        
    arguments:
        lmmin, lmmax: min and max log stellar masses (solar units) 
            [default 10.4, 11.2]
        covera_max: max c/a to select MW as a disk galaxy [default 0.45]
        covera_inner: use robust measure of c/a instead of full subhalo
                        [default: False]
        visual: use Pillepich visual flat classification: 
            'only' for only criterion of visual flatness
            'and' for (c/a < max) AND visual
            # 'or' for (c/a < max) OR visual
            'no' for disregard visual
        centrals_only: only select central subhalos [default: True]
        
    returns:
        df of selected subhalos, df of all subhalos in mass range, df of all groups
        
    author: Gary Mamon (gam AAT iap.fr)
    """
    # extract interesting parameters for all subhalos
    # params=['SubhaloMass','SubhaloMassType','SubhaloMassInRadType',
    #         'SubhaloPos','SubhaloVel','SubhaloHalfmassRadType',
    #         'SubhaloGrNr','SubhaloFlag','SubhaloGasMetallicity','SubhaloLenType',
    #         'SubhaloSFR','SubhaloSFRinRad','SubhaloSpin','SubhaloStarMetallicity',
    #         'SubhaloStellarPhotometrics','SubhaloVmax','SubhaloVmaxRad']
    if verbose > 0:
        print("reading subhalos...")
    
    # read joint subhalo group data
    df = ReadFITS('subhalos_groups',data_dir=tng_dir + sim + '/',
                  verbose=verbose)
    
    # restrict to stellar mass range and good flags
    dfM = df.loc[df.lM_stars.between(lmmin,lmmax) & df.SubhaloFlag]

    # restrict to centrals
    if centrals_only:
        dfM = dfM.loc[dfM.is_central]
    if verbose > 0:
        print("len df dfM = ",len(df),len(dfM))
         
    # read supplementary table: Stellar Circularities
    if verbose > 0:
        print("reading stellar circularities...")
    df_SC = getStellarCircularities()
    df_SC['covera'] = df_SC.MassTensorEigenVals0/df_SC.MassTensorEigenVals2
    df_SC['covera_inner'] = df_SC.ReducedMassTensorEigenVals0 \
                            /df_SC.ReducedMassTensorEigenVals2
                            
    # merge with Stellar Circularities  
    df2 = dfM.merge(df_SC,how='inner',left_on='SubhaloID',right_on='SubfindID')
    # df2.astype:({'SubhaloID':'int64','group_id':'int64'})

    # select flat systems:
    if verbose > 0:
        print("len df2=",len(df2))
        print("selecting flat galaxies...")
    if covera_inner:
        df_flat = df2.loc[df2.covera_inner<covera_max]
    else:
        df_flat = df2.loc[df2.covera<covera_max]
        
    # read supplemntary table of Pillepich+23 MWs
    
    if sim[0:5] != 'TNG50':
        print("not reading MW file for sim=",sim)
        return df_flat
    
    df_flat0 = df_flat.copy()
    if verbose > 0:
        print("selecting MWs...")
    df_MW = getMWM31(newcols=False)

    df_merge = dfM.merge(df_MW,how='inner',
                         left_on='SubhaloID',right_on='SubfindID')
    if visual == 'only':
        df_flat = df_merge.loc[df_merge.FlagDiskyVisual==1]
    # elif visual == 'or':
    #     df_flat = df_flat0.merge(df_merge,how='outer', on='SubhaloID')
    #     for kx in df_flat.columns:
    #         if ('is_central' in kx) or ('SubhaloFlag' in kx):
    #             print("***",kx)
    #         # print(kx,"...")
    #         if '_x' in kx:
    #         # if ('_x' in kx) & (kx != 'v_x') & (kx != 'SubhaloFlag_x') \
    #         # & (kx != 'is_central_x'):  
    #             print(kx)
    #             k = kx[0:-2]
    #             ky = k + '_y'
    #             if ky not in df_flat.columns:
    #                 print("!!!!",ky,"not in columns!")
    #             if np.min(np.abs(df_flat[ky]-df_flat[kx]) < 0.0001):
    #                 df_flat[k] = df_flat[kx]
    #             else:
    #                 # print("disgreement for",k)
    #                 # print(df_flat[[kx,ky]])
    #                 # sys.exit()
    #                 df_flat[k] = np.where(np.isnan(df_flat[ky]),
    #                                             df_flat[kx],df_flat[ky])
    #                 df_flat.drop(columns=[kx,ky])

    return df_flat
 
def rotCurve(sub,df,sim='TNG50-1',snapnum=99,extract=False,Rmax_spin=27):
    # extract particles
    cutout = ['Coordinates','Velocities','Masses','GFM_StellarPhotometrics']
    p = particles(sub,sim=sim,snapnum=snapnum,params=cutout,extract=extract)
    
    # relative positions
    dx = FixPeriodic(p['Coordinates'][:,0] - df.x,sim=sim)
    dy = FixPeriodic(p['Coordinates'][:,0] - df.y,sim=sim)
    dz = FixPeriodic(p['Coordinates'][:,0] - df.z,sim=sim)
    r = np.sqrt(x*x + y*y + z*z)
    
    # relative velocites (ONLY WORKS FOR z=0!!!)
    dv_x = p['Velocities'][:,0] - df.v_x
    dv_y = p['Velocities'][:,1] - df.v_y
    dv_z = p['Velocities'][:,2] - df.v_z
    
    dfgal = pd.DataFrame(data={'x':dx,'y':dy,'z':dz,'r':r,
                               'v_x':dv_x,'v_y':dv_y,'v_z':dv_z,
                               'm':p['Masses'][:]})
    
    # restrict to Rmax_spin
    dfgal_R = dfgal.loc[dfgal.r<Rmax_spin]
    
    # spin angular momentum
    pos = np.transpose([dfgal_R.x,dfgal_R.y,dfgal_R.z])
    vel = np.transpose([dfgal_R.v_x,dfgal_R.v_y,dfgal_R.v_z])
    
    # rotate to new axes
    pos_new, vel_new = rotate_to_angular_momentum_frame(pos,vel)
    
    
def rotate_to_angular_momentum_frame(positions, velocities):
    """author: ChatGPT """
    
    # Calculate the angular momentum vector
    angular_momentum = np.cross(positions, velocities)
    
    # Normalize the angular momentum vector
    angular_momentum /= np.linalg.norm(angular_momentum)
    
    # Calculate the new third axis (z-axis) of the frame
    z_axis = angular_momentum
    
    # Calculate the new first axis (x-axis) of the frame
    x_axis = np.cross(z_axis, [0, 0, 1])
    x_axis /= np.linalg.norm(x_axis)
    
    # Calculate the new second axis (y-axis) of the frame
    y_axis = np.cross(z_axis, x_axis)
    
    # Create a rotation matrix to transform to the new frame
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    print("shape rotation_matrix=",rotation_matrix.shape)
    print("shape positions=",positions.shape)
    
    # Rotate positions to the new frame
    rotated_positions = np.dot(rotation_matrix.T, positions)
    
    # Rotate velocities to the new frame
    rotated_velocities = np.dot(rotation_matrix.T, velocities)
    
    return rotated_positions, rotated_velocities
    
def test_rotate_to_angular_momentum_frame(sub,df,sim='TNG50-1',Rmax=None,
                                          extract=True):
    p = particles(sub,sim=sim,snapnum=99,extract=extract)
    df_gal = df.loc[df.SubhaloID==sub]
    print("len df_gal=",len(df_gal))
    print("x y z vx vy vz =",df_gal[['x','y','z','v_x','v_y','v_z']].values)
    
    # relative positions
    dx = FixPeriodic(p['Coordinates'][:,0] - df_gal.x.values,sim=sim)
    dy = FixPeriodic(p['Coordinates'][:,1] - df_gal.y.values,sim=sim)
    dz = FixPeriodic(p['Coordinates'][:,2] - df_gal.z.values,sim=sim)
    r = np.sqrt(dx*dx + dy*dy + dz*dz)
    
    plt.scatter(dx,dz,s=0.5)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('raw')
    plt.show()
    
    # relative velocites (ONLY WORKS FOR z=0!!!)
    dv_x = p['Velocities'][:,0] - df_gal.v_x.values
    dv_y = p['Velocities'][:,1] - df_gal.v_y.values
    dv_z = p['Velocities'][:,2] - df_gal.v_z.values
    
    # build dataframe
    dfgal = pd.DataFrame(data={'x':dx,'y':dy,'z':dz,'r':r,
                               'v_x':dv_x,'v_y':dv_y,'v_z':dv_z})
                               
    # restrict to Rmax_spin
    dfgal_R = dfgal.loc[dfgal.r<Rmax]
    
    # 2D positions and velocities
    pos = np.transpose([dfgal_R.x,dfgal_R.y,dfgal_R.z])
    vel = np.transpose([dfgal_R.v_x,dfgal_R.v_y,dfgal_R.v_z])
    
    # rotated
    pos_new, vel_new = rotate_to_angular_momentum_frame(pos,vel)
    # print("z min max = ",pos_new[:,2])
    
    # edge-on view
    plt.scatter(pos_new[:,0],pos_new[:,2],s=0.5)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('rotated')
    
def CartesianToRotated(pos,vel,mass=None,angmom=None,rmax_angmom=10,
                       method='abhner',verbose=0):
    """Cartesian components in frame where 
       the 3rd axis is aligned with the angular momentum vector 
       
    arguments:
        pos: positions ([N,3] array)
        vel: velocities ([N,3] array)
        mass: masses (N array)
        angmom: angular momentum ([N,3] array)
        rmax_angmom: maximum distance to center to compute angular momentum
        method: 'abhner' or 'gary' --> give same
        
    source:
        Abhner Pinto de Almeida
        slightly modified by Gary Mamon
    """

    # compute angular momentum if not supplied
    if mass is None:
        mass = np.ones(len(pos))
    if angmom is None:
        if mass is None:
            angmom = np.sum(np.cross(pos,vel),axis=0)
        else:
            angmom = np.sum(mass[:,None]*np.cross(pos,vel),axis=0)
        r = np.linalg.norm(pos,axis=1)
        cond = r < rmax_angmom
        _pos = pos[cond]
        _vel = vel[cond]
        _mass = mass[cond]
        angmom = np.sum(_mass[:,None]*np.cross(_pos,_vel),axis=0)
        
    # modulus of angular momentum
    J = np.sqrt(np.sum(angmom*angmom))    
    
    if method == 'abhner':
        # trigonometry to obtain matrix
        costheta = np.clip(angmom[2] / J,-1,1)
        theta = np.arccos(costheta)
        sintheta = np.clip(np.sin(theta),-1,1)    
        sinphi = angmom[1] / (J*sintheta)
        cosphi = angmom[0] / (J*sintheta)
        if verbose > 0:
            print("theta_deg cos(theta) sin(theta) sin(phi) cos(phi)=",
              theta*180/np.pi,costheta,sintheta,cosphi,sinphi)
            print("angmom = ",angmom/J)
         
        matrix = np.array([[costheta*cosphi , costheta*sinphi, -sintheta],	
    		       [-sinphi, cosphi, 0],
    		       [sintheta*cosphi  , sintheta*sinphi, costheta]])
    elif method == 'gary': # adapted from pynbody
        [jx,jy,jz] = angmom/J
        matrix = np.array([[jz,-jx*jy,jz],[0,jx*jx+jy*jy,jy],[-jx,-jy*jz,jz]])

    # new positions and velocities
    POS = np.dot(matrix, np.transpose(pos))
    VEL = np.dot(matrix, np.transpose(vel))
    
    POS = np.around(POS, decimals=10)
    VEL = np.around(VEL, decimals=10)

    return np.transpose(POS), np.transpose(VEL)

def MapGalaxy(sub,sim='TNG50-1',pos=None,vel=None,angmom=None,PartType=4,
              view='xz',extract=False,size=20,rotation=True,method='abhner',
              rmax_angmom=10,verbose=0):
    if pos is None:
        p = particles(sub,sim=sim,PartType=PartType,extract=extract)
        pos = p['Coordinates']/h
        vel = p['Velocities']
        m = p['Masses']
    elif m is None:
        m = np.ones(len(pos))
    # dens = np.log10(m)
    f = getsubhalo(sub,simulation=sim)
    pos0 = np.array([f['pos_x'],f['pos_y'],f['pos_z']])/h
    vel0 = np.array([f['vel_x'],f['vel_y'],f['vel_z']])
    dpos = FixPeriodic(pos-pos0,sim=sim)
    dvel = vel - vel0
    if rotation:
        POS, VEL = CartesianToRotated(dpos, dvel, m, rmax_angmom=rmax_angmom,
                                      verbose=verbose)
    else:
        POS, VEL = dpos, dvel
    coords = ['x','y','z']
    icoord0 = coords.index(view[0])
    icoord1 = coords.index(view[1])
    plt.hist2d(POS[:,icoord0],POS[:,icoord1],weights=m,bins=[100,100],norm='log',
               range=[[-size/2,size/2],[-size/2,size/2]])
    plt.xlabel('$' + view[0] + '$')
    plt.ylabel('$' + view[1] + '$')
    plt.xlim(-size/2,size/2)
    plt.ylim(-size/2,size/2)
    ax = plt.gca()
    ax.set_aspect('equal')
    
def RotationCurve(sub,sim='TNG50-1',cutout_request=None,
                  PartType=4,pos=None,vel=None,
                  angmom=None,rmax_angmom=10,
                  extract=False,Rmax=20,Rmin=0,Nbins=20,loglog=False,
                  vcirc=False,
                  title=True,savefig=False):
    if pos is None:
        p = particles(sub,sim=sim,cutout_request=cutout_request,
                      PartType=PartType,extract=extract)
        pos = p['Coordinates']/h
        vel = p['Velocities']
        m = p['Masses']
    # dens = np.log10(m)
    f = getsubhalo(sub,simulation=sim)
    pos0 = np.array([f['pos_x'],f['pos_y'],f['pos_z']])/h
    vel0 = np.array([f['vel_x'],f['vel_y'],f['vel_z']])
    dpos = FixPeriodic(pos-pos0,sim=sim)
    dvel = vel - vel0
    
    # positions and velocities in rotating frame
    POS,VEL = CartesianToRotated(dpos,dvel,m,rmax_angmom=rmax_angmom)
    
    # projected distances
    X = POS[:,0]
    Y = POS[:,1]
    R = np.sqrt(X*X + Y*Y)
    phi = np.arctan2(Y,X)
    vphi = -VEL[:,0]*np.sin(phi) + VEL[:,1]*np.cos(phi)
    
    # rotation curve

    vrot = np.zeros(Nbins)
    evrot = np.zeros(Nbins)
    if loglog:
        lR = np.log10(R)
        lRmin = np.log10(Rmin)
        lRmax = np.log10(Rmax)
        dlR = (lRmax-lRmin)/Nbins
        lRbins = np.arange(lRmin+0.5*dlR,lRmax+0.5*dlR,dlR)
        for j, lRb in enumerate(lRbins):
            cond = np.abs(lR-lRb) < dlR/2
            _vphi = vphi[cond]
            _m = m[cond]
            vrot[j] = np.average(_vphi,weights=_m)
            evrot[j] = np.std(_vphi)/np.sqrt(len(_m))
        Rbins = 10**lRbins
    else:
        dR = (Rmax-Rmin)/Nbins
        Rbins = np.arange(Rmin+0.5*dR,Rmax+0.5*dR,dR)
        for j, Rb in enumerate(Rbins):
            cond = np.abs(R-Rb) < dR/2
            _vphi = vphi[cond]
            _m = m[cond]
            vrot[j] = np.average(_vphi,weights=_m)
            evrot[j] = np.std(_vphi)/np.sqrt(len(_m))
            
    if vcirc: # predicted rotation curve
        Z = POS[:,2]
        r = np.sqrt(R*R + Z*Z)
        
        

    # plt.scatter(pos_new[:,0],pos_new[:,2],s=1)
    # ggu.plothex(pos_new[:,0],pos_new[:,2],z=None)
    plt.figure()
    plt.errorbar(Rbins,np.abs(vrot),evrot,ls='none',mfc='r',ecolor='k',marker='o')
    plt.xlabel('$R$ (kpc)')
    plt.ylabel('rotation curve (km/s)')
    if loglog:
        plt.xscale('log')
        plt.yscale('log')
        yticks = [5,10,20,50,100,200]
        yticklabels = [str(yt) for yt in yticks]
        plt.yticks(yticks)
        plt.gca().set_yticklabels(yticklabels)
    subhalo_str = str(sub)
    if title:
        plt.title(subhalo_str)
    if savefig:
        if loglog:
            plt.savefig(subhalo_str + '_loglog.pdf')
        else:
            plt.savefig(subhalo_str + '.pdf')

def SubhaloPot(sub,sim='TNG50-1',snap=99,verbose=0):
    particle_mostbound_id = History(sub,plot=False)['SubhaloIDMostbound'][0]
    lens = getsubhalo(sub,simulation=sim,snapnum=snap,
                      parameter=['len_gas','len_dm','len_stars'])
    types = [0,1,4]
    for i, typ in enumerate(types):
        if verbose > 0:
            print("type=",typ)
        # skip types with no particles
        if lens[i] == 0: 
            continue
        p = particles(sub,sim=sim,snapnum=snap,PartType=typ,
                  params=['ParticleIDs','Potential'],extract=True)
        if verbose > 0:
            print("converting to df...")
        df = ConvertDict(p,df=True)
        df_mostbound = df.loc[df['ParticleIDs']==particle_mostbound_id]
        if len(df_mostbound) > 0:
            return df_mostbound.Potential.values[0]
    raise ValueError("Could not find most bound particle")
    
def SubhaloOrbitalEnergy(sub,sim='TNG50-1',snap=99,verbose=0):
    particle_mostbound_id = History(sub,plot=False)['SubhaloIDMostbound'][0]
    lens = getsubhalo(sub,simulation=sim,snapnum=snap,
                      parameter=['len_gas','len_dm','len_stars'])
    types = [0,1,4]
    for i, typ in enumerate(types):
        if verbose > 0:
            print("type=",typ)
        # skip types with no particles
        if lens[i] == 0: 
            continue
        p = particles(sub,sim=sim,snapnum=snap,PartType=typ,
                  params=['ParticleIDs','Potential','Velocities'],extract=True)
        if verbose > 0:
            print("converting to df...")
        df = ConvertDict(p,df=True)
        df_mostbound = df.loc[df['ParticleIDs']==particle_mostbound_id]
        vel_sub = np.array(getsubhalo(sub,simulation=sim,snapnum=snap,
                             parameter=['vel_x','vel_y','vel_z']))
        kin_sub = 0.5*np.sum(vel_sub*vel_sub)
        if len(df_mostbound) > 0:
            pot = df_mostbound.Potential.values[0]
            kin_mostbound = \
                + 0.5*df_mostbound.Velocities0.values[0]**2 \
                + 0.5*df_mostbound.Velocities1.values[0]**2 \
                + 0.5*df_mostbound.Velocities2.values[0]**2
            return pot, kin_mostbound, kin_sub
    raise ValueError("Could not find most bound particle")
    

def OrbitalEnergy(sub,cen,sim='TNG50-1',snap=99):
    pot_sub = SubhaloPot(sub,sim=sim,snap=snap)
    pot_cen = SubhaloPot(can,sim=sim,snap=snap)
    # TO BE FINISHED!!!
    
def rhoofr(sub,sim='TNG50-1',snap=99,extract=True,
           rmin=0.01,rmax=100,Nbins=20,title=None,verbose=0):
    # subhalo center
    if verbose > 0:
        print("extracting subhalo...")
    pos_sub = getsubhalo(sub,simulation=sim,snapnum=snap,
                         parameter=['pos_x','pos_y','pos_z'])
    cutout_request = {'gas':'Coordinates,Masses',
                      'dm':'Coordinates',
                      'stars':'Coordinates,Masses'}
    m_DM = 450000.
    if verbose > 0:
        print("extracting particles...")
    p = particles(sub,sim=sim,snapnum=snap,cutout_request=cutout_request,
                  extract=extract)
    if verbose > 0:
        print("converting data...")
    all_Pos = []
    all_r = []
    all_Masses = []
    if Nbins <= 30:
        mec = 'k'
    else:
        mec = 'none'
    for i in [0,1,4]:
        pType = 'PartType' + str(i)
        if pType in p.keys():
            if i == 0:
                gas_Pos = FixPeriodic(p[pType]['Coordinates'][:]-pos_sub)
                gas_Masses = 1e10/h * p[pType]['Masses'][:]
                gas_r = np.sqrt(np.sum(gas_Pos*gas_Pos,axis=1))
                ggu.Plotrhoofr(gas_r, rmin, rmax, Nbins, weights=gas_Masses,title=title,
                               xlab='$r$ (kpc)',
                               ylab='3D density ($\mathrm{M}_\odot/\mathrm{kpc}^3$)',
                               label='gas',mfc='g',mec=mec)
                all_Pos.extend(gas_Pos)
                all_r.extend(gas_r)
                all_Masses.extend(gas_Masses)
            elif i == 1:
                dm_Pos = FixPeriodic(p[pType]['Coordinates'][:]-pos_sub)
                dm_Masses = m_DM*np.ones(len(dm_Pos))
                dm_r = np.sqrt(np.sum(dm_Pos*dm_Pos,axis=1))
                ggu.Plotrhoofr(dm_r, rmin, rmax, Nbins, weights=dm_Masses,
                               label='DM',mfc='purple',mec=mec,initialize=False)
                all_Pos.extend(dm_Pos)
                all_r.extend(dm_r)
                all_Masses.extend(dm_Masses)
            elif i == 4:
                star_Pos = FixPeriodic(p[pType]['Coordinates'][:]-pos_sub)
                star_Masses = 1e10/h * p[pType]['Masses'][:]
                star_r = np.sqrt(np.sum(star_Pos*star_Pos,axis=1))
                ggu.Plotrhoofr(star_r, rmin, rmax, Nbins, weights=star_Masses,
                               label='stars',mfc='b',mec=mec,initialize=False)
                all_Pos.extend(star_Pos)
                all_r.extend(star_r)
                all_Masses.extend(star_Masses)
    # return np.array(all_Pos),np.array(all_Masses)
    r = np.array(all_r)
    m = np.array(all_Masses)
    if verbose > 0:
        print("len r m = ", len(r), len(m))
    if verbose > 0:
        print("plotting...")
    if title == 'auto':
        title = str(sub)
    ggu.Plotrhoofr(r, rmin, rmax, Nbins, weights=m,title=title,mfc='k',
                   label='total',initialize=False)
    xlim = plt.xlim()
    ylim = plt.ylim()
    xmid = np.sqrt(xlim[0]*xlim[1])
    ymid = np.sqrt(ylim[0]*ylim[1])
    xx = np.geomspace(xmid/3,3*xmid,101)
    yy = ymid / (xx/xmid)**2
    plt.legend()
    plt.plot(xx,yy,'g-')
    plt.savefig('rhoofr_' + str(sub) + '.pdf')
    
    # add slope -2
    
    # return all_Pos,all_r,all_Masses
