"""Central style registry (lightweight + backward compatible).

This module replaces the previous "fully-expanded" style dictionaries with:

1) a small set of *base* definitions (e.g., Normal/SBC/MBC colors, markers), and
2) rule-based resolution for derived keys (e.g., *Empty*, *Error*, *Colorbar*,
   *ColorbarEdge*, *ColorbarEmpty*, *...Legend*).

The aim is to avoid a combinatorial explosion of entries like:
    SBCColorbarEmpty, SBCCentralError, NormalSatelliteDMrichError, ...

while keeping backward compatibility for your existing plotting code that does:
    colors.get(name, 'black')
    markers.get(name, None)
    titles.get(name, name)

If an old key is requested, we try to *derive* it from the base style + tokens
present in the key, and cache the result.

Author: Abhner P. de Almeida
"""

from __future__ import annotations


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def extract_NameBase(name: str) -> str:
    """Return the param/object name and sidename"""
    
    sidename = None
    specialcase = None
    side_first = False
    typename  = None

    if 'Scatter' in name or name == 'Bian et al. (2025)':
        name = name.replace('Scatter', '')
        
    if 'BlackLine' in name:
        name = name.replace('BlackLine', '')
        specialcase = 'BlackLine'
        
        if 'Empty' in name:
            name = name.replace('Empty', '')
            
    elif 'Empty' in name:
        name = name.replace('Empty', '')
        specialcase = 'Empty'
        
    elif 'Selected' in name:
        name = name.replace('Selected', '')
        specialcase = 'Selected'
        
    elif 'Colorbar' in name:
        name = name.replace('Colorbar', '')
        specialcase = 'Colorbar'
            
    
    for t in SIDE_TOKENS:
        if t in name:
            
            pos = name.find(t)
        
            if pos != -1:
                if sidename == None:
                    sidename = t
                    side_first = (pos == 0)
                else:
                    sidename = sidename + t
                
                name = name.replace(t, "")
    
    for t in TYPE_TOKENS:
        if t in name:
            typename = t

            
    return name, sidename, typename, specialcase, side_first


SIDE_TOKENS = (
    "Satellite", "DMrich", "DMpoor",
    "Central", "WithoutBH", "WithBH",
    "NotInteract", "Interact", "Dont",
    "LoseTheirGas"
)

TYPE_TOKENS = (
    "Type0", "Type1", "Type4"
)



# -----------------------------------------------------------------------------
# extract parameters
# -----------------------------------------------------------------------------


def colors(name: str) :
    name, sidename, typename, specialcase, side_first = extract_NameBase(name)
    if specialcase in ('Empty', 'Colorbar'):
        return 'white'
    
    elif specialcase == 'Selected' and sidename == 'Satellite':
        return 'black'
    
    
    elif side_first:
        return BASE_colors.get(sidename, 'black')

    else:
        return BASE_colors.get(name, 'black')


def edgecolors(name: str) :
    
    name, sidename, typename,  specialcase, side_first = extract_NameBase(name)

    if specialcase in ('Empty', 'Colorbar'):
        if side_first:
            return BASE_edgecolors.get(sidename, 'black')
        
        if sidename == None:
            return BASE_edgecolors.get(name, 'black')
        elif 'LoseTheirGas' in sidename:
            return BASE_edgecolors.get(sidename, 'k')
    
        else:
            return BASE_edgecolors.get(name, 'black')
    else:
        return 'black'


def lines(name: str) :
    name, sidename, typename, specialcase,  side_first = extract_NameBase(name)

    if typename != None:
        return BASE_lines.get(typename, 'solid')

    if sidename == None:
        return BASE_lines.get(name, 'solid')

    elif 'LoseTheirGas' in sidename:
        return BASE_lines.get(sidename, 'solid')
    
    return BASE_lines.get(sidename, 'solid')


def linesthicker(name: str) :
    name, sidename, typename,  specialcase,  side_first = extract_NameBase(name)
    
    if typename != None:
        return BASE_linesthicker.get(typename, 1.6)
    
    elif side_first:
        return BASE_linesthicker.get(sidename, 1.6)

    else:
        return BASE_linesthicker.get(name,  1.6)

def markers(name: str) :
    name, sidename, typename,  specialcase,  side_first = extract_NameBase(name)
    if side_first:
        return BASE_markers.get(sidename, 'o')

    else:
        if specialcase == None:
            return BASE_markers.get(name, 'o')

        elif 'Colorbar' in specialcase:
            if sidename == None:
                return BASE_markers.get(name+'Colorbar', 'o')

            else:
                return BASE_markers.get(name+'Colorbar', 'o')

        else:
            return BASE_markers.get(name, 'o')
        
def msize(name: str) :
    name, sidename, typename, specialcase,  side_first = extract_NameBase(name)
    
    
    if side_first:
        return BASE_msize.get(sidename, 8)

    else:
        if specialcase  == None :
            return BASE_msize.get(name, 8)
        
        elif 'Colorbar' in  specialcase:
            if sidename == None:
                return BASE_msize.get(name+'Colorbar', 8)
            elif 'LoseTheirGas' in sidename:
                return BASE_msize.get(sidename+'Colorbar', 8)
            else:
                return BASE_msize.get(name+'Colorbar', 8)
            
        else:
            return BASE_msize.get(name, 8)
        
    
def scales(name: str) :
    name, sidename, typename, specialcase,  side_first = extract_NameBase(name)
    
    if side_first:
        return BASE_scales.get(sidename, 'linear')

    else:
        return BASE_scales.get(name, 'linear')
    
def capstyles(name: str):
    name, sidename, typename, specialcase,  side_first = extract_NameBase(name)
    
    if side_first:
        return BASE_capstyles.get(sidename, 'projecting')

    else:
        return BASE_capstyles.get(name, 'projecting')
    
def titles(name: str) :
    name, sidename, typename, specialcase,  side_first = extract_NameBase(name)
    
    if side_first:
        return BASE_titles.get(sidename, sidename)

    else:
        return BASE_titles.get(name, name)



# -----------------------------------------------------------------------------
# Base parameters 
# -----------------------------------------------------------------------------

BASE_colors = {
        #Size classes
        'Normal': 'darkorange',
        'NormalDif': 'darkorange',

        'SBC': 'forestgreen',
        'MBC': 'royalblue',
        
        'Diffuse': 'crimson',
        'SubDiffuse': '#8c6bb1',
        'Compact': 'forestgreen',
        'SubCompact': 'royalblue',
        
        #Special classes
        'SBCBornYoung': 'lime',
        
        'BornYoungSystems': 'white',


        'TNGrage':  'gray',
        'TNGAllrage':  'gray',
        'TNGrageCentral':  'gray',

        'Selected':  'none',
        'SatelliteSelected':  'black',
        
        'BadFlag':  'red',
        'Satellite': 'none',
        'GMM': 'red',
        
        'AllSB_TEST': 'green',
        'BadFlagAllSB_TEST':  'red',
        'SBCSatelliteAllSB_TEST': 'tab:green',
        

        'SBCGamaColor': 'darkseagreen', 
        'MBCGamaColor': 'lightblue', 
        'NormalGamaColor': 'navajowhite', 
        'GAMAColor': 'crimson', 
        
        'SatelliteDMrich':  '#9467BD',
        'SatelliteDMpoor':  '#8C564B',
        'SatelliteNotInteract':  'blue',
        'SatelliteInteract':  'red',
        'Central':  'black',
        
        'LoseTheirGas': 'white',
        'DontLoseTheirGas': 'white',

        #Error
        'NormalError': 'tab:orange',
        'SBCError': 'tab:green',
        'MBCError': 'tab:blue',
        'DiffuseError': 'tab:red',
        'SubDiffuseError': 'tab:purple',
        
        'DiffuseCentralError': 'tab:red',
        'SubDiffuseCentralError': 'tab:purple',

        'SatelliteDMrichError':  'tab:purple',
        'SatelliteDMpoorError':  'tab:brown',
        'SatelliteNotInteractError':  'tab:blue',
        'SatelliteInteractError':   'tab:red',
        'CentralError':  'gray',
        
        #Random
        '0': 'red', 
        '1': 'sienna',
        '2': 'darkorange',
        '3': 'purple',
        '4': 'lime',
        '5': 'g',
        '6': 'dodgerblue',
        '7':  'b',
        '8': 'brown',
        '9': 'lawngreen', 
        '10': 'turquoise',
        '11': 'steelblue',
        '12':  'indigo',
        '13': 'tab:blue',
        '14': 'gold',   
        '15': 'deeppink',
        '16': 'tab:purple',
        '17': 'darkcyan',
        '18': 'tab:red',
        '19': 'olive',
        '20': 'navy',
        '21': 'teal',
        '22': 'mediumvioletred',
        '23': 'darkolivegreen',
        '24': 'coral',
        '25': 'slateblue',
        '26': 'chocolate',
        '27': 'firebrick',
        '28': 'darkmagenta',
        '29': 'tab:green',
       
}

BASE_edgecolors = {
    
        #Size classes
        'Normal': 'darkorange',
        'NormalDif': 'darkorange',

        'SBC': 'forestgreen',
        'MBC': 'royalblue',
        
        'Diffuse': 'crimson',
        'SubDiffuse': '#8c6bb1',
        'Compact': 'forestgreen',
        'SubCompact': 'royalblue',
        
        'BornYoungSystems': 'red',
        
        'LoseTheirGas': 'k',
        'DontLoseTheirGas': 'k',

        
        #Special classes
        
        'BadFlag':  'black',
        'Central': 'black',        
        
}
    
BASE_lines = {
        #Special classes

        'SatelliteDMrich': 'solid',
        'SatelliteDMpoor': (0, (10, 4)),
        'DMrich': 'solid',
        'DMpoor':  (0, (10, 8)),
        'Central': (0, (10, 8)),
        'WithoutBH': (0, (10, 8)),
        'WithBH': 'solid',
        
        'LoseTheirGas': 'solid',
        'DontLoseTheirGas': 'dashed',

        #EVOLUTION
        
        'Type0': (0, (10, 8)),
        'Type1': (0,(0.1,2)),
        'Type4': 'solid',
        
        'Mbar_Norm_Max': 'solid',
        'Mgas_Norm_Max': (0, (10, 8)),

        #Evolution
        
        'MassExNormalize': 'solid',
        'MassExNormalizeAll': (0, (10, 8)),
        'StellarMassExSituMinor': (0,(0.1,2)),
        'StellarMassExSituIntermediate': (0, (10, 8)),
        'StellarMassExSituMajor': 'solid',
        
        
        'GasMass_In_TrueRhpkpc': 'solid',
        'GasMass_Above_TrueRhpkpc': (0, (10, 8)),
        'sSFR_In_TrueRhpkpc': 'solid',
        'sSFR_Above_TrueRhpkpc': (0, (10, 8)),
    
}


BASE_scales = {
    #SUBHALO
    
    #Masses
    'Mgas_Norm_Max_99': 'log',
    'MDM_Norm_Max_99': 'log',
    'Mstar_Norm_Max_99': 'log',
    
    'Mgas_Norm_Max':   'log',
    'MDM_Norm_Max':   'log',
    'Mstar_Norm_Max':  'log',
    'Mbar_Norm_Max':  'log',
    
    'GasFrac_99': 'log',
    'StarFrac_99': 'log',
    'DMFrac_99': 'linear',
    
    'MDM_Norm_Max_99': 'log',
    
    'DMFrac_Birth': 'logit',

    #ExSitu
    'MassExNormalize':  'log',
    'MassExNormalizeAll':  'log',
    'Frac_ExSitu': 'log',
    #Group
    'GroupNsubsFinalGroup':  'log',
    'rOverR200Min': 'log',
    
    'rToRNearYoung': 'log',
    'r_over_R_Crit200': 'log',

    'rOverR200Mean': 'log',
    'rOverR200Mean_New': 'log',

    #Others
    
    'z_Birth': 'log',
    
    
    None: 'linear'
}

BASE_linesthicker = {
        
        'Normal': 1.1,
        'NormalDif': 1.1,

        'SBC': 1.1,
        'MBC': 1.1,
        
        'SatelliteDMrich': 1.,
        'SatelliteDMpoor': 1.,
        'Central':  0.8,
        
        
        #Evolution
        
        'Type0': 1.1,
        'Type1': 1.5,
        'Type4': 1.1,
        
        
        'GasMass_In_TrueRhpkpc': 0.8,
        'GasMass_Above_TrueRhpkpc': 0.8,
        'sSFR_In_TrueRhpkpc': 0.8,
        'sSFR_Above_TrueRhpkpc': 0.8,
        
        
}

BASE_markers = {
    #Size classes
    'Normal': 'o',
    'NormalDif': 'o',

    'SBC': 'o',
    'MBC': 'o',
    
    'SubDiffuse': 'o',
    'Diffuse': 'o',
    'SBCBornYoung': '^',

    #Special classes
    'TNGrage': 'o',
    'TNGAllrage': 'o',
    'TNGrageCentral':  'o',
    
    'BornYoungSystems': 'o',


    'Selected': 'o',
    'SatelliteSelected': 'o',
    'BadFlag': 'o',

    'SBCGamaColor': 'D', 
    'MBCGamaColor': 'o', 
    'NormalGamaColor': '^', 
    'GAMAColor': '*',
    
    #Colorbar
    'NormalColorbar': '^',
    'SBCColorbar': 'D',
    'MBCColorbar': 'o',
    
    'DontLoseTheirGasColorbar': 'o',
    'LoseTheirGasColorbar': 'o',
    
    'DontLoseTheirGas': 'o',
    'LoseTheirGas': 'o',
    
    'SubDiffuseColorbar': 'H',
    'DiffuseColorbar': 's',

    'GAMAColorbar': '*', 
    
    'SatelliteDontLoseTheirGasColorbar': '*',
    'SatelliteLoseTheirGasColorbar': 's',
    'CentralColorbar': 'o',



} 

BASE_msize = {
    #Size classes
    'Normal': 3.3,
    'NormalDif': 3.3,

    #Special classes
    'SBCBornYoung': 11,
    'TNGrage':  5,
    'TNGAllrage': 5,
    'TNGrageCentral':  5,
    


    'BadFlag':  5,
    'GAMAColor': 9.5, 


    #Colorbar
    'NormalColorbar': 8,
    'SubDiffuseColorbar': 8,
    'GAMAColorbar': 9.5, 
    
    'SatelliteDontLoseTheirGasColorbar': 36,
    'SatelliteLoseTheirGasColorbar': 12,
    
    # 'DontLoseTheirGasColorbar': 12,
    # 'LoseTheirGasColorbar': 8,
    
    # 'DontLoseTheirGas': 12,
    # 'LoseTheirGas': 8,
    

}

BASE_capstyles = {}


BASE_titles = {
    #Size classes
    'Normal': r'Normals',
    'SBC': r'Compacts$_\mathrm{SB}$',
    'SBCBornYoung': 'Young \n Compacts$_\mathrm{SB} $',
    'MBC': r'Compacts$_\mathrm{MB}$',
    
    'Diffuse': r'Diffuse',
    'SubDiffuse': r'Sub-Diffuse',
    'SBsat': 'Satellites in SB',
    #Special classes
    
    'BornYoungSystems': 'Young subhalos',

    
    'TNGrage':  'Central galaxies',
    'TNGAllrage': 'All galaxies',
    'TNGrageCentral':  'Central galaxies',

    'Selected':  'Selected',
    'SatelliteSelected': 'Satellite',
    'Satellite':  'Satellite',
    'Central':  'Central',
    'BadFlag':  'Bad flags',
    
    'AllSB_TEST': 'All',
    'BadFlagAllSB_TEST':  'Bad flags',
    'GMM': 'GMM',

    'SBCGamaColor': r'$\mathrm{Compacts_{SB}}$',
    'MBCGamaColor': r'$\mathrm{Compacts_{MB}}$',
    'NormalGamaColor': 'Normals', 
    'GAMAColor': 'GAMA',

    'DMrich': r'$f_\mathrm{DM} > 0.7$',
    'DMpoor': r'$f_\mathrm{DM} < 0.7$',
    
    'SatelliteDMpoorMetalRich': r'$f_\mathrm{DM} < 0.7$ and $\log(Z/Z_\odot)<0.3$',
    'SatelliteDMpoorMetalUltraRich': r'$f_\mathrm{DM} < 0.7$ and $\log(Z/Z_\odot)>0.3$',

    'DMFracHigher': r'$f_\mathrm{DM} > 0.93$',
    'DMFracLower': r'$f_\mathrm{DM} \leq 0.93$',
    
    'DMFracHigher': r'$f_\mathrm{DM} > 0.93$',
    'DMFracLower': r'$f_\mathrm{DM} \leq 0.93$',
    
    'SatelliteNotInteract': r'Sats loEnv',
    'SatelliteInteract': r'Sats hiEnv',

    'WithBH': 'With BH',
    'WithoutBH': 'Without BH',
    
    'SatelliteDMrich': r'Satellites $f_\mathrm{DM} > 0.7$',
    'SatelliteDMpoor': r'Satellites $f_\mathrm{DM} < 0.7$',

    'SatelliteDMpoorEntryToNoGas':  'With gas',
    'SatelliteDMpoorNoGasToFinal':  'After gas loss',

    'DontLoseTheirGasColorbar': 'Retain their gas',
    'LoseTheirGasColorbar': 'Lose their gas',
    
    'DontLoseTheirGas': 'Retain their gas',
    'LoseTheirGas': 'Lose their gas',
    
    'CentralColorbar': 'Central',
    
    
    #EVOLUTION
    
    #ExSitu 
    'MassExNormalize': 'by \n $M_{\mathrm{ex-situ}, \; z = 0}$',
    'MassExNormalizeAll': r'by $M_{\star, \; z = 0}$',
    'StellarMassExSituMinor': r'Minor mergers',
    'StellarMassExSituIntermediate': r'Intermediate mergers',
    'StellarMassExSituMajor': r'Major mergers',
    
    
    #Mass
    
    'GasMass_In_TrueRhpkpc':   r'$r \leq  r_{1/2,\; z=0}$',
    'GasMass_Above_TrueRhpkpc':   r'$r >  r_{1/2,\; z=0}$',
    'sSFR_In_TrueRhpkpc': r'$r <  r_{1/2,\; z=0}$',
    'sSFR_Above_TrueRhpkpc': r'$r > r_{1/2,\; z=0}$',
    
    'Mbar_Norm_Max': '$M_\star + M_\mathrm{gas}$',
    'Mgas_Norm_Max': '$M_\mathrm{gas}$',
    
    #SPECIAL
    'Type0': 'Gas',
    'Type1': 'DM',
    'Type4': 'Stars',
    

}

## -----------------------------------------------------------------------------
# Labels 
# -----------------------------------------------------------------------------

labelsequal = {
    #SPECIAL
    'Type0': 'Gas',
    'Type1': 'DM',
    'Type4': 'Stars',
    
    #SUBHALO
    
    #Sizes
    'logHalfRadstar_99': r'$\log(r_{1/2, \, z = 0}/\mathrm{kpc})$',

    'SubhaloHalfmassRadType0': r'$\log(r_{1/2}/\mathrm{kpc})$',  
    'SubhaloHalfmassRadType1': r'$\log(r_{1/2}/\mathrm{kpc})$',  
    'SubhaloHalfmassRadType4': r'$\log(r_{1/2}/\mathrm{kpc})$',  


    #Masses
    'SubhaloMassType0': r'$\log(M/\mathrm{M}_\odot)$',
    'SubhaloMassType1': r'$\log(M/\mathrm{M}_\odot)$',
    'SubhaloMassType4': r'$\log(M/\mathrm{M}_\odot)$',
    
    'Mgas_Norm_Max_99':r'$M_{z = 0}/ M_\mathrm{max}$',
    'MDM_Norm_Max_99': r'$M_{z = 0}/ M_\mathrm{max}$',
    'Mstar_Norm_Max_99': r'$M_{z = 0}/ M_\mathrm{max}$',
    
    'Mgas_Norm_Max':r'$M/ M_\mathrm{max}$',
    'MDM_Norm_Max': r'$M/ M_\mathrm{max}$',
    'Mstar_Norm_Max': r'$M/ M_\mathrm{max}$',
    'Mbar_Norm_Max': r'$M/ M_\mathrm{max}$',

    
    'GasMass_In_Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_TrueRhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_TrueRhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    
    #SFR
    'SubhalosSFRInHalfRad': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFRCoreRatio': r'$\mathrm{sSFR} / \mathrm{sSFR}}$',
    'sSFR_Ratio_07_5_pkpc': r'$\mathrm{sSFR}_{r < 0.7  \, \mathrm{kpc}} / \mathrm{sSFR}_{r < 5 \,  \mathrm{kpc}}}$',
    'sSFR_Ratio_05_2p5_pkpc': r'$\mathrm{sSFR}_{r < 0.5  \, \mathrm{kpc}} / \mathrm{sSFR}_{r < 2.5 \,  \mathrm{kpc}}$',
    'sSFR_In_Rhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_Above_Rhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_TrueRhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_Above_TrueRhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    
    'SubhalosSFR': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',

    #ExSitu
    'MassExNormalize': r'Normalized $M_\mathrm{ex-situ}$',
    'MassExNormalizeAll': r'Normalized $M_\mathrm{ex-situ}$',
    'StellarMassExSituMinor': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)$',
    'StellarMassExSituIntermediate': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)$',
    'StellarMassExSituMajor': r'$\log(M_\mathrm{ex-situ}/\mathrm{M}_\odot)$',
    
    'ExMassType0Evolution': r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType1Evolution': r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType4Evolution': r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',
    
    'RadIn': r'$\overline{r}_\mathrm{in-situ} [\mathrm{kpc}]$',
    'RadEx': r'$\overline{r}_\mathrm{ex-situ}  [\mathrm{kpc}]$',
    'SigmaIn': r'$\sigma_\mathrm{v, in-situ}  [\mathrm{km\, s}^{-1}]$',
    
    'logSUM_Mstar_merger_Corotate' : r'$\log(M_{\star,\,\mathrm{mergers}} / \mathrm{M}_\odot)$',
    'logSUM_Mstar_merger_Perpendicular' : r'$\log(M_{\star,\,\mathrm{mergers}} / \mathrm{M}_\odot)$',
    'logSUM_Mstar_merger_Counterotating' : r'$\log(M_{\star,\,\mathrm{mergers}} / \mathrm{M}_\odot)$',
    
    #Group
    'GroupNsubsFinalGroup': r'Number of satellites',
    'Group_M_Crit200FinalGroup': r'$\log(M_{200}/\mathrm{M}_\odot)$',
    
    'rOverR200Min': r'$(R/R_{200})_\mathrm{min}$',
    
    'M200Mean': r'$\overline{\log(M_{200} / \mathrm{M}_\odot)}$',
    'rOverR200Mean': r'$\overline{(\mathrm{R} / \mathrm{R}_{200})}$',
    'rOverR200Mean_New': r'$\overline{(\mathrm{R} / \mathrm{R}_{200})}$',

    #AngularMomentum
    'MassTensorEigenVals': r'$\mu_1 / \sqrt{\mu_2 \mu_3}$',
    'logjProfile': r'$\log (j_{\mathrm{gas}} / \, \, [\mathrm{kpc \; km  \; s^{-1}}])$',
    
    #Specific time
    'LBTimeMajorMerger': 'Lookback time  [Gyr]',
    'LBTimeMinorMerger': 'Lookback time  [Gyr]',
    'LBTimeIntermediateMerger': 'Lookback time  [Gyr]',
    
    'z_At_FinalEntry':  r'$z_{\mathrm{infall}}$ in final host',
    'SnapLostGas': 'Gas loss lookback time  [Gyr]', #$M_\mathrm{gas} = 0$',

    #Others
    
    'Decrease_Entry_To_NoGas_Norm_Delta': r'$(\Delta r_{1/2} / (r_{1/2}^\mathrm{entry}  \Delta t))^\mathrm{entry-to-gas-loss}\, \mathrm{[Gyr^{-1}]}$',
    'Decrease_NoGas_To_Final_Norm_Delta': r'$(\Delta r_{1/2} / (r_{1/2}^\mathrm{no-gas}  \Delta t))^\mathrm{no-gas}\, \mathrm{[Gyr^{-1}]}$', 
    
    'Decrease_Entry_To_NoGas': r'$(\Delta r_{1/2} / r_{1/2}^\mathrm{entry} )^\mathrm{entry-to-gas-loss}$',
    'Decrease_NoGas_To_Final': r'$(\Delta r_{1/2} / r_{1/2}^\mathrm{entry} )^\mathrm{no-gas}$',
    
    'dt_Entry_To_NoGas_Gyr': r'$(\Delta t)$^\mathrm{entry-to-gas-loss}',
    
    'l200': r'$\lambda_{200}$',
    
    'l200_NewMeanAfter1Gyr': r'$\lambda_{200}$',
    'l200_NewMeanAfter5Gyr': r'$\lambda_{200}$',
    'l200_NewMeanAfter8Gyr': r'$\lambda_{200}$',

    'l200_New_at_99': r'$\lambda_{200}$',
    
    'CumulativeCorotateFraction_at_1': r'$f_{\mathrm{acc,\, Corotate}}$',
    'CumulativeCorotateFraction_at_5': r'$f_{\mathrm{acc,\, Corotate}}$',
    'CumulativeCorotateFraction_at_8': r'$f_{\mathrm{acc,\, Corotate}}$',

    'CumulativeCorotateFraction': r'$f_{\mathrm{acc,\, Corotate}}$',


    }

labels = {
    #SPECIAL
    'Type0': 'Gas',
    'Type1': 'DM',
    'Type4': 'Stars',
    
    #SUBHALO
    
    #Sizes
    'logHalfRadstar_99': r'$\log(r_{1/2, \, z = 0}/\mathrm{kpc})$',

    'SubhaloHalfmassRadType0': r'$\log(r_{1/2, \mathrm{gas}}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType1': r'$\log(r_{1/2, \mathrm{DM}}/\mathrm{kpc})$',
    'SubhaloHalfmassRadType4': r'$\log(r_{1/2}/\mathrm{kpc})$',
    
    'deltaSize_at_Entry':  r'$[(r_{{1/2}} -\left<r_{{1/2}}\right>) / \sigma_{r_{{1/2}}}]^{\mathrm{entry}}$ ',
    
    'Relative_Rhalf_MaxProfile_Minus_HalfRadstar_Entry': r'$(r_{1/2,\; \mathrm{iSP}} - r_{1/2, z_\mathrm{entry}})/  r_{1/2, z_\mathrm{entry}}$',
    'Relative_Rhalf_MinProfile_Minus_HalfRadstar_Entry': r'$(r_{1/2,\; \mathrm{oSP}} - r_{1/2, z_\mathrm{entry}}) / r_{1/2, z_\mathrm{entry}}$',


    #Masses
    'SubhaloMassType0': r'$\log(M_{\mathrm{gas}}/\mathrm{M}_\odot)$',
    'SubhaloMassType1': r'$\log(M_{\mathrm{DM}}/\mathrm{M}_\odot)$',
    'SubhaloMassType4': r'$\log(M_{\star}/\mathrm{M}_\odot)$',
    
    'Mgas_Norm_Max_99':r'$(M_{z = 0}/ M_\mathrm{max})_\mathrm{gas}$',
    'MDM_Norm_Max_99': r'$(M_{z = 0}/ M_\mathrm{max})_\mathrm{DM}$',
    'Mstar_Norm_Max_99': r'$(M_{z = 0}/ M_\mathrm{max})_\star$',
    
    'Mgas_Norm_Max':r'$(M/M_\mathrm{max})_\mathrm{gas}$',
    'MDM_Norm_Max': r'$(M/ M_\mathrm{max})_\mathrm{DM}$',
    'Mstar_Norm_Max': r'$(M/ M_\mathrm{max})_\star$',
    'Mbar_Norm_Max': r'$(M/ M_\mathrm{max})_\mathrm{bar}$',

    'GasFrac_99': r'$(M_\mathrm{gas}/ M)_{z = 0}$',
    'StarFrac_99': r'$(M_\star/ M)_{z = 0}$',
    'DMFrac_99': r'$f_{\rm DM,\,z=0}$',
    
    'MassIn_Infall_to_GasLost': r'$(\Delta M_\star)_{\mathrm{inner}}^\mathrm{entry-to-gas-loss} / M_\star^\mathrm{entry}$', #'Relative inner stellar mass \n change during period', #
    'MassAboveAfterInfall_Lost': r'$(\Delta M_\star)_{\mathrm{outer}} ^\mathrm{no-gas} / M_{\star}^{\mathrm{gas-loss}}$', #r'$(\Delta M_\star)_{r > r_{1/2, z = 0},  M_\mathrm{gas, \, min} \mathrm{\, to \,} z = 0} / M_{\star, M_\mathrm{gas, \, min} }$',
    'MassAboveAfter_Infall_to_GasLost': r'$(\Delta M_\star)_{\mathrm{outer}}^\mathrm{entry-to-gas-loss} / M_\star^{\mathrm{entry}}$', #'Relative inner stellar mass \n change during period', #
    
    'GasMass_In_Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_Rhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_In_TrueRhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    'GasMass_Above_TrueRhpkpc':  r'$\log(M_{\mathrm{gas}}/\mathrm{M_\odot})$',
    
    'DMFrac_Birth': r'$(M_\mathrm{DM}/M)_\mathrm{birth}$',

    
    #SFR
    'SubhalosSFRInHalfRad': r'$\log(\mathrm{sSFR}_{r < r_{1/2}}/\mathrm{yr}^{-1})$',
    'sSFRCoreRatio': r'$\mathrm{sSFR}_{r < r_{1/2}} / \mathrm{sSFR}_{r > r_{1/2}}$',
    'sSFR_Ratio_07_5_pkpc': r'$\mathrm{sSFR}_{r < 0.7 \, \mathrm{kpc}} / \mathrm{sSFR}_{r < 5 \, \mathrm{kpc}}$',
    'sSFR_Ratio_05_2p5_pkpc': r'$\mathrm{sSFR}_{r < 0.5  \mathrm{\,kpc}} / \mathrm{sSFR}_{r < 2.5 \mathrm{\,kpc}}$',
    'SubhalosSFRInHalfRad': r'$\log(\mathrm{sSFR}_{r < r_{1/2}}/\mathrm{yr}^{-1})$',
    'sSFRinHalfRadAfterz5': r'$\overline{\log{(\mathrm{sSFR}_{r < r_{1/2}}/\mathrm{yr^{-1}})}}_{z < 5}$',

    'log_sSFR__fixed_0p5pkpc': r'$\log{(\mathrm{sSFR}_{r < 0.5  \mathrm{\,kpc}}/\mathrm{yr^{-1}})}$',

    'sSFR_In_Rhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_Above_Rhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_In_TrueRhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    'sSFR_Above_TrueRhpkpc': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    
    'sSFRTrueInner_BeforeEntry': r'$\log{(\overline{\mathrm{sSFR}}/\mathrm{yr^{-1}}})_{\mathrm{inner}}^{\mathrm{entry}}$',
    'sSFRTrueInner_Entry_to_Nogas': r'$\log{(\overline{\mathrm{sSFR}}/\mathrm{yr^{-1}}})_{\mathrm{inner}}^{\mathrm{entry-to-gas-loss}}$',
    'sSFRTrueRatio_Entry_to_Nogas':  r'$(\overline{\mathrm{sSFR}_\mathrm{inner}/\mathrm{sSFR}_{\mathrm{outer}}})^{\mathrm{entry-to-gas-loss}}$',

    'sSFRRatioPericenter': r'$(\overline{\mathrm{sSFR}_\mathrm{After}} / \overline{\mathrm{sSFR}_\mathrm{Before}})^\mathrm{1st\,pericenter}_\mathrm{inner}$ ',

    'SubhalosSFR': r'$\log(\mathrm{sSFR}/\mathrm{yr}^{-1})$',
    
    #ExSitu
    'MassExNormalize': r'$(M_{\mathrm{ex-situ}} / M_{\mathrm{ex-situ},\; z = 0})$',
    'MassExNormalizeAll': r'$(M_{\mathrm{ex-situ}} / M_{\star,\; z = 0})$',
    
    'Frac_ExSitu': r'$(M_{\mathrm{ex-situ}} / M_{\star})$',
    
    'StellarMassExSitu': r'$\log(M_{\mathrm{ex-situ}}/\mathrm{M}_\odot)$',

    'StellarMassExSituMinor': r'$\log(M_{\mathrm{ex-situ,\; minor\; merger}}/\mathrm{M}_\odot)$',
    'StellarMassExSituIntermediate': r'$\log(M_{\mathrm{ex-situ,\; intermediate\; merger}}/\mathrm{M}_\odot)$',
    'StellarMassExSituMajor': r'$\log(M_{\mathrm{ex-situ,\; major\; merger}}/\mathrm{M}_\odot)$',
    
    'ExMassType0Evolution': r'$\log(M_{\mathrm{gas, \; ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType1Evolution': r'$\log(M_{\mathrm{DM, \; ex-situ}}/\mathrm{M}_\odot)$',
    'ExMassType4Evolution': r'$\log(M_{\mathrm{\star, \; ex-situ}}/\mathrm{M}_\odot)$',
    
    'RadIn': r'$\overline{r}_\mathrm{in-situ} [\mathrm{kpc}]$',
    'RadEx': r'$\overline{r}_\mathrm{ex-situ}  [\mathrm{kpc}]$',
    'SigmaIn': r'$\sigma_\mathrm{v, in-situ}  [\mathrm{km\, s}^{-1}]$',
    
    'logSUM_Mstar_merger_Corotate' : r'$\log(M_{\star,\,\mathrm{mergers}} / \mathrm{M}_\odot)$',
    'logSUM_Mstar_merger_Perpendicular' : r'$\log(M_{\star,\,\mathrm{mergers}} / \mathrm{M}_\odot)$',
    'logSUM_Mstar_merger_Counterotating' : r'$\log(M_{\star,\,\mathrm{mergers}} / \mathrm{M}_\odot)$',

    #Group
    'GroupNsubsFinalGroup': 'Number of satellites \n Final host',
    'Group_M_Crit200FinalGroup': r'$\log(M_{200}/\mathrm{M}_\odot)$',
    
    'rOverR200Min': r'$(R/R_{200})_\mathrm{min}$',
    
    'M200Mean': r'$\overline{\log(M_{200} / \mathrm{M}_\odot)}$',
    'rOverR200Mean': r'$\overline{(\mathrm{R} / \mathrm{R}_{200})}$',
    'rOverR200Mean_New': r'$\overline{(\mathrm{R} / \mathrm{R}_{200})}$',
    
    'rToRNearYoung': r'$d_{\mathrm{NNB}}$ [kpc]', 
    'r_over_R_Crit200': r'$R/R_{200}$',


    #AngularMomentum
    'MassTensorEigenVals': r'$\mu_1 / \sqrt{\mu_2 \mu_3}$',
    'logjProfile': r'$\log (j_{\mathrm{gas}} / \, \, [\mathrm{kpc \; km  \; s^{-1}}])$',
    
    #Specific time
    'LBTimeMajorMerger': 'Lookback time \n Major Merger [Gyr]',
    'LBTimeMinorMerger': 'Lookback time \n  Minor Merger [Gyr]',
    'LBTimeIntermediateMerger': 'Lookback time \n Intermediate Merger [Gyr]',
    
    'z_At_FinalEntry':  r'$z_{\mathrm{infall}}$ in final host',
    'z_At_FirstEntry':  r'$z_{\rm entry}$',
    'SnapLostGas': 'Gas loss lookback time  [Gyr]', #$M_\mathrm{gas} = 0$',

    'z_Birth': r'$z_\mathrm{birth}$',

    #Others
    'U-r': r'$u-r \; [\mathrm{mag}]$',
    'ratio_color_U_minus_r_0p5xRh': r'$[u-r]_\mathrm{inner} - [u-r]_\mathrm{outer} $',
    'ratio_color_U_minus_r_1xRh': r'$[u-r]_\mathrm{inner} - [u-r]_\mathrm{outer} $',
    'inner_color_U_minus_r_1xRh': r'$[u-r]_\mathrm{inner}$',
    'outer_color_U_minus_r_1xRh': r'$[u-r]_\mathrm{outer}$',
    'global_color_U_minus_r_1xRh': r'$[u-r]_\mathrm{global}$',

    'Decrease_Entry_To_NoGas_Norm_Delta': r'$(\Delta r_{1/2} / (r_{1/2}^\mathrm{entry}  \Delta t))^\mathrm{entry-to-gas-loss}\, \mathrm{[Gyr^{-1}]}$',
    'Decrease_NoGas_To_Final_Norm_Delta': r'$(\Delta r_{1/2} / (r_{1/2}^\mathrm{no-gas}  \Delta t))^\mathrm{no-gas}\, \mathrm{[Gyr^{-1}]}$', 
   
    'Decrease_Entry_To_NoGas': r'$(\Delta r_{1/2} / r_{1/2}^\mathrm{entry} )^\mathrm{entry-to-gas-loss}$',
    'Decrease_NoGas_To_Final': r'$(\Delta r_{1/2} / r_{1/2}^\mathrm{entry} )^\mathrm{no-gas}$',
    
    'dt_Entry_To_NoGas_Gyr': r'$(\Delta t)^\mathrm{entry-to-gas-loss}$',
    'dt_NoGas_OVER_dt_EntryToGasLoss': r'$(\Delta t)^\mathrm{no-gas} / (\Delta t)^\mathrm{entry-to-gas-loss} $',

    'Relative_logZ_At_Entry': r'$[\log Z_\star -\left<\log Z_\star\right>]^{\mathrm{entry}}$ ',
    'Relative_logInnerZ_At_Entry': r'$[\log Z_\star -\left<\log Z_\star\right>]^{\mathrm{entry}}_\mathrm{inner}$ ',
    'logStarZ_99': r'$\log( Z_\star / Z_\odot)_{z = 0}$',
    'logStarZ_99_75dex': r'$\log( Z_\star / Z_\odot)_{z = 0}$',
    'l200': r'$\lambda_{200}$',
    
    'l200_NewMeanAfter1Gyr': r'$\lambda_{200}$',
    'l200_NewMeanAfter5Gyr': r'$\lambda_{200}$',
    'l200_NewMeanAfter8Gyr': r'$\lambda_{200}$',

    'l200_New_at_99': r'$\lambda_{200}$',
    
    'CumulativeCorotateFraction_at_1': r'$f_{\mathrm{acc,\, Corotate}}$',
    'CumulativeCorotateFraction_at_5': r'$f_{\mathrm{acc,\, Corotate}}$',
    'CumulativeCorotateFraction_at_8': r'$f_{\mathrm{acc,\, Corotate}}$',

    'CumulativeCorotateFraction': r'$f_{\mathrm{acc,\, Corotate}}$',


    
    }

texts = {
    #SPECIAL
    'Type0': 'Gas',
    'Type1': 'DM',
    'Type4': 'Stars',
    
    #SUBHALO
    
    #Sizes
    'SubhaloHalfmassRadType0': r'$r_{1/2, \mathrm{gas}}$',
    'SubhaloHalfmassRadType1': r'$r_{1/2, \mathrm{DM}}$',
    'SubhaloHalfmassRadType4': r'$r_{1/2, \star}$',


    #Masses
    'SubhaloMassType0': r'$M_{\mathrm{gas}}$',
    'SubhaloMassType1': r'$M_{\mathrm{DM}}$',
    'SubhaloMassType4': r'$M_{\star}$',
    
    #SFR
    'SubhalosSFRInHalfRad': r'$\mathrm{sSFR}_{r < r_{1/2}}$',
    'sSFRCoreRatio': r'$\mathrm{sSFR}_{r < r_{1/2}} / \mathrm{sSFR}_{r > r_{1/2}}$',

    #ExSitu
    'MassExNormalize': r'(M_{\star, \mathrm{ex-situ}} / M_{\star, \mathrm{ex-situ}, z = 0})',
    'MassExNormalizeAll': r'(M_{\star, \mathrm{ex-situ}} / M_{\star})',
    'StellarMassExSituMinor': r'$M_{\mathrm{ex-situ,\; minor\; merger}}$',
    'StellarMassExSituIntermediate': r'$M_{\mathrm{ex-situ,\; intermediate\; merger}}$',
    'StellarMassExSituMajor': r'$M_{\mathrm{ex-situ,\; major\; merger}}$',
    
    'ExMassType0Evolution': r'$M_{\mathrm{gas, \; ex-situ}}$',
    'ExMassType1Evolution': r'$M_{\mathrm{DM, \; ex-situ}}$',
    'ExMassType4Evolution': r'$M_{\mathrm{\star, \; ex-situ}}$',
    
    #Group
    'Group_M_Crit200FinalGroup': r'$M_{200}$',
    'GroupNsubsFinalGroup': 'Number of satellites \n Final host',
    
    'rOverR200Min': r'$(R/R_{200})_\mathrm{min}$',

    #AngularMomentum
    'MassTensorEigenVals': r'$\mu_1 / \sqrt{\mu_2 \mu_3}$',
    'logjProfile': r'$\log (j_{\mathrm{gas}} / \, \, [\mathrm{kpc \; km  \; s^{-1}}])$',
    
    'l200_NewMeanAfter1Gyr': r'$z = 5.5$', #r'$1$ Gyr after birth',
    'l200_NewMeanAfter5Gyr': r'$z = 1.2$', #r'$5$ Gyr after birth',
    'l200_NewMeanAfter8Gyr': r'$z = 0.6$', # r'$8$ Gyr after birth',
    'l200_New_at_99': r'$z = 0$', # r'$8$ Gyr after birth',


    }