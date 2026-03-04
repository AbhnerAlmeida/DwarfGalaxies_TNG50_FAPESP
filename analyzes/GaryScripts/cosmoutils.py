#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mathutils as mu

"""
Created on Thu Feb  6 18:11:56 2020

@author: gam
"""

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

def LookbackTime(Omegam0,h,z,zobs=0):
    """lookback time for given redshift
    args:
        Omegam0: density parameter at z=0
        h:       H_0/(100 km/s/Mpc)
        z:       redshift
    author: Gary Mamon"""
    return AgeUniverse(Omegam0,h,zobs)- AgeUniverse(Omegam0,h,z)    

def Mass_tilde_NFW(X):
    """Dimensionless mass profile for an NFW profile.
    arg: X = R/r_{-2}( positive float or array of positive floats), where r_{-2} is scale radius (slope -2)
    returns: M(r)/M(r_{-2}) (float, or array of floats)
    """
    return (np.log(X+1)-X/(X+1)) / (np.log(2.)-0.5)

def SurfaceDensity_tilde_NFW(X):
    """Dimensionless cluster surface density for an NFW profile. 
    arg: X = R/r_{-2} (positive float or array of positive floats), where r_{-2} is scale radius (slope -2)
    returns: Sigma(r_{-2} X) / [N(r_{-2})/pi r_{-2}^2] (float, or array of floats)"""

    # author: Gary Mamon

    #  t = type(X)
    
    # stop with error message if input values are 0 (infinite surface density) or < 0 (unphysical)
    minX = np.min(X)
    if np.min(X) <= 0.:
        raise ValueError("ERROR in SurfaceDensity_tilde_NFW: min(X) = ", 
                         minX, " cannot be <= 0")

    # compute dimensionless surface density
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 1.2x10^-6)
    denom          = np.log(4.)-1.
    Xminus1        = X-1.
    Xsquaredminus1 = X*X - 1.
    return   ( np.where(abs(Xminus1) < 0.001, 
                        1./3. - 0.4*Xminus1, 
                        (1. 
                         - mu.ACO(1./X) / np.sqrt(abs(Xsquaredminus1))
                         ) 
                        / Xsquaredminus1 
                        ) \
               / denom 
             )

def SurfaceDensity_hat_NFW(X,c):
    arg = c*c/(2*np.pi*(np.log(c+1)-c/(c+1)))
    Xminus1 = X - 1.
    Xsquaredminus1 = X*X - 1.
    return arg * np.where(abs(Xminus1) < 0.001, 
                        1./3. - 0.4*Xminus1, 
                        (1. 
                         - mu.ACO(1./X) / np.sqrt(abs(Xsquaredminus1))
                         ) 
                        / Xsquaredminus1 
                        ) 

# RoverRvir = float(input("Enter R/r_vir: "))
# c = float(input("Enter c: "))
# print("Surf_tilde = ", SurfaceDensity_tilde_NFW(c*RoverRvir))
# print("Surf_hat = ", SurfaceDensity_hat_NFW(c*RoverRvir,c))

def cofM(M,h=1,slope=-0.098,norm=6.76):
    """Concentration vs. mass"""
    hM12 = h*M / 1e12
    return norm * hM12**slope

def MockNFW(N=1000,roverrs_max=4, return_r=False):
    """Mock NFW model
        arguments:
            N: number of points (default 1000)
            roverrs_max: max r/r_s (default 4)
            return_r: return only radial coordinates? (default False)
        returns: radial coordinates if return_r=True, else cartesian coordinates
        Author: Gary Mamon (gam AAT iap.fr)
    """
    # radii
    qr = np.random.rand(N)
    asinhroverrs = np.linspace(0,np.arcsinh(roverrs_max),101)
    roverrs_tmp = np.sinh(asinhroverrs)
    m = du.Number_tilde_NFW(roverrs_tmp)
    movermaxm = m/np.max(m)
    cs = CubicSpline(movermaxm,asinhroverrs)
    asinhrRandom = cs(qr)
    rRandom = np.sinh(asinhrRandom)
    if return_r:
        return rRandom
    
    # other 2 spherical coordinates
    qtheta = np.random.rand(N)
    theta = np.arccos(1-2*qtheta)
    qphi = np.random.rand(N)
    phi = 2*np.pi*qphi
    
    # cartesian coordinates
    xRandom = rRandom*np.sin(theta)*np.cos(phi)
    yRandom = rRandom*np.sin(theta)*np.sin(phi)
    zRandom = rRandom*np.cos(theta)
    return xRandom,yRandom,zRandom

def CenterIterative(x,y,z,nmin=100,factor=2):
    n = len(x)
    pos = np.array([x,y,z])
    ipass = 0
    while n >= nmin:
        ipass += 1
        print("ipass = ", ipass, "n=",n)
        # center at median
        x0 = np.median(x)
        y0 = np.median(y)
        z0 = np.median(z)
        
        # radius
        r = radius(x,y,z,x0,y0,z0)
        rmax = np.max(r)
        
        # extract sphere around new center out to rmax/factor
        x = x[r < rmax/factor]
        y = y[r < rmax/factor]
        z = z[r < rmax/factor]
        n = len(x)
    return x0,y0,z0
        
def radius(x,y,z,x0,y0,z0):
    return np.sqrt((x-x0)**2 + (y-y0)**2 + (z-z0)**2)

def convertYangvir(lMYang,z,h=1,Omegam=0.3):
    """Convert Yang group catalog log virial mass to log_10(M200c/MSun) and r200c/kpc
    author: Gary Mamon (gam AAT iap.fr)
    source: Appendix A of Trevisan, Mamon & Khosroshahi 2017 (MNRAS 464, 4593)"""
    
    # work with h_new = 1 until end
    GNewton = 43.012 # in units where sizes are in kpc, masses in 10^11 M_sun,
                     # velocities in 100 km/s
    M = 10**lMYang
    lMvirtry = np.linspace(8,15.6,761)
    Mvirtry = 10**lMvirtry
    ctry = cofM(Mvirtry)
    Mtildec = Mass_tilde_NFW(ctry)
    E = Eofz(Omegam,1-Omegam,z)
    cst = (0.9*Omegam)**(-1/3)
    oldadiff = 1e6 * np.ones(len(lMYang))
    Mnew = np.zeros(len(lMYang)) + 1e7
    for i, Mtry in enumerate(Mvirtry):
        c = ctry[i]
        Mtilde = Mtildec[i]
        arg = cst * c * E**(2/3)/(1+z) * (M/Mtry)**(1/3) / Mtilde
        lhs = np.log10(Mass_tilde_NFW(arg) * Mtry)
        rhs = lMYang
        adiff = np.abs(lhs-rhs)
        Mnew = np.where(adiff < oldadiff, Mtry, Mnew)
        oldadiff = np.where(adiff < oldadiff, adiff, oldadiff)
    print("max(adiff)=",np.max(oldadiff))
    Mnew = Mnew/h
    rnew = (0.01*GNewton*(Mnew/1e11)/(1e-3*h)**2)**(1/3)
    return np.log10(Mnew),rnew


    
