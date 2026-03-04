import numpy as np
from scipy import integrate
from scipy import interpolate
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from . import PROFCL_mathutils as mu
from . import PROFCL_astroutils as au
from . import PROFCL_constants as cst
# import PROFCL_mathutils as mu
# import PROFCL_astroutils as au
# import PROFCL_constants as cst

def SurfaceDensity_tilde_NFW(X):
    """Dimensionless cluster surface density for an NFW profile. 
    arg: X = R/r_{-2} (positive float or array of positive floats), where r_{-2} is scale radius (slope -2)
    returns: Sigma(r_{-2} X) / [N(r_{-2})/pi r_{-2}^2] (float, or array of floats)"""

    # author: Gary Mamon

    #  t = type(X)
    # globals iPassSDtNFW
    
    # iPassSD

    # check that input is integer or float or numpy array
    # lu.CheckType(X,'SurfaceDensity_tilde_NFW','X')
    
    # stop with error message if input values are 0 (infinite surface density) or < 0 (unphysical)
    if np.any(X) <= 0.:
        raise ValueError('SurfaceDensity_tilde_NFW: min(X) = ' +
                         str(np.min(X)) + ' cannot be <= 0')

    # compute dimensionless surface density
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 1.2x10^-6)
    denom = cst.LN4MINUS1
    Xminus1 = X-1.
    Xsquaredminus1 = X*X - 1.
    # y =  np.where(abs(Xminus1) < 0.001, 
    #                     1./3. - 0.4*Xminus1, 
    #                     (1. 
    #                      - mu.ACO(1./X) / np.sqrt(abs(Xsquaredminus1))
    #                      ) 
    #                     / Xsquaredminus1 
    #                     ) / denom
    # print("num NaN = ", np.isnan(y).sum())
    return   np.where(abs(Xminus1) < 0.001, 
                        1./3. - 0.4*Xminus1, 
                        (1. 
                         - mu.ACO(1./X) / np.sqrt(abs(Xsquaredminus1))
                         ) 
                        / Xsquaredminus1 
                        ) / denom 

def SurfaceDensity_tilde_coredNFW(X):
    """Dimensionless cluster surface density for a cored-NFW profile: rho(x) ~ 1/(1+x)^3
    arg: X = R/r_{-2} (positive float or array of positive floats), where r_{-2} is radius of slope -2 
      (not the natural scale radius for which x=r/a!)
    returns: Sigma(r_{-2} X) / [N(r_{-2})/pi r_{-2}^2] (float, or array of floats)"""

    # author: Gary Mamon

    #  t = type(X)

    # check that input is integer or float or numpy array
    # lu.CheckType(X,"SurfaceDensity_tilde_coredNFW",'X')
    
    # stop with error message if input values are 0 (infinite surface density) or < 0 (unphysical)
    if np.any(X) <= 0.:
        raise ValueError("SurfaceDensity_tilde_coredNFW: min(X) = "
                         + str(np.min(X)) + " cannot be <= 0")

    # compute dimensionless surface density
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 2.1x10^-6)
    denom = cst.LN3MINUS8OVER9
    Xsquared = X*X
    Xminushalf = X-0.5
    Xsquaredtimes4minus1 = 4.*Xsquared-1.
    return np.where(abs(Xminushalf) < 0.001, 
                     0.4 - 24./35.*Xminushalf, 
                     (8.*Xsquared + 1. 
                      - 12.*Xsquared*mu.ACO(0.5/X) / np.sqrt(abs(Xsquaredtimes4minus1))
                      )
                     / Xsquaredtimes4minus1**2
                     ) / denom

def SurfaceDensity_tilde_tNFW(X,Xm):
    """Dimensionless cluster surface density for a truncated NFW profile. 
    args: X = R/r_{-2} (positive float or array of positive floats), where r_{-2} is scale radius (slope -2), Xm = R_truncation/R_{-2}
    returns: Sigma(r_{-2} X) / [N(r_{-2})/pi r_{-2}^2] (float, or array of floats)"""

    # author: Gary Mamon
    # source: Mamon, Biviano & Murante (2010), eq. (B.4)

    #  t = type(X)

    # check that input is integer or float or numpy array
    # lu.CheckType(X,'SurfaceDensity_tilde_tNFW','X')
    # lu.CheckType(Xm,'SurfaceDensity_tilde_tNFW','Xm')


    # stop with error message if input values are 0 (infinite surface density) or < 0 (unphysical)
    if np.any(X) <= 0.:
        raise ValueError('SurfaceDensity_tilde_tNFW: min(X) = '
                    + str(np.min(X)) + ' cannot be <= 0')

    # compute dimensionless surface density
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 1.2x10^-6)
    denom = cst.LN4MINUS1
    Xminus1 = X - 1
    argtmp = np.where(X>Xm,1,Xm*Xm-1)
    sqrtXmsquaredminus1 = np.where(X>Xm,1,np.sqrt(argtmp))
    argtmp2 = np.where(X>Xm,1,Xm*Xm-X*X)
    sqrtXm2minusX2 = np.where(X>Xm,1,np.sqrt(argtmp2))
    Xmplus1 = Xm + 1

    # return   np.select(
    #     [abs(Xminus1) < 0.001, X > 0 and X < Xm],
    #     [sqrtXmsquaredminus1*(Xm+2)/(3.*Xmplus1**2.) + Xminus1*(2-Xm-4*Xm*Xm-2*Xm**3)/(5*Xmplus1**2.*sqrtXmsquaredminus1),
    #      mu.ACO((X*X + Xm)/(X*(Xm+1.))) / ((1.-X*X) * np.sqrt(np.abs(X*X-1.))) - np.sqrt(Xm*Xm-X*X)/((1.-X*X)*(Xm+1))],
    #     default=0) / denom
    
    # fix by Yuzheng Kang
    return   np.select(
        [abs(Xminus1) < 0.001, np.logical_and(X > 0 , X < Xm)],
        [sqrtXmsquaredminus1*(Xm+2)/(3.*Xmplus1**2.) + Xminus1*(2-Xm-4*Xm*Xm-2*Xm**3)/(5*Xmplus1**2.*sqrtXmsquaredminus1),
         mu.ACO((X*X + Xm)/(X*(Xm+1.))) / ((1.-X*X) * np.sqrt(np.abs(X*X-1.))) - sqrtXm2minusX2/((1.-X*X)*(Xm+1))],
        default=0) / denom

def SurfaceDensity_tilde_Uniform(X):
    """Dimensionless cluster surface density for uniform model.
    arg: X = R/R_1 (positive float or array of positive floats), where R_1 is scale radius (radius where uniform model stops)
    returns: Sigma(R_1 X) (float, or array of floats)"""
    return np.where(X<1.,1.,0.)

def ProjectedNumber_tilde_NFW(X):
    """Dimensionless cluster projected number for an NFW profile. 
    arg: X = R/r_s (positive float or array of positive floats)
    returns: N_proj(X r_{-2}) / N_proj(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    # lu.CheckType(X,'ProjectedNumber_tilde_NFW','X')

    # stop with error message if input values are < 0 (unphysical)
    if np.any(X) < 0.:
        raise ValueError("ProjectedNumber_tilde_NFW: min(X) = "
                         + str(np.min(X)) + "cannot be <= 0")

    # compute dimensionless projected number
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 1.2x10^-7)
    denom = cst.LN2MINUSHALF
    Xtmp0 = np.where(X==0,1,X)
    Xtmp1 = np.where(X==1,0,X)
    # y =  np.where(X==0.,
    #                   0.,
    #                   np.where(abs(X-1.) < 0.001, 
    #                            1. - np.log(2.) + (X-1.)/3., 
    #                            mu.ACO(1./Xtmp0) / 
    #                            np.sqrt(abs(1.-Xtmp1*Xtmp1)) 
    #                            + np.log(0.5*Xtmp0)
    #                   ) / denom
    #             )
    # print("num NaN = ", np.isnan(y).sum())
    return np.where(X==0.,
                      0.,
                      np.where(abs(X-1.) < 0.001, 
                               1. - np.log(2.) + (X-1.)/3., 
                               mu.ACO(1./Xtmp0) / 
                               np.sqrt(abs(1.-Xtmp1*Xtmp1)) 
                               + np.log(0.5*Xtmp0)
                      ) / denom
                )

def ProjectedNumber_tilde_coredNFW(X):
    """Dimensionless cluster projected number for a cored NFW profile: rho(x) ~ 1/(1+x)^3
    arg: X = R/r_{-2} (positive float or array of positive floats), where r_{-2} is radius of slope -2 
      (not the natural scale radius for which x=r/a!)
    returns: N(X r_{-2}) / N(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    # CheckType(X,'ProjectedNumber_tilde_coredNFW','X')

    # stop with error message if input values are < 0 (unphysical)
    if np.any(X) < 0.:
        raise ValueError("ProjectedNumber_tilde_coredNFW: min(X) = "
                         + str(np.min(X)) + "cannot be < 0")

    # compute dimensionless projected number
    #   using series expansion for |X-1/2| < 0.001 (relative accuracy better than 4.11x10^-7)
    denom = cst.LN3MINUS8OVER9
    Xsquared = X*X
    Xminushalf = X-0.5
    Xsquaredtimes4minus1 = 4.*Xsquared-1.
    
    return np.where(X==0.,0.,
                     np.where(abs(Xminushalf) < 0.001, 
                     5./6. - np.log(2.) + 0.4*Xminushalf, 
                     (
                (6*Xsquared-1.)*mu.ACO(0.5/X) / np.sqrt(abs(Xsquaredtimes4minus1))
                + np.log(X)*Xsquaredtimes4minus1 - 2.*Xsquared
                )
                     /Xsquaredtimes4minus1
                     )
            ) / denom

def ProjectedNumber_tilde_tNFW(X,Xm):
    """Dimensionless cluster projected number for a truncated NFW profile. 
    args: X = R/r_{-2} (positive float or array of positive floats), where r_{-2} is scale radius (slope -2), Xm = R_truncation/R_{-2}
    returns: N(r_{-2} X) / N(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon
    # source: Mamon, Biviano & Murante (2010), eq. (B.4)

    # check that input is integer or float or numpy array
    # CheckType(X,'ProjectedNumber_tilde_tNFW','X')
    # CheckType(Xm,'ProjectedNumber_tilde_tNFW','Xm')

    # stop with error message if input values are 0 (infinite surface density) or < 0 (unphysical)
    if np.any(X) <= 0.:
        raise ValueError('ProjectedNumber_tilde_tNFW: min(X) = '
                  +  str(np.min(X)) + ' cannot be <= 0')

    # compute dimensionless surface density
    #   using series expansion for |X-1| < 0.001 (relative accuracy better than 1.2x10^-6)
    denom = cst.LN2MINUSHALF
    Xminus1 = X - 1.
    argtmp = np.where(X>Xm,1,Xm*Xm-X*X)
    sqrtXmsqminusXsq = np.where(X>Xm,1,np.sqrt(argtmp))
    Xmplus1 = Xm + 1.

    # return np.select(
    #             [abs(Xminus1) < 0.001, X > 0 and X < Xm],
    #             [np.log((Xm+1.)*(Xm-sqrtXmsqminusXsq)) - Xm/(Xm+1) + 2.*np.sqrt((Xm+1.)/(Xm-1.)),
    #              (sqrtXmsqminusXsq-Xm)/(Xm+1.) + np.log(Xmplus1*(Xm-sqrtXmsqminusXsq)/X) + mu.ACO((X*X+Xm)/(X*(Xm+1.)))
    #             ],
    #             default=np.log(Xm+1)-Xm/(Xm+1.)
    #        ) / denom

    # with Yuzheng Kang's idea of using np.logical_and
    return   np.select(
        [abs(Xminus1) < 0.001, np.logical_and(X > 0, X < Xm), X > Xm],
                [np.log((Xm+1.)*(Xm-sqrtXmsqminusXsq)) - Xm/(Xm+1) + 2.*np.sqrt((Xm-1.)/(Xmplus1)),
                 (sqrtXmsqminusXsq-Xm)/(Xm+1.) + np.log(Xmplus1*(Xm-sqrtXmsqminusXsq)/X) + mu.ACO((X*X+Xm)/(X*(Xm+1.)))/np.sqrt(np.abs(X*X-1)),
                 np.log(Xm+1)-Xm/(Xm+1)
                ],
                default=0
        ) / denom

def ProjectedNumber_tilde_Uniform(X):
    """Dimensionless cluster projected number for a uniform surface density profile
    arg: X = R/R_cut (positive float or array of positive floats), where R_cut is radius of slope -2 
      (not the natural scale radius for which x=r/a!)
    returns: N(X R_cut) / N(R_cut) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    # CheckType(X,"ProjectedNumber_tilde_Uniform","X")

    # stop with error message if input values are < 0 (unphysical)
    if np.any(X) < 0.:
        raise ValueError("ProjectedNumber_tilde_Uniform: X cannot be < 0")

    return np.where(X<1, X*X, 1.)

def Number_tilde_NFW(x):
    """Dimensionless cluster 3D number for an NFW profile. 
    arg: x = r/r_s (positive float or array of positive floats)
    returns: N_3D(x r_{-2}) / N_3D(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    # CheckType(x,"Number_tilde_NFW","x")

    # stop with error message if input values are < 0 (unphysical)
    if np.any(x) < 0.:
        raise ValueError("Number_tilde_NFW: min(x) = "
                         + str(np.min(x)) + " cannot be < 0")

    return ( (np.log(x+1)-x/(x+1)) / cst.LN2MINUSHALF)

def Number_tilde_coredNFW(x):
    """Dimensionless cluster 3D number for a cored NFW profile. 
    arg: x = r/r_s (positive float or array of positive floats)
    returns: N_3D(x r_{-2}) / N_3D(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # check that input is integer or float or numpy array
    # CheckType(x,"Number_tilde_coredNFW",'x')

    # stop with error message if input values are < 0 (unphysical)
    if np.any(x) < 0.:
        raise ValueError("Number_tilde_coredNFW: min(x) = "
                         + str(np.min(x)) + " cannot be < 0")

    return ( (np.log(2*x+1)-2*x*(3*x+1)/(2*x+1)**2) / cst.LN3MINUS8OVER9 )

def Number_tilde_tNFW(x,x_cut):
    """Dimensionless cluster 3D number for an NFW profile. 
    args: 
        x = r/r_s (positive float or array of positive floats)
        x_cut = r_cut/r_s
    returns: N_3D(x r_{-2}) / N_3D(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    # cutoff radius / scale radius
    # x_cut = 10.**(log_R_cut-log_scale_radius)
    
    # check that input is integer or float or numpy array
    # CheckType(x,"Number_tilde_tNFW",'x')

    # stop with error message if input values are < 0 (unphysical)
    if np.any(x) < 0.:
        raise ValueError("Number_tilde_tNFW: min(x) = "
                         + str(np.min(x)) + " cannot be < 0")

    xtmp = np.where(x < x_cut, x, x_cut)
    return Number_tilde_NFW(xtmp)

def Number_tilde_Uniform(x):
    """Dimensionless cluster 3D number for a uniform surface density profile. 
    arg: x = r/R_1 (cutoff radius)
    returns: N_3D(x R_1) / (Sigma/R_1) (float, or array of floats)"""

    # author: Gary Mamon

     # check that input is integer or float or numpy array
    # CheckType(x,"Number_tilde_Uniform",'x')

    # stop with error message if input values are < 0 (unphysical)
    if np.any(x) < 0.:
        raise ValueError("Number_tidle_Uniform: min(x) = "
                         + str(np.min(x)) + " cannot be < 0")

    return np.where(x >= 1, 0, 1 / (np.pi * np.sqrt(1-x*x)))
   
def Number_tilde_Kazantzidis(x):
    """Dimensionless cluster 3D number for Kazantzidis model.
    arg: x = r/a (cutoff radius)
    returns: N_3D(x a) / N_#D(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon

    return (1 - (1+x)*np.exp(-x)) * np.exp(1)/(np.exp(1)-2)

def Number_tilde(x,model,x_cut=1):
     """Dimensionless cluster 3D number"""
     
     # author : Gary Mamon
     
     if model == "NFW":
         return Number_tilde_NFW(x)
     elif model == "cNFW":
         return Number_tilde_coredNFW(x)
     elif model == "tNFW":
         return Number_tilde_tNFW(x,x_cut)
     elif model == "Kazantzidis":
         return Number_tilde_Kazantzidis(x)
     else:
         raise ValueError("Cannot recognize model=" + model)
        
def Random_radius(xmax,model,Npoints,Nknots=300,verbosity=0,t=None):
    """Random r/r_{-2} from Monte Carlo for circular NFW model
    args: r_max/r_{-2} model (NFW|coredNFW) number-of-random-points number-of-knots"""

    # author: Gary Mamon

    # CheckType(Xmax,"Random_radius","Xmax")
    # CheckTypeInt(Npoints,"Random_radius","Npoints")

    # random numbers
    q = np.random.random_sample(Npoints)

    # equal spaced knots in arcsinh x
    # print("asinhx...")
    asinhx0 = np.linspace(0., np.arcsinh(xmax), num=Nknots+1)
    
    x0 = np.sinh(asinhx0)
    ratio = Number_tilde(x0,model)/Number_tilde(xmax,model)
    # print("x0...")
    # if model == "NFW":
    #     N0 = Number_tilde_NFW(xmax)
    # elif model == "coredNFW":
    #     N0 = Number_tilde_coredNFW(xmax)
    # elif model == 'tNFW':
    #     N0 = Number_tilde_tNFW(xmax)
    # else:
    #     raise ValueError ("Random_radius: model = " + model + " not recognized")
    
    # ratio = Number_tilde_NFW(x0) / N0
    if verbosity >= 2:
        print("ratio = ",ratio)

    # spline on knots of asinh(equal spaced)
    asinhratio = np.arcsinh(ratio)
    t = time.process_time()
    spline = interpolate.splrep(asinhratio,asinhx0,s=0)
    if verbosity >= 2:
        print("t = ", t)
        print("compute spline: time = ", time.process_time()-t)
    t = time.process_time()        
    asinhq = np.arcsinh(q)
    if verbosity >= 2:
        print("asinh(q): time = ", time.process_time()-t)
    t = time.process_time()        
    asinhx_spline = interpolate.splev(asinhq, spline, der=0, ext=2)
    if verbosity >= 2:
        print("evaluate spline: time = ", time.process_time()-t)
    return np.sinh(asinhx_spline)
         
def Random_projected_radius(Xmax,model,Npoints,Nknots=300,verbosity=0,t=None):
    """Random R/r_{-2} from Monte Carlo for circular NFW model
    args: R_max/R_{-2} model (NFW|coredNFW) number-of-random-points number-of-knots"""

    # author: Gary Mamon

    # CheckType(Xmax,"Random_radius","Xmax")
    # CheckTypeInt(Npoints,"Random_radius","Npoints")

    # random numbers
    q = np.random.random_sample(Npoints)

    # equal spaced knots in arcsinh X
    asinhX0 = np.linspace(0., np.arcsinh(Xmax), num=Nknots+1)

    X0 = np.sinh(asinhX0)
    if model == "NFW":
        N0 = ProjectedNumber_tilde_NFW(Xmax)
    elif model == "coredNFW":
        N0 = ProjectedNumber_tilde_coredNFW(Xmax)
    elif model == 'tNFW':
        N0 = ProjectedNumber_tilde_tNFW(Xmax)
    else:
        raise ValueError ("Random_radius: model = " + model + " not recognized")
    
    ratio = ProjectedNumber_tilde_NFW(X0) / N0

    # spline on knots of asinh(equal spaced)
    asinhratio = np.arcsinh(ratio)
    # t = time.process_time()
    spline = interpolate.splrep(asinhratio,asinhX0,s=0)
    if verbosity >= 2:
        print("compute spline: time = ", time.process_time()-t)
    # t = time.process_time()        
    asinhq = np.arcsinh(q)
    if verbosity >= 2:
        print("asinh(q): time = ", time.process_time()-t)
    # t = time.process_time()        
    asinhX_spline = interpolate.splev(asinhq, spline, der=0, ext=2)
    if verbosity >= 2:
        print("evaluate spline: time = ", time.process_time()-t)
    return np.sinh(asinhX_spline)

def Random_xy(Rmax,model,Npoints,Nknots,scale_radius,ellipticity,PA,
              RA_cen,Dec_cen,RA_cen_init,Dec_cen_init,verbosity=0):
    """Random x & y (in deg) from Monte Carlo for circular model (NFW or coredNFW)
    args: R_max model (NFW|coredNFW) number-of-random-points number-of-knots ellipticity PA"""
    R_random = scale_radius * Random_projected_radius(Rmax/scale_radius,model,Npoints,Nknots)
    # PA_random = 2 * np.pi * np.random.random_sample(Npoints) # in rd
    theta_random = 2 * np.pi * np.random.random_sample(Npoints) # in rd
    u_random = R_random * np.cos(theta_random)
    v_random = R_random * np.sin(theta_random) * (1-ellipticity)
    x0_random = RA_cen_init + (RA_cen - RA_cen_init) * mu.cosd(Dec_cen_init) 
    y0_random = Dec_cen
    x_random = x0_random - u_random*mu.sind(PA)- v_random*mu.cosd(PA)
    y_random = y0_random + u_random*mu.cosd(PA) - v_random*mu.sind(PA)

    if verbosity >= 3:
        print("IS TRUE : ", x0_random == RA_cen_init, y0_random == Dec_cen_init)
    return x_random, y_random
    
def ProjectedNumber_tilde_ellip_NFW(X,ellipticity,PA,
                                    N_points,
                                    RA_cen,Dec_cen,RA_cen_init,Dec_cen_init,
                                    DeltaCenter,min_R_over_rminus2,
                                    DeltaCenter_over_a,
                                    TINY_SHIFT_POS,
                                    verbosity=0):
    """Dimensionless projected mass for non-circular NFW models
    args:
    X = R_sky/r_{-2}  (positive float or array of positive floats), where r_{-2} is radius of slope -2
    ellipticity = 1-b/a (0 for circular)
    returns: N(X r_{-2}) / N(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon
    # source: ~gam/EUCLID/CLUST/PROCL/denomElliptical.nb

    # N_points = int(1e5)
    # N_knots  = int(1e4)

    if verbosity >= 5:
        print("using 2D polynomial for ellipticity = ", ellipticity)
    e = ellipticity
    if N_points == 0 and DeltaCenter < TINY_SHIFT_POS and X < min_R_over_rminus2:
        # print("ProjectedNumber_tilde_ellip_NFW: series expansion")
        Nprojtilde = X*X/cst.LN16MINUS2 / (1.-e) * (-1. - 2.*np.log(0.25*X*(2.-e)/(1.-e)))
    elif N_points == 0 and DeltaCenter < TINY_SHIFT_POS:
        # analytical approximation, only for centered ellipses
        lX = np.log10(X)
        lNprojtilde = (
            0.2002159160168399 + 0.23482973073606245*e + 
            0.028443816507694702*e**2 + 3.0346488960850246*e**3 - 
            32.92395216847275*e**4 + 155.31214454203342*e**5 - 
            384.7956580823655*e**6 + 524.185430033757*e**7 - 
            372.1186576278279*e**8 + 107.73575331855518*e**9 + 
            1.084957208707404*lX - 0.15550331288872482*e*lX + 
            0.19686182416407058*e**2*lX - 4.369613060146462*e**3*lX + 
            25.786051119038408*e**4*lX - 76.01463442935163*e**5*lX + 
            118.12160576401868*e**6*lX - 93.01512548879035*e**7*lX + 
            29.20821583872627*e**8*lX - 0.350953887814871*lX**2 - 
            0.024303352180603605*e*lX**2 - 0.1287797997529538*e**2*lX**2 + 
            2.173015888479342*e**3*lX**2 - 8.937397688350035*e**4*lX**2 + 
            17.468998705433673*e**5*lX**2 - 16.251979189333717*e**6*lX**2 + 
            5.885069528670919*e**7*lX**2 - 0.012941807861877342*lX**3 + 
            0.03797412877170318*e*lX**3 + 0.10378160335237464*e**2*lX**3 - 
            0.5184579855114362*e**3*lX**3 + 1.2834168703734363*e**4*lX**3 - 
            1.4165466726424798*e**5*lX**3 + 0.5653714436995129*e**6*lX**3 + 
            0.04317093844555722*lX**4 + 0.013619786789711666*e*lX**4 - 
            0.07157446386996426*e**2*lX**4 + 0.12635271935992576*e**3*lX**4 - 
            0.1623323869598711*e**4*lX**4 + 0.06594832410639553*e**5*lX**4 + 
            0.0005189446937787153*lX**5 - 0.012170985301529685*e*lX**5 + 
            0.0078104820069108665*e**2*lX**5 - 
            0.012168623850966566*e**3*lX**5 + 0.01120734375450095*e**4*lX**5 - 
            0.0063164825849164104*lX**6 - 0.0003229562648197668*e*lX**6 + 
            0.004797249087277705*e**2*lX**6 - 
            0.0006839516501486773*e**3*lX**6 + 0.0005190658690337241*lX**7 + 
            0.001323550523203948*e*lX**7 - 0.0009722709478153854*e**2*lX**7 + 
            0.0004615622537881461*lX**8 - 0.0002037464879060379*e*lX**8 - 
            0.00008236148148039739*lX**9
        )
        Nprojtilde = 10. ** lNprojtilde
    elif N_points < 0 and DeltaCenter < TINY_SHIFT_POS:
        # evaluate double integral
        print("ProjectedNumber_tilde_ellip_NFW: quadrature")
        f = lambda V, U: SurfaceDensity_tilde_NFW(np.sqrt(U*U+V*V/(1-ellipticity)**2))
        Nprojtilde = integrate.dblquad(f,0,X,lambda U: 0, lambda U: np.sqrt(X*X-U*U), epsabs=0., epsrel=0.001)
        Nprojtilde = 4/(np.pi*(1.-ellipticity)) * Nprojtilde[0]
    else:
        # Monte Carlo integration
        print("ProjectedNumber_tilde_ellip_NFW: Monte Carlo")
        if DeltaCenter_over_a < 1.e-6:
            X_ellip = Random_radius(X/(1.-ellipticity), "NFW", N_points, 100)
        else:
            X_ellip = Random_radius((X+DeltaCenter_over_a)/(1.-ellipticity), "NFW", N_points, 100)
        phi = 2. * np.pi * np.random.random_sample(N_points)
        U = X_ellip * np.cos(phi)
        V = (1.-ellipticity) * X_ellip * np.sin(phi)
        if DeltaCenter_over_a < 1.e-6:
            X_sky_MC = np.sqrt(U*U + V*V)
        else:
            dX = - U*mu.sind(PA) - V*mu.cosd(PA)
            dY =   U*mu.cosd(PA) - V*mu.coss(PA)
            X_MC = -(RA_cen - RA_cen_init)/mu.cosd(Dec_cen_init) + dX
            Y_MC = Dec_cen - Dec_cen_init + dY
            X_sky_MC = np.sqrt(X_MC*X_MC + Y_MC*Y_MC)

        X_in_circle = X_sky_MC[X_sky_MC < X]
        frac = len(X_in_circle) / N_points
        Nprojtilde = frac * ProjectedNumber_tilde_NFW(X/(1.-ellipticity))

    # print ("ProjectedNumber_tilde_ellip_NFW: Xmax e Nprojtilde = ", X, ellipticity, Nprojtilde)
    return Nprojtilde

def ProjectedNumber_tilde_ellip_coredNFW(X,ellipticity,PA,
                                    N_points,
                                    RA_cen,Dec_cen,RA_cen_init,Dec_cen_init,
                                    DeltaCenter,min_R_over_rminus2,
                                    DeltaCenter_over_a,
                                    TINY_SHIFT_POS,
                                    TINY=1.e-38,
                                    verbosity=0):
    """Dimensionless projected mass for non-circular cored NFW models
    args:
    X = R/r_{-2}  (positive float or array of positive floats), where r_{-2} is radius of slope -2
    ellipticity = 1-b/a (0 for circular)
    returns: N(X r_{-2}) / N(r_{-2}) (float, or array of floats)"""

    # author: Gary Mamon
    # source: ~gam/EUCLID/CLUST/PROCL/denomElliptical.nb

    if np.abs(ellipticity) < TINY:
        return (ProjectedNumber_tilde_coredNFW(X))
    
    if verbosity >= 3:
        print("using 2D polynomial")
    lX = np.log10(X)
    e = ellipticity
    if N_points == 0:
        lNprojtilde = (
            0.21076779174081403 - 0.1673534076933856*e - 
            0.9471677808222536*e**2 + 11.648473045614114*e**3 - 
            91.92475409478227*e**4 + 422.8544124895236*e**5 - 
            1206.605470683992*e**6 + 2152.6556515394586*e**7 - 
            2336.252720403306*e**8 + 1409.651246367505*e**9 - 
            362.82577003936643*e**10 + 1.1400775160218775*lX - 
            0.24603956803791907*e*lX + 0.4746353855804624*e**2*lX - 
            5.213784368168905*e**3*lX + 28.349333190289443*e**4*lX - 
            95.44143806235569*e**5*lX + 196.77037041806182*e**6*lX - 
            242.5768688683326*e**7*lX + 164.00212699954048*e**8*lX - 
            46.77921433973666*e**9*lX - 0.47280190984201714*lX**2 + 
            0.030724988640708772*e*lX**2 + 0.14209201391142387*e**2*lX**2 - 
            0.755436616271162*e**3*lX**2 + 3.306367265271173*e**4*lX**2 - 
            7.25557673533242*e**5*lX**2 + 9.429315278575027*e**6*lX**2 - 
            6.660238987320651*e**7*lX**2 + 2.04545992649397*e**8*lX**2 + 
            0.03394971337975079*lX**3 + 0.09887824821508472*e*lX**3 - 
            0.18041596878156793*e**2*lX**3 + 0.6289610806099004*e**3*lX**3 - 
            1.4556318193802276*e**4*lX**3 + 2.1239832585391083*e**5*lX**3 - 
            1.8325143147948293*e**6*lX**3 + 0.6369289158521704*e**7*lX**3 + 
            0.07315774564006589*lX**4 - 0.037041022300377306*e*lX**4 + 
            0.0029908382801743685*e**2*lX**4 - 0.03572991462536126*e**3*lX**4 - 
            0.05039173454869054*e**4*lX**4 + 0.06826024306255776*e**5*lX**4 - 
            0.028441143677024536*e**6*lX**4 - 0.019219238751868855*lX**5 - 
            0.02361318179363677*e*lX**5 + 0.0405966969727285*e**2*lX**5 - 
            0.052053157027219105*e**3*lX**5 + 0.05969376194544227*e**4*lX**5 - 
            0.01240643979930337*e**5*lX**5 - 0.01026942895674158*lX**6 + 
            0.01301415707276946*e*lX**6 - 0.007109228236235994*e**2*lX**6 + 
            0.014751475808259498*e**3*lX**6 - 0.008400229615749667*e**4*lX**6 + 
            0.004545329673990146*lX**7 + 0.0011480281966753895*e*lX**7 - 
            0.002874103492006819*e**2*lX**7 - 0.0009871609971554144*e**3*lX**7 + 
            0.0003921813852493623*lX**8 - 0.0014751021188585689*e*lX**8 + 
            0.0006830554871586946*e**2*lX**8 - 0.0004114331203583239*lX**9 + 
            0.00020132121960998451*e*lX**9 + 0.00005094309326516718*lX**10
        )
        Nprojtilde = 10. ** lNprojtilde
        
    return Nprojtilde

def ProjectedNumber_tilde_Uniform(X):
    """Dimensionless cluster projected number for a uniform model.
    arg: X = R/R_1 (positive float or array of positive floats), where R_1 is scale radius (radius where uniform model stops)
    returns: N(R_1 X) / N(R_1) (float, or array of floats)"""

    # author: Gary Mamon

    return np.where(X<1, X*X, 1.)

def ProjectedNumber_tilde_ellip(R_over_a, model, e, sinPA, cosPA, background, 
                                DeltaRA, DeltaDec,
                                scale_radius,DeltaCenter,
                                RA_cen_init,Dec_cen_init,
                                N_points,
                                Tiny_Shift_Pos,TINY=1.e-38,
                                ):
    """Dimensionless cluster projected number of elliptical and/or shifted models relative to circular region
    arguments:
        R_over_a = R/r_{-2} (dimensionless radius) [positive float or array of positive floats]
        model: 'NFW' or 'coredNFW'
        e: ellipticity (default=0.) [dimensionless]
        DeltaRA, DeltaDec: shift of position of model relative to circular region [deg]
    returns: N_proj(R) / N(r_{-2}) (float or array of floats)"""

    # author: Gary Mamon

    # short-named variables for clarity
    a = scale_radius
    Z = R_over_a
    
    if np.abs(e) < TINY and np.abs(DeltaCenter) < Tiny_Shift_Pos:
        # centered circular
        # print("ProjectedNumber_tilde_ellip: circular ...")
        return (ProjectedNumber_tilde(R_over_a,model))
    
    elif np.abs(e) < TINY and model == 'uniform':
        # shifted Uniform
        # print("ProjectedNumber_tilde_ellip: uniform shifted ...")
        shift_x, shift_y = au.dxdy_from_RADec(RA_cen_init+DeltaRA,Dec_cen_init+DeltaDec,
                                              RA_cen_init,Dec_cen_init)
        d = np.sqrt(shift_x**shift_x + shift_y*shift_y)
        # area is intersection of circles
        # from Wolfram MathWorld http://mathworld.wolfram.com/Circle-CircleIntersection.html
        Rtmp = a * R_over_a
        area =   Rtmp*Rtmp * np.arccos((d*d + Rtmp*Rtmp - a*a) / (2*d*Rtmp)) \
               + a*a * np.arccos((d*d + a*a - Rtmp*Rtmp) / (2*d*a)) \
               - 0.5*np.sqrt((-d+a+Rtmp) * (d+a-Rtmp) * (d-a+Rtmp) * (d+a+Rtmp))
        # FOLLOWING IS PROBABLY INCORRECT!!!
        # needs to be in units of N(a)?
        return background * area

    elif DeltaCenter < Tiny_Shift_Pos and N_points == 0 and model in ('NFW','coredNFW'):
        # centered elliptical with polynomial approximation
        # print("ProjectedNumber_tilde_ellip: polynomial ...")
        if model == 'NFW':
            return (ProjectedNumber_tilde_ellip_NFW(R_over_a,e))
        elif model == 'coredNFW':
            return (ProjectedNumber_tilde_ellip_coredNFW(R_over_a,e))

    elif N_points < 0:
        # double integral by quadrature
        # print("ProjectedNumber_tilde_ellip: quadrature ...")
        tol_quadrature = 10.**N_points
        if DeltaCenter < Tiny_Shift_Pos:
            f = lambda V, U: SurfaceDensity_tilde(np.sqrt(U*U+V*V/(1-e)**2),model)
            Nprojtilde = integrate.dblquad(f,0,Z,lambda U: 0, lambda U: np.sqrt(Z*Z-U*U), epsabs=0., epsrel=tol_quadrature)     
            Nprojtilde = 4./(np.pi*(1.-e)) * Nprojtilde[0]
        else:
            # CHECK FOLLOWING LINE!
            f = lambda Y, X: SurfaceDensity_tilde(au.R_ellip_from_xy(a*X,a*Y)/a,model)
            Nprojtilde = integrate.dblquad(f,-Z,Z,lambda X: -np.sqrt(Z*Z-X*X), 
                                           lambda X: np.sqrt(Z*Z-X*X), epsabs=0., epsrel=tol_quadrature)
            Nprojtilde = 1./(np.pi*(1.-e)) * Nprojtilde[0]

    elif N_points >= 1000:
        # Monte Carlo
        # print("ProjectedNumber_tilde_ellip: Monte Carlo ... N_points = ", N_points)
        N_knots = 100
        Z_ellip_MC = Random_radius(R_over_a/(1.-e), model, N_points, N_knots)
        phi = 2. * np.pi * np.random.random_sample(N_points)
        U = Z_ellip_MC * np.cos(phi)
        V = (1.-e) * Z_ellip_MC * np.sin(phi)
        if np.abs(DeltaCenter) < Tiny_Shift_Pos:
            Z_MC = np.sqrt(U*U + V*V)
        else:
            # add shift of center
            dX,dY = au.dxdy_from_uv(U,V,sinPA,cosPA)
            X = -1. * DeltaRA/mu.cosd(Dec_cen_init) + dX
            Y =       DeltaDec                              + dY
            Z_MC = np.sqrt(X*X + Y*Y)
        Z_in_circle = Z_MC[Z_MC < R_over_a]
        frac = len(Z_in_circle) / N_points
        # circular N_proj_tilde times fraction of points inside oversized circle
        Nprojtilde = frac * ProjectedNumber_tilde(R_over_a/(1.-e),model)
    else:
        raise ValueError("ProjectedNumber_tilde_ellip: N_points = " 
                         + str(N_points) + "DeltaCenter = " + str(DeltaCenter))
    return Nprojtilde

def SurfaceDensity_tilde(X,model,Xcut=0):
    """Dimensionless cluster surface density
    arguments:
        X = R/r_{-2} (dimensionless radius) [positive float or array of positive floats]
        model: 'NFW' or 'coredNFW'
        Xcut = Rmax-allowed/r_{-2} (dimensionless cutoff radius)
    returns: Sigma(r_{-2} X) / [N(r_{-2})/pi r_{-2}^2]  (float or array of floats)"""

    # author: Gary Mamon

    if model == "NFW":
        return SurfaceDensity_tilde_NFW(X)
    elif model == "coredNFW":
        return SurfaceDensity_tilde_coredNFW(X)
    elif model == 'tNFW':
        return SurfaceDensity_tilde_tNFW(X,Xcut)
    elif model == "uniform":
        return SurfaceDensity_tilde_Uniform(X,Xcut)
    else:
        raise ValueError("SurfaceDensity_tilde: model = " + model + " is not recognized")

def ProjectedNumber_tilde(X,e,Xcut,model,DeltaRA,DeltaDec,DeltaCenter,
                          min_R_over_rminus2=0,Tiny_Shift_Pos=0,
                          TINY=1.e-10,verbosity=0):
    """Dimensionless cluster projected number
    arguments:
        X = R/r_{-2} (dimensionless radius) [positive float or array of positive floats]
        model: 'NFW' or 'coredNFW'
        e: ellipticity (default=0.) [dimensionless]
        DeltaRA, DeltaDec: shift of position of model relative to circular region [deg]
        min_R_over_rminus2: min allowed R/r_minus2 [default: 0]
        Tiny_Shift_Pos: smallest shift in position for full fit [deg, default: 0]
        TINY: maximum ellipticity for circular fit [default: 0]
        verbosity: verbosity [default: 0]
    returns: N_proj(R) / N(r_{-2}) (float or array of floats)"""

    # author: Gary Mamon

    if verbosity >= 4:
        print("ProjectedNumber_tilde: ellipticity=",e)
    if X < -TINY:
        raise ValueError("ProjectedNumber_tilde: X = " + str(X) + " cannot be negative")
    elif X < TINY:
        return 0
    # elif X < min_R_over_rminus2-TINY:
    #     raise ValueError("ProjectedNumber_tilde: X = " + str(X) + 
    #                      " <= critical value = " + str(min_R_over_rminus2))

    if np.abs(e) < TINY and DeltaCenter < Tiny_Shift_Pos:
        if model == "NFW":
            return ProjectedNumber_tilde_NFW(X)
        elif model == "coredNFW":
            return ProjectedNumber_tilde_coredNFW(X)
        elif model == 'tNFW':
            return ProjectedNumber_tilde_tNFW(X,Xcut)
        elif model == "uniform":
            return ProjectedNumber_tilde_Uniform(X)
        else:
            raise ValueError("ProjectedNumber_tilde: cannot recognize model " + model)
    else:
        return ProjectedNumber_tilde_ellip(X,model,e,DeltaRA,DeltaDec)

def Density_tilde_NFW(x):
    """Dimensionless NFW 3D density (for testing)
    arguments:
        r/r_{-2}
    output:
        rho(r) / { N(r_{-2}) / [4 pi r_{-2}^3)] }
    """

    # author: Gary Mamon

    return 1/(x*(1+x)**2) / cst.LN2MINUSHALF

def Density_tilde_coredNFW(x):
    """Dimensionless coredNFW 3D density (for testing)
    arguments:
        r/r_{-2}
    output:
        rho(r) / { N(r_{-2}) / [4 pi r_{-2}^3)] }
    """

    # author: Gary Mamon

    return 8 / (2*x+1)**3 / cst.LN3MINUS8OVE9R

def Density_tilde_tNFW(x,xcut):
    """Dimensionless truncated NFW 3D density (for testing)
    arguments:
        r/r_{-2} r_cut/r_{-2}
    output:
        rho(r) / { N(r_{-2}) / [4 pi r_{-2}^3)] }
    """

    # author: Gary Mamon
    
    return np.where(x<xcut,Density_tilde_NFW(x),0)

def Density_tilde_Kazantzidis(x):
    """Dimensionless Kazantzidis 3D density (for testing)
    arguments:
        r/r_{-2}
    output:
        rho(r) / { N(r_{-2}) / [4 pi r_{-2}^3)] }
    """

    # author: Gary Mamon
    
    return np.exp(-x) / x / (1-2/np.exp(1))

   
def Density_tilde(x,model,xcut):
    """Dimensionless cluster 3D number density (for testing)
    arguments:
        x = r/r_{-2} (dimensionless 3D radius) [positive float or array of positive floats]
        model: 'NFW' or 'tNFW'
        x_cut = r_cut/r_{-2}
         where r_{-2} is radius of slope -2."""
    
    # author: Gary Mamon

    if model == "NFW":
        return Density_tilde_NFW(x)
    elif model == "tNFW":
        return Density_tilde_tNFW(x,xcut)
    elif model == "cNFW":
        return Density_tilde_coredNFW(x)
    elif model == "Kazantzidis":
        return Density_tilde_Kazantzidis(x)
    else:
        raise ValueError("Number_tilde: model = " + model + " is not recognized")

def PlotDensity(x,xmin,xmax,Nbins,dim=2,xlims=None,ylims=None,
               xlab=None,ylab=None,RadiusUnit=None,DensityUnit=None,
               normData=None,normModel=None,normModelAlt=None,
               rminus2=None,rminus2Alt=None,
               fieldDensity=0,
               fieldDensityAlt=None,
               model=None,device=None):
    """Plot surface density profile in bins of log projected radius
    overplotting best-fit profile
    optionally overplot alternative profile (e.g. true profile for mock)
    """
    
    # author: Gary Mamon
        
    rc('text', usetex=True)
    lx = np.log10(x)
    hist, bin_edges = np.histogram(lx,bins=Nbins,range=(np.log10(xmin),np.log10(xmax)))
    xminbins = 10**bin_edges[:-1]
    xmaxbins = 10**bin_edges[1:]
    area = np.pi * (xmaxbins**2-xminbins**2)
    volume = 4*np.pi/3 * (xmaxbins**3-xminbins**3)
    if normData is None:
        norm = 1
    else:
        norm = normData
    if dim == 3:
        dens = norm*hist/volume
        edens = norm*np.sqrt(hist)/volume
        xlab = 'radius'
        ylab = 'density'
    elif dim == 2:
        dens = norm*hist/area
        edens = norm*np.sqrt(hist)/area
        xlab = 'projected radius'
        ylab = 'surface density'
    else:
        raise ValueError("cannot understand dim = " + str(dim)) 
    # nonzerodens = dens[dens>0]
    # print("bin_edges=",bin_edges)
 
    fig = plt.figure(figsize=(6,6))
    ax = fig.gca()

    bins = (bin_edges[1:]+bin_edges[:-1])/2
    # dexbins = 10**bins
    plt.errorbar(10**bins,dens,yerr=edens,marker='o',c='k',mfc='r',mec='k',ls='None')
    plt.xscale('log')
    plt.yscale('log')
    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
    if (RadiusUnit is None):
        plt.xlabel(xlab)
    else:
        plt.xlabel(xlab + ' (' + RadiusUnit + ')')
    if (DensityUnit is None):
        # plt.ylabel("$" + ylab + "$")
        plt.ylabel(ylab)
    else:
        # plt.ylabel("$" + ylab + "$" + ' (' + "$" + DensityUnit + "$" + ')')
        plt.ylabel(ylab + ' (' + DensityUnit + ')')
    if xlims is None:
        xlims = ax.get_xlim()
    plt.grid()
    xx = 10**np.linspace(np.log10(xlims[0]),np.log10(xlims[-1]), 201)
    if model == 'NFW' and rminus2 is not None:
        if normModel is None:
            norm = 1
        else:
            norm = normModel
        if dim == 2:
            plt.plot(xx,fieldDensity + norm*SurfaceDensity_tilde(xx/rminus2,model),'b')
        else:
            plt.plot(xx,fieldDensity + norm*Density_tilde_NFW(xx/rminus2),'b')
        if rminus2Alt is not None:
            if dim == 2:
                plt.plot(xx,fieldDensityAlt + norm*SurfaceDensity_tilde(xx/rminus2Alt,model),'g')
            else:
                plt.plot(xx,fieldDensityAlt + norm*Density_tilde_NFW(xx/rminus2Alt),'g')
    elif model == 'NFW':
        raise ValueError("PlotDensity requires scale radius")
    else:
        raise ValueError("PlotDensity cannot recognize model " + model)
    # plt.title('all galaxies (from ' + str(numhalos) + ' halos)')
    plt.grid(True)
    if device is None:
        plt.show
    else:
        plt.savefig(device)
