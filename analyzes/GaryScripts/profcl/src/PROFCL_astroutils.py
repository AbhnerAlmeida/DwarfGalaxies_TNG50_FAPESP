import numpy as np
from astropy.coordinates import SkyCoord
from . import PROFCL_mathutils as mu
from . import PROFCL_constants as cst

def AngularSeparation(RA1,Dec1,RA2,Dec2,choice='cart'):
    """angular separation given (RA,Dec) pairs [input and output in degrees]
    args: RA_1, Dec_1, RA_2, Dec_2, choice
        choice can be 'astropy', 'trig', or 'cart'
            'trig' is like 'astropy', but 50x faster!
            default: 'cart'"""

    # author: Gary Mamon

    # tests indicate that:
    # 1) trig is identical to astropy, but 50x faster!
    # 2) cart is within 10^-4 for separations of 6 arcmin, but 2x faster than trig
    
    if choice == 'cart':
        DeltaRA = (RA2-RA1) * mu.cosd(0.5*(Dec1+Dec2))
        DeltaDec = Dec2-Dec1
        separation = np.sqrt(DeltaRA*DeltaRA + DeltaDec*DeltaDec)
    elif choice == 'trig':
        cosSeparation = mu.sind(Dec1)*mu.sind(Dec2) \
                        + mu.cosd(Dec1)*mu.cosd(Dec2)*mu.cosd((RA2-RA1))
        separation = mu.ACO(cosSeparation) / cst.DEGREE
    elif choice == 'astropy':
        c1 = SkyCoord(RA1,Dec1,unit='deg')
        c2 = SkyCoord(RA2,Dec2,unit='deg')
        separation = c1.separation(c2).deg
    else:
        raise ValueError("AngularSeparation: cannot recognize choice = " + choice)

    return separation

def RADec_from_dxdy(dx,dy,RA_cen,Dec_cen,cosDec_cen_init):
    """celestial coordinates given cartesian coordinates relative to new center (all in deg)"""

    # author: Gary Mamon
    # cosDec_cen_init, RA_cen, Dec_cen arrive as global variables
    
    RA  = cosDec_cen_init * RA_cen - dx
    Dec = Dec_cen                  + dy

    # handle crossing of RA=0
    RA = np.where(RA >= 360.,  RA - 360., RA)

    return RA,Dec

def RADec_from_uv(u,v,RA_cen,Dec_cen,cosDec_cen_init,sinPA,cosPA):
    """celestial coordinates given elliptical coordinates u (along major axis) and v (along minor axis)"""
    # author: Gary Mamon

    # cartesian coordinates around new center    
    dx,dy = dxdy_from_uv(u,v,sinPA,cosPA)
    RA,Dec = RADec_from_dxdy(dx,dy,RA_cen,Dec_cen,cosDec_cen_init)
    
    return RA,Dec

def xy_from_dxdy(dx,dy,Delta_x_cen,Delta_y_cen):
    """cartesian coordinates relative to center of circular region (x points to -RA, y to +Dec) 
    given cartesian coordinates relative to new center
    output in degrees"""

    # author: Gary Mamon
    # Delta_x_cen and Delta_y_cen arrive as global variables
    
    x = dx + Delta_x_cen
    y = dy + Delta_y_cen
    return x,y

def dxdy_from_xy(x,y,Delta_x_cen,Delta_y_cen):
    """cartesian coordinates relative to new center (x points to -RA, y to +Dec) 
    given cartesian coordinates (in deg) relative to center fo circular region"""
    # author: Gary Mamon
    # Delta_x_cen and Delta_y_cen arrive as global variables
    
    dx = x + Delta_x_cen
    dy = y - Delta_y_cen
    return dx,dy

def dxdy_from_uv(u,v,sinPA,cosPA):
    """cartesian coordinates relative to center (x points to -RA, y to +Dec) given elliptical coordinates"""
    # author: Gary Mamon
     
    dx = -u * sinPA - v * cosPA
    dy =  u * cosPA - v * sinPA
    return dx,dy

def dxdy_from_RADec(RA,Dec,RA_cen,Dec_cen,cosDec_cen=None):
    """cartesian coordinates (in deg) relative to center (x points to -RA, y to +Dec) 
    given celestial coordinates"""
    # author: Gary Mamon

    if cosDec_cen is None:
        cosDec_cen = mu.cosd(Dec_cen)
    dx =  -1.* (RA  - RA_cen) * cosDec_cen
    dy =        Dec - Dec_cen   
    
    # handle crossing of RA=0
    dx = np.where(dx >= 180.,  dx - 360., dx)
    dx = np.where(dx <= -180., dx + 360., dx)

    return dx,dy

def uv_from_dxdy(dx,dy,sinPA,cosPA):
    """elliptical coordinates u (along major axis) and v (along minor axis), given cartesian coordinates relative to ellipse (all in deg)"""
    # author: Gary Mamon
    # sinPA and cosPA arrives as a global variables

    # rotate to axes of cluster (careful: PA measures angle East from North)
    u = - dx * sinPA + dy * cosPA
    v = - dx * cosPA - dy * sinPA
    return u,v

def uv_from_xy(x,y,Delta_x_cen,Delta_y_cen,sinPA,cosPA):
    """elliptical coordinates u (along major axis) and v (along minor axis), given cartesian coordinates relative to circle (all in deg)"""
    # author: Gary Mamon

    # cartesian coordinates around new center
    dx,dy = dxdy_from_xy(x,y,Delta_x_cen,Delta_y_cen)

    # rotate to axes of cluster
    u,v = uv_from_dxdy(dx,dy,sinPA,cosPA)

    return u,v
    
def uv_from_RADec(RA,Dec,RA_cen,Dec_cen,sinPA,cosPA):
    """elliptical coordinates u (along major axis) and v (along minor axis), given celestial coordinates and PA (all in deg)"""
    # author: Gary Mamon

    # cartesian coordinates around new center
    dx,dy = dxdy_from_RADec(RA,Dec,RA_cen,Dec_cen)
    
    # rotate to axes of cluster
    u,v = uv_from_dxdy(dx,dy,sinPA,cosPA)

    return u,v


def R_ellip_from_dxdy(dx,dy,ellipticity,sinPA,cosPA):
    """elliptical equivalent radius, given celestial coordinates and PA (all in deg) and ellipticity"""
    # author: Gary Mamon
    # ellipticity arrives as a global variable

    # rotate to axes of cluster
    u,v = uv_from_dxdy(dx,dy,sinPA,cosPA)
    v_decompressed = v/(1.-ellipticity)
    return np.sqrt(u*u + v_decompressed*v_decompressed)

def R_ellip_from_RADec(RA,Dec,ellipticity,RA_cen,Dec_cen,sinPA,cosPA):
    """elliptical equivalent radius, given celestial coordinates and PA (all in deg) and ellipticity"""
    # author: Gary Mamon
    # ellipticity arrives as a global variable
    
    # rotate to axes of cluster
    u,v = uv_from_RADec(RA,Dec,RA_cen,Dec_cen,sinPA,cosPA)
    v_decompressed = v/(1.-ellipticity)
    return np.sqrt(u*u + v_decompressed*v_decompressed)

def R_ellip_from_xy(x,y,ellipticity,Delta_x_cen,Delta_y_cen,sinPA,cosPA):
    """elliptical equivalent radius, given cartesian coordinates around circular region (all in deg)"""
    # author: Gary Mamon

    # rotate to axes of cluster
    u,v = uv_from_xy(x,y,Delta_x_cen,Delta_y_cen,sinPA,cosPA)
    v_decompressed = v/(1.-ellipticity)
    return np.sqrt(u*u + v_decompressed*v_decompressed)

def NewCoords(RA_old,Dec_old):
    """coordinates in new frame where pole is at 0,90"""

    # author: Gary Mamon

    # convert from degrees to radians
    
    # RA_old           = RA_old * degree
    # Dec_old          = Dec_old * degree

    Dec_new_rd         = np.arcsin(mu.cosd(Dec_old) * mu.cosd(RA_old)) 
    # sinRA_new        = mu.sind(RA_old) * mu.cosd(Dec_old) / mu.cosd(Dec_new)
    
    tanRA_new2_num   = mu.cosd(Dec_old)*mu.sind(RA_old)
    tanRA_new2_denom = mu.sind(Dec_old)
    RA_new_rd           = np.arctan2(tanRA_new2_num,tanRA_new2_denom)
    return (RA_new_rd/cst.DEGREE,Dec_new_rd/cst.DEGREE)

def GuessCenter(RA,Dec,prob_membership,guess_center_flag,
                RA_cen=None,Dec_cen=None):
    """Guess center of cluster using (posisbly weighted) median
    arguments: 
        RA, Dec, 
        prob_membership
        guess_ center_flag (d for data-given, m for median, b for brightest)"""

    # author: Gary Mamon
    
    # return center if provided
    
    if RA_cen is not None and Dec_cen is not None and guess_center_flag == 'd':
        return RA_cen,Dec_cen
    
    
    if guess_center_flag == 'b':
        raise ValueError("GuessCenter flag b for brightest galaxy not yet implemented")

    # return median (weighted by probability if p_mem are not all unity)

    if np.any(prob_membership < 1.):
        return mu.WeightedMedian_Ash(RA,prob_membership), mu.WeightedMedian_Ash(Dec,prob_membership)
    else:
        return np.median(RA), np.median(Dec)

def GuessEllipPA(RA,Dec,RA_cen_init,Dec_cen_init):
    """ guess ellipticity and PA of cluster according to its 2nd moments
        RA_cen_init, Dec_cen_init passed as globals
    """

    # author: Gary Mamon
    
    dx,dy = dxdy_from_RADec(RA,Dec,RA_cen_init,Dec_cen_init)

    # guess ellipticity and PA from 2nd moments

    xymean = np.mean(dx*dy)
    x2mean = np.mean(dx*dx)
    y2mean = np.mean(dy*dy)
    tan2theta = 2*xymean/(x2mean-y2mean)
    theta = 0.5*np.arctan(tan2theta)
    PA_pred = theta / cst.DEGREE
    if PA_pred < 0.:
        PA_pred = PA_pred + 180.
    A2 = (x2mean+y2mean)/2. + np.sqrt((x2mean-y2mean)**2/4+xymean*xymean)
    B2 = (x2mean+y2mean)/2. - np.sqrt((x2mean-y2mean)**2/4+xymean*xymean)
    ellipticity_pred = 1 - np.sqrt(B2/A2)
    return ellipticity_pred, PA_pred
