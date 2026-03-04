import numpy as np
import os
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.stats import poisson
import scipy.integrate as integrate
# from scipy.optimize import minimize_scalar
# from scipy.optimize import fmin_tnc
# from scipy import interpolate
# from scipy import integrate
import matplotlib as mpl
from matplotlib import rcParams
import profcl.PROFCL_astroutils as au
import profcl.PROFCL_langutils as lu
import profcl.PROFCL_mathutils as mu
import profcl.PROFCL_densutils as du
import profcl.PROFCL_constants as cst
import sys
import os
home_dir = os.getenv("HOME") + '/'
import emcee # MCMC
import time # for tracking MCMC
from multiprocessing import Pool # emcee parallelization
import matplotlib.pyplot as plt
import corner_ev

cst.InitConstants()
method_dict         = {
    "Brent"         : "br",
    "Diff-Evol"     : "de",
    "L-BFGS-B"      : "l",
    "Nelder-Mead"   : "nm",
    "Powell"        : "p", 
    "SLSQP"         : "s",
    "TNC"           : "t",
    "Basin-Hop"     : "bh",
    "SHGO"          : "sh",
    "Dual-Anneal"   : "da",
    }
# methods Diff-Evol and TNC are the ones that work well

model_dict = {
    "NFW"       : "n",
    "cNFW"      : "c",
    "tNFW"      : "tn",
    "Unif"      : "u"
}

def PenaltyFunction(x, boundMin, boundMax, verbosity=0):
    """normalized penalty Function applied to likelihood 
    when parameters goes beyond bounds"""
    if boundMax < boundMin:
        raise ValueError("boundMax = " + str(boundMax) +\
                         "cannot be smaller than boundMin = " + str(boundMin))
    elif boundMax == boundMin:
        # penalty = 1000 sqrt(distance to bound)
        pf = 1000 * np.sqrt(np.abs(x-boundMin))
    else:
        # relative to bounds, from -1 to 1
        xRelative = -1 + 2 * (x-boundMin)/(boundMax-boundMin)
        
        # absolute value
        xRelAbs = np.abs(xRelative)
        
        # penalty: 1000 * sqrt(distance to nearest bound)
        pf = np.where(xRelAbs>1,1000*np.sqrt(np.abs(xRelAbs-1)),0)
        if verbosity >= 3:
            print("PenaltyFunction: x boundMin boundMax = ", x, 
                  boundMin, boundMax,  " penalty = ", pf)
    return pf

def guess_center(RA, Dec):
    return np.median(RA), np.median(Dec)

def PROFCL_ProbGalaxy(R_ellip_over_a, R_min, R_max, N_tot, 
                      scale_radius, Nofa, field_surfdensity,
                      ellipticity, R_cut, DeltaCenter, model,
                      min_R_over_rminus2, DeltaRA, DeltaDec,
                      Tiny_Shift_Pos, delta_phis=None, iPass=0, verbosity=0):
    """probability of projected radii for given galaxy position, 
        model parameters and field surface density
    arguments:
        RA, Dec: celestial coordinates [deg]
        scale_radius: radius of 3D density slope -2) [deg]
        field_surfdensity: uniform field surface density [deg^{-2}]
        ellipticity [dimensionless]
        RA_cen, Dec_cen: (possibly new) center of model [deg]
        model: 'NFW' or 'coredNFW' or 'Uniform'
    returns probablity density p(data|model)"""  

    # author: Gary Mamon
    
    a = scale_radius    # for clarity
    e = ellipticity     # for clarity
    if verbosity >= 2:
        print("PROFCL_ProbGalaxy: R_min=", R_min, "R_max=",R_max,"a=", a)

    if delta_phis is None:
        delta_phis = 0*R_ellip_over_a + 2*np.pi
    if delta_phis.min() > 2*np.pi*(1-cst.TINY):
        DeltaNproj_tilde = du.ProjectedNumber_tilde(R_max/a,e,R_cut/a,model, 
                                                    DeltaRA,DeltaDec,
                                                    DeltaCenter,
                                                    min_R_over_rminus2,
                                                    Tiny_Shift_Pos) \
                         - du.ProjectedNumber_tilde(R_min/a,e,R_cut/a,model, 
                                                    DeltaRA,DeltaDec,
                                                    DeltaCenter,
                                                    min_R_over_rminus2,
                                                    Tiny_Shift_Pos) 
    else:
        raise ValueError("mask not yet implemented in ProbGalaxy")
        # numerator = delta_phis*R_ellip_over_a*(Nofa/)
    if field_surfdensity < 1e-8:
        numerator = du.SurfaceDensity_tilde(R_ellip_over_a, model, R_cut/a)
        denominator = a*a * (1.-e) * DeltaNproj_tilde
    else:
        area = np.pi*(R_max*R_max-R_min*R_min)
        if N_tot > 0: # PoissonCount = False
            Nofa = (N_tot - area*field_surfdensity) \
                    / DeltaNproj_tilde
            denominator = N_tot
        else:   # PoissonCount = True
            denominator = Nofa*DeltaNproj_tilde + area*field_surfdensity
        numerator = Nofa/(np.pi * a*a * (1-e)) \
                     * du.SurfaceDensity_tilde(R_ellip_over_a,model,R_cut/a) \
                    + field_surfdensity

    return (numerator / denominator)

def PROFCL_ProbGalaxyCircular(data, R_min, R_max, params, aux):

    """probability of 2D radii for given galaxy position and scale radius 
    (assuming circular model)
    arguments:
        data
            R: projected radius [deg or Mpc/h]
            p_mem: probabilities of membership or None
        R_min, R_max: minimum and maximum considered R [deg or Mpc/h]
        params: 
            scale_radius: radius of 3D density slope -2) [deg or Mpc/h]
            Nofa: model normalization, N(r_{-2})
            r_cut: model physical  truncation radius (big value if none)
            field_surface_density: [deg^-2 or (Mpc/h)^-2]
        aux:
            DeltaNproj_tilde: dimensionless projected number between R_min and R_max
            model: 'NFW' or 'tNFW' or 'cNFW'
            fieldratio: use log(Sigma_field/Sigma_model) as variable 
                        instead of log(Sigma_field)
            denomNtotObs: True for cheat on denominator
            noMix: True for no mixed model-field
            deltaphiGals: premeasured azimuthal ranges at radial positions of galaxies
            deltaphiLin: linearly spaced azmuthal ranges
            thetaLin: linearly spaced angular radii
            verbosity: 0 for none
    returns probablity density p(data|model)"""  

    # author: Gary Mamon
 
    theta, p_mem = data
    [DeltaNproj_tilde,model,fieldratio,denomNtotObs,noMix,
     deltaphiGals,deltaphiLin,thetaLin,iPass,verbosity] = aux
    
            
    if verbosity >= 1:
        print("in prob: N_data=",len(data[0]))
        print("DeltaNproj_tilde model fieldratio denomNtotObs deltaphiGals=",
              DeltaNproj_tilde,model,fieldratio,denomNtotObs,deltaphiGals)
    
    # convert None or scalar p_mem to array of 1s.
    if verbosity >= 1:
        if p_mem is None:
            print("p_mem = None")
        elif type(p_mem) is int:
            print("p_mem = ", p_mem)
        else:
            print("p_mem.min =",p_mem.min())
    if (p_mem is None) or (p_mem is int):
        p_mem = 0*theta + 1

    # parameters
    if verbosity >= 1:
        print("len params = ",len(params))
    if len(params) == 4:
        if fieldratio:
            log_scale_radius, log_Nofa, log_r_cut, log_field_ratio = params
        else:
            [log_scale_radius, log_Nofa, log_r_cut, log_field_surface_density] \
             = params
    elif len(params) == 3:
        if fieldratio:
            log_scale_radius, log_Nofa, log_field_ratio = params
        else:
            log_scale_radius, log_Nofa, log_field_surface_density = params
        log_r_cut = 10
        
    # simpler notation
    scale_radius = 10**log_scale_radius
    a = scale_radius
    Nofa = 10**log_Nofa
    r_cut = 10**log_r_cut
    Sigma_field = 10**log_field_surface_density
    Nobs = len(theta)
    DeltaRsq = R_max*R_max - R_min*R_min
    if (verbosity >= 1) & (iPass==1):
        print("a N(a) r_cut=",a,Nofa,r_cut)
        print("min-theta max-theta R-min R-max=",theta.min(),theta.max(),
              R_min,R_max)
    
    # shortcuts for no mask
    if deltaphiGals is None:
        DeltaNprojtilde = du.ProjectedNumber_circ_tilde(R_max/a,model,r_cut/a) \
                         - du.ProjectedNumber_circ_tilde(R_min/a,model,r_cut/a) 
                      
        Npred = Nofa*DeltaNprojtilde + np.pi*DeltaRsq*Sigma_field
        Sigma_model = Nofa/(np.pi*a*a)*du.SurfaceDensity_tilde(theta/a, model)
 
    # case of galaxy p(theta) independent  of membership probabilities 
    if (p_mem.min() > 0.99) or noMix: # same p_mem values
        if deltaphiGals is None: # no mask
            numerator = 2*np.pi*theta*(Sigma_model + Sigma_field)
        else: # with mask
            numerator = deltaphiGals*theta*(Sigma_model + Sigma_field)
        if denomNtotObs: # (Sarazin 80):
            if deltaphiGals is None: # no mask
                denominator = Nobs
                if (verbosity >= 1) & (iPass==1):
                    print('Sarazin 80')
            else:
                raise ValueError("Cannot have denomNtotObs=True with mask")
        else: # predicted denominator
           if deltaphiGals is None: # no mask
               denominator = Npred
               if (verbosity >= 1) & (iPass==1):
                    print('Sarazin 80 with predicted N_tot')
           else: #mask
               y = deltaphiLin*thetaLin * \
                     (Nofa/(np.pi*a*a)* \
                      du.SurfaceDensity_tilde(thetaLin/a, model) + Sigma_field)
               denominator = integrate.simpson(y,thetaLin)
        prob = numerator / denominator
    else:
        # mix cluster and field models
        if deltaphiGals is None: # no mask
            p_cluster = 2*theta/(a*a)*du.SurfaceDensity_tilde(theta/a, model) \
                    / DeltaNprojtilde
            p_field = 2*theta/DeltaRsq
        else: # mask
            p_cluster_num = deltaphiGals * theta \
                * du.SurfaceDensity_tilde(theta/a, model)
            y = deltaphiLin*thetaLin*du.SurfaceDensity_tilde(thetaLin/a, model)
            p_cluster_denom = integrate.simpson(y,thetaLin)
            p_cluster = p_cluster_num / p_cluster_denom
            numerator = deltaphiGals * theta # *Sigma_field (omitted for speed)
            y = deltaphiLin*thetaLin         # *Sigma_field (omitted for speed)
            denominator = integrate.simpson(y,thetaLin)
            p_field = numerator/denominator
        prob = p_mem*p_cluster + (1-p_mem)*p_field

    # return probability
    
    prob = np.where(theta > r_cut, 0, numerator/denominator)
    return prob

def PROFCL_LogPoissonCount(R, Nexpected, R_min, R_max):
    """"ln(Poisson probability) of having a given number of galaxies in a given range, 
    given its expectation"""
    # author: Gary Mamon
    
    # extract objects in desired range of projected radii
    R = R[((R>R_min) & (R<R_max))]
    
    return poisson.logpmf(len(R),Nexpected)

def PROFCL_Rescale_Params(params_in_fit, R_min, R_max):
    [RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_field_surface_density, 
     log_R_cut] = params_in_fit

    # re-sccale
    RA_cen          *= R_max * np.cos(Dec_cen*cst.DEGREE)
    Dec_cen         *= R_max
    ellipticity     *= 5.
    PA              *= 500.
    log_field_surface_density  += 1.   # 10 x field 
    log_R_cut       += 0.7  #  5 x R_cut
    return [RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, 
        log_field_surface_density, log_R_cut]

def PosLogLikelihood(params, *args):
    """Positive general ln(L) for emcee"""
    return -1*PROFCL_LogLikelihood(params,args)

def PosLogLikelihoodCircular(params, *args):
    """Positive circular ln(L) for emcee"""
    return -1*PROFCL_LogLikelihoodCircular(params,args)

def PROFCL_LogLikelihood(params, *args, **kwargs):
    """general negative log likelihood of cluster given galaxy positions
    arguments:
        params: (RA_cen, Dec_cen, log_scale_radius, log_Nofa, ellipticity, PA,
                 log_field_surface_density, log_r_cut)
            RA_cen, Dec_cen: coordinates of cluster center [floats]
            log_scale_radius: log of scale radius of the profile (deg) [float]
            log_Nofa: log of model normalization N(a), 
                number of galaxies in sphere of radius a [float]
            ellipticity: ellipticity of cluster 
                (1-b/a, so that 0 = circular, and 1 = linear) [float]
            PA: position angle of cluster (DEGREEs from North to East) [float]
            log_field_surface_density: uniform field of cluster (in deg^{-2}) [float]
            log_r_cut: log of radius where model is cut (in deg) [float]
        args: (data, aux, bounds, mask_fun)
            data: RA,Dec,p_mem [np array of floats with first two in deg]
            aux: see below
            bounds: ((min,max) for each parameter)
            mask_fun: function(RA,Dec) to read mask
        aux:  (model,method,tolerance,N_points,R_min,R_max,theta_sky,N_tot,
               rescale_flag,
              RA_cen_init,Dec_cen_init,Tiny_Shift_Pos,
              min_R_over_rminus2,max_R_over_rminus2,
              # min_ellip,max_ellip,
              ellipFit,
              mask_fun,delta_phis,
              debug_file,iPass,verbosity)
            
    returns: -log likelihood [float]
    assumptions: not too close to celestial pole 
    (solution for close to celestial pole [not yet implemented]: 
    1. convert to galactic coordinates, 
    2. fit,
    3. convert back to celestial coordinates for PA)"""

    # authors: Gary Mamon with help from Yuba Amoura, Christophe Adami 
    #          & Eliott Mamon
        
    global iPass
    (data,aux,bounds,mask_fun) = args
    
    (RA_gal,Dec_gal,prob_membership) = data
    
    (model,method,tolerance,N_points,R_min,R_max,theta_sky,N_tot,rescale_flag,
     RA_cen_init,Dec_cen_init,Tiny_Shift_Pos,
     min_R_over_rminus2,max_R_over_rminus2,
     # min_ellip,max_ellip,
     ellipFit,
     mask_fun,delta_phis,
     debug_file,iPass,verbosity) = aux
    
    # check that all parameters are within bounds
    for j,par in enumerate(params):
        # print("type(par)=",type(par),"type(boundsmin)=",type(bounds[j,0]))
        if ((par < bounds[j,0]) or (par > bounds[j,1])):
            if verbosity >= 3:
                print("log_prior: par = ",par,"is out of bounds:", 
                      bounds[j,0],bounds[j,1])
            return np.inf
    
    if N_tot == 0:
        PoissonCount = True
    else:
        PoissonCount = False
        
    if verbosity >= 2:
        print("ellipFit=",ellipFit,"PoissonCount=",PoissonCount)
    if verbosity >= 3:
        for i in range(len(params)):
            print("i param bounds=",i,params[i],bounds[i])

    # if verbosity >= 4:
    #     print("entering LogLikelihood: R_min=",R_min)
    try: iPass
    except NameError:
        iPass = 1
    else:
        iPass += 1

    # read function arguments (parameters and extra arguments)
    if rescale_flag:
        params = PROFCL_Rescale_Params(params)
        
    # individual parameters
    [RA_cen, Dec_cen, log_scale_radius, log_Nofa, ellipticity, PA, 
     log_field_surface_density, log_r_cut] =  params
    
    scale_radius = 10 ** log_scale_radius
    a = scale_radius # shortcut
    Nofa = 10**log_Nofa
    field_surface_density = 10**log_field_surface_density
    r_cut = 10 ** log_r_cut

    if verbosity >= 2:
        print("min(R) max(R) R_min R_max r_s=",np.min(theta_sky),np.max(theta_sky),
              R_min,R_max,scale_radius)

    # tests
    
    if (len(RA_gal) == 0 or len(Dec_gal) == 0 or len(prob_membership) == 0):
        print("ERROR in PROFCL_lnlikelihood: ",
          "no galaxies found with projected radius between ",
          R_min, " and ", R_max,
          " around RA = ", RA_cen, " and Dec = ", Dec_cen)
        return cst.HUGE

    # if R_max/scale_radius > max_R_over_rminus2:
    #     if verbosity >= 1:
    #         print("PROFCL_lnlikelihood: R_max a max_Rovera = ", R_max, 
    #               scale_radius, max_R_over_rminus2)
    #     # return a -ln L that increases with off bound on R/r_minus2
    #     return 1000*np.sqrt(R_max/scale_radius-max_R_over_rminus2)
 
    # # check that coordinates are not too close to Celestial Pole
    # max_allowed_Dec = 80.
    # Dec_abs_max = np.max(np.abs(Dec_gal))
    # if Dec_abs_max > max_allowed_Dec:
    #     raise ValueError("in PROFCL_lnlikelihood: max(abs(Dec)) = "
    #                 + str(Dec_abs_max) + " too close to pole!")
    
    if ellipFit:
        ## transform from equatorial to cartesian and then to projected radii
        
        cosPA         = mu.cosd(PA)
        sinPA         = mu.sind(PA)
    
        # a = 10. ** log_scale_radius
        # DeltaRA = RA_cen - RA_cen_init
        # DeltaDec = Dec_cen - Dec_cen_init
        Delta_x_cen,Delta_y_cen = au.dxdy_from_RADec(RA_cen,Dec_cen,
                                                     RA_cen_init,Dec_cen_init)
    
        DeltaCenter = np.sqrt(Delta_x_cen*Delta_x_cen+Delta_y_cen*Delta_y_cen)
    
        # some minimizers will change the center when this is meant to be fixed
        # for N_points = 0, we change N_points to 1000 to compute a penalty
        # flag, and then turn back N_points to 0
        if DeltaCenter > Tiny_Shift_Pos and N_points == 0:
            N_points = 1000
            N_points_flag = True
        else:
            N_points_flag = False
            
        # DeltaCenter_over_a = DeltaCenter / a
        u,v = au.uv_from_RADec(RA_gal,Dec_gal,RA_cen,Dec_cen,
                               sinPA,cosPA)
        if verbosity >= 2:
            print("Log_Likelihood: len(u) = ", len(u))
    
        # elliptical radii (in DEGREEs)
        R_ellip = au.R_ellip_from_RADec(RA_gal,Dec_gal,ellipticity,
                                        RA_cen,Dec_cen,
                                        sinPA,cosPA)
    else: # circular fit
        R_ellip = au.AngularSeparation(RA_gal, Dec_gal, RA_cen, Dec_cen)
        DeltaCenter = 0
        N_points_flag = False
    # print("past IF block: N_points_flag=", N_points_flag)
    
    # dimensionless elliptical radii
    R_ellip_over_a = R_ellip / scale_radius
    if verbosity >= 3:
        print("R_min=",R_min,"R_max=",R_max,"r-2=", scale_radius,
              " Sigma_Field=",field_surface_density)
        print("median(R_ellip/a) = ",np.median(R_ellip_over_a))
        
    # offsets
    DeltaRA = RA_cen - RA_cen_init
    DeltaDec = Dec_cen - Dec_cen_init
    # DeltaRA_over_a = (RA_cen - RA_cen_init) / scale_radius
    # DeltaDec_over_a = (Dec_cen - Dec_cen_init) / scale_radius
    
    # check that elliptical radii are within limits of Mathematica fit
    
    if N_points == 0 and ellipFit and (np.any(R_ellip_over_a<min_R_over_rminus2) 
                          or np.any(R_ellip_over_a > max_R_over_rminus2)):
        if verbosity >= 2:
            print("log_likelihood: off limits for r_s minXallow maxallow",
                  " min(X) max(X)= ", 
                  scale_radius, min_R_over_rminus2, max_R_over_rminus2, 
                  np.min(R_ellip_over_a), np.max(R_ellip_over_a))
        return cst.HUGE
    elif verbosity >= 3:
        print("OK for r_s = ", scale_radius)

    ## likelihood calculation
    
    iPass2 = iPass
    prob = PROFCL_ProbGalaxy(R_ellip_over_a, R_min, R_max, N_tot,
                             scale_radius, Nofa, field_surface_density,
                             ellipticity, r_cut, DeltaCenter, model,
                             min_R_over_rminus2, DeltaRA, DeltaDec,
                             Tiny_Shift_Pos, delta_phis, iPass2, verbosity)
    
    if verbosity >= 3:
        print("\n\n PROBABILITY = ", prob, "\n\n")
    if np.any(prob<=0):
        if verbosity >= 1:
            print("one p = 0, EXITING LOG_LIKELIHOOD FUNCTION")
            print("1st 3 p = ",np.transpose([R_ellip_over_a,prob]))
        return cst.HUGE

    if verbosity >= 3:
        print("OK")

    # log likelihood
    #    (2nd term is useless for parameter optimization, 
    #    but is to give a reasonable value)
    lnlikminus_radii = -1*np.sum(prob_membership*np.log(prob)) \
        - np.sum(np.log(2*np.pi*R_ellip))
    if np.isnan(lnlikminus_radii).sum() > 0:
        print("lnlikminus contains NaNs!")
    if verbosity >= 2:
        if iPass == 1:
            print(np.transpose([R_ellip_over_a,prob]))
        print("ipass N log(theta_s) log(theta_cut) log(sdens) e PA -ln L = ",
              iPass,
              len(R_ellip_over_a),log_scale_radius,log_r_cut,
              log_field_surface_density,ellipticity,PA,RA_cen,Dec_cen,lnlikminus_radii)
        
    # penalization
    sumPenalization = 0.
    if len(params) != len(bounds):
        raise ValueError(str(len(bounds)) + "pairs of bounds do not match " \
                         + str(len(params)) + "parameters")
    if verbosity >= 3:
        print("bounds=",bounds)
    for i in range(len(params)):
        penalization = PenaltyFunction(params[i],bounds[i][0],bounds[i][1])
        sumPenalization += penalization
    lnlikminus_radii += sumPenalization
    
    if verbosity > 0 and penalization > 0:
        print("penalization > 0")
        for i in range(len(params)):
            print("i param bounds=",i,params[i],bounds[i])
            
    if not ellipFit:
        DeltaNproj_tilde = du.ProjectedNumber_tilde(R_max/a,0,r_cut/a,model, 
                                                   0,0,0,
                                                   0.001,1000) \
                         - du.ProjectedNumber_tilde(R_min/a,0,r_cut/a,model, 
                                                   0,0,0,
                                                   0.001,1000)
    if PoissonCount is not None:
        Nexpected = Nofa*DeltaNproj_tilde \
            + np.pi*(R_max*R_max-R_min*R_min)*field_surface_density
        if PoissonCount == 'expo':
            lnlikminus_count = -Nexpected
        elif PoissonCount == 'poisson':
            lnlikminus_count = -1 * poisson.logpmf(len(R_ellip),Nexpected)
        lnlikminus = lnlikminus_radii + lnlikminus_count
        if verbosity > 0:
            print("-lnL radii counts total = ",lnlikminus_radii,lnlikminus_count,
                  lnlikminus)
            print("Nexpected len(N)=",Nexpected,len(R_ellip))
    else:
        lnlikminus = lnlikminus_radii

    if N_points_flag: # WHAT IS THIS?
        N_points = 0
        N_points_flag = False
        
    # optional print 
    if verbosity >= 2:
        print(">>> pass = ", iPass,
              "-lnlik = ",       lnlikminus,
              # "penalization = ",  sumPenalization,
              "RA_cen = ",           RA_cen,
              "Dec_cen = ",          Dec_cen,
              "log_scale_radius = ", log_scale_radius,
              "ellipticity = ",      ellipticity,
              "PA = ",               PA,
              "log_field_surface_density = ",   log_field_surface_density
              )
        print("pwd --> ", os.getcwd())
        np.savetxt(debug_file + str(iPass),
                   np.c_[RA_gal,Dec_gal,u,v,R_ellip,prob])
        ftest = open(debug_file + str(iPass),'a')
        # ftest.write("{0:8.4f} {1:8.4f} {2:10.3f} {3:5.3f} {4:3.0f} {5:6.2f} {6:10.3f}\n".format(
        #     RA_cen, Dec_cen, log_scale_radius, log_Nofa, ellipticity, PA, 
        #     log_field_surface_density, lnlikminus))
        ftest.write('%8.4f'%RA_cen + ' %8.4f'%Dec_cen \
                    + ' %8.3f'%log_scale_radius \
                    + '%8.3f'%log_Nofa + ' %5.3f'%ellipticity + '3.0f'%PA \
                    + '6.2f'%log_field_surface_density + ' %5.3f'%log_r_cut \
                    + '%10.3f'%lnlikminus)
        ftest.close()

    # return -ln likelihood (which may include penalization, see above)
    return lnlikminus

def PROFCL_LogLikelihoodCircular(params, *args):
    """general -log likelihood of cluster given galaxy positions
    arguments:
        params: (log_scale_radius, log_R_cut)
            log_scale_radius: log of scale radius of the profile 
              (where scale radius is in Mpc/h) [float]
            log r_cut: log of truncation radius (Mpc/h)
        bounds: bounds on parameters
        data:
            R (deg or Mpc/h)
            p_mem (array or None)
        aux: (model, method, tolerance, maxfev, move, Nwalkers, seed, pool, autocorrectquiet,
              Rmin & Rmax (deg or Mpc/h), denomNtotObs, noMix, PoissonCount, ipass, 
              fieldratio, rescale_flag, min_R_over_rminus2,max_R_over_rminus2,
              mask_fun,deltaphiGals,deltaphiLin,thetaLin,debug_file,iPass,
              verbosity)
    returns: -log likelihood [float]"""

    # author: Gary Mamon
        
    # print("in LogLikelihoodCircular: args=",args)
    # print("len(args)=",len(args))
    # print("len(args[0])=",len(args[0]))
    
    # if len(args) == 1:
    #     args = args[0]
    #     # print("args is now",args)
    #     # print("len(args) is now",len(args))
    # (R,aux,bounds) = args
    # (model,method,tolerance,R_min,R_max,PoissonCount,fieldratio,iPass,
    #    verbosity) = aux
    # print("aux=",aux)
    # if verbosity >= 1:
    print("LogLikelihood_Circular: len(args)=",len(args))
    print("args=",args)
    if len(args) == 1:
        args = args[0]
    [data, aux, bounds] = args
    [model,R_min,R_max,
     denomNtotObs,noMix, PoissonCount,fieldratio,rescale_flag,
         min_R_over_rminus2,max_R_over_rminus2,
         mask_fun,deltaphiGals,deltaphiLin,thetaLin,debug_file,iPass,
         verbosity] = aux
    theta, p_mem = data
    if verbosity >= 1:
        print("type(p_mem)=",type(p_mem))
    Ndata = len(theta)

    iPass += 1

    # check if parameters are in bounds
    if verbosity > 0:
        print("bounds = ", bounds)
    for j,par in enumerate(params):
        # print("type(par)=",type(par),"type(boundsmin)=",type(bounds[j,0]))
        if verbosity >= 1:
            print("bounds-j=",bounds[j],"type=",type(bounds[j]))
            print("par=",par)

        if ((par < bounds[j,0]) or (par > bounds[j,1])):
            if verbosity >= 1:
                print("*** log_prior: par = ",par,"is out of bounds:", 
                      bounds[j,0],bounds[j,1])
            return np.inf

    # read function arguments (parameters and extra arguments)
    if verbosity >= 2:
        print("fieldratio len(params)=",fieldratio,len(params))
    if len(params) == 4:
        if fieldratio:
            log_scale_radius, log_Nofa, log_r_cut, log_field_ratio = params
        else:
            [log_scale_radius, log_Nofa, log_r_cut, log_field_surface_density] \
             = params
    elif len(params) == 3:
        if fieldratio:
            log_scale_radius, log_Nofa, log_field_ratio = params
        else:
            log_scale_radius, log_Nofa, log_field_surface_density = params
        log_r_cut = 10
    else:
        raise ValueError("params should include log(r_s), log(N_3D(r_s))," \
                         + " log(Sigma_field), and possibly log(r_cut)")
        # log_scale_radius = params
        # log_field_surface_density = -99
        # log_R_cut = 1
    scale_radius = 10 ** log_scale_radius
    a = scale_radius # for short
    Nofa = 10 ** log_Nofa 
    if fieldratio:
        field_surface_density = 10**log_field_ratio*Nofa/(np.pi*a**2)
    else:
        field_surface_density = 10**log_field_surface_density
    r_cut = 10 ** log_r_cut

    ## likelihood calculation
    
    DeltaNproj_tilde = du.ProjectedNumber_circ_tilde(R_max/a,model,r_cut/a) \
                     - du.ProjectedNumber_circ_tilde(R_min/a,model,r_cut/a)

    # prob = PROFCL_ProbGalaxyCircular(R, R_min, R_max, scale_radius, Nofa, 
    #                                  r_cut, field_surface_density,  
    #                                  DeltaNproj_tilde, model, verbosity)
    aux = [DeltaNproj_tilde,model,fieldratio,denomNtotObs,noMix,
           deltaphiGals,deltaphiLin,thetaLin,iPass,verbosity]
    prob = PROFCL_ProbGalaxyCircular(data, R_min, R_max, params, aux)
    if (verbosity >= 1) & (iPass==1):
        print(np.transpose([data[0],prob]))
    
    if verbosity >= 3:
        print("params=",params)
        print("\n\n PROBABILITY = ", prob, "\n\n")
    if np.any(prob<=0):
        if verbosity >= 1:
            print("one p = 0, EXITING LOG_LIKELIHOOD FUNCTION")
        return cst.HUGE

    if verbosity >= 1:
        print("PoissonCount denomNtotObs =",PoissonCount,denomNtotObs)
    if verbosity >= 2:
        print("OK")

    # -ln L on radii 
    #    (2nd term is useless for parameter optimization, 
    #    but is to give a reasonable value)
    lnlikminus_radii = -1*np.sum(np.log(prob)) - np.sum(np.log(2*np.pi*theta))
    if noMix: # -ln L = - sum_i q_mem_i ln p(theta_i)
        lnlikminus_radii = p_mem*lnlikminus_radii

    # extra term in -ln L for shot noise
    # default 0
    lnlikminus_count = 0*lnlikminus_radii
    if PoissonCount is not None:
        Nexpected = Nofa*DeltaNproj_tilde \
           + np.pi*(R_max*R_max-R_min*R_min)*field_surface_density
        if PoissonCount == 'expo':
           lnlikminus_count = Nexpected
           if (verbosity >= 1) & (iPass==1):
               print("PoissonCount=expo")
        elif PoissonCount == 'poisson':
           lnlikminus_count = -1 * poisson.logpmf(Ndata,Nexpected)
        lnlikminus = lnlikminus_radii + lnlikminus_count
        if verbosity >= 0.5:
            print("params lnLR lnLct lnL =",params,lnlikminus_radii,
                  lnlikminus_count,lnlikminus)
        if verbosity >= 2:
           print("-lnL radii counts total = ",lnlikminus_radii,lnlikminus_count,
                 lnlikminus)
           print("Nexpected len(N)=",Nexpected,Ndata)
    else:
        if (verbosity >= 1) & (iPass==1):
            print('No added term to -ln L')
    lnlikminus = lnlikminus_radii + lnlikminus_count

    # penalize field
    # if penalize_field:
    #     penalty = np.sqrt(log10(field_surface_density_max))
    # penalty = 0
    # for i, param in enumerate(params):
    #     penalty += Penalty(param)
    if verbosity >= 1:
        print("params nll_R nll=",params,lnlikminus_radii,lnlikminus,
              PoissonCount)
        
    return lnlikminus 

def Azimuth(RA,Dec,RA_cen,Dec_cen,mask_fun,phiStep=1):
    # grid of phi
    phis = np.arange(0,360,phiStep)
    # compute angles
    thetas = au.AngularSeparation(RA,Dec,RA_cen,Dec_cen)
    Azimuth = 0
    for i, theta in enumerate(thetas):
        # convert theta,phi to RA,Dec
        RAdum,Decdum = au.polar2eq(thetas, phis, RA_cen, Dec_cen)
        masktheta = mask_fun(RAdum,Decdum)
        Azimuth += masktheta
    # convert Azimuth to radians
    return Azimuth*cst.DEGREE

def AzimuthCircular(thetas,RA_cen,Dec_cen,mask_fun,phiStep=1):
    # grid of phi
    phis = np.arange(0,360,phiStep)    
    Azimuth = 0
    for i, theta in enumerate(thetas):
        # convert theta,phi to RA,Dec
        RAdum,Decdum = au.polar2eq(thetas, phis, RA_cen, Dec_cen)
        masktheta = mask_fun(RAdum,Decdum)
        Azimuth += masktheta
    # convert Azimuth to radians
    return Azimuth * cst.DEGREE

def PROFCL_FitCircular(params, bounds, data, aux):
    '''Maxmimum Likelihood Estimate of 2D scale radius 
    of isolated spherical system of known center
       (where slope of 3D density profile is -2) given galaxy positions on sky
    arguments:
        params = (log_scale_radius,log_Nofa,log_r_cut,log_field_surfdensity, 
                  where:
                log_scale_radius: log of scale radius (deg or Mpc/h) [float]
                log_Nofa: model normalization log(N(r_{-2}))
                log_r_cut: sharp truncation of 3D density profile 
                    (deg or Mpc/h, 0 for none) [float]
                log_field_surface_density: log of field surface density 
                    (deg^-2 or [Mpc/h]^-2)
        bounds = ((min,max),...) for each parameter
        data:
            R = 2D radii (deg or Mpc/h) (before cut in interval)
            p_membership (array or None)
        aux = (model,method,tolerance, maxfev,
               move, Nwalkers, seed, pool, autocorrquiet,
               R_min,R_max,denomNtotObs,PoissonCount,fieldRatio,rescale_flag,
               min_R_over_rminus2,max_R_over_rminus2, 
               mask_fun, deltaphiGals,deltaphiLin,thetaLin,
               debug_file, iPass, verbosity), 
            where
                model: density model (NFW or coredNFW or Uniform) [char]
                method:    minimization method (first two recommended) [string]
                        'diff-evol':    Differential Evolution
                        'Powell':       Powell
                        'Nelder-Mead':  Simplex 
                        'BFGS':         Broyden-Fletcher-Goldfarb-Shanno, 
                            using gradient
                        'L-BFGS-B':     Broyden-Fletcher-Goldfarb-Shanno, 
                            using gradient
                        'SLSQP':        Sequential Least-Squares Programming
                        'Newton-CG':    Newton's conjugate gradient
                        'Brent':        Brent method (only for univariate data)
                tolerance: relative allowed tolerance [float[]]
                R_min:     minimum allowed projected radius (deg or Mpc/h) [float]
                R_max:     maximum allowed projected radius (deg or Mpc/h) [float]
                denomNtotObs:
                    True: for cheat on denominator of prob = N_tot,obs
                    False: denominayor of prob = N_tot,pred
                PoissonCount: 
                   'expo': constant exp(-Ntot,model), recommended
                   'poisson': naive Poisson extra term in likelihood
                   None: no handling of Poisson counts
                   this adds free parameter N(a)
                   and handle Poisson galaxy count [str]
                fieldRatio: use log(Sigma_Field/Sigma_mean_model(r_-2))
                  instead of log(Sigma_field)
          mask_fun: function (RA,Dec) returning mask (0, 1 and in between)
                '''

    # authors: Gary Mamon
    
    global iPass
    
    R, p_mem = data
    p_mem = FixProbMembership(p_mem,len(R))
    [model,method,tolerance,maxfev,move,Nwalkers,seed,pool,autocorrquiet,
     R_min,R_max,
     denomNtotObs,noMix,PoissonCount,fieldratio,rescale_flag,
     min_R_over_rminus2,max_R_over_rminus2,
     mask_fun,deltaphiGals,deltaphiLin,thetaLin,debug_file,
     iPass,verbosity] = aux
    if verbosity >= 1:
        print("in FitCircular: R = ", R)
        print("aux=",aux)
    ## checks on values of arguments
    
    # check that R_min > 0 (to avoid infinite surface densities)
    if R_min < 0:
        raise ValueError("_min = " + str(R_min) + "must be >= 0")
                             
    # check that R_max > R_min
    if R_max <= R_min:
        raise ValueError("R_max = " + str(R_max) +
                         " must be > R_min = " + str(R_min))
                             
    # check model
    
    # print("*** in PROFCL_Fit, verbosity=",verbosity)
    allowedmodels = "NFW", "cNFW", "tNFW", "Kazantzidis"
    
    found = False
    for mod in allowedmodels:
        if mod == model:
            found = True
            break
    if found == False:
        raise ValueError("model = " +
                         model + " not recognized... must be in " \
                             + str(list(model_dict.keys())))
    
    # force radii in bounds
    if verbosity >= 0.5:
        print("before data filter: N_data=",len(R))  
    cond = (R>R_min) & (R<R_max)
    R = R[cond]
    if p_mem is not None:
        p_mem = p_mem[cond]
    data2 = [R,p_mem]
    if verbosity >= 0.5:
        print("after data filter: N_data=",len(R))
        
    # if verbosity >= 0.9:
    #     print()
    
    # fit
    
    iPass = 0
    myargs = (data2,aux,bounds)
    auxShort = [model,R_min,R_max,
                denomNtotObs,noMix,PoissonCount,fieldratio,rescale_flag,
                min_R_over_rminus2,max_R_over_rminus2,
                mask_fun,deltaphiGals,deltaphiLin,thetaLin,debug_file,
                iPass,verbosity]
    myargsShort = (data2,auxShort,bounds)
    if verbosity >= 1:
        print("FitCircular: myargs = ", myargs)
        print("myargs[0]=",myargs[0])
        print("myargs[1]=",myargs[1])
        print("myargs[2]=",myargs[2])
        print("verbosity=",verbosity)
        # sys.exit(0)
    if method[0:5] == "emcee":
        # random initial conditions
        # boundsfree = bounds[bounds[:,1]-bounds[:,0]>0.001]
        # ndim = len(boundsfree)
        # params0 = boundsfree[:,0] \
        #       + np.random.rand(Nwalkers,ndim)*(boundsfree[:,1]-boundsfree[:,0])
        ndim = len(bounds)
        params0 = bounds[:,0] \
              + np.random.rand(Nwalkers,ndim)*(bounds[:,1]-bounds[:,0])
        print("params0=",params0)
        if pool:
            print("parallel MCMC run...")
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(Nwalkers, ndim, 
                                                PosLogLikelihoodCircular,
                                                args=myargs,
                                                moves=move,pool=pool)
                start = time.time()
                sampler.run_mcmc(params0, maxfev, progress=True)
                end = time.time()
                multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        else:
            print("scaler MCMC run... args=",myargs)
            sampler = emcee.EnsembleSampler(Nwalkers, ndim, 
                                            PosLogLikelihoodCircular,
                                            args=myargs,
                                            moves=move())
            start = time.time()
            sampler.run_mcmc(params0, maxfev, progress=True)
            end = time.time()
            scalar_time = end - start
            print("Scaler run took {0:.1f} seconds".format(scalar_time))
        # print("{0:.1f} times faster than serial".format(serial_time / multi_time))
        # tau = sampler.get_autocorr_time(quiet=autocorrquiet)
        # samples = sampler.get_chain(discard=int(maxfev/10))
        # print("autocorrquiet=",autocorrquiet)
        # tau = emcee.autocorr.integrated_time(samples,quiet=autocorrquiet)
        return sampler

    elif method == ["emceeMPI"]:
        pass
    elif method == "Nelder-Mead":
        # return minimize(PROFCL_LogLikelihood, params, args=(RA, Dec, prob_membership, R_min, R_max, model), 
        #                 method=method, tol=tolerance, bounds=bounds, options={'fatol':tol, 'maxfev':maxfev})
        return minimize (PROFCL_LogLikelihoodCircular, params, args=myargsShort, 
                         method=method, tol=tolerance,
                         options={"maxfev":maxfev})
    elif method == "Powell":
        return minimize (PROFCL_LogLikelihoodCircular, params, args=myargsShort, 
                        method=method, tol=tolerance, bounds=bounds, 
                        options={"ftol":tolerance, "maxfev":maxfev})
    # elif method == "CG" or method == "BFGS":
    #     return minimize              (PROFCL_LogLikelihood, params, args=myargsShort, 
    #                     method=method, tol=tolerance, bounds=bounds, options={"gtol":tolerance, "maxiter":maxfev})
    # elif method == "Newton-CG":
    #     return minimize              (PROFCL_LogLikelihood, params, args=myargsShort, 
    #                     method=method, tol=tolerance, bounds=bounds, options={"xtol":tolerance, "maxiter":maxfev})
    elif method == "L-BFGS-B":
        return minimize (PROFCL_LogLikelihoodCircular, params, args=myargsShort, 
                        method=method, tol=tolerance, bounds=bounds, 
                        options={"ftol":tolerance, "maxfun":maxfev})
    
    elif method == "SLSQP":
        return minimize (PROFCL_LogLikelihoodCircular, params, args=myargsShort, 
                        method=method, jac=None, bounds=bounds, tol=None, 
                        options={"ftol":tolerance, "maxiter":maxfev})
    
    elif method == "TNC":
        return minimize (PROFCL_LogLikelihoodCircular, params, args=myargsShort, 
                        method=method, jac=None, bounds=bounds, tol=None, 
                        options={"xtol":tolerance, "maxiter":maxfev})
    
    elif method in ["Diff-Evol","DiffEvol","DE"]:
        return differential_evolution(PROFCL_LogLikelihoodCircular, bounds, 
                                      args=myargsShort, atol=tolerance)
    
    # elif method == "SHGO":
    #     return shgo(PROFCL_LogLikelihood, bounds, args=myargsShort)

    elif method == "Dual-Anneal":
        return dual_annealing(PROFCL_LogLikelihoodCircular, bounds, 
                              args=myargsShort, maxiter=500)
    
    # elif method == "Basin-Hop":
    #     return basinhopping(PROFCL_LogLikelihood, params, minimizer_kwargs = {"args": myargsShort})
    
    # elif method == "Brent":
    #     print ("bounds=",bounds)
    #     bounds2 = (bounds[0], (bounds[0]+bounds[1])/2, bounds[1])
    #     return brent                (PROFCL_LogLikelihood, brack=bounds2, 
    #                                   args=myargsShort, tol=tolerance)
    else:
        raise ValueError("method = " + method + " is not yet implemented")

def log_prior(pars,bounds,npars,verbosity=0):
    if len(bounds) != npars:
        print("len(bounds)=",len(bounds))
    for j,par in enumerate(pars):
        # print("type(par)=",type(par),"type(boundsmin)=",type(bounds[j,0]))
        if ((par < bounds[j,0]) or (par > bounds[j,1])):
            if verbosity >= 2:
                print("log_prior: par = ",par,"is out of bounds:", 
                      bounds[j,0],bounds[j,1])
            return -np.inf
    return 0

def PROFCL_Fit(params, bounds, data, aux):
    '''Maxmimum Likelihood Estimate of 3D scale radius 
       (where slope of 3D density profile is -2) given galaxy positions on sky
    arguments:
        params = (log_scale_radius, Nofa, log_r_cut, log_field_surfdens, 
                  ellipticity, PA, RA_cen, Dec_cen), where
                log_scale_radius: log of scale radius of the profile 
                    (where scale radius is in degrees) [float]
                log_Nofa: model normalization, N(a) 
                    (not used if PoissonCount [in aux] is False) [float]
                log_r_cut: log of sharp truncation of density profile 
                    (degrees, big for none) [float]
                ellipticity: ellipticity of cluster 
                    (1-b/a, so that 0 = circular, and 1 = linear) [float]
                PA: position angle of cluster (degrees from North to East)
                    [float]
                log_field_surfdens: uniform field surface density of cluster 
                    (in deg^{-2}) [float]
                RA_cen, Dec_cen: coordinates of cluster center (deg) [floats]
        bounds = ((min,max),...) for each parameter
        data = (RA_gal, Dec_gal, prob_membership)
        aux = (model, method, tolerance, maxfev, move, Nwalkers, seed, N_points, 
                theta_min, theta_max, denomNtotObs, PoissonCount, fieldRatio,
                rescale_flag, RA_cen_init, Dec_cen_init,  
                Tiny_Shift_Pos,min_R_over_rminus2,max_R_over_rminus2, 
                mask_fun, deltaphiGals,deltaphiLin,thetaLin,
                debug_file, iPass, verbosity)
                model: density model (NFW or coredNFW or Uniform) [char]
                method:    minimization method [string]
                        'Powell':       Powell (with bounds)
                        'Nelder-Mead':  Simplex 
                        'BFGS':         Broyden-Fletcher-Goldfarb-Shanno, 
                                            using gradient
                        'L-BFGS-B':     Broyden-Fletcher-Goldfarb-Shanno, 
                                            using gradient
                        'SLSQP':        Sequential Least-Squares Programming
                        'Newton-CG':    Newton's conjugate gradient
                        'Brent':        Brent method (only for univariate data)
                        'diff-evol':    Differential Evolution
                tolerance: relative allowed tolerance [float[]]
                maxfev: max number of function evaluations [int]
                move: emcee Move strategy
                Nwalkers: number of walkers (chains) for MCMC
                seed: random seed for Differential-Evolution & emcee methods
                pool: parallel?
                autocorrquiet: do not stop if corrleation length is too small
                N_points: number of points for elliptical solutions [int]:
                        0: 2D polynomial approximation
                        >0: number of points for Monte Carlo integration
                        <0: log_10(tolerance) for 2D integration
                theta_min:     minimum allowed projected radius (degrees) [float]
                theta_max:     maximum allowed projected radius (degrees) [float]
                denomNtotObs: 
                    True for cheat on denominator of prob = N_tot,obs
                    False for denominator of prob = N_tot,pred
                PoissonCount: 
                    'expo': constant exp(-Ntot,model), recommended
                    'poisson': naive Poisson extra term in likelihood
                    None: no handling of Poisson counts
                    this adds free parameter N(a)
                    and handle Poisson galaxy count [str]
                fieldRatio: field parameter is log(field/Sigma_model(a))? [bool]
                rescale_flag:   rescale-data [bool]
                RA_cen_init:    initial RA_cen (degrees)
                Dec_cen_init:   initial Dec_cen (degrees)
                Tiny_Shift_Pos: maximum position shift to keep circular fit 
                    [degrees]
                min_R_over_rminus2: minimum allowed R/theta_minus2
                max_R_over_rminus2: maximum allowed R/theta_minus2
                mask_fun: function (RA,Dec) returning mask (0, 1 and in between)
                NthetasAzimuth: number of geometrically-spaced angular radii
                 for azimuthal range evaluation from mask
                deltaphiAzimuth: step in azimuths for  azimuthal range 
                 evaluation from mask (deg)
                debug_file
                iPass: set to 0
                verbosity: > 0 for debugging output
                (field_flag: flag for field fit 
                     (True for including field in fit, False for no field)
                recenter_flag: flag for recentering 
                    (True for recentering, False for keeping DETCL center)
                elliptical_flag: flag for elliptical fit 
                    (True for elliptical fit, False for circular fit)
                )
      
    '''

    # authors: Gary Mamon, with help from Yuba Amoura and Eliott Mamon
    
    global iPass
    
    # print("### in PROFCL_Fit")
    
    models = ["NFW", "cNFW", "tNFW", "Uniform"]
    methods = ['emcee','emceeMPI','Nelder-Mead','Powell','L-BFGS-B','SLSQP',
               'TNC','Diff-Evol','Dual-Anneal']
    moves = [eval('emcee.moves.' + move) for move in dir(emcee.moves)[0:9]]
    
    # auxiliary fit parameters
    [model,method,tolerance,maxfev,move,Nwalkers,seed,pool,autocorrquiet,
     backend,N_points,
     theta_min,theta_max,denomNtotObs,
     PoissonCount,fieldRatio,rescale_flag,
     RA_cen_init,Dec_cen_init,Tiny_Shift_Pos,
     min_R_over_rminus2,max_R_over_rminus2,
     mask_fun,NthetasAzimuth,deltaphiAzimuth,
     debug_file,iPass,verbosity] = aux
    
    # data
    RA,Dec,prob_membership = data
    prob_membership = FixProbMembership(prob_membership,len(RA))
    if verbosity >= 1:
        print("len RA Dec =",len(RA),len(Dec))
        if prob_membership is not None:
            print("len(p)=",len(prob_membership))
        else:
            print("p_mem = None")
        print("Fit: aux = ",aux)
    if verbosity >= 2:
        print("Fit: RA = ", RA)
        print("Fit: Dec = ", Dec)
        print("Fit: p_mem = ", prob_membership)
    
    # check model and method
    # convert model and method
    model = cst.Convert(model,cst.MODEL_DICT)
    method = cst.Convert(method,cst.METHOD_DICT)

    # projected radii (in degrees)
    # if theta_sky is None:
    #     print("RA_cen_init =",RA_cen_init)
    #     print("Dec_cen_init =",Dec_cen_init)
    if verbosity >= 0.5:
        print("before data filter: N_data=",len(RA))
    theta_sky = au.AngularSeparation(RA,Dec,RA_cen_init,Dec_cen_init,
                                 choice='trig')

    # if verbosity >= 1:
    #     print("len(theta_sky)=",len(theta_sky))
    # if verbosity >= 2:
    #     print("orig theta_sky = ", theta_sky)
    # force data to data limits
    # cond = np.logical_and(theta_sky >= theta_min, theta_sky <= theta_max)
    cond = (theta_sky >= theta_min) & (theta_sky <= theta_max)
    # if verbosity >= 1:
    #     print("len(cond) sum(cond)=",len(cond),sum(cond))
    # if verbosity >= 2:
    #     print("Fit: N_tot = ", len(theta_sky[cond]))
    #     # print("RA Dec=",RA,Dec)

    RA_good = RA[cond]
    Dec_good = Dec[cond]
    theta_sky_good = theta_sky[cond]
    if prob_membership is not None:
        p_mem_good = prob_membership[cond]
    else:
        p_mem_good = None
    if verbosity >= 0.5:
        print("after data filter: N_data=",len(RA_good))
    
    # if PoissonCount:
    #     N_tot = 0
    # else:
    #     N_tot = len(theta_sky_good)

    if verbosity >= 1:
        print("original bounds...")
        for i in range(len(bounds)):
            print(i,bounds[i])
            
    boundsFixed = []
    # fit for center?
    if ((np.abs(bounds[-2,1]-bounds[-2,0]) < cst.TINY) \
        & (np.abs(bounds[-1,1]-bounds[-1,0]) < cst.TINY)):
        centerFit = False
        boundsFixed.append(bounds[-2,0])
        boundsFixed.append(bounds[-1,0])
    else:
        centerFit = True
        
    # elliptical fit?
    if ((np.abs(bounds[4,1]-bounds[4,0]) < cst.TINY) \
        & (np.abs(bounds[5,1]-bounds[5,0]) < cst.TINY)):
        ellipFit = False
        boundsFixed.append((bounds[4,0]))
        boundsFixed.append((bounds[5,0]))
    else:
        ellipFit = True
        
    if verbosity >= 1:
        print("Fit: centerFit ellipFit=",centerFit,ellipFit)

    # Azimuthal intervals of good masks at data points
    if ((not ellipFit) & (mask_fun is not None)):
        delta_phis = Azimuth(RA,Dec,RA_cen_init,Dec_cen_init,mask_fun) 
    else:
        delta_phis = 0*RA + 2*np.pi
    
    if verbosity >= 4:
        print("entering Fit: theta_min=",theta_min)

    ## checks on types of arguments

    # check that input positions are in numpy arrays
    if not isinstance(RA,np.ndarray):
        raise ValueError("in PROFCL_fit: RA must be numpy array")

    if not isinstance(Dec,np.ndarray):
        raise ValueError("in PROFCL_fit: Dec must be numpy array")

    # check that min and max projected radii are floats or ints
    lu.CheckTypeIntorFloat(theta_min,"PROFCL_fit","theta_min")
    lu.CheckTypeIntorFloat(theta_max,"PROFCL_fit","theta_max")
                             
    ## checks on values of arguments
    
    # check that theta_min > 0 (to avoid infinite surface densities)
    if theta_min <= 0:
        raise ValueError("in PROFCL_fit: theta_min = " + str(theta_min) \
                         + "must be > 0")
                             
    # check that theta_max > theta_min
    if theta_max < theta_min:
        raise ValueError("in PROFCL_fit: theta_max = " + str(theta_max) +
                         " must be > theta_min = " + str(theta_min))
                             
    # check model
    # if model != "NFW" and model != "coredNFW" and  model != "Uniform":
    if model not in models:
        raise ValueError("in PROFCL_fit: model = " + model \
                         + " not recognized... must be NFW,coredNFW,Uniform")
    
    # function of one variable
    if np.isnan(RA_cen_init):
        raise ValueError("in PROFCL_fit: RA_cen_init is NaN!")
    if np.isnan(Dec_cen_init):
        raise ValueError("in PROFCL_fit: Dec_cen_init is NaN!")

    iPass = 0
        
    if ((~ellipFit) & (~centerFit)): # circular fit with fixed center
        print("circular fit...")
        data2 = [theta_sky_good,p_mem_good]
        np.savetxt(home_dir + 'TMP/R.dat',theta_sky_good,delimiter=' ')
        nllfun = PROFCL_LogLikelihoodCircular
        llfun = PosLogLikelihoodCircular
        params = params[0:4]
        bounds = bounds[0:4]
        if bounds[2,1]-bounds[2,0] < cst.TINY:
            # omit r_cut parameter
            bounds = np.delete(bounds,2,axis=0)
            params = np.delete(params,2)
            
        # azimuthal ranges
        if mask_fun is not None:
            deltaphiGals = AzimuthCircular(theta_sky_good, RA_cen_init, Dec_cen_init,
                                            mask_fun, deltaphiAzimuth)
            thetaLin = np.linspace(theta_min,theta_max,NthetasAzimuth)
            deltaphiLin = AzimuthCircular(thetaLin, RA_cen_init, Dec_cen_init,
                                            mask_fun, deltaphiAzimuth)
        else:
            deltaphiGals = None
            thetaLin = None
            deltaphiLin = None
        aux2 = [model,method,tolerance,maxfev,move,Nwalkers,seed,pool,
                autocorrquiet,
                theta_min,theta_max,
                denomNtotObs,PoissonCount,fieldRatio,rescale_flag,
                min_R_over_rminus2,max_R_over_rminus2,
                mask_fun,deltaphiGals,deltaphiLin,thetaLin,debug_file,iPass,
                verbosity]

        if verbosity >= 1:
            # print("Fit: len(data0) len(data1)=",len(data2[0]),len(data2[1]))
            print("Fit: aux2 = ",aux2)
            print("Fit: bounds=",bounds)
        
        return PROFCL_FitCircular(params, bounds, data2, aux2)
    else:
        nllfun = PROFCL_LogLikelihood
        llfun = PosLogLikelihood
        aux2 = (model,method,tolerance,N_points,theta_min,theta_max,theta_sky_good,
                denomNtotObs,PoissonCount,
                fieldRatio,rescale_flag,RA_cen_init,Dec_cen_init,Tiny_Shift_Pos,
                min_R_over_rminus2,max_R_over_rminus2,ellipFit,
                mask_fun,deltaphiGals,deltaphiLin,thetaLin,debug_file,iPass,
                verbosity)
        myargs = (data2,aux2,bounds)

    # fit
    
    if method[0:5] == "emcee":
        # random initial conditions
        # boundsfree = bounds[bounds[:,1]-bounds[:,0]>0.001]
        # ndim = len(boundsfree)
        # params0 = boundsfree[:,0] \
        #       + np.random.rand(Nwalkers,ndim)*(boundsfree[:,1]-boundsfree[:,0])
        ndim = len(bounds)
        params0 = bounds[:,0] \
              + np.random.rand(Nwalkers,ndim)*(bounds[:,1]-bounds[:,0])   
        print("before pool: params0=",params0)
        if pool:
            print("parallel MCMC...")
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(Nwalkers, ndim, llfun,
                                                args=[aux2,bounds],
                                                quiet=autocorrquiet,
                                                moves=move,pool=pool)
                start = time.time()
                sampler.run_mcmc(params0, maxfev, progress=True)
                end = time.time()
                multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))
        else:
            print("scalar MCMC...")
            sampler = emcee.EnsembleSampler(Nwalkers, ndim, llfun,
                                            args=[aux2,bounds],
                                            backend=backend,
                                            quiet=autocorrquiet,
                                            moves=move)
            start = time.time()
            sampler.run_mcmc(params0, maxfev, progress=True)
            end = time.time()
            scalar_time = end - start
            print("Scalar run took {0:.1f} seconds".format(scalar_time))
        # print("{0:.1f} times faster than serial".format(serial_time / multi_time))
        # tau = sampler.get_autocorr_time()
        # samples = sampler.get_chain(discard=int(maxfev/10))
        # print("samples.shape=",samples.shape)
        # print("autocorrquiet=",autocorrquiet)
        # tau = emcee.autocorr.integrated_time(samples,quiet=autocorrquiet)
        return sampler

    elif method == ["emceeMPI"]:
        pass
    elif method == "Nelder-Mead":
        # return minimize(llfun, params, args=(RA, Dec, prob_membership, theta_min, theta_max, model), 
        #                 method=method, tol=tolerance, bounds=bounds, options={'fatol':tol, 'maxfev':maxfev})
        return minimize  (nllfun, params, args=myargs, 
                          method=method, tol=tolerance, bounds=bounds,
                          options={"maxfev":maxfev})
    elif method == "Powell":
        if verbosity >= 1:
            print("launching Powell...")
        return minimize  (nllfun, params, args=myargs, 
                          method=method, tol=tolerance, bounds=bounds,
                          options={"ftol":tolerance, "maxfev":maxfev})

    # elif method == "CG" or method == "BFGS":
    #     return minimize              (llfun, params, args=myargs, 
    #                     method=method, tol=tolerance, bounds=bounds, options={"gtol":tolerance, "maxiter":maxfev})
    # elif method == "Newton-CG":
    #     return minimize              (llfun, params, args=myargs, 
    #                     method=method, tol=tolerance, bounds=bounds, options={"xtol":tolerance, "maxiter":maxfev})
    elif method == "L-BFGS-B":
        return minimize  (nllfun, params, args=myargs, 
                          method=method, tol=tolerance, bounds=bounds, 
                          options={"ftol":tolerance, "maxfun":maxfev})
    elif method == "SLSQP":
        return minimize  (nllfun, params, args=myargs, 
                        method=method, jac=None, bounds=bounds, tol=None, 
                        options={"ftol":tolerance, "maxiter":maxfev})
    elif method == "TNC":
        return minimize  (nllfun, params, args=myargs, 
                        method=method, jac=None, bounds=bounds, tol=None, 
                        options={"xtol":tolerance, "maxiter":maxfev})
    # elif method == "fmin_TNC":
    #     return fmin_tnc              (llfun, params, fprime=None, args=myargs,  approx_grad=True, bounds=bounds, epsilon=1.e-8, ftol=tolerance)
    elif method == "Diff-Evol":
        return differential_evolution(nllfun,bounds,seed=seed,
                                      args=myargs, atol=tolerance)
    
    elif method == "Dual-Anneal":
        return dual_annealing(nllfun, bounds, args=myargs, 
                              maxiter=500)
    else:
        raise ValueError("in PROFCL_fit: method = " + method + " is not yet implemented")


def PlotChains(samples,labels=None,title=None,prefix=None,nll=None):
    """plot emcee chains
    following example in 
    https://emcee.readthedocs.io/en/stable/tutorials/line/"""
    # samples = sampler.get_chain()
    ndim = samples.shape[-1]
    if nll is not None:
        nplots = ndim + 1
    else:
        nplots = ndim
    # rcParams['text.usetex'] = False
    formatter_limits = mpl.rcParams['axes.formatter.limits']
    mpl.rcParams['axes.formatter.limits']= [-8,8]
    fig, axes = plt.subplots(nplots, figsize=(10, 7), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], 'k', alpha=0.3)
        ax.set_xlim(0, len(samples))
        if labels is not None:
            ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        if (i == 0) & (title is not None):
            ax.set_title(title)
    if nll is not None:
        ax = axes[ndim]
        ax.plot(nll[:],'b',alpha=0.3)
        ax.set_ylabel('$-\ln L$')
    axes[-1].set_xlabel("step number")
    # if title is not None:
    #     fig.title(title)
    if prefix is not None:
        plt.savefig(prefix + '.pdf')
    else:
        plt.show()
    mpl.rcParams['axes.formatter.limits']= formatter_limits
    
def PlotCorner(flatsample,boundsfree,burnin=100,thin=1,labels=None,truths=None,title=None):
    # tau = sampler.get_autocorr_time()


    # if tau < len(samples)/50:
    #     flat_samples = sampler.get_chain(discard=3*tau, thin=int(tau/2), flat=True)
    # else:
    # samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    corner_ev.corner(flatsample,range=boundsfree,labels=labels,truths=truths,
                     title=title,smooth=0.05)
    return

def StatsEmcee(sampler,burnin=None,thin=1,autocorrquiet=False):
    if burnin is None:
        burnin = int(len(sampler.get_chain())/10)
    samples = sampler.get_chain(discard=burnin, thin=thin)
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    AcceptFrac = sampler.acceptance_fraction.mean()
    # tau = sampler.get_autocorr_time().mean()
    tau = emcee.autocorr.integrated_time(flat_samples,quiet=autocorrquiet).mean()
    quantiles = [0.16,0.50,0.84]
    parquantiles = np.quantile(flat_samples,quantiles,axis=0)
    parmedians = parquantiles[1]
    parsigminus = parquantiles[0]
    parsigplus = parquantiles[2]
    log_prob_flat = sampler.get_log_prob(flat=True,discard=burnin)
    idx = np.argsort(log_prob_flat,axis=0)
    parbest = flat_samples[idx[-1]]
    return [parmedians, parsigminus, parsigplus, parbest, AcceptFrac, tau, flat_samples,
            samples]

def FixProbMembership(p,N,pmin=0.999):
    """Convert probability of membership as follows:
    if p = constant > pmin (e.g. 1) OR p = array of numbers close to 1
        --> None
    """
    p_ones = np.ones(N)
    if p is None:
        p = p_ones
    elif len(np.array(p)) == 1:
        if p == 1:
            p = p_ones
        else:
            raise ValueError("cannot understand p=" + p)
    elif len(np.array(p)) != N:
        raise ValueError("p_mem should have same dimension as data")
    return p
