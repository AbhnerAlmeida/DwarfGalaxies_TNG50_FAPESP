import numpy as np
import os
from scipy.optimize import minimize, differential_evolution, dual_annealing
# from scipy.optimize import minimize_scalar
# from scipy.optimize import fmin_tnc
# from scipy import interpolate
# from scipy import integrate
from . import PROFCL_astroutils as au
from . import PROFCL_langutils as lu
from . import PROFCL_mathutils as mu
from . import PROFCL_densutils as du
from . import PROFCL_constants as cst

def PenaltyFunction(x, boundMin, boundMax, verbosity=0):
    """normalized penalty Fuunction applied to likelihood when parameters goes beyond bounds"""
    if boundMax <= boundMin:
        pf = 0.
    else:
        # rescale to min=0 and max=1
        xRelative01 = (x-boundMin) / (boundMax-boundMin)
        # rescale to min=-1 and max=1
        xRelativemin1plus1 = 2. * np.abs(xRelative01-0.5)
        # absolute value of rescaled
        abs_xRelativemin1plus1 = np.abs(xRelativemin1plus1)
        pf = np.where(abs_xRelativemin1plus1>1,1000*np.sqrt(np.abs(abs_xRelativemin1plus1-1)),0)
        if verbosity >= 3:
            print("PenaltyFunction: x boundMin boundMax = ", x, boundMin, boundMax,\
              " penalty = ", pf)
    return pf

def guess_center(RA, Dec):
    return np.median(RA), np.median(Dec)

def PROFCL_ProbGalaxy(R_ellip_over_a, R_min, R_max, N_tot, scale_radius, 
                      field_surfdensity,
                      ellipticity, R_cut, DeltaCenter, model,
                      min_R_over_rminus2, DeltaRA, DeltaDec,
                      Tiny_Shift_Pos, verbosity):
    """probability of projected radii for given galaxy position, model parameters 
       and field surface density
    arguments:
        RA, Dec: celestial coordinates [deg]
        scale_radius: radius of 3D density slope -2) [deg]
        ellipticity [dimensionless]
        field_surfdensity: uniform field surface density [deg^{-2}]
        RA_cen, Dec_cen: (possibly new) center of model [deg]
        model: 'NFW' or 'coredNFW' or 'Uniform'
    returns probablity density p(data|model)"""  

    # author: Gary Mamon
    
    a = scale_radius    # for clarity
    e = ellipticity     # for clarity
    
    if verbosity >= 2:
        print("PROFCL_ProbGalaxy: R_min = ", R_min, "a=", a, "R_min/a=",R_min/a)
    DeltaNproj_tilde =    du.ProjectedNumber_tilde(R_max/a,e,R_cut/a,model, 
                                                   DeltaRA,DeltaDec,DeltaCenter,
                                                   min_R_over_rminus2,Tiny_Shift_Pos) \
                        - du.ProjectedNumber_tilde(R_min/a,e,R_cut/a,model, 
                                                   DeltaRA,DeltaDec,DeltaCenter,
                                                   min_R_over_rminus2,Tiny_Shift_Pos) 
    if field_surfdensity < cst.TINY:
        numerator = du.SurfaceDensity_tilde(R_ellip_over_a, model)
        denominator = a*a * (1.-e) * DeltaNproj_tilde
    else:
        Nofa = (N_tot - np.pi * (R_max*R_max - R_min*R_min) * field_surfdensity) / DeltaNproj_tilde
        numerator = Nofa/(np.pi * a*a * (1.-e)) * du.SurfaceDensity_tilde(R_ellip_over_a, model) + field_surfdensity
        denominator = N_tot
    if verbosity >= 2:
        print("num = ", numerator, "denom = ", denominator)
    return (numerator / denominator)

def PROFCL_Rescale_Params(params_in_fit, R_min, R_max):
    RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_field_surfdens, log_R_cut = params_in_fit

    # re-sccale
    RA_cen          *= R_max * np.cos(Dec_cen*cst.DEGREE)
    Dec_cen         *= R_max
    ellipticity     *= 5.
    PA              *= 500.
    log_field_surfdens  += 1.   # 10 x field 
    log_R_cut       += 0.7  #  5 x R_cut
    return RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_field_surfdens, log_R_cut
    
def PROFCL_LogLikelihood(params, *args):
    """general -log likelihood of cluster given galaxy positions
    arguments:
        params: (RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_field_surfdens, log_R_cut)
            RA_cen, Dec_cen: coordinates of cluster center [floats]
            log_scale_radius: log of scale radius of the profile (where scale radius is in DEGREEs) [float]
            ellipticity: ellipticity of cluster (1-b/a, so that 0 = circular, and 1 = linear) [float]
            PA: position angle of cluster (DEGREEs from North going East) [float]
            log_field_surfdens: uniform field of cluster (in deg^{-2}) [float]
        aux: (bounds, flags, model, other)
            bounds: ((min,max) for each parameter)
            flags: (field_flag, recenter_flag, ellipticity_flag)
            model: (NFW etc.)
            N_points: number of points for Monte-Carlo integration of elliptical fit
    returns: -log likelihood [float]
    assumptions: not too close to celestial pole 
    (solution for close to celestial pole [not yet implemented]: 
    1. convert to galactic coordinates, 
    2. fit,
    3. convert back to celestial coordinates for PA)"""

    # authors: Gary Mamon with help from Yuba Amoura, Christophe Adami & Eliott Mamon
        
    (data,aux) = args
    (RA_gal,Dec_gal,prob_membership) = data
    (model,method,tolerance,N_points,R_min,R_max,N_tot,rescale_flag,
     RA_cen_init,Dec_cen_init,
     R_sky,Tiny_Shift_Pos,min_R_over_rminus2,max_R_over_rminus2,debug_file,
     iPass,verbosity) = aux
    # rescale_flag = flags
    # global iPass
    # global Delta_x_cen, Delta_y_cen
    # global DeltaCenter, DeltaCenter_over_a
    # global DeltaRA, DeltaDec
    # global RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, field, log_R_cut
    # global scale_radius
    # global PA_in_rd, cosPA, sinPA
    # global R_ellip_over_a, DeltaRA_over_a, DeltaDec_over_a
    # global in_annulus
    # global N_points, N_points_flag

    # if verbosity >= 4:
    #     print("entering LogLikelihood: R_min=",R_min)
    iPass += 1

    # read function arguments (parameters and extra arguments)

    if rescale_flag:
        RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_field_surfdens, \
            log_R_cut =  PROFCL_Rescale_Params(params)
           
    else:
        RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_field_surfdens, \
            log_R_cut = params
    scale_radius = 10 ** log_scale_radius
    R_cut = 10 ** log_R_cut
    if verbosity >= 1:
        print("min(R) max(R) R_min R_max r_s=",np.min(R_sky),np.max(R_sky),
              R_min,R_max,scale_radius)

    # tests
    
    if (len(RA_gal) == 0 or len(Dec_gal) == 0 or len(prob_membership) == 0):
        print("ERROR in PROFCL_lnlikelihood: ",
          "no galaxies found with projected radius between ",
          R_min, " and ", R_max,
          " around RA = ", RA_cen, " and Dec = ", Dec_cen)
        return cst.HUGE

    if R_max/scale_radius > max_R_over_rminus2:
        if verbosity >= 1:
            print("PROFCL_lnlikelihood: R_max a max_Rovera = ", R_max, 
                  scale_radius, max_R_over_rminus2)
        # return a -ln L that increases with off bound on R/r_minus2
        return 1000*np.sqrt(R_max/scale_radius-max_R_over_rminus2)
 
    # check that coordinates are not too close to Celestial Pole
    max_allowed_Dec = 80.
    Dec_abs_max = np.max(np.abs(Dec_gal))
    if Dec_abs_max > max_allowed_Dec:
        raise ValueError("in PROFCL_lnlikelihood: max(abs(Dec)) = "
                    + str(Dec_abs_max) + " too close to pole!")

    ## transform from RA,Dec to cartesian and then to projected radii
    
    cosPA         = mu.cosd(PA)
    sinPA         = mu.sind(PA)

    # a = 10. ** log_scale_radius
    # DeltaRA = RA_cen - RA_cen_init
    # DeltaDec = Dec_cen - Dec_cen_init
    Delta_x_cen,Delta_y_cen = au.dxdy_from_RADec(RA_cen,Dec_cen,
                                                 RA_cen_init,Dec_cen_init)

    
    DeltaCenter = np.sqrt(Delta_x_cen*Delta_x_cen + Delta_y_cen*Delta_y_cen)

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

    # linear variables
    scale_radius = 10.**log_scale_radius            # in deg
    field_surfdens   = 10.**log_field_surfdens      # in deg^{-2}
    if verbosity >= 2:
        print("R_min=",R_min,"R_max=",R_max,"r-2=", scale_radius,
              " Sigma_Field=",field_surfdens)

    # elliptical radii (in DEGREEs)
    R_ellip = au.R_ellip_from_RADec(RA_gal,Dec_gal,ellipticity,
                                    RA_cen,Dec_cen,
                                    sinPA,cosPA)
    R_ellip_over_a = R_ellip / scale_radius
    DeltaRA = RA_cen - RA_cen_init
    DeltaDec = Dec_cen - Dec_cen_init
    # DeltaRA_over_a = (RA_cen - RA_cen_init) / scale_radius
    # DeltaDec_over_a = (Dec_cen - Dec_cen_init) / scale_radius
    
    # check that elliptical radii are within limits of Mathematica fit
    
    if N_points == 0 and (np.any(R_ellip_over_a < min_R_over_rminus2) 
                          or np.any(R_ellip_over_a > max_R_over_rminus2)):
        if verbosity >= 1:
            print("log_likelihood: off limits for r_s minXallow maxallow min(X) max(X)= ", 
                  scale_radius, min_R_over_rminus2, max_R_over_rminus2, 
                  np.min(R_ellip_over_a), np.max(R_ellip_over_a))
        return cst.HUGE
    elif verbosity >= 1:
        print("OK for r_s = ", scale_radius)

    ## likelihood calculation
    
    prob = PROFCL_ProbGalaxy(R_ellip_over_a, R_min, R_max, N_tot, scale_radius, 
                      field_surfdens,
                      ellipticity, R_cut, DeltaCenter, model,
                      min_R_over_rminus2, DeltaRA, DeltaDec,
                      Tiny_Shift_Pos, verbosity)
    
    if verbosity >= 1:
        print("\n\n PROBABILITY = ", prob, "\n\n")
    if np.any(prob<=0):
        if verbosity >= 1:
            print("one p = 0, EXITING LOG_LIKELIHOOD FUNCTION")
        return cst.HUGE

    if verbosity >= 2:
        print("OK")

    # print("prob=",prob)
    lnlikminus = -1*np.sum(prob_membership*np.log(prob))
    if np.isnan(lnlikminus).sum() > 0:
        print("lnlikminus contains NaNs!")
    if verbosity >= 1:
        print("log(theta_s) -ln L = ",log_scale_radius,lnlikminus)
        
    # penalization

    # sumPenalization = 0.
    # for i in range(len(params)):
    #     penalization = PenaltyFunction(params[i],bounds[i,0],bounds[i,1])
    #     sumPenalization += penalization

    if N_points_flag:
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
              "log_field_surfdens = ",   log_field_surfdens
              )
        print("pwd --> ", os.getcwd())
        np.savetxt(debug_file + str(iPass),np.c_[RA_gal,Dec_gal,u,v,R_ellip,prob])
        ftest = open(debug_file + str(iPass),'a')
        ftest.write("{0:8.4f} {1:8.4f} {2:10.3f} {3:5.3f} {4:3.0f} {5:6.2f} {6:10.3f}\n".format(
            RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, 
            log_field_surfdens, lnlikminus))
        ftest.close()
        ftest = open(debug_file,'a')
        ftest.write("{0:8.4f} {1:8.4f} {2:10.3f} {3:5.3f} {4:3.0f} {5:6.2f} {6:10.3f}\n".format(
            RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, 
            log_field_surfdens, lnlikminus))
        ftest.close()

    # return -ln likelihood + penalization
    return lnlikminus # + sumPenalization

def PROFCL_Fit(RA, Dec, p_mem, RAcen=None, Deccen=None, Rmin=None, Rmax=None,
               rs_min=None, rs_max=None, rcut_min=None,rcut_max=None,
               log_surfdens_field_min=None, log_surfdens_field_max=None,
               ellipmin=0,ellipmax=0,PAmin=0,PAmax=180, 
               RAcenMin=None, RAcenMax=None, 
               DeccenMin=None, DeccenMax=None, 
               use_pmem=True,
               model="NFW", method="Powell", tolerance=0.001,
               Npoints=0, rescale_flag=False, Tiny_Shift_Pos=None,
               min_R_over_rminus2=None,max_R_over_rminus2=None,
               debug_file=None, verbosity=0):
    '''Maxmimum Likelihood Estimate of 3D scale radius 
       (where slope of 3D density profile is -2) given galaxy positions on sky
    arguments:
      Data:
        RA, Dec: positions of galaxies [np.array, deg]
        p_mem: membership probabilities [np.array or 1]
        RAcen, Deccen: position of cluster center, default None, None to fit center [deg]
        Rmin, Rmax: min and max considered data semi-major coordinates [min]
      Model parameter bounds:
        rs_min, rs_max: min and max r_scale [arcmin]
        rcut_min, rcut_max: min and max r_cut (tNFW models only) [arcmin]
        log_surfdens_field_min, log_surfdens_field_max: 
            min and max log_10(field surface density) (-99 -99 for no field) [1/arcmin^2]
        ellip_min, ellip_max: min and max ellipticity (default 0 0 for circular fit)
        PAmin PAmax: min and max position angle (East from North) (default 0 180)
        RAcenMin RAcenMax: min and max RA of center [deg]
        DeccenMin DeccenMax: min and max Dec of center [deg]
        use_pmem: flag to consider pmem values  (default: True)
        model: density model
            'NFW': Navarro-Frenck-White 96
            'tNFW': truncated NFW
            'cNFW': cored NFW
            'Uniform': uniform (for tests)
            or coredNFW or Uniform) [char]
        method: minimization method (all with bounds)
            'Powell':       Powell
            'TNC': truncated Newton Conjugate Gradient
            'Nelder-Mead':  Simplex 
            'L-BFGS-B':     Broyden-Fletcher-Goldfarb-Shanno, using gradient
            'SLSQP':        Sequential Least-Squares Programming
            'Brent':        Brent method (only for univariate data)
            'DiffEvol':     Differential Evolution
        tolerance: relative allowed tolerance
        N_points: number of points for elliptical solutions:
                0: 2D polynomial approximation
                >0: number of points for Monte Carlo integration
                <0: log_10(tolerance) for 2D integration
        rescale_flag:   rescale-data [bool]
        Tiny_Shift_Pos: maximum position shift to keep circular fit [degrees]
        min_R_over_rminus2: minimum allowed R/r_scale
        max_R_over_rminus2: maximum allowed R/r_scale
        debug_file
        verbosity: > 0 for debugging output (default 0)
        )
    '''

    # authors: Gary Mamon & Yuba Amoura, with help from Eliott Mamon
    
    global iPass

    ### basic checks
    
    ## check that initial center is provided
    ## (will be removed when fits for center are ready)
    if RAcen is None:
        raise ValueError("RAcen must be set")
    if Deccen is None:
        raise ValueError("Dec be set")    
        
    ## checks that parameter bounds are set
    
    if len(Dec) != len(RA):
        raise ValueError("len(RA) = " + len(RA) + " does not match len(Dec) = " + len(Dec))

    # convert p_mem = 1 to all p_mem = 1
    if np.isscalar(p_mem):
        if p_mem == 1:
            p_mem = 0*RA + 1
        else:
            raise ValueError("Cannot understand p_mem=" + str(p_mem))

    if len(p_mem) != len(RA):
        raise ValueError("p_mem has different length (" + str(len(p_mem)) + ") than RA (" + str(len(RA)) + ")")
    
    # projected radii (in degrees)
    # start with centroid if center not provided
    # if RAcen is None or Deccen is None:
        
    R_sky = au.AngularSeparation(RA,Dec,RAcen,Deccen,choice='trig')
    N_tot = len(R_sky)

    # check on argument values
                            
    # min and max projected radii in arcmin
    if Rmin is None:
        Rmin = 0.5 * 60 * np.min(R_sky)
    if Rmax is None:
        Rmax = 1.1 * 60 * np.max(R_sky)

    # check that Rmin > 0 (to avoid infinite surface densities)
    if Rmin <= 0:
        raise ValueError("in PROFCL_fit: R_min = " + str(Rmin) + "must be > 0")
                             
    # check that Rmax > Rmin
    if Rmax < Rmin:
        raise ValueError("in PROFCL_fit: Rmax = " + str(Rmax) +
                         " must be > Rmin = " + str(Rmin))
                             
    # # check model
    # models = ["NFW", "cNFW", "tNFW", "Uniform"]
    # methods = ["Powell","Nelder-Mead","BFGS","L-BFGS-B","SLSQP",
    #            "Newton-CG","Brent","DiffEvol", "DualAnneal"]


    # if model not in models:
    #     raise ValueError("model " + model + " not understood; must be in " + models)
    # if method not in methods :
    #     raise ValueError("method " + method + " not understood; must be in " + methods)

    # If near RA=0 shift by RA by 180 deg
    
    if (np.min(RA) < 1) or (np.max(RA) > 359):
        RAshift = True
        RA = RA + 180
    else:
        RAshift = False

    maxabsDec = np.max(np.abs(Dec))
    if maxabsDec > 88:
        print("Galaxies closer to pole than",'%4.2f' % (90-maxabsDec))
        return -99
    
    ### end of checks

    # convert model and method
    model = cst.Convert(model,cst.MODEL_DICT)
    method = cst.Convert(method,cst.METHOD_DICT)
    
    # select data by projected distance bounds
    cond = np.logical_and(R_sky >= Rmin, R_sky <= Rmax)
    if verbosity >= 2:
        print("Fit: N_tot = ", len(R_sky[cond]))
        # print("RA Dec=",RA,Dec)
    data = [RA,Dec,p_mem]
    dataTranspose = np.transpose(data)
    dataTrGood = dataTranspose[cond]
    if verbosity >= 1:
        print("len(data) len(dataTranspose) len(dataTrGood) = ",
              len(data), len(dataTranspose), len(dataTrGood))
    data_inbounds = np.transpose(dataTrGood)
    R_sky2 = R_sky[cond]
    if verbosity >= 1:
        print("min(R_sky2) = ", np.min(R_sky2))
    
    if verbosity >= 4:
        print("entering Fit: Rmin=",Rmin)
        
    maxfev = 500
    iPass = 0
    
    # initial guesses at middle of bounds
    if RAcen is None:
        RAcen = np.mean([RAcenMin,RAcenMax])
    if Deccen is None:
        Deccen = np.mean([DeccenMin,DeccenMax])
    log_scale_radius_min = np.log10(rs_min)
    log_scale_radius_max = np.log10(rs_max)
    log_scale_radius = np.mean([log_scale_radius_min,log_scale_radius_max])
    ellipticity = np.mean([ellipmin,ellipmax])
    PA = np.mean([PAmin,PAmax])
    log_surfdens_field = np.mean([log_surfdens_field_min,log_surfdens_field_max])
    log_R_cut_min = np.log10(rcut_min)
    log_R_cut_max = np.log10(rcut_max)
    log_R_cut = np.mean([log_R_cut_min,log_R_cut_max])
    
    # arguments to minimization function
    params = (RAcen, Deccen, log_scale_radius, ellipticity, PA, log_surfdens_field, 
              log_R_cut)
    bounds = ((RAcenMin,RAcenMax), 
              (DeccenMin,DeccenMax), 
              (log_scale_radius_min,log_scale_radius_max),
              (ellipmin,ellipmax), 
              (PAmin,PAmax), 
              (log_surfdens_field_min,log_surfdens_field_max),
              (log_R_cut_min,log_R_cut_max))
    aux = (model,method,tolerance,Npoints,Rmin,Rmax,N_tot,rescale_flag,
     RAcen,Deccen,
     R_sky2,Tiny_Shift_Pos,min_R_over_rminus2,max_R_over_rminus2,debug_file,
     iPass,verbosity)
    myargs = (data_inbounds,aux)
    
    # minimization
    if method == "Nelder-Mead":
        res = minimize (PROFCL_LogLikelihood, params, args=myargs, 
                            method=method, tol=tolerance, bounds=bounds,
                            options={"maxfev":maxfev})
    elif method == "Powell":
        res = minimize  (PROFCL_LogLikelihood, params, args=myargs, 
                            method=method, tol=tolerance, bounds=bounds,
                            options={"ftol":tolerance, "maxfev":maxfev})
    elif method == "L-BFGS-B":
        res = minimize  (PROFCL_LogLikelihood, params, args=myargs, 
                            method=method, tol=tolerance, bounds=bounds, 
                            options={"ftol":tolerance, "maxfun":maxfev})
    elif method == "TNC":
        res = minimize  (PROFCL_LogLikelihood, params, args=myargs, 
                            method=method, jac=None, bounds=bounds, tol=None, 
                            options={"xtol":tolerance, "maxiter":maxfev})
    
    elif method == "SLSQP":
        res = minimize  (PROFCL_LogLikelihood, params, args=myargs, 
                            method=method, jac=None, bounds=bounds, tol=None, 
                            options={"ftol":tolerance, "maxiter":maxfev})

    # elif method == "fmin_TNC":
    #     return fmin_tnc              (PROFCL_LogLikelihood, params, fprime=None, args=myargs,  approx_grad=True, bounds=bounds, epsilon=1.e-8, ftol=tolerance)
    elif method == "DiffEvol":
        res = differential_evolution(PROFCL_LogLikelihood, bounds, args=myargs, 
                                      atol=tolerance)
    
    elif method == "DualAnneal":
        res = dual_annealing(PROFCL_LogLikelihood, bounds, args=myargs, 
                              maxiter=500)
    else:
        raise ValueError("in PROFCL_fit: method = " + method + " is not yet implemented")
        
    if RAshift:
        res.x[0] = res.x[0] - 180
    
    return res.x

def PROFCL_Fit_old(params, bounds, data, aux):
    '''Maxmimum Likelihood Estimate of 3D scale radius 
       (where slope of 3D density profile is -2) given galaxy positions on sky
    arguments:
        params = (RA_cen, Dec_cen, log_scale_radius, ellipticity, PA, log_field_surfdens, R_cut), where
                RA_cen, Dec_cen: coordinates of cluster center (deg) [floats]
                log_scale_radius: log of scale radius of the profile (where scale radius is in degrees) [float]
                ellipticity: ellipticity of cluster (1-b/a, so that 0 = circular, and 1 = linear) [float]
                PA: position angle of cluster (degrees from North going East) [float]
                log_field_surfdens: uniform field surface density of cluster (in arcmin^{-2}) [float]
                R_cut: sharp truncation of density profile (degrees, 0 for none) [float]
        bounds = ((min,max),...) for each parameter
        data = (RA_gal, Dec_gal, prob_membership)
        aux = (model, method, tolerance, N_points, R_min, R_max, N_tot, rescale_flag, 
                RA_cen_init, Dec_cen_init, R_sky, Tiny_Shift_Pos,min_R_over_rminus2,max_R_over_rminus2,
                debug_file, iPass, verbosity)
                model: density model (NFW or coredNFW or Uniform) [char]
                method:    minimization method [string]
                        'Powell':       Powell (with bounds)
                        'Nelder-Mead':  Simplex 
                        'BFGS':         Broyden-Fletcher-Goldfarb-Shanno, using gradient
                        'L-BFGS-B':     Broyden-Fletcher-Goldfarb-Shanno, using gradient
                        'SLSQP':        Sequential Least-Squares Programming
                        'Newton-CG':    Newton's conjugate gradient
                        'Brent':        Brent method (only for univariate data)
                        'diff-evol':    Differential Evolution
                tolerance: relative allowed tolerance [float[]]
                N_points: number of points for elliptical solutions [int]:
                        0: 2D polynomial approximation
                        >0: number of points for Monte Carlo integration
                        <0: log_10(tolerance) for 2D integration
                R_min:     minimum allowed projected radius (degrees) [float]
                R_max:     maximum allowed projected radius (degrees) [float]
                R_sky:          radial positions on sky (to save CPU time) [degrees, default=None]
                N_tot:     length of data (input not used for now)
                rescale_flag:   rescale-data [bool]
                RA_cen_init:    initial RA_cen (degrees)
                Dec_cen_init:   initial Dec_cen (degrees)
                Tiny_Shift_Pos: maximum position shift to keep circular fit [degrees]
                min_R_over_rminus2: minimum allowed R/r_minus2
                max_R_over_rminus2: maximum allowed R/r_minus2
                debug_file
                iPass: set to 0
                verbosity: > 0 for debugging output
                (field_flag: flag for field fit (True for including field in fit, False for no field)
                recenter_flag: flag for recentering (True for recentering, False for keeping DETCL center)
                elliptical_flag: flag for elliptical fit (True for elliptical fit, False for circular fit)
                )
    '''

    # authors: Gary Mamon & Yuba Amoura, with help from Eliott Mamon
    
    global iPass

    models = ["NFW", "cNFW", "tNFW", "Uniform"]
    # data
    RA,Dec,prob_membership = data
    
    # auxiliary fit parameters
    (model,method,tolerance,N_points,R_min,R_max,R_sky,N_tot,rescale_flag,
     RA_cen_init,Dec_cen_init,
     Tiny_Shift_Pos,min_R_over_rminus2,max_R_over_rminus2,debug_file,
     iPass,verbosity) = aux

    # projected radii (in degrees)
    if R_sky is None:
        R_sky = au.AngularSeparation(RA,Dec,RA_cen_init,Dec_cen_init,choice='trig')
    if N_tot is None:
        N_tot = len(R_sky)
    
    # force data to data limits
    cond = np.logical_and(R_sky >= R_min, R_sky <= R_max)
    if verbosity >= 2:
        print("Fit: N_tot = ", len(R_sky[cond]))
        # print("RA Dec=",RA,Dec)
    dataTranspose = np.transpose(data)
    dataTrGood = dataTranspose[cond]
    if verbosity >= 1:
        print("len(data) len(dataTranspose) len(dataTrGood) = ",
              len(data), len(dataTranspose), len(dataTrGood))
    # data2 = np.zeros((3,len(data[0])))
    # # R_sky2 = np.zeros(R_sky)
    # print("shape(data) = ",data.shape)
    # for i, d in enumerate(data):
    #     data2[i] = d[cond]
    data2 = np.transpose(dataTrGood)
    R_sky2 = R_sky[cond]
    if verbosity >= 1:
        print("min(R_sky2) = ", np.min(R_sky2))
    aux2 = (model,method,tolerance,N_points,R_min,R_max,N_tot,rescale_flag,
     RA_cen_init,Dec_cen_init,
     R_sky2,Tiny_Shift_Pos,min_R_over_rminus2,max_R_over_rminus2,debug_file,
     iPass,verbosity)
    
    if verbosity >= 4:
        print("entering Fit: R_min=",R_min)

    ## checks on types of arguments

    # check that input positions are in numpy arrays
    if not isinstance(RA,np.ndarray):
        raise ValueError("in PROFCL_fit: RA must be numpy array")

    if not isinstance(Dec,np.ndarray):
        raise ValueError("in PROFCL_fit: Dec must be numpy array")

    # check that min and max projected radii are floats or ints
    lu.CheckTypeIntorFloat(R_min,"PROFCL_fit","R_min")
    lu.CheckTypeIntorFloat(R_max,"PROFCL_fit","R_max")

    # check that model is a string
    t = type(model)
    if t is not str:
        raise ValueError("in PROFCL_fit: model is ", 
                         t, " ... it must be a str")
                             
    # # check that flags are boolean
    # lu.CheckTypeBool(field_flag,"PROFCL_fit","field_flag")
    # lu.CheckTypeBool(recenter_flag,"PROFCL_fit","recenter_flag")
    # lu.CheckTypeBool(ellipticity_flag,"PROFCL_fit","ellipticity_flag")
    
    # check that method is a string
    t = type(method)
    if t is not str:
        raise ValueError("in PROFCL_fit: method is ", 
                         t, " ... it must be a str")
                             
    ## checks on values of arguments
    
    # check that R_min > 0 (to avoid infinite surface densities)
    if R_min <= 0:
        raise ValueError("in PROFCL_fit: R_min = " + str(R_min) + "must be > 0")
                             
    # check that R_max > R_min
    if R_max < R_min:
        raise ValueError("in PROFCL_fit: R_max = " + str(R_max) +
                         " must be > R_min = " + str(R_min))
                             
    # check model
    # if model != "NFW" and model != "coredNFW" and  model != "Uniform":
    if model not in models:
        raise ValueError("in PROFCL_fit: model = " +
                         model + " not recognized... must be NFW or coredNFW or Uniform")
    
    # function of one variable
    if np.isnan(RA_cen_init):
        raise ValueError("in PROFCL_fit: RA_cen_init is NaN!")
    if np.isnan(Dec_cen_init):
        raise ValueError("in PROFCL_fit: Dec_cen_init is NaN!")

    # params = np.array([
    #                    RA_cen,Dec_cen,log_scale_radius,
    #                    ellipticity,PA,log_field_surfdens
    #                   ]
    #                  )

    # force radii in bounds
        
    maxfev = 500
    iPass = 0
    myargs = (data2,aux2)
    
    # minimization
    # if method == 'brent' or method == 'Brent':
    #     if recenter_flag:
    #         raise ValueError('ERROR in PROFCL_fit: brent minimization method cannot be used for fits with re-centering')
    #     if ellipticity_flag:
    #         raise ValueError('ERROR in PROFCL_fit: brent minimization method cannot be used for elliptical fits')
        
    # else:

    if method == "Nelder-Mead":
        # return minimize(PROFCL_LogLikelihood, params, args=(RA, Dec, prob_membership, R_min, R_max, model), 
        #                 method=method, tol=tolerance, bounds=bounds, options={'fatol':tol, 'maxfev':maxfev})
        return minimize              (PROFCL_LogLikelihood, params, args=myargs, 
                                      method=method, tol=tolerance,
                                      options={"maxfev":maxfev})
    elif method == "Powell":
        return minimize              (PROFCL_LogLikelihood, params, args=myargs, 
                                      method=method, tol=tolerance, bounds=bounds,
                                      options={"ftol":tolerance, "maxfev":maxfev})
    # elif method == "Powell":
    #     return minimize              (PROFCL_LogLikelihood, params, args=myargs, 
    #                     method=method, tol=tolerance, bounds=bounds, options={"ftol":tolerance, "maxfev":maxfev})
    # elif method == "CG" or method == "BFGS":
    #     return minimize              (PROFCL_LogLikelihood, params, args=myargs, 
    #                     method=method, tol=tolerance, bounds=bounds, options={"gtol":tolerance, "maxiter":maxfev})
    # elif method == "Newton-CG":
    #     return minimize              (PROFCL_LogLikelihood, params, args=myargs, 
    #                     method=method, tol=tolerance, bounds=bounds, options={"xtol":tolerance, "maxiter":maxfev})
    elif method == "L-BFGS-B":
        return minimize              (PROFCL_LogLikelihood, params, args=myargs, 
                        method=method, tol=tolerance, bounds=bounds, 
                        options={"ftol":tolerance, "maxfun":maxfev})
    elif method == "SLSQP":
        return minimize              (PROFCL_LogLikelihood, params, args=myargs, 
                        method=method, jac=None, bounds=bounds, tol=None, 
                        options={"ftol":tolerance, "maxiter":maxfev})
    elif method == "TNC":
        return minimize              (PROFCL_LogLikelihood, params, args=myargs, 
                        method=method, jac=None, bounds=bounds, tol=None, 
                        options={"xtol":tolerance, "maxiter":maxfev})
    # elif method == "fmin_TNC":
    #     return fmin_tnc              (PROFCL_LogLikelihood, params, fprime=None, args=myargs,  approx_grad=True, bounds=bounds, epsilon=1.e-8, ftol=tolerance)
    elif method == "Diff-Evol":
        return differential_evolution(PROFCL_LogLikelihood, bounds, args=myargs, atol=tolerance)
    
    elif method == "Dual-Anneal":
        return dual_annealing(PROFCL_LogLikelihood, bounds, args=myargs, maxiter=500)
    else:
        raise ValueError("in PROFCL_fit: method = " + method + " is not yet implemented")



