
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 2022

@author: enterprise
"""
###########################################################################
########################################################################### 

import numpy as np
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from scipy.stats import ranksums
from scipy.stats import median_test
from scipy.stats import kstest
import ExtractTNG

from sklearn.metrics import mean_squared_error, r2_score

##########################################################################
########################################################################### 


def bootstrap(x):
    values = np.random.choice(x,replace=True,size=len(x))
    return values


def boostrap_func(data,func=np.nanmedian,num_boots=1000):
    """bootstrap measure of standard deviation of function of 1D data set
    arguments:
        x: data (1d numpy array)
        func: function (default np.median)
        num_trials: number of bootstrap samples (defult 1000)
    returns: standard deviation of function
    requirements: numpy, bootstrap [in this file]
    author: Gary Mamon (gam AAT iap.fr)"""

    val = np.zeros(num_boots)
    # loop over bootstrap samples
    
    data = data[~np.isinf(data)]

    if len(data) > 1:
        for n in range(num_boots):
            # function for the given bootstrap sample
            values = bootstrap(data)
            val[n] = func(values)
        return np.std(val)
    return np.nan
    
    
# def jackknife(data, func=np.median):
#     n = len(data)
#     t = np.zeros(n)
#     inds = np.arange(n)

#     ## 'jackknifing' by leaving out an observation for  each i                                                                                                                      
#     for i in range(n):
#         t[i] = func(np.delete(data,i))

#     return func(t)


def jackknife(data, func=np.median):
    data = np.asarray(data)
    n = data.size
    if n < 2:
        raise ValueError("Precisa de n >= 2 para jackknife leave-one-out.")

    t = np.empty(n, dtype=float)
    for i in range(n):
        sample = np.concatenate((data[:i], data[i+1:]))
        t[i] = func(sample)

    return func(t)
 

def TestPermutation(population_1, population_2, num_permutations = 50000, roundmedian = 3, KStest = False, RankSums = False, Moodtest = False):
    # Define the observed test statistic (e.g., mean difference)
    print('Medians: ', round(np.nanmedian(population_1), roundmedian) , round(np.nanmedian(population_2), roundmedian))

    observed_statistic = np.nanmedian(population_1) - np.nanmedian(population_2)
    if observed_statistic < 0:
        population_2_old = population_2
        population_2 = population_1
        population_1 = population_2_old
        observed_statistic = np.nanmedian(population_1) - np.nanmedian(population_2)


    if RankSums:
        print('RankSums')
        _, p_value = ranksums(population_1, population_2)
    
    elif KStest:
        print('kstest')
        res =  kstest(population_1, population_2)
        p_value = res.pvalue
    elif Moodtest:
        print('Mood')
        res = median_test(population_1, population_2)
        p_value = res.pvalue
    else:
        # Initialize an array to store permuted statistics
        permuted_statistics = np.zeros(num_permutations)
    
        # Combine the two populations
        combined_population = np.concatenate((population_1, population_2))
    
        # Perform the permutation test
        for i in range(num_permutations):
            # Shuffle the data randomly
            np.random.shuffle(combined_population)
    
            # Calculate the test statistic for the permuted data
            permuted_statistic = np.nanmedian(combined_population[:len(population_1)]) - np.nanmedian(combined_population[len(population_1):])
    
            # Store the permuted statistic
            permuted_statistics[i] = permuted_statistic
    
        # Calculate the p-value
        p_value = np.sum(np.abs(permuted_statistics) >= np.abs(observed_statistic)) / num_permutations #np.sum(permuted_statistics >= observed_statistic) / num_permutations

    # Output the results

    print("Observed Test Statistic:", round(observed_statistic, 5))
    print("Permutation Test P-Value:", round(p_value, 5))
    
    return observed_statistic, p_value

def median(data, Nboots = 1000):

    snaps = [ str(int(i)) for i in np.arange(data.shape[1]-1)]

    ysigma = np.array([])
    y = np.array([])


    for i, snap in enumerate(snaps):
        
        array = np.array([])
        
        for value in data[snap].values:
            if not np.isnan(value):
                array = np.append(array,value)

        if len(array) >= 5:
            # create an empty list to store the median of each bootstrap sample
            medians = []
    
            n_size = int(len(array) * 0.50) 
            for i in range(Nboots):
                bootstrap_sample = resample(array, n_samples = n_size)
                medians.append(np.median(bootstrap_sample))
            
            # calculate the bootstrap error
            original_median = np.median(array)
            bootstrap_median = np.median(medians)
            if np.isinf(bootstrap_median):
                bootstrap_error = np.median(ysigma)
            else:
                bootstrap_error = bootstrap_median - original_median

            
            ysigma = np.append(ysigma, bootstrap_error)
            y = np.append(y , original_median)
        else:
            ysigma = np.append(ysigma,np.nan)
            y = np.append(y ,np.nan)

    return y, ysigma

def weighted_median(values, weights):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, 0.5 * c[-1])]]

def split_quantiles(x, y, total_bins = 20, quantile = 0.95):

    '''
    bins = np.linspace(min_value, max_value, num_bins+1)
    
    x = np.sort(x)
    bins = np.linspace(x.min() , x.max() + 0.001 * x.max() ,  total_bins)
    print(bins, x.max() )
    delta = bins[1]-bins[0]
    idx  = np.digitize(x,bins)
    '''
    
    x = np.array([value for value in x])
    y = np.array([value for value in y])

    #Remove NaN and inf
    argNaN = np.argwhere(np.isnan(x)).transpose()[0]

    y = np.delete(y, argNaN)
    x = np.delete(x, argNaN)
    
    argInf = np.argwhere(np.isinf(x)).transpose()[0]
    y = np.delete(y, argInf)
    x = np.delete(x, argInf)

    xargs = np.argsort(x)
    x = x[xargs]
    y = y[xargs]

    if min(x) !=0 and min(x)*max(x) > 0:
        xmean = np.geomspace(min(x), max(x), total_bins + 1)
    else:
        xmean = np.linspace(min(x), max(x), total_bins + 1)

    yquantile95 = np.array([])
    yquantile5 = np.array([])
    ymedian = np.array([])
    xmeanfinal = np.array([])


    for k, value in enumerate(xmean):
        
        if value == xmean[-2]:
            array = y[np.where(x >= xmean[k - 1])]
        elif k == len(xmean) - 1:
            continue
        else:
            array = y[np.where((x >= value) & (x <xmean[k + 1]))]
        if len(array) < 5:
            continue
        array = array[~np.isnan(array)]
        ymedian = np.append(ymedian, np.median(array))
        yquantile95 = np.append(yquantile95, np.quantile(array, quantile))
        yquantile5 = np.append(yquantile5, np.quantile(array, 1-quantile))
        xmeanfinal = np.append(xmeanfinal, value)

    array = y[np.where(x >= xmean[-2])]
    if len(array) >= 5:
        ymedian = np.append(ymedian, np.median(array))
        yquantile95 = np.append(yquantile95, np.quantile(array, quantile))
        yquantile5 = np.append(yquantile5, np.quantile(array, 1-quantile))
        xmeanfinal = np.append(xmeanfinal, max(x))
        

    return xmeanfinal, ymedian, yquantile95, yquantile5

##########################################################################
########################################################################### 

def quant(data):

    snaps = [ str(int(i)) for i in np.linspace(99, 0, 100)]

    quant_16 = np.array([])
    quant_84 = np.array([])
    y = np.array([])

    for i, snap in enumerate(snaps):
        
        array = np.array([])
        for value in data[snap].values:
            if not np.isnan(value):
                array = np.append(array,value)

            
        if len(array) >= 5:
            quant16 =  np.quantile(array, 0.16)
            quant84 =  np.quantile(array, 0.84) 
            med = np.median(array)
            
            '''
            if sigma > med and med >= 0:
                ysigma = np.append(ysigma,np.nan)
                y = np.append(y ,np.nan)
            else:
            '''
            quant_16 = np.append(quant_16, quant16)
            quant_84 = np.append(quant_84, quant84)
            y = np.append(y , med)
            
        else:
            quant_16 = np.append(quant_16, np.nan)
            quant_84 = np.append(quant_84, np.nan)
            y = np.append(y , np.nan)

    return y, quant_16, quant_84

def randomList(nmin, nmax, N, seed = 123):
    np.random.seed(seed)
    values = []
    # traversing the loop 15 times
    while len(values) < N:
        for i in range(N):
           if len(values) == N - 1:
               return values
           # generating a random number in the range 1 to 100
           r= np.random.randint(nmin,nmax)
           # checking whether the generated random number is not in the
           # randomList
           if r not in values:
              # appending the random number to the resultant list, if the condition is true
              values.append(r)
    return values


##########################################################################
########################################################################### 

def fit_func(X, a, b, c, d):
    return a + b * X + c * X**2 + d * X**3

def fit_func_inc(X, asigma, bsigma, csigma, dsigma):
    return np.sqrt( (asigma)**2. + (bsigma * X)**2. + (csigma * X**2)**2. + (dsigma * X**3)**2. )


def fit_linear_function(X, Y, fz):
    
    
    for value in Y:
        if np.isinf(value):
            print('Inf')
        elif np.isnan(value):
            print('NaN, Y')
            
    popt, pcov = curve_fit(lambda X, a, b, c, d : fit_func(X, a, b, c, d), X, Y)
    a, b, c = popt
    a_uncertainty, b_uncertainty, c_uncertainty = np.sqrt(np.diag(pcov))
    
    coefficients = np.array([ a, b, c])
    uncertainties = np.array([ a_uncertainty, b_uncertainty, c_uncertainty])
    for i, value in enumerate(uncertainties):
        if np.isinf(value):
            uncertainties[i] = np.nan
    
    return  coefficients, uncertainties


def CartesiantToRotated(pos,vel,angmom):
    """Cartesian components in frame where 
       the 3rd axis is aligned with the angular momentum vector 
       
    arguments:
        pos: positions ([N,3] array)
        vel: velocities ([N,3] array)
        angmom: angular momentum ([N,3] array)
        
    source:
        Gary Mamon and Abhner Pinto de Almeida
        """
    # modulus of position
    r = np.sqrt(np.sum(pos*pos))
    
    # modulus of angular momentum
    J = np.sqrt(np.sum(angmom*angmom))
    
    # dot product r . J
    r_dot_J = np.dot(pos,angmom)
    
    # trigonometry to obtain matrix
    costheta = np.clip(angmom[2] / J,-1,1)
    theta = np.arccos(costheta)
    
    sintheta = np.clip(np.sin(theta),-1,1)
    
    sinphi = angmom[1] / (J*sintheta)
    cosphi = angmom[0] / (J*sintheta)
    
    matrix = np.array([[sinphi, -1.*cosphi, 0],
		       [cosphi * costheta, sinphi * costheta, - sintheta],
		       [sintheta * cosphi , sintheta * sinphi, costheta]])
    
    # new positions and velocities
    print(matrix, pos)
    POS = np.dot(matrix, pos)
    VEL = np.dot(matrix, vel)

    return POS, VEL


###########################################################################
###########################################################################

def Rotate(file, Rad, z = 0, TNG = False):
    """Rotate galaxy hdf5 file
       
    arguments:
        file: galaxy TNG file (hdf5)
        Rad: galaxy Rad ([N,3] array)
        
    source:
        Abhner Pinto de Almeida
    """

    #Positions, velocities, masses, cen and bulk velocity for each component
    
    pos = []
    mass = []
    vel = []
    Cen = []
    VelBulk = []
    
    factor = 1. / (1 + z)
    scalefactorsqrt = np.sqrt(1. / (1 + z))
    h = 0.6778
    
    for TypeNum in range(6):
        try:
            pos.append(file['PartType'+str(int(TypeNum))]['Coordinates'][:] * factor / h)
            vel.append(file['PartType'+str(int(TypeNum))]['Velocities'][:] * scalefactorsqrt)
            
            if TypeNum == 1:
                mass.append(file['Header'].attrs['MassTable'][1]*np.ones(len(file['PartType'+str(int(TypeNum))]['Coordinates'])) * 1e10 / h )
                Cen.append( np.sum(pos[TypeNum][:] * mass[TypeNum][:, np.newaxis], axis=0) / np.sum(mass[TypeNum][:]))
                VelBulk.append( np.sum(vel[TypeNum][:] * mass[TypeNum][:, np.newaxis], axis=0) / np.sum(mass[TypeNum][:]))

            else:
                mass.append(file['PartType'+str(int(TypeNum))]['Masses'][:] * 1e10 / h)
                Cen.append( np.sum(pos[TypeNum][:] * mass[TypeNum][:, np.newaxis], axis=0) / np.sum(mass[TypeNum][:]))
                VelBulk.append( np.sum(vel[TypeNum][:] * mass[TypeNum][:, np.newaxis], axis=0) / np.sum(mass[TypeNum][:]))

                
                
        except:
            pos.append(np.array([0, 0, 0]))
            vel.append(np.array([0, 0, 0]))
            Cen.append(np.array([0, 0, 0]))
            VelBulk.append(np.array([0, 0, 0]))
            mass.append(np.array([0]))
            
    #Center and galay velocity
    CenGalaxy = np.sum(np.array([Cen[i] * np.sum(mass[i][:]) for i in range(6)]), axis = 0) / np.sum(np.array([np.sum(mass[i][:]) for i in range(6)]))
    VelGalaxy = np.sum(np.array([VelBulk[i] * np.sum(mass[i][:]) for i in range(6)]), axis = 0) / np.sum(np.array([np.sum(mass[i][:]) for i in range(6)]))
    
    if TNG:
        for TypeNum in range(6):
            pos[TypeNum] = ExtractTNG.FixPeriodic(pos[TypeNum] - CenGalaxy)
            vel[TypeNum] = ExtractTNG.FixPeriodic(vel[TypeNum] - VelGalaxy)


    for TypeNum in range(6): 

        if not 'PartType'+str(TypeNum) in file.keys():

            continue
           
        #Align the galaxy in terms of the gaseous component. 
        #If the galaxy has no gas then stellar component and, 
        #if the galaxy don't have stars so dm component
        
        if 'PartType0' in file.keys():
            RadMass = np.linalg.norm(pos[0], axis = 1)
            MassInRad = mass[0][RadMass < Rad]
            ang_mom = np.sum(np.cross(pos[0][RadMass < Rad], MassInRad[:, np.newaxis] * vel[0][RadMass < Rad]), axis=0)
        elif 'PartType4' in file.keys():
            RadMass = np.linalg.norm(pos[4], axis = 1)
            MassInRad = mass[4][RadMass < Rad]
            ang_mom = np.sum(np.cross(pos[4][RadMass < Rad], MassInRad[:, np.newaxis] * vel[4][RadMass < Rad]), axis=0)
        else:
            RadMass = np.linalg.norm(pos[1], axis = 1)
            MassInRad = mass[1][RadMass < Rad]
            ang_mom = np.sum(np.cross(pos[1][RadMass < Rad], MassInRad[:, np.newaxis] * vel[1][RadMass < Rad]), axis=0)
        
        
        POS, VEL = CartesiantToRotated(pos[TypeNum],vel[TypeNum],ang_mom)

        file['PartType'+str(TypeNum)]['Coordinates'][:,0] = POS[:, 0] 
        file['PartType'+str(TypeNum)]['Coordinates'][:,1] = POS[:, 1]
        file['PartType'+str(TypeNum)]['Coordinates'][:,2] = POS[:, 2]
        
        
        file['PartType'+str(TypeNum)]['Velocities'][:,0] = VEL[:, 0]
        file['PartType'+str(TypeNum)]['Velocities'][:,1] = VEL[:, 1] 
        file['PartType'+str(TypeNum)]['Velocities'][:,2] = VEL[:, 2] 


    return file


###########################################################################
###########################################################################

def CartesiantToRotated(pos,vel,angmom):
    """Cartesian components in frame where 
       the 3rd axis is aligned with the angular momentum vector 
       
    arguments:
        pos: positions ([N,3] array)
        vel: velocities ([N,3] array)
        angmom: angular momentum ([N,3] array)
        
    source:
        Abhner Pinto de Almeida
    """

    # modulus of angular momentum
    J = np.sqrt(np.sum(angmom*angmom))
    

    # trigonometry to obtain matrix
    costheta = np.clip(angmom[2] / J,-1,1)
    theta = np.arccos(costheta)
    
    sintheta = np.clip(np.sin(theta),-1,1)
    
    sinphi = angmom[1] / (J*sintheta)
    cosphi = angmom[0] / (J*sintheta)
    
     
    matrix = np.array([[costheta*cosphi , costheta*sinphi, -sintheta],	
		       [-sinphi, cosphi, 0],
		       [sintheta*cosphi  , sintheta*sinphi, costheta]])
    

    # new positions and velocities
    POS = np.dot(matrix, np.transpose(pos))
    VEL = np.dot(matrix, np.transpose(vel))
    
    POS = np.around(POS, decimals=10)
    VEL = np.around(VEL, decimals=10)

    return np.transpose(POS), np.transpose(VEL)


##########################################################################
########################################################################### 
#MORTO

##########################################################################
########################################################################### 
def linearFit(veclist,value):
    """linear fit
    
    Arguments:
        veclist: list of vectors
        value: data
    Returns:
        coefficients, uncertainties-on-coefficients

    Author: Gary Mamon (gam AAT iap.fr)
    Source:
        translated from SM (SuperMongo) source code for `linfit'
        
    Example:
        fit Z = a*X + b*X^2 + c*sin(Y) + d
        Coeffs, errCoeffs = linearFit(np.array[X,X*X,np.sin(Y),0*X+1])
    """
    N = len(veclist)
    a = np.zeros(N)
    vara = np.zeros(N)
    for i in range(N):
        vara[i] = np.sum(veclist[i,:]*value)
    M = eqnorm(veclist)
    Minv = np.linalg.inv(M)
    for i in range(N):
        a[i] = np.sum(Minv[i,:]*vara)
    D1 = value
    for i in range(N):
        D1 = D1 - a[i]*veclist[i,:]
    D2 = np.sum(D1*D1) / len(D1)
    for i in range(N):
        vara[i] = D2 * Minv[i,i]
    return a, np.sqrt(vara)

def eqnorm(veclist):
    """kernal for linearFit"""
    N = len(veclist)
    M = np.zeros((len(veclist),len(veclist)))
    for i in range(N):
        for j in range(N):
            M[i,j] = np.sum(veclist[i,:]*veclist[j,:])
    return M



##########################################################################
########################################################################### 

def fit_function(X, Y, z):
    """
    Fits the generic linear function Y = a*z + b X + c X^2
    to a set of data using linear regression.

    Parameters:
        X (array): 1-D array of X values
        Y (array): 1-D array of Y values
        z (float): A float number

    Returns:
        coefficients (array): Array of coefficients [a, b, c]
        uncertainties (array): Array of uncertainties on the coefficients
    """
    # Create the design matrix with polynomial features
    X_poly = np.column_stack([z * np.ones_like(X), X, X**2])

    # Fit the linear regression model to the data
    reg = LinearRegression().fit(X_poly, Y)

    # Get the coefficients and uncertainties
    coefficients = reg.coef_
    uncertainties = np.sqrt(np.diag(np.linalg.inv(X_poly.T @ X_poly)))

    return coefficients, uncertainties

def LinerReg(Xs, Ys, test_size=0.5, random_state=None, verbose=True):
    """
    Ajusta uma regressão linear simples (ou múltipla) e retorna o modelo e métricas.

    Parâmetros
    ----------
    Xs : array-like, shape (n_samples,) ou (n_samples, n_features)
        Variáveis preditoras.
    Ys : array-like, shape (n_samples,)
        Variável alvo.
    test_size : float
        Fração para o conjunto de teste (padrão 0.5).
    random_state : int ou None
        Semente para reprodutibilidade.
    verbose : bool
        Se True, imprime coeficientes e métricas.

    Retorna
    -------
    model : LinearRegression
        Modelo treinado.
    metrics : dict
        Dicionário com 'coef_', 'intercept_', 'mse', 'r2'.
    """

    Xs = np.asarray(Xs)
    Ys = np.asarray(Ys)

    # Garantir que X tenha forma 2D (n_samples, n_features)
    if Xs.ndim == 1:
        Xs = Xs.reshape(-1, 1)

    # Remover linhas com NaN/inf (opcional, mas útil)
    mask = np.isfinite(Xs).all(axis=1) & np.isfinite(Ys)
    Xs = Xs[mask]
    Ys = Ys[mask]

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, Ys, test_size=test_size, random_state=random_state
    )

    # Ajuste do modelo
    regr = LinearRegression()
    regr.fit(X_train, y_train)

    # Predição e métricas
    y_pred = regr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if verbose:
        print("Coeficientes:", regr.coef_)
        print("Intercepto:", regr.intercept_)
        print(f"Mean squared error: {mse:.4f}")
        print(f"R²: {r2:.4f}")

    return regr, {"coef_": regr.coef_, "intercept_": regr.intercept_, "mse": mse, "r2": r2}


def power_law(r, a, b):
    return a * r**b  # Power law function


def compute_density_profile(r, masses, nbins=30):
    """
    Compute the density profile of an N-body system.
    
    Parameters:
        r (array): Radial distances of the particles from the center.
        masses (array): Masses of the particles.
        nbins (int): Number of radial bins.
    
    Returns:
        bin_centers (array): Radial positions of the bins.
        densities (array): Mass density in each bin.
    """
    r_bins = np.logspace(np.log10(np.min(r[r>0])), np.log10(np.max(r)), nbins+1)  # Log-spaced bins
    bin_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
    volumes = (4/3) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
    
    density = np.zeros(nbins)
    for i in range(nbins):
        in_bin = (r >= r_bins[i]) & (r < r_bins[i+1])
        density[i] = np.sum(masses[in_bin]) / volumes[i] if np.sum(in_bin) > 0 else np.nan
    
    return bin_centers, density



def critical_density(z, H0=0.6774*100, Omega_m=0.3089, Omega_Lambda=0.6911, G = 4.300917270038e-06):
    """
    Compute the critical density at redshift z.

    Parameters:
        z (float): Redshift.
        H0 (float): Hubble constant in km/s/Mpc.
        Omega_m (float): Matter density parameter.
        Omega_Lambda (float): Dark energy density parameter.

    Returns:
        rho_crit_z (float): Critical density at redshift z in M_sun/kpc³.
    """
    H0 = H0 / (1000)  # Convert to km/s 1/kpc
    H_z = H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)  # H(z) in 1/s
    rho_crit_z = (3 * H_z**2) / (8 * np.pi * G)  # Msun/kpc3
    
    return rho_crit_z


def angle_between_vectors(A, B):
    """Compute the angle between two vectors in degrees."""
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    
    # Prevent domain errors due to floating-point precision
    cos_theta = np.clip(dot_product / (norm_A * norm_B), -1.0, 1.0)
    
    # Compute angle in radians and convert to degrees
    theta_rad = np.arccos(cos_theta)
    theta_deg = np.degrees(theta_rad)
    
    return theta_deg



def first_pericenter_index(r):
    """
    Retorna o índice do primeiro pericentro (primeiro mínimo local) em uma série r(t)
    em ordem temporal crescente (entry -> z=0).
    """
    r = np.asarray(r, dtype=float)

    # máscara para pontos válidos
    ok = np.isfinite(r)
    if ok.sum() < 3:
        return None

    # trabalhamos em índices "reais" (da array original) para não bagunçar posições
    idx = np.where(ok)[0]
    rr = r[idx]

    # 1) tenta mínimo local estrito: r[i-1] > r[i] < r[i+1]
    for j in range(1, len(rr) - 1):
        if (rr[j] < rr[j-1]) and (rr[j] < rr[j+1]):
            return idx[j]

    # 2) fallback: pericentro como primeiro ponto onde a derivada muda de negativa p/ positiva
    dr = np.diff(rr)
    for j in range(1, len(dr)):
        if (dr[j-1] < 0) and (dr[j] > 0):
            return idx[j]  # j refere ao ponto "central" (aprox. mínimo)
    
    return None


def ratio_mean_ssfr_after_before_peri(rOverR200, ssfr):
    """
    ssfr deve estar alinhada 1-para-1 com rOverR200 (mesmo slicing e flip).
    Retorna ratio, kperi e as médias (before, after).
    """
    r = np.asarray(rOverR200, dtype=float)
    s = np.asarray(ssfr, dtype=float)

    if len(r) != len(s):
        raise ValueError("rOverR200 e ssfr precisam ter o mesmo comprimento (mesmo slicing/flip).")

    k = first_pericenter_index(r)
    if k is None:
        return np.nan, None, np.nan, np.nan

    # precisa ter k-2,k-1 e k+1,k+2
    if k < 2 or k+2 >= len(s):
        return np.nan, k, np.nan, np.nan

    before = np.nanmean(s[k-2:k])      # k-2, k-1
    after  = np.nanmean(s[k+1:k+3])    # k+1, k+2

    # evita divisão por zero / valores inválidos
    if (not np.isfinite(before)) or before <= 0:
        return np.nan, k, before, after

    ratio = 10**after / 10**before
    return ratio, k, before, after