import numpy as np
import math as m
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
# import m.pi as pi
# import np.sqrt as sqrt
# import np.log as log
# import np.log10 as log10

def Gauss(x,mean,sig):
    """Gaussian function
    arguments:
        x
        mean
        standard deviation
    returns: Gaussian
    """
    sig2 = sig*sig
    dx = x-mean
    num = np.exp(-0.5*dx*dx/sig2)
    denom = np.sqrt(2*np.pi*sig2)
    return num/denom

def lnGauss(x,mean,sig):
    """ln Gaussian function
    arguments:
        x
        mean
        standard deviation
    returns: ln(Gaussian)
    """
    sig2 = sig*sig
    dx = x-mean
    return -0.5*dx*dx/sig2 - 0.5*np.log(2*np.pi*sig2)

def fitpolyn(x,y,ord):
    """fit polynomial of order ord to numpy array y(x)
    returns: coefficients (0 1 ... ord), total squared difference, rank, s
    Author: Gary Mamon (gam AAT iap.fr)"""
    A = np.zeros([len(x),ord+1])
    for pow in range(ord+1):
        A[:,pow] = x**pow
    return np.linalg.lstsq(A, y, rcond=None)[0]

def buildpolyn(x,coeff):
    """build polyomial from x and coefficient numpy arrays
    Author: Gary Mamon gam AAT iap.fr"""
    
    val = 0.
    for pow in range(0,len(coeff)):
        val = val + coeff[pow]*x**pow
    return val

def fitpolyn2d(x,y,z,ord):
    """fit 2D polynomial of order ord to numpy array z(x,y)
    returns: coefficients (00 10 01 20 11 02 ...), total squared difference, rank, s
    Author: Gary Mamon (gam AAT iap.fr)"""

    numsumpow = int(np.rint((ord+1)*(ord+2)/2))
    A = np.zeros([len(x),numsumpow])

    pow = 0
    for sumpow in range(ord+1):
        for powy in range(sumpow+1):
            powx = sumpow - powy
            A[:,pow] = x**powx * y**powy
            pow = pow + 1
    # print("A = ", A)
    return np.linalg.lstsq(A, z, rcond=None)
    
def buildpoly2d(x,y,coeffs):
    """build 2D polyomial from x and y numpy arrays and from coefficient numpy array
    Author: Gary Mamon gam AAT iap.fr"""
    
    ord = int(np.rint(-1.5+0.5*np.sqrt(9+8*len(coeffs))))
    val = 0.
    idx = 0
    for sumpow in range(ord+1):
        for powy in range(sumpow+1):
            powx = sumpow -powy
            val = val + coeffs[idx]* x **powx * y**powy
            idx = idx + 1
    return val

def NumPolyCoeffs(n):
    """number of coefficients of polynomial of order n
    Author: Gary Mamon (gam AAT iap.fr)"""
    
    return int(np.rint((n+1)*(n+2)/2))

def RMS(x):
    """root mean square of numpy array
    Author: Gary Mamon (gam AAT iap.fr)"""
    
    return np.sqrt(np.mean(x*x))

def bootstrap(x):
    """bootstrap numpy array
    Requirements: import numpy as np
    Author: Gary Mamon (gam AAT iap.fr),
    see also scipy.stats.bootstrap"""
    i = np.random.randint(low=0,high=len(x)-1,size=len(x))
    return x[i]

def boostrap_func(x,func=np.median,num_boots=1000):
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
    for n in range(num_boots):
        # function for the given bootstrap sample
        val[n] = func(bootstrap(x))
    return np.std(val)
 
def shuffle(x,y):
    """Random shuffle two numpy arrays
    Arguments:
        x, y: two numpy arrays (not necessarily of same length)
    Returns: two shuffled numpy arrays with same respective lengths
    
    Requirements: import numpy as np
    
    Author: Gary Mamon (gam AAT iap.fr)
    see also scipy.stats.permutation_test
    """

    xy = np.concatenate((x,y))
    np.random.shuffle(xy)
    return xy[0:len(x)],xy[len(x):]
    
def shuffle2D(x1,y1,x2,y2):
    """Random shuffle two 2D numpy arrays
    Arguments:
        x1, y1: 1st x,y pair of same length
        x2, y2: 2nd x,y pair of same length (possibly different from previous pair)
    Returns: two shuffled numpy arrays with same respective lengths
    
    Requirements: import numpy as np
    
    Author: Gary Mamon (gam AAT iap.fr)
    """
    
    x1x2 = np.concatenate((x1,x2))
    y1y2 = np.concatenate((y1,y2))
    xy = np.transpose([x1x2,y1y2])
    np.random.shuffle(xy)
    return xy[0:len(x1)],xy[len(x1):]

def CompareByShuffles(x,y,N=10000,stat='median',val=1,verbosity=0):
    """Compare two distributions 
        according to the difference in a statsitic
        using random shuffles
        
       Arguments :
           x, y: two numpy arrays (not necessarily of same length)
           N: number of random shuffles (default: 100000)
           stat: statistic used for difference (default: 'median')
           val: optional parameter for count of x and y above value
           
       Returns: f
           fraction of trials with greater stat_sample1 - stat_sample2
           
       Requires: 
           import numpy as np
           import scipy.stats as stats
           
       Author: Gary Mamon (gam AAT iap.fr)
        """
    # statistic on data
    if stat == 'median':
        stat_data = np.median(x) - np.median(y)
    elif stat == 'mean':
        stat_data = np.mean(x) - np.mean(y)
    elif stat == 'stdev':
        stat_data = np.std(x) - np.std(y)
    elif stat == 'kurt':
        stat_data = stats.kurtosis(x) - stats.kurtosis(y)
    elif stat == 'count':
        stat_data = len(x[x>val])/len(x) - len(y[y>val])/len(y)
    else:
        stat_data = stat(x) - stat(y)
    if verbosity > 0:
        print("stat_data=",stat_data)
        
    # loop over trials
    stat_shuf = np.zeros(N)
    for n in range(N):
        # shuffle data into same size components
        xshuffle, yshuffle = shuffle(x,y)
        
        # statistic on shuffled data
        if stat == 'median':
            stat_shuf[n] = np.median(xshuffle) - np.median(yshuffle)
        elif stat == 'mean':
            stat_shuf[n] = np.mean(xshuffle) - np.mean(yshuffle)
        elif stat == 'stdev':
            stat_shuf[n] = np.std(xshuffle) - np.std(yshuffle)
        elif stat == 'kurt':
            stat_shuf[n] = stats.kurtosis(xshuffle) - stats.kurtosis(yshuffle)
        else:
            stat_shuf[n] = stat(xshuffle) - stat(yshuffle)
            
    # check fraction of data with higher statistic
    Nworse = np.sum(stat_shuf > stat_data)
    if verbosity > 0:
        print("median stat of shuffled = ",np.median(stat_shuf))
    
    return Nworse/N

def ACO(X):
    """ArcCos for |X| < 1, ArcCosh for |X| >= 1
    arg: X (float, int, or numpy array)"""

    # author: Gary Mamon
    
    # temperorary X values to avoid error messages with arccos(X>1) and arccosh(X<1)
    tmpXbig = np.where(np.abs(X) > 1, X, 1)
    tmpXsmall = np.where(np.abs(X) <=  1, X, 1)
    
    return np.where(np.abs(X) <= 1, np.arccos(tmpXsmall),np.arccosh(tmpXbig))

def StandardDevGapper(x):
    """Gapper measure of standard deviation
    Author: Gary Mamon (gam AAT iap.fr)
    Sources: Wainer & Thiussen (1976), Beers, Flynn & Gebhardt (1990), AJ 100, 32"""
    
    # differences of sorted values
    diff = np.diff(np.sort(x))
    
    # weights
    N = len(x)
    i = np.arange(1,N)
    weight = i*(N-i)
    
    return np.sqrt(np.pi) / (N*(N-1)) * np.sum(weight*diff)

def BoxcarSmooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def derivative(x,y,order=1):
    """Derivative
    Args: x-array y-array order
        order=0: [f(x+h)-f(x)]/h (N-1)
        order=1: [f(x+h)-f(x-h)]/(2h) (N-2)
        order=2: [-f(x+2h)+8f(x+h)-8f(x-h)+f(x-2h)]/(12h) (N-4) padded by order=1
    Author: Gary Mamon (gam AAT iap.fr)
    """
    
    if len(y) != len(x):
        raise ValueError("xvec and yvec musthave same length")
    N = len(y)
    if np.std(np.diff(x)) > 0.001:
        raise ValueError("xvec must be equally spaced")
    h = np.mean(np.diff(x))
    if order == 0:
        i = np.arange(N-1)
        yplus = y[i+1]
        yminus = y[i]
        return (yplus-yminus)/h
    elif order == 1:
        i = np.arange(1,N-1)
        yplus = y[i+1]
        yminus = y[i-1]
        dydx = np.concatenate(
            (
                np.array([(y[1]-y[0])/h]),
                (yplus-yminus)/(2*h),
                np.array([(y[N-1]-y[N-2])/h])
            ))
        return dydx
    elif order == 2:
        i = np.arange(2,N-2)
        yplus2 = y[i+2]
        yplus1 = y[i+1]
        yminus1 = y[i-1]
        yminus2 = y[i-2]
        dydx = np.concatenate(
            (
                np.array([(y[1]-y[0])/h]),
                np.array([(y[2]-y[0])/(2*h)]),
                (-1*yplus2+8*yplus1-8*yminus1+yminus2)/(12*h),
                np.array([(y[N-1]-y[N-3])/(2*h)]),
                np.array([(y[N-1]-y[N-2])/h])
            )
        )
        return dydx
    else:
        raise ValueError("cannot recognize order = " + str(order))

def BinomialError(N,n,Wilson_nsigma=1.65,Wilson_center_to_zero=False):
    """Binomial fractions
    Arguments:
        N: total number of points in bin (numpy array)
        n: number of interesting points in bin (numpy array of same length as N)
        Wilson_nsigma: 1 for 1 sigma, 1.65 for 95% upper limit
        Wilson_center_to_zero: center points to 0 for Wilson
        recommend default values for 1 sigma on regular points and 1.65 sigma 
                (95% confidence upper [n=0] or lower [n=N] limits)
    Author: Gary Mamon (gam AAT iap.fr)"""
    if not isinstance(N,np.ndarray):
        raise ValueError("N must be a numpy array")
    if not isinstance(n,np.ndarray):
        raise ValueError("n must be a numpy array")
    if len(n) != len(N):
        raise ValueError("N and n must have same length")
    N_tmp = np.where(N==0,1,N)
    p_orig = np.where(N==0,-1,n/N_tmp)
    error_p_orig = np.where(N==0,-1,np.sqrt(p_orig*(1-p_orig)/N))
    Wilson_nsigma2 = Wilson_nsigma*Wilson_nsigma
    if np.max(Wilson_nsigma) > 0:
        if Wilson_center_to_zero:
            p = np.select([n==0,n==N],[0,1],p_orig)
        else:
            p = np.where(((n==0) | (n==N)),(n+0.5*Wilson_nsigma2)/(N+Wilson_nsigma2),p_orig)
        error_p = np.where(((n==0) | (n==N)),Wilson_nsigma*np.sqrt(n*(1-n/N)+0.25*Wilson_nsigma2)/(N+Wilson_nsigma2),
                              error_p_orig)
    else:
        p = p_orig
        error_p = error_p_orig
    return p, error_p

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
    Minv = inv(M)
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

def Intersect(x,y1,y2,verbose=0,plot=False):
    """Find 1st abscissa where two  curves intersect
    
    arguments: 
        x: x-vector
        y1 & y2: y1 & y2 vectors
        verbose: verbosity (0 for none)
        plot: optional plot
        
    returns:
        solution x_sol where y1(x_sol) = y2(x_sol)
        
    requires:
        from scipy.interpolate import interp1d
        from matplotlib import pyplot as plt
    author:
        Gary Mamon (gam AAT iap.fr)
    """
    
    # order of interpolation
    if len(x) > 3:
        kind = 'cubic'
    else:
        kind = 'linear'
        
    # sort by x
    tab = np.array([x,y1,y2]).T
    if verbose > 0:
        print(tab)
    tab = tab[tab[:,0].argsort()]
    if verbose > 0:
        print(tab)
    x = tab[:,0]
    y1 = tab[:,1]
    y2 = tab[:,2]
    
    # x(y1) & x(y2)
    fun_y1ofx = interp1d(x,y1,assume_sorted=True,kind=kind)
    fun_y2ofx = interp1d(x,y2,assume_sorted=True,kind=kind)
    
    # evaluate on fine grid
    x_fine = np.linspace(x.min(),x.max(),10001)
    y1_fine = fun_y1ofx(x_fine)
    y2_fine = fun_y2ofx(x_fine)
    diff_fine = y2_fine-y1_fine
    
    # quick solution on fine grid: first change of sign of y2-y1
    i_fine = np.arange(len(x_fine)).astype(int)
    if diff_fine.min()*diff_fine.max() > 0:
        raise ValueError("no solution!")
    if diff_fine[0] > 0:
        isol_coarse = i_fine[diff_fine<0][0]
    elif diff_fine[0] < 0:
        isol_coarse = i_fine[diff_fine>0][0]
    else:
        if verbose > 0:
            print("diff_fine[0]=0")
        return x_fine[0]
    
    # linear interpolation of two closest points to solution
    x4interp = x_fine[isol_coarse-1:isol_coarse+1]
    diff4interp = diff_fine[isol_coarse-1:isol_coarse+1]
    f = interp1d(diff4interp,x4interp,kind='linear')

    # plot if requested    
    if plot:
        plt.figure()
        plt.scatter(x,y1,s=20,c='b')
        # plt.scatter(x,y2,s=20,c='g')
        plt.plot(x_fine,y1_fine,lw=0.5,c='b')
        plt.plot(x_fine,y2_fine,lw=0.5,c='g')
        x = np.array([f(0)])
        y = fun_y1ofx(x)
        plt.scatter(x,y,s=10,c='r')
        plt.yscale('log')
        plt.show()
    return f(0), fun_y1ofx(f(0))

def CartesianToSphericalVelocities(pos,vcart):
    """Velocities in spherical coordinates 
        given velocities in cartesian coordinates
        
    Arguments:
        pos: positions ([N,3] array)
        vcart: cartesian velocities ([N,3] array)
        
    Returns:
        vr, vtheta, vphi ([N] arrays])
    
    Author: Gary Mamon (gam AAT iap.fr)"""
    
    vel = vcart
    R = np.sqrt(pos[:,0]*pos[:,0] + pos[:,1]*pos[:,1])
    r = np.sqrt(R*R + pos[:,2]*pos[:,2])
    vr = np.sum(pos*vel,axis=1)/r
    vtheta = ((pos[:,0]*vel[:,0] + pos[:,1]*vel[:,1])*pos[:,2] - R*R*vel[:,2]) \
              / (r*R)
    vphi = (pos[:,0]*vel[:,1]-pos[:,1]*vel[:,0]) / R
    return [vr,vtheta,vphi]

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
    costheta = np.clip(r_dot_J / (r*J),-1,1)
    theta = np.arccos(costheta)
    sintheta = np.clip(np.sin(theta),-1,1)
    theta2 = theta+theta
    sec2theta = 1/np.cos(theta2)
    matrix = np.array([[-1*sec2theta*sintheta,costheta*sec2theta,0],
              [0,0,1],
              [costheta*sec2theta,-1*sintheta*sec2theta,0]
              ])
    
    # new positions and velocities
    POS = np.dot(matrix, pos)
    VEL = np.dot(matrix, vel)

    return POS, VEL

def AICc(ln_prob,N_params,N_data):
    """Akaike Information Criterion (corrected for small samples)
    arguments: 
        ln_Prob: log (likelihood or poserior)
        N_params: number of free parameters
        N_data: number of data points
    returns AICc
    source: Wikipedia
    author of algorithm: 
            AIC: H. Akaike (1973) 
              "Information theory and an extension of the maximum likelihood principle", in Petrov, B. N.; Csáki, F. (eds.), 2nd International Symposium on Information Theory, Tsahkadsor, Armenia, USSR, September 2-8, 1971, Budapest: Akadémiai Kiadó, pp. 267–281. Republished in Kotz, S.; Johnson, N. L., eds. (1992), Breakthroughs in Statistics, vol. I, Springer-Verlag, pp. 610–624. for AIC
            AICc:
                Sugiura (1978) Comm. in Stat. - Theory and Methods, 7: 13, 
                    for linear regression
                Hurvich & Tsai (1989) Biometrika, 76 (2): 297, 
                    for more general problems
    author of code: Gary Mamon (gam AAT iap.fr)"""
    
    AIC = -2*ln_prob + 2*N_params
    AICc = AIC + 2*N_params*(N_params+1)/(N_data-N_params-1)
    return AICc

def BIC(ln_prob,N_params,N_data):
    """Bayes Information Criterion
    arguments: 
        ln_Prob: log (likelihood or poserior)
        N_params: number of free parameters
        N_data: number of data points
    returns BIC
    source: Wikipedia
    author of algorithm: 
            G. Schwarz (1978), Annals of Statistics, 6 (2): 461–464
    author of code: Gary Mamon (gam AAT iap.fr)"""
    BIC = -2*ln_prob + np.log(N_data)*N_params
    return BIC

def WeightedMedian_proofreader(values, weights):
    """Weighted median using dataframes"""
    # author: proofreader (https://stackoverflow.com/users/2774479/prooffreader)
    # source: https://stackoverflow.com/questions/26102867/python-weighted-median-algorithm-with-pandas
    # 10x slower than WeightedMedian_Ash
    
    df = pd.DataFrame(np.column_stack((values,weights)),columns=['values','weights'])
    df.sort_values('values', inplace=True)
    cumsum = df.weights.cumsum()
    cutoff = df.weights.sum() / 2.0
    return df.values[cumsum >= cutoff][0,0]

def WeightedMedian_Ash(values, weights):
    '''Weighted median of values, as follows:
         1- sort both lists (values and weights) based on values.
         2- select the 0.5 point from the weights 
              and return the corresponding values as results
         e.g. values = [1, 3, 0] and weights=[0.1, 0.3, 0.6] assuming weights are probabilities.
         sorted values = [0, 1, 3] and corresponding sorted weights = [0.6, 0.1, 0.3] the 0.5 point on
         weight corresponds to the first item which is 0. so the weighted median is 0.'''

    # author: Ash (https://stackoverflow.com/users/956730/ash)
    # source: https://stackoverflow.com/questions/26102867/python-weighted-median-algorithm-with-pandas
    # 10x faster than WeightedMedian_proofreader
    
    # convert the weights into probabilities
    sum_weights              = sum(weights)
    weights                  = np.array([(w*1.0)/sum_weights for w in weights])
    # sort values and weights based on values
    values                   = np.array(values)
    sorted_indices           = np.argsort(values)
    values_sorted            = values[sorted_indices]
    weights_sorted           = weights[sorted_indices]
    # select the median point
    it                       = np.nditer(weights_sorted, flags=['f_index'])
    accumulative_probability = 0
    median_index             = -1
    while not it.finished:
        accumulative_probability  += it[0]
        if accumulative_probability > 0.5:
            median_index           = it.index
            return values_sorted[median_index]
        elif accumulative_probability == 0.5:
            median_index           = it.index
            it.iternext()
            next_median_index      = it.index
            return np.mean(values_sorted[[median_index, next_median_index]])
        it.iternext()

    return values_sorted[median_index]
