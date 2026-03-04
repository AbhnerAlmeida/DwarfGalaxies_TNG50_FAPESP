import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mplcol             # for log colorbars

import numpy as np
import langutils as lu
import mathutils as mu

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] \
    + plt.rcParams['font.serif']
    
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

#-------------------------------------------------------------

def plot2(xvec,yvec,xlog=0,ylog=0,xerrvec=None,yerrvec=None,xlims=None,ylims=None,
          size=None,grid=1,xlab=None,ylab=None,labsize=20,ticklabsize=16,maxmajticks=6,leg=None,tit=None,dev=None):
    """scatter plot with automatic limits and axis labels
    authors: Eliott & Gary Mamon
    arguments:
        xvec:        np.array of x-values or np.array of np.arrays of x-values
        yvec:        np.array of y-values or np.array of np.arrays of y-values
        xerrvec:     np.array of x-value errors or np.array of np.arrays of x-value errors
        yerrvec:     np.array of y-value errors or np.array of np.arrays of y-value errors
        xlims:       list of min and max x for plot limits
        ylims:       list of min and max y for plot limits
        size:        sizes of plot markers (default: 2000/len(xvec) in range 0,360)
        grid:        0 --> no grid, 1: major grid, 2: major and minor grids
        xlab:        xlabel (e.g. '$x_2$' for x subscript 2 in LaTeX mode)
        ylab:        ylabel (same syntax)
        labsize:     label size (default 20)
        ticklabsize: tick number label size (default 16)
        maxmajticks: maximum number of major ticks per axis (default 6)
        leg:         plot legend
        tit:         plot title
        dev:         device (None for screen, 'test' for 'test.pdf', else string)
"""
    plt.rc('text', usetex=True)
    fig,ax = plt.subplots(figsize=(6,6))  # inner ticks on all 4 sides  

    # log axes if necessary
    if xlog:
        plt.xscale('log')
    if ylog:
        plt.yscale('log')

    # automatic point size according to length of array
    if size is None:
        size = 2000/len(xvec)
    smin=0
    smax=360
    if size > smax:
        size=smax
    if size < smin:
        size=smin
    # add minor ticks
    plt.minorticks_on()

    ## automatic plot limits (going slightly beyond min-max of data)
    if xlims is not None:
        plt.xlim(xlims)
    if ylims is not None:
        plt.ylim(ylims)
        
    # add grid if desired
    if grid:
        plt.grid(which='major',color='gray',linestyle='--',axis='both')
        if grid >= 2:
            plt.grid(which='minor',color='gray',linestyle=':',axis='both')

    # TeX labels
    Nxvecs = xvec.ndim
    # scatter plot for list of xvec,yvec or for single xvec,yvec
    mycolors = ['k','r','g','b','c','m','orange']
    if Nxvecs > 1:
        xoffset = (1,1)
        for i in range(Nxvecs):
            if len(yvec[i]) > 0:
                plt.scatter(xoffset[i]*xvec[i], yvec[i], s=size, c=mycolors[i], label=leg[i])
                plt.errorbar(xoffset[i]*xvec[i],yvec[i],xerr=xerrvec[i],yerr=yerrvec[i],c=mycolors[i],ls='none',elinewidth=2)
    else:
        plt.scatter(xvec, yvec, s=size, c='r', label=leg)
        plt.errorbar(xvec,yvec,xerr=xerrvec,yerr=yerrvec,c='r',ls='none',elinewidth=2)

    # automatic axis labels
    if xlab is None and Nxvecs==1:
        xlab = lu.getvarname(xvec)
        print("xlab is now", xlab)
    if ylab is None and Nxvecs==1:
        ylab = lu.getvarname(yvec)
    plt.xlabel(xlab,fontsize=labsize)
    plt.ylabel(ylab,fontsize=labsize)

    # tickmarks and number labels
    plt.tick_params(labelsize=ticklabsize)
    ax.xaxis.set_major_locator(MaxNLocator(maxmajticks, prune="lower"))
    ax.yaxis.set_major_locator(MaxNLocator(maxmajticks, prune="lower"))

    if leg is not None:
        plt.legend(loc='best')
    if tit is not None:
        plt.title(tit)

    # choose plot style from argument
    # plt.style.use(style)

    # show full plot or save to device
    if dev is None:
        plt.show()
    else:
        if dev.find('.pdf')==len(dev)-4:
            dev2 = dev[0:len(dev)-4]
        else:
            dev2 = dev
        fig.savefig("%s.pdf" % dev2, bbox_inches='tight')        

def func_count(C):
    return len(C)

def func_dispersion(C):
    C = np.asarray(C)
    N_points = len(C)
    if (N_points < min_len):
        return np.nan
    variance = N_points/(N_points-1) * np.nanmean(C*C - np.nanmean(C)*np.nanmean(C))
    return np.sqrt(variance)

def func_mean(C):
    C = np.asarray(C)
    if (len(C) < min_len):
        return np.nan
    return np.nanmean(C)

def func_median(C):
    C = np.asarray(C)
    if (len(C) < min_len):
        return np.nan
    return np.nanmedian(C)

def plothex(x,y,z,title=None,xlims=None,ylims=None,xlab=None,ylab=None,zlab=None,
            n_per_cell=None,color_map=None,colorbar=True,zscale='lin',n_grid=None,
            flag_z='count',grid=None):
    """ 2D plot in hexagonal cells
    author: Eduardo Vitral (retouched by Gary Mamon with added z_flag='counts')
    """

    if (len(x) != len(y) or len(x) != len(z) or len(y) != len(z)):
        string = "Error: x y and z arrays must have the same length"
        raise ValueError(string)
    if (len(x) < 2 or len(y) < 2 or len(z) < 2):
        string = "Error: arrays must have length > 2"
        raise ValueError(string)
    # if flag_z not in ['count','mean','median','dispersion','std']:
    #     raise ValueError("flag_z must be one of ['count','mean','median','std']")
    if (n_grid == None):
        n_grid = int(len(x)/20)
        if (n_grid == 0):
            n_grid = 2
            
    # define custom (segmented rainbow) color map
    if (color_map == None):
        n_bins    = 1000
        
        colors    = [(0,0,50/255),(102/255,178/255,1),
                     (1,102/255,102/255),(1,1,1)]
        cmap_name = 'my_rainbow'
        rainbow   = LinearSegmentedColormap.from_list(cmap_name,colors,
                                                      N=n_bins)
        
        color_map = rainbow
    
    global min_len
    
    if (n_per_cell == None):
        min_len = 1
    else:
        if (n_per_cell < 1 or (type(n_per_cell) != int)):
            string = "Error: please provide a valid argument for" + \
                        "'n_per_cell', i.e. (int) and (>=1)"
            raise ValueError(string)
        else:
            min_len = n_per_cell
       
    if (flag_z == 'mean'):
        C_function = func_mean
    elif (flag_z == 'count'):
        C_function = func_count
    elif (flag_z == 'median'):
        C_function = func_median
    elif (flag_z in ['std','dispersion']):
        C_function = func_dispersion
    else:
        C_function = flag_z
        # string = "Error: provide a valid argument for" + \
        #                 "'flag_z': 'count', 'mean', 'median' or 'dispersion'"
        # raise ValueError(string)

    # start figure
        
    fig,axs = plt.subplots(facecolor='w', edgecolor='k')

    if xlims != None:
        plt.xlim(xlims)
    if ylims != None:
        plt.ylim(ylims)
    if (title == None):
        pass
    else:
        axs.set_title(title, fontsize = 14)
    if zscale == 'log':
        norm = mplcol.LogNorm()
    else:
        norm = None
    axs.set_facecolor('w')
    c=axs.hexbin(x,y,C=z,cmap=color_map,gridsize=n_grid,norm=norm,
                 reduce_C_function=C_function)

    if colorbar:
        cbar = fig.colorbar(c, ax = axs)
        cbar.ax.tick_params(labelsize=12.5)

        if (zlab == None):
            pass
        else:
            cbar.ax.set_title(zlab,fontsize = 13)

    # Set common labels
    if (xlab == None):
        pass
    else:
        plt.xlabel(xlab,fontsize=14)
    
    if (ylab == None):
        pass
    else:
        plt.ylabel(ylab,fontsize=14)


    axs.tick_params(labeltop=False, labelright=False, top = True, right = True, \
                    axis='both', which='major', labelsize=13, direction="in", \
                    length = 8)
    axs.tick_params(labeltop=False, labelright=False, top = True, right = True, \
                    axis='both', which='minor', labelsize=13, direction="in", \
                    length = 4)

    axs.set_axisbelow(False)
    plt.minorticks_on()
    if grid != None and grid != False:
        plt.grid(which='minor',color='gray',linestyle=':',axis='both')
        plt.grid(which='major',color='gray',linestyle='--',axis='both')
    else:
        plt.grid(False)

def autolims(y,scale='linear',padding=0.075):
    """Automatic axis limits
    arguments: 
        y: list or numpy array of values
        scale: 'linear' or 'log'
        padding: extra padding (in units of range of y)

    returns: ymin,ymax

    author: Gary Mamon (gam AAT iap.fr)
    """

    if not isinstance(y,np.ndarray):
        y = np.array(y)
    if scale == 'log':
        yplus = y[y>0]
        lyplus = np.log10(yplus)
        minlyplus = np.min(lyplus)
        maxlyplus = np.max(lyplus)
        rangelyplus = maxlyplus-minlyplus
        ymin = 10**(minlyplus-padding*rangelyplus)
        ymax = 10**(maxlyplus+padding*rangelyplus)
    else:
        miny = np.min(y)
        maxy = np.max(y)
        rangey = maxy - miny
        ymin = miny-padding*rangey
        ymax = maxy+padding*rangey
    return ymin,ymax

def binomialerrorplot(x,N,n,color='k',marker='o',mec='k',capsize=0,capthick=1,
                       markersize=30,label=None,scale='linear',eyFactor=None,
                       ax=None,verbose=0):
    """binomial plot
    arguments:
        x: x array
        N: array of total points
        n: array of good points
        color: face color of points, color of error bars, and color of upper and lower limits
        marker: marker shape
        markersize: markersize
        mec: color surrounding points
        capsize: size of error bar edges and of error bar arrow heads
        label: label for legend
        scale: scale of y axis: 'linear' or 'log'
        eyFactor: length of upper and lower limit symbols (number of dex for scale='log')
        
    author: Gary Mamon (gam AAT iap.fr)
        """
        
    # restrict to bins with points
    condGood = N>0
    x = x[condGood]
    N = N[condGood]
    n = n[condGood]
    
    # statistics on bins with points
    p, ep = mu.BinomialError(N,n)
    condUpper = n==0
    condLower = n==N
    condPoints = np.logical_not(np.logical_or((condUpper),(condLower)))
    if verbose > 0:
        print(" x   fraction    err(frac)  type")
        print(np.transpose([x,p,ep,condPoints]))
        
    # points with error bars
    if ax is None:
        ax = plt.gca()
    ax.errorbar(x[condPoints],p[condPoints],ep[condPoints],
                 marker=marker,mec=mec,mfc=color,ecolor=color,ls='none',
                 ms=markersize,
                 capsize=capsize,capthick=capthick,label=label)
    

    
    # points with upper limits
    if scale=='log':
        if eyFactor is None:
            eyFactor = 0.5
        ep = p*(1-10**(-eyFactor))
    else:
        if eyFactor is None:
            eyFactor = 0.075
        ep = 0*ep + eyFactor
    # ensure that upper limit arrows do not extend past y=0
    # apply /2 fudge factor since capsize adds to length of arrow and is only known in pixels
    ep = np.where(p-ep<0,p/2,ep)

    if verbose >= 1:
        print("ep=",ep)
    uplims = np.full(len(x[condUpper]),True)
    ax.errorbar(x[condUpper],p[condUpper],ep[condUpper],
                 marker=',',mec=mec,mfc=color,ecolor=color,ls='none',
                 capsize=capsize,capthick=capthick,uplims=uplims)
    if verbose > 0:
        print("points with upper limits:")
        print()
    
    
    # points with lower limits
    if scale == 'log':
        ep = p*(10**eyFactor-1)
    # ensure that arrows do not extend past y=1
    # apply /2 fudge factor since capsize adds to length of arrow and is only known in pixels
    ep = np.where(p+ep>1,(1-p)/2,ep)
    lolims = np.full(len(x[condLower]),True)
    ax.errorbar(x[condLower],p[condLower],ep[condLower],
                 marker=',',mec=mec,mfc=color,ecolor=color,ls='none',
                 capsize=capsize,capthick=capthick,lolims=lolims)

    if scale=='log':
        ax.set_yscale('log')

def Markersize(N,dim=2,division=1):
    """ Automatic marker size
    arguments:
        N: number of bins (1d) or points (2d)
        dim: 1 or 2
        division: extra reduction (default 1)
    
    author: Gary Mamon (gam AAT iap.fr)
    """
    if dim == 1: # histogram
        if N > 50:
            markersize = 3
        else:
            markersize = 150/N
    elif dim == 2: # scatter plot
        if N > 2500:
            markersize = 3
        else:
            markersize = 150/np.sqrt(N)
    else:
        raise ValueError("dim must be 1 or 2")
    return markersize/division

def PlotCDF(a,reverse=False,norm=True,pad=0,color='b',lw=2,label=None):
    """Plot cumulative distribution function
    a: numpy array
    reverse: reerse CDF (default False)
    norm: normalize to unity (deault True)
    pad: pad edges to 0, 1 (number of values, for norm=True, default 0 [no pad])

    Author: Gary Mamon (gam AAT iap.fr)
    adapted from grand_chat
    https://stackoverflow.com/questions/24788200/calculate-the-cumulative-distribution-function-cdf-in-python"""
    # compute CDF
    x, counts = np.unique(a, return_counts=True)
    x = np.sort(x)
    cusum = np.cumsum(counts)
    if norm:
        cusum = cusum/cusum.max()
    if reverse:
        print("before reverse: cusum=",cusum)
        cusum = 1 - cusum
    # pad on edges
    for i in range(pad):
        xmin = x.min()
        xmax = x.max()
        xrange = xmax - xmin
        dx = 0.04*xrange
        x = np.pad(x,1,'constant',constant_values=[xmin-dx,xmax+dx])
    if ((pad > 0) & norm):
        if reverse:
            yvals = [1,0]
        else:
            yvals = [0,1]
        cusum = np.pad(cusum,pad,'constant',constant_values=yvals)

    # plot
    plt.plot(x,cusum,drawstyle='steps-post',lw=lw,color=color,label=label)

def Plotrhoofr(x,xmin,xmax,Nbins,weights=None,dim=3,norm=1,conc=None,
               mark='o',mec='k',labfontsize=16,
               titlesize='large',
               device=None,xlims=None,ylims=None,title=None,
               xlab=None,ylab=None,mfc=None,markersize=None,radiusUnit=None,
               densityUnit=None,
               label=None,usetex=False,initialize=True,finalize=False,
               plot=True,
               legend=False,
               showField=True,
               xfitmin=None,xfitmax=None,params=None,bounds=None,
               aux=None,Nclus=1,fixfield=False,probcounts=False,
               fieldratio=False,chisq=False,
               model="NFW",method="Powell",tol=0.0001,
               verbosity=0):
    """Plot 3D density or 2D surface density profiles
    arguments:
        x np array to plot (radii or projected radii, could be dimensionless)
        xmin, xmax: limits to x
        Nbins: number of bins between xmin and xmax
        norm: normalization (e.g. 1/N(r_vir) if in virial units)
        dim: dimension (3 for 3D density, 2 for surface density)
        aux: auxiliary fixed parameters (default None)
        Nclus: number of clusters (default 1)
        device: ouptut device
        xlims, ylims: x and y limits of plot
        fixfield: fix surface density bounds (dim=2 only) (default False)
        xlab, ylab: x- and y- labels
        mark: marker shape
        mfc: marker face color
        radiusUnit: unit of radius for plot (e.g. "kpc" or "virial")
        densityUnit: unit of density for plot (e.g. kpc^{-3})
        label: for legend
        title: plot title
        legend: plot legend (default False)
        labfontsize: label font size (default 'medium')
        titlesize: title font size (default 'medium')
        usetex: use TeX labels? (slow)
        initialize: start new plot (default True)
        finalize: finalize plot (default False)
        xfitmin, xfitmax: limits to fit region
        params: parameters of fit
        bounds: bounds of fit (pairs of parameters)
        fixfield: change param and bounds of field for dim=2 (default False)
        probcounts: handle properly the counts (default False)
        fieldratio: consider log(Sigma_field/<Sigma_model(a)>) instead of log(Sigma_field) (default False)
    author: Gary Mamon (gam AAT iap.fr)
  """
    # marker size
    if markersize is None:
        markersize = Markersize(Nbins,dim=1)
      
    # counts in bins
    # 1) restrict to bounds
    cond = (x>xmin) & (x<xmax)
    x = x[cond]
    if weights is not None:
        weights = weights[cond]
    # counts in bins
    lx = np.log10(x)
    # if xlims is None:
    #     counts, bin_edges = np.histogram(lx,bins=Nbins)    
    # else:
    if weights is None:
        counts, bin_edges = np.histogram(lx,bins=Nbins)
    else:
        counts, bin_edges = np.histogram(lx,bins=Nbins,weights=weights)
    xminbins = 10**bin_edges[:-1]
    xmaxbins = 10**bin_edges[1:]
    lxbins = (bin_edges[1:]+bin_edges[:-1])/2
    xbins = 10**lxbins
    if verbosity >= 1:
        print("verbosity=",verbosity,"dim=",dim)
        print("min max radii of bins = ",xbins.min(),xbins.max())

    # 95% confidence upper limits for counts = 0
    # see http://ms.mcmaster.ca/peter/s743/poissonalpha.html
    ecounts_tmp = np.where(counts==0,3.69,np.sqrt(counts))
    upperlimits = np.where(counts==0,True,False)

    # density and error
    area = np.pi * (xmaxbins**2-xminbins**2)
    volume = 4/3*np.pi * (xmaxbins**3-xminbins**3)
    if norm is None:
        norm = 1/np.sum(counts)
    if dim == 3:
        dens = norm*counts/volume
        edens = norm/volume * ecounts_tmp
    elif dim == 2:
        dens = norm*counts/area
        edens = norm/area * ecounts_tmp
    else:
        raise ValueError("cannot understand dim = " + str(dim))
        
    if ((params is None) & (bounds is not None)):
        bounds = np.array(bounds)
        params = (bounds[:,0]+bounds[:,1])/2
    elif bounds is None:
        bounds = np.array([[-2,0.5],[-2,4],[-3,6]])
        params = (bounds[:,0]+bounds[:,1])/2
    # print("params=",params,"bounds=",bounds)
    # if (chisq & (dim==2)):
    #     res = FitChisqNFW(xbins,dens,edens,bounds,R_min=xmin,R_max=xmax,
    #                       verbosity=verbosity)
    #     a = 10**res.x[0]
    #     c = 1/a
    #     Nofa = 10**res.x[1]
    #     Sigma_field = 10**res.x[2]
    #     if verbosity >= 1:
    #         print("chi^2: res.x=",res.x)
    #         print("chi^2: c N(a) Sigma_field=",c,Nofa,Sigma_field)
        
    # plot
    if plot:
        if verbosity >= 1:
            print("plot: dim=",dim,len(x),"galaxies")
        if initialize:
            fig = plt.figure(figsize=(6,6))
        ax = plt.gca()
        # limit on field surface density
        if ((dim == 2) & fixfield):
            # max from surface density in last factor 2
            xout = x[x>xmax/2]
            if verbosity >= 1:
                print("len(xout)=",len(xout))
            log_Sigma_field_max = np.log10(len(xout)/(np.pi*xmax**2*(1-0.25)))
            if verbosity >= 1:          
                print("old: params, bounds = ",params,bounds)
            # print("log_Sigma_field_max=",log_Sigma_field_max)
            params[2] = log_Sigma_field_max-0.5
            bounds[2] = [log_Sigma_field_max-1,log_Sigma_field_max]
            if verbosity >= 1:
                print("new: params, bounds = ",params,bounds)

        if mfc is None:
            mfc = 'r'
            
        # shade region of fit
        plt.axvspan(xfitmin,xfitmax,color='bisque',alpha=0.3)

        # plt.text(np.sqrt(xfitmin*xfitmax),0.9,'fit region',transform=ax.tra,
        #          ha='center',color='brown',fontsize=12)
        plt.errorbar(xbins,dens,yerr=edens,marker=mark,c='k',mfc=mfc,mec=mec,
                     markersize=markersize,ls='None',uplims=upperlimits,label=label)

        # finish plot
        plt.xscale('log')
        plt.yscale('log')
        if xlims is not None:
            if verbosity>=1:
                print("xlims=",xlims)
            plt.xlim(xlims)
        if ylims is not None:
            plt.ylim(ylims)
        ylims = ax.get_ylim()

        if ((xlab is None) & (radiusUnit is None)):
            xlab = 'radius'
        elif xlab is None:
            xlab = 'radius' + ' (' + radiusUnit + ')'
        plt.xlabel(xlab,fontsize=labfontsize)
        if ((ylab is None) & (densityUnit is None)):
            if dim == 2:
                ylab = 'surface density ($N_\mathrm{vir}/r_\mathrm{vir}^2$)'
            else:
                ylab = 'density ($N_\mathrm{vir}/r_\mathrm{vir}^3$)'
        elif ylab is None:
            if dim == 2:
                ylab = 'surface density'
            else:
                ylab = 'density'
            ylab = ylab + ' (' + densityUnit+ ')'
        plt.ylabel(ylab,fontsize=labfontsize)
        if legend:
            plt.legend()
        if title is not None:
            plt.title(title,fontsize=titlesize)
        plt.tight_layout()
    if finalize:
        if device is not None:
            print("in device block")
            if verbosity > 0:
                print("saving figure to",device)
            plt.savefig(device)
        else:
            if verbosity > 0:
                print("plt.show...")
            plt.show()
    # return quantities
    if initialize:
        return xminbins, xmaxbins, counts, dens, edens, fig
    else:
        return xminbins, xmaxbins, counts, dens, edens
    
def Ternary(ftop=None,fleft=None,fright=None,inputs='fracs',
            labeltop=None,labelleft=None,labelright=None,
            points='scatter',gridsize=10,func=len,norm=None,
            ms=5,ls='-',lw=3,color='r',marker='o',
            # tlim=None,llim=None,rlim=None,
            edgecolor='face',fontsize=18,title=None,saveprefix=None):
    """ternary (triangle) plots (for 3 variables in [0,1] that sum to unity)
    
    arguments:
        ftop,fleft,fright: 3 variables in np.arrays, according to argument inputs:
            input='fracs': 3 variables are fractions [0,1] that sum to unity
            input='vars': 3 variables that are internally converted to fractions
            input='log-vars': 3 log-variables that are internally converted to fractions
        labeltop,labelleft,labelright: labels [without 'fraction', which is added automatically]
        points: one of 'scatter', 'plot', 'tribin', 'hexbin'
                    (the latter two for density plots)
        func: function for density plots (default len for counts)
        norm: None or 'log' for density plots in log counts
        ms: plt.scatter marker size (default 5)
        edgecolor: color of symbol edges (default 'face' for None)
        other arguments are graphics related
        
    requirements:
        import mpltern
        
    author: Gary Mamon (gam AAT iap.fr)
    """
    if inputs == 'fracs':
        frac_top = ftop
        frac_left = fleft
        frac_right = fright
        sumfracs = frac_top + frac_left + frac_right
        if np.max(np.abs(sumfracs-1)) > 0.01:
            string = "fractions do not sum well to unity:\n" \
                     + "  min and max sum fractions=%.2f"%sumfracs.min() \
                     + ", %.2f"%sumfracs.max()
            raise ValueError(string)
    elif inputs == 'vars':
        sumvars = ftop + fleft + fright
        frac_top = ftop/sumvars
        frac_left = fleft/sumvars
        frac_right = fright/sumvars
    elif inputs == 'log-vars':
        frac_top = 10**ftop/sumvars
        frac_left = 10**fleft/sumvars
        frac_right = 10**fright/sumvars
    else:
        raise ValueError("Cannot recognize inputs=" + inputs)

    ax = plt.subplot(projection="ternary")
    if norm == 'log':
        norm = mplcol.LogNorm()
    if points == 'scatter':
        ax.scatter(frac_top,frac_left,frac_right,s=ms,color=color,marker=marker)
    elif points == 'plot':
        ax.plot(frac_top,frac_left,frac_right,color=color,ls=ls,lw=lw)
    elif points == 'tribin':
        ax.tribin(frac_top,frac_left,frac_right,gridsize=gridsize,
                  color=color,cmap='Greys',reduce_C_function=func,
                  edgecolors=edgecolor)
    elif points == 'hexbin':
        ax.hexbin(frac_top,frac_left,frac_right,gridsize=gridsize,norm=norm,
                  color=color,cmap='Greys',reduce_C_function=func,
                  edgecolors=edgecolor)
    if labeltop is not None:
        ax.set_tlabel('$\leftarrow$ ' + labeltop + ' fraction' ,fontsize=fontsize)
    if labelleft is not None:
        ax.set_llabel('$\leftarrow$ ' + labelleft + ' fraction',fontsize=fontsize)
    if labelright is not None:
        ax.set_rlabel(labelright + ' fraction $\\rightarrow$',fontsize=fontsize)
    position = 'tick1'
    ax.taxis.set_label_position(position)
    ax.laxis.set_label_position(position)
    ax.raxis.set_label_position(position)
    # if (tlim is not None) & (llim is not None) & (rlim is not None):
    #     # ax.set_ternary_lim(tlim[0],tlim[1],llim[0],llim[1],rlim[0],rlim[1])
    #     ax.set_tlim(tlim[0],tlim[1])
    #     ax.set_llim(llim[0],llim[1])
    #     ax.set_rlim(rlim[0],rlim[1])
    #     ax.set_aspect('equal')
    if title is not None:
        ax.set_title(title)
    if saveprefix is not None:
        plt.savefig(saveprefix + '.pdf')

def xticks(xticks):
    """Force major x-axis ticks with associated labels
    (useful for log-scale x-axis)
    argument: 
        xticks: lost or array of x-axis ticks
    author:
        Gary Mamon (gam AAT iap.fr)"""
    plt.xticks(xticks)
    xticklabels = [str(xt) for xt in xticks]
    plt.gca().set_xticklabels(xticklabels)
    
def yticks(yticks):
    """Force major y-axis ticks with associated labels
    (useful for log-scale y-axis)
    argument: 
        yticks: lost or array of y-axis ticks
    author:
        Gary Mamon (gam AAT iap.fr)"""
    plt.yticks(yticks)
    yticklabels = [str(yt) for yt in yticks]
    plt.gca().set_yticklabels(yticklabels)
