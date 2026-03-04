
import numpy as np
import pandas as pd
from datetime import datetime as dt
def getvarname(var):
    """Variable name string"""
    # author: Eliott Mamon

    # copy globals dictionary
    vars_dict = globals().copy()

    # loop over variables in dictinary and return variable string related to variable name
    lastFoundUnderscoreName = None
    for key in vars_dict:
        if vars_dict[key] is var:
            if key[0] != '_':
                return key
            lastFoundUnderscoreName = key
    return lastFoundUnderscoreName

def variablename(var):
    import itertools
    return [tpl[0] for tpl in 
            itertools.ifilter(lambda x: var is x[1], globals().items())]

def slicer_vectorized(a,start,end):
    """Slice numpy array of strings 
    arguments: 
            a: numpy array of strings
            start: index to start
            end: index to end
    returns: array of strings limited to string indices start to end
    source: divakar and John Zwinck in Stack Overflow
    at https://stackoverflow.com/questions/39042214/how-can-i-slice-each-element-of-a-numpy-array-of-strings"""
    b = a.view((str,1)).reshape(len(a),-1)[:,start:end]
    return np.frombuffer(b.tobytes(),dtype=(str,end-start))

def yearDecimal(dates):
    """approximate decimal year
    argument: date (string array of format 'YYYY-MM-DD' or aray of datetimes)
    returns: decimal year (float array)
    author: Gary Mamon (gam AAT iap.fr)"""
    
    # np arrays of year month and day
    if not isinstance(dates,np.ndarray):
        dates = np.array(dates)
    
    # convert datetime to string
    if isinstance(dates[0],dt):
        dates = pd.to_datetime(dates).strftime('%Y-%m-%d').to_numpy().astype(str)
        
    # extract np arrays of years, months and days-of-month
    y = dates.astype('<U4').astype(int)
    m = slicer_vectorized(dates, 5, 7).astype(int)
    d = slicer_vectorized(dates, 8, 10).astype(int)
    
    # days in year and days in February
    daysinyear = np.where(y % 4 == 0, 366, 365)
    febdays = np.where(daysinyear==366,29,28)
    
    # cumulative days in previous months
    cond = [m==1,m==2,m==3,m==4,m==5,m==6,m==7,m==8,m==9,m==10,m==11,m==12]
    choice = [0,31,31+febdays,62+febdays,92+febdays,123+febdays,153+febdays,
              184+febdays,215+febdays,245+febdays,276+febdays,306+febdays]
    prevcumdays = np.select(cond,choice)
    
    # float-year = int-year + (days-in-past-months+day-of-month-1)/days-in-year
    yearDecimal = y + (prevcumdays + d - 1)/daysinyear
    
    return yearDecimal