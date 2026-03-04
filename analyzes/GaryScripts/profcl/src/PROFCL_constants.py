#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:40:34 2020

@author: gam
"""

import numpy as np
def InitConstants():
    global HUGE,TINY,DEGREE,EXP1,EXP10
    global LN16MINUS2,LN4MINUS1,LN2MINUSHALF,LN3MINUS8OVER9,LN2PI,LOG60
    global METHOD_DICT, MODEL_DICT
    HUGE           = 1.e38
    TINY           = 1.e-8
    DEGREE         = np.pi/180.
    EXP1           = np.exp(1.)
    EXP10          = np.exp(10.)
    LN16MINUS2     = np.log(16.) - 2.
    LN4MINUS1      = np.log(4.) - 1.
    LN2MINUSHALF   = np.log(2.) - 0.5
    LN3MINUS8OVER9 = np.log(3.) - 8./9.
    LN2PI          = np.log(2.*np.pi)
    LOG60          = np.log10(60.)
    METHOD_DICT = {
        "br"       : "Brent",
        "de"       : "Diff-Evol",
        "diffevol" : "Diff-Evol",
        "lbb"      : "L-BFGS-B",
        "nm"       : "Nelder-Mead",
        "p"        : "Powell",
        "s"        : "SLSQP",
        "t"        : "TNC",
        "tnc"      : "TNC",
        "bh"       : "Basin-Hop",
        "sh"       : "SHGO",
        "da"       : "Dual-Anneal"
        }
    MODEL_DICT = {
        "nfw"      : "NFW",
        "n"        : "NFW",
        "cnfw"     : "cNFW",
        "c"        : "cNFW",
        "tnfw"     : "tNFW",
        "t"        : "tNFW",
        "unif"     : "Uniform",
        "u"        : "Uniform"
        }
    
def Convert(abbrev,dictionary,invert=True):
    """Convert abbreviations according to dictionary
    args: 
        abbreviation, dictionary
        where:
            dictionary is a dictionary of:
                "abbrev": "full" 
        invert: set True for case 2
    returns: full name"""
    if abbrev in dictionary.values():
        return abbrev
    elif abbrev in dictionary:
        return dictionary[abbrev]
    else:
        raise ValueError("cannot find abbreviation `" + abbrev + "' among values of dict")
