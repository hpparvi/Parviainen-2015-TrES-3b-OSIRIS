import sys
import math as m
import numpy as np
import pandas as pd
import seaborn as sb
import pyfits as pf

from os.path import join, exists, basename
from matplotlib.dates import datestr2num
from matplotlib import rc
from numpy import pi, array, exp, abs, sum
from numpy.polynomial.chebyshev import Chebyshev
from scipy.constants import k, G, proton_mass

from exotk.constants import rjup, mjup, rsun, msun

AAOCW, AAPGW = 3.4645669, 7.0866142

class WavelengthSolution(object):
    def __init__(self):
        self.fitted_lines = None
        self.reference_lines = None
        self._cp2w = None
        self._cw2p = None

    def fit(self, fitted_lines, reference_lines):
        self.fitted_lines = fitted_lines
        self.reference_lines = reference_lines
        self._cp2w = Chebyshev.fit(self.fitted_lines, self.reference_lines, 5, domain=[0,2051])
        self._cw2p = Chebyshev.fit(self.reference_lines, self.fitted_lines, 5, domain=[400,1000])
        
    def pixel_to_wl(self, pixels):
        return self._cp2w(pixels)
    
    def wl_to_pixel(self, wavelengths):
        return self._cw2p(wavelengths)

class GeneralGaussian(object):
    def __init__(self, name, c, s, p):
        self.name = name
        self.c = c
        self.s = s
        self.p = p
        
    def __call__(self, x):
        return np.exp(-0.5*np.abs((x-self.c)/self.s)**(2*self.p))
    
class WhiteFilter(object):
    def __init__(self, name, filters):
        self.name = name
        self.filters = filters
        
    def __call__(self, x):
        return np.sum([f(x) for f in self.filters], 0)


def H(T,g,mu=None):
    """Atmospheric scale height [m]"""
    mu = mu or 2.3*proton_mass
    return T*k/(g*mu)

## Stellar parameters
## ------------------
## From Torres et al. 2008
## 
## Note: the stellar density estimate
MSTAR = 0.915*msun # [m]
RSTAR = 0.812*rsun # [m]
TSTAR = 5650       # [K]
TEQ   = 1620       # [K]
LOGGS = 4.4       
LOGGP = 3.452  

dir_data    = 'data'
dir_plots   = 'plots'
dir_results = 'results'

fn_white_lc = '20140707-tres_0003_b-gtc_osiris-wlc-ori.dat'
fn_spect_lc = '20140707-tres_0003_b-gtc_osiris-nblcs-250.dat'

## Matplotlib configuration
rc('figure', figsize=(13,5))
rc(['axes', 'ytick', 'xtick'], labelsize=8)
rc('font', size=6)

rc_paper = {"lines.linewidth": 1,
            'ytick.labelsize': 6.5,
            'xtick.labelsize': 6.5,
            'axes.labelsize': 6.5,
            'figure.figsize':(AAOCW,0.65*AAOCW)}

rc_notebook = {'figure.figsize':(13,5)}

sb.set_context('notebook', rc=rc_notebook)
sb.set_style('white')

## Color definitions
##
c_ob = "#002147" # Oxford blue
c_bo = "#CC5500" # Burnt orange
cp = sb.color_palette([c_ob,c_bo], n_colors=2)+sb.color_palette(name='deep',n_colors=4)
sb.set_palette(cp)

## Potassium and Kalium resonance line centers [nm]
##
wlc_k  = array([766.5,769.9])
wlc_na = array([589.4])

## Narrow-band filters
##
pb_centers    = 540. + np.arange(16)*25
pb_filters_nb = [GeneralGaussian('', c, 12, 20) for c in pb_centers]
pb_filter_bb  = WhiteFilter('white', pb_filters_nb)
