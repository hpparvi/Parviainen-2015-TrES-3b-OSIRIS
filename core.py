import sys
import math as m
import numpy as np
import pandas as pd

try:
    import seaborn as sb
    sb.set_style('white')
except ImportError:
    pass

from os.path import join, exists
from matplotlib.dates import datestr2num
from matplotlib import rc
from numpy import pi, array, exp, abs, sum
from numpy.polynomial.chebyshev import Chebyshev

from exotk.constants import rjup, rsun

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


rc('font', size=6)
rc('axes', labelsize=8)
rc('ytick', labelsize=8)
rc('xtick', labelsize=8)


rstar = 0.829*rsun

dir_data    = 'data'
dir_plots   = 'plots'
dir_results = 'results'

fn_white_lc = '20140707-tres_0003_b-gtc_osiris-wlc-ori.dat'
fn_spect_lc = '20140707-tres_0003_b-gtc_osiris-nblcs-250.dat'

c_ob = "#002147" # Oxford blue
c_bo = "#CC5500" # Burnt orange

pb_centers    = 540. + np.arange(16)*25
pb_filters_nb = [GeneralGaussian('', c, 12, 20) for c in pb_centers]
pb_filter_bb  = WhiteFilter('white', pb_filters_nb)

pb_ld = array(
    [[5300,5425,5550], [5550,5675,5800], [5800,5925,6050], [6050,6175,6300], [6300,6425,6550], [6550,6675,6800], [6800,6925,7050], 
     [7050,7175,7300], [7300,7425,7550], [7550,7675,7800], [7800,7925,8050], [8050,8175,8300], [8300,8425,8550], [8550,8675,8800], 
     [8800,8925,9050], [9050,9175,9300]]) * 1e-1
pbc_ld = pb_ld[:,1]

pb_KI = array(
    [[7020,7110,7200], [7200,7290,7380], [7380,7470,7560], [7560,7650,7740], [7740,7830,7920],
     [7920,8010,8100], [8100,8190,8280], [8280,8370,8460], [8460,8550,8640]]) * 1e-1
pbc_KI = pb_KI[:,1]

pb_NaI = array(
    [[5445,5495,5545], [5545,5595,5645], [5645,5695,5745], [5745,5795,5845], [5845,5895,5945], [5945,5995,6045], [6045,6095,6145], 
     [6145,6195,6245], [6245,6295,6345], [6345,6395,6445], [6445,6495,6545], [6545,6595,6645]]) * 1e-1
pbc_NaI = pb_NaI[:,1]

pb_full = array(
    [[5300.0,5425.0,5550.0], [5550.0,5675.0,5800.0], [5800.0,5925.0,6050.0], [6050.0,6175.0,6300.0],
     [6300.0,6425.0,6550.0], [6550.0,6675.0,6800.0], [6800.0,6925.0,7050.0], [7050.0,7175.0,7300.0],
     [7300.0,7425.0,7550.0], [7550.0,7675.0,7800.0], [7800.0,7925.0,8050.0], [8050.0,8175.0,8300.0],
     [8300.0,8425.0,8550.0], [8550.0,8675.0,8800.0], [8800.0,8925.0,9050.0], [9050.0,9175.0,9300.0]]) * 1e-1
pbc_full = pb_full[:,1]

def import_white_lc():
    cols = ("jd flux target_flux comparison_flux seeing pixpos sgnamp ra dec rotang azimuth "
            "elevation airmass humidity pressure tamb expt targname date_obs ut_start filename inf1 inf2").split()

    dt = np.loadtxt(join(dir_data,fn_white_lc), converters = {17:lambda s:0, 18:datestr2num, 19:lambda s:0, 20:lambda s:0, 21:lambda s:0})
    df = pd.DataFrame(dt[:225,:], columns=cols)
    df['time'] = df.jd - int(df.jd[0])
    return df


def import_spect_lc():
    cols = (["jd"] 
            + ["flux_{:d}".format(i+1) for i in range(16)] 
            + ["flux_rt_{:d}".format(i+1) for i in range(16)] 
            + ["flux_rf_{:d}".format(i+1) for i in range(16)]
            + ("seeing pixpos sgnamp ra dec rotang azimuth elevation airmass humidity "
               "pressure tambient expt targname date ut_start infile inf1 inf2").split())

    dt = np.loadtxt(join(dir_data,fn_spect_lc), converters = {63:datestr2num, 62:lambda s:0, 64:lambda s:0, 65:lambda s:0})
    df = pd.DataFrame(dt, columns=cols)
    df['time'] = df.jd - int(df.jd[0])
    return df, dt
