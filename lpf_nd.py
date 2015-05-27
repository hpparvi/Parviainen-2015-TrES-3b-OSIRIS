import math as mt
import numpy as np

from numpy import array, ones, zeros, ones_like, inf, newaxis, sqrt, s_

from pytransit import MandelAgol as MA
from ldtk import LDPSetCreator
from exotk.priors import UP, NP, PriorSet
from exotk.utils.orbits import as_from_rhop
from exotk.utils.likelihood import ll_normal_es
from core import *

class LPFunction(object):
    def __init__(self, time, flux, airmass, nthreads=2):
        self.tm = MA(interpolate=True, klims=(0.15,0.20), nthr=nthreads, nk=512) 
        self.nthr = nthreads

        self.time     = array(time)
        self.flux_o   = array(flux)
        self.airmass  = array(airmass)
        self.npt      = self.flux_o.shape[0]
        self.npb      = self.flux_o.shape[1]
        self._w_bl    = ones_like(flux)
        self._w_ld    = zeros(2*self.npb)
        self._w_pv    = zeros(4+6*self.npb)
        
        sc = LDPSetCreator(teff=(5650,35),logg=(4.58,0.015),z=(-0.19,0.08), filters=[pb_filter_bb]+pb_filters_nb)
        self.lds = sc.create_profiles(500)
        self.lds.resample_linear_z()
        
        self.priors = [UP(  0.605,   0.615,   'tc'),  ##  0  - Transit centre
                       NP(  1.306,   1e-7,     'p'),  ##  1  - Period
                       UP(  1.000,   3.00,   'rho'),  ##  2  - Stellar density
                       UP(  0.000,   0.99,     'b')]  ##  3  - Impact parameter
        
        for ipb in range(self.npb):
            self.priors.extend([
                       UP( .15**2, .20**2, 'k2_{:02d}'.format(ipb)),  ##  4 + 5*i  - planet-star area ratio
                       NP(    1.0,   0.01,  'c_{:02d}'.format(ipb)),  ##  5 + 5*i  - Baseline constant
                       NP(    0.0,   0.01,  'x_{:02d}'.format(ipb)),  ##  6 + 5*i  - Residual extinction coefficient
                       UP(   -1.0,    1.0,  'u_{:02d}'.format(ipb)),  ##  7 + 5*i  - limb darkening u
                       UP(   -1.0,    1.0,  'v_{:02d}'.format(ipb))]) ##  8 + 5*i  - limb darkening v
            
        self.wn_start = len(self.priors)
        for ipb in range(self.npb-1):
            self.priors.append(UP(1e-4,  30e-4,  'e_{:02d}'.format(ipb))) ## wn_start + i - White noise std
            
        self.cr_start = len(self.priors)
        for ipb in range(self.npb-1):
            self.priors.append(UP(0.00,  2.000,  'f_{:02d}'.format(ipb))) ## cr_start + i - Broadband correction
        self.ps = PriorSet(self.priors)
        
        self._sk2 = s_[4:self.wn_start:5]
        self._sc  = s_[5:self.wn_start:5]
        self._sx  = s_[6:self.wn_start:5]
        self._su  = s_[7:self.wn_start:5]
        self._sv  = s_[8:self.wn_start:5]
        self._sn  = s_[self.wn_start:self.wn_start+self.npb]
        self._sf  = s_[self.cr_start:self.cr_start+self.npb]
        
        
    def compute_baseline(self, pv):
        return pv[self._sc]*np.exp(pv[self._sx]*self.airmass[:,newaxis])
    
    
    def compute_transit(self, pv):
        _a  = as_from_rhop(pv[2], pv[1])
        _i  = mt.acos(pv[3]/_a)
        _k  = sqrt(pv[self._sk2])
        self._w_ld[0::2] = pv[self._su]
        self._w_ld[1::2] = pv[self._sv]                 
        return self.tm.evaluate(self.time, _k, self._w_ld, pv[0], pv[1], _a, _i)
    
    
    def compute_lc_model(self, pv):
        return self.compute_baseline(pv)*self.compute_transit(pv)


    def compute_flux_ratios(self, pv, ipb, flux_m):
        b = pv[self.cr_start+ipb-1]
        fro = self.flux_o[:,ipb] / (1.+b*(self.flux_o[:,0]-1.))
        frm = flux_m[:,ipb]      / (1.+b*(flux_m[:,0]-1.))
        return fro, frm
    
    
    def log_posterior(self, pv):
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs):
            return -inf
        
        flux_m = self.compute_lc_model(pv)
        lnlike_lc = ll_normal_es(self.flux_o[:,0], flux_m[:,0], 4e-4)
        lnlike_lr = sum([ll_normal_es(*self.compute_flux_ratios(pv, ipb, flux_m), 
                                      e=pv[self.wn_start+ipb-1]) for ipb in range(1,self.npb)])

        lnlike_ld = self.lds.lnlike_qd(self._w_ld)                
        return self.ps.c_log_prior(pv) + lnlike_lr + lnlike_lc + lnlike_ld

    def __call__(self, pv):
        return self.log_posterior(pv)
