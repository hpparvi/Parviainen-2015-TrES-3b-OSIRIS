import math as mt
import numpy as np

from numpy import array, ones, inf

from pytransit import MandelAgol as MA
from ldtk import LDPSetCreator
from exotk.priors import UP, NP, PriorSet
from exotk.utils.orbits import as_from_rhop
from exotk.utils.likelihood import ll_normal_es
from lpf_ww import LPFunction
from core import *

from george import GP
from george.kernels import ExpSquaredKernel, ExpKernel

class LPFunctionRN(LPFunction):
    def __init__(self, time, flux, airmass, nthreads=2):
        super(LPFunctionRN, self).__init__(time, flux, airmass, nthreads)
        self.gp = GP(ExpKernel(1))
        self.priors.append(UP(1e-5,1e-3, 'gp_std'))
        self.priors.append(UP(  -5,   6, 'gp_log_inv_length'))
        self.ps = PriorSet(self.priors)
        
    def log_posterior(self,pv):
        if np.any(pv < self.ps.pmins) or np.any(pv>self.ps.pmaxs): 
            return -1e18

        self.gp.kernel = pv[10]**2*ExpKernel(1./10**pv[11])
        self.gp.compute(self.time, pv[5])

        lnlike_ld = self.lds.lnlike_qd(pv[8:10])
        lnlike_lc = self.gp.lnlikelihood((self.flux_o - self.compute_lc_model(pv)))
        lnprior   = self.ps.c_log_prior(pv)
        return lnprior + lnlike_lc + lnlike_ld
