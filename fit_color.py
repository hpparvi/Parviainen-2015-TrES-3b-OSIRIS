#!/usr/bin/env python
from argparse import ArgumentParser
from scipy.optimize import fmin
from numpy.random import normal, permutation
from emcee import EnsembleSampler
from pyde.de import DiffEvol
from pytransit import MandelAgol
from exotk.priors import UP,NP, PriorSet
from exotk.lpf import SingleTransitMultiColorLPF as LPF
from exotk.utils.limb_darkening import quadratic_law
from core import *

from george import GP
from george.kernels import ExpSquaredKernel, ExpKernel

import matplotlib.pyplot as pl

df, dt = import_spect_lc()

class GLPF_WN(object):
    def __init__(self, nthr=2, mode='relative'):
        fluxes = df[df.columns[1:17]]
        self.npb = fluxes.shape[1]

        priors = {'transit_center':  UP(  0.605, 0.615,  'tc'),             
                  'period':          NP(  1.306, 1e-7,    'p'),             
                  'stellar_density': NP(  1.646, 0.05,  'rho', lims=[1,3]), 
                  'radius_ratio':    UP(  0.150, 0.22,    'k'),
                  'baseline':        UP(  0.980, 1.02,   'bl'),
                  'impact_parameter':UP(  0.600, 0.99,    'b')}
                  #'impact_parameter':NP(  0.827, 0.01, 'b', lims=[0.,1.])}

        for ipb in range(self.npb):
            for ild in range(2):
                key = 'ldc_{}_{}'.format(ipb,ild)
                priors[key] = UP(-5, 5, key)  


        self.tm = MandelAgol(nthr=nthr, lerp=True, klims=(0.15,0.22), nk=512)
        self.lpf = LPF(df.time, fluxes, df.airmass, 
                       tcenter=0.61, tduration=0.06, priors=priors, 
                       tmodel=self.tm)

        self.ps = self.lpf.ps

        ## Load the stellar brightness profiles
        ##
        lddata = np.load('data/tres_3_limb_darkening_c.npz')
        self.im = lddata['im'][:-1,:].T
        self.ie = lddata['ie'][:-1,:].T
        self.iv = self.ie**2
        self.mu = lddata['mu'][:-1]

        mo = self.lpf.mask_oot
        self.ld0 = [fmin(lambda pv: ((im - quadratic_law(self.mu, pv))**2).sum(), [0.5, 0.2], disp=False) for im in self.im]
        self.pbl = [fmin(lambda pv: sum((self.lpf.flux[mo,i]/self.flux_baseline(self.lpf.airmass[mo], pv)-1)**2), [1,0], disp=False) for i in range(self.npb)]


    def flux_baseline(self, airmass, pv):
        return pv[0]/np.exp(pv[1]*airmass)
        

    def ld_log_likelihood(self,pv):
        i_model = quadratic_law(self.mu, pv[self.lpf.ldc_slice].reshape((self.npb,2)))
        chi_sqr = ((self.im - i_model)**2/self.iv).sum()
        log_l   = -0.5*self.im.size*np.log(2*pi) -np.log(self.ie).sum() - 0.5*chi_sqr
        return log_l
        

    def log_posterior(self,pv):
        return self.lpf.log_posterior(pv) + self.ld_log_likelihood(pv)



class GLPF_GP(GLPF_WN):
    def __init__(self, nthr=2, flux_mode='relative'):
        super(GLPF_GP, self).__init__(nthr, flux_mode)
        self.lpf.priors.append(UP(1e-5,1e-2,'gp_std'))
        self.lpf.priors.append(UP(-5, 6,'gp_log_inv_length'))
        self.lpf.ps = PriorSet(self.lpf.priors)
        self.ps = self.lpf.ps
        self.lpf.gp_start = self.lpf.bsl_slices[-1].stop
        self.gp = GP(ExpKernel(1.))

    def log_posterior(self,pv):
        if np.any(pv < self.ps.pmins) or np.any(pv>self.ps.pmaxs): return -1e18
        residuals = self.lpf.normalize_flux(pv) - self.lpf.compute_lc_model(pv)

        log_l = 0.
        mer = pv[self.lpf.err_slice].mean()
        gps = self.lpf.gp_start

        self.gp.kernel = pv[gps]**2*ExpKernel(1./10**pv[gps+1])
        self.gp.compute(self.lpf.time.ravel(), mer)

        for ipb in range(self.npb):
            log_l += self.gp.lnlikelihood(residuals[:,ipb])

        log_p = self.ps.c_log_prior(pv)
        return log_p + log_l + self.ld_log_likelihood(pv)
        

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--de-n-iterations', type=int, default=1000)
    ap.add_argument('--mc-n-runs',       type=int, default=3)
    ap.add_argument('--mc-n-iterations', type=int, default=2000)
    ap.add_argument('--mc-thin'  ,       type=int, default=50)
    ap.add_argument('--n-walkers',       type=int, default=600)
    ap.add_argument('--n-threads',       type=int, default=2)
    ap.add_argument('--do-de', action='store_true', default=False)
    ap.add_argument('--do-mc', action='store_true', default=False)
    ap.add_argument('--dont-continue-mc', dest='continue_mc', action='store_false', default=True)
    ap.add_argument('--noise-model', default='wn', choices=['wn','gp'])

    args = ap.parse_args()

    mc_wn_file = join(dir_results,'TrES_3b_color_wn_mc.npz')
    mc_gp_file = join(dir_results,'TrES_3b_color_gp_mc.npz')

    de_file = join(dir_results,'TrES_3b_color_de.npz')
    mc_file = join(dir_results,'TrES_3b_color_{:s}_mc.npz').format(args.noise_model)

    do_de = args.do_de or not exists(de_file)
    do_mc = args.do_mc or not exists(mc_file)
    continue_mc = args.continue_mc and exists(mc_file)

    if args.noise_model == 'wn':
        lpf = GLPF_WN(args.n_threads)
    else:
        lpf = GLPF_GP(args.n_threads)

    if do_de:
        de = DiffEvol(lpf.log_posterior, lpf.ps.bounds, args.n_walkers, maximize=True)

        bs,ls = lpf.lpf.baseline_start, lpf.lpf.ldc_start
        for ipb in range(lpf.npb):
            ## Create the initial limb darkening coefficient population based on the fitted values
            ##
            for ild in range(2):
                de._population[:,ls+2*ipb+ild] = normal(lpf.ld0[ipb][ild], 0.05, size=args.n_walkers)

            ## Create the initial baseline parameter population based on the fitted values
            ##
            de._population[:,bs+2*ipb  ] = normal(lpf.pbl[ipb][0], 0.001, size=args.n_walkers)
            de._population[:,bs+2*ipb+1] = normal(lpf.pbl[ipb][1], 0.001, size=args.n_walkers)

        for ide, (der,dev) in enumerate(de(args.de_n_iterations)):
            sys.stdout.write('\r{:4d}/{:4d} -- {:8.2f}'.format(ide,args.de_n_iterations,dev))
            sys.stdout.flush()
        print ""
        np.savez(de_file, population=de.population, best_fit=de.minimum_location)

    if do_mc:
        sampler = EnsembleSampler(args.n_walkers, lpf.ps.ndim, lpf.log_posterior)        
  
        if continue_mc:
            population = np.load(mc_file)['chains'][:,-1,:]
            print "Continuing MCMC from the previous run"
        else:
            if args.noise_model == 'wn':
                population = np.load(de_file)['population']
                print "Starting MCMC from the DE population"
            else:
                print "Starting new GP MCMC from the white noise population"
                population = np.load(mc_wn_file)['chains'][:,-1,:]
                population = np.concatenate([population,np.ones([population.shape[0],2])], axis=1)
                population[:,lpf.lpf.gp_start] = np.random.permutation(population[:,lpf.lpf.err_start])
                population[:,lpf.lpf.gp_start+1] = np.random.normal(4.5, 0.01, size=population.shape[0])

        for irun in range(args.mc_n_runs):
            sys.stdout.write('')
            sampler.reset()
            for ismp, e in enumerate(sampler.sample(population, iterations=args.mc_n_iterations, thin=args.mc_thin)):
                sys.stdout.write('\r{:2d} -- {:4d}/{:4d}'.format(irun,ismp,args.mc_n_iterations))
                sys.stdout.flush()
            print ""

            np.savez(mc_file, chains=sampler.chain)
            chains = sampler.chain.copy()
            population = chains[:,-1,:]
