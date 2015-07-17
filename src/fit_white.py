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

df = import_white_lc()

class GLPF_WN(object):
    def __init__(self, nthr=2, mode='relative'):
        priors = {'transit_center':  UP(  0.605, 0.615,  'tc'),             
          'period':          NP(  1.306, 1e-7,   'p'),             
          'stellar_density': NP(  1.646, 0.05, 'rho', lims=[1,3]), 
          'radius_ratio':    UP(  0.150, 0.28,   'k'),
          'baseline':        UP(  0.980, 1.02,  'bl'),
          'ldc_0_0':         UP(     -5,    5,  'ldc_0_0'),
          'ldc_0_1':         UP(     -5,    5,  'ldc_0_1')}
        
        if mode == 'relative':
            flux = df[df.columns[1:2]]
        elif mode == 'raw':
            flux = df[df.columns[2:3]]/df.target_flux[0]
        else:
            print "Unknown flux mode, choose either 'relative' or 'raw'"
            exit()

        self.tm = MandelAgol(nthr=nthr, lerp=True, klims=(0.15,0.28), nk=512)
        self.lpf = LPF(df.time, flux, df.airmass, 
                       tcenter=0.61, tduration=0.06, priors=priors, 
                       tmodel=self.tm)

        if mode == 'raw':
            self.lpf.priors[8] = UP( 0.90, 1.10, 'bl_0')
            self.lpf.priors[9] = UP(-0.20, 0.20, 'am_0')
            self.lpf.ps = PriorSet(self.lpf.priors)

        self.ps = self.lpf.ps

        ## Load the stellar brightness profiles
        ##
        lddata = np.load('data/tres_3_limb_darkening_w.npz')
        self.im = lddata['im'][:-1]
        self.ie = lddata['ie'][:-1]
        self.iv = self.ie**2
        self.mu = lddata['mu'][:-1]
        self.ld0 = fmin(lambda pv: ((self.im - quadratic_law(self.mu, pv))**2).sum(), [0.5, 0.2], disp=False)


    def ld_log_likelihood(self,pv):
        chi_sqr = ((self.im - quadratic_law(self.mu, pv[self.lpf.ldc_slice]))**2/self.iv).sum()
        log_l   = -0.5*self.mu.size*np.log(2*pi) -np.log(self.ie).sum() - 0.5*chi_sqr
        return log_l

        
    def log_posterior(self,pv):
        return self.lpf.log_posterior(pv) + self.ld_log_likelihood(pv)



class GLPF_GP(GLPF_WN):
    def __init__(self, nthr=2, flux_mode='relative'):
        super(GLPF_GP, self).__init__(nthr, flux_mode)
        self.gp = GP(ExpKernel(1))
        self.lpf.priors.append(UP(1e-5,1e-3,'gp_std'))
        self.lpf.priors.append(UP(-5, 6,'gp_log_inv_length'))
        self.lpf.ps = PriorSet(self.lpf.priors)
        self.ps = self.lpf.ps

        
    def log_posterior(self,pv):
        if np.any(pv < self.ps.pmins) or np.any(pv>self.ps.pmaxs): return -1e18
        self.gp.kernel = pv[10]**2*ExpKernel(1./10**pv[11])
        self.gp.compute(self.lpf.time.ravel(), pv[5])

        log_l = self.gp.lnlikelihood((self.lpf.normalize_flux(pv) - self.lpf.compute_lc_model(pv)).ravel())
        log_p = self.ps.c_log_prior(pv)
        return log_p + log_l + self.ld_log_likelihood(pv)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--de-n-iterations', type=int, default=1000)
    ap.add_argument('--mc-n-runs',       type=int, default=3)
    ap.add_argument('--mc-n-iterations', type=int, default=2000)
    ap.add_argument('--mc-thin'  ,       type=int, default=50)
    ap.add_argument('--n-walkers',       type=int, default=100)
    ap.add_argument('--n-threads',       type=int, default=2)
    ap.add_argument('--do-de', action='store_true', default=False)
    ap.add_argument('--do-mc', action='store_true', default=False)
    ap.add_argument('--dont-continue-mc', dest='continue_mc', action='store_false', default=True)
    ap.add_argument('--noise-model', default='wn', choices=['wn','gp'])
    ap.add_argument('--flux-mode', default='relative', choices=['relative','raw'])

    args = ap.parse_args()

    mc_wn_file = join(dir_results,'TrES_3b_white_{:s}_wn_mc.npz'.format(args.flux_mode))
    mc_gp_file = join(dir_results,'TrES_3b_white_{:s}_gp_mc.npz'.format(args.flux_mode))

    de_file = join(dir_results,'TrES_3b_white_{:s}_de.npz'.format(args.flux_mode))
    mc_file = join(dir_results,'TrES_3b_white_{:s}_{:s}_mc.npz').format(args.flux_mode,args.noise_model)

    do_de = args.do_de or not exists(de_file)
    do_mc = args.do_mc or not exists(mc_file)
    continue_mc = args.continue_mc and exists(mc_file)

    if args.noise_model == 'wn':
        lpf = GLPF_WN(args.n_threads, args.flux_mode)
    else:
        lpf = GLPF_GP(args.n_threads, args.flux_mode)

    if do_de:
        de = DiffEvol(lpf.log_posterior, lpf.ps.bounds, args.n_walkers, maximize=True)
        for i in range(2):
            de._population[:,6+i] = normal(lpf.ld0[i], 0.05, size=args.n_walkers)
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
                population = np.load(mc_wn_file)['chains'][:,-1,:]
                population = np.concatenate([population,np.ones([population.shape[0],2])], axis=1)
                population[:,10] = np.random.permutation(population[:,5])
                population[:,11] = np.random.normal(4.5, 0.01, size=population.shape[0])

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
