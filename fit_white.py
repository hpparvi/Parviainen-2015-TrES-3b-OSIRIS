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
from hpgp.gp import GP
from core import *

import matplotlib.pyplot as pl

df = import_white_lc()

class GLPF_WN(object):
    def __init__(self, nthr=2, mode='relative'):
        priors = {'transit_center':  UP(  0.605, 0.615,  'tc'),             
          'period':          NP(  1.306, 1e-7,   'p'),             
          'stellar_density': NP(  1.646, 0.05, 'rho', lims=[1,3]), 
          'radius_ratio':    UP(  0.150, 0.25,   'k'),
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

        self.tm = MandelAgol(nthr=nthr, lerp=True, klims=(0.15,0.25), nk=256)
        self.lpf = LPF(df.time, flux, df.airmass, 
                       tcenter=0.61, tduration=0.06, priors=priors, 
                       tmodel=self.tm)

        if mode == 'raw':
            self.lpf.priors[8] = UP( 0.90, 1.10, 'bl_0')
            self.lpf.priors[9] = UP(-0.20, 0.20, 'am_0')
            self.lpf.ps = PriorSet(self.lpf.priors)

        self.ps = self.lpf.ps

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
        self.gp = GP(self.lpf.time.ravel(), self.lpf.flux.ravel(), 'e')
        
    def set_gp(self, sigma, l):
        print sigma, l
        self.gp(sigma, l)

    def log_posterior(self,pv):
        if np.any(pv < self.ps.pmins) or np.any(pv>self.ps.pmaxs): return -1e18

        self.gp.y[:] = 1e4*(self.lpf.normalize_flux(pv) - self.lpf.compute_lc_model(pv)).ravel()
        self.gp(1e4*pv[5], 0.01)

        log_l = self.gp.log_likelihood()
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

        #if args.noise_model == 'gp':
            #ch_wn = np.load(mc_wn_file)['chains']
            #fc_wn = ch_wn.reshape([-1,lpf.ps.ndim])
            #scatter_means = fc_wn[::50,lpf.lpf.err_slice].mean(0)
            #scatter_stds  = fc_wn[::50,lpf.lpf.err_slice].std(0)

            #for i,(sm,ss) in enumerate(zip(scatter_means, scatter_stds)):
            #    name = lpf.ps.priors[lpf.lpf.err_start+i].name
            #    lpf.ps.priors[lpf.lpf.err_start+i] = NP(sm, ss, name, lims=(0,1))

            #mpv    = np.median(fc_wn, axis=0)
            #resids = 1e4*(lpf.lpf.normalize_flux(mpv) - lpf.lpf.compute_lc_model(mpv))
            #gp = GP(lpf.lpf.time, resids, 'e')
            #pve = fmin(lambda pv: -gp(*pv), [resids.std(), 1], disp=False)
            #lpf.set_gp(*pve)
            #lpf.set_gp(2.5, 0.01)

        if continue_mc:
            population = np.load(mc_file)['chains'][:,-1,:]
            print "Continuing MCMC from the previous run"
        else:
            if args.noise_model == 'wn':
                population = np.load(de_file)['population']
                print "Starting MCMC from the DE population"
            else:
                population = ch_wn[:,-1,:]

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
