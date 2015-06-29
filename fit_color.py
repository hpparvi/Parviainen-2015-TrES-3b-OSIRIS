#!/usr/bin/env python
import numpy as np
import pandas as pd

from numpy import array, ones_like
from numpy.random import normal

from argparse import ArgumentParser
from emcee import EnsembleSampler
from pyde.de import DiffEvol
from lpf_ww import LPFunction as LPFS
from lpf_nw import LPFunction as LPFM
from core import *
     
class PE(object):
    def __init__(self, wfname, n_walkers=100, n_threads=4, ipb=None):
        df_aux = pd.read_hdf('results/light_curves.h5', 'aux')
        df_lc  = pd.read_hdf('results/light_curves.h5', wfname)
        msk = array(df_aux.bad_mask, dtype=np.bool)
        
        if ipb is None:
            self.lpf = LPFM(array(df_aux.mjd-56846+0.5)[msk], array(df_lc)[msk,:], 
                                  df_aux.airmass[msk], n_threads)
        else:
            self.lpf = LPFS(array(df_aux.mjd-56846+0.5)[msk], array(df_lc)[msk,ipb], 
                                  df_aux.airmass[msk], n_threads, filters=[pb_filters_nb[ipb]])
            
        self.de = DiffEvol(self.lpf, self.lpf.ps.bounds, n_walkers, maximize=True, C=0.85, F=0.25)
        self.sampler = EnsembleSampler(self.de.n_pop, self.lpf.ps.ndim, self.lpf) 
                
        qc = self.lpf.lds.coeffs_qd()[0]
        for ipb in range(self.lpf.npb):
            self.de._population[:,8+6*ipb] = normal(qc[ipb,0], 0.05, size=n_walkers) 
            self.de._population[:,9+6*ipb] = normal(qc[ipb,1], 0.05, size=n_walkers)
            
    def run_de(self, n_iter=250):
        for ide, (der,dev) in enumerate(self.de(n_iter)):
            sys.stdout.write('\r{:4d}/{:4d} -- {:8.2f}'.format(ide+1,n_iter,dev))
            sys.stdout.flush()
        print ""
            
    def run_mcmc(self, n_iter=2500, thin=50, irun=0, population=None):
        p0 = population if population is not None else self.de.population
        for ismp, e in enumerate(self.sampler.sample(p0, iterations=n_iter, thin=thin)):
            sys.stdout.write('\r{:2d} -- {:4d}/{:4d}'.format(irun,ismp,n_iter))
            sys.stdout.flush()
        print ""

    def create_dataframe(self, burn=0, thin=1):
        self.df = pd.DataFrame(self.fc(burn,thin), columns=self.lpf.ps.names)
        self.df['k'] = sqrt(self.df['k2'])
        return self.df    

    def fc(self, burn=0, thin=1):
        return self.chain[:,burn::thin,:].reshape([-1, self.chain.shape[2]])

    @property
    def chain(self):
        return self.sampler.chain


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--de-n-iterations', type=int, default=1000)
    ap.add_argument('--mc-n-runs',       type=int, default=3)
    ap.add_argument('--mc-n-iterations', type=int, default=2000)
    ap.add_argument('--mc-thin'  ,       type=int, default=50)
    ap.add_argument('--n-walkers',       type=int, default=500)
    ap.add_argument('--n-threads',       type=int, default=2)
    ap.add_argument('--do-de', action='store_true', default=False)
    ap.add_argument('--do-mc', action='store_true', default=False)
    ap.add_argument('--dont-continue-mc', dest='continue_mc', action='store_false', default=True)
    ap.add_argument('--lc-name', default='final/nb_nomask')
    ap.add_argument('--run-name', default='nomask')

    args = ap.parse_args()
    de_file = join(dir_results,'TrES_3b_color_{:s}_wn_de.npz'.format(args.run_name))
    mc_file = join(dir_results,'TrES_3b_color_{:s}_wn_mc.npz'.format(args.run_name))

    do_de = args.do_de or not exists(de_file)
    do_mc = args.do_mc or not exists(mc_file)
    continue_mc = args.continue_mc and exists(mc_file)

    pe = PE(args.lc_name, args.n_walkers, n_threads=args.n_threads)

    if do_de:
        pes = [PE(args.lc_name, n_walkers=args.n_walkers, n_threads=args.n_threads, ipb=ipb) for ipb in range(16)]
        [p.run_de(100) for p in pes]

        pe.de._population[:,:4] = pes[0].de.population[:,:4]
        for ipb in range(pe.lpf.npb):
            pe.de._population[:,4+6*ipb:4+6*(1+ipb)] = pes[ipb].de.population[:,4:]

        pe.run_de(args.de_n_iterations)
        np.savez(de_file, population=pe.de.population, best_fit=pe.de.minimum_location)

    if do_mc:  
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

        for irun in range(args.mc_n_runs):
            sys.stdout.write('')
            pe.sampler.reset()
            pe.run_mcmc(args.mc_n_iterations, thin=args.mc_thin, irun=irun, population=population)
            np.savez(mc_file, chains=pe.sampler.chain)
            population = pe.chain[:,-1,:].copy()
