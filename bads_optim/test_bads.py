#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 17:02:16 2022

@author: edoardo
"""
import matlab
import matlab.engine as eng
import numpy as np
import sys
import seaborn as sbs
import matplotlib.pylab as plt
plt.close('all')
sys.path.append('/Users/edoardo/Work/Code/P-GPCCA/core/')
from data_processing_tools import *
from learnGPParams import all_trial_GPLL

res = np.load('/Users/edoardo/Work/Code/P-GPCCA/inference_syntetic_data/em20iter_sim_150Trials.npz',allow_pickle=True)
dat = res['dat'].all()


binSize = dat.binSize
tau2lam = lambda x,binSize: np.log(binSize**2 / ((x * 1000) ** 2))
lam2tau = lambda lam,binSize: np.exp(-lam/2)/1000*binSize


np.random.rand(4)
tau = dat.init_priorPar[1]['tau']
#tau=np.array([0.61061895, 0.2510141, 0.57811751, 0.85722733, 0.5862096])

lam0 = tau2lam(tau,binSize)




# eng.testPassVar(seq)
param = {'eps':[0.001]*len(lam0),
         'gamma':matlab.double(np.exp(lam0).tolist())}

seq = {'T':[],'VsmGP':[],'xsm':[]}
xdim=3
for tr in dat.trialDur.keys():
    # if tr >2:
    #     break
    seq['T'].append(matlab.double(list(np.arange(dat.trialDur[tr]))))
    
    cov = np.transpose(dat.posterior_inf[tr].cov_k[1],(1,2,0))
    seq['VsmGP'] .append(matlab.double((cov).tolist()))
    seq['xsm'] .append(matlab.double((dat.posterior_inf[tr].mean[1]).tolist()))
eng = eng.start_matlab()
s = eng.genpath('/Users/edoardo/Work/Code/bads')
eng.addpath(s)
ev,grev = eng.allTrial_loglike_RBF_wrt(matlab.double(lam0.tolist()), param, seq, matlab.int64([50]),nargout=2)
f,g = eng.bads_fit(matlab.double(lam0.tolist()), param, seq, matlab.int64([50]),nargout=2)

f = np.array(f)
print('init',tau)
print('object',dat.ground_truth_priorPar[1]['tau'])
print('bads',lam2tau(f,50))
print('L-BGSF-B',dat.priorPar[1]['tau'])

f1= all_trial_GPLL(2 * np.log(((dat.priorPar[1]['tau'] * dat.binSize/1000))), dat, 1, block_trials=150, isGrad=False)
Tsamp = np.arange(50).tolist()
mTsamp = matlab.double(Tsamp)

Kbig,K_big_inv,logdetK = eng.make_K_big_trialBasedDT(param, mTsamp,nargout=3)
Kbig = np.array(Kbig)
K_big_inv = np.array(K_big_inv)
K1, Kbig1, K_big_inv1,  logdet_K_big1 = makeK_big(xdim, tau, 50*50, 50, epsNoise=0.001, T=None, computeInv=True)
K0 = len(tau)
T=50
idx_rot = np.tile(np.arange(0, K0 * T, T), T) + np.arange(K0 * T) // K0

Kbig1 = Kbig1[idx_rot][:,idx_rot]
K_big_inv1 = K_big_inv1[idx_rot][:,idx_rot]














