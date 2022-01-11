#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 21:03:41 2022

@author: edoardo
"""
import matlab
import matlab.engine as eng
import numpy as np



class badsOptim(object):
    def __init__(self,dat):
        print('preparing for bads optim')
        self.eng = eng.start_matlab()
        self.eng.addpath(self.eng.genpath('/Users/edoardo/Work/Code/bads'))
        self.eng.addpath(self.eng.genpath('badsOptim/'))
        self.x = []
        for kk in range(len(dat.zdims)-1):
            self.x += [[]]
            for tr in dat.trialDur.keys():
                xList = dat.get_observations(tr)[1][kk]
                self.x[-1].append(matlab.int32(xList.tolist()))
        # aa= self.eng.testPassVar(self.x)
        
    def bads_optim(self,dat,i_latent):
        
        tau = dat.priorPar[i_latent]['tau']
        binSize = dat.binSize
        tau2lam = lambda x,binSize: np.log(binSize**2 / ((x * 1000) ** 2))
        lam2tau = lambda lam,binSize: np.exp(-lam/2)/1000*binSize
        lam0 = tau2lam(tau,binSize)
        
        param = {'eps':[0.001]*len(lam0),
             'gamma':matlab.double(np.exp(lam0).tolist())}
    
        seq = {'T':[],'VsmGP':[],'xsm':[]}
        for tr in dat.trialDur.keys():
            seq['T'].append(matlab.double(list(np.arange(dat.trialDur[tr]))))
            cov = np.transpose(dat.posterior_inf[tr].cov_k[i_latent],(1,2,0))
            # print(cov.shape)
            seq['VsmGP'] .append(matlab.double((cov).tolist()))
            seq['xsm'] .append(matlab.double((dat.posterior_inf[tr].mean[i_latent]).tolist()))
            
        f,g = self.eng.bads_fitGP(matlab.double(lam0.tolist()), param, seq, matlab.int64([binSize]),nargout=2)
        f = np.array([f])
        g = np.array([g])
        return lam2tau(f, binSize), g

    def bads_optimPoisson(self, dat, i_xVar):
        W0 = matlab.double(dat.xPar[i_xVar]['W0'].tolist())
        W1 = matlab.double(dat.xPar[i_xVar]['W1'].tolist())
        d = matlab.double(dat.xPar[i_xVar]['d'].tolist())
        K0 = dat.xPar[i_xVar]['W0'].shape[1]
        K1 = dat.xPar[i_xVar]['W1'].shape[1]


        mean_post = []
        cov_post = []

        for tr in dat.trialDur.keys():
            cov_00 = dat.posterior_inf[tr].cov_t[0]
            cov_ii = dat.posterior_inf[tr].cov_t[i_xVar+1]
            cov_0i = dat.posterior_inf[tr].cross_cov_t[i_xVar+1]
            T = cov_00.shape[0]
            cov = np.zeros((T,K0+K1,K0+K1))
            cov[:, :K0,:K0] = cov_00
            cov[:, K0:, K0:] = cov_ii
            cov[:, :K0, K0:] = cov_0i
            cov[:, K0:, :K0] = np.transpose(cov_0i,(0,2,1))
            mean = np.hstack((dat.posterior_inf[tr].mean[0].T, dat.posterior_inf[tr].mean[i_xVar+1].T))
            cov_post.append(matlab.double(cov.tolist()))
            mean_post.append(matlab.double(mean.tolist()))
        C,C1,d,f = self.eng.bads_fitPoisson(W0,W1,d,self.x,mean_post,cov_post,matlab.int32([i_xVar+1]),nargout=4)
        return np.array(C),np.array(C1),np.array(d).flatten(),f