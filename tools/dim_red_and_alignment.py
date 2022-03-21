#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:41:58 2022

@author: Edoardo Balzani & Pedro Herrera Vidal
"""

import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis

from scipy.linalg  import orthogonal_procrustes
from scipy.spatial import procrustes
from scipy.stats import multivariate_normal
import matplotlib.pylab as plt


def probCCA_MLE(X, Y, latentDim=2):
    # prob CCA with analytic solution
    cov = np.cov(np.hstack((X,Y)).T)
    # standard cca for weight updates
    dim_X = X.shape[1]
    ee, U = np.linalg.eigh(cov[:dim_X, :dim_X])
    ee = np.abs(ee) # should be
    srt_idx = np.argsort(ee)
    ee = ee[srt_idx]
    U = U[:, srt_idx]
    ee = ee[::-1]
    U = U[:, ::-1]
    sqrtY11 = np.dot(np.dot(U, np.diag(np.sqrt(ee))), U.T)
    ee2, U2 = np.linalg.eigh(cov[dim_X:, dim_X:])
    srt_idx = np.argsort(ee2)
    ee2 = ee2[srt_idx]
    U2 = U2[:, srt_idx]
    ee2 = ee2[::-1]
    U2 = U2[:, ::-1]
    sqrtY22 = np.dot(np.dot(U2, np.diag(np.sqrt(ee2))), U2.T)
    M = np.dot(np.dot(np.linalg.pinv(sqrtY11), cov[:dim_X, dim_X:]), np.linalg.pinv(sqrtY22))
    VV1, cancorr, VV2 = np.linalg.svd(M)

    # std canonical directions & mle proj
    U = np.dot(np.linalg.pinv(sqrtY11), VV1)
    V = np.dot(np.linalg.pinv(sqrtY22), VV2.T)
    Mi = np.diag(np.sqrt(cancorr[:latentDim]))
    WX = np.dot(np.dot(cov[:dim_X,:dim_X], U[:,:latentDim]), Mi)
    WY = np.dot(np.dot(cov[dim_X:, dim_X:], V[:,:latentDim]),Mi)
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    PhiX = cov[:dim_X,:dim_X] - np.dot(WX, WX.T)
    PhiY = cov[dim_X:, dim_X:] - np.dot(WY, WY.T)
    E_z_X = np.dot(np.dot(Mi.T, U[:,:latentDim].T), (X - muX).T).T
    E_z_Y = np.dot(np.dot(Mi.T, V[:,:latentDim].T), (Y - muY).T).T
    cov_z_X = np.eye(latentDim) - np.dot(Mi,Mi.T)
    cov_z_Y = np.eye(latentDim) - np.dot(Mi, Mi.T)

    sigma_zx = np.block([WX.T,WY.T])
    s00 = np.dot(WX, WX.T) + PhiX
    s11 = np.dot(WY, WY.T) + PhiY
    s10 = np.dot(WY, WX.T)
    sigma_xx_inv = np.linalg.pinv(np.block([[s00, s10.T], [s10, s11]]))
    muXY = np.hstack((muX, muY))
    E_z_XY = np.einsum('ij,tj->ti', np.dot(sigma_zx, sigma_xx_inv), np.hstack((X,Y)) - muXY)

    cov_z_XY = np.linalg.pinv( np.eye(latentDim) + np.dot(np.dot(WX.T, np.linalg.pinv(PhiX)),WX) +
                              np.dot(np.dot(WY.T, np.linalg.pinv(PhiY)), WY))

    einv = np.abs(np.linalg.eigh(sigma_xx_inv)[0])
    log_det_sigma_xx = np.log(1/einv).sum()
    n_half = X.shape[0]*0.5
    log_like = n_half * (dim_X + Y.shape[1]) * np.log(np.pi*2) + n_half * log_det_sigma_xx +\
               n_half * np.dot(cov.flatten(), sigma_xx_inv.T.flatten())

    return E_z_X, E_z_Y, E_z_XY, cov_z_X, cov_z_Y, cov_z_XY, WX, WY, PhiX, PhiY, muX, muY,cancorr,log_like, np.dot(Mi.T, U[:,:latentDim].T)


def probPCA(X, latentDim):
    """
    Analytic MLE for prob. cca (Bishop Pattern Recognition and Machine Learning,
                                2011 chapter 12.2).

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    latentDim : TYPE
        DESCRIPTION.

    Returns
    -------
    E_z_X : TYPE
        DESCRIPTION.
    cov_z_X : TYPE
        DESCRIPTION.
    W : TYPE
        DESCRIPTION.
    sigma2 : TYPE
        DESCRIPTION.
    mu : TYPE
        DESCRIPTION.
    var_expl : TYPE
        DESCRIPTION.
    log_like : TYPE
        DESCRIPTION.

    """
    assert(latentDim < X.shape[1])
    cov = np.cov(X.T)
    ee, U = np.linalg.eigh(cov)
    srt_idx = np.argsort(ee)
    ee = np.abs(ee[srt_idx])[::-1]
    U = U[:, srt_idx]
    U = U[:, ::-1]
    UM = U[:, :latentDim]
    var_expl = ee[:latentDim]/np.sum(ee)
    LM = np.diag(ee[:latentDim])
    mu = X.mean(axis=0)
    sigma2 = np.mean(ee[latentDim:])
    W = np.dot(UM, np.sqrt(LM - np.eye(LM.shape[0])*sigma2))
    Minv = np.linalg.pinv(np.dot(W.T,W) + np.eye(latentDim)*sigma2)
    E_z_X = np.dot(Minv, np.dot(W.T, (X-mu).T)).T
    cov_z_X = np.linalg.pinv(np.eye(latentDim) + (np.dot(W.T, W)/sigma2))
    C = np.dot(W,W.T) + np.eye(X.shape[1])*sigma2
    log_like = multivariate_normal.logpdf(X, mean=mu, cov=C).sum()
    return E_z_X, cov_z_X, W, sigma2, mu, var_expl, log_like



class LSM_Procrustes():
    """
    Latent alignment
    """
    def __init__(self, populations = ['A', 'B']):
        self.pop_ = populations
        
    def import_data(self, path = 'spikes_sim_3hz_median.npz'):
        temp_ = np.load(path, mmap_mode='r')
        
        self.data_    = {ii: temp_['spk_'+ii] for ii in self.pop_} # timepoints x units
        self.sm_data_ = {ii: temp_['sm_spk_'+ii] for ii in self.pop_}
        
        self.xDim_ = {ii: temp_['spk_'+ii].shape[1] for ii in self.pop_}
        self.T_    = {ii: temp_['spk_'+ii].shape[0] for ii in self.pop_}
    
        
    # overrides imported smoothed data
    def smoothSpikes(self, method = 'sqrt'): # sqrt transform idea from Yu et al. 2009
        self.sm_data_ = {ii: np.sqrt(self.data_[ii]) for ii in self.pop_}
        
    def mean_subtract(self):
        for ii in self.pop_:
            self.sm_data_[ii] -= np.mean(self.sm_data_[ii], axis=0)
            
    def get_LatDim_FA(self, range_ = np.arange(2,10)):
        self.range_ = range_
        self.score_ = {ii: [] for ii in self.pop_}
        self.LogLs_ = {ii: [] for ii in self.pop_}
        self.aic_ = {ii: [] for ii in self.pop_}
        
        for ii in self.pop_: # this shouldn't be two loops
            for dim in range_:
                fa = FactorAnalysis(n_components=dim)
                fa.fit(self.sm_data_[ii])
                self.LogLs_[ii].append(fa.loglike_[-1])
                self.score_[ii].append(fa.score(self.sm_data_[ii]))
                npar = fa.noise_variance_.shape[0] + \
                    np.prod(fa.components_.shape) + fa.mean_.shape[0]
                self.aic_[ii].append(2*npar - 2*fa.loglike_[-1])
                
    def get_LatDim_PPCA(self, range_ = np.arange(2,10)):
        self.range_ = range_
        self.LogLs_ = {ii: [] for ii in self.pop_}
        self.aic_ = {ii: [] for ii in self.pop_}
        
        for ii in self.pop_: # this shouldn't be two loops            
            for dim in range_:
                E_z_X, _, W, sigma2, mu, _, log_like = probPCA(self.sm_data_[ii],dim)
                self.LogLs_[ii].append(log_like)
                npar = np.prod(W.shape) + 1 + mu.shape[0]
                self.aic_[ii].append(2*npar - 2*log_like)
                
    def get_LatDim_PCCA(self, range_ = np.arange(2,10)):
        self.range_ = range_
        self.LogLs_ = []
        self.aic_ = []
        for dim in range_:
            _, _, _, _, _, _,\
                WX, WY, PhiX, PhiY, muX, muY,\
                    _, log_like, _ = \
                        probCCA_MLE(self.sm_data_[self.pop_[0]],
                                    self.sm_data_[self.pop_[1]],dim)
            self.LogLs_.append(log_like)
            npar = np.prod(WX.shape) + np.prod(WY.shape) + muX.shape[0] +\
                muY.shape[0] + np.prod(PhiX.shape) + np.prod(PhiY.shape)
            self.aic_.append(2*npar - 2*log_like)
                
    
    def get_latents_FA(self, zDim = 2): # zDim should be set to argmax(LL)
        self.Z_ = {}
        
        for ii in self.pop_:
            fa = FactorAnalysis(n_components=zDim)
            self.Z_[ii] = fa.fit_transform(self.sm_data_[ii])
            
    def get_latents_PPCA(self, zDim = 2): # zDim should be set to argmax(LL)
        self.Z_ = {}
        
        for ii in self.pop_:
            E_z_X = probPCA(self.sm_data_[ii],zDim)[0] #PPCA_EM(temp_cov, self.xDim_[ii], zDim, self.T_[ii])
            self.Z_[ii] = E_z_X
            
    def get_latents_PCCA(self, zDim = 2): # zDim should be set to argmax(LL)
        assert(len(self.pop_)==2)
        # this works only for 2 pops
        self.Z_ = {}
        tmp = probCCA_MLE(self.sm_data_[self.pop_[0]],self.sm_data_[self.pop_[1]],zDim) #PPCA_EM(temp_cov, self.xDim_[ii], zDim, self.T_[ii])
        self.Z_[self.pop_[0]] = tmp[0]
        self.Z_[self.pop_[1]] = tmp[1]
            
    def procrustean_align(self, method ='regular'): # note that this requires tied latent dim.
        self.Z_align_ = {}
        
        if method == 'orthgonal':
            R, _ = orthogonal_procrustes(self.Z_[self.pop_[0]], self.Z_[self.pop_[1]])
            self.Z_align_[self.pop_[0]] = self.Z_[self.pop_[0]] @ R
            self.Z_align_[self.pop_[1]] = self.Z_[self.pop_[1]]
            
        elif method == 'regular':
            tempA, tempB,dissim = procrustes(self.Z_[self.pop_[0]], self.Z_[self.pop_[1]])
            self.Z_align_[self.pop_[0]] = tempA
            self.Z_align_[self.pop_[1]] = tempB
            self.dissim = dissim
            
    def plot_dim_FA(self):
        f, (ax1, ax2) = plt.subplots(1, 2, facecolor='w', figsize=(8,5))
        for ii in self.pop_:
            ax1.plot(self.range_, self.LogLs_[ii])
            ax2.plot(self.range_, self.score_[ii])
        ax1.set_xlabel('# dim.'); ax1.legend(self.pop_)
        ax1.set_ylabel('LL'); ax2.set_ylabel('Score')
        
    def plot_dim_PPCA(self):
        plt.figure(facecolor='w', figsize=(4,5))
        for ii in self.pop_:
            plt.plot(self.range_, self.LogLs_[ii])
        plt.xlabel('# dim.'); plt.legend(self.pop_); plt.ylabel('LL');
        
    def plot_latents(self):
        plt.figure(facecolor='w', figsize=(5,5))
        for ii in self.pop_:
            plt.plot(self.Z_align_[ii][:, 0], self.Z_align_[ii][:, 1], lw=2);
        plt.xlabel('LD-1'); plt.ylabel('LD-2'); plt.legend(self.pop_)
        

if __name__ == '__main__':
    from scipy.stats import zscore
    #load_dat = 'spikes_sim_3hz_median.npz' # mid rate
    load_dat = 'spikes_sim.npz' # high rate
    plt.close('all')
    FA_align = LSM_Procrustes()
    FA_align.import_data('/Users/edoardo/Work/Code/FF_dimReduction/P-GPCCA_analyze/Area_Interaction_noStim/'+load_dat)
    FA_align.get_LatDim_FA(range_=np.arange(1,12))
    FA_align.get_latents_FA(2)
    FA_align.procrustean_align()

    
    # import stuff
    PCCA_align = LSM_Procrustes()
    PCCA_align.import_data('/Users/edoardo/Work/Code/FF_dimReduction/P-GPCCA_analyze/Area_Interaction_noStim/'+load_dat)
    PCCA_align.get_LatDim_PCCA(range_=np.arange(1,12))
    PCCA_align.get_latents_PCCA(2)
    PCCA_align.procrustean_align()

    
    PPCA_align = LSM_Procrustes()
    PPCA_align.import_data('/Users/edoardo/Work/Code/FF_dimReduction/P-GPCCA_analyze/Area_Interaction_noStim/'+load_dat)
    PPCA_align.get_LatDim_PPCA(range_=np.arange(1,12))
    PPCA_align.get_latents_PPCA(2)
    PPCA_align.procrustean_align()
    
    plt_end = 10
    plt.figure(figsize=(5,3))
    
    plt.subplot(121)
    plt.title('AIC')
    cmFA = plt.get_cmap('Reds')
    cmPPCA = plt.get_cmap('Blues')
    
    plt.plot(range(1,12)[:plt_end],zscore(FA_align.aic_['A'][:plt_end]),color=cmFA(0.4))
    plt.plot(range(1,12)[:plt_end],zscore(FA_align.aic_['B'][:plt_end]),color=cmFA(0.8))
    
    
    plt.plot(range(1,12)[:plt_end],zscore(PPCA_align.aic_['A'][:plt_end]),color=cmPPCA(0.4))
    plt.plot(range(1,12)[:plt_end],zscore(PPCA_align.aic_['B'][:plt_end]),color=cmPPCA(0.8))
    
    plt.plot(range(1,12)[:plt_end], zscore(PCCA_align.aic_[:plt_end]),color='k')
    plt.xlabel('latent dim.')
    plt.ylabel('z-score')
    plt.xticks(range(1,12)[:plt_end:2])

    plt.subplot(122)
    plt.title('Log-likelihood')
    
    plt.plot(range(1,12)[:plt_end],zscore(FA_align.LogLs_['A'][:plt_end]),color=cmFA(0.4))
    plt.plot(range(1,12),zscore(FA_align.LogLs_['B']),color=cmFA(0.8),label='FA')
    
    
    plt.plot(range(1,12)[:plt_end],zscore(PPCA_align.LogLs_['A'][:plt_end]),color=cmPPCA(0.4))
    plt.plot(range(1,12)[:plt_end],zscore(PPCA_align.LogLs_['B'][:plt_end]),color=cmPPCA(0.8),label='PPCA')
    
    plt.plot(range(1,12)[:plt_end], zscore(PCCA_align.LogLs_[:plt_end]),color='k',label='PCCA')
    
    plt.xticks(range(1,12)[:plt_end:2])
    plt.xlabel('latent dim.')
    plt.legend()
    plt.tight_layout()
    
    
    
    
    
    
    