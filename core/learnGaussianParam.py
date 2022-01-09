import numpy as np
from scipy.optimize import minimize
from data_processing_tools import approx_grad
from copy import deepcopy

def MStepGauss(x1, mean_post, cov_post):
    """
    M-step updates for trial stacked data\n
    Parameters
    ==========
    :param x1: 
        - Gaussian observations for all trials  
        
            .. math::  x_1 \\in  \\mathbb{R}^{ \\sum_{ \\text{trial}} T_i \\times N}
        

        
    :param mean_post: posterior mean for all trials
    
        .. math:: \mathbb{E}[\mathbf{z} ] \in \mathbb{ R}^{  \sum_{\\text{trial}} T_i \\times K}
        
        
    :param cov_post: 
        - posterior covariance
    
        .. math:: \\text{cov} ( \mathbf{z}) \in \mathbb{ R}^{  \sum_{\\text{trial}} T_i \\times K \\times K}
    
    
    :return:
    """
    mumuT = np.einsum('tj,tk->jk', mean_post, mean_post)
    cov = cov_post.sum(axis=0)
    mu = mean_post.sum(axis=0)
    Ezz = cov + mumuT

    T = mean_post.shape[0]
    xMu = np.einsum('tj,tk->jk',x1,mean_post)
    sX = x1.sum(axis=0)

    # method based on the 2D system
    # t0 = perf_counter()
    # MM = np.block([[Ezz, mu.reshape(-1, 1)], [mu.reshape(1,-1), T]])
    # ee = np.block([xMu, sX.reshape(-1,1)])
    # Cd = np.linalg.solve(MM,ee.T).T
    # Wother = Cd[:, :mu.shape[0]]
    # dother = Cd[:,-1]
    # t1=perf_counter()
    # print(t1-t0)

    W1 = np.linalg.solve(cov + mumuT - np.einsum('i,j->ij',mu,mu)/T,(xMu - np.einsum('i,j->ij', sX, mu) / T).T).T
    d1 = (sX - np.dot(W1, mu)) / T

    # Psi update
    term_0 = np.einsum('ti,tj->ij', x1, x1)
    term_1 = -np.einsum('ij,kj->ik',xMu,W1) #np.einsum('tj,ti,ki->jk',x1,mean_post,W1)
    term_1 = term_1+term_1.T
    term_2 = -np.einsum('j,i->ji', sX, d1)
    term_2 = term_2 + term_2.T
    term_3 = np.einsum('ij,jk,km->im',W1,Ezz.T,W1.T)
    term_4 = np.einsum('j,k,ik->ji', d1, mu,W1)
    term_4 = term_4 + term_4.T
    term_5 = mean_post.shape[0] * np.einsum('j,k->jk', d1, d1)
    R = 1 / T * (term_0+term_1+term_2+term_3+term_4+term_5)
    return W1, d1, R



def logLike_Gauss(x1,W1,d1,Rinv,mean_post,cov_post):
    if len(Rinv.shape) == 1:
        logDet = np.log(Rinv).sum()
        Rinv = np.diag(Rinv)
    else:
        logDet = np.log(np.linalg.eigh(Rinv)[0]).sum()
    Ezz = cov_post.sum(axis=0) + np.einsum('tj,tk->jk',mean_post,mean_post)
    term0 = -0.5*np.einsum('ti,ij,tj',x1,Rinv,x1)
    term1 = np.einsum('tj,jk,km,tm', x1, Rinv, W1, mean_post)#np.einsum('tj,jk,km,tm',x1,Rinv,W1,mean_post)
    term2 = np.einsum('tj,jk,k->t', x1.sum(axis=0), Rinv, d1)#np.einsum('tj,jk,k->t', x1, Rinv, d1).sum()
    term3 = -0.5 * np.trace(np.einsum('ij,jk,kl,lh->ih',W1.T,Rinv,W1,Ezz))#-0.5 * np.trace(np.einsum('ij,jk,kl,lt->it',W1.T,Rinv,W1,Ezz))
    term4 = - np.einsum('j,jk,kw,tw->t', d1, Rinv, W1, mean_post.sum(axis=0))#-np.einsum('j,jk,kw,tw->t',d1,Rinv,W1,mean_post).sum()
    term5 = -0.5 * mean_post.shape[0] * np.einsum('j,jk,k', d1, Rinv, d1)#-0.5 * mean_post.shape[0] * np.einsum('j,jk,k',d1,Rinv,d1)
    term6 = 0.5 * logDet * mean_post.shape[0]
    loss = term0 + term1 + term2 + term3 + term4 + term5 + term6
    return loss

def grag_GaussLL_wrt_Wd(x1,W1,d1,Rinv,mean_post,cov_post):#dLoss_dWd
    mumuT = np.einsum('tj,tk->jk', mean_post, mean_post)
    cov = cov_post.sum(axis=0)
    mu = mean_post.sum(axis=0)
    Ezz = cov + mumuT

    T = mean_post.shape[0]
    xMu = np.einsum('tj,tk->jk', x1, mean_post)
    sX = x1.sum(axis=0)

    # method based on the 2D system
    MM = np.block([[Ezz, mu.reshape(-1, 1)], [mu.reshape(1,-1), T]])
    ee = np.block([xMu, sX.reshape(-1,1)])
    Wd = np.block([W1, d1.reshape(-1,1)])
    grad = np.dot(Wd,MM) - ee
    return grad

def grag_GaussLL_wrt_Rinv(x1,W1,d1,Rinv,mean_post,cov_post):#dLoss_dRinv
    if len(Rinv.shape) == 1:
        Rinv = np.diag(Rinv)

    Ezz = cov_post.sum(axis=0) + np.einsum('tj,tk->jk',mean_post,mean_post)
    mu = mean_post.sum(axis=0)

    xMu = np.einsum('tj,tk->jk', x1, mean_post)
    sX = x1.sum(axis=0)

    term_0 = -0.5 * np.einsum('ti,tj->ij', x1, x1)
    term_1 = np.einsum('ij,kj->ik', xMu, W1)  # np.einsum('tj,ti,ki->jk',x1,mean_post,W1)
    term_1 = 0.5 * (term_1 + term_1.T)
    term_2 = np.einsum('j,i->ji', sX, d1)
    term_2 = 0.5 * (term_2 + term_2.T)
    term_3 = -0.5 * np.einsum('ij,jk,km->im', W1, Ezz.T, W1.T)
    term_4 = -np.einsum('j,k,ik->ji', d1, mu, W1)
    term_4 = 0.5 * (term_4 + term_4.T)
    term_5 = -0.5 * mean_post.shape[0] * np.einsum('j,k->jk', d1, d1)
    grad_loss = term_0 + term_1 + term_2 + term_3 + term_4 + term_5 + 0.5 * mean_post.shape[0] * np.linalg.inv(Rinv)
    return grad_loss

def gaussStepTrial(x1, mean_post, cov_post):
    """
    Compute the compoents needed for the parameter updates of a single 
    (or multiple stacked) trial

    Parameters
    ----------
    x1 : numpy.array of dimension T x N
        The gaussian observations.
    mean_post : numpy.array of dimension T x K
        The posterior mean.
    cov_post : numpy.array of dimension T x K x K
        The posterior covariance at time T.

    Returns
    -------
    sX : numpy.array of dimension N
        the sum of the observations over time.
        
        .. math:: \\sum_t \\mathbf{x}_t
        
    mu : numpy.array of dimension K
    
        The sum over time of the posterior mean
        
        .. math:: \\sum_t \mathbb{E}[\mathbf{z}_t]
        
    xxT : mpy.array of dimension N x N 
        Outer product of the observations summed over time.
        
        .. math:: \\sum_t \mathbf{x}_t \mathbf{x}_t^{\\top}  
        
    xMu : numpy.array of dimension N x K
        outer product of the observation with the posterior mean summed over
        time
    Ezz : numpy.array of dimension K x K
        .. math:: \\sum_t \\text{cov}(\mathbf{z}_t) 

    """
    mumuT = np.einsum('tj,tk->jk', mean_post, mean_post)
    cov = cov_post.sum(axis=0)
    mu = mean_post.sum(axis=0)
    Ezz = cov + mumuT

    xMu = np.einsum('tj,tk->jk', x1, mean_post)
    sX = x1.sum(axis=0)

    xxT = np.einsum('ti,tj->ij', x1, x1)

    return sX, mu, xxT, xMu, Ezz

def updateGauss(data, T, sX, mu, xxT, xMu, Ezz):
    """
    Compute the parameter updates for the Gaussian observation and store it 
    in the parameter dictionary

    Parameters
    ----------
    data : P_GPCCA
        The data container.
        
    T : int
        Total recording duration intime steps.
        
    sX : numpy.array of dimension N
        the sum of the observations over time.
        
        .. math:: \\sum_t \\mathbf{x}_t
        
    mu : numpy.array of dimension K
    
        The sum over time of the posterior mean
        
        .. math:: \\sum_t \mathbb{E}[\mathbf{z}_t]
        
    xxT : mpy.array of dimension N x N 
        Outer product of the observations summed over time.
        
        .. math:: \\sum_t \mathbf{x}_t \mathbf{x}_t^{\\top}  
        
    xMu : numpy.array of dimension N x K
        outer product of the observation with the posterior mean summed over
        time
    Ezz : numpy.array of dimension K x K
        .. math:: \\sum_t \\text{cov}(\mathbf{z}_t) 

    Returns
    -------
    None.

    """
    sXMuT = np.einsum('i,j->ij', sX, mu)
    sMuSMuT = np.einsum('i,j->ij', mu, mu)
    W = np.linalg.solve(Ezz - sMuSMuT / T, (xMu - sXMuT / T).T).T
    d = (sX - np.dot(W, mu)) / T

    term_0 = xxT
    term_1 = -np.einsum('ij,kj->ik', xMu, W)  # np.einsum('tj,ti,ki->jk',x1,mean_post,W1)
    term_1 = term_1 + term_1.T
    term_2 = -np.einsum('j,i->ji', sX, d)
    term_2 = term_2 + term_2.T
    term_3 = np.einsum('ij,jk,km->im', W, Ezz.T, W.T)
    term_4 = np.einsum('j,k,ik->ji', d, mu, W)
    term_4 = term_4 + term_4.T
    term_5 = T * np.einsum('j,k->jk', d, d)
    R = 1 / T * (term_0 + term_1 + term_2 + term_3 + term_4 + term_5)

    data.stimPar['W0'] = W
    data.stimPar['d'] = d
    data.stimPar['PsiInv'] = np.linalg.inv(R)
    return

def full_GaussLL(data):
    # T, sX, mu, _, xMu, Ezz = learn_GaussianParams(data, test=False, isMPI=True)
    T = np.sum(list(data.trialDur.values()))
    W1 = data.stimPar['W0']
    d1 = data.stimPar['d']
    Rinv = data.stimPar['PsiInv']
    N,K = W1.shape
    x1 = np.zeros((T,N))
    mean_post = np.zeros((T,K))
    cov_post = np.zeros((T, K, K))
    t0 = 0
    for tr in data.trialDur.keys():
        x1[t0:t0+data.trialDur[tr]] = data.get_observations(tr)[0]
        mean_post[t0:t0+data.trialDur[tr],:] = data.posterior_inf[tr].mean[0].T
        cov_post[t0:t0 + data.trialDur[tr]] = data.posterior_inf[tr].cov_t[0]
        t0 += data.trialDur[tr]
    logDet = np.log(np.linalg.eigh(Rinv)[0]).sum()
    Ezz = cov_post.sum(axis=0) + np.einsum('tj,tk->jk', mean_post, mean_post)
    term0 = -0.5 * np.einsum('ti,ij,tj', x1, Rinv, x1)
    term1 = np.einsum('tj,jk,km,tm', x1, Rinv, W1, mean_post)  # np.einsum('tj,jk,km,tm',x1,Rinv,W1,mean_post)
    term2 = np.einsum('j,jk,k->', x1.sum(axis=0), Rinv, d1)  # np.einsum('tj,jk,k->t', x1, Rinv, d1).sum()
    term3 = -0.5 * np.trace(np.einsum('ij,jk,kl,lh->ih', W1.T, Rinv, W1, Ezz))  # -0.5 * np.trace(np.einsum('ij,jk,kl,lt->it',W1.T,Rinv,W1,Ezz))
    term4 = - np.einsum('j,jk,kw,w->', d1, Rinv, W1,mean_post.sum(axis=0))  # -np.einsum('j,jk,kw,tw->t',d1,Rinv,W1,mean_post).sum()
    term5 = -0.5 * T * np.einsum('j,jk,k', d1, Rinv,d1)  # -0.5 * mean_post.shape[0] * np.einsum('j,jk,k',d1,Rinv,d1)
    return term0 + term1 + term2 + term3 + term4 + term5

def learn_GaussianParams(data, test=False, isMPI=False):
    """
    This function assumes that the inference step is already completed
    :param data: the cca_input data
    :return:
    """
    sX, mu, xxT, xMu, Ezz = [0] * 5

    for tr in data.trialDur.keys():
        stim, _ = data.get_observations(tr)
        mean_post = data.posterior_inf[tr].mean[0].T
        cov_post = data.posterior_inf[tr].cov_t[0]
        p0,p1,p2,p3,p4 = gaussStepTrial(stim,mean_post,cov_post)
        sX = sX + p0
        mu = mu + p1
        xxT = xxT + p2
        xMu = xMu + p3
        Ezz = Ezz + p4

    T = np.sum(list(data.trialDur.values()))

    if isMPI:
        return T, sX, mu, xxT, xMu, Ezz

    updateGauss(data, T, sX, mu, xxT, xMu, Ezz)

    if test:
        first = True
        for tr in data.trialDur.keys():
            stm, _ = data.get_observations(tr)
            mn = data.posterior_inf[tr].mean[0].T
            cov = data.posterior_inf[tr].cov_t[0]
            if first:
                first = False
                mean_post = deepcopy(mn)
                cov_post = deepcopy(cov)
                stim = deepcopy(stm)
            else:
                mean_post = np.vstack((mean_post, mn))
                cov_post = np.vstack((cov_post, cov))
                stim = np.vstack((stim,stm))
        sX1, mu1, xxT1, xMu1, Ezz1 = gaussStepTrial(stim, mean_post, cov_post)
        W11,d11,R1 = MStepGauss(stim,mean_post,cov_post)
    return
