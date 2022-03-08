"""
Core inference functions.
Likelihoods gradients and hessians of all model components are implemented as individual functions and combined in
a single method.
"""
import numpy as np
from time import perf_counter

import scipy.sparse
from scipy.optimize import minimize
from scipy.linalg import block_diag, lapack
import scipy.sparse as sparse
from data_processing_tools import *
import scipy.sparse as sparse
from numba import jit
import csr
from data_processing_tools import block_inv

def GPLogLike(z, Kinv):
    """
    GP prior log-likelihood
    :param z: TxK latent factor
    :param Kinv: KT x KT precision matrix (K must be blocked according to the latent dimensions)
    :return: the prior log likelihood (up ot a constant in z)
    """
    zbar = z.T.reshape(-1, )
    return -0.5*np.einsum('i,ij,j->',zbar,Kinv,zbar, optimize=True)

def grad_GPLogLike(z, Kinv):
    """
    Gradient of the GP prior in z
    :param z:
    :param Kinv:
    :return:
    """
    zbar = z.T.reshape(-1, )
    return -np.dot(Kinv,zbar)

def hess_GPLogLike(z, Kinv):
    """
    Hessian  of the GP prior in z
    :param Kinv:
    :return:
    """
    return -Kinv

def gaussObsLogLike(s, z, C, d, PsiInv):
    """
    Log-likelihood of the gaussian observations s
    :param s: T x D
    :param z: T x K
    :param C: D x K
    :param d: D
    :param Psi: D x D
    :return:
    """

    s_center = s - np.einsum('ij,tj->ti', C, z) - d
    res = np.einsum('tj,ti->ij', s_center, s_center, optimize=False)
    res = np.einsum('ij,ji->', res, PsiInv, optimize=False)
    return -0.5*res

def grad_gaussObsLogLike(s, z, C, d, PsiInv):
    """
    gradient of the gaussian observations
    :param s:
    :param z:
    :param C:
    :param d:
    :param PsiInv:
    :return:
    """
    CTPsiInv = np.dot(C.T, PsiInv)
    CPsiInvC = np.dot(CTPsiInv,C)
    CPsiInvs = np.einsum('ij,tj->ti', CTPsiInv, s)
    normLL = CPsiInvs \
             - np.einsum('ij,tj->ti', CPsiInvC, z) \
             - np.ones(z.shape) * np.einsum('ij,j->i', CTPsiInv, d)

    return normLL.flatten()


def hess_gaussObsLogLike(s, z, C, d, PsiInv,return_blocks=False):
    """
    Hessian of the gaussian observartions
    :param z:
    :param C:
    :param PsiInv:
    :return:
    """
    CTPsiInv = np.dot(C.T, PsiInv)
    CPsiInvC = np.dot(CTPsiInv, C)
    if return_blocks:
        M = np.zeros((z.shape[0],z.shape[1],z.shape[1]))
        M[:] = CPsiInvC
    else:
        M = block_diag(*[CPsiInvC] * z.shape[0])
    return -M

def poissonLogLike(x, z0, z1, W0, W1, d):
    """
    Log-likelihood of the poisson population
    :param x:
    :param z0:
    :param z1:
    :param W0:
    :param W1:
    :param d:
    :return:
    """
    LL = (x * (np.einsum('ij,tj->ti', W1, z1) + np.einsum('ij,tj->ti', W0, z0) + d)).sum() -\
         np.exp(np.einsum('ij,tj->ti', W1, z1) + np.einsum('ij,tj->ti', W0, z0) + d).sum()
    return LL

def grad_poissonLogLike(x, z0, z1, W0, W1, d):
    """
    Gradient of the log-likelihood of the poisson population in z_j
    """
    EXP = np.exp(np.einsum('ij,tj->ti', W0, z0) + np.einsum('ij,tj->ti', W1, z1) + d)
    poissLL_z0 = np.einsum('ij,tj->ti',W0.T,x) - np.einsum('tj,ji->ti',EXP,W0)
    poissLL_z1 = np.einsum('ij,tj->ti', W1.T, x) - np.einsum('tj,ji->ti', EXP, W1)
    return poissLL_z0,poissLL_z1


def hess_poissonLogLike(x, z0, z1, W0, W1, d, return_blocks=False):
    """
    compute the hessian for the poisson Log-likelihood
    :param x:
    :param z0:
    :param z1:
    :param W0:
    :param W1:
    :param d:
    :return:
    """
    EXP = np.exp(np.einsum('ij,tj->ti', W0, z0) + np.einsum('ij,tj->ti', W1, z1) + d)
    T,K0 = z0.shape
    K1 = z1.shape[1]
    precision_z0 = np.zeros((T, K0, K0))
    precision_z1 = np.zeros((T, K1, K1))
    precision_z0z1 = np.zeros((T, K0, K1))
    ## check if it is possible to get rid of the loop with einsum
    for t in range(T):
        precision_z0[t] = precision_z0[t] + np.dot(W0.T * EXP[t], W0)
        precision_z1[t] = precision_z1[t] + np.dot(W1.T * EXP[t], W1)
        precision_z0z1[t] = precision_z0z1[t] + np.dot(W0.T * EXP[t], W1)
    if return_blocks:
        return -precision_z0, -precision_z1, -precision_z0z1
    return block_diag(*(-precision_z0)),block_diag(*(-precision_z1)),block_diag(*(-precision_z0z1))


def PpCCA_logLike(zstack, stim, xList, priorPar, stimPar, xPar, binSize, epsNoise=0.001,useGauss=1):
    # extract dim z0, stim and trial time
    #K0 = priorPar[0]['tau'].sh
    tau0 = priorPar[0]['tau']
    K0 = tau0.shape[0]
    if not stim is None:
        T, stimDim = stim.shape
    else:
        T,_ = xList[0].shape

    # extract z0 and its params
    z0 = zstack[:T*K0].reshape(T, K0)
    K0_big_inv = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

    # compute log likelihood for the stimulus and the GP
    if stim.shape[1] == 0:
        logLike = GPLogLike(z0, K0_big_inv)
    else:
        logLike = useGauss * gaussObsLogLike(stim, z0, stimPar['W0'], stimPar['d'], stimPar['PsiInv']) + GPLogLike(z0, K0_big_inv)

    i0 = K0*T
    for k in range(len(xList)):
        N, K = xPar[k]['W1'].shape
        counts = xList[k].reshape(T, N)
        z = zstack[i0: i0+T*K].reshape(T, K)
        K_big_inv = makeK_big(K, priorPar[k+1]['tau'], None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

        logLike += GPLogLike(z,K_big_inv) + poissonLogLike(counts, z0, z,
                                                           xPar[k]['W0'], xPar[k]['W1'], xPar[k]['d'])
        i0 += K*T
    return logLike


def grad_PpCCA_logLike(zstack, stim, xList, priorPar, stimPar, xPar, binSize, epsNoise=0.001,useGauss=1):
    # extract dim z0, stim and trial time
    #K0 = stimPar['W0'].shape[1]
    tau0 = priorPar[0]['tau']
    K0 = tau0.shape[0]

    if stim.shape[1]==0:
        T, stimDim = stim.shape
    else:
        T, _ = xList[0].shape

    # extract z0 and its params
    z0 = zstack[:T*K0].reshape(T,K0)
    K0_big_inv = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

    # compute log likelihood for the stimulus and the GP
    grad_logLike = np.zeros(zstack.shape)
    grad_z0 = grad_GPLogLike(z0, K0_big_inv)

    grad_z0 = grad_z0.reshape(K0,T).T.flatten()
    if stim.shape[1] != 0:
        grad_z0 = grad_z0 + useGauss*grad_gaussObsLogLike(stim, z0, stimPar['W0'], stimPar['d'], stimPar['PsiInv'])
    grad_logLike[:T*K0] = grad_z0

    i0 = K0*T
    for k in range(len(xList)):
        N, K = xPar[k]['W1'].shape
        counts = xList[k].reshape(T,N)
        z = zstack[i0: i0+T*K].reshape(T, K)
        K_big_inv = makeK_big(K, priorPar[k+1]['tau'], None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]
        grad_z0, grad_z1 = grad_poissonLogLike(counts, z0, z, xPar[k]['W0'], xPar[k]['W1'], xPar[k]['d'])
        grad_z1 = grad_z1.flatten() + grad_GPLogLike(z,K_big_inv).reshape(K,T).T.flatten()
        grad_logLike[:K0*T] = grad_logLike[:K0*T] + grad_z0.flatten()
        grad_logLike[i0: i0 + K * T] = grad_z1
        i0 += K * T
    return grad_logLike



def hess_PpCCA_logLike(zstack, stim, xList, priorPar, stimPar, xPar, binSize, epsNoise=0.001, usePrior=1, useGauss=1):
    """

    :param zstack:
    :param stim:
    :param xList:
    :param priorPar:
    :param stimPar:
    :param xPar:
    :param binSize:
    :param epsNoise:
    :param usePrior: only for debug reasons, remove the temporal dependency to obtain block structure for the post cov
    :return:
    """
    # extract dim z0, stim and trial time
    #K0 = stimPar['W0'].shape[1]
    tau0 = priorPar[0]['tau']
    K0 = tau0.shape[0]

    if stim.shape[1] != 0:
        T, stimDim = stim.shape
    else:
        T, _ = xList[0].shape

    # extract z0 and its params
    z0 = zstack[:T*K0].reshape(T, K0)
    K0_big_inv = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]
    # compute log likelihood for the stimulus and the GP
    hess_logLike = np.zeros([zstack.shape[0]]*2, dtype=float)
    hess_z0 = hess_GPLogLike(z0, K0_big_inv)
    idx_rot = np.tile(np.arange(0, K0*T, T), T) + np.arange(K0*T)//K0
    if stim.shape[1] == 0:
        hess_logLike[:T * K0, :T * K0] = usePrior * hess_z0[idx_rot, :][:, idx_rot]
    else:
        hess_logLike[:T*K0,:T*K0] = usePrior*hess_z0[idx_rot,:][:,idx_rot] + useGauss*hess_gaussObsLogLike(stim,z0,stimPar['W0'],
                                                                                     stimPar['d'],stimPar['PsiInv'])
    i0 = K0*T
    for k in range(len(xList)):
        N, K = xPar[k]['W1'].shape
        counts = xList[k].reshape(T, N)
        z = zstack[i0: i0+T*K].reshape(T, K)
        K_big_inv = makeK_big(K, priorPar[k+1]['tau'], None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

        hess_z0, hess_z1, hess_z0z1 = hess_poissonLogLike(counts, z0, z, xPar[k]['W0'], xPar[k]['W1'], xPar[k]['d'])
        tmp = hess_GPLogLike(z, K_big_inv)
        idx_rot = np.tile(np.arange(0, K * T, T), T) + np.arange(K * T) // K
        hess_z1 = hess_z1 + usePrior*tmp[idx_rot,:][:,idx_rot]
        hess_logLike[:T*K0, :T*K0] = hess_logLike[:T*K0, :T*K0] + hess_z0
        hess_logLike[i0: i0 + K * T, i0: i0 + K * T] = hess_z1
        hess_logLike[:K0 * T, i0: i0 + K * T] = hess_z0z1
        hess_logLike[i0: i0 + K * T, :K0 * T] = hess_z0z1.T

        i0 += K * T
    return hess_logLike

def inferTrial(data, trNum, zbar=None, useGauss=1, returnLogDetPrecision=False,remove_neu_dict=None,
               savepath=None,rank=None):
    """
    Laplace inference for an individual trial
    :param data: P_GPCCA
        The P-GPCCA input 
    :param trNum: int
    :param zbar: 
    :return: 
    """
    # retrive outputs
    stim, xList = data.get_observations(trNum)
    if savepath and (rank == 0):
        with open(savepath,'a') as fh:
            string = 'trial %d obs extracted\n'%trNum
            fh.write(string)
            fh.close()

    priorPar = data.priorPar
    stimPar = data.stimPar
    xPar = deepcopy(data.xPar)
    if savepath and (rank == 0):
        with open(savepath, 'a') as fh:
            string = 'trial %d params extracted\n' % trNum
            fh.write(string)
            fh.close()

    if not remove_neu_dict is None:
        for k in range(len(xList)):
            W0 = xPar[k]['W0']
            W1 = xPar[k]['W1']
            d = xPar[k]['d']

            keep_neu = np.ones(xList[0].shape[1], dtype=bool)
            if k in remove_neu_dict.keys():
                keep_neu[remove_neu_dict[k]] = False
                xList[k] = xList[k][:, keep_neu]
                W0 = W0[keep_neu]
                W1 = W1[keep_neu]
                d = d[keep_neu]

            xPar[k]['W0'] = W0
            xPar[k]['W1'] = W1
            xPar[k]['d'] = d


    if zbar is None:
        zdim = 0
        for priorP in priorPar:
            zdim += priorP['tau'].shape[0]

        zbar = np.random.normal(size=zdim*stim.shape[0])*0.01

    # create the lambda function for the numerical MAP optimization
    func = lambda z: -PpCCA_logLike(z, stim, xList, priorPar=priorPar, stimPar=stimPar, xPar=xPar,
                  binSize=data.binSize, epsNoise=data.epsNoise,useGauss=useGauss)
    grad_fun = lambda z: -grad_PpCCA_logLike(z, stim, xList, priorPar=priorPar, stimPar=stimPar, xPar=xPar,
                  binSize=data.binSize, epsNoise=data.epsNoise,useGauss=useGauss)
    #dispFlag = (trNum == 706) or (trNum == 514)
    dispFlag = False
    if savepath and (rank == 0):
        with open(savepath, 'a') as fh:
            string = 'trial %d zbar.shape %s\n' %(trNum, str(zbar.shape))
            fh.write(string)
            fh.close()
    try:
        res = minimize(func, zbar, jac=grad_fun, method='L-BFGS-B',options={'disp':dispFlag})
    except Exception as e:
        if savepath and (rank == 0):
            with open(savepath, 'a') as fh:
                string = 'trial %d exception: '%trNum + e + '\n'
                fh.write(string)
                fh.close()

    if not res.success:
        print('unable to find MAP for trial', trNum)

    zbar = res.x
    precision = -(hess_PpCCA_logLike(zbar, stim, xList, priorPar=priorPar, stimPar=stimPar, xPar=xPar,
                  binSize=data.binSize, epsNoise=data.epsNoise,useGauss=useGauss))
    if savepath and (rank == 0):
        with open(savepath, 'a') as fh:
            string = 'trial %d precison computed \n' % trNum
            fh.write(string)
            fh.close()

    if returnLogDetPrecision:
        lgdet = logDetHessBlock(precision, data.zdims, data.trialDur[trNum])
        return lgdet
    laplAppCov = invertHessBlock(precision, data.zdims, data.trialDur[trNum])
    if savepath and (rank == 0):
        with open(savepath, 'a') as fh:
            string = 'trial %d precison invert precison, ok \n' % trNum
            fh.write(string)
            fh.close()
    return zbar, laplAppCov

def multiTrialInference(data, plot_trial=False, trial_list=None, return_list_post=False, useGauss=1,
                        returnLogDetPrecision=False, remove_neu_dict=None,savepath=None, rank=None):
    """
    Laplace inference for all trials and store the result in the data structure.
    :param data: CCA_input_data
        - the whole input data (or a subset)
    :return:
        - None
    """
    if 'posterior_inf' not in data.__dict__.keys():
        data.posterior_inf = {}
    if return_list_post:
        list_mean_post = []
        list_cov_post = []
    if trial_list is None:
        trial_list = list(data.trialDur.keys())
    if savepath and (rank == 0):
        with open(savepath,'a') as fh:
            string = 'multitrial inf start\n'
            fh.write(string)
            fh.close()
    if returnLogDetPrecision:
        logDetPrecision = []
    cnt = 1
    for tr in trial_list:
        if plot_trial:
            print('infer trial: %d/%d'%(cnt,len(trial_list)))
            if savepath and (rank == 0):
                with open(savepath, 'a') as fh:
                    string = 'infer trial: %d/%d\n'%(cnt,len(trial_list))
                    fh.write(string)
                    fh.close()
        cnt += 1
        if tr not in data.posterior_inf.keys():
            zbar = None
        else:
            # reconstruct zbar
            zbar = np.zeros(np.sum(data.zdims) * data.trialDur[tr])
            i0 = 0
            for k in range(len(data.zdims)):
                zbar[i0:i0 + data.zdims[k] * data.trialDur[tr]] = data.posterior_inf[tr].mean[k].T.flatten()
                i0 += data.zdims[k] * data.trialDur[tr]
        if savepath and (rank == 0):

            with open(savepath, 'a') as fh:
                string = 'reconstructed zbar\n'
                fh.write(string)
                fh.close()
        # set all the attributes related to trial as dictionaries
        T = data.trialDur[tr]
        if returnLogDetPrecision:
            logDetPrecision.append(inferTrial(data, tr, zbar=zbar, useGauss=useGauss, returnLogDetPrecision=returnLogDetPrecision,remove_neu_dict=remove_neu_dict))
        else:
            meanPost, covPost = inferTrial(data, tr, zbar=zbar, useGauss=useGauss,remove_neu_dict=remove_neu_dict,
                                           savepath=savepath,rank=rank)
            if savepath and (rank == 0):
                with open(savepath, 'a') as fh:
                    string = 'inference ok\n'
                    fh.write(string)
                    fh.close()
            if return_list_post:
                list_mean_post.append(meanPost)
                list_cov_post.append(covPost)
            # retrive the K x T x T submarix of the posterior cov and the K x T mean for all the latent variables
            # this will be used for the GP proir time constant learning
            mean_k, cov_ii_k = parse_fullCov_latDim(data, meanPost, covPost, T)
            if savepath and (rank == 0):
                with open(savepath, 'a') as fh:
                    string = 'parsing ok\n'
                    fh.write(string)
                    fh.close()

            # retrive the T x K x K  covariance  and Tx K0 x K cross-cov used in the learning of the observation
            # parameters
            _, cov_ii_t, cov_0i_t = parse_fullCov(data, meanPost, covPost, T)

            # create the structure containing the results
            data.posterior_inf[tr] = emptyStruct()
            data.posterior_inf[tr].mean = mean_k
            data.posterior_inf[tr].cov_t = cov_ii_t
            data.posterior_inf[tr].cross_cov_t = cov_0i_t
            data.posterior_inf[tr].cov_k = cov_ii_k
            if savepath and (rank == 0):
                with open(savepath, 'a') as fh:
                    string = 'storing ok\n'
                    fh.write(string)
                    fh.close()
    if return_list_post:
        return list_mean_post,list_cov_post
    if returnLogDetPrecision:
        return logDetPrecision
    return



#
# if __name__ == '__main__':
#
#     from gen_synthetic_data import *
#     T = 50
#     data = dataGen(5,T=T)
#     sub = data.cca_input.subSampleTrial(np.arange(1,4))
#     multiTrialInference(sub)
#     dat = data.cca_input
#     # test factorized
#     trNum = 1
#     stim, xList = dat.get_observations(trNum)
#     func = lambda zbar: factorized_logLike(dat,trNum,stim, xList, zbar=zbar)
#     grad_func = lambda zbar: grad_factorized_logLike(dat, trNum, stim, xList, zbar=zbar)[0]
#     hess_func = lambda zbar: hess_factorized_logLike(dat, trNum, stim, xList, zbar=zbar)
#
#
