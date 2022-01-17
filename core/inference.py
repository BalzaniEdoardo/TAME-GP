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


def PpCCA_logLike(zstack, stim, xList, priorPar, stimPar, xPar, binSize, epsNoise=0.001):
    # extract dim z0, stim and trial time
    K0 = stimPar['W0'].shape[1]
    tau0 = priorPar[0]['tau']
    T, stimDim = stim.shape

    # extract z0 and its params
    z0 = zstack[:T*K0].reshape(T, K0)
    K0_big_inv = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

    # compute log likelihood for the stimulus and the GP
    logLike = gaussObsLogLike(stim, z0, stimPar['W0'], stimPar['d'], stimPar['PsiInv']) + GPLogLike(z0, K0_big_inv)

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


def grad_PpCCA_logLike(zstack, stim, xList, priorPar, stimPar, xPar, binSize, epsNoise=0.001):
    # extract dim z0, stim and trial time
    K0 = stimPar['W0'].shape[1]
    tau0 = priorPar[0]['tau']
    T, stimDim = stim.shape

    # extract z0 and its params
    z0 = zstack[:T*K0].reshape(T,K0)
    K0_big_inv = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

    # compute log likelihood for the stimulus and the GP
    grad_logLike = np.zeros(zstack.shape)
    grad_z0 = grad_GPLogLike(z0, K0_big_inv)

    grad_z0 = grad_z0.reshape(K0,T).T.flatten()
    grad_z0 = grad_z0 + grad_gaussObsLogLike(stim, z0, stimPar['W0'], stimPar['d'], stimPar['PsiInv'])
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



def hess_PpCCA_logLike(zstack, stim, xList, priorPar, stimPar, xPar, binSize, epsNoise=0.001, usePrior=1):
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
    K0 = stimPar['W0'].shape[1]
    tau0 = priorPar[0]['tau']
    T, stimDim = stim.shape

    # extract z0 and its params
    z0 = zstack[:T*K0].reshape(T, K0)
    K0_big_inv = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]
    # compute log likelihood for the stimulus and the GP
    hess_logLike = np.zeros([zstack.shape[0]]*2, dtype=float)
    hess_z0 = hess_GPLogLike(z0, K0_big_inv)
    idx_rot = np.tile(np.arange(0, K0*T, T), T) + np.arange(K0*T)//K0
    hess_logLike[:T*K0,:T*K0] = usePrior*hess_z0[idx_rot,:][:,idx_rot] + hess_gaussObsLogLike(stim,z0,stimPar['W0'],
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

def inferTrial(data, trNum, zbar=None):
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
    priorPar = data.priorPar
    stimPar = data.stimPar
    xPar = data.xPar

    if zbar is None:
        zdim = 0
        for priorP in priorPar:
            zdim += priorP['tau'].shape[0]

        zbar = np.random.normal(size=zdim*stim.shape[0])*0.01

    # create the lambda function for the numerical MAP optimization
    func = lambda z: -PpCCA_logLike(z, stim, xList, priorPar=priorPar, stimPar=stimPar, xPar=xPar,
                  binSize=data.binSize, epsNoise=data.epsNoise)
    grad_fun = lambda z: -grad_PpCCA_logLike(z, stim, xList, priorPar=priorPar, stimPar=stimPar, xPar=xPar,
                  binSize=data.binSize, epsNoise=data.epsNoise)

    res = minimize(func, zbar, jac=grad_fun, method='L-BFGS-B')
    if not res.success:
        print('unable to find MAP for trial', trNum)

    zbar = res.x
    precision = -(hess_PpCCA_logLike(zbar, stim, xList, priorPar=priorPar, stimPar=stimPar, xPar=xPar,
                  binSize=data.binSize, epsNoise=data.epsNoise))
    laplAppCov = invertHessBlock(precision, data.zdims, data.trialDur[trNum])

    return zbar, laplAppCov

def multiTrialInference(data, plot_trial=False, trial_list=None, return_list_post=False):
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
    for tr in trial_list:
        if plot_trial:
            print('infer trial: %d/%d'%(tr,len(data.trialDur.keys())))
        if tr not in data.posterior_inf.keys():
            zbar = None
        else:
            # reconstruct zbar
            zbar = np.zeros(np.sum(data.zdims) * data.trialDur[tr])
            i0 = 0
            for k in range(len(data.zdims)):
                zbar[i0:i0 + data.zdims[k] * data.trialDur[tr]] = data.posterior_inf[tr].mean[k].T.flatten()
                i0 += data.zdims[k] * data.trialDur[tr]

        # set all the attributes related to trial as dictionaries
        T = data.trialDur[tr]
        meanPost, covPost = inferTrial(data, tr, zbar=zbar)
        if return_list_post:
            list_mean_post.append(meanPost)
            list_cov_post.append(covPost)
        # retrive the K x T x T submarix of the posterior cov and the K x T mean for all the latent variables
        # this will be used for the GP proir time constant learning
        mean_k, cov_ii_k = parse_fullCov_latDim(data, meanPost, covPost, T)

        # retrive the T x K x K  covariance  and Tx K0 x K cross-cov used in the learning of the observation
        # parameters
        _, cov_ii_t, cov_0i_t = parse_fullCov(data, meanPost, covPost, T)

        # create the structure containing the results
        data.posterior_inf[tr] = emptyStruct()
        data.posterior_inf[tr].mean = mean_k
        data.posterior_inf[tr].cov_t = cov_ii_t
        data.posterior_inf[tr].cross_cov_t = cov_0i_t
        data.posterior_inf[tr].cov_k = cov_ii_k
    if return_list_post:
        return list_mean_post,list_cov_post
    return

def factorized_logLike(dat, trNum, stim, xList, zbar=None,idx_sort=None,rev_idx_sort=None):
    # extract latent init if zbar not given
    T = dat.trialDur[trNum]
    sumK = np.sum(dat.zdims)
    if (zbar is None) and ('posterior_inf' not in dat.__dict__.keys()):
        zbar = np.zeros(T * sumK)
    elif zbar is None:
        zbar = np.zeros(T * sumK)
        # start orderig it [K0*T, K1 *T, ... ]
        i0 = 0
        for j in range(len(dat.zdims)):
            mn = dat.posterior_inf[trNum].mean[j].T  # T x K
            zbar[i0: i0 + dat.zdims[j] * T] = mn.flatten()
            i0 += dat.zdims[j] * T
    # else:
    #     if idx_sort is None:
    #         rev_idx_sort = rev_sortGradient_idx(T,dat.zdims)
    #     zbar = zbar[rev_idx_sort]

    stimPar = dat.stimPar
    xPar = dat.xPar

    # stim, xList = dat.get_observations(trNum)

    # extract dim z0, stim and trial time
    K0 = stimPar['W0'].shape[1]
    T, stimDim = stim.shape

    # extract z0 and its params
    z0 = zbar[:T*K0].reshape(T, K0)

    # compute log likelihood for the stimulus and the GP
    logLike = gaussObsLogLike(stim, z0, stimPar['W0'], stimPar['d'], stimPar['PsiInv'])

    i0 = K0*T
    for k in range(len(xList)):
        N, K = xPar[k]['W1'].shape
        counts = xList[k].reshape(T, N)
        z = zbar[i0: i0+T*K].reshape(T, K)

        logLike += poissonLogLike(counts, z0, z, xPar[k]['W0'], xPar[k]['W1'], xPar[k]['d'])
        i0 += K*T
    return logLike



def grad_factorized_logLike(dat, trNum, stim, xList, zbar=None, idx_sort=None,
                            rev_idx_sort=None,useGauss=1,usePoiss=1):

    # vec_time = []
    # vec_latent = []
    T = dat.trialDur[trNum]
    sumK = np.sum(dat.zdims)

    # initialize zbar if none
    if (zbar is None) and ('posterior_inf' not in dat.__dict__.keys()):
        zbar = np.zeros(T * sumK)
    elif zbar is None:
        zbar = np.zeros(T * sumK)
        # start orderig it [K0*T, K1 *T, ... ]
        i0 = 0
        for j in range(len(dat.zdims)):
            mn = dat.posterior_inf[trNum].mean[j]  # K x T
            zbar[i0: i0 + dat.zdims[j] * T] = mn.flatten()  # here you have stacked [z_{0,1:T},z_{1,1:T}, ...]
            i0 += dat.zdims[j] * T
    else:
        if rev_idx_sort is None:
            rev_idx_sort = sortGradient_idx(T, dat.zdims, isReverse=True)
        zbar = zbar[rev_idx_sort]

    # retrive stim par
    C = dat.stimPar['W0']
    d = dat.stimPar['d']
    PsiInv = dat.stimPar['PsiInv']
    _, K0 = C.shape


    # extract grad z0
    grad_factorized = np.zeros(T*sumK)
    grad_z0 = useGauss * grad_gaussObsLogLike(stim,zbar[:T*K0].reshape(T,K0), C, d, PsiInv)
    grad_factorized[:K0*T] = grad_z0.flatten()#.reshape(T,K0).T.flatten()
    # vec_time = np.hstack([vec_time, np.repeat(np.arange(T),K0)])
    # vec_latent = np.hstack([vec_latent, np.repeat(np.ones(K0)*0,T)])
    i0 = K0*T
    for j in range(len(xList)):
        x = xList[j]
        W0 = dat.xPar[j]['W0']
        W1 = dat.xPar[j]['W1']
        d1 = dat.xPar[j]['d']
        K = W1.shape[1]
        z0 = zbar[:T*K0].reshape(T, K0)
        zj = zbar[i0: i0+K*T].reshape(T, K)
        tmp_z0, grad_z1 = grad_poissonLogLike(x,z0,zj,W0,W1,d1)
        # vec_time = np.hstack([vec_time, np.repeat(np.arange(T), K)])
        # vec_latent = np.hstack([vec_latent, np.repeat(np.ones(K) * (j+1), T)])

        grad_factorized[:K0*T] = grad_factorized[:K0*T] + usePoiss * tmp_z0.flatten()#.T.flatten()
        grad_factorized[i0: i0+K*T] = usePoiss * grad_z1.flatten()#.T.flatten()

        i0 += K * T

    if idx_sort is None:
        idx_sort = sortGradient_idx(T, dat.zdims, isReverse=False)
        # idx_sort = np.arange(T*sumK,dtype=int)
    return grad_factorized[idx_sort], idx_sort, rev_idx_sort


def hess_factorized_logLike(dat, trNum, stim, xList, zbar=None, idx_sort=None,
                            rev_idx_sort=None, indices=None, indptr=None, inverse=False):
    # retrive data
    T = dat.trialDur[trNum]
    sumK = np.sum(dat.zdims)

    # initialize zbar if none
    if (zbar is None) and ('posterior_inf' not in dat.__dict__.keys()):
        zbar = np.zeros(T * sumK)
    elif zbar is None:
        zbar = np.zeros(T * sumK)
        # start orderig it [K0*T, K1 *T, ... ]
        i0 = 0
        for j in range(len(dat.zdims)):
            mn = dat.posterior_inf[trNum].mean[j]  # K x T
            zbar[i0: i0 + dat.zdims[j] * T] = mn.flatten()  # here you have stacked [z_{0,1:T},z_{1,1:T}, ...]
            i0 += dat.zdims[j] * T
    else:
        if rev_idx_sort is None:
            rev_idx_sort = sortGradient_idx(T, dat.zdims, isReverse=True)
        zbar = zbar[rev_idx_sort]


    # retrive stim par
    C = dat.stimPar['W0']
    d = dat.stimPar['d']
    PsiInv = dat.stimPar['PsiInv']
    _, K0 = C.shape

    # extract grad z0
    z0 = zbar[:T * K0].reshape(T, K0)

    # compute log likelihood for the stimulus
    i0 = K0 * T
    ii0 = K0
    if not inverse:
        H = np.zeros([T, sumK, sumK], dtype=float)
        H[:, :K0, :K0] = hess_gaussObsLogLike(stim, z0, C, d, PsiInv, return_blocks=True)
    else:
        inverseBlocks = []
        corssBlocks = []#np.zeros((K0,T*(sumK-K0)),dtype=np.float64)
        A = hess_gaussObsLogLike(stim, z0, C, d, PsiInv, return_blocks=True)

    for k in range(len(xList)):
        N, K = dat.xPar[k]['W1'].shape
        counts = xList[k].reshape(T, N)
        z = zbar[i0: i0 + T * K].reshape(T, K)
        hess_z0, hess_z1, hess_z0z1 = hess_poissonLogLike(counts, z0, z, dat.xPar[k]['W0'], dat.xPar[k]['W1'],
                                                          dat.xPar[k]['d'],return_blocks=True)

        if not inverse:
            H[:, :K0, :K0] = H[:, :K0, :K0] + hess_z0
            H[:, ii0:ii0+K, ii0:ii0+K] = hess_z1
            H[:, :K0, ii0:ii0 + K] = hess_z0z1
            H[:, ii0:ii0+K, :K0] = np.transpose(hess_z0z1,(0,2,1))

        else:
            inverseBlocks.append(block_inv(hess_z1))
            corssBlocks.append(hess_z0z1)
            A = A + hess_z0

        i0 += K * T
        ii0 += K
    if inverse:
        H = invertLoop(inverseBlocks, corssBlocks, A)

    # create template for csr (use full blocks in case the inverse is computed)
    if indptr is None:
        mn = H.min()
        # make sure there are no unwanted zeros
        H = H - mn + 1
        spHess = sparse.block_diag(H, format='csr')
        indices = spHess.indices
        indptr = spHess.indptr
        # revert transform
        spHess.data = spHess.data + mn - 1
        # numba compatible
        spHess = csr.CSR.from_scipy(spHess)
    else:
        spHess = sparse.csr_matrix((H.flatten(), indices, indptr))
        # numba compatible
        spHess = csr.CSR.from_scipy(spHess)

    return spHess, indices, indptr


jit(nopython=True)
def invertLoop(Binvs,Cs,As):
    sumK = 0
    for k in range(len(Binvs)):
        block = Binvs[k]
        sumK += block.shape[1]
        T = block.shape[0]

    K0 = As.shape[1]
    sumK += K0
    M = np.zeros((T, sumK, sumK), dtype=np.float64)
    #MM = np.zeros((T, sumK, sumK), dtype=np.float64)
    for t in range(T):
        #btmp = []
        A = As[t]
        CBinv = np.zeros((K0, sumK-K0),dtype=np.float64)
        i0 = 0
        C_all = np.zeros((K0, sumK-K0))
        Binv_all = np.zeros((sumK-K0, sumK-K0))
        for k in range(len(Binvs)):
            B = Binvs[k]
            Bt = B[t]
            #btmp.append(np.linalg.inv(Bt))
            C = Cs[k]
            Ct = C[t]
            K = Bt.shape[0]
            CBinv[:, i0:i0+K] = np.dot(Ct,Bt)
            C_all[:, i0:i0+K] = Ct
            Binv_all[i0:i0+K,i0:i0+K] = Bt
            i0 += K

        CTBC = np.dot(CBinv, C_all.T)
        Ainv = np.linalg.inv(A - CTBC)
        ABinv = -np.dot(Ainv, CBinv)
        ABinvT = ABinv.T
        M[t, :K0, :K0] = Ainv
        M[t, :K0, K0:] = ABinv
        M[t, K0:, :K0] = ABinvT
        M[t, K0:, K0:] = Binv_all + np.dot(np.dot(CBinv.T, Ainv), CBinv)
        #MM[t] = np.block([[A, C_all],[C_all.T,block_diag(*btmp)]])
    return M


jit(nopython=True)
def sortGradient_idx( T, zdims, isReverse=False):
    """
    Sort array for gradient so that the latent are first stacked togheter on a certain time point
    :return:
    """
    idx_sort = np.zeros(T*np.sum(zdims),dtype=int)
    sumK = np.sum(zdims)
    cc = 0
    for jj in range(len(zdims)):
        i0 = np.sum(zdims[:jj])
        for tt in range(T):
            idx_sort[cc: cc+ zdims[jj]] = np.arange(sumK * tt + i0, sumK*tt + i0 + zdims[jj])
            cc += zdims[jj]
    if not isReverse:
        idx_sort = np.argsort(idx_sort)
    return idx_sort


@jit(nopython=True)
def slice(A, i0, i1, j0, j1):
    slice = np.zeros((i1-i0, j1-j0), dtype=np.float64)

    ci = 0
    for row in range(i0,i1):
        cj = 0
        for col in range(j0,j1):
            slice[ci,cj] = A[row,col]
            cj += 1
        ci += 1

    return slice

@jit(nopython=True)
def invertBlocks(*args):
    """
    Invert a list of T x Kj x Kj blocks of a block diagonal matri
    we want to invert
    :param args:
    :return:
    """
    Kmax = 0
    cc = 0
    for k in range(len(args)):
        blocks = args[k]
        Kmax = max(Kmax, blocks.shape[1])
        cc += blocks.shape[0]

    M = np.zeros((cc, Kmax, Kmax),dtype=np.float64)
    cc = 0
    for k in range(len(args)):
        blocks = args[k]
        K = blocks.shape[1]
        for t in range(blocks.shape[0]):
            B = blocks[t]
            M[cc, :K, :K] = np.linalg.inv(B)
            cc += 1
    return M

def all_trial_inverseHess_and_grad_factorized(dat, post_mean):
    indices = None
    indptr = None
    idx_sort = None
    rev_idx_sort = None
    first = True
    stackNum = len(dat.trialDur.keys())
    sumK = np.sum(dat.zdims)
    totDur = np.sum(list(dat.trialDur.values()))
    grad = np.zeros(totDur * sumK, dtype=np.float64, order='C')
    i0 = 0

    for trNum in dat.trialDur.keys():
        stim, xList = dat.get_observations(trNum)
        print('tr %d'%trNum)
        t0 = perf_counter()
        hesInv, indices, indptr = hess_factorized_logLike(dat, trNum, stim, xList, zbar=post_mean[trNum],
                                                          inverse=True, indices=indices, indptr=indptr)
        tt1 = perf_counter()
        grad[i0:i0 + dat.trialDur[trNum]*sumK], idx_sort, rev_idx_sort = grad_factorized_logLike(dat, trNum, stim,
                                                                                                     xList,
                                                                                                     zbar=post_mean[trNum],
                                                                                                     idx_sort=idx_sort,
                                                                    rev_idx_sort=rev_idx_sort, useGauss=1, usePoiss=1)
        i0 += dat.trialDur[trNum]*sumK
        t1 = perf_counter()
        print('hess compute',tt1-t0)
        print('grad compute', t1 - tt1)
        if first:
            indPTRSize = (hesInv.rowptrs.shape[0] - 1) * stackNum + 1
            indValSize = (hesInv.values.shape[0]) * stackNum
            indColSize = (hesInv.values.shape[0]) * stackNum
            vals = -np.ones(indValSize, dtype=np.float64, order='C')
            indcols = -np.ones(indColSize, dtype=np.int32, order='C')
            rowptr = np.zeros(indPTRSize, dtype=np.int32, order='C')
            i0Val = 0
            i0PTR = 0
            nrows = 0
            ncols = 0
            nnz = 0
            first = False

        (nrows, ncols, nnz, rowptr, indcols,
         vals, i0Val, i0PTR) = fast_stackCSRHes_memoryPreAllocation(vals, rowptr, indcols, nnz, nrows,
                                                                    ncols, i0PTR, i0Val, hesInv, sumK)
    hesInv = csr.CSR(nrows, ncols, nnz, rowptr, indcols, vals)
    return grad, hesInv



@jit(nopython=True)
def slice3(A,i0,i1,j0,j1,t):
    slice = np.zeros((i1 - i0, j1 - j0), dtype=np.float64)
    ci = 0
    for row in range(i0, i1):
        cj = 0
        for col in range(j0, j1):
            slice[ci, cj] = A[t, row, col]
            cj += 1
        ci += 1
    return slice


if __name__ == '__main__':

    from gen_synthetic_data import *
    T = 50
    data = dataGen(5,T=T)
    sub = data.cca_input.subSampleTrial(np.arange(1,4))
    multiTrialInference(sub)
    dat = data.cca_input
    # test factorized
    trNum = 1
    stim, xList = dat.get_observations(trNum)
    func = lambda zbar: factorized_logLike(dat,trNum,stim, xList, zbar=zbar)
    grad_func = lambda zbar: grad_factorized_logLike(dat, trNum, stim, xList, zbar=zbar)[0]
    hess_func = lambda zbar: hess_factorized_logLike(dat, trNum, stim, xList, zbar=zbar)


    z0 = np.random.normal(size=dat.trialDur[trNum]*np.sum(dat.zdims))
    apgrad = approx_grad(z0,z0.shape[0],func,epsi=10**-5)
    aphes = approx_grad(z0,(z0.shape[0],)*2,grad_func,10**-5)
    grd = grad_func(z0)
    hes,ind,indptr = hess_func(z0)
    hesinv,_,_ = hess_factorized_logLike(dat, trNum, stim, xList, zbar=z0,inverse=True,indices=ind,indptr=indptr)
    t0 = perf_counter()
    hes.multiply(hesinv)
    t1=perf_counter()
    print(t1-t0)
    hh = hes.to_scipy().toarray()
    hhinv = hesinv.to_scipy().toarray()
    t0 = perf_counter()
    hh.dot(hhinv)
    t1=perf_counter()
    print(t1-t0)
    A = sparse.spdiags(np.ones(10**4),0,10**4,10**4)

    t0 = perf_counter()
    res = csr.CSR.from_scipy(A)
    t1 = perf_counter()

    stackNum = 5
    timeComp = np.zeros(stackNum)
    timeCompMemoryAlloc = np.zeros(stackNum)
    timeComp_Reg = np.zeros(stackNum)

    # create hessian with no memory alloc
    t0 = perf_counter()
    hes2 = deepcopy(hes)
    timeComp[0] = perf_counter() - t0
    for k in range(stackNum-1):
        print('noalloc elaborate',k+1)
        t0 = perf_counter()
        hes2 = fast_stackCSRHes(hes2, hes)
        timeComp[k+1] = perf_counter() - t0
    # tt1_noAlloc = perf_counter()


    t0 = perf_counter()
    indPTRSize = (hes.rowptrs.shape[0] - 1) * stackNum + 1
    indValSize = (hes.values.shape[0]) * stackNum
    indColSize = (hes.values.shape[0]) * stackNum
    vals = -np.ones(indValSize,dtype=np.float64,order='C')
    indcols = -np.ones(indColSize,dtype=np.int32,order='C')
    rowptr = np.zeros(indPTRSize,dtype=np.int32,order='C')
    i0Val = 0
    i0PTR = 0
    nrows = 0
    ncols = 0
    nnz = 0
    init_time = perf_counter() - t0
    for k in range(stackNum):
        print('elaborate',k)
        t0 = perf_counter()
        (nrows, ncols, nnz, rowptr, indcols,
         vals, i0Val, i0PTR) = fast_stackCSRHes_memoryPreAllocation(vals, rowptr, indcols, nnz, nrows,
                                                                    ncols, i0PTR, i0Val, hes, np.sum(dat.zdims))

        timeCompMemoryAlloc[k] = perf_counter() - t0
    t0 = perf_counter()
    hes3 = csr.CSR(nrows, ncols, nnz, rowptr, indcols,vals)
    end_time = perf_counter() - t0

    # create hessian with no memory alloc
    hes_sci = hes.to_scipy()
    t0 = perf_counter()
    hes_sci2 = deepcopy(hes_sci)
    timeComp_Reg[0] = perf_counter() - t0
    for k in range(stackNum - 1):
        print('regular elaborate', k + 1)
        t0 = perf_counter()
        hes_sci2 = sparse.block_diag([hes_sci2,hes_sci])
        timeComp_Reg[k + 1] = perf_counter() - t0

    post_mean = preproc_post_mean_factorizedModel(dat)
    grad, hesInv = all_trial_inverseHess_and_grad_factorized(dat, post_mean)
