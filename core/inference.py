"""
Core inference functions.
Likelihoods gradients and hessians of all model components are implemented as individual functions and combined in
a single method.
"""
import numpy as np
from time import perf_counter
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

    vec_time = []
    vec_latent = []
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
    vec_time = np.hstack([vec_time, np.repeat(np.arange(T),K0)])
    vec_latent = np.hstack([vec_latent, np.repeat(np.ones(K0)*0,T)])
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
        vec_time = np.hstack([vec_time, np.repeat(np.arange(T), K)])
        vec_latent = np.hstack([vec_latent, np.repeat(np.ones(K) * (j+1), T)])

        grad_factorized[:K0*T] = grad_factorized[:K0*T] + usePoiss * tmp_z0.flatten()#.T.flatten()
        grad_factorized[i0: i0+K*T] = usePoiss * grad_z1.flatten()#.T.flatten()

        i0 += K * T

    if idx_sort is None:
        idx_sort = sortGradient_idx(T, dat.zdims, isReverse=False)
        # idx_sort = np.arange(T*sumK,dtype=int)
    return grad_factorized[idx_sort],vec_time


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
    hess_logLike2 = np.zeros([T, sumK, sumK], dtype=float)
    hess_logLike2[:, :K0, :K0] = hess_gaussObsLogLike(stim, z0, C, d, PsiInv,return_blocks=True)
    i0 = K0 * T
    ii0 = K0
    for k in range(len(xList)):
        N, K = dat.xPar[k]['W1'].shape
        counts = xList[k].reshape(T, N)
        z = zbar[i0: i0 + T * K].reshape(T, K)
        hess_z0, hess_z1, hess_z0z1 = hess_poissonLogLike(counts, z0, z, dat.xPar[k]['W0'], dat.xPar[k]['W1'],
                                                          dat.xPar[k]['d'],return_blocks=True)

        if inverse:
            hess_z1_inv = block_inv(hess_z1)

        hess_logLike2[:, :K0, :K0] = hess_logLike2[:, :K0, :K0] + hess_z0
        hess_logLike2[:, ii0:ii0+K, ii0:ii0+K] = hess_z1
        hess_logLike2[:, :K0, ii0:ii0 + K] = hess_z0z1
        hess_logLike2[:, ii0:ii0+K, :K0] = np.transpose(hess_z0z1,(0,2,1))

        i0 += K * T
        ii0 += K

    # create template for csr (use full blocks in case the inverse is computed)
    if indptr is None:
        mn = hess_logLike2.min()
        # make sure there are no unwanted zeros
        hess_logLike2 = hess_logLike2 - mn + 1
        spHess = sparse.block_diag(hess_logLike2, format='csr')
        indices = spHess.indices
        indptr = spHess.indptr
        # revert transform
        spHess.data = spHess.data + mn - 1
        # numba compatible
        spHess = csr.CSR.from_scipy(spHess)
    else:
        spHess = sparse.csr_matrix((hess_logLike2.flatten(), indices, indptr))
        # numba compatible
        spHess = csr.CSR.from_scipy(spHess)

    return spHess, indices, indptr

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


    #     Binv = np.linalg.inv(A[t].reshape(A.shape[1],A.shape[2]))
    #     for k in range(Binv.shape[0]):
    #         for j in range(Binv.shape[1]):
    #             B[t,k,j] = Binv[k,j]
    # return B



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
    data = dataGen(10,T=T)
    sub = data.cca_input.subSampleTrial(np.arange(1,4))
    multiTrialInference(sub)
    dat = data.cca_input
    # test factorized
    trNum = 5
    stim, xList = dat.get_observations(trNum)
    func = lambda zbar: factorized_logLike(dat,trNum,stim, xList, zbar=zbar)
    grad_func = lambda zbar: grad_factorized_logLike(dat, trNum, stim, xList, zbar=zbar)[0]
    hess_func = lambda zbar: hess_factorized_logLike(dat, trNum, stim, xList, zbar=zbar)


    z0 = np.random.normal(size=dat.trialDur[5]*np.sum(dat.zdims))
    apgrad = approx_grad(z0,z0.shape[0],func,epsi=10**-5)
    aphes = approx_grad(z0,(z0.shape[0],)*2,grad_func,10**-5)
    grd = grad_func(z0)
    hes,spHess,spHess2 = hess_func(z0)
    vt = grad_factorized_logLike(dat, trNum, stim, xList, zbar=z0)[1]
    # lt = grad_factorized_logLike(dat, trNum, stim, xList, zbar=z0)[1]
    t0 = perf_counter()
    idx_sort = sortGradient_idx(T, dat.zdims)
    t1 = perf_counter()
    print(t1-t0)

    MM = np.zeros((np.sum(dat.zdims),)*2)
    cc = 0
    for k in dat.zdims:
        MM[cc:cc+k,cc:cc+k] = 1
        cc += k
    MM[:dat.zdims[0],:] = 1
    MM[:, :dat.zdims[0]] = 1
    M = sparse.block_diag([MM]*T)


    # t0 = perf_counter()
    # hes, indices, indptr = hess_factorized_logLike(dat,trNum,stim,xList,zbar=z0,indices=None,indptr=None)
    # t1 = perf_counter()
    # print('first trial', t1 - t0)
    #
    # tt0 = perf_counter()
    # hes, indices, indptr = hess_factorized_logLike(dat, trNum, stim, xList, zbar=z0, indices=indices, indptr=indptr)
    # tt1 = perf_counter()
    # print('passed indices', tt1-tt0)


    A = np.random.normal(size=(300,300))

    tt0 = perf_counter()
    B = slice(A,10,200,39,298)
    tt1 = perf_counter()
    print(tt1-tt0)
    tt0 = perf_counter()
    B = slice(A, 10, 200, 39, 298)
    tt1 = perf_counter()
    print(tt1 - tt0)

    tt0 = perf_counter()
    BB = A[10:200,39:298]
    tt1 = perf_counter()
    print(tt1 - tt0)

    ABig = np.random.normal(size=(100*3,100*3))
    ABig = np.dot(ABig, ABig.T)

    t0 = perf_counter()
    np.linalg.inv(ABig)
    t1 = perf_counter()
    print('srandard inv', t1-t0)
    AA = np.random.normal(size=(50,3,3))
    for kk in range(AA.shape[0]):
        AA[kk] = np.dot(AA[kk],AA[kk].T)
    t0 = perf_counter()
    block_inv(AA)
    t1 = perf_counter()
    print('numba inv', t1 - t0)



