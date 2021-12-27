"""
Core inference functions.
Likelihoods gradients and hessians of all model components are implemented as individual functions and combined in
a single method.
"""
import numpy as np
from time import perf_counter
from scipy.optimize import minimize
from scipy.linalg import block_diag
import scipy.sparse as sparse

def compileKBig_Fast(K, K_big, T, binSize, epsNoise, epsSignal, tau, computeInv=True):
    """
    Compute the RBF covariance for given parameters.
    :param K:
    :param K_big:
    :param T:
    :param binSize:
    :param epsNoise:
    :param epsSignal:
    :param tau:
    :param computeInv:
    :return:
    """
    # compute log det
    idx = np.arange(0, K.shape[0]*T.shape[0], K.shape[0])
    if computeInv:
        K_big_inv = np.zeros(K_big.shape)
        logdet_K_big = 0
    else:
        logdet_K_big = None
        K_big_inv = None

    for xd in range(K.shape[0]):
        xx, yy = np.meshgrid(idx + xd, idx + xd)
        K[xd] = epsSignal * np.exp(
            -0.5 * (np.tile(T,T.shape[0]).reshape(T.shape[0],T.shape[0]) - np.repeat(T,T.shape[0]).reshape(T.shape[0],T.shape[0]))**2
                * binSize**2 / ((tau[xd] * 1000) ** 2)) + epsNoise * np.eye(len(T))

        K_big[xx, yy] = K[xd]
        if computeInv:
            chl = np.linalg.cholesky(K[xd])
            Linv = np.linalg.solve(chl, np.eye(chl.shape[0]))
            Kinv = np.dot(Linv.T, Linv)
            logdet_K = 2 * np.sum(np.log(np.diag(chl)))
            K_big_inv[xx, yy] = Kinv  # invToeplitz(K);
            logdet_K_big = logdet_K_big + logdet_K
        else:
            logdet_K_big
    return K, K_big, K_big_inv,  logdet_K_big

def makeK_big(xdim, tau, trialDur, binSize, epsNoise=0.001, T=None, computeInv=False):
    """
    Compute the GP covariance, its inverse and the log-det
    :param params:
    :param trialDur:
    :param binSize:
    :param epsNoise:
    :param T:
    :param xdim:
    :param computeInv:
    :return:
    """

    epsSignal = 1 - epsNoise
    if T is None:
        T = np.arange(0, int(trialDur / binSize))
    else:
        T = np.arange(0, T)
    K = np.zeros([xdim, len(T), len(T)])
    K_big = np.zeros([xdim * len(T), xdim * len(T)])
    K, K_big, K_big_inv,  logdet_K_big = compileKBig_Fast(K, K_big, T, binSize, epsNoise, epsSignal, tau,
                                                          computeInv=computeInv)

    return K, K_big, K_big_inv,  logdet_K_big

def GPLogLike(z, Kinv):
    """
    GP prior log-likelihood
    :param z: KxT latent factor
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


def hess_gaussObsLogLike(s, z, C, d, PsiInv):
    """
    Hessian of the gaussian observartions
    :param z:
    :param C:
    :param PsiInv:
    :return:
    """
    CTPsiInv = np.dot(C.T, PsiInv)
    CPsiInvC = np.dot(CTPsiInv, C)
    M = block_diag(*[CPsiInvC]*z.shape[0])
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


def hess_poissonLogLike(x, z0, z1, W0, W1, d):
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
    return block_diag(*(-precision_z0)),block_diag(*(-precision_z1)),block_diag(*(-precision_z0z1))


def PpCCA_logLike(zstack, stim, xList, priorPar, stimPar, xPar, binSize,epsNoise=0.001):
    # extract dim z0, stim and trial time
    K0 = stimPar['d'].shape[0]
    tau0 = priorPar[0]['tau']
    T, stimDim = stim.shape

    # extract z0 and its params
    z0 = zstack[:T*K0].reshape(T,K0)
    K0_big_inv = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

    # compute log likelihood for the stimulus and the GP
    logLike = GPLogLike(z0, K0_big_inv) + gaussObsLogLike(stim, z0, stimPar['W0'], stimPar['d'], stimPar['PsiInv'])


    i0 = K0
    for k in range(len(xList)):
        N, K = xPar[k]['W1'].shape
        counts = xList[k].reshape(T,N)
        z = zstack[i0: i0+T*K].reshape(T, K)
        K_big_inv = makeK_big(K, priorPar[k]['tau'], None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

        logLike += GPLogLike(z,K_big_inv) + poissonLogLike(counts, z0, z,
                                                           xPar[k]['W0'], xPar[k]['W1'], xPar[k]['d'])
    return logLike




def grad_PpCCA_logLike(zstack, stim, xList, priorPar, stimPar, xPar, binSize,epsNoise=0.001):
    # extract dim z0, stim and trial time
    K0 = stimPar['d'].shape[0]
    tau0 = priorPar[0]['tau']
    T, stimDim = stim.shape

    # extract z0 and its params
    z0 = zstack[:T*K0].reshape(T,K0)
    K0_big_inv = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

    # compute log likelihood for the stimulus and the GP
    grad_logLike = grad_GPLogLike(z0, K0_big_inv) + gaussObsLogLike(stim, z0, stimPar['W0'], stimPar['d'], stimPar['PsiInv'])


    i0 = K0
    for k in range(len(xList)):
        N, K = xPar[k]['W1'].shape
        counts = xList[k].reshape(T,N)
        z = zstack[i0: i0+T*K].reshape(T, K)
        K_big_inv = makeK_big(K, priorPar[k]['tau'], None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

        grad_logLike += grad_GPLogLike(z,K_big_inv) + grad_poissonLogLike(counts, z0, z,
                                                           xPar[k]['W0'], xPar[k]['W1'], xPar[k]['d'])
    return grad_logLike


if __name__ == '__main__':
    import sys, os, inspect

    basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
    sys.path.append(os.path.join(basedir, 'firefly_utils'))
    sys.path.append(os.path.join(basedir, 'core'))
    from behav_class import emptyStruct
    from inference import makeK_big
    from scipy.linalg import block_diag
    from data_structure import *
    import seaborn as sbn

    preproc = emptyStruct()
    preproc.numTrials = 1
    preproc.ydim = 50
    preproc.binSize = 50

    preproc.T = np.array([100])

    tau = np.array([0.9, 0.2, 0.4])
    K0 = 3
    epsNoise = 0.000001
    K_big = makeK_big(K0, tau, None, preproc.binSize, epsNoise=epsNoise, T=preproc.T[0], computeInv=False)[1]
    z = np.random.multivariate_normal(mean=np.zeros(K0 * preproc.T[0]), cov=K_big, size=1).reshape(preproc.T[0], K0)

    # create the stim vars
    PsiInv = np.eye(2)
    W = np.random.normal(size=(2, K0))
    d = np.zeros(2)
    preproc.covariates = {}
    preproc.covariates['var1'] = [np.random.multivariate_normal(mean=np.dot(W, z.T)[0], cov=np.eye(preproc.T[0]))]
    preproc.covariates['var2'] = [np.random.multivariate_normal(mean=np.dot(W, z.T)[1], cov=np.eye(preproc.T[0]))]

    # create the counts
    tau = np.array([1.1])
    K_big = makeK_big(1, tau, None, preproc.binSize, epsNoise=epsNoise, T=preproc.T[0], computeInv=False)[1]
    z1 = np.random.multivariate_normal(mean=np.zeros(preproc.T[0]), cov=K_big, size=1).reshape(preproc.T[0], 1)

    W1 = np.random.normal(size=(preproc.ydim, 1))
    W0 = np.random.normal(size=(preproc.ydim, 1))
    d = -0.2
    preproc.data = [
        {'Y': np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', W0, z) + np.einsum('ij,tj->ti', W1, z1) + d))}]

    # create the data struct
    struc = GP_pCCA_input(preproc, ['var1', 'var2'], ['PPC'], np.array(['PPC'] * preproc.ydim),
                          np.ones(preproc.ydim, dtype=bool))
    struc.initializeParam([2, 1])
    struc.get_observations(0)


