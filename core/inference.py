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
    if computeInv:
        K_big_inv = np.zeros(K_big.shape)
        logdet_K_big = 0
    else:
        logdet_K_big = None
        K_big_inv = None
    ii = 0
    for xd in range(K.shape[0]):
        K[xd] = epsSignal * np.exp(
            -0.5 * (np.tile(T,T.shape[0]).reshape(T.shape[0],T.shape[0]) - np.repeat(T,T.shape[0]).reshape(T.shape[0],T.shape[0]))**2
                * binSize**2 / ((tau[xd] * 1000) ** 2)) + epsNoise * np.eye(len(T))

        K_big[ii:ii+T.shape[0], ii:ii+T.shape[0]] = K[xd]

        if computeInv:
            chl = np.linalg.cholesky(K[xd])
            # Linv = np.linalg.solve(chl, np.eye(chl.shape[0]))
            # chlInv = lapack.dtrtri(np.asfortranarray(np.array(chl, order='F')),lower=1)[0]
            # Kinv = np.dot(chlInv.T,chlInv)
            Kinv = np.linalg.inv(K[xd])
            logdet_K = 2 * np.sum(np.log(np.diag(chl)))
            K_big_inv[ii:ii+T.shape[0], ii:ii+T.shape[0]] = Kinv  # invToeplitz(K);
            logdet_K_big = logdet_K_big + logdet_K
        ii += T.shape[0]

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
    K0 = stimPar['W0'].shape[1]
    tau0 = priorPar[0]['tau']
    T, stimDim = stim.shape

    # extract z0 and its params
    z0 = zstack[:T*K0].reshape(T,K0)
    K0_big_inv = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

    # compute log likelihood for the stimulus and the GP
    logLike = gaussObsLogLike(stim, z0, stimPar['W0'], stimPar['d'], stimPar['PsiInv']) + GPLogLike(z0, K0_big_inv) #


    i0 = K0*T
    for k in range(len(xList)):
        N, K = xPar[k]['W1'].shape
        counts = xList[k].reshape(T,N)
        z = zstack[i0: i0+T*K].reshape(T, K)
        K_big_inv = makeK_big(K, priorPar[k]['tau'], None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

        logLike += GPLogLike(z,K_big_inv) + poissonLogLike(counts, z0, z,
                                                           xPar[k]['W0'], xPar[k]['W1'], xPar[k]['d'])
        i0 += K*T
    return logLike


def grad_PpCCA_logLike(zstack, stim, xList, priorPar, stimPar, xPar, binSize,epsNoise=0.001):
    # extract dim z0, stim and trial time
    K0 = stimPar['W0'].shape[1]
    tau0 = priorPar[0]['tau']
    T, stimDim = stim.shape

    # extract z0 and its params
    z0 = zstack[:T*K0].reshape(T,K0)
    K0_big_inv = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]
    # print(K0, tau0, None, binSize, epsNoise, T, True)
    # compute log likelihood for the stimulus and the GP
    grad_logLike = np.zeros(zstack.shape)
    grad_z0 = grad_GPLogLike(z0, K0_big_inv)
    # print(grad_GPLogLike(z0, K0_big_inv).reshape(K0,T).T.flatten()[-5:])

    grad_z0 = grad_z0.reshape(K0,T).T.flatten()
    grad_z0 = grad_z0 + grad_gaussObsLogLike(stim, z0, stimPar['W0'], stimPar['d'], stimPar['PsiInv'])
    grad_logLike[:T*K0] = grad_z0
    # print(grad_logLike[:T*K0][-5:])

    i0 = K0*T
    for k in range(len(xList)):
        N, K = xPar[k]['W1'].shape
        counts = xList[k].reshape(T,N)
        z = zstack[i0: i0+T*K].reshape(T, K)
        K_big_inv = makeK_big(K, priorPar[k]['tau'], None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]
        grad_z0, grad_z1 = grad_poissonLogLike(counts, z0, z, xPar[k]['W0'], xPar[k]['W1'], xPar[k]['d'])
        grad_z1 = grad_z1.flatten() + grad_GPLogLike(z,K_big_inv).reshape(K,T).T.flatten()
        grad_logLike[:K0*T] = grad_logLike[:K0*T] + grad_z0.flatten()
        grad_logLike[i0: i0 + K * T] = grad_z1
        i0 += K * T
    return grad_logLike


def hess_PpCCA_logLike(zstack, stim, xList, priorPar, stimPar, xPar, binSize,epsNoise=0.001):
    # extract dim z0, stim and trial time
    K0 = stimPar['W0'].shape[1]
    tau0 = priorPar[0]['tau']
    T, stimDim = stim.shape

    # extract z0 and its params
    z0 = zstack[:T*K0].reshape(T,K0)
    K0_big_inv = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]
    # compute log likelihood for the stimulus and the GP
    hess_logLike = np.zeros([zstack.shape[0]]*2, dtype=np.float)
    hess_z0 = hess_GPLogLike(z0, K0_big_inv)
    idx_rot = np.tile(np.arange(0, K0*T, T), T) + np.arange(K0*T)//K0
    hess_logLike[:T*K0,:T*K0] = hess_z0[idx_rot,:][:,idx_rot] + hess_gaussObsLogLike(stim,z0,stimPar['W0'],
                                                                                     stimPar['d'],stimPar['PsiInv'])
    i0 = K0*T
    for k in range(len(xList)):
        N, K = xPar[k]['W1'].shape
        counts = xList[k].reshape(T,N)
        z = zstack[i0: i0+T*K].reshape(T, K)
        K_big_inv = makeK_big(K, priorPar[k]['tau'], None, binSize, epsNoise=epsNoise, T=T, computeInv=True)[2]

        hess_z0, hess_z1, hess_z0z1 = hess_poissonLogLike(counts, z0, z, xPar[k]['W0'], xPar[k]['W1'], xPar[k]['d'])
        tmp = hess_GPLogLike(z, K_big_inv)
        idx_rot = np.tile(np.arange(0, K * T, T), T) + np.arange(K * T) // K
        hess_z1 = hess_z1 + tmp[idx_rot,:][:,idx_rot]
        hess_logLike[:T*K0,:T*K0] = hess_logLike[:T*K0,:T*K0] + hess_z0
        hess_logLike[i0: i0 + K * T, i0: i0 + K * T] = hess_z1
        hess_logLike[:K0 * T, i0: i0 + K * T] = hess_z0z1
        hess_logLike[i0: i0 + K * T, :K0 * T] = hess_z0z1.T

        i0 += K * T
    return hess_logLike

def approx_grad(x0, dim, func, epsi):
    grad = np.zeros(shape=dim)
    for j in range(grad.shape[0]):
        if np.isscalar(x0):
            ej = epsi
        else:
            ej = np.zeros(x0.shape[0])
            ej[j] = epsi
        grad[j] = (func(x0 + ej) - func(x0 - ej)) / (2 * epsi)
    return grad

def reshapeHessianGP(zdim, T, hess):
    assert(hess.shape[0]==zdim*T)
    hess_resh = np.zeros(hess.shape)
    idx = np.arange(0,zdim*T,zdim)
    for k in range(zdim):
        hess_resh[k*T:(k+1)*T, k*T:(k+1)*T] = hess[k::zdim, k::zdim]
    return hess_resh

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
    import matplotlib.pylab as plt
    np.random.seed(4)
    preproc = emptyStruct()
    preproc.numTrials = 1
    preproc.ydim = 50
    preproc.binSize = 50

    preproc.T = np.array([7])

    tau = np.array([0.9, 0.2, 0.4, 0.2, 0.8])
    K0 = 5
    tau = tau[:K0]
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
    tau = np.array([1.1,1.3])
    K_big = makeK_big(2, tau, None, preproc.binSize, epsNoise=epsNoise, T=preproc.T[0], computeInv=False)[1]
    z1 = np.random.multivariate_normal(mean=np.zeros(preproc.T[0]*len(tau)), cov=K_big, size=1).reshape(preproc.T[0], len(tau))

    W1 = np.random.normal(size=(preproc.ydim, len(tau)))
    W0 = np.random.normal(size=(preproc.ydim, K0))
    d = -0.2
    preproc.data = [
        {'Y': np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', W0, z) + np.einsum('ij,tj->ti', W1, z1) + d))}]

    # create the data struct
    struc = GP_pCCA_input(preproc, ['var1', 'var2'], ['PPC'], np.array(['PPC'] * preproc.ydim),
                          np.ones(preproc.ydim, dtype=bool))
    struc.initializeParam([K0, z1.shape[1]])
    stim, xList = struc.get_observations(0)
    zstack = np.hstack((z.flatten(), z1.flatten()))


    res = PpCCA_logLike(zstack, stim, xList,priorPar=struc.priorPar,stimPar=struc.stimPar,xPar=struc.xPar,
                  binSize=struc.binSize,epsNoise=0.0001)

    grad_res = grad_PpCCA_logLike(zstack, stim, xList,priorPar=struc.priorPar,stimPar=struc.stimPar,xPar=struc.xPar,
                  binSize=struc.binSize,epsNoise=0.0001)

    func = lambda z: PpCCA_logLike(z, stim, xList,priorPar=struc.priorPar,stimPar=struc.stimPar,xPar=struc.xPar,
                  binSize=struc.binSize,epsNoise=0.0001)
    app_grad = approx_grad(zstack,zstack.shape[0],func,epsi=10**-4)

    plt.figure()
    plt.title('grad check: %f'%np.max(np.abs(app_grad.flatten()-grad_res.flatten())))
    plt.scatter(app_grad,grad_res)


    hess_res = hess_PpCCA_logLike(zstack, stim, xList, priorPar=struc.priorPar,stimPar=struc.stimPar,xPar=struc.xPar,
                  binSize=struc.binSize,epsNoise=0.0001)

    func_res = lambda z: grad_PpCCA_logLike(z, stim, xList,priorPar=struc.priorPar,stimPar=struc.stimPar,xPar=struc.xPar,
                  binSize=struc.binSize,epsNoise=0.0001)
    app_hess_res = approx_grad(zstack,(zstack.shape[0],zstack.shape[0]),func_res,epsi=10**-4)
    grad_res = func_res(zstack)

    plt.figure()
    plt.title('hessian check: %f'%np.max(np.abs(app_hess_res.flatten()-hess_res.flatten())))
    # plt.scatter(app_grad, grad_res)
    plt.scatter(app_hess_res.flatten(), hess_res.flatten())
    # plt.figure()
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    # sbn.heatmap(app_hess[:35,:35], linecolor='g', lw=0.5, ax=ax1, cbar=False)
    # sbn.heatmap(hess_res, linecolor='g', lw=0.5, ax=ax2, cbar=False)


    # # plt.scatter(app_hess.flatten(), hess_res.flatten())
    z0 = zstack[:K0*preproc.T[0]].reshape(preproc.T[0],K0)
    K0_big_inv = makeK_big(K0, struc.priorPar[0]['tau'], None, struc.binSize, epsNoise=0.0001, T=preproc.T[0], computeInv=True)[2]
    # print(K0, struc.priorPar[0]['tau'], None, struc.binSize, 0.0001, preproc.T[0],True)

    # K0_big_inv = np.diag(np.arange(1,1+K0_big_inv.shape[0]))
    # K0_big_inv[1,0] = 1
    # K0_big_inv[0, 1] = 1

    hess_z0 = hess_GPLogLike(z0, K0_big_inv)

    func = lambda z: grad_GPLogLike(z, K0_big_inv).reshape(K0,preproc.T[0]).T.flatten()
    grad_z0 = func(z0)
    print(grad_GPLogLike(z0, K0_big_inv).reshape(K0,preproc.T[0]).T.flatten()[-5:])

    func2 = lambda z: z.reshape(K0, preproc.T[0]).T.flatten()
    #
    app_hess = approx_grad(z0.flatten(),(z0.flatten().shape[0],z0.flatten().shape[0]),func,epsi=10**-4)

    rot = approx_grad(z0.flatten(),(z0.flatten().shape[0],z0.flatten().shape[0]),func2,epsi=10**-4)

    idx_rot = np.tile(np.arange(0, K0*preproc.T[0], preproc.T[0]), preproc.T[0]) + np.arange(K0*preproc.T[0])//K0
    hess_z0 = hess_z0[idx_rot].T
    # rot2 = np.eye(K0*preproc.T[0])[idx_rot,:]
    #
    # plt.figure()
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    # sbn.heatmap(app_hess, linecolor='g', lw=0.5,ax=ax1,cbar=False)
    # sbn.heatmap(hess_res, linecolor='g', lw=0.5,ax=ax2,cbar=False)
    #
    #
    # plt.figure()
    # plt.scatter(app_hess.flatten(), hess_z0.flatten())
    #

    # A = np.zeros((K0*preproc.T[0],)*2)
    # i0=0
    # idx = np.arange(0,K0*preproc.T[0],K0)
    # for xd in range(K0):
    #     xx, yy = np.meshgrid(idx + xd, idx + xd)
    #     A[xx,yy] = np.ones((preproc.T[0],)*2) * (xd+1)
    # # for tt in range(preproc.T[0]):
    # #     A[i0:i0+K0,i0:i0+K0] = np.ones((K0,K0)) *(tt+1)
    # #     i0 = i0+K0
    #
    # func = lambda z: reshapeHessianGP(K0, preproc.T[0], z.reshape(K0* preproc.T[0], K0* preproc.T[0]))
    # # app_hess = approx_grad(A.flatten(), (A.flatten().shape[0], A.flatten().shape[0]), func, epsi=10 ** -4)
    # plt.figure()
    # ax1 = plt.subplot(121)
    # ax2 = plt.subplot(122)
    # sbn.heatmap(A, linecolor='g', lw=2,ax=ax1)
    # sbn.heatmap(func(A), linecolor='g', lw=2,ax=ax2)
    #
    #
    # func = lambda z: z.reshape(3,4).T.flatten()
    # app_hess = approx_grad(np.arange(12), (12, 12), func,10** -4)
    # plt.figure()
    # sbn.heatmap(app_hess, linecolor='g', lw=2)