import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy
from numba import jit
import csr
from scipy.stats import multivariate_normal


def compileTrialStackedObsAndLatent(data, idx_latent, trial_list, T, xDim, K0, K1):
    x = np.zeros((T, xDim))
    mean_post = np.zeros((T, K0 + K1))
    cov_post = np.zeros((T, K1 + K0, K1 + K0))
    t0 = 0
    for tr in trial_list:
        T_tr = data.trialDur[tr]
        x[t0:t0 + T_tr, :] = data.get_observations(tr)[1][idx_latent - 1]
        cov_post[t0:t0 + T_tr, :K0, :K0] = data.posterior_inf[tr].cov_t[0]
        cov_post[t0:t0 + T_tr, K0:, K0:] = data.posterior_inf[tr].cov_t[idx_latent]
        cov_post[t0:t0 + T_tr, :K0, K0:] = data.posterior_inf[tr].cross_cov_t[idx_latent]
        cov_post[t0:t0 + T_tr, K0:, :K0] = np.transpose(cov_post[t0: t0 + T_tr, :K0, K0:], (0, 2, 1))
        mean_post[t0:t0 + T_tr, :K0] = data.posterior_inf[tr].mean[0].T
        mean_post[t0:t0 + T_tr, K0:] = data.posterior_inf[tr].mean[idx_latent].T
        t0 += T_tr
    return x, mean_post, cov_post

def fast_stackCSRHes_memoryPreAllocation(vals, rowsptr, colindices, nnz, nrows, ncols, i0PTR, i0Val, newHes, sumK):
    """
    Fast stacking of csr matrix format hessian of different trials. Strongly uses the fact that across trials the
    structure of the matrix is the same. The important bit is to make sure that from one trial to another there are no
    additional zeros in the block structure of the matrix; thi in the factorized model computation of the hessian is
    done by adding  (min + 1) to vector of values that go into the matrix.
    :param spHess: hessian for the log likelihood of multiple trials in csr.CSR format
    :param newHes: hessian for the log likelihood of a new trial in csr.CSR format that we want to stack as a new
    block-diagonal component
    :return:
    """
    # update indices
    new_indices = newHes.colinds + colindices[i0Val-1] + 1
    new_ptr = newHes.rowptrs + rowsptr[i0PTR]

    # create the concatenated hessian
    vals[i0Val: i0Val + newHes.values.shape[0]] = newHes.values
    rowsptr[i0PTR: i0PTR + new_ptr.shape[0]] = new_ptr
    colindices[i0Val: i0Val + newHes.values.shape[0]] = new_indices

    nnz = nnz + newHes.nnz
    nrows = nrows + newHes.nrows
    ncols = ncols + newHes.ncols

    i0Val = i0Val + newHes.values.shape[0]
    i0PTR = i0PTR + new_ptr.shape[0] - 1

    return nrows, ncols, nnz, rowsptr, colindices, vals, i0Val, i0PTR

class emptyStruct(object):
    def __init__(self):
        return

@jit(nopython=True)
def block_inv(A):
    Binv = np.zeros((A.shape),dtype=np.float64)
    for t in range(A.shape[0]):
        Binv[t] = np.linalg.inv(A[t])
    return Binv

def logDetHessBlock(B,zdim,T):
    """
        Invert matrix using the block matrix inversion formula given the structure of the M.
        This should reduce the order of the computation from (sum(K_i)xT)^3 to sum ( K_i x T)^3
        :param M:
        :param zdims:
        :return:
        """
    K0 = zdim[0]
    zdim = zdim[1:]
    A = B[:K0 * T, :K0 * T]
    C = B[K0 * T:, :K0 * T]
    i0 = K0 * T
    detBblocks = 0
    invList = []
    for K in zdim:
        KT = K * T
        B_block = B[i0:i0 + KT, i0:i0 + KT]
        eig, u = np.linalg.eigh(B_block)
        detBblocks += np.log(eig).sum()
        invList.append(np.linalg.inv(B_block))
        i0 += KT

    Binv = block_diag(*invList)
    CTBinv = np.dot(C.T, Binv)
    CTBC = np.dot(CTBinv, C)

    e, u = np.linalg.eig(A - CTBC)
    logdetM = np.log(e).sum() + detBblocks
    return logdetM

def invertHessBlock(B, zdim, T):
    """
    Invert matrix using the block matrix inversion formula given the structure of the M.
    This should reduce the order of the computation from (sum(K_i)xT)^3 to sum ( K_i x T)^3
    :param M:
    :param zdims:
    :return:
    """
    K0 = zdim[0]
    zdim = zdim[1:]
    A = B[:K0*T,:K0*T]
    C = B[K0*T:,:K0*T]
    i0 = K0*T
    invList = []
    for K in zdim:
        KT = K*T
        B_block = B[i0:i0+KT, i0:i0+KT]
        invList.append(np.linalg.inv(B_block))
        i0 += KT


    Binv = block_diag(*invList)
    CTBinv = np.dot(C.T,Binv)
    CTBC = np.dot(CTBinv,C)

    Ainv = np.linalg.inv(A-CTBC)
    ABinv = -np.dot(Ainv, CTBinv)

    Minv = np.block([[Ainv, ABinv], [ABinv.T, Binv + np.dot(np.dot(CTBinv.T,Ainv),CTBinv)]])
    return Minv



def logDetCompute(K):
    chl = np.linalg.cholesky(K)
    return 2 * np.sum(np.log(np.diag(chl)))

def compileKBig_Fast(K, K_big, T, binSize, epsNoise, epsSignal, tau, computeInv=True, returnSqrt=False):
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
        if not returnSqrt:
            K_big_inv = np.zeros(K_big.shape)
        else:
            K_big_inv = np.zeros(K.shape)
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
            # eig = np.linalg.eigh(K[xd])[0]
            # if any(np.isnan(eig)):
            #     xxx=1
            
            
            if not returnSqrt:
                logdet_K = logDetCompute(K[xd])
                Kinv = np.linalg.inv(K[xd])
                K_big_inv[ii:ii+T.shape[0], ii:ii+T.shape[0]] = Kinv
            else:
                vals, vecs = np.linalg.eigh(K[xd])
                logdet_K = np.sum(np.log(vals))
                U = vecs * np.sqrt(1/vals)
                K_big_inv[xd] = U
            logdet_K_big = logdet_K_big + logdet_K
        ii += T.shape[0]

    return K, K_big, K_big_inv,  logdet_K_big

def makeK_big(xdim, tau, trialDur, binSize, epsNoise=0.001, T=None, computeInv=False,returnSqrt=False):
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
                                                          computeInv=computeInv,returnSqrt=returnSqrt)

    return K, K_big, K_big_inv,  logdet_K_big


def retrive_t_blocks_fom_cov(data, trNum, i_Latent, meanPost, covPost):
    """
    this function returns the t x k_i x k_i covariance block for the posterior. the posterior is arranged into
    T blocks.
    :return:
    """
    # i0 = np.sum(data.zdims[:i_Latent])
    K = data.zdims[i_Latent]
    K0 = data.zdims[0]
    T = data.trialDur[trNum]
    i0 = np.sum(data.zdims[:i_Latent])*T
    cov_00 = covPost[trNum][: K0 * T, : K0 * T]
    mean_0 = meanPost[trNum][:K0 * T]
    if i_Latent == 0:
        mean_tt = np.zeros((T, K0))
        cov_tt = np.zeros((T, K0, K0))
        for t in range(T):
            cov_tt[t] = cov_00[t * K0: (t + 1) * K0, t * K0: (t + 1) * K0]
            mean_tt[t] = mean_0[t*K0:(t+1)*K0]

    else:
        mean_i = meanPost[trNum][i0: i0 + K * T]
        cov_ii = covPost[trNum][i0: i0 + K * T, i0: i0 + K * T]
        cov_i0 = covPost[trNum][:K0 * T, i0: i0 + K * T]
        cov_tt = np.zeros((T, K+K0, K+K0))
        mean_tt = np.zeros((T,K+K0))
        for t in range(T):
            cov_tt[t][:K0, :K0] = cov_00[t*K0: (t+1)*K0, t*K0: (t+1)*K0]
            cov_tt[t][:K0, K0:] = cov_i0[t*K0: (t+1)*K0, t*K: (t+1)*K]
            cov_tt[t][K0:, :K0] = cov_tt[t][:K0, K0:].T
            cov_tt[t][K0:, K0:] = cov_ii[t * K: (t + 1) * K, t * K: (t + 1) * K]
            mean_tt[t][:K0] = mean_0[t*K0:(t+1)*K0]
            mean_tt[t][K0:] = mean_i[t * K:(t + 1) * K]
    return mean_tt, cov_tt

def parse_fullCov(data, meanPost, covPost, T):
    """
    this function returns the t x k_i x k_i covariance block for the posterior. the posterior is arranged into
    T blocks.
    :return:
    """

    cov_ii_t = {}
    mean_t = {}
    cov_0i_t = {}
    K0 = data.zdims[0]

    cnt_dim = 0
    for K in data.zdims:
        i0 = int(np.sum(data.zdims[:cnt_dim])*T)

        mean_k = meanPost[i0:i0 + K * T]
        cov_k = covPost[i0: i0 + K * T, i0: i0 + K * T]
        crosscov_k = covPost[:K0 * T, i0: i0 + K * T]

        mean_t_k = np.zeros((T, K))
        cov_ii_k = np.zeros((T, K, K))
        cov_0i_k = np.zeros((T, K0, K))
        for t in range(T):
            # print(cnt_dim,K,t)
            cov_ii_k[t] = cov_k[t * K: (t + 1) * K, t * K: (t + 1) * K]
            mean_t_k[t] = mean_k[t * K:(t + 1) * K]
            if cnt_dim != 0:
                cov_0i_k[t] = crosscov_k[t*K0: (t+1)*K0, t*K: (t+1)*K]


        cov_ii_t[cnt_dim] = deepcopy(cov_ii_k)
        mean_t[cnt_dim] = deepcopy(mean_t_k)
        if cnt_dim !=0:
            cov_0i_t[cnt_dim] = deepcopy(cov_0i_k)
        cnt_dim += 1

    return mean_t, cov_ii_t, cov_0i_t

def parse_fullCov_latDim(data, meanPost, covPost, T):
    """
    this function returns the t x k_i x k_i covariance block for the posterior. the posterior is arranged into
    T blocks.
    :return:
    """

    cov_ii_t = {}
    mean_t = {}

    cnt_dim = 0

    for K in data.zdims:
        idx = np.arange(0, K * T, K)
        i0 = int(np.sum(data.zdims[:cnt_dim])*T)
        mean_k = meanPost[i0:i0 + K * T] # dim KT
        cov_k = covPost[i0: i0 + K * T, i0: i0 + K * T] # dim KT x KT

        # print(i0)

        mean_t_k = np.zeros((K, T))
        cov_ii_k = np.zeros((K,T, T))
        for k in range(K):
            # print(cnt_dim,K,t)
            xx, yy = np.meshgrid(idx + k, idx + k)
            cov_ii_k[k] = cov_k[xx, yy]
            mean_t_k[k] = mean_k[idx+k]


        cov_ii_t[cnt_dim] = deepcopy(cov_ii_k)
        mean_t[cnt_dim] = deepcopy(mean_t_k)

        cnt_dim += 1

    return mean_t, cov_ii_t

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

def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

def logpdf_multnorm(x, mean, covInv, logdet):
    rank = mean.shape[0]
    dev = x - mean
    # "maha" for "Mahalanobis distance".
    maha = np.dot(np.dot(dev,covInv),dev)
   # print('maha2', maha)
    log2pi = np.log(2 * np.pi)
    return -0.5 * (rank * log2pi + maha + logdet)




if __name__ == '__main__':
    from gen_synthetic_data import dataGen
    dat = dataGen(1)
