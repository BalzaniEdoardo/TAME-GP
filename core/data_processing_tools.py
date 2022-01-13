import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy

class emptyStruct(object):
    def __init__(self):
        return

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
            # eig = np.linalg.eigh(K[xd])[0]
            # if any(np.isnan(eig)):
            #     xxx=1
            Kinv = np.linalg.inv(K[xd])
            logdet_K = logDetCompute(K[xd])
            K_big_inv[ii:ii+T.shape[0], ii:ii+T.shape[0]] = Kinv
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
        i0 = np.sum(data.zdims[:cnt_dim])*T

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
        i0 = np.sum(data.zdims[:cnt_dim])*T

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

if __name__ == '__main__':
    from time import perf_counter
    from gen_synthetic_data import dataGen
    import seaborn as sbn
    import matplotlib.pylab as plt

    # zdim = [2]+[10]*5
    # T = 60
    # B = np.zeros((np.sum(zdim)*T, np.sum(zdim)*T))
    #
    # ii = 0
    # for k in range(len(zdim)):
    #     AA = np.random.normal(size=(zdim[k]*T, zdim[k]*T))
    #     AA = np.dot(AA, AA.T)
    #     B[ii: ii + zdim[k]*T, ii: ii + zdim[k]*T] = AA
    #     ii += zdim[k]*T
    #
    # ii = zdim[0]*T
    # for k in range(1,len(zdim)):
    #     AA = np.random.normal(size=(zdim[0]*T,zdim[k]*T))
    #     B[: zdim[0]*T, ii: ii + zdim[k]*T] = AA
    #     B[ii: ii + zdim[k]*T, :zdim[0]*T] = AA.T
    #     ii += zdim[k]*T
    #
    # # t0 = perf_counter()
    # # Binv = np.linalg.inv(B)
    # # t1 = perf_counter()
    # # print(t1 - t0)
    # tt0 = perf_counter()
    # Binv2 = invertHessBlock(B, zdim, T)
    # tt1 = perf_counter()
    # print(tt1-tt0)
    # # print((t1-t0)/(tt1-tt0))
    #
    # # gen syntetic data
    dat = dataGen(1)


    # invertHessBlock(precision, dat.cca_input.zdims, dat.cca_input.trialDur[0])
    # mn,cov,crcov = parse_fullCov(dat.cca_input, dat.meanPost[0], dat.covPost[0],dat.cca_input.trialDur[0])