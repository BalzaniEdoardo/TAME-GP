"""
Core inference functions.
Likelihoods gradients and hessians of all model components are implemented as individual functions and combined in
a single method.
"""
import numpy as np
import os,sys,inspect
basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(basedir,'core'))
from time import perf_counter
from data_processing_tools import block_inv, approx_grad
from data_preprocessing_tools_factorized import fast_stackCSRHes_memoryPreAllocation, preproc_post_mean_factorizedModel
import scipy.sparse as sparse
from numba import jit
import csr
from inference import gaussObsLogLike, poissonLogLike, grad_gaussObsLogLike, grad_poissonLogLike,\
    hess_gaussObsLogLike, hess_poissonLogLike

def factorized_logLike(dat, trNum, stim, xList, zbar=None, idx_sort=None,rev_idx_sort=None):
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

    stimPar = dat.stimPar
    xPar = dat.xPar

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

def all_trial_ll_grad_hess_factorized(dat, post_mean, tr_dict={}, isDict=False, returnLL=False):
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
    LL = 0

    for trNum in dat.trialDur.keys():
        stim, xList = dat.get_observations(trNum)
        print('tr %d'%trNum)
        if isDict:
            zbar = post_mean[trNum]
        else:
            zbar = post_mean[tr_dict[trNum]]
        LL += factorized_logLike(dat, trNum, stim, xList, zbar=zbar, idx_sort=idx_sort, rev_idx_sort=rev_idx_sort)
        if returnLL:
            continue
        t0 = perf_counter()
        hesInv, indices, indptr = hess_factorized_logLike(dat, trNum, stim, xList, zbar=zbar,
                                                          inverse=True, indices=indices, indptr=indptr)
        tt1 = perf_counter()
        grad[i0:i0 + dat.trialDur[trNum]*sumK], idx_sort, rev_idx_sort = grad_factorized_logLike(dat, trNum, stim,
                                                                                                     xList,
                                                                                                     zbar=zbar,
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
    if returnLL:
        return LL
    hesInv = csr.CSR(nrows, ncols, nnz, rowptr, indcols, vals)
    return LL, grad, hesInv



def newton_optim_map(dat, tol=10 ** -10, max_iter=100, max_having=15,
                     indices=None, indptr=None,
                     indices_up=None, indptr_up=None, disp_ll=False):
    eps_float = np.finfo(float).eps
    Z0, tr_dict = preproc_post_mean_factorizedModel(dat,returnDict=False)

    ii = 0
    delta_ll = np.inf

    tmpZ = Z0.copy()
    while ii < max_iter and delta_ll > tol:

        # t0 = perf_counter()
        log_like, grad_ll, hess_ll = all_trial_ll_grad_hess_factorized(dat, Z0, tr_dict=tr_dict, isDict=True)
        if disp_ll:
            print('newton optim iter', ii, 'log-like', log_like)
        delt = hess_ll.mult_vec(grad_ll)
        # print('sparse solve time', perf_counter() - t0)
        step = 1
        new_ll = -np.inf
        step_halv = 0
        # t0 = perf_counter()
        while new_ll < log_like and step_halv < max_having:

            tmpZ = Z0 - step * delt.reshape(Z0.shape)
            new_ll = all_trial_ll_grad_hess_factorized(dat, Z0, tr_dict=tr_dict, isDict=True, returnLL=True)
            delta_ll = new_ll - log_like
            if disp_ll:
                print('halv step log-like', step_halv + 1, new_ll)
            step = step / 2
            step_halv += 1
        # print('step halving time', perf_counter() - t0)
        if new_ll == -np.inf or (log_like - new_ll > np.sqrt(eps_float)):
            return Z0, False
        Z0 = tmpZ
        if disp_ll:
            print('delta_ll', delta_ll)

        ii += 1
    # only criteria for convergence used
    flag_convergence = delta_ll <= tol
    return Z0, flag_convergence



if __name__ == '__main__':

    from gen_synthetic_data import *
    from copy import deepcopy
    from data_preprocessing_tools_factorized import fast_stackCSRHes
    T = 50
    data = dataGen(50,T=T)
    sub = data.cca_input.subSampleTrial(np.arange(1,4))
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
    ll, grad, hesInv = all_trial_ll_grad_hess_factorized(dat, post_mean)

    res = newton_optim_map(dat, tol=10 ** -10, max_iter=100, max_having=15,
                     indices=None, indptr=None,
                     indices_up=None, indptr_up=None, disp_ll=True)