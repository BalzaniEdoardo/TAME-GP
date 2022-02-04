"""
Core inference functions.
Likelihoods gradients and hessians of all model components are implemented as individual functions and combined in
a single method.
"""
import numpy as np
import os,sys,inspect
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(basedir,'core'))
from time import perf_counter
from data_processing_tools import block_inv, approx_grad, emptyStruct, fast_stackCSRHes_memoryPreAllocation,sortGradient_idx
from data_processing_tools_factorized import preproc_post_mean_factorizedModel
import scipy.sparse as sparse
from numba import jit
import csr
from inference import gaussObsLogLike, poissonLogLike, grad_gaussObsLogLike, grad_poissonLogLike,\
    hess_gaussObsLogLike, hess_poissonLogLike


def reconstruct_post_mean_and_cov(dat, zbar, index_dict):
    """

    :param dat: P_GPCCA inputt
    :param zbar: the output of newton optim (which is the posterior mean for all trials)
    :param index_dict: index of zbar for each trial
    :return:
    """
    if 'posterior_inf' not in dat.__dict__.keys():
        dat.posterior_inf = {}

    for tr in index_dict.keys():
        stim, xList = dat.get_observations(tr)
        post_mean = zbar[index_dict[tr]]
        post_cov = -hess_factorized_logLike(dat, tr, stim, xList, zbar=post_mean, inverse=True,
                            return_tensor=True)
        T = dat.trialDur[tr]
        rev_idx_sort = sortGradient_idx(T, dat.zdims, isReverse=True)
        post_mean = post_mean[rev_idx_sort]

        if tr not in dat.posterior_inf.keys():
            dat.posterior_inf[tr] = emptyStruct()
            dat.posterior_inf[tr].mean = {}
            dat.posterior_inf[tr].cov_t = {}
            dat.posterior_inf[tr].cross_cov_t = {}

        i0 = 0
        c0 = 0
        for j in range(len(dat.zdims)):
            dat.posterior_inf[tr].mean[j] = post_mean[i0: i0 + dat.zdims[j] * T].reshape(T, dat.zdims[j]).T
            dat.posterior_inf[tr].cov_t[j] = post_cov[:, c0: c0 + dat.zdims[j], c0: c0 + dat.zdims[j]]
            if j > 0:
                dat.posterior_inf[tr].cross_cov_t[j] = post_cov[:, :dat.zdims[0], c0: c0 + dat.zdims[j]]

            i0 += dat.zdims[j] * T
            c0 += dat.zdims[j]
    return True

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
    else:
        if rev_idx_sort is None:
            rev_idx_sort = sortGradient_idx(T, dat.zdims, isReverse=True)
        zbar = zbar[rev_idx_sort]


    stimPar = dat.stimPar
    xPar = dat.xPar

    # extract dim z0, stim and trial time
    K0 = stimPar['W0'].shape[1]
    T, stimDim = stim.shape

    # extract z0 and its params
    z0 = zbar[:T*K0].reshape(T, K0)


    # compute log likelihood for the stimulus and the GP
    logLike = gaussObsLogLike(stim, z0, stimPar['W0'], stimPar['d'], stimPar['PsiInv']) - 0.5 * (zbar*zbar).sum()

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
    grad_z0 = useGauss * grad_gaussObsLogLike(stim,zbar[:T*K0].reshape(T,K0), C, d, PsiInv) - zbar[:T*K0]
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
        grad_factorized[i0: i0+K*T] = usePoiss * grad_z1.flatten() - zbar[i0: i0+K*T]#.T.flatten()

        i0 += K * T

    if idx_sort is None:
        idx_sort = sortGradient_idx(T, dat.zdims, isReverse=False)
        # idx_sort = np.arange(T*sumK,dtype=int)
    return grad_factorized[idx_sort], idx_sort, rev_idx_sort


def hess_factorized_logLike(dat, trNum, stim, xList, zbar=None, idx_sort=None,
                            rev_idx_sort=None, indices=None, indptr=None, inverse=False,
                            return_tensor=False):
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
        H = H - np.eye(sumK)
        H[:, :K0, :K0] = H[:, :K0, :K0] + hess_gaussObsLogLike(stim, z0, C, d, PsiInv, return_blocks=True)
    else:
        inverseBlocks = []
        corssBlocks = []#np.zeros((K0,T*(sumK-K0)),dtype=np.float64)
        A = hess_gaussObsLogLike(stim, z0, C, d, PsiInv, return_blocks=True) - np.eye(K0)

    for k in range(len(xList)):
        N, K = dat.xPar[k]['W1'].shape
        counts = xList[k].reshape(T, N)
        z = zbar[i0: i0 + T * K].reshape(T, K)
        hess_z0, hess_z1, hess_z0z1 = hess_poissonLogLike(counts, z0, z, dat.xPar[k]['W0'], dat.xPar[k]['W1'],
                                                          dat.xPar[k]['d'],return_blocks=True)

        if not inverse:
            H[:, :K0, :K0] = H[:, :K0, :K0] + hess_z0

            H[:, ii0:ii0+K, ii0:ii0+K] = H[:, ii0:ii0+K, ii0:ii0+K] + hess_z1
            H[:, :K0, ii0:ii0 + K] = hess_z0z1
            H[:, ii0:ii0+K, :K0] = np.transpose(hess_z0z1,(0,2,1))

        else:
            hess_z1 = hess_z1 - np.eye(K)
            inverseBlocks.append(block_inv(hess_z1))
            corssBlocks.append(hess_z0z1)
            A = A + hess_z0

        i0 += K * T
        ii0 += K
    if inverse:
        H = invertLoop(inverseBlocks, corssBlocks, A)

    if return_tensor:
        return H

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
        # numba compatible
        nnz = np.prod(H.shape)
        ncols = sumK * T
        nrows = sumK * T
        spHess = csr.CSR(nrows, ncols, nnz, indptr, indices, H.flatten())

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


def all_trial_ll_grad_hess_factorized(dat, post_mean, tr_dict={}, isDict=True, returnLL=False, inverse=True,
                                      trialDur_variable=False):
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
    tmax = max(list(dat.trialDur.values()))
    indicesMax = np.zeros(tmax*sumK**2,dtype=np.int32)
    indptrMax = np.zeros(tmax*sumK+1,dtype=np.int32)

    for trNum in dat.trialDur.keys():
        stim, xList = dat.get_observations(trNum)
        # print('tr %d'%trNum)
        if trialDur_variable:
            T = dat.trialDur[trNum]
            rev_idx_sort = sortGradient_idx(T, dat.zdims, isReverse=True)
            idx_sort = np.argsort(rev_idx_sort)
            indicesMax[:T*sumK**2] = np.repeat(np.arange(T), sumK ** 2) * sumK + np.tile(np.arange(sumK), T * sumK)
            indptrMax[:T*sumK+1] = sumK * np.arange(sumK*T+1)
        else:
            indicesMax = indices
            indptrMax = indptr
            if indicesMax is None:
                T = dat.trialDur[trNum]
                indicesMax = np.repeat(np.arange(T), sumK ** 2) * sumK + np.tile(np.arange(sumK), T * sumK)
                indptrMax = sumK * np.arange(sumK*T+1)


        if isDict:
            zbar = post_mean[trNum]
        else:
            zbar = post_mean[tr_dict[trNum]]
        LL += factorized_logLike(dat, trNum, stim, xList, zbar=zbar, idx_sort=idx_sort, rev_idx_sort=rev_idx_sort)
        if returnLL:
            continue
        # t0 = perf_counter()

        hesInv, indices, indptr = hess_factorized_logLike(dat, trNum, stim, xList, zbar=zbar,
                                                          inverse=inverse, indices=indicesMax[:T*sumK**2], indptr=indptrMax[:T*sumK+1])
        # tt1 = perf_counter()
        grad[i0:i0 + dat.trialDur[trNum]*sumK], idx_sort, rev_idx_sort = grad_factorized_logLike(dat, trNum, stim,
                                                                                                     xList,
                                                                                                     zbar=zbar,
                                                                                                     idx_sort=idx_sort,
                                                                    rev_idx_sort=rev_idx_sort, useGauss=1, usePoiss=1)
        i0 += dat.trialDur[trNum]*sumK

        if first:
            indValSize = (totDur) * (sumK ** 2)
            indColSize = (totDur) * (sumK ** 2)
            indPTRSize = (totDur) * sumK + 1

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


def trialByTrialInference(dat, tol=10 ** -10, max_iter=100, max_having=30,disp_ll=False,init_zeros=False,
                     useNewton=True, trialDur_variable=False):
    eps_float = np.finfo(float).eps

    sumK = np.sum(dat.zdims)
    LL = 0
    tmax = max(list(dat.trialDur.values()))
    indicesMax = np.zeros(tmax * sumK ** 2, dtype=np.int32)
    indptrMax = np.zeros(tmax * sumK + 1, dtype=np.int32)
    Z0, tr_dict = preproc_post_mean_factorizedModel(dat, returnDict=False)
    Z0 = (1 - init_zeros) * Z0
    init_ll = all_trial_ll_grad_hess_factorized(dat, Z0, tr_dict=tr_dict, isDict=False, returnLL=True)
    print('initial MAP log post',init_ll)

    for trNum in dat.trialDur.keys():
        stim, xList = dat.get_observations(trNum)
        # print('tr %d'%trNum)

        T = dat.trialDur[trNum]
        rev_idx_sort = sortGradient_idx(T, dat.zdims, isReverse=False)
        idx_sort = np.argsort(rev_idx_sort)
        indicesMax[:T * sumK ** 2] = np.repeat(np.arange(T), sumK ** 2) * sumK + np.tile(np.arange(sumK), T * sumK)
        indptrMax[:T * sumK + 1] = sumK * np.arange(sumK * T + 1)

        zbar = Z0[tr_dict[trNum]]
        ii = 0
        delta_ll = np.inf

        tmpZ = zbar.copy()
        # ll_hist = []
        while ii < max_iter and delta_ll > tol:
            log_like = factorized_logLike(dat, trNum, stim, xList, zbar=zbar, idx_sort=idx_sort, rev_idx_sort=rev_idx_sort)
            if ii == 0:
                print('trial %d:'%trNum, 'initial LL',log_like)

            hess_ll = hess_factorized_logLike(dat, trNum, stim, xList, zbar=zbar,
                                                              inverse=True, indices=indicesMax[:T * sumK ** 2],
                                                              indptr=indptrMax[:T * sumK + 1])[0]
            grad_ll = grad_factorized_logLike(dat, trNum, stim, xList, zbar=zbar, idx_sort=idx_sort,rev_idx_sort=rev_idx_sort,
                                                                                                       useGauss=1,
                                                                                                       usePoiss=1)[0]

            # if disp_ll:
            #     print('tr %d newton optim iter'%trNum, ii, 'log-like', log_like)
            if useNewton:
                delt = hess_ll.mult_vec(grad_ll)
            else:
                delt = -grad_ll  # hess_ll.mult_vec(grad_ll)
                # print('sparse solve time', perf_counter() - t0)
            step = 1
            new_ll = -np.inf
            step_halv = 0

            # t0 = perf_counter()
            while new_ll < log_like and step_halv < max_having:
                tmpZ = zbar - step * delt.reshape(zbar.shape)
                new_ll = factorized_logLike(dat, trNum, stim, xList, zbar=tmpZ, idx_sort=idx_sort, rev_idx_sort=rev_idx_sort)
                #all_trial_ll_grad_hess_factorized(dat, tmpZ, tr_dict=tr_dict, isDict=False, returnLL=True)

                delta_ll = new_ll - log_like
                #print('halving #%d, ' % step_halv, delta_ll)
                step = step / 2
                step_halv += 1

            # print('step halving time', perf_counter() - t0)
            if new_ll == -np.inf or (log_like - new_ll > np.sqrt(eps_float)):
                # print(log_like - new_ll)
                LL += log_like
                Z0[tr_dict[trNum]] = zbar
                #ll = all_trial_ll_grad_hess_factorized(dat, tmpZ, tr_dict=tr_dict, isDict=False, returnLL=True)
                continue
            #ll_hist.append(new_ll)
            zbar = tmpZ

            ii += 1
        LL += new_ll
        print('trial %d:' % trNum, 'final LL', new_ll)
        Z0[tr_dict[trNum]] = zbar
    print('final MAP log post', LL)
    return Z0,tr_dict,LL


def newton_optim_map(dat, tol=10 ** -10, max_iter=100, max_having=30,disp_ll=False,init_zeros=False,
                     useNewton=True, trialDur_variable=False, randInit=False):
    eps_float = np.finfo(float).eps
    Z0, tr_dict = preproc_post_mean_factorizedModel(dat,returnDict=False)
    Z0 = (1-init_zeros)*Z0
    if all(Z0==0) and randInit:
        # random small coeff init
        llabs = np.inf
        n = 1
        while np.isinf(llabs):
            Z0 = 0.5**n * np.random.normal(size=Z0.shape)
            llabs = np.abs(all_trial_ll_grad_hess_factorized(dat, Z0, tr_dict=tr_dict, isDict=False, returnLL=True))
            n += 1
    ii = 0
    delta_ll = np.inf

    tmpZ = Z0.copy()
    ll_hist = []
    while ii < max_iter and delta_ll > tol:

        # t0 = perf_counter()
        log_like, grad_ll, hess_ll = all_trial_ll_grad_hess_factorized(dat, Z0, tr_dict=tr_dict, isDict=False,trialDur_variable=trialDur_variable)
        if disp_ll:
            print('newton optim iter', ii, 'log-like', log_like)
        if useNewton:
            delt = hess_ll.mult_vec(grad_ll)
        else:
            delt = -grad_ll#hess_ll.mult_vec(grad_ll)
        # print('sparse solve time', perf_counter() - t0)
        step = 1
        new_ll = -np.inf
        step_halv = 0

        # t0 = perf_counter()
        while new_ll < log_like and step_halv < max_having:

            tmpZ = Z0 - step * delt.reshape(Z0.shape)
            new_ll = all_trial_ll_grad_hess_factorized(dat, tmpZ, tr_dict=tr_dict, isDict=False, returnLL=True)

            delta_ll = new_ll - log_like
            #print('halving #%d, ' % step_halv, delta_ll)
            step = step / 2
            step_halv += 1

        # print('step halving time', perf_counter() - t0)
        if new_ll == -np.inf or (log_like - new_ll > np.sqrt(eps_float)):
            print(log_like - new_ll)
            ll = all_trial_ll_grad_hess_factorized(dat, tmpZ, tr_dict=tr_dict, isDict=False, returnLL=True)
            return Z0, False, tr_dict, ll, ll_hist
        ll_hist.append(new_ll)
        Z0 = tmpZ
        # if disp_ll:
        #     print('delta_ll', delta_ll)

        ii += 1
    # only criteria for convergence used
    flag_convergence = delta_ll <= tol
    ll = all_trial_ll_grad_hess_factorized(dat, tmpZ, tr_dict=tr_dict, isDict=False, returnLL=True)
    return Z0, flag_convergence, tr_dict, ll, ll_hist



if __name__ == '__main__':

    from gen_synthetic_data import *
    from copy import deepcopy
    from data_processing_tools_factorized import fast_stackCSRHes
    import matplotlib.pylab as plt
    from inference import multiTrialInference

    T = 50
    data = dataGen(3,T=T,infer=False)
    sub = data.cca_input.subSampleTrial(np.arange(1,4))
    dat = data.cca_input
    # test factorized
    trNum = 0
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

    post_mean,td = preproc_post_mean_factorizedModel(dat,returnDict=False)
    # ll, grad, hesInv = all_trial_ll_grad_hess_factorized(dat, post_mean)

    zmap,success,td,ll,ll_hist = newton_optim_map(dat, tol=10 ** -10, max_iter=100, max_having=20,
                     indices=None, indptr=None,
                     indices_up=None, indptr_up=None, disp_ll=True, init_zeros=True,useNewton=True)

    dat2 = deepcopy(dat)
    reconstruct_post_mean_and_cov(dat2,zmap,td)
    multiTrialInference(dat)
    plt.figure(figsize=(10,5))
    cc = 1
    for k in range(2):
        plt.subplot(2,2,k+cc)
        plt.title('Factorized: coord %d'%(k+1))
        std_fact = np.sqrt(dat2.posterior_inf[2].cov_t[0][:, k, k])
        p, = plt.plot(dat2.posterior_inf[2].mean[0][k])
        plt.fill_between(np.arange(T), dat2.posterior_inf[2].mean[0][k] - 1.96 * std_fact,
                         dat2.posterior_inf[2].mean[0][k] + 1.96 * std_fact, alpha=0.4, color=p.get_color())
        plt.plot(dat.ground_truth_latent[2][:,k],color='k')
        cc+=1

        plt.subplot(2, 2, k + cc)
        plt.title('GP: coord %d' % (k + 1))
        std_gp = np.sqrt(dat.posterior_inf[2].cov_t[0][:, k, k])
        p, = plt.plot(dat.posterior_inf[2].mean[0][k])
        plt.fill_between(np.arange(T), dat.posterior_inf[2].mean[0][k] - 1.96 * std_gp,
                         dat.posterior_inf[2].mean[0][k] + 1.96 * std_gp, alpha=0.4, color=p.get_color())
        plt.plot(dat.ground_truth_latent[2][:, k], color='k')
    plt.tight_layout()

    # func = lambda z0: -all_trial_ll_grad_hess_factorized(dat, z0, tr_dict=td, isDict=False, returnLL=True)
    # grad_func = lambda z0: -all_trial_ll_grad_hess_factorized(dat, z0, tr_dict=td, isDict=False, returnLL=False)[1]
    # hessian_func = lambda z0: -all_trial_ll_grad_hess_factorized(dat, z0, tr_dict=td, isDict=False, returnLL=False,inverse=False)[2].to_scipy().toarray()

    # stim,xList = dat.get_observations(10)
    # H = -hess_factorized_logLike(dat, trNum, stim, xList, zbar=None, idx_sort=None,
    #                         rev_idx_sort=None, indices=None, indptr=None, inverse=True,
    #                         return_tensor=True)

    # apgrad = approx_grad(post_mean*0,post_mean.shape[0],func,epsi=10**-5)
    # gr = grad_func(post_mean*0)
    # hes = hessian_func(post_mean*0)
    # aphes = approx_grad(post_mean*0,(gr.shape[0],)*2, grad_func,10**-5)
