"""
Some of the code here is adapted from Machens et al. implementation of P-GPFA.
"""
import numpy as np
from scipy.optimize import minimize
from data_processing_tools import approx_grad,block_inv, fast_stackCSRHes_memoryPreAllocation
import scipy.sparse as sparse
import csr

def expectedLLPoisson(x, C, d, mean_post,cov_post, C1=None):
    '''
    The Spike count observation expected likelihood
    '''
    ydim, xdim = C.shape
    mean0 = mean_post[:, :xdim]
    if not C1 is None:
        xdim1 = C1.shape[1]
        mean1 = mean_post[:, xdim:]
    else:
        xdim1 = 0
    yhat = 0
    for yd in range(ydim):

        if not C1 is None:
            Cout = np.hstack((C[yd],C1[yd]))
            CC = np.outer(Cout, Cout)
        else:
            Cout = C[yd]
            CC = np.outer(C[yd, :], C[yd, :])
        yhat += np.exp(0.5*np.sum(cov_post.reshape(cov_post.shape[0],(xdim+xdim1)**2) * CC.reshape((xdim+xdim1)**2),axis=1)+\
            d[yd] + np.einsum('j,tj->t', Cout, mean_post))
    yhat = yhat.sum()
    if not C1 is None:
        hh = np.einsum('tj,jk,tk->', x, C1, mean1, optimize=True) + np.einsum('tj,jk,tk->', x, C, mean0, optimize=True) + np.dot(x.sum(axis=0), d)
    else:
        hh = np.einsum('tj,jk,tk->', x, C, mean0, optimize=True) + np.dot(x.sum(axis=0),d)
    return hh - yhat


def grad_expectedLLPoisson(x, C, d, mean_post, cov_post, C1=None):
    ydim, xdim = C.shape
    # mean0 = mean_post[:,:xdim]
    if not C1 is None:
        xdim1 = C1.shape[1]
        # mean1 = mean_post[:, xdim:]
        dyhat_C1 = np.zeros((ydim, xdim1))
    else:
        xdim1 = 0
    dyhat_C = np.zeros((ydim, xdim))
    dyhat_d = np.zeros((ydim,))
    for yd in range(ydim):
        if C1 is None:
            dyhat_d[yd], dyhat_C[yd] = grad_Poisson_ithunit(C[yd], C1[yd], d[yd], mean_post, cov_post)
        else:
            dyhat_d[yd], dyhat_C[yd], dyhat_C1[yd] = grad_Poisson_ithunit(C[yd], C1[yd], d[yd], mean_post, cov_post)
    dhh_d = x.sum(axis=0)
    dhh_Call = np.einsum('ti,tj->ij', x, mean_post)
    dhh_C = dhh_Call[:,:xdim]
    if not C1 is None:
        dhh_C1 = dhh_Call[:, xdim:]
        return np.hstack(((dhh_C - dyhat_C).flatten(), (dhh_C1 - dyhat_C1).flatten(), dhh_d - dyhat_d))
    return np.hstack(((dhh_C - dyhat_C).flatten(), dhh_d - dyhat_d))#dhh_d - dyhat_d, dhh_C - dyhat_C

def grad_Poisson_ithunit(C, C1, d, mean_post, cov_post):
    xdim = C.shape[0]
    xdim1 = C1.shape[0]
    if not C1 is None:
        Cout = np.hstack((C, C1))
        CC = np.outer(Cout, Cout)
        covC = np.einsum('tij,j->ti', cov_post[:, :xdim, :xdim], C) + np.einsum('tij,j->ti',
                                                                                    cov_post[:, :xdim, xdim:], C1)
        covC1 = np.einsum('tij,j->ti', cov_post[:, xdim:, xdim:], C1) + np.einsum('tij,j->ti',
                                                                                      cov_post[:, xdim:, :xdim], C)
    else:
        Cout = C
        CC = np.outer(C, C)
        covC = np.einsum('tij,j->ti', cov_post, C)

    EXP = np.exp(
        0.5 * np.sum(cov_post.reshape(cov_post.shape[0], (xdim + xdim1) ** 2) * CC.reshape((xdim + xdim1) ** 2),
                     axis=1) + \
        d + np.einsum('j,tj->t', Cout, mean_post))
    dyhat_C = np.einsum('t,tj->j', EXP, mean_post[:, :xdim] + covC)
    dyhat_d = EXP.sum()
    if not C1 is None:
        dyhat_C1 = np.einsum('t,tj->j', EXP, mean_post[:, xdim:] + covC1)
        return dyhat_d,dyhat_C,dyhat_C1
    return dyhat_d, dyhat_C


def hess_Poisson_ithunit(parStack, mean_t, cov_t, inverse=True):
    """
    Use the factorization of the model to return a sparse block hessian of the parameters and then use the newton
    optim step to learn efficiently the M-step
    :param parStack:
        K0+K1+1 projection weights and intercept for an initial neuron
    :param mean_t
        T x (K0+K1) posterior mean
    :param cov_t
        T x (K0+K1) posterior mean
    :return:
    """
    h = parStack[-1]
    W = parStack[:-1]
    hess = np.zeros((parStack.shape[0],parStack.shape[0]))
    covW = np.einsum('tij,j->ti', cov_t, W)
    WW = np.outer(W, W)
    EXP = np.exp(
        0.5 * np.sum(cov_t.reshape(cov_t.shape[0], (W.shape[0]) ** 2) * WW.reshape((W.shape[0]) ** 2),
                     axis=1) + h + np.einsum('j,tj->t', W, mean_t))

    covW_p_mu = covW+mean_t
    hess[-1, -1] = -EXP.sum()
    tmp = np.einsum('ti,t->it',covW_p_mu,EXP)
    tmp = -np.einsum('it,tj->ij ',tmp,covW_p_mu)
    hess[:-1, :-1] = tmp - np.einsum('t,tij->ij',EXP,cov_t)
    hess[-1,:-1] = -np.einsum('t,ti->i', EXP,covW_p_mu)
    hess[:-1,-1] = hess[-1,:-1].T
    if inverse:
        return np.linalg.inv(hess)
    return hess

def hess_Poisson_all_trials(d, W0, W1, idx_latent, data, trial_list=None, inverse=True, sparse=True):
    if trial_list is None:
        trial_list = list(data.trialDur.keys())
    xDim, K1 = W1.shape
    K0 = W0.shape[1]
    T = np.sum(list(data.trialDur.values()))
    x, mean_post, cov_post = compileTrialStackedObsAndLatent(data, idx_latent, trial_list, T, xDim, K0, K1)
    HessBlock = np.zeros((xDim, K0+K1+1, K0+K1+1))
    for icoord in range(xDim):
        HessBlock[icoord] = hess_Poisson_ithunit(np.hstack((W0[icoord], W1[icoord],d[icoord])), mean_post, cov_post, inverse=False)
    if inverse:
        HessBlock = block_inv(HessBlock)
    if sparse:
        HessBlock = csr_blockdiag_hessian_block(HessBlock)
    return HessBlock

def grad_Poisson_all_trials(d, W0, W1, idx_latent, data, trial_list=None):
    if trial_list is None:
        trial_list = list(data.trialDur.keys())
    xDim, K1 = W1.shape
    K0 = W0.shape[1]
    T = np.sum(list(data.trialDur.values()))
    x, mean_post, cov_post = compileTrialStackedObsAndLatent(data, idx_latent, trial_list, T, xDim, K0, K1)
    dhh_d = x.sum(axis=0)
    dhh_Call = np.einsum('ti,tj->ij', x, mean_post)
    GradBlock = np.zeros((xDim, K0+K1+1))
    for icoord in range(xDim):
        C = W0[icoord]
        C1 = W1[icoord]
        di = d[icoord]
        grad_d,grad_C,grad_C1 = grad_Poisson_ithunit(C, C1, di, mean_post, cov_post)
        grad_d = dhh_d[icoord] - grad_d
        grad_C_all = dhh_Call[icoord] - np.hstack((grad_C,grad_C1))
        GradBlock[icoord] = np.hstack((grad_C_all, grad_d))

    return GradBlock.flatten()


def newton_opt_CSR(func, grad, hessInv, Z0,tol=10**-8,max_iter=1000, disp_eval=True,max_having=30):
    """
    :param func: the expected log loss for all trials
    :param grad: the grad of the expected for all trial
    :param hessInv: the hess of the expected for all trial, must return a csr.CSR sparse matrix
    :param Z0: initial condition
    :param rot:
    :param tol:
    :param max_iter:
    :param disp_eval:
    :param max_having:
    :param rotation: the rotation to apply to the gradient after the computation for aligning it to the hess
    :return:
    """
    eps_float = np.finfo(float).eps

    tmpZ = Z0.copy()
    feval_hist = []
    ii  = 0
    delta_eval = np.inf
    while ii < max_iter and delta_eval > tol:

        # t0 = perf_counter()
        feval = func(Z0)
        fgrad = grad(Z0)
        fhess = hessInv(Z0)
        if disp_eval:
            print('newton optim iter', ii, 'log-like', feval)

        delt = fhess.mult_vec(fgrad)
        # print('sparse solve time', perf_counter() - t0)
        step = 1
        new_eval = -np.inf
        step_halv = 0

        # t0 = perf_counter()
        while new_eval < feval and step_halv < max_having:
            tmpZ = Z0 - step * delt.reshape(Z0.shape)
            new_eval = func(tmpZ)

            delta_eval = new_eval - feval
            # if disp_ll:
            #     print('halv step log-like', step_halv + 1, new_ll)
            step = step / 2
            step_halv += 1

        # print('step halving time', perf_counter() - t0)
        if new_eval == -np.inf or (feval - new_eval > np.sqrt(eps_float)):
            print(feval - new_eval)
            feval = func(Z0)
            return Z0, feval, feval_hist
        feval_hist.append(new_eval)
        Z0 = tmpZ
        # if disp_ll:
        #     print('delta_ll', delta_ll)

        ii += 1
    # only criteria for convergence used
    flag_convergence = delta_eval <= tol
    feval = func(Z0)
    return Z0, feval, feval_hist



# def all_trial_poissonLL(C, d, x_list, mean_post, cov_post, C1=None, isGrad=False):
#     """
#     Looop over trial and compute expected LL or its gradent for poisson obs
#     :param C:
#     :param d:
#     :param x_list:
#     :param mean_post:
#     :param cov_post:
#     :param C1:
#     :param isGrad:
#     :return:
#     """
#     if isGrad:
#         func = grad_expectedLLPoisson
#     else:
#         func = expectedLLPoisson
#     f = 0
#     for i in range(len(x_list)):
#         x = x_list[i]
#         meanPost = mean_post[i]
#         covPost = cov_post[i]
#         f = f + func(x, C, d, meanPost, covPost, C1=C1)
#
#     return f
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

def multiTrial_PoissonLL(W0, W1, d, data, idx_latent, trial_num=None, isGrad=False, trial_list=None, test=False):
    """

    :param isGrad: True return the gradient, False returns the funcion evaluation
    :return:
    """
    if isGrad:
        func = grad_expectedLLPoisson
    else:
        func = expectedLLPoisson

    if trial_num is None:
        trial_num = len(list(data.trialDur.values()))

    if trial_list is None:
        trial_list = list(data.trialDur.keys())

    xDim,K0 = W0.shape
    K1 = W1.shape[1]
    T = 0
    for tr in trial_list:
        T += data.trialDur[tr] #np.sum(list(data.trialDur.values()))

    x, mean_post, cov_post = compileTrialStackedObsAndLatent(data, idx_latent, trial_list, T, xDim, K0, K1)

    f = func(x, W0, d, mean_post, cov_post, C1=W1)/trial_num
    if test:
        ff = lambda xx: -expectedLLPoisson(x, xx[:K0 * xDim].reshape(xDim, K0), xx[(K0+K1)*xDim:], mean_post,
                                          cov_post, C1=xx[K0 * xDim:(K0+K1)*xDim].reshape(xDim, K1))
        grad_ff = lambda xx: -grad_expectedLLPoisson(x, xx[:K0 * xDim].reshape(xDim, K0), xx[(K0+K1)*xDim:],
                                                     mean_post, cov_post, C1=xx[K0 * xDim:(K0+K1)*xDim].reshape(xDim, K1))
        xx0 = np.zeros(xDim*(K0+K1)+xDim)
        ap_grad = approx_grad(xx0,xx0.shape[0],ff,10**-4)
        grad = grad_ff(xx0)
        err = np.abs(grad - ap_grad).mean()/np.abs(ap_grad).mean()
        return err
    return f

def all_trial_PoissonLL(W0,W1,d, data, idx_latent, block_trials=None, isGrad=False):
    if block_trials is None:
        block_trials = len(data.trialDur.keys())
    all_trials = list(data.trialDur.keys())

    trial_list = []
    nBlocks = int(np.ceil(len(all_trials) /block_trials))
    for k in range(nBlocks):
        trial_list.append(all_trials[k*block_trials:(k+1)*block_trials])
    f = 0
    for tl in trial_list:
        f = f + multiTrial_PoissonLL(W0,W1,d,data, idx_latent, trial_num=len(all_trials), isGrad=isGrad, trial_list=tl, test=False)
    return f


def csr_blockdiag_hessian_block(hess, test=False):
    if test:
        sparse_hess_scipy = sparse.block_diag(hess)

    stackNum = hess.shape[0]

    indPTRSize = (hess.shape[1]) * stackNum + 1
    indValSize = np.prod(hess.shape)
    indColSize = np.prod(hess.shape)
    vals = -np.ones(indValSize, dtype=np.float64, order='C')
    indcols = -np.ones(indColSize, dtype=np.int32, order='C')
    rowptr = np.zeros(indPTRSize, dtype=np.int32, order='C')
    i0Val = 0
    i0PTR = 0
    nrows = 0
    ncols = 0
    nnz = 0

    indices = np.tile(np.arange(hess.shape[1], dtype=np.int32), hess.shape[1])
    indptr = np.arange(0, hess.shape[1] ** 2 + 1, hess.shape[1], dtype=np.int32)
    nnz0 = hess.shape[1]**2
    for k in range(stackNum):
        values = hess[k].flatten()
        hes0 = csr.CSR(hess.shape[1], hess.shape[2], nnz0, indptr, indices, values)
        (nrows, ncols, nnz, rowptr, indcols,
         vals, i0Val, i0PTR) = fast_stackCSRHes_memoryPreAllocation(vals, rowptr, indcols, nnz, nrows,
                                                                    ncols, i0PTR, i0Val, hes0, None)
    sparse_hess = csr.CSR(nrows, ncols, nnz, rowptr, indcols,vals)

    if test:
        assert((sparse_hess_scipy - sparse_hess.to_scipy()).nnz == 0)

    return sparse_hess


def poissonELL_Sparse(par, idx_latent, data, rotInv):
    xx = par[rotInv]
    K0 = data.zdims[0]
    K1 = data.zdims[idx_latent]
    N = data.xPar[idx_latent - 1]['d'].shape[0]
    W0 = xx[:K0 * N].reshape(N, K0)
    W1 = xx[K0 * N:(K0 * N) + (K1 * N)].reshape(N, K1)
    d = xx[(K0 * N) + (K1 * N):]
    trNum = len(data.trialDur.keys())
    f = multiTrial_PoissonLL(W0, W1, d, data, idx_latent, trial_num=None, isGrad=False, trial_list=None, test=False) * trNum
    return f


def grad_poissonELL_Sparse(par, idx_latent, data, rotInv):
    xx = par[rotInv]
    K0 = data.zdims[0]
    K1 = data.zdims[idx_latent]
    N = data.xPar[idx_latent - 1]['d'].shape[0]
    W0 = xx[:K0 * N].reshape(N, K0)
    W1 = xx[K0 * N:(K0 * N) + (K1 * N)].reshape(N, K1)
    d = xx[(K0 * N) + (K1 * N):]
    grd = grad_Poisson_all_trials(d, W0, W1, idx_latent, data, trial_list=None)
    grd = grd  # [rotinv]
    return grd


def hess_poissonELL_Sparse(par, idx_latent, data, rotInv, inverse=True, sparse=True):
    xx = par[rotInv]
    K0 = data.zdims[0]
    K1 = data.zdims[idx_latent]
    N = data.xPar[idx_latent - 1]['d'].shape[0]
    W0 = xx[:K0 * N].reshape(N, K0)
    W1 = xx[K0 * N:(K0 * N) + (K1 * N)].reshape(N, K1)
    d = xx[(K0 * N) + (K1 * N):]
    hes = hess_Poisson_all_trials(d, W0, W1, idx_latent, data, trial_list=None, inverse=inverse, sparse=sparse)
    return hes


if __name__ == '__main__':
    from inference import *
    from time import perf_counter

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

    dat = np.load('/Users/edoardo/Work/Code/P-GPCCA/inference_syntetic_data/sim_150Trials.npy', allow_pickle=True).all()

    # extract initial par values
    idx_latent = 1
    C = dat.xPar[idx_latent - 1]['W0']
    C1 = dat.xPar[idx_latent - 1]['W1']
    d = dat.xPar[idx_latent - 1]['d']
    N,K0 = C.shape
    K1=C1.shape[1]

    f = all_trial_PoissonLL(C,C1,d, dat, idx_latent, block_trials=31, isGrad=False)
    g = all_trial_PoissonLL(C,C1,d,dat, idx_latent, block_trials=31, isGrad=True)
    xx = np.hstack((C.flatten(),C1.flatten(),d))
    func = lambda xx: -all_trial_PoissonLL(xx[:N*K0].reshape(N,K0), xx[N*K0:N*(K0+K1)].reshape(N,K1),xx[N*(K0+K1):],
                               dat, idx_latent, block_trials=31, isGrad=False)
    gr_func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0), xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                          xx[N * (K0 + K1):],
                                          dat, idx_latent, block_trials=31, isGrad=True)

    # ap_grad = approx_grad(xx,xx.shape[0],func,10**-5)
    res = minimize(func,np.zeros(xx.shape),jac=gr_func,method='L-BFGS-B',tol=10**-10)