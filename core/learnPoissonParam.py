"""
Some of the code here is adapted from Machens et al. implementation of P-GPFA.
"""
import numpy as np
from scipy.optimize import minimize
from data_processing_tools import approx_grad

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
        if not C1 is None:
            Cout = np.hstack((C[yd], C1[yd]))
            CC = np.outer(Cout, Cout)
            covC = np.einsum('tij,j->ti', cov_post[:, :xdim, :xdim], C[yd]) + np.einsum('tij,j->ti', cov_post[:, :xdim, xdim:], C1[yd])
            covC1 = np.einsum('tij,j->ti', cov_post[:, xdim:, xdim:], C1[yd]) + np.einsum('tij,j->ti', cov_post[:, xdim:, :xdim], C[yd])

        else:
            Cout = C[yd]
            CC = np.outer(C[yd, :], C[yd, :])
            covC = np.einsum('tij,j->ti', cov_post, C[yd])

        EXP = np.exp(0.5 * np.sum(cov_post.reshape(cov_post.shape[0], (xdim+xdim1) ** 2) * CC.reshape((xdim+xdim1) ** 2), axis=1) + \
                     d[yd] + np.einsum('j,tj->t', Cout, mean_post))
        dyhat_C[yd] = np.einsum('t,tj->j', EXP, mean_post[:,:xdim] + covC)
        if not C1 is None:
            dyhat_C1[yd] = np.einsum('t,tj->j', EXP, mean_post[:, xdim:] + covC1)
        dyhat_d[yd] = EXP.sum()
    dhh_d = x.sum(axis=0)
    dhh_Call = np.einsum('ti,tj->ij', x, mean_post)
    dhh_C = dhh_Call[:,:xdim]
    if not C1 is None:
        dhh_C1 = dhh_Call[:, xdim:]
        return np.hstack(((dhh_C - dyhat_C).flatten(), (dhh_C1 - dyhat_C1).flatten(), dhh_d - dyhat_d))
    return np.hstack(((dhh_C - dyhat_C).flatten(), dhh_d - dyhat_d))#dhh_d - dyhat_d, dhh_C - dyhat_C

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

    # C1=0
    # C=0
    # d=0


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