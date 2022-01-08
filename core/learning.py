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

    # np.random.seed(4)
    #
    # # create the input data
    # preproc = emptyStruct()
    # preproc.numTrials = 1
    # preproc.ydim = 50
    # preproc.binSize = 50
    # preproc.T = np.array([50])
    # tau0 = np.array([0.9])#np.array([0.9, 0.2, 0.4, 0.2, 0.8])
    # K0 = len(tau0)
    # epsNoise = 0.000001
    # K_big = makeK_big(K0, tau0, None, preproc.binSize, epsNoise=epsNoise, T=preproc.T[0], computeInv=False)[1]
    # z0 = np.random.multivariate_normal(mean=np.zeros(K0 * preproc.T[0]), cov=K_big, size=1).reshape(K0, preproc.T[0]).T
    #
    # # create the stim vars
    # PsiInv = np.eye(2)
    # W = np.random.normal(size=(2, K0))
    # d = np.zeros(2)
    # preproc.covariates = {}
    # preproc.covariates['var1'] = [np.random.multivariate_normal(mean=np.dot(W, z0.T)[0], cov=np.eye(preproc.T[0]))]
    # preproc.covariates['var2'] = [np.random.multivariate_normal(mean=np.dot(W, z0.T)[1], cov=np.eye(preproc.T[0]))]
    # trueStimPar = {'W0': W, 'd': d, 'PsiInv': PsiInv}
    #
    # # create the counts
    # tau = np.array([1.1, 1.3])
    # K_big = makeK_big(len(tau), tau, None, preproc.binSize, epsNoise=epsNoise, T=preproc.T[0], computeInv=False)[1]
    # z1 = np.random.multivariate_normal(mean=np.zeros(preproc.T[0] * len(tau)), cov=K_big, size=1).reshape(len(tau),
    #                                                                                                       preproc.T[
    #                                                                                                           0]).T
    #
    # W1 = np.random.normal(size=(preproc.ydim, len(tau)))
    # W0 = np.random.normal(size=(preproc.ydim, K0))
    # d = -0.2
    # preproc.data = [
    #     {'Y': np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', W0, z0) + np.einsum('ij,tj->ti', W1, z1) + d))}]
    #
    # # create the true observation par dict
    # trueObsPar = [{
    #     'W0': W0,
    #     'W1': W1,
    #     'd': np.ones(preproc.ydim) * d
    # }]
    #
    # # true Prior params
    # truePriorPar = [{'tau': tau0}, {'tau': tau}]
    #
    # # create the data struct
    # struc = P_GPCCA(preproc, ['var1', 'var2'], ['PPC'], np.array(['PPC'] * preproc.ydim),
    #                       np.ones(preproc.ydim, dtype=bool))
    # struc.initializeParam([K0, z1.shape[1]])
    # stim, xList = struc.get_observations(0)
    # zstack = np.hstack((z0.flatten(), z1.flatten()))
    #
    # ## inference of the latent variables
    # # set the pars to the true
    # struc.xPar = trueObsPar
    # struc.priorPar = truePriorPar
    # struc.stimPar = trueStimPar
    # struc.epsNoise = epsNoise
    #
    # # call the optimization function
    # meanPost, covPost = inferTrial(struc, 0)
    # mean_t, cov_t = retrive_t_blocks_fom_cov(struc, 0, 1, [meanPost], [covPost])
    #
    # PARStack = np.hstack((W1.flatten(), np.ones(preproc.ydim) * d))
    # # Wsize =
    # grad_fun = lambda xx: -grad_expectedLLPoisson(xList[0], xx[:np.prod(W1.shape)].reshape(W1.shape),
    #                                                 xx[np.prod(W1.shape):], mean_t[:,K0:], cov_t[:,K0:,K0:])
    #
    # func = lambda xx: -expectedLLPoisson(xList[0], xx[:np.prod(W1.shape)].reshape(W1.shape),
    #                                                 xx[np.prod(W1.shape):], mean_t[:, K0:], cov_t[:, K0:, K0:])
    #
    # print('first optimization')
    # res = minimize(func,np.zeros(PARStack.shape[0]),jac=grad_fun)
    # app_grad1 = approx_grad(PARStack,PARStack.shape[0],func,10**-4)
    #
    #
    # # repeat with 2 factors
    # PARStack2 = np.hstack((W0.flatten(), W1.flatten(), np.ones(preproc.ydim) * d))
    # # Wsize =
    # grad_fun2 = lambda xx: -grad_expectedLLPoisson(xList[0], xx[:np.prod(W0.shape)].reshape(W0.shape),
    #                                      xx[np.prod(W0.shape)+np.prod(W1.shape):], mean_t, cov_t,
    #                                      C1=xx[np.prod(W0.shape):np.prod(W0.shape)+np.prod(W1.shape)].reshape(W1.shape))
    #
    # func2 = lambda xx: -expectedLLPoisson(xList[0], xx[:np.prod(W0.shape)].reshape(W0.shape),
    #                                      xx[np.prod(W0.shape)+np.prod(W1.shape):], mean_t, cov_t,
    #                                      C1=xx[np.prod(W0.shape):np.prod(W0.shape)+np.prod(W1.shape)].reshape(W1.shape))
    #
    # print('second optimization')
    # res2 = minimize(func2, np.zeros(PARStack2.shape[0]), jac=grad_fun2)
    #
    # if preproc.T[0] >= 500:
    #     plt.figure(figsize=[6.4 , 3.54])
    #     plt.title('M-step Poisson Observation')
    #     plt.plot(PARStack2,label='true parameter')
    #     plt.plot(res2.x,label='recovered parameter')
    #     plt.ylabel('parameter palue')
    #     plt.xlabel('parameter index')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig('/Users/edoardo/Work/Code/P-GPCCA/inference_syntetic_data/M_step_poisson.jpg')
    #
    # app_grad = approx_grad(PARStack2,PARStack2.shape[0],func2,10**-5)
    # grad = grad_fun2(PARStack2)
    #
    # Wnew, dnew, PsiNew = MStepGauss()
    # # plt.figure()
    # # plt.subplot(121)
    # # plt.scatter(app_grad1, grad_fun(PARStack))
    # # plt.subplot(122)
    # # plt.scatter(app_grad,grad_fun2(PARStack2))
    # # print('GRADIENT NOT WORKING! fix')

    dat = np.load('/Users/edoardo/Work/Code/P-GPCCA/inference_syntetic_data/sim_150Trials.npy', allow_pickle=True).all()
    dat.cca_input = dat.cca_input.subSampleTrial(np.arange(1, 4))
    multiTrialInference(dat.cca_input)
    learn_GaussianParams(dat.cca_input,test=True)