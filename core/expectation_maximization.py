import numpy as np
from inference import multiTrialInference
from learnGaussianParam import learn_GaussianParams,full_GaussLL
from learnPoissonParam import all_trial_PoissonLL
from learnGPParams import all_trial_GPLL
from copy import deepcopy
from scipy.optimize import minimize

def expectation_mazimization(data, maxIter=10, tol=10**-3):
    # save the intial parameters
    data.init_priorPar = deepcopy(data.priorPar)
    data.init_xPar = deepcopy(data.xPar)
    data.init_stimPar = deepcopy(data.stimPar)

    # compute initial expected LL
    LL_list = []
    LL = full_GaussLL(data)
    for k in range(len(data.zdims) - 1):
        C = data.xPar[k]['W0']
        C1 = data.xPar[k]['W1']
        d = data.xPar[k]['d']
        N, K0 = C.shape
        K1 = C1.shape[1]
        parStack = np.hstack((C.flatten(), C1.flatten(), d))
        func = lambda xx: all_trial_PoissonLL(xx[:N * K0].reshape(N, K0), xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                               xx[N * (K0 + K1):],
                                               data, k + 1, block_trials=100, isGrad=False)
        LL += func(parStack)

    for k in range(len(data.zdims)):
        tau = data.priorPar[k]['tau']
        lam0 = 2 * np.log(((tau * data.binSize / 1000)))
        func = lambda lam0: all_trial_GPLL(lam0, data, k, block_trials=1, isGrad=False)
        LL += func(lam0)

    LL_list.append(LL)

    # start the EM
    for ii in range(maxIter):
        print('EM iteration: %d/%d'%(ii+1,maxIter))
        # infer latent
        print('- E-step')
        multiTrialInference(data)

        # learn gaussian params
        print('- Gaussian M-step')
        learn_GaussianParams(data, test=False, isMPI=False)
        nLL = -full_GaussLL(data)

        # learn Poisson obs param
        for k in range(len(data.zdims)-1):
            C = data.xPar[k]['W0']
            C1 = data.xPar[k]['W1']
            d = data.xPar[k]['d']
            N, K0 = C.shape
            K1 = C1.shape[1]
            parStack = np.hstack((C.flatten(), C1.flatten(), d))
            func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0), xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                                   xx[N * (K0 + K1):],
                                                   data, k+1, block_trials=100, isGrad=False)
            gr_func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0),
                                                      xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                                      xx[N * (K0 + K1):],
                                                      data, k+1, block_trials=100, isGrad=True)

            print('- Poisson M-step, observed population: %d/%d'%(k+1,len(data.zdims)-1))
            res = minimize(func, np.zeros(parStack.shape), jac=gr_func, method='L-BFGS-B', tol=10 ** -10)
            if res.success or res.fun < func(parStack):
                data.xPar[k]['W0'] = res.x[:N * K0].reshape(N, K0)
                data.xPar[k]['W1'] = res.x[N * K0:N * (K0 + K1)].reshape(N, K1)
                data.xPar[k]['d'] = res.x[N * (K0 + K1):]
                nLL += res.fun
            else:
                nLL += func(parStack)
        # learn GP param
        for k in range(len(data.zdims)):
            tau = data.priorPar[k]['tau']
            lam0 = 2 * np.log(((tau * data.binSize/1000)))
            func = lambda lam0: -all_trial_GPLL(lam0, data, k, block_trials=1, isGrad=False)
            gr_func = lambda lam0: -all_trial_GPLL(lam0, data, k, block_trials=1, isGrad=True)

            print('- GP M-step, latent factor: %d/%d'%(k+1,len(data.zdims)))
            res = minimize(func, lam0, jac=gr_func, method='L-BFGS-B', tol=10 ** -10)
            if res.success or res.fun < func(lam0):
                data.priorPar[k]['tau'] = np.exp(res.x/2)*1000/data.binSize
                nLL += res.fun
            else:
                nLL += func(lam0)

        LL_list.append(-nLL)
        if ii == 0:
            continue
        if np.abs(LL_list[-2]-LL_list[-1])/np.abs(LL_list[0]-LL_list[1]) < tol:
            break
    return LL_list

if __name__ == '__main__':
    from gen_synthetic_data import dataGen
    dat = dataGen(trNum=150, T=50, D=2, N=20,N1=20, K0=1, K2=3, K3=3)
    dat = dat.cca_input
    # dat = np.load('/Users/edoardo/Work/Code/P-GPCCA/inference_syntetic_data/sim_150Trials.npy', allow_pickle=True).all()
    dat.initializeParam(dat.zdims)
    ll_list = expectation_mazimization(dat, maxIter=20, tol=10 ** -4)
    np.savez('/Users/edoardo/Work/Code/P-GPCCA/inference_syntetic_data/em10iter_sim_150Trials.npz',dat=dat,ll=ll_list)

