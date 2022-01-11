import numpy as np
from inference import multiTrialInference
from learnGaussianParam import learn_GaussianParams,full_GaussLL
from learnPoissonParam import all_trial_PoissonLL
from learnGPParams import all_trial_GPLL
from copy import deepcopy
from scipy.optimize import minimize
import sys
sys.path.append('../bads_optim')
from badsOptim import badsOptim

def computeLL(data):
    llgauss = full_GaussLL(data)
    LL = full_GaussLL(data)
    ll_poiss = []
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
        tmp = func(parStack)
        ll_poiss.append(tmp)
        LL += tmp

    ll_gp = []
    for k in range(len(data.zdims)):
        tau = data.priorPar[k]['tau']
        lam0 = 2 * np.log(((tau * data.binSize / 1000)))
        func = lambda lam0: all_trial_GPLL(lam0, data, k, block_trials=1, isGrad=False)
        tmp = func(lam0)
        ll_gp.append(tmp)
        LL += tmp
    return LL,llgauss,ll_poiss,ll_gp


def expectation_mazimization(data, maxIter=10, tol=10**-3, use_badsGP=False,
                             use_badsPoisson=False):
    # save the intial parameters
    data.init_priorPar = deepcopy(data.priorPar)
    data.init_xPar = deepcopy(data.xPar)
    data.init_stimPar = deepcopy(data.stimPar)
    
    if use_badsPoisson or use_badsGP:
        gpOptim = badsOptim(data)

    # save the ll_list
    if 'll_iter' not in data.__dict__.keys():
        data.ll_iter = []

    # compute initial expected LL
    LL_list = []
    # LL = full_GaussLL(data)
    # for k in range(len(data.zdims) - 1):
    #     C = data.xPar[k]['W0']
    #     C1 = data.xPar[k]['W1']
    #     d = data.xPar[k]['d']
    #     N, K0 = C.shape
    #     K1 = C1.shape[1]
    #     parStack = np.hstack((C.flatten(), C1.flatten(), d))
    #     func = lambda xx: all_trial_PoissonLL(xx[:N * K0].reshape(N, K0), xx[N * K0:N * (K0 + K1)].reshape(N, K1),
    #                                            xx[N * (K0 + K1):],
    #                                            data, k + 1, block_trials=100, isGrad=False)
    #     LL += func(parStack)
    #
    # for k in range(len(data.zdims)):
    #     tau = data.priorPar[k]['tau']
    #     lam0 = 2 * np.log(((tau * data.binSize / 1000)))
    #     func = lambda lam0: all_trial_GPLL(lam0, data, k, block_trials=1, isGrad=False)
    #     LL += func(lam0)
    #
    # LL_list.append(LL)

    # start the EM
    for ii in range(maxIter):
        print('EM iteration: %d/%d'%(ii+1,maxIter))
        # infer latent
        print('- E-step')
        multiTrialInference(data)
        if ii == 0:
            print('initial LL:', computeLL(data)[0],'\n')
        # learn gaussian params
        print('- Gaussian M-step')
        learn_GaussianParams(data, test=False, isMPI=False)
        nLL = -full_GaussLL(data)

        # learn Poisson obs param
        for k in range(len(data.zdims) - 1):
            if not use_badsPoisson:

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
            else:
                C, C1, d, f = gpOptim.bads_optimPoisson(data, k)
                data.xPar[k]['W0'] = C
                data.xPar[k]['W1'] = C1
                data.xPar[k]['d'] = d
                nLL += f/len(data.trialDur.keys())



            # learn GP param
        for k in range(len(data.zdims)):
            tau = data.priorPar[k]['tau']
            if use_badsGP:
                f,g = gpOptim.bads_optim(data,k)

                f = np.squeeze(f)
                if len(f.shape) is 0:
                    f = np.reshape(f,1)
                data.priorPar[k]['tau'] = f
                # print(data.priorPar[k]['tau'],g)
                nLL += g
                
            else:
                lam0 = 2 * np.log(((tau * data.binSize/1000)))
                func = lambda lam0: -all_trial_GPLL(lam0, data, k, block_trials=1, isGrad=False)
                gr_func = lambda lam0: -all_trial_GPLL(lam0, data, k, block_trials=1, isGrad=True)
                f0 = func(lam0)
                print('- GP M-step, latent factor: %d/%d'%(k+1,len(data.zdims)))
                res = minimize(func, lam0, jac=gr_func, method='L-BFGS-B', tol=10 ** -10)
                if res.success or res.fun < func(lam0):
                    data.priorPar[k]['tau'] = np.exp(res.x/2)*1000/data.binSize
                    nLL += res.fun
                    print('nLL prior before/after optim:',f0,res.fun)
                else:
                    nLL += func(lam0)

        LL_list.append(-nLL)
        print('current LL: ',LL_list[-1])
        if ii == 0:
            continue
        if np.abs(LL_list[-2]-LL_list[-1])/np.abs(LL_list[0]-LL_list[1]) < tol:
            break
    data.ll_iter.append(LL_list)
    print('Final Posterior Inference....')
    multiTrialInference(data, plot_trial=True)
    LL = computeLL(data)
    data.ll_iter.append([LL])
    return LL_list

if __name__ == '__main__':
    from gen_synthetic_data import dataGen
    use_badsGP = True
    use_badsPoisson = False
    K0=1
    K2=2
    K3=3
    trNum = 300
    emIter = 4
    dat = dataGen(trNum=trNum, T=50, D=2, N=3,N1=20, K0=K0, K2=K2, K3=K3)

    cca_input1 = deepcopy(dat.cca_input)
    cca_input2 = deepcopy(dat.cca_input)
    cca_input2.initializeParam([K0,K2,K3], use_poissonPCA=True)


    multiTrialInference(cca_input1, plot_trial=True)
    LL = computeLL(cca_input1)[0]

    # multiTrialInference(cca_input2, plot_trial=True)

    print(computeLL(cca_input1))
    print(computeLL(cca_input2))
    print('MAX LL: ',LL)
    ll = expectation_mazimization(cca_input2,maxIter=emIter,use_badsGP=use_badsGP,use_badsPoisson=use_badsPoisson)
    if use_badsGP and use_badsPoisson:
        np.savez('/Users/edoardo/Work/Code/P-GPCCA/inference_syntetic_data/BADSGP_BADSPoisson_em%diter_sim_150Trials.npz'%emIter,dat=cca_input2)
    elif use_badsGP:
        np.savez('/Users/edoardo/Work/Code/P-GPCCA/inference_syntetic_data/BADSGP_em%diter_sim_150Trials.npz'%emIter,dat=cca_input2)
    elif use_badsPoisson:
        np.savez('/Users/edoardo/Work/Code/P-GPCCA/inference_syntetic_data/BADSPoisson_em%diter_sim_150Trials.npz'%emIter,dat=cca_input2)
    else:
        np.savez('/Users/edoardo/Work/Code/P-GPCCA/inference_syntetic_data/L_BFGS_B_em%diter_sim_150Trials.npz'%emIter,dat=cca_input2)



