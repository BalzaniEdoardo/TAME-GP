import numpy as np
from inference import multiTrialInference
from learnGaussianParam import learn_GaussianParams,full_GaussLL
from learnPoissonParam import all_trial_PoissonLL,poissonELL_Sparse,grad_poissonELL_Sparse,hess_poissonELL_Sparse,newton_opt_CSR
from learnGPParams import all_trial_GPLL
from data_processing_tools import sortGradient_idx
from copy import deepcopy
from scipy.optimize import minimize
import sys,os
import inspect
basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
print(basedir)

# requires matlab, with the matlab api for python installed and bads package from Luigi Acerbi
# https://github.com/lacerbi/bads
try:
    sys.path.append(os.path.join(basedir,'bads_optim'))
    from badsOptim import badsOptim
except:
    badsOptim=None

def computeLL(data):
    llgauss = full_GaussLL(data)
    LL = deepcopy(llgauss)
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


def expectation_maximization(data, maxIter=10, tol=10**-3, use_badsGP=False,
                             method='sparse-Newton', tolPoissonOpt=10**-12,
                             boundsW0=None, boundsW1=None, boundsD=None):
    """

    :param data: P_GPCCA structure
        The input data
    :param maxIter: int
        max number of EM iterations
    :param tol:
        stopping parameter for the EM incremental in LL.
    :param use_badsGP: bool
        True: use the BADS optimiizer for global optimimum (empirically it doesn't make a difference)
        False: use L-BGSF-B algorithm for the GP parameters
    :param use_badsPoisson:
        True: use the BADS optimiizer for global optimimum (strongly not suggested, the problem should be convex)
        False: use L-BGSF-B algorithm for the GP parameters
    :param tolPoissonOpt:
        tolerance parameter for L-BGFS-B optimizer for the poisson optimization
    :param boundsW0: Poissno projection weights bound
        None: set to default values:
         Poisson W0 bound: bounds [-4,4] for all W0_{ij}.
            very large range considering that for an exponential link exp(Wz) will have a mean of exp(W^2 / 2), for
            z normal N(0,1)... this ranges ranging from 1 to e^8.
        [lb,ub] : pair of lower and upper bounds
    :param boundsW1:
        Same as bound W0
    :param boundsD:
        None: set default to [-10,5] for Poisson variables
        [lb,ub] pair of lower and upper bounds
    :return:
    """
    # save the intial parameters
    data.init_priorPar = deepcopy(data.priorPar)
    data.init_xPar = deepcopy(data.xPar)
    data.init_stimPar = deepcopy(data.stimPar)
    
    if method=='BADS' or use_badsGP:
        gpOptim = badsOptim(data)

    # save the ll_list
    if 'll_iter' not in data.__dict__.keys():
        data.ll_iter = []

    # initialize expected LL list
    LL_list = []

    # start the EM
    for ii in range(maxIter):
        print('EM iteration: %d/%d'%(ii+1,maxIter))
        # infer latent
        print('- E-step')
        nLL = multiTrialInference(data,plot_trial=True)
        if ii == 0:
            print('initial LL:', computeLL(data)[0],'\n')
        # learn gaussian params
        print('- Gaussian M-step')
        learn_GaussianParams(data, test=False, isMPI=False)


        # learn Poisson obs param
        for k in range(len(data.zdims) - 1):
            if method == 'L-BFGS-B':
                # extract parameters
                C = data.xPar[k]['W0']
                C1 = data.xPar[k]['W1']
                d = data.xPar[k]['d']
                N, K0 = C.shape
                K1 = C1.shape[1]
                # L-BGFS-B optimization
                parStack = np.hstack((C.flatten(), C1.flatten(), d))
                if boundsW0 is None:
                    bW0 = np.array([-4, 4] * C.flatten().shape[0]).reshape(-1,2)
                else:
                    bW0 = np.array(list(boundsW0) * C.flatten().shape[0]).reshape(-1,2)

                if boundsW1 is None :
                    bW1 = np.array([-4, 4] * C1.flatten().shape[0]).reshape(-1,2)
                else:
                    bW1 = np.array(list(boundsW1) * C1.flatten().shape[0]).reshape(-1,2)

                if boundsD is None:
                    bD = np.array(list([-10,5]) * d.shape[0]).reshape(-1,2)
                else:
                    bD = np.array(boundsD * d.shape[0]).reshape(-1,2)

                bounds = np.vstack([bW0, bW1, bD])

                func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0), xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                                       xx[N * (K0 + K1):],
                                                       data, k+1, block_trials=100, isGrad=False)
                gr_func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0),
                                                          xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                                          xx[N * (K0 + K1):],
                                                          data, k+1, block_trials=100, isGrad=True)

                print('- Poisson M-step, observed population: %d/%d'%(k+1,len(data.zdims)-1))
                res = minimize(func, np.zeros(parStack.shape), jac=gr_func, method='L-BFGS-B', bounds=bounds, tol=tolPoissonOpt)
                if res.success or res.fun < func(parStack):
                    data.xPar[k]['W0'] = res.x[:N * K0].reshape(N, K0)
                    data.xPar[k]['W1'] = res.x[N * K0:N * (K0 + K1)].reshape(N, K1)
                    data.xPar[k]['d'] = res.x[N * (K0 + K1):]

                else:
                    pass
            elif method == 'BADS':
                C, C1, d, f = gpOptim.bads_optimPoisson(data, k)
                data.xPar[k]['W0'] = C
                data.xPar[k]['W1'] = C1
                data.xPar[k]['d'] = d


            elif method == 'sparse-Newton':
                C = data.xPar[k]['W0']
                C1 = data.xPar[k]['W1']
                d = data.xPar[k]['d']
                N, K0 = C.shape
                K1 = C1.shape[1]
                # fast sparse matrix based newton optim

                rot = sortGradient_idx(C.shape[0], [C.shape[1], C1.shape[1], 1], isReverse=False)
                rotInv = sortGradient_idx(C.shape[0], [C.shape[1], C1.shape[1], 1], isReverse=True)
                parStack = np.hstack((C.flatten(), C1.flatten(), d))[rot]
                ff = lambda par: poissonELL_Sparse(par, k+1, data, rotInv)
                gg = lambda par: grad_poissonELL_Sparse(par, k+1, data, rotInv)
                hh = lambda par: hess_poissonELL_Sparse(par, k+1, data, rotInv, inverse=True, sparse=True)
                Z0, feval, feval_hist = newton_opt_CSR(ff, gg, hh, parStack, tol=10 ** -8, max_iter=1000,
                                                       disp_eval=True, max_having=30)
                par_optim = Z0[rotInv]
                data.xPar[k]['W0'] = par_optim[:N * K0].reshape(N, K0)
                data.xPar[k]['W1'] = par_optim[N * K0:N * (K0 + K1)].reshape(N, K1)
                data.xPar[k]['d'] = par_optim[N * (K0 + K1):]


        # learn GP param
        for k in range(len(data.zdims)):
            tau = data.priorPar[k]['tau']
            if use_badsGP:
                f,g = gpOptim.bads_optim(data,k)

                f = np.squeeze(f)
                if len(f.shape) == 0:
                    f = np.reshape(f,1)
                data.priorPar[k]['tau'] = f

                
            else:
                lam0 = 2 * np.log(((tau * data.binSize/1000)))
                func = lambda lam0: -all_trial_GPLL(lam0, data, k, block_trials=1, isGrad=False)
                gr_func = lambda lam0: -all_trial_GPLL(lam0, data, k, block_trials=1, isGrad=True)
                f0 = func(lam0)
                print('- GP M-step, latent factor: %d/%d'%(k+1,len(data.zdims)))
                res = minimize(func, lam0, jac=gr_func, method='L-BFGS-B', tol=10 ** -12)
                if res.success or res.fun < func(lam0):
                    data.priorPar[k]['tau'] = np.exp(res.x/2)*1000/data.binSize

                    print('nLL prior before/after optim:',f0,res.fun)
                else:
                    pass

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
    from time import perf_counter
    from gen_synthetic_data import dataGen,dataGen_poissonOnly
    if os.path.exists('../inference_syntetic_data/L_BFGS_B_em4iter_sim_150Trials.npz'):
        dat = np.load('../inference_syntetic_data/L_BFGS_B_em4iter_sim_150Trials.npz',
                      allow_pickle=True)['dat'].all()
    else:
        gen_dat = dataGen_poissonOnly(10, T=50)
        dat = gen_dat.cca_input
    dat.genNewData(10, 50)

    subStruc = dat.subSampleTrial(dat.new_trial_list)
    multiTrialInference(subStruc, trial_list=dat.new_trial_list, plot_trial=True)
    dat_true = deepcopy(subStruc)
    dat_true.xPar = subStruc.ground_truth_xPar
    dat_true.stimPar = subStruc.ground_truth_stimPar
    dat_true.priorPar = subStruc.ground_truth_priorPar
    # Psi = np.linalg.inv(subStruc.ground_truth_stimPar['PsiInv'])
    # ee = np.linalg.eigh(Psi)[0]
    # dat_true.stimPar['PsiInv'] = np.diag(1/np.sqrt(ee))
    # multiTrialInference(dat_true, trial_list=dat.new_trial_list, plot_trial=True)
    # print('fit par')
    # print(computeLL(subStruc)[0]-computeLL(dat_true)[0])
    # print('true par')
    expectation_maximization(dat_true,maxIter=44,boundsW0=[-3,3],boundsD=[-10,10])




    # plt.figure()
    # plt.title('EM reconstructed posterior: N %d'%N)
    # p, = plt.plot(mn)
    # plt.fill_between(np.arange(mn.shape[0]), mn - 1.96 * std, mn + 1.96 * std, color=p.get_color(), alpha=0.4)
    # plt.plot(lt, color='k')
    # model = LinearRegression()
    # MN_post = dat2.posterior_inf[tr].mean[1]
    # lat = dat2.ground_truth_latent[tr][:, ii:ii+dat2.zdims[lat]]
    # res = model.fit(MN_post,lat.T)
    #
    # plt.figure(figsize=(12,4))
    # lat = 2
    # plt.subplot(121)
    # plt.title('True parameter latent posterior:')
    # tr = 29
    # mn0,mn1 = dat.posterior_inf[tr].mean[lat]
    # p0, = plt.plot(mn0)
    # p1, = plt.plot(mn1)
    # std0 = np.sqrt(dat.posterior_inf[tr].cov_t[lat][:, 0, 0])
    # std1 = np.sqrt(dat.posterior_inf[tr].cov_t[lat][:, 1, 1])
    # plt.fill_between(range(mn0.shape[0]), mn0-std0, mn0+std0,alpha=0.4,color=p0.get_color())
    # plt.fill_between(range(mn1.shape[0]), mn1-std1, mn1+std1,alpha=0.4,color=p1.get_color())
    #
    # plt.subplot(122)
    # plt.title('EM parameter latent posterior:')
    #
    # mn0, mn1 = dat2.posterior_inf[tr].mean[lat]*-1
    # p1, = plt.plot(mn1)
    # p0, = plt.plot(mn0)
    #
    # std1 = np.sqrt(dat2.posterior_inf[tr].cov_t[lat][:, 0, 0])
    # std0 = np.sqrt(dat2.posterior_inf[tr].cov_t[lat][:, 1, 1])
    # plt.fill_between(range(mn0.shape[0]), mn0 - std0, mn0 + std0, alpha=0.4, color=p0.get_color())
    # plt.fill_between(range(mn1.shape[0]), mn1 - std1, mn1 + std1, alpha=0.4, color=p1.get_color())
    #
    #
    #
    #
