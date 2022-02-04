import numpy as np
import os,sys
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(basedir,'core'))
from learnGaussianParam import learn_GaussianParams,full_GaussLL
from learnPoissonParam import all_trial_PoissonLL,poissonELL_Sparse,grad_poissonELL_Sparse,hess_poissonELL_Sparse,newton_opt_CSR
from data_processing_tools import sortGradient_idx
from data_processing_tools_factorized import preproc_post_mean_factorizedModel
from copy import deepcopy
from inference_factorized import newton_optim_map, reconstruct_post_mean_and_cov,trialByTrialInference
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

def computeLL_factorized(data):
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

    ## needs to be tested
    print('TEST THAT THIS IS A GOOD $E_q[z\\tr z]$')
    for k in range(len(data.zdims)):
        T = np.sum(list(data.trialDur.values()))
        var_stack = np.zeros((T, data.zdims[k]))
        mean_stack = np.zeros((T, data.zdims[k]))
        cc = 0
        i0 = 0
        for tr in data.trialDur.keys():
            T_tr = data.trialDur[tr]
            idx0, idx1 = np.diag_indices(data.zdims[k])
            #print(k,tr,mean_stack[i0:i0+T_tr,:].shape,data.posterior_inf[tr].mean[k].T.shape)
            mean_stack[i0:i0+T_tr,:] = data.posterior_inf[tr].mean[k].T
            var_stack[i0:i0+T_tr,:] = data.posterior_inf[tr].cov_t[k][:,idx0,idx1]
            cc += 1
            i0 += T_tr
        LL += -0.5 * (var_stack + mean_stack**2).sum()
        # mean_t = data.posterior_inf[tr].mean[k].T
        # mumuT = np.einsum()

    return LL,llgauss,ll_poiss

def grad_ascent(func,grad,Z0,tol=10**-8,max_iter=1000, disp_eval=True,max_having=30):
    eps_float = np.finfo(float).eps

    tmpZ = Z0.copy()
    feval_hist = []
    ii  = 0
    delta_eval = np.inf
    while ii < max_iter and delta_eval > tol:

        # t0 = perf_counter()
        feval = func(Z0)
        fgrad = grad(Z0)
        if disp_eval:
            print('newton optim iter', ii, 'log-like', feval)

        delt = -fgrad  # hess_ll.mult_vec(grad_ll)
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


def expectation_maximization_factorized(data, maxIter=10, tol=10 ** -8,
                             method='sparse-Newton', tolPoissonOpt=10 ** -12,
                             boundsW0=None, boundsW1=None, boundsD=None,trialDur_variable=False,
                             useNewton=False,trial_block=100):
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


    # save the ll_list
    if 'll_iter' not in data.__dict__.keys():
        data.ll_iter = []

    # initialize expected LL list
    LL_list = []

    # start the EM
    for ii in range(maxIter):
        print('EM iteration: %d/%d' % (ii + 1, maxIter))
        # infer latent
        print('- E-step')
        # find zmap latent
        if trial_block==1:
            zmap, td,ll = trialByTrialInference(data, tol=10 ** -10, max_iter=100, max_having=30,
                                                              disp_ll=True, init_zeros=True, useNewton=useNewton,
                                                              trialDur_variable=trialDur_variable)
            reconstruct_post_mean_and_cov(data, zmap, td)


        elif trial_block < len(data.trialDur.keys()):
            tr_list = list(data.trialDur.keys())
            Z0, tr_dict = preproc_post_mean_factorizedModel(data, returnDict=False)
            for k in range(0, len(tr_list), trial_block):
                print('trial block: [%d, %d]'%(k,min(k+trial_block,len(tr_list))))
                trs = tr_list[k:k+trial_block]
                sub_data = data.subSampleTrial(trs)
                zmap, success, td, ll, ll_hist = newton_optim_map(sub_data, tol=10 ** -10, max_iter=100, max_having=30,
                                                                  disp_ll=True, init_zeros=True, useNewton=useNewton,
                                                                  trialDur_variable=trialDur_variable)
                # store the results
                for tr in trs:
                    Z0[tr_dict[tr]] = zmap[td[tr]]
            #recompute the map
            reconstruct_post_mean_and_cov(data, Z0, tr_dict)


        else:
            zmap, success, td, ll, ll_hist = newton_optim_map(data, tol=10 ** -10, max_iter=100, max_having=20,
                                                          disp_ll=True, init_zeros=True, useNewton=useNewton,
                                                              trialDur_variable=trialDur_variable)
            # reconstruct the usual posterior structure and store
            reconstruct_post_mean_and_cov(data, zmap, td)


        if ii == 0:
            print('initial LL:', computeLL_factorized(data)[0], '\n')
        # learn gaussian params
        print('- Gaussian M-step')
        learn_GaussianParams(data, test=False, isMPI=False)
        nLL = -full_GaussLL(data)

        # learn Poisson obs param
        for k in range(len(data.zdims) - 1):
            # extract parameters
            C = data.xPar[k]['W0']
            C1 = data.xPar[k]['W1']
            d = data.xPar[k]['d']
            N, K0 = C.shape
            K1 = C1.shape[1]
            print('- Poisson M-step, observed population: %d/%d' % (k + 1, len(data.zdims) - 1))

            parStack = np.hstack((C.flatten(), C1.flatten(), d))
            if method == 'L-BFGS-B':
                # L-BGFS-B optimization
                if boundsW0 is None:
                    bW0 = np.array([-4, 4] * C.flatten().shape[0]).reshape(-1, 2)
                else:
                    bW0 = np.array(list(boundsW0) * C.flatten().shape[0]).reshape(-1, 2)

                if boundsW1 is None:
                    bW1 = np.array([-4, 4] * C1.flatten().shape[0]).reshape(-1, 2)
                else:
                    bW1 = np.array(list(boundsW1) * C1.flatten().shape[0]).reshape(-1, 2)

                if boundsD is None:
                    bD = np.array(list([-10, 5]) * d.shape[0]).reshape(-1, 2)
                else:
                    bD = np.array(boundsD * d.shape[0]).reshape(-1, 2)

                bounds = np.vstack([bW0, bW1, bD])

                func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0),
                                                       xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                                       xx[N * (K0 + K1):],
                                                       data, k + 1, block_trials=100, isGrad=False)
                gr_func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0),
                                                          xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                                          xx[N * (K0 + K1):],
                                                          data, k + 1, block_trials=100, isGrad=True)


                res = minimize(func, parStack, jac=gr_func, method='L-BFGS-B', bounds=bounds,
                               tol=tolPoissonOpt)
                if res.success or res.fun < func(parStack):
                    data.xPar[k]['W0'] = res.x[:N * K0].reshape(N, K0)
                    data.xPar[k]['W1'] = res.x[N * K0:N * (K0 + K1)].reshape(N, K1)
                    data.xPar[k]['d'] = res.x[N * (K0 + K1):]
                    nLL += res.fun
                else:
                    nLL += func(parStack)
            elif method == 'gradient-ascent':

                func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0),
                                                       xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                                       xx[N * (K0 + K1):],
                                                       data, k + 1, block_trials=100, isGrad=False)
                gr_func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0),
                                                          xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                                          xx[N * (K0 + K1):],
                                                          data, k + 1, block_trials=100, isGrad=True)
                ff = lambda x: -func(x)
                gg = lambda x: -gr_func(x)
                Z0, feval, feval_hist = grad_ascent(ff, gg, parStack, tol=10 ** -8, max_iter=100, disp_eval=False)
                data.xPar[k]['W0'] = Z0[:N * K0].reshape(N, K0)
                data.xPar[k]['W1'] = Z0[N * K0:N * (K0 + K1)].reshape(N, K1)
                data.xPar[k]['d'] = Z0[N * (K0 + K1):]
                nLL += feval

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
                nLL += feval / len(data.trialDur.keys())

        LL_list.append(-nLL)
        print('current LL: ', LL_list[-1])
        if ii == 0:
            continue
        if np.abs(LL_list[-2] - LL_list[-1]) / np.abs(LL_list[0] - LL_list[1]) < tol:
            break
    data.ll_iter.append(LL_list)
    print('Final Posterior Inference....')
    zmap, success, td, ll, ll_hist = newton_optim_map(data, tol=10 ** -10, max_iter=100, max_having=20,
                                                      disp_ll=True, init_zeros=True, useNewton=True,
                                                      trialDur_variable=trialDur_variable)
    # reconstruct the usual posterior structure and store
    reconstruct_post_mean_and_cov(data, zmap, td)


    LL = computeLL_factorized(data)[0]
    data.ll_iter.append([LL])
    return LL_list


if __name__ == '__main__':
    from gen_synthetic_data import dataGen
    import matplotlib.pylab as plt
    cc = 0

    mn_all = np.zeros((100,1,50))
    std_all = np.zeros((100, 1, 50))
    latent = np.zeros((100, 1, 50))
    for N in [500]:
        lat = 2
        if lat == 2:
            gen_dat = dataGen(100, N=200, N1=N,K2=2, K3=2, T=50,infer=False,setTruePar=True)
        else:
            gen_dat = dataGen(100, N=N, N1=200, K2=2, K3=2, T=50, infer=False, setTruePar=True)

        dat = gen_dat.cca_input
        zmap, success, td, ll, ll_hist = newton_optim_map(dat, tol=10 ** -10, max_iter=100, max_having=20,
                                                          disp_ll=True, init_zeros=True, useNewton=True)
        reconstruct_post_mean_and_cov(dat, zmap, td)


        ii = dat.zdims[:lat].sum()
        for tr in range(100):
            mn = dat.posterior_inf[tr].mean[lat][0]
            std = np.sqrt(dat.posterior_inf[tr].cov_t[lat][:, 0, 0])
            mn_all[tr,cc] = mn
            std_all[tr, cc] = std
            latent[tr,cc] = dat.ground_truth_latent[tr][:,ii]
        cc+=1

    cc = 1
    tr = 11
    plt.figure(figsize=(12, 4.5))
    for N in [500]:
        mn = mn_all[tr,cc-1]
        std = std_all[tr,cc-1]
        lt = latent[tr,cc-1]
        plt.subplot(1, 4, cc)
        plt.title('N=%d units' % N)
        p, = plt.plot(mn)
        plt.fill_between(np.arange(mn.shape[0]), mn - 1.96 * std, mn + 1.96 * std, color=p.get_color(), alpha=0.4)

        plt.plot(lt, color='k')
        cc+=1
    plt.tight_layout()

    ## fit em
    dat2 = deepcopy(dat)
    dat2.initializeParam(dat2.zdims)
    ll_list = expectation_maximization_factorized(dat2, maxIter=20, boundsW0=[-3, 3], boundsD=[-10, 10])
    mn = dat2.posterior_inf[tr].mean[lat][0]
    std = np.sqrt(dat2.posterior_inf[tr].cov_t[lat][:, 0, 0])
    lt = dat2.ground_truth_latent[tr][:, ii]
    plt.figure()
    plt.title('EM reconstructed posterior: N %d'%N)
    p, = plt.plot(mn)
    plt.fill_between(np.arange(mn.shape[0]), mn - 1.96 * std, mn + 1.96 * std, color=p.get_color(), alpha=0.4)
    plt.plot(lt, color='k')
    model = LinearRegression()
    MN_post = dat2.posterior_inf[tr].mean[1]
    lat = dat2.ground_truth_latent[tr][:, ii:ii+dat2.zdims[lat]]
    res = model.fit(MN_post,lat.T)

    plt.figure(figsize=(12,4))
    lat = 2
    plt.subplot(121)
    plt.title('True parameter latent posterior:')
    tr = 29
    mn0,mn1 = dat.posterior_inf[tr].mean[lat]
    p0, = plt.plot(mn0)
    p1, = plt.plot(mn1)
    std0 = np.sqrt(dat.posterior_inf[tr].cov_t[lat][:, 0, 0])
    std1 = np.sqrt(dat.posterior_inf[tr].cov_t[lat][:, 1, 1])
    plt.fill_between(range(mn0.shape[0]), mn0-std0, mn0+std0,alpha=0.4,color=p0.get_color())
    plt.fill_between(range(mn1.shape[0]), mn1-std1, mn1+std1,alpha=0.4,color=p1.get_color())

    plt.subplot(122)
    plt.title('EM parameter latent posterior:')

    mn0, mn1 = dat2.posterior_inf[tr].mean[lat]*-1
    p1, = plt.plot(mn1)
    p0, = plt.plot(mn0)

    std1 = np.sqrt(dat2.posterior_inf[tr].cov_t[lat][:, 0, 0])
    std0 = np.sqrt(dat2.posterior_inf[tr].cov_t[lat][:, 1, 1])
    plt.fill_between(range(mn0.shape[0]), mn0 - std0, mn0 + std0, alpha=0.4, color=p0.get_color())
    plt.fill_between(range(mn1.shape[0]), mn1 - std1, mn1 + std1, alpha=0.4, color=p1.get_color())





