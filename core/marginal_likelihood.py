import numpy as np
from expectation_maximization import computeLL
from scipy.stats import multivariate_normal, poisson
from scipy.linalg import block_diag
from data_processing_tools import makeK_big, logpdf_multnorm, logDetCompute
from time import perf_counter
from copy import deepcopy
from inference import multiTrialInference

def jointLL_at_MAP(data, trial_list=None, remove_neu_dict=None):
    ll = 0
    ll_spikes = 0
    ll_stim = 0
    ll_prior = 0
    stimPar = data.stimPar
    xPar = data.xPar
    priorPar = data.priorPar
    try:
        cov_gauss = np.linalg.pinv(stimPar['PsiInv'])
        log_det_gauss = logDetCompute(cov_gauss)
    except:
        pass
    if trial_list is None:
        trial_list = list(data.trialDur.keys())
    for tr in trial_list:
        # t0 = perf_counter()
        T = data.trialDur[tr]
        stim, xList = data.get_observations(tr)
        zmap = data.posterior_inf[tr].mean
        # gaussian likelihood
        if stim.shape[1] > 0:
            mean_gauss = np.dot(stimPar['W0'],zmap[0]).T + stimPar['d']# D x T
            blk_cov_inv = block_diag(*[stimPar['PsiInv']]*T)

            val = logpdf_multnorm(stim.flatten(), mean_gauss.flatten(), blk_cov_inv, log_det_gauss*stim.shape[0])
            ll_stim += val
            ll += val


        for k in range(len(xPar)):
            W0 = xPar[k]['W0']
            W1 = xPar[k]['W1']
            d = xPar[k]['d']
            if (not remove_neu_dict is None) and (k in remove_neu_dict.keys()):
                keep_neu = np.ones(W0.shape[0],dtype=bool)
                keep_neu[remove_neu_dict[k]] = False
                W0 = W0[keep_neu]
                W1 = W1[keep_neu]
                d = d[keep_neu]
            else:
                keep_neu = np.ones(W0.shape[0],dtype=bool)
            # t0 = perf_counter()
            m0 = np.dot(W0, zmap[0]).T
            m1 = np.dot(W1, zmap[k+1]).T
            val = np.sum(poisson.logpmf(xList[k][:,keep_neu], mu=np.exp(m0+m1+d)))
            ll_spikes += val
            ll += val
            # print('poisson ', perf_counter() - t0)

        for k in zmap.keys():
            # t0 = perf_counter()
            tau = priorPar[k]['tau']
            _, Kbig, Kbiginv, log_det = makeK_big(tau.shape[0], tau, None, data.binSize, epsNoise=data.epsNoise, T=T,
                                                  computeInv=True)
            val = logpdf_multnorm(zmap[k].flatten(), np.zeros(Kbig.shape[0]), Kbiginv, log_det)
            ll_prior += val
            ll += val
            # print('GP ', perf_counter() - t0)
    # print('LL stim', ll_stim)
    # print('LL spikes', ll_spikes)
    # print('LL prior', ll_prior)
    return ll,ll_prior,ll_stim,ll_spikes

def marginal_likelihood(data, remove_neu_dict=None, trial_list=None, return_all=False,return_inf=False):
    if trial_list is None:
        trial_list = list(data.trialDur.keys())

    if not remove_neu_dict is None:
        data_inf = deepcopy(data)
        # infer posterior without the neuron
        multiTrialInference(data_inf, trial_list=trial_list, useGauss=1, returnLogDetPrecision=False,
                        remove_neu_dict=remove_neu_dict)
    else:
        data_inf = data
    ll0, ll_prior,ll_stim,ll_spikes = jointLL_at_MAP(data_inf,trial_list=trial_list, remove_neu_dict=remove_neu_dict)
    ll1 = 0.5 * np.sum(multiTrialInference(data_inf, trial_list=trial_list, useGauss=1, returnLogDetPrecision=True, remove_neu_dict=remove_neu_dict)) # this is log(1/|cov|)

    T = 0
    for tr in trial_list:
        T += data.trialDur[tr]
    #print('sum(K_i) * T_tot * log 2 \pi', 0.5 * np.sum(data.zdims) * T * np.log(2 * np.pi))
    if return_inf:
        return ll0 - ll1 + 0.5 * np.sum(data.zdims) * T * np.log(2 * np.pi), data_inf
    if not return_all:
        return ll0 - ll1 + 0.5 * np.sum(data.zdims) * T * np.log(2*np.pi)

    return ll0 - ll1 + 0.5 * np.sum(data.zdims) * T * np.log(2*np.pi),  ll_prior,ll_stim,ll_spikes, ll1,  0.5 * np.sum(data.zdims) * T * np.log(2 * np.pi)




if __name__ == '__main__':
    from inference import inferTrial,multiTrialInference
    data = np.load(
        '/Users/edoardo/Work/Code/P-GPCCA/fit_cluster/neural_communic_subspace_nostim_fit_1.npz',
        allow_pickle=True)['data_cca'].all()

    K0 = 0
    for parDict in data.xPar:
        K0 += np.prod(parDict['W0'].shape)
        K0 += np.prod(parDict['W1'].shape)
        K0 += np.prod(parDict['d'].shape)
    try:
        K0 += np.prod(data.stimPar['W0'].shape)
        K0 += np.prod(data.stimPar['d'].shape)
        K0 += np.prod(data.stimPar['PsiInv'].shape)
    except:
        pass
    for parDict in data.priorPar:
        K0 += np.prod(parDict['tau'].shape)


    #print(jointLL_at_MAP(data))
    # data_1 = deepcopy(data)
    # data_1.zdims = [1,data.zdims[1]]
    # K1 = 0
    # for tr in data_1.trialDur.keys():
    #     data_1.posterior_inf[tr].mean[0] = data.posterior_inf[tr].mean[0][:1,:]
    #     data_1.posterior_inf[tr].cov_k[0] = data.posterior_inf[tr].cov_k[0][:1,:,:]
    #     data_1.posterior_inf[tr].cross_cov_t[1] = data.posterior_inf[tr].cross_cov_t[1][:,:1,:]
    #     try:
    #         data_1.stimPar['W0'] = data_1.stimPar['W0'][:,:1]
    #     except:
    #         pass
    #     data_1.xPar[0]['W0'] = data_1.xPar[0]['W0'][:, :1]
    #     data_1.priorPar[0]['tau'] = data_1.priorPar[0]['tau'][:1]
    #     for parDict in data_1.xPar:
    #         K1 += np.prod(parDict['W0'].shape)
    #         K1 += np.prod(parDict['W1'].shape)
    #         K1 += np.prod(parDict['d'].shape)
    #     try:
    #         K1 += np.prod(data_1.stimPar['W0'].shape)
    #         K1 += np.prod(data_1.stimPar['d'].shape)
    #         K1 += np.prod(data_1.stimPar['PsiInv'].shape)
    #     except:
    #         pass
    #
    #     for parDict in data_1.priorPar:
    #         K1 += np.prod(parDict['tau'].shape)





    data_sub = deepcopy(data)
    data_sub.initializeParam(data.zdims)

    #print(2*K0 - 2*marginal_likelihood(data, trial_list=[0,1,2,6,7,8,9]))
    # print(2*K1 - 2*marginal_likelihood(data_1, trial_list=[0,1,2,6,7,8,9]))
    #print(2*K0 - 2*marginal_likelihood(data_sub, trial_list=[0,1,2,6,7,8,9]))

    # print(marginal_likelihood(data, trial_list=[0,1,2],remove_neu_dict={0:[]}))
    # print(marginal_likelihood(data, trial_list=[0,1,2],remove_neu_dict={0:[8]}))
    #



    # print(marginal_likelihood(data, trial_list=[0,1,2,3,4]))

    #print(marginal_likelihood(data))