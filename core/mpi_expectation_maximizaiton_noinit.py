from mpi4py import MPI
import numpy as np
from inference import multiTrialInference
from learnGaussianParam import learn_GaussianParams,full_GaussLL
from learnPoissonParam import all_trial_PoissonLL,poissonELL_Sparse,grad_poissonELL_Sparse,hess_poissonELL_Sparse,newton_opt_CSR
from learnGPParams import all_trial_GPLL
from data_processing_tools import sortGradient_idx
from copy import deepcopy
from scipy.optimize import minimize
from expectation_maximization import computeLL
import sys,os
import inspect
from time import perf_counter
sys.path.append('../initialization/')
from expectation_maximization_factorized import expectation_maximization_factorized
# input file name

# fh_name = '../fit_cluster/sim_static_stim.npz'
# save_path = '../fit_cluster/%s'%sys.argv[1]
# iter_save = '../fit_cluster/iter_fit_%s.txt'%sys.argv[2]

fh_name = '../fit_cluster/%s'%sys.argv[1]
save_path = '../fit_cluster/%s'%sys.argv[2]
iter_save = '../fit_cluster/iter_fit_%s.txt'%sys.argv[3]
maxIter = int(sys.argv[4])

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
np.random.seed(rank+400)

def mpi_em(data,trial_dict, maxIter=10, tol=10**-3, method='sparse-Newton', tolPoissonOpt=10**-12,
           boundsW0=None, boundsW1=None, boundsD=None,
           save_every=10,save_path='pgpcca_fit.npz',iter_save='iter.txt'):

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
        string = 'iter %d/%d...\n'%(ii+1,maxIter)
        with open(iter_save, 'a') as fh:
            fh.write(string)
            fh.close()
        # flag for no intteruption of worker while loop
        comm.bcast(False, root=0)
        print('EM iteration: %d/%d' % (ii + 1, maxIter))
        # infer latent

        print('- E-step - using workers')
        t0 = perf_counter()
        # broadcast model params
        comm.bcast(data.xPar, root=0)
        comm.bcast(data.stimPar, root=0)
        comm.bcast(data.priorPar, root=0)

        # run the inference for the root worker trials
        # del non-root post inferred trials (not necessary precaution)
        if hasattr(data, 'posterior_inf'):
            print('Clearing non-root inferred trial before E-step')
            lst_pop = []
            for tr in data.posterior_inf.keys():
                if not tr in trial_dict[0]:
                    lst_pop.append(tr)
            for tr in lst_pop:
                data.posterior_inf.pop(tr)

        print('infer root 0 - start')
        with open(iter_save, 'a') as fh:
            fh.write('infer root 0 - start\n')
            fh.close()
        multiTrialInference(data, trial_list=trial_dict[0], plot_trial=True)
        t1 = perf_counter()
        print('infer root 0 - end')
        with open(iter_save, 'a') as fh:
            fh.write('infer root 0 - end\ntot time: %f sec\n'%(t1-t0))
            fh.close()


        # gather all the other inf results

        posterior_list = comm.gather(data.posterior_inf, root=0)
        print('gathered inference')

        # store in dict (the list have the root as first element, then the other workers)
        post_inf = {}
        for pst in posterior_list:
            for tr in pst.keys():
                post_inf[tr] = pst[tr]
        data.posterior_inf = post_inf
        del posterior_list
        print('stored newly inferred trials')

        if ii % save_every == 0:
            iteration =  ii + 1
            np.savez(save_path, data_cca = data, iteration=iteration)

        if ii == 0:
            print('initial LL:', computeLL(data)[0], '\n')
        # learn gaussian params
        print('- Gaussian M-step')
        t0 = perf_counter()
        learn_GaussianParams(data, test=False, isMPI=False)
        t1 = perf_counter()
        with open(iter_save, 'a') as fh:
            fh.write('Gauss M-Step tot time: %f sec\n'%(t1-t0))
            fh.close()
        nLL = -full_GaussLL(data)

        # learn Poisson obs param
        t0 = perf_counter()
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

                print('- Poisson M-step, observed population: %d/%d' % (k + 1, len(data.zdims) - 1))
                res = minimize(func, np.zeros(parStack.shape), jac=gr_func, method='L-BFGS-B', bounds=bounds,
                               tol=tolPoissonOpt)
                if res.success or res.fun < func(parStack):
                    data.xPar[k]['W0'] = res.x[:N * K0].reshape(N, K0)
                    data.xPar[k]['W1'] = res.x[N * K0:N * (K0 + K1)].reshape(N, K1)
                    data.xPar[k]['d'] = res.x[N * (K0 + K1):]
                    nLL += res.fun
                else:
                    nLL += func(parStack)

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
                ff = lambda par: poissonELL_Sparse(par, k + 1, data, rotInv)
                gg = lambda par: grad_poissonELL_Sparse(par, k + 1, data, rotInv)
                hh = lambda par: hess_poissonELL_Sparse(par, k + 1, data, rotInv, inverse=True, sparse=True)
                Z0, feval, feval_hist = newton_opt_CSR(ff, gg, hh, parStack, tol=10 ** -8, max_iter=1000,
                                                       disp_eval=True, max_having=30)
                par_optim = Z0[rotInv]
                data.xPar[k]['W0'] = par_optim[:N * K0].reshape(N, K0)
                data.xPar[k]['W1'] = par_optim[N * K0:N * (K0 + K1)].reshape(N, K1)
                data.xPar[k]['d'] = par_optim[N * (K0 + K1):]
                nLL += feval / len(data.trialDur.keys())

        t1 = perf_counter()
        with open(iter_save, 'a') as fh:
            fh.write('Poisson M-Step tot time: %f sec\n' % (t1 - t0))
            fh.close()
        # learn GP param
        t0 = perf_counter()
        for k in range(len(data.zdims)):
            tau = data.priorPar[k]['tau']

            lam0 = 2 * np.log(((tau * data.binSize / 1000)))
            func = lambda lam0: -all_trial_GPLL(lam0, data, k, block_trials=1, isGrad=False)
            gr_func = lambda lam0: -all_trial_GPLL(lam0, data, k, block_trials=1, isGrad=True)
            f0 = func(lam0)
            print('- GP M-step, latent factor: %d/%d' % (k + 1, len(data.zdims)))
            res = minimize(func, lam0, jac=gr_func, method='L-BFGS-B', tol=10 ** -12)
            if res.success or res.fun < func(lam0):
                data.priorPar[k]['tau'] = np.exp(res.x / 2) * 1000 / data.binSize
                nLL += res.fun
                print('nLL prior before/after optim:', f0, res.fun)
            else:
                nLL += func(lam0)

        t1 = perf_counter()
        with open(iter_save, 'a') as fh:
            fh.write('GP M-Step tot time: %f sec\n' % (t1 - t0))
            fh.write('GP pars: %s\n' % (str(data.priorPar)))

            fh.close()

        LL_list.append(-nLL)
        print('current LL: ', LL_list[-1])

        if ii == 0:
            continue
        if np.abs(LL_list[-2] - LL_list[-1]) / np.abs(LL_list[0] - LL_list[1]) < tol:
            break

    data.ll_iter.append(LL_list)
    print('Final Posterior Inference....')
    multiTrialInference(data, plot_trial=True)
    LL = computeLL(data)
    data.ll_iter.append([LL])
    comm.bcast(True, root=0)
    iteration = ii + 1
    np.savez(save_path, data_cca=data, iteration=iteration)
    return LL_list



if rank == 0:
    ## load data

    data_cca = np.load(fh_name, allow_pickle=True)['data_cca'].all()
    #data_cca = data_cca.subSampleTrial(np.arange(0,600,60))
    all_trials = np.array(list(data_cca.trialDur.keys()))
    trial_x_proc = all_trials.shape[0] // size + 1
    tr_dict = {}
    i0 = 0
    idx = -1
    for idx in range(size-1):
        tr_dict[idx] = all_trials[np.arange(i0, i0 + trial_x_proc)]
        i0 += trial_x_proc
    tr_dict[idx + 1] = all_trials[i0:]
    string = ''
    for k in tr_dict.keys():
        print(k, tr_dict[k])
        string += '%d - %s\n'%(k,str(tr_dict[k]))
    with open(iter_save, 'a') as fh:
        fh.write(string)
        fh.close()



else:
    tr_dict = {}
    data_cca = {}
    interrupt = False
# broadcast inputs
tr_dict = comm.bcast(tr_dict, root=0)
data_cca = comm.bcast(data_cca, root=0)

if rank != 0:
    # for the workers, just keep the trial needed for the optimization



    data_cca = data_cca.subSampleTrial(tr_dict[rank])
    if maxIter > 0:
        while True:
            interrupt = comm.bcast(interrupt, root=0)
            if interrupt:
                break
            # wait for the parameter update
            data_cca.xPar = comm.bcast(data_cca.xPar, root=0)
            data_cca.stimPar = comm.bcast(data_cca.stimPar, root=0)
            data_cca.priorPar = comm.bcast(data_cca.priorPar, root=0)

            # infer trials
            print('infer worker %d - start'%rank)
            multiTrialInference(data_cca, trial_list = tr_dict[rank])
            print('infer worker %d - end'%rank)
            posterior_list = comm.gather(data_cca.posterior_inf, root=0)

    # wait for the parameter update
    data_cca.xPar = comm.bcast(data_cca.xPar, root=0)
    data_cca.stimPar = comm.bcast(data_cca.stimPar, root=0)
    data_cca.priorPar = comm.bcast(data_cca.priorPar, root=0)
    multiTrialInference(data_cca, trial_list=tr_dict[rank],useGauss=0)
    posterior_list = comm.gather(data_cca.posterior_inf, root=0)



else:

    if maxIter > 0:
        ll = mpi_em(data_cca, tr_dict, maxIter=maxIter, tol=10 ** -3, method='sparse-Newton', tolPoissonOpt=10 ** -12,
           boundsW0=None, boundsW1=None, boundsD=None,save_every=1,save_path=save_path,iter_save=iter_save)

    # wait for the parameter update
    data_cca.xPar = comm.bcast(data_cca.xPar, root=0)
    data_cca.stimPar = comm.bcast(data_cca.stimPar, root=0)
    data_cca.priorPar = comm.bcast(data_cca.priorPar, root=0)
    data_noStim = deepcopy(data_cca)

    if hasattr(data_noStim, 'posterior_inf'):
        print('Clearing non-root inferred trial before E-step')
        lst_pop = []
        for tr in data_noStim.posterior_inf.keys():
            if not tr in tr_dict[0]:
                lst_pop.append(tr)
        for tr in lst_pop:
            data_noStim.posterior_inf.pop(tr)

    # final inference with no stimulus
    multiTrialInference(data_noStim, trial_list=tr_dict[rank], useGauss=0)
    posterior_list = comm.gather(data_noStim.posterior_inf, root=0)

    #np.savez('%d_fit_1iter.npz'%size,data_cca = data_cca)

    post_inf = {}
    for pst in posterior_list:
        for tr in pst.keys():
            post_inf[tr] = pst[tr]
    data_cca.posterior_inf_noStim = post_inf
    np.savez(save_path, data_cca=data_cca, iteration=maxIter)





