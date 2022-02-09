import numpy as np
import os,sys
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(basedir,'core'))
from inference import (inferTrial,makeK_big,retrive_t_blocks_fom_cov,multiTrialInference)
from data_structure import P_GPCCA
import unittest
from learnGaussianParam import MStepGauss,grag_GaussLL_wrt_Rinv, grag_GaussLL_wrt_Wd,learn_GaussianParams,\
    compileTrialStackedObsAndLatent_gauss,logLike_Gauss
from data_processing_tools import approx_grad,emptyStruct,sortGradient_idx
from learnPoissonParam import expectedLLPoisson,grad_expectedLLPoisson,compileTrialStackedObsAndLatent,all_trial_PoissonLL,\
    hess_Poisson_ithunit,grad_Poisson_ithunit,hess_Poisson_all_trials,multiTrial_PoissonLL,grad_Poisson_all_trials,newton_opt_CSR
from copy import deepcopy
from scipy.optimize import minimize
from scipy.linalg import block_diag
from time import perf_counter
np.random.seed(4)

class TestMStep(unittest.TestCase):

    def setUp(self):
        np.random.seed(4)

        super(TestMStep,self).__init__()
        self.eps = 10 ** -6
        self.D = 4
        self.K2 = 5
        self.K3 = 3
        self.K0 = 2
        self.N = 9
        self.N1 = 6
        ## Errors in gradient approximation have an average positive bias for each time point (due to the
        ## derivative of the exponential being monotonic). The larger the T the more the error is accumulating so
        # eventually the precision in the approx derivative will be lost.
        ## Using 0 < T < 100 should be enough for 10^-7 precisioo
        self.T = 50
        binSize = 50
        epsNoise = 0.001
        trialNum = 10
        # set up param
        self.W1 = np.random.normal(size=(self.D, self.K0))  # / 0.9

        self.W12 = 1 * np.random.normal(size=(self.N, self.K2))  # / 1.8

        self.W02 = np.random.normal(size=(self.N, self.K0))  # / 0.3
        self.W03 = np.random.normal(size=(self.N1, self.K0))  # / 0.3

        self.W13 = 1 * np.random.normal(size=(self.N1, self.K3))  # / .8

        self.R = np.random.uniform(0.1, 3, size=self.D)
        self.d1 = np.random.uniform(size=(self.D))
        self.d2 = 1 * np.random.uniform(-1, 0.2, size=(self.N))
        self.d3 = 1 * np.random.uniform(-1, 0.2, size=(self.N1))

        self.tau0 = np.random.uniform(0.2, 1, self.K0)
        self.tau2 = np.random.uniform(0.2, 1, self.K2)
        self.tau3 = np.random.uniform(0.2, 1, self.K3)

        K_big0 = makeK_big(self.K0, self.tau0, None, binSize, epsNoise=epsNoise, T=self.T, computeInv=False)[1]
        K_big2 = makeK_big(self.K2, self.tau2, None, binSize, epsNoise=epsNoise, T=self.T, computeInv=False)[1]
        K_big3 = makeK_big(self.K3, self.tau3, None, binSize, epsNoise=epsNoise, T=self.T, computeInv=False)[1]

        A = np.random.normal(size=(self.R.shape[0],) * 2)
        A = np.dot(A, A.T)
        _, U = np.linalg.eig(A)

        # rotate R
        self.Psi = np.dot(np.dot(U.T, np.diag(self.R)), U)

        # create latent and observation containers
        self.z = []
        self.z2 = []
        self.z3 = []
        self.x1 = []
        self.x2 = []
        self.x3 = []

        ## infer trials
        # create a fake data
        preproc = emptyStruct()
        preproc.numTrials = trialNum
        preproc.ydim = self.N + self.N1
        preproc.binSize = binSize
        preproc.T = np.array([self.T]*trialNum)
        preproc.covariates = {}
        for k in range(self.D):
            preproc.covariates['var%d' % k] = []
        preproc.data = []

        trueStimPar = {'W0': self.W1, 'd': self.d1, 'PsiInv': np.linalg.inv(self.Psi)}
        trueObsPar = [{'W0': self.W02, 'W1': self.W12, 'd': self.d2},
                      {'W0': self.W03, 'W1': self.W13, 'd': self.d3}]
        truePriorPar = [{'tau': self.tau0}, {'tau': self.tau2}, {'tau': self.tau3}]

        for tr in range(trialNum):

            self.z.append(np.random.multivariate_normal(mean=np.zeros(self.K0 * self.T), cov=K_big0, size=1).reshape(self.K0,
                                                                                                               self.T).T)
            self.z2.append(np.random.multivariate_normal(mean=np.zeros(self.K2 * self.T), cov=K_big2, size=1).reshape(self.K2,
                                                                                                                self.T).T)
            self.z3.append(np.random.multivariate_normal(mean=np.zeros(self.K3 * self.T), cov=K_big3, size=1).reshape(self.K3,
                                                                                                                self.T).T)

            mu = np.einsum('ij,tj->ti', self.W1, self.z[-1]) + self.d1
            x1 = np.zeros((self.T, self.W1.shape[0]))
            for t in range(mu.shape[0]):
                x1[t] = np.random.multivariate_normal(mean=mu[t], cov=self.Psi)
            self.x1.append(x1)

            self.x2.append(np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', self.W02, self.z[-1]) +
                                                   np.einsum('ij,tj->ti', self.W12, self.z2[-1]) + self.d2)))
            self.x3.append(np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', self.W03, self.z[-1]) +
                                                   np.einsum('ij,tj->ti', self.W13, self.z3[-1]) + self.d3)))
            for k in range(self.D):
                preproc.covariates['var%d' % k].append(self.x1[-1][:, k])


            preproc.data.append({'Y': np.hstack([self.x2[-1], self.x3[-1]])})

        # create the data structure
        self.struc = P_GPCCA(preproc, list(preproc.covariates.keys()), ['A', 'B'],
                                   np.array(['A'] * self.N + ['B'] * self.N1),
                                   np.ones(preproc.ydim, dtype=bool))
        self.struc.initializeParam([self.K0, self.K2, self.K3])

        # set the parameters to the true value
        self.struc.xPar = trueObsPar
        self.struc.priorPar = truePriorPar
        self.struc.stimPar = trueStimPar
        self.struc.epsNoise = epsNoise

        # infer a trial
        self.meanPost_list, self.covPost_list = multiTrialInference(self.struc, plot_trial=True,return_list_post=True)#muti(self.struc, 0)

        # posterior mean and covariance of factor self.z and self.z2
        _, self.mean_01, self.cov_01 = compileTrialStackedObsAndLatent(self.struc, 1, [0], self.struc.preproc.T[0], self.N, self.K0, self.K2)

    def test_GaussMStep(self):
        print('Test Gaussian Update')
        W = self.W1
        d = self.d1
        Psi = self.Psi
        # x1 = self.x1[0]
        # mean_t, cov_t = retrive_t_blocks_fom_cov(self.struc, 0, 0, self.meanPost_list, self.covPost_list)

        x1, mean_t, cov_t = compileTrialStackedObsAndLatent_gauss(self.struc, list(self.struc.trialDur.keys()),
                                                                        np.sum(list(self.struc.trialDur.values())), W.shape[0], W.shape[1])

        Wnew, dnew, PsiNew = MStepGauss(x1, mean_t, cov_t)

        grad1 = grag_GaussLL_wrt_Wd(x1, Wnew,dnew,np.linalg.inv(PsiNew),mean_t,cov_t)
        err1 = (np.abs(grad1).max())
        print('gradient theo [W,d] max error', err1)

        # func = lambda xx: logLike_Gauss(x1, Wnew, dnew, xx.reshape(Psi.shape), mean_t, cov_t)
        grad2 = grag_GaussLL_wrt_Rinv(x1, Wnew,dnew,np.linalg.inv(PsiNew),mean_t,cov_t)
        err2 = (np.abs(grad2).max())

        print('gradient theo Psi^-1 max error', err2)
        self.assertLessEqual(err1, self.eps, msg='parameter W or d are not 0s of the gradient: %f' % err1)
        self.assertLessEqual(err2, self.eps, msg='parameter Psi^-1 does not 0s of the gradient: %f' % err2)
        data = deepcopy(self.struc)
        learn_GaussianParams(data, test=False, isMPI=False)
        PsiNew2 = np.linalg.inv(data.stimPar['PsiInv'])
        T = np.sum(list(self.struc.trialDur.values()))
        N, K = W.shape
        x1, mean, cov = compileTrialStackedObsAndLatent_gauss(data, self.struc.trialDur.keys(), T, N, K)

        print('\nUse non gradient based method for omptimizing the parameters...')
        # create lambda func (this is needed to numerically fit a covariance, instead of optimizing for a generic matrix
        # optimize for a lower triangular matrix with positive diagonal element imposed as optimization constraint.
        # this basically will find numerically the cholesky factor of the covariance;
        def funcTril(x,N):
            y = np.zeros((N,N))
            y[np.tril_indices(N)] = x
            YYT = np.dot(y,y.T)
            return YYT

        bounds = np.array([[-10,10]]*(N*(N+1)//2),dtype=float)
        bounds[np.tril_indices(N)[0] == np.tril_indices(N)[1]] = [0.001,10] # force positive constraints

        func = lambda x: -logLike_Gauss(x1, data.stimPar['W0'], data.stimPar['d'], funcTril(x,N), mean, cov)
        xx = np.zeros(int(N*(N+1)//2))
        xx[np.tril_indices(N)[0] == np.tril_indices(N)[1]] = 1
        res = minimize(func,xx,bounds=bounds,method='SLSQP', tol=10**-12)
        covSLSQP = np.linalg.inv(funcTril(res.x,N))
        L = np.linalg.cholesky(np.linalg.inv(PsiNew2))
        x_anal = L[np.tril_indices(N)]
        self.assertLessEqual(func(x_anal),res.fun)
        err = np.abs(func(x_anal) - res.fun)
        self.assertLessEqual(err,10**-5)
        err = np.abs(covSLSQP - PsiNew2).max()
        print('max error covariance estimation:',err)
        self.assertLessEqual(err,10**-5)
        func = lambda x: -logLike_Gauss(x1, x[:N*K].reshape(N,K), x[N*K:], np.linalg.inv(PsiNew2), mean, cov)
        xx = np.zeros(N*(K+1))
        bounds = np.array([[-100,100]]*(N*(K+1)),dtype=float)
        res = minimize(func, xx, bounds=bounds, method='SLSQP', tol=10**-10)
        d_opt = res.x[-N:]
        err = np.abs(d_opt - data.stimPar['d']).max()
        print('max error d estimation:',err)
        self.assertLessEqual(err,10**-6)
        W_opt = res.x[:K * N].reshape(N, K)
        err = np.abs(W_opt - data.stimPar['W0']).max()
        print('max error W0 estimation:',err)
        self.assertLessEqual(err,2*10**-6)


    def test_PoissonMStep(self):
        _,xList = self.struc.get_observations(0)
        W0 = self.W02
        W1 = self.W12
        d = self.d2
        mean_t = self.mean_01
        cov_t = self.cov_01

        grad_func = lambda xx: -grad_expectedLLPoisson(xList[0], xx[:np.prod(W0.shape)].reshape(W0.shape),
                                                       xx[np.prod(W0.shape) + np.prod(W1.shape):], mean_t, cov_t,
                                                       C1=xx[np.prod(W0.shape):np.prod(W0.shape) + np.prod(
                                                           W1.shape)].reshape(W1.shape))

        func = lambda xx: -expectedLLPoisson(xList[0], xx[:np.prod(W0.shape)].reshape(W0.shape),
                                              xx[np.prod(W0.shape) + np.prod(W1.shape):], mean_t, cov_t,
                                              C1=xx[np.prod(W0.shape):np.prod(W0.shape) + np.prod(W1.shape)].reshape(
                                                  W1.shape))

        PARStack = np.hstack((W0.flatten(), W1.flatten(), d))

        app_grad = approx_grad(PARStack, PARStack.shape[0], func, 10 ** -5)
        grad = grad_func(PARStack)

        err = np.abs(app_grad-grad).mean() / np.mean(np.abs(grad))
        print('ERROR in Poisson MSetp Gradient:',err)
        self.assertLessEqual(err, self.eps, msg='average Poisson observation logLikelihood Hessian_z0z1 error: %f' % err)

        ## test the exact function used in the eM which iterates over trials
        N, K0 = W0.shape
        K1 = W1.shape[1]
        parStack = np.hstack((W0.flatten(), W1.flatten(), d))
        func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0), xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                               xx[N * (K0 + K1):],
                                               self.struc, 1, block_trials=100, isGrad=False)
        gr_func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0),
                                                  xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                                  xx[N * (K0 + K1):],
                                                  self.struc, 1, block_trials=100, isGrad=True)
        app_grad = approx_grad(parStack, parStack.shape[0], func, 10 ** -5)
        err = np.abs(gr_func(parStack) - app_grad).max()
        bounds = [[-10, 10]] * len(parStack)
        t0 = perf_counter()
        res_BFGS = minimize(func, np.zeros(parStack.shape), bounds=bounds, jac=gr_func, method='L-BFGS-B', tol=10 ** -12)
        t1 = perf_counter()

        tt0 = perf_counter()
        res_SLSQP = minimize(func, np.zeros(parStack.shape),bounds=bounds, jac=gr_func, method='SLSQP', tol=10 ** -10)
        tt1 = perf_counter()
        tt0_1 = perf_counter()
        res_SLSQP_wo = minimize(func, np.zeros(parStack.shape),bounds=bounds, method='SLSQP', tol=10 ** -10)
        tt1_1 = perf_counter()

        err = np.abs(res_BFGS.fun-res_SLSQP.fun)
        print('BFGS conv time: %f - nfev: %d'%(t1-t0,res_BFGS.nfev))
        print('SLSQP conv time with gradient: %f - nfev: %d'%(tt1-tt0,res_SLSQP.nfev))
        print('SLSQP conv time without gradient: %f - nfev: %d'%(tt1_1-tt0_1,res_SLSQP_wo.nfev))
        print('error in gradient vs non grediant method:',err)
        self.assertLessEqual(err,10**-4)

    def test_PoissonMStep_hess(self):
        _, xList = self.struc.get_observations(0)
        W0 = self.W02
        W1 = self.W12
        d = self.d2
        mean_t = self.mean_01
        cov_t = self.cov_01
        x = xList[0]

        dhh_d = x.sum(axis=0)
        dhh_Call = np.einsum('ti,tj->ij', x, mean_t)

        xdim = W0.shape[1]
        icord = 0
        parStack = np.hstack((W0[icord],W1[icord],d[icord]))
        hess = hess_Poisson_ithunit(parStack, mean_t, cov_t, inverse=False)

        def grad_poiss(pars,dhh_d,dhh_Call,icord,xdim):
            d_d,d_C,d_C1 = grad_Poisson_ithunit(pars[:xdim], pars[xdim:-1], pars[-1], mean_t, cov_t)
            concat = np.hstack(( dhh_Call[icord] - np.hstack((d_C,d_C1)),[dhh_d[icord] - d_d] ))
            return concat

        grad = lambda x: grad_poiss(x,dhh_d,dhh_Call,icord,xdim)
        ap_hes = approx_grad(parStack,(parStack.shape[0],)*2, grad,10**-5)
        err = np.linalg.norm(hess-ap_hes)
        self.assertLessEqual(err, 10 ** -6)

        def grad_func( xx, data):

            N = len(data.trialDur.keys())
            grd = multiTrial_PoissonLL(xx[:np.prod(W0.shape)].reshape(W0.shape), xx[np.prod(W0.shape):np.prod(W0.shape) + np.prod(
                                       W1.shape)].reshape(W1.shape), xx[np.prod(W0.shape) + np.prod(W1.shape):], data,
                                       1, trial_num=None, isGrad=True, trial_list=None, test=False)*N
            return grd

        rot = sortGradient_idx(W0.shape[0], [W0.shape[1], W1.shape[1],1],isReverse=False)
        gr_func = lambda xx: grad_func(xx, self.struc)
        parStack_all = np.hstack((W0.flatten(), W1.flatten(),d))#[rot]


        H = hess_Poisson_all_trials(d,W0,W1,1,self.struc,list(self.struc.trialDur.keys()),inverse=False,sparse=False)
        hess_ap = approx_grad(parStack_all,(parStack_all.shape[0],)*2,gr_func,10**-5)
        hess_ap = hess_ap[rot,:][:,rot]
        hess_blk = block_diag(*H)
        self.assertLessEqual(np.abs(hess_ap-hess_blk).max()/np.mean(np.abs(H)),10**-5)




        ## test optim

        def fun(par, idx_latent, data, rotInv):
            xx = par[rotInv]
            K0 = data.zdims[0]
            K1 = data.zdims[idx_latent]
            N = data.xPar[idx_latent - 1]['d'].shape[0]
            W0 = xx[:K0 * N].reshape(N, K0)
            W1 = xx[K0 * N:(K0 * N) + (K1 * N)].reshape(N, K1)
            d = xx[(K0 * N) + (K1 * N):]
            trNum = len(data.trialDur.keys())
            f = multiTrial_PoissonLL(W0, W1, d, data, 1, trial_num=None, isGrad=False, trial_list=None, test=False) * trNum
            return f

        def grad_fun(par, idx_latent, data, rotInv):
            xx = par[rotInv]
            K0 = data.zdims[0]
            K1 = data.zdims[idx_latent]
            N = data.xPar[idx_latent - 1]['d'].shape[0]
            W0 = xx[:K0 * N].reshape(N, K0)
            W1 = xx[K0 * N:(K0 * N) + (K1 * N)].reshape(N, K1)
            d = xx[(K0 * N) + (K1 * N):]
            grd = grad_Poisson_all_trials(d, W0, W1, idx_latent, data, trial_list=None)
            grd = grd#[rotinv]
            return grd

        def hess_fun(par, idx_latent,data, rotInv, inverse=True,sparse=True):
            xx = par[rotInv]
            K0 = data.zdims[0]
            K1 = data.zdims[idx_latent]
            N = data.xPar[idx_latent-1]['d'].shape[0]
            W0 = xx[:K0*N].reshape(N,K0)
            W1 = xx[K0 * N:(K0*N)+ (K1*N)].reshape(N, K1)
            d = xx[(K0*N)+ (K1*N):]
            hes = hess_Poisson_all_trials(d, W0, W1, idx_latent, data, trial_list=None, inverse=inverse, sparse=sparse)
            return hes


        rot = sortGradient_idx(W0.shape[0], [W0.shape[1], W1.shape[1],1],isReverse=False)
        rotInv = sortGradient_idx(W0.shape[0], [W0.shape[1], W1.shape[1], 1], isReverse=True)
        ff = lambda par: fun(par, 1, self.struc, rotInv)
        gg = lambda par: grad_fun(par, 1, self.struc, rotInv)
        hh = lambda par: hess_fun(par, 1, self.struc, rotInv, inverse=True, sparse=True)


        parStack_all = np.hstack((W0.flatten(), W1.flatten(),d))[rot]*0
        # HH = block_diag(*hh(parStack_all))
        t0 = perf_counter()
        Z0, feval, feval_hist = newton_opt_CSR(ff, gg, hh, parStack_all, tol=10**-8, max_iter=1000, disp_eval=True,max_having=30)
        par_fit = Z0[rotInv]
        t1=perf_counter()

        print('SPARSE HESS NEWTON %.5fsec'%(t1-t0))
        N,K0 = W0.shape
        K1 = W1.shape[1]
        func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0), xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                               xx[N * (K0 + K1):],
                                               self.struc, 1, block_trials=100, isGrad=False)
        gr_func = lambda xx: -all_trial_PoissonLL(xx[:N * K0].reshape(N, K0),
                                                  xx[N * K0:N * (K0 + K1)].reshape(N, K1),
                                                  xx[N * (K0 + K1):],
                                                  self.struc, 1, block_trials=100, isGrad=True)

        bounds = [[-10, 10]] * len(parStack_all)
        t0 = perf_counter()
        res_BFGS = minimize(func, np.zeros(parStack_all.shape), bounds=bounds, jac=gr_func, method='L-BFGS-B',
                            tol=10 ** -12)
        t1 = perf_counter()
        print('BFGS %.5fsec'%(t1-t0))
        err = np.abs(func(par_fit) - res_BFGS.fun)
        self.assertLessEqual(err,10**-4)


if __name__ == "__main__":
    unittest.main()
