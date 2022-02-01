import numpy as np
import os,sys
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('base folder:', basedir)
sys.path.append(os.path.join(basedir,'core'))
sys.path.append(os.path.join(basedir,'initialization'))
from inference_factorized import reconstruct_post_mean_and_cov, factorized_logLike,\
    grad_factorized_logLike,hess_factorized_logLike,all_trial_ll_grad_hess_factorized,reconstruct_post_mean_and_cov,\
    newton_optim_map
from inference import makeK_big
from data_structure import P_GPCCA
import unittest
from scipy.linalg import block_diag
from data_processing_tools import emptyStruct,approx_grad
from data_processing_tools_factorized import preproc_post_mean_factorizedModel

class TestLogLikelihood(unittest.TestCase):
    def setUp(self):
        np.random.seed(4)

        super(TestLogLikelihood, self).__init__()
        self.eps = 10 ** -5
        self.D = 4
        self.K2 = 5
        self.K3 = 3
        self.K0 = 2
        self.N = 50
        self.N1 = 6
        ## Errors in gradient approximation have an average positive bias for each time point (due to the
        ## derivative of the exponential being monotonic). The larger the T the more the error is accumulating so
        # eventually the precision in the approx derivative will be lost.
        ## Using 0 < T < 100 should be enough for 10^-7 precisioo
        self.T = 50
        binSize = 50
        epsNoise = 0.001
        trialNum = 3
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
        preproc.T = np.array([self.T] * trialNum)
        preproc.covariates = {}
        for k in range(self.D):
            preproc.covariates['var%d' % k] = []
        preproc.data = []

        trueStimPar = {'W0': self.W1, 'd': self.d1, 'PsiInv': np.linalg.inv(self.Psi)}
        trueObsPar = [{'W0': self.W02, 'W1': self.W12, 'd': self.d2},
                      {'W0': self.W03, 'W1': self.W13, 'd': self.d3}]
        truePriorPar = [{'tau': self.tau0}, {'tau': self.tau2}, {'tau': self.tau3}]

        for tr in range(trialNum):

            self.z.append(
                np.random.multivariate_normal(mean=np.zeros(self.K0 * self.T), cov=K_big0, size=1).reshape(self.K0,
                                                                                                           self.T).T)
            self.z2.append(
                np.random.multivariate_normal(mean=np.zeros(self.K2 * self.T), cov=K_big2, size=1).reshape(self.K2,
                                                                                                           self.T).T)
            self.z3.append(
                np.random.multivariate_normal(mean=np.zeros(self.K3 * self.T), cov=K_big3, size=1).reshape(self.K3,
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

    # def test_unblock(self):
    #     # create a struct with 5x2 latent dim and a block matrix
    #     struc = emptyStruct()
    #     struc.zdims = [5,2]
    #     struc.trialDur = [50]
    #     A = block_diag(*([np.ones((5,5))]*50))
    #     B = block_diag(*([np.ones((5,2))*2]*50))
    #     C = block_diag(*([np.ones((2,2))*3]*50))
    #     M = np.block([[A, B], [B.T, C]])
    #     mu = np.ones(M.shape[0])
    #     MMOrig = block_diag(*([np.block([[np.ones((5,5)), np.ones((5,2))*2],
    #                                      [np.ones((2,5))*2,np.ones((2,2))*3]])]*50))
    #     _,t_M = retrive_t_blocks_fom_cov(struc, 0, 1, [mu],[M])
    #     MM = block_diag(*t_M)
    #     self.assertEqual(np.abs(MM-MMOrig).sum(),0)

    def test_gradLogLike(self):
        stim, xList = self.struc.get_observations(0)
        zstack = np.random.normal(size=self.struc.trialDur[0]*np.sum(self.struc.zdims))

        func = lambda zbar: factorized_logLike(self.struc, 0, stim, xList, zbar=zbar)
        grad_func = lambda zbar: grad_factorized_logLike(self.struc, 0, stim, xList, zbar=zbar)[0]

        app_grad = approx_grad(zstack, zstack.shape[0], func, epsi=10 ** -5)
        grad_res = grad_func(zstack)
        err = np.max(np.abs(app_grad-grad_res)/np.mean(app_grad))
        self.assertLessEqual(err,10**-5)

        print('\ngrad error: ',err, 'max grad:',np.max(grad_res))

    def test_hessLogLike(self):
        stim, xList = self.struc.get_observations(0)
        zstack = np.random.normal(size=self.struc.trialDur[0] * np.sum(self.struc.zdims))

        grad_func = lambda zbar: grad_factorized_logLike(self.struc, 0, stim, xList, zbar=zbar)[0]
        hess_func = lambda zbar: hess_factorized_logLike(self.struc, 0, stim, xList, zbar=zbar)[0]
        app_hess = approx_grad(zstack,(zstack.shape[0],zstack.shape[0]),grad_func,epsi=10**-5)
        hess_res = hess_func(zstack)
        hess_res = hess_res.to_scipy().toarray()
        err = np.max(np.abs(app_hess-hess_res))
        self.assertLessEqual(err,10**-4)
        print('\nhess error: ',err, 'max hess:',np.max(hess_res))

    def test_newtonOptim(self):
        zmap, success, td, ll, ll_hist = newton_optim_map(self.struc, tol=10 ** -10, max_iter=100, max_having=20,
                                                  disp_ll=True, init_zeros=True,
                                                 useNewton=True)
        self.assertGreaterEqual(np.min(np.diff(ll_hist)),0)
        reconstruct_post_mean_and_cov(self.struc, zmap, td)
        Zopt, _ = preproc_post_mean_factorizedModel(self.struc, returnDict=False)

        grd_at_opt = all_trial_ll_grad_hess_factorized(self.struc, zmap, tr_dict=td, isDict=False, returnLL=False)[1]
        self.assertLessEqual(np.abs(zmap-Zopt).max(),10**-5)
        self.assertLessEqual(np.abs(grd_at_opt).max(),10**-6)
        ## check that the posterior still looks good


if __name__ == '__main__':
    unittest.main()