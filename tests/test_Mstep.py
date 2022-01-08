import numpy as np
import os,sys,inspect
basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(basedir,'core'))
sys.path.append(os.path.join(basedir,'firefly_utils'))
from inference import (inferTrial,makeK_big,retrive_t_blocks_fom_cov)
from behav_class import emptyStruct
from data_structure import P_GPCCA
import unittest
from learnGaussianParam import MStepGauss,grag_GaussLL_wrt_Rinv, grag_GaussLL_wrt_Wd
from data_processing_tools import approx_grad
from learnPoissonParam import expectedLLPoisson,grad_expectedLLPoisson


class TestMStep(unittest.TestCase):

    def setUp(self):
        super(TestMStep,self).__init__()
        self.eps = 10 ** -6
        self.D = 4
        self.K2 = 5
        self.K3 = 3
        self.K0 = 2
        self.N = 7
        self.N1 = 6
        ## Errors in gradient approximation have an average positive bias for each time point (due to the
        ## derivative of the exponential being monotonic). The larger the T the more the error is accumulating so
        # eventually the precision in the approx derivative will be lost.
        ## Using 0 < T < 100 should be enough for 10^-7 precisioo
        self.T = 200
        binSize = 50
        epsNoise = 0.001
        self.tau0 = np.random.uniform(0.2, 1, self.K0)

        K_big = makeK_big(self.K0, self.tau0, None, binSize, epsNoise=epsNoise, T=self.T, computeInv=False)[1]
        self.z = np.random.multivariate_normal(mean=np.zeros(self.K0 * self.T), cov=K_big, size=1).reshape(self.K0,
                                                                                                           self.T).T
        self.tau2 = np.random.uniform(0.2, 1, self.K2)
        K_big = makeK_big(self.K2, self.tau2, None, binSize, epsNoise=epsNoise, T=self.T, computeInv=False)[1]
        self.z2 = np.random.multivariate_normal(mean=np.zeros(self.K2 * self.T), cov=K_big, size=1).reshape(self.K2,
                                                                                                            self.T).T

        self.tau3 = np.random.uniform(0.2, 1, self.K3)
        K_big = makeK_big(self.K3, self.tau3, None, binSize, epsNoise=epsNoise, T=self.T, computeInv=False)[1]
        self.z3 = np.random.multivariate_normal(mean=np.zeros(self.K3 * self.T), cov=K_big, size=1).reshape(self.K3,
                                                                                                            self.T).T
        self.W1 = np.random.normal(size=(self.D, self.K0)) #/ 0.9

        self.W12 = 1 * np.random.normal(size=(self.N, self.K2)) #/ 1.8

        self.W02 = np.random.normal(size=(self.N, self.K0)) #/ 0.3
        self.W03 = np.random.normal(size=(self.N1, self.K0)) #/ 0.3

        self.W13 = 1 * np.random.normal(size=(self.N1, self.K3)) #/ .8

        self.R = np.random.uniform(0.1, 3, size=self.D)
        self.d1 = np.random.uniform(size=(self.D))
        self.d2 = 1 * np.random.uniform(-1, 0.2, size=(self.N))
        self.d3 = 1 * np.random.uniform(-1, 0.2, size=(self.N1))
        A = np.random.normal(size=(self.R.shape[0],)*2)
        A = np.dot(A,A.T)
        _,U = np.linalg.eig(A)

        self.Psi = np.dot(np.dot(U.T,np.diag(self.R)),U)
        self.x1 = np.random.normal(loc=np.einsum('ij,tj->ti', self.W1, self.z) + self.d1,
                                   scale=np.tile(np.sqrt(self.R), self.T).reshape(self.T, self.D))
        self.x2 = np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', self.W02, self.z) +
                                               np.einsum('ij,tj->ti', self.W12, self.z2) + self.d2))
        self.x3 = np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', self.W03, self.z) +
                                               np.einsum('ij,tj->ti', self.W13, self.z3) + self.d3))

        ## infer trials
        # create a fake data
        preproc = emptyStruct()
        preproc.numTrials = 1
        preproc.ydim = self.N + self.N1
        preproc.binSize = binSize
        preproc.T = np.array([self.T])
        preproc.covariates = {}
        for k in range(self.D):
            preproc.covariates['var%d' % k] = [self.x1[:, k]]
        trueStimPar = {'W0': self.W1, 'd': self.d1, 'PsiInv': np.linalg.inv(self.Psi)}

        preproc.data = [{'Y': np.hstack([self.x2, self.x3])}]
        trueObsPar = [{'W0': self.W02, 'W1': self.W12, 'd': self.d2},
                      {'W0': self.W03, 'W1': self.W13, 'd': self.d3}]

        truePriorPar = [{'tau': self.tau0}, {'tau': self.tau2}, {'tau': self.tau3}]

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
        self.meanPost, self.covPost = inferTrial(self.struc, 0)

        # posterior mean and covariance of factor self.z and self.z2
        self.mean_01, self.cov_01 = retrive_t_blocks_fom_cov(self.struc, 0, 1, [self.meanPost], [self.covPost])

    def test_GaussMStep(self):
        W = self.W1
        d = self.d1
        Psi = self.Psi
        x1 = self.x1
        mean_t, cov_t = retrive_t_blocks_fom_cov(self.struc, 0, 0, [self.meanPost], [self.covPost])
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



if __name__ == "__main__":
    unittest.main()
