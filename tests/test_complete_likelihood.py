import numpy as np
import os,sys,inspect
basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(basedir,'core'))
from inference import (PpCCA_logLike,grad_PpCCA_logLike,hess_PpCCA_logLike,makeK_big,approx_grad)
basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(basedir, 'firefly_utils'))
sys.path.append(os.path.join(basedir, 'core'))
from behav_class import emptyStruct
from data_structure import GP_pCCA_input
import unittest

class TestLogLikelihood(unittest.TestCase):
    def setUp(self):
        self.eps = 10 ** -7
        np.random.seed(4)
        # set up synthetic data
        preproc = emptyStruct()
        preproc.numTrials = 1
        preproc.covariates = {}

        preproc.ydim = 50
        preproc.binSize = 50
        preproc.T = np.array([20])

        tau = np.array([0.9, 0.2, 0.4, 0.2, 0.8])
        K0 = len(tau)
        epsNoise = 0.001
        K_big = makeK_big(K0, tau, None, preproc.binSize, epsNoise=epsNoise,
                          T=preproc.T[0], computeInv=False)[1]
        self.z0 = np.random.multivariate_normal(mean=np.zeros(K0 * preproc.T[0]), cov=K_big,
                                          size=1).reshape(preproc.T[0], K0)

        # latent for the counts
        tau1 = np.array([1.1, 1.3])
        K_big = makeK_big(len(tau1), tau1, None, preproc.binSize, epsNoise=epsNoise, T=preproc.T[0], computeInv=False)[1]
        self.z1 = np.random.multivariate_normal(mean=np.zeros(preproc.T[0] * len(tau1)), cov=K_big, size=1).reshape(
            preproc.T[0], len(tau1))

        W1 = np.random.normal(size=(preproc.ydim, len(tau1)))
        W0 = np.random.normal(size=(preproc.ydim, K0))
        d = -0.2
        preproc.data = [
            {'Y': np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', W0, self.z0) + np.einsum('ij,tj->ti', W1, self.z1) + d))}]

        preproc.covariates['var1'] = [
            np.random.multivariate_normal(mean=np.dot(W0/2., self.z0.T)[0], cov=np.eye(preproc.T[0]))]
        preproc.covariates['var2'] = [
            np.random.multivariate_normal(mean=np.dot(W0, self.z0.T)[1], cov=np.eye(preproc.T[0]))]


        # create the data struct
        self.struc = GP_pCCA_input(preproc, ['var1', 'var2'], ['PPC'], np.array(['PPC'] * preproc.ydim),
                              np.ones(preproc.ydim, dtype=bool))
        self.struc.initializeParam([K0, self.z1.shape[1]])
        self.epsNoise=0.001

    def test_gradLogLike(self):
        stim, xList = self.struc.get_observations(0)
        zstack = np.hstack((self.z0.flatten(), self.z1.flatten()))

        # res = PpCCA_logLike(zstack, stim, xList, priorPar=self.struc.priorPar, stimPar=self.struc.stimPar, xPar=self.struc.xPar,
        #                     binSize=self.struc.binSize, epsNoise=0.0001)

        grad_res = grad_PpCCA_logLike(zstack, stim, xList, priorPar=self.struc.priorPar, stimPar=self.struc.stimPar,
                                      xPar=self.struc.xPar,
                                      binSize=self.struc.binSize, epsNoise=self.epsNoise)

        func = lambda z: PpCCA_logLike(z, stim, xList, priorPar=self.struc.priorPar, stimPar=self.struc.stimPar, xPar=self.struc.xPar,
                                       binSize=self.struc.binSize, epsNoise=self.epsNoise)
        app_grad = approx_grad(zstack, zstack.shape[0], func, epsi=10 ** -4)
        err = np.max(np.abs(app_grad-grad_res))
        self.assertLessEqual(err,self.eps)
        print('\ngrad error: ',err, 'max grad:',np.max(grad_res))

    def test_hessLogLike(self):
        stim, xList = self.struc.get_observations(0)
        zstack = np.hstack((self.z0.flatten(), self.z1.flatten()))

        hess_res = hess_PpCCA_logLike(zstack, stim, xList, priorPar=self.struc.priorPar, stimPar=self.struc.stimPar,
                                      xPar=self.struc.xPar,
                                      binSize=self.struc.binSize, epsNoise=self.epsNoise)

        func = lambda z: grad_PpCCA_logLike(z, stim, xList, priorPar=self.struc.priorPar, stimPar=self.struc.stimPar, xPar=self.struc.xPar,
                                       binSize=self.struc.binSize, epsNoise=self.epsNoise)
        app_hess = approx_grad(zstack,(zstack.shape[0],zstack.shape[0]),func,epsi=10**-4)
        err = np.max(np.abs(app_hess-hess_res))
        self.assertLessEqual(err,self.eps)
        print('\nhess error: ',err, 'max hess:',np.max(hess_res))

if __name__ == '__main__':
    unittest.main()