import numpy as np
import os,sys,inspect
basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(basedir,'core'))
from inference import (makeK_big,GPLogLike,grad_GPLogLike,hess_GPLogLike,
                       gaussObsLogLike,grad_gaussObsLogLike,hess_gaussObsLogLike,
                       poissonLogLike,grad_poissonLogLike,hess_poissonLogLike,logDetCompute)
import unittest
from data_processing_tools import approx_grad

np.random.seed(4)



class TestLogLikelihood(unittest.TestCase):
    def setUp(self):
        super(TestLogLikelihood,self).__init__()
        self.eps = 10**-7
        self.D = 4
        self.K = 5
        self.K0 = 2
        self.N = 50
        self.T = 100

        self.z = np.random.normal(size=(self.T, self.K))
        self.W1 = np.random.normal(size=(self.D, self.K)) / 0.3

        self.W2 = 1 * np.random.normal(size=(self.N, self.K)) / 1.8
        self.W0 = np.random.normal(size=(self.N, self.K0)) / 0.3

        self.W3 = 0 * np.random.normal(size=(self.N, self.K)) / .8

        self.R = np.random.uniform(0.1, 3, size=self.D)
        self.d1 = np.random.uniform(size=(self.D))
        self.d2 = 0 * np.random.uniform(-1, 0.2, size=(self.N))
        self.d3 = 0 * np.random.uniform(-1, 0.2, size=(self.N))

        self.x1 = np.random.normal(loc=np.einsum('ij,tj->ti', self.W1, self.z) + self.d1,
                                   scale=np.tile(np.sqrt(self.R), self.T).reshape(self.T, self.D))
        self.x2 = np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', self.W2, self.z) + self.d2))
        self.x3 = np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', self.W3, self.z) + self.d3))

        self.z0 = np.random.normal(size=(self.T, self.K0))

        trialDur = None
        binSize = 50
        TT = 100
        self.KK, self.K_big, self.K_big_inv, self.logdet_K_big = makeK_big(self.K, np.random.uniform(0.4, 1, size=self.K), trialDur, binSize, epsNoise=0.0000001, T=TT,

                                                       computeInv=True)

        self.zbar = np.random.multivariate_normal(mean=np.zeros(self.K_big.shape[0]), cov=self.K_big, size=1)[0]

    def test_KBigInv(self):
        err = np.max(np.abs(np.eye(self.K_big.shape[0]) - np.dot(self.K_big, self.K_big_inv)))
        print('\nERROR INV', err)
        self.assertLessEqual(err,self.eps)

    def test_logDetKbig(self):
        # log(det(A)) = 5*4/2 = 10
        A = np.diag(np.exp(np.arange(1,5)))

        # crete a rotation matrix that does not change the det
        tmp = np.random.normal(size=(4,4))
        tmp = np.dot(tmp,tmp.T)
        _,U = np.linalg.eig(tmp)
        ARot = np.dot(np.dot(U.T, A),U) # logdet = 10

        # check the logdet
        print('log-det err',np.abs(logDetCompute(ARot) - 10))
        self.assertLessEqual(np.abs(logDetCompute(ARot) - 10), self.eps)

    def test_GPLike(self):
        func = lambda z: GPLogLike(z, self.K_big_inv)
        grad = grad_GPLogLike(self.zbar, self.K_big_inv)
        app_grad = approx_grad(self.zbar, self.zbar.shape[0], func, 10 ** -4)
        err = (np.linalg.norm(grad - app_grad, axis=0) / np.abs(grad.shape[0] * np.mean(np.abs(grad))))
        print('\nERROR in GP gradient',err)
        self.assertLessEqual(err, self.eps, msg='average GP logLikelihood gradient error: %f'%err)

        func = lambda z: grad_GPLogLike(z, self.K_big_inv)
        hess = hess_GPLogLike(self.z, self.K_big_inv)
        app_hes = approx_grad(self.zbar, (self.zbar.shape[0], self.zbar.shape[0]), func, 10 ** -4)
        err = (np.linalg.norm(app_hes - hess) / np.abs(hess.shape[0] * hess.shape[1] * np.mean(np.abs(hess))))
        print('ERROR in GP hessian',err)
        self.assertLessEqual(err, self.eps, msg='average GP logLikelihood Hessian error: %f' % err)

    def test_GaussOBSLike(self):
        RR = np.random.uniform(size=np.diag(self.R).shape)
        RR = np.dot(RR, RR.T)
        res2 = grad_gaussObsLogLike(self.x1, self.z, self.W1, self.d1, RR)
        func = lambda z: gaussObsLogLike(self.x1, z.reshape(self.T, self.K), self.W1, self.d1, RR)
        app_grad = approx_grad(self.z.flatten(), np.prod(self.z.shape), func, 10 ** -4)
        err = (np.linalg.norm(res2 - app_grad, axis=0) / np.abs(res2.shape[0] * np.mean(np.abs(res2)))).mean()
        print('\nERROR in Gaussian gradient',err)
        self.assertLessEqual(err, self.eps, msg='average Gaussian observation logLikelihood gradient error: %f' % err)

        hess = hess_gaussObsLogLike(self.x1, self.z, self.W1, self.d1, RR)
        func = lambda z: grad_gaussObsLogLike(self.x1, z.reshape(self.T, self.K), self.W1, self.d1, RR)
        app_hes = approx_grad(self.z.flatten(), (self.K * self.T, self.K * self.T), func, 10 ** -4)
        err = (np.linalg.norm(app_hes - hess) / np.abs(hess.shape[0] * hess.shape[1] * np.mean(np.abs(hess))))
        print('ERROR in Gaussian hessian',err)
        self.assertLessEqual(err, self.eps, msg='average Gaussian observation logLikelihood Hessian error: %f' % err)

    def test_PoissonOBSLike(self):

        func = lambda zz: poissonLogLike(self.x2, zz.reshape(self.T, self.K),
                                         self.z.reshape(self.T, self.K)*0.2, self.W2, self.W2 * 0.5, self.d2)
        grad_z0, grad_z1 = grad_poissonLogLike(self.x2, self.z, self.z*0.2, self.W2, self.W2 * 0.5, self.d2)
        app_grad = approx_grad(self.z.flatten(), np.prod(self.z.shape), func, 10 ** -4)
        err = np.linalg.norm(grad_z0.flatten() - app_grad, axis=0) / np.abs(grad_z0.flatten().shape[0] * np.mean(np.abs(grad_z0.flatten())))
        self.assertLessEqual(err, self.eps, msg='average Poisson observation logLikelihood gradient error: %f' % err)

        hess_z0,hess_z1,hess_z0z1 = hess_poissonLogLike(self.x2, self.z, self.z*0.2, self.W2, self.W2 * 0.5, self.d2)
        func = lambda zz: grad_poissonLogLike(self.x2, zz.reshape(self.T, self.K),
                                         self.z.reshape(self.T, self.K)*0.2, self.W2, self.W2 * 0.5, self.d2)[0].flatten()
        app_hess_z0 = approx_grad(self.z.flatten(), (self.K * self.T, self.K * self.T), func, 10 ** -4)
        err = (np.linalg.norm(hess_z0 - app_hess_z0, axis=0) / np.abs(app_hess_z0.shape[0] * np.mean(np.abs(hess_z0)))).mean()
        self.assertLessEqual(err, self.eps, msg='average Poisson observation logLikelihood Hessian_z0 error: %f' % err)

        func = lambda zz: grad_poissonLogLike(self.x2, self.z.reshape(self.T, self.K),
                                              zz.reshape(self.T, self.K), self.W2, self.W2 * 0.5, self.d2)[
            1].flatten()
        app_hess_z1 = approx_grad(self.z.flatten()*0.2, (self.K * self.T, self.K * self.T), func, 10 ** -4)
        err = (np.linalg.norm(hess_z1 - app_hess_z1, axis=0) / np.abs(
            app_hess_z1.shape[0] * np.mean(np.abs(hess_z1)))).mean()
        print('\nERROR in Poisson gradient',err)
        self.assertLessEqual(err, self.eps, msg='average Poisson observation logLikelihood Hessian_z1 error: %f' % err)

        func = lambda zz: grad_poissonLogLike(self.x2, self.z.reshape(self.T, self.K),
                                              zz.reshape(self.T, self.K), self.W2, self.W2 * 0.5, self.d2)[
            0].flatten()
        app_hess_z0z1 = approx_grad(self.z.flatten() * 0.2, (self.K * self.T, self.K * self.T), func, 10 ** -4)
        err = (np.linalg.norm(hess_z0z1 - app_hess_z0z1, axis=0) / np.abs(
            app_hess_z0z1.shape[0] * np.mean(np.abs(hess_z0z1)))).mean()
        print('ERROR in Poisson hessian',err)
        self.assertLessEqual(err, self.eps, msg='average Poisson observation logLikelihood Hessian_z0z1 error: %f' % err)

if __name__ == "__main__":
    unittest.main()
