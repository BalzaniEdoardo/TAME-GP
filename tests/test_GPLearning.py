import numpy as np
import os,sys,inspect
basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(basedir,'core'))

from data_structure import *
import unittest
from scipy.linalg import block_diag
from scipy.optimize import minimize
from scipy.stats import pearsonr
from gen_synthetic_data import dataGen
from learnGPParams import allTrial_grad_expectedLLGPPrior
from data_processing_tools import parse_fullCov_latDim,approx_grad,emptyStruct

class TestGPLearning(unittest.TestCase):
    def setUp(self):
        super(TestGPLearning,self).__init__()
        np.random.seed(4)
        basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
        pathFile = os.path.join(basedir,'inference_syntetic_data','sim_150Trials.npy')
        if pathFile:
            dat = np.load(pathFile,allow_pickle=True).all()
        else:
            dat = dataGen(150, T=50)
        lam_0 = 2 * np.log(
            ((dat.cca_input.priorPar[1]['tau'] * dat.cca_input.binSize / 1000))
        )
        eps = dat.cca_input.epsNoise
        mu_list = []
        cov_list = []
        for k in range(len(dat.cca_input.trialDur)):
            m, s = parse_fullCov_latDim(dat.cca_input, dat.meanPost[k], dat.covPost[k], dat.cca_input.trialDur[k])
            mu_list.append(m[1])
            cov_list.append(s[1])

        func = lambda lam_0: -allTrial_grad_expectedLLGPPrior(lam_0, mu_list, cov_list, eps,
                                                              max(dat.cca_input.trialDur), isGrad=False)
        func_grad = lambda lam_0: -allTrial_grad_expectedLLGPPrior(lam_0,  mu_list, cov_list,
                                                                   dat.cca_input.binSize, eps,
                                                                   max(dat.cca_input.trialDur) + 1, isGrad=True)
        self.res = minimize(func, np.zeros(len(lam_0)), jac=func_grad, method='L-BFGS-B', tol=10 ** -10)
        self.target = lam_0
        self.dat = dat

    def test_gradExpLLGP(self):
        dat = self.dat
        eps = self.dat.cca_input.epsNoise

        mu_list = []
        cov_list = []
        for k in range(len(dat.cca_input.trialDur)):
            m, s = parse_fullCov_latDim(dat.cca_input, dat.meanPost[k], dat.covPost[k], dat.cca_input.trialDur[k])
            mu_list.append(m[1])
            cov_list.append(s[1])

        func = lambda lam_0: -allTrial_grad_expectedLLGPPrior(lam_0,  mu_list, cov_list,
                                                              dat.cca_input.binSize, eps,
                                                              max(dat.cca_input.trialDur), isGrad=False)
        func_grad = lambda lam_0: -allTrial_grad_expectedLLGPPrior(lam_0, mu_list, cov_list,
                                                                   dat.cca_input.binSize, eps,
                                                                   max(dat.cca_input.trialDur) + 1, isGrad=True)
        xDim = m[1].shape[0]
        x0 = np.random.uniform(-5,-3,size=xDim)
        ap_grad = approx_grad(x0, xDim, func,10**-5)
        err = np.linalg.norm(ap_grad-func_grad(x0))
        print('grad error: ',err)
        self.assertLessEqual(err/np.mean(ap_grad),10**-5)

    def test_GPOoptimCorr(self):
        cor = pearsonr(self.res.x,self.target)[0]
        self.assertGreaterEqual(cor,0.99)

if __name__ == "__main__":
    unittest.main()

