import numpy as np
from inference import *
from learnPoissonParam import *
from data_structure import *
from data_processing_tools import emptyStruct

class dataGen(object):

    def __init__(self, trNum, T=50, D=4, K0=2, K2=5, K3=3, N=7, N1=6):
        super(dataGen,self).__init__()
        ## Errors in gradient approximation have an average positive bias for each time point (due to the
        ## derivative of the exponential being monotonic). The larger the T the more the error is accumulating so
        # eventually the precision in the approx derivative will be lost.
        ## Using 0 < T < 100 should be enough for 10^-7 precisioo

        binSize = 50
        epsNoise = 0.001

        tau0 = np.random.uniform(0.2, 1, K0)
        K_big0 = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=False)[1]

        tau2  = np.random.uniform(0.2, 1, K2)
        K_big2 = makeK_big(K2, tau2 , None, binSize, epsNoise=epsNoise, T=T, computeInv=False)[1]

        tau3 = np.random.uniform(0.2, 1, K3)
        K_big3 = makeK_big(K3, tau3, None, binSize, epsNoise=epsNoise, T=T, computeInv=False)[1]

        W1 = np.random.normal(size=(D, K0)) #/ 0.9

        W12 = 1 * np.random.normal(size=(N, K2)) #/ 1.8

        W02 = np.random.normal(size=(N, K0)) #/ 0.3
        W03 = np.random.normal(size=(N1, K0)) #/ 0.3

        W13 = 1 * np.random.normal(size=(N1, K3)) #/ .8

        R = np.random.uniform(0.1, 3, size=D)
        d1 = np.random.uniform(size=(D))
        d2 = 1 * np.random.uniform(-1, 0.2, size=(N))
        d3 = 1 * np.random.uniform(-1, 0.2, size=(N1))
        A = np.random.normal(size=(R.shape[0],)*2)
        A = np.dot(A,A.T)
        _,U = np.linalg.eig(A)

        Psi = np.dot(np.dot(U.T,np.diag(R)),U)

        # create a fake data
        preproc = emptyStruct()
        preproc.numTrials = trNum
        preproc.ydim = N + N1
        preproc.binSize = binSize
        preproc.T = np.array([T]*trNum)
        preproc.covariates = {}
        preproc.data = []

        for k in range(D):
            preproc.covariates['var%d' % k] = []
        self.ground_truth_latent = []
        for tr in range(trNum):
            z = np.random.multivariate_normal(mean=np.zeros(K0 * T), cov=K_big0, size=1).reshape(K0, T).T
            z2 = np.random.multivariate_normal(mean=np.zeros(K2 * T), cov=K_big2, size=1).reshape(K2,T).T
            z3 = np.random.multivariate_normal(mean=np.zeros(K3 * T), cov=K_big3, size=1).reshape(K3,T).T

            x1 = np.random.normal(loc=np.einsum('ij,tj->ti', W1, z) + d1,
                                       scale=np.tile(np.sqrt(R), T).reshape(T, D))
            x2 = np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', W02, z) +
                                                   np.einsum('ij,tj->ti', W12, z2) + d2))
            x3 = np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', W03, z) +
                                                   np.einsum('ij,tj->ti', W13, z3) + d3))

            for k in range(D):
                preproc.covariates['var%d' % k].append(x1[:, k])

            preproc.data.append({'Y': np.hstack([x2, x3])})

            self.ground_truth_latent.append(np.hstack((z,z2,z3)))

        ## infer trials


        trueStimPar = {'W0': W1, 'd': d1, 'PsiInv': np.linalg.inv(Psi)}


        trueObsPar = [{'W0': W02, 'W1': W12, 'd': d2},
                      {'W0': W03, 'W1': W13, 'd': d3}]

        truePriorPar = [{'tau': tau0}, {'tau': tau2 }, {'tau': tau3}]

        # create the data structure
        self.cca_input = P_GPCCA(preproc, list(preproc.covariates.keys()), ['A', 'B'],
                                   np.array(['A'] * N + ['B'] * N1),
                                   np.ones(preproc.ydim, dtype=bool))
        self.cca_input.initializeParam([K0, K2, K3])
        # set the parameters to the true value
        self.cca_input.xPar = trueObsPar
        self.cca_input.priorPar = truePriorPar
        self.cca_input.stimPar = trueStimPar
        self.cca_input.epsNoise = epsNoise

        # infer trials
        self.meanPost = []
        self.covPost = []

        multiTrialInference(self.cca_input)
        # for tr in range(trNum):
        #     print('infer trial %d'%tr)
        #     meanPost, covPost = inferTrial(self.cca_input, tr)
            # self.meanPost.append(meanPost)
            # self.covPost.append(covPost)

if __name__ == '__main__':
    from data_processing_tools import retrive_t_blocks_fom_cov
    import matplotlib.pylab as plt
    from time import perf_counter
    t0 = perf_counter()
    data = dataGen(150,T=50)
    t1 = perf_counter()
    print(t1-t0)
    i_latent = 2
    tr = 0
    mean_t,cov_t = retrive_t_blocks_fom_cov(data.cca_input,tr, i_latent,data.meanPost,data.covPost)
    K0 = (i_latent!=0)*data.cca_input.zdims[0]
    i0 = np.sum(data.cca_input.zdims[:i_latent])
    K = data.cca_input.zdims[i_latent]
    plt.figure(figsize=[12,3.5])
    for tt in range(K):
        plt.subplot(1,K,tt+1)
        p=plt.plot(mean_t[:, K0+tt],ls='--')
        plt.plot(data.ground_truth_latent[tr][:, i0+tt],color=p[0].get_color())