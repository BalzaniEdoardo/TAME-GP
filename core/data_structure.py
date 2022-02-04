"""
Implement a class that handles the input dataset conveniently.
The class needs to store spikes and task variables, initialize parameters and select appropriately the data for the fits.
"""
import numpy as np
from data_processing_tools import emptyStruct
from copy import deepcopy

class P_GPCCA(object):
    def __init__(self, preProc, var_list, area_list, unit_area, filter_unit, binSize=50, epsNoise=0.001,
                 transposeY = False):
        """
        :param preProc: structure with attributes:
            * numTrials: int, number of trials
            * ydim: dimension of the simulatenously recorded population
            * data: spike counts, list with one element per trial
                * data[tr]['Y']: N x T spike count matrix for a trial (T varies)
            * binSize: temporal sampling in ms (usually 50ms)
            * T: list of trial durations in # of time points
            * covariates: dictionary containing the task variables
                * covariates['varname']: list, each element is a trial
                    *  covariates['varname'][trialNum]: array of length T, the time course of the variable

            * trialId: list of trial identifiers, used to pair from the fits trial with the original recordings
        :param var_list: list of variables that are included as stimulus in the regression
        :param area_list: list of simultaneously recorded brain regions to be added to the model
        :param unit_area: array of length ydim, that with the brain area identifier of the unit
        :param filter_unit: array of boolean of length ydim, which units to include in the fit
        """
        # the spike counts and
        self.preproc = preProc
        self.filter_unit = filter_unit
        self.unit_area = unit_area
        self.area_list = area_list
        self.var_list = var_list
        self.binSize = binSize
        self.epsNoise = epsNoise


        if not type(self.preproc.data) is dict:
            # create a data dictionary with trial information
            self.trialDur = {}  # np.zeros(self.preproc.numTrials, dtype=int)
            data = {}
            for tr in range(self.preproc.numTrials):
                if not transposeY:
                    self.trialDur[tr] = self.preproc.data[tr]['Y'].shape[0]
                    data[tr] = self.preproc.data[tr]
                else:
                    self.trialDur[tr] = self.preproc.data[tr]['Y'].shape[1]
                    tmp = self.preproc.data[tr]['Y'].T
                    data[tr] = {'Y':tmp}

            cov = {}
            for var in self.preproc.covariates.keys():
                cov[var] = {}
                for tr in range(self.preproc.numTrials):
                    cov[var][tr] = self.preproc.covariates[var][tr]
            self.preproc.data = data
            self.preproc.T = deepcopy(self.trialDur)
            self.preproc.covariates = cov
        else:
            self.trialDur = {}
            for tr in self.preproc.data.keys():
                self.trialDur[tr] = self.preproc.data[tr]['Y'].shape[0]


    def initializeParam(self, zdims, use_poissonPCA=False):
        """
        Naive params initialization using random projection weights.
        :param zdims: list of the latent dimensions

        """
        assert((len(zdims)-1) == len(self.area_list))
        # get the observation dim
        stimDim = len(self.var_list)

        # get means of stim and spike counts
        stimMean = np.zeros(stimDim)
        stimCov = np.zeros((stimDim,stimDim))

        for tr in self.preproc.data.keys():
            cc = 0
            trStim = np.zeros((stimDim, self.preproc.covariates[self.var_list[0]][tr].shape[0]))
            for var in self.var_list:
                stimMean[cc] = (stimMean[cc] + np.nanmean(self.preproc.covariates[var][tr])/self.preproc.numTrials)
                trStim[cc] = self.preproc.covariates[var][tr]
                cc += 1
            stimCov = stimCov + np.cov(trStim)/self.preproc.numTrials

        # extract all spikes
        spikes = np.zeros([self.preproc.ydim, np.sum(list(self.trialDur.values()))])
        stim_all = np.zeros([len(self.var_list), np.sum(list(self.trialDur.values()))])
        cc = 0
        for tr in self.trialDur.keys():
            T = self.preproc.data[tr]['Y'].shape[0]
            spikes[:, cc:cc + T] = self.preproc.data[tr]['Y'].T
            var_cc = 0
            for var in self.var_list:
                stim_all[var_cc, cc:cc + T] = self.preproc.covariates[var][tr]
            cc += T

        xDims = []
        xLogMeans = []
        W1_list = []
        W0_list = []
        cc = 1
        for area in self.area_list:
            sel = self.filter_unit * (self.unit_area == area)
            xDims.append(sel.sum())
            xMeans_area = np.mean(spikes[sel],1) + 1e-10
            xLogMeans.append(np.log(xMeans_area))
            if use_poissonPCA:
                covY = np.cov(spikes[sel])

                # moment conversion between Poisson & Gaussian with exponential nonlinearity (taken from K. Machens code)
                lamb = np.log(np.abs(covY + np.outer(xMeans_area, xMeans_area) - np.diag(xMeans_area))) - np.log(np.outer(xMeans_area, xMeans_area))
                # PCA
                evals, evecs = np.linalg.eig(lamb)
                idx = np.argsort(evals)[::-1]
                evecs = evecs[:, idx]
                # sort eigenvectors according to same index
                evals = evals[idx]
                # select the first xdim eigenvectors
                evecs = evecs[:, :zdims[cc]]
                W1_list.append(evecs)

                # repeat for CCA
                Y = np.vstack((spikes[sel], stim_all))
                covY = np.cov(Y)
                ee,U = np.linalg.eigh(covY[:sel.sum(),:sel.sum()])
                sqrtY11  = np.dot(np.dot(U, np.diag(np.sqrt(ee))), U.T)
                ee2, U2 = np.linalg.eigh(covY[sel.sum():, sel.sum():])
                sqrtY22 = np.dot(np.dot(U2, np.diag(np.sqrt(ee2))), U2.T)
                M = np.dot(np.dot(sqrtY11, covY[:sel.sum(),sel.sum():]),sqrtY22)
                V1,d,V2 = np.linalg.svd(M)
                V1 = np.dot(sqrtY11, V1)
                idx = np.argsort(d)
                V1 = V1[:, idx]
                W0_list.append(np.log(np.abs(V1[:,:zdims[0]])))

            else:
                W1_list.append(0.01 * np.random.normal(size=(xDims[-1], zdims[cc])))
                W0_list.append(0.01 * np.random.normal(size=(xDims[-1], zdims[0])))
            cc += 1

        # zdims are the dimensinons of the latent variables
        priorPar = []
        for kk in zdims:
            priorPar.append({'tau': np.random.rand(kk)*0.5})

        stimPar = {
            'W0': 0.01 * np.random.normal(size=(stimDim, zdims[0])),
            'd' : stimMean,
            'PsiInv': np.linalg.pinv(stimCov + np.eye(stimDim)*0.0001)
        }

        xParams = []
        for kk in range(len(xDims)):
            pars = {
                'W0': W0_list[kk],
                'W1': W1_list[kk],
                'd' : xLogMeans[kk]
            }
            xParams.append(pars)
        self.priorPar = priorPar
        self.stimPar = stimPar
        self.xPar = xParams
        self.zdims = np.array(zdims)

        # self.initPar = emptystruct()
        return

    def get_observations(self, trNum):
        """
        This function extracts the observed variables for a specific trial
        :param trNum: int, trial number
        :return:
            * stim: T x m0 task variable time course for trial trNum
            * xList: list of spike counts by brain region
                * xList[i]: T x Ni spike count matrix for trial trNum and brain area self.area_list[i]
        """
        counts = self.preproc.data[trNum]['Y']
        taskVars = self.preproc.covariates
        stim = np.zeros((counts.shape[0], len(self.var_list)))

        # create the stim var
        cc = 0
        for var in self.var_list:
            stim[:, cc] = taskVars[var][trNum]
            cc += 1

        keep_idx = ~np.isnan(stim.sum(axis=1))
        stim = stim[keep_idx]

        # create the list
        xList = []
        for area in self.area_list:
            sel = self.filter_unit * (self.unit_area == area)
            xList.append(counts[keep_idx][:,sel])

        return stim, xList

    def subSampleTrial(self, trialVec):
        subStruct = deepcopy(self)
        unwanted = set(self.trialDur.keys()) - set(trialVec)

        for unwanted_key in unwanted: del subStruct.trialDur[unwanted_key]
        for unwanted_key in unwanted: del subStruct.preproc.data[unwanted_key]
        for unwanted_key in unwanted: del subStruct.preproc.T[unwanted_key]
        if 'posterior_inf' in self.__dict__.keys():
            for unwanted_key in unwanted: del subStruct.posterior_inf[unwanted_key]
        for var in subStruct.preproc.covariates.keys():
            for key in unwanted:
                subStruct.preproc.covariates[var].pop(key)


        if 'ground_truth_latent' in self.__dict__.keys():
            idxTr = np.array(list(self.trialDur.keys()))
            keep = np.ones(idxTr.shape[0],dtype=bool)
            for tr in unwanted:
                keep[np.where(idxTr == tr)[0][0]] = False
            subStruct.ground_truth_latent = np.array(subStruct.ground_truth_latent,dtype=object)[keep]


        subStruct.preproc.numTrials = len(subStruct.trialDur.keys())
        return subStruct

    def genNewData(self, trNum, T):
        """
        Sample trNum new trials of duration T using model parameters
        :param trNum:
        :param T:
        :return:
        """
        from data_processing_tools import makeK_big
        if 'ground_truth_xPar' not in self.__dict__.keys():
            print('No ground truth parameters available!')
        K0 = self.zdims[0]
        tau0 = self.ground_truth_priorPar[0]['tau']
        binSize = self.binSize
        epsNoise = self.epsNoise
        K_big0 = makeK_big(K0, tau0, None, binSize, epsNoise=epsNoise, T=T, computeInv=False)[1]
        trial_0 = max(list(self.trialDur.keys()))+1
        for tr in range(trNum):
            z0 = np.random.multivariate_normal(mean=np.zeros(K0 * T), cov=K_big0, size=1).reshape(K0, T).T
            W0 = self.stimPar['W0']
            d = self.stimPar['d']
            Psi = np.linalg.inv(self.stimPar['PsiInv'])
            mu = np.einsum('ij,tj->ti', W0, z0) + d
            x1 = np.zeros((T,W0.shape[0]))
            for t in range(mu.shape[0]):
                x1[t] = np.random.multivariate_normal(mean=mu[t],cov=Psi)
            k=0
            for kk in self.var_list:
                self.preproc.covariates[kk][trial_0+tr] = x1[:, k]
                k+=1
            gtlatent = deepcopy(z0)
            Y = np.zeros((T, self.preproc.ydim))
            for k in range(1, len(self.zdims)):
                K = self.zdims[k]
                taui = self.priorPar[k]['tau']
                K_bigi = makeK_big(K, taui, None, binSize, epsNoise=epsNoise, T=T, computeInv=False)[1]
                zi = np.random.multivariate_normal(mean=np.zeros(K * T), cov=K_bigi, size=1).reshape(K, T).T

                W0 = self.xPar[k - 1]['W0']
                W1 = self.xPar[k - 1]['W1']
                d = self.xPar[k - 1]['d']
                xi = np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', W0, z0) +
                                                  np.einsum('ij,tj->ti', W1, zi) + d))
                sel = self.unit_area == self.area_list[k-1]
                Y[:, sel] = xi
                gtlatent = np.hstack((gtlatent,zi))
            self.preproc.data[trial_0+tr] = {'Y': Y}
            self.ground_truth_latent.append(gtlatent)
            self.trialDur[trial_0+tr] = T

        self.new_trial_list = np.arange(trial_0,trial_0+trNum)



if __name__ == '__main__':
    from inference import makeK_big
    preproc = emptyStruct()
    preproc.numTrials = 10
    preproc.ydim = 50
    preproc.binSize = 50

    preproc.T = np.array([100]*preproc.numTrials)

    preproc.covariates = {}
    preproc.covariates['var1'] = []
    preproc.covariates['var2'] = []
    preproc.data = []
    for k in range(preproc.numTrials):
        tau = np.array([0.9, 0.2, 0.4])
        K0 = 3
        epsNoise = 0.000001
        K_big = makeK_big(K0, tau, None, preproc.binSize, epsNoise=epsNoise, T=preproc.T[0], computeInv=False)[1]
        z = np.random.multivariate_normal(mean=np.zeros(K0*preproc.T[0]),cov=K_big,size=1).reshape(preproc.T[0],K0)

        # create the stim vars
        PsiInv = np.eye(2)
        W = np.random.normal(size=(2, K0))
        d = np.zeros(2)

        preproc.covariates['var1'] += [np.random.multivariate_normal(mean=np.dot(W,z.T)[0],cov=np.eye(preproc.T[0]))]
        preproc.covariates['var2'] += [np.random.multivariate_normal(mean=np.dot(W,z.T)[1],cov=np.eye(preproc.T[0]))]


        # create the counts
        tau = np.array([1.1])
        K_big = makeK_big(1, tau, None, preproc.binSize, epsNoise=epsNoise, T=preproc.T[0], computeInv=False)[1]
        z1 = np.random.multivariate_normal(mean=np.zeros(preproc.T[0]),cov=K_big,size=1).reshape(preproc.T[0],1)

        W1 = np.random.normal(size=(preproc.ydim, 1))
        W0 = np.random.normal(size=(preproc.ydim, 1))
        d = -0.2
        preproc.data += [{'Y': np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', W0, z) + np.einsum('ij,tj->ti', W1, z1) + d))}]


    # create the data struct
    struc = P_GPCCA(preproc,['var1','var2'],['PPC'],np.array(['PPC']*preproc.ydim),np.ones(preproc.ydim,dtype=bool))
    struc.initializeParam([2,1])
    a = struc.subSampleTrial(np.arange(2,5))
