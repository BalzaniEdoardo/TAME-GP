"""
Implement a class that handles the input dataset conveniently.
The class needs to store spikes and task variables, initialize parameters and select appropriately the data for the fits.
"""
import numpy as np

class GP_pCCA_input(object):
    def __init__(self, preProc, var_list, area_list, unit_area, filter_unit):
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

    def initializeParam(self, zdims):
        """
        Naive params initialization using random projection weights.
        :param zdims: list of the latent dimensions

        """
        # get total tp
        T = 0
        for tr in range(self.preproc.numTrials):
            T += self.preproc.data[tr]['Y'].shape[1]

        # get the observation dim
        stimDim = len(self.var_list)

        # get means of stim and spike counts
        stimMean = np.zeros(stimDim)
        stimCov = np.zeros((stimDim,stimDim))

        for tr in range(self.preproc.numTrials):
            cc = 0
            trStim = np.zeros((stimDim, self.preproc.covariates[self.var_list[0]][tr].shape[0]))
            for var in self.var_list:
                stimMean[cc] = (stimMean[cc] + np.nanmean(self.preproc.covariates[var][tr])/self.preproc.numTrials)
                trStim[cc] = self.preproc.covariates[var][tr]
                cc += 1
            stimCov = stimCov + np.cov(trStim)/self.preproc.numTrials

        xDims = []
        xLogMeans = []
        for area in self.area_list:
            sel = self.filter_unit * (self.unit_area == area)
            xDims.append(sel.sum())
            xMeans_area = np.zeros(xDims[-1])
            for tr in range(self.preproc.numTrials):
                xMeans_area = (xMeans_area + np.nanmean(self.preproc.data[tr]['Y'][:,sel],axis=0) /self.preproc.numTrials)
            xLogMeans.append(np.log(xMeans_area))

        # zdims are the dimensinons of the latent variables
        priorPar = []
        for kk in zdims:
            priorPar.append({'tau': np.random.rand(kk)*0.5})

        stimPar = {
            'W0': 0.01 * np.random.normal(size=(stimDim, zdims[0])),
            'd' : stimMean,
            'PsiInv': np.linalg.pinv(stimCov + np.eye(stimDim)*0.001)
        }

        xParams = []
        for kk in range(len(xDims)):
            pars = {
                'W0': 0.01 * np.random.normal(size=(xDims[kk], zdims[0])),
                'W1': 0.01 * np.random.normal(size=(xDims[kk], zdims[kk+1])),
                'd' : xLogMeans[kk]
            }
            xParams.append(pars)
        self.priorPar = priorPar
        self.stimPar = stimPar
        self.xPar = xParams

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

if __name__ == '__main__':
    import sys,os,inspect
    basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
    sys.path.append(os.path.join(basedir, 'firefly_utils'))
    sys.path.append(os.path.join(basedir, 'core'))
    from behav_class import emptyStruct
    from inference import makeK_big
    from scipy.linalg import block_diag
    import seaborn as sbn
    preproc = emptyStruct()
    preproc.numTrials = 1
    preproc.ydim = 50
    preproc.binSize = 50

    preproc.T = np.array([100])

    tau = np.array([0.9,0.2,0.4])
    K0 = 3
    epsNoise=0.000001
    K_big = makeK_big(K0, tau, None, preproc.binSize, epsNoise=epsNoise, T=preproc.T[0], computeInv=False)[1]
    z = np.random.multivariate_normal(mean=np.zeros(K0*preproc.T[0]),cov=K_big,size=1).reshape(preproc.T[0],K0)

    # create the stim vars
    PsiInv = np.eye(2)
    W = np.random.normal(size=(2, K0))
    d = np.zeros(2)
    preproc.covariates = {}
    preproc.covariates['var1'] = [np.random.multivariate_normal(mean=np.dot(W,z.T)[0],cov=np.eye(preproc.T[0]))]
    preproc.covariates['var2'] = [np.random.multivariate_normal(mean=np.dot(W,z.T)[1],cov=np.eye(preproc.T[0]))]


    # create the counts
    tau = np.array([1.1])
    K_big = makeK_big(1, tau, None, preproc.binSize, epsNoise=epsNoise, T=preproc.T[0], computeInv=False)[1]
    z1 = np.random.multivariate_normal(mean=np.zeros(preproc.T[0]),cov=K_big,size=1).reshape(preproc.T[0],1)

    W1 = np.random.normal(size=(preproc.ydim, 1))
    W0 = np.random.normal(size=(preproc.ydim, 1))
    d = -0.2
    preproc.data = [{'Y': np.random.poisson(lam=np.exp(np.einsum('ij,tj->ti', W0, z) + np.einsum('ij,tj->ti', W1, z1) + d))}]


    # create the data struct
    struc = GP_pCCA_input(preproc,['var1','var2'],['PPC'],np.array(['PPC']*preproc.ydim),np.ones(preproc.ydim,dtype=bool))
    struc.initializeParam([2,1])