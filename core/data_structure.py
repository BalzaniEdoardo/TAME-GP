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
            T += self.preproc.data['Y'].shape[1]

        # get the observation dim
        stimDim = len(self.var_list)

        # get means of stim and spike counts
        stimMean = np.zeros(stimDim)
        cc = 0
        for var in self.var_list:
            for tr in range(self.preproc.numTrials):
                stimMean[cc] = (stimMean[cc] + np.nanmean(self.preproc.covariates[var][tr])/self.preproc.numTrials)
            cc += 1

        xDims = []
        xLogMeans = []
        for area in self.area_list:
            sel = self.filter_unit * (self.unit_area == area)
            xDims.append(sel.sum())
            xMeans_area = np.zeros(xDims[-1])
            for tr in range(self.preproc.numTrials):
                xMeans_area = (xMeans_area + np.nanmean(self.preproc.data[tr]['Y'][sel],axis=1) /self.preproc.numTrials)
            xLogMeans.append(np.log(xMeans_area))

        # zdims are the dimensinons of the latent variables
        priorPar = {}
        stimPar = {
            'W0': 0.01 * np.random.normal(size=(stimDim, zdims[0])),
            'd' : stimMean
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
        stim = np.zeros((counts.shape[1], len(self.var_list)))

        # create the stim var
        cc = 0
        for var in self.area_list:
            stim[:, cc] = taskVars[var][trNum]
            cc += 1

        keep_idx = ~np.isnan(stim.sum(axis=1))
        stim = stim[keep_idx]

        # create the list
        xList = []
        for area in self.area_list:
            sel = self.filter_unit * (self.unit_area == area)
            xList.append(counts[sel, keep_idx].T)

        return stim, xList