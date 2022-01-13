import numpy as np
from data_processing_tools import makeK_big

def allTrial_grad_expectedLLGPPrior(lam , meanPost, covPost, binSize,eps=0.001,Tmax=600,isGrad=False, trial_num=None):
    """
    Average over trial of the expected log-likelihood of the GP prior as a funciton of the time constant
    :param lam: transformed time constant such that K(t,s) = exp(-lam / 2 * |t-s|^2)
    :param data: object of the class gen_synthetic_data.dataGen
    :param meanPost: list, one element for each trial. meanPost[tr] is a K x T tensor containing the posterior mean
    :param covPost: list, one element for each trial. meanPost[tr] is a K x T x T tensor containing a slice the posterior
     covariance for each laetnt dimension. Do not need the full posterior since the RBF kernel factorizes over
     coordinates.
    :param binSize: time sampling rate in ms
    :param eps: noise parameter
    :param Tmax: max duration of a trial
    :param isGrad: True return the gradient, False returns the funcion evaluation
    :return:
    """
    #trial_num = len(data.trialDur)
    if trial_num is None:
        trial_num = len(meanPost)
    xDim = len(lam)

    idx_max = np.arange(0, (xDim) * (Tmax ), xDim)
    covX_max = np.zeros((Tmax * xDim, Tmax * xDim))
    muX_max = np.zeros((Tmax * xDim, 1))

    if not isGrad:
        f = 0
    else:
        f = np.zeros(xDim, )
    for i in range(len(meanPost)):
        T = meanPost[i].shape[1]
        idx = idx_max[:T]
        covX = covX_max[:T* xDim, :T * xDim]*0.
        # print(covX.shape,covPost[0].shape,idx_max.shape,T)
        muX = muX_max[:T * xDim]*0.
        for h in range(xDim):
            xx,yy= np.meshgrid(idx+h,idx+h)
            covX[xx,yy] = covPost[i][h, :,:]
            muX[idx + h] = meanPost[i][h,:].reshape((-1,1))
        # muX = muX.reshape(-1,1)
        f_tr = grad_expectedLLGPPrior(lam , eps, covX, muX, binSize,T,isGrad=isGrad)
        f = f + (f_tr / trial_num)
    return f

def grad_expectedLLGPPrior(lam , eps, covX, muX, binSize, T,isGrad=False):
    """
    Computes the expected log-likelihood of the GP prior as a funciton of the time constant
    :param lam_0: transformed time constant such that K(t,s) = exp(-lam / 2 * |t-s|^2)
    :param eps: noise parameter
    :param covX: list, each element is a trial
        contains the covariance in KT x KT formats (do not need the full posterior covariance since
    the RBF covariance factorizes across latent dimensions
    :param muX: list, each element is a trial
        contains the posterior mean KT x 1
    :param binSize: time sampling rate in ms
    :param T: number of time points of a trial
    :param isGrad: True: return gradient, False: return the function evaluation
    :return:
    """
    term1, term2 = compGrad_expectedLLGPPrior(lam , eps, covX, muX, binSize, T,isGrad=isGrad)

    f = -0.5 * (term1 + term2)

    return f


def compGrad_expectedLLGPPrior(lam_0, eps, covX, muX, binSize, T,isGrad=False):
    """
    Computes the two components of expected log-likelihood of the GP prior as a funciton of the time constant or
     its gradient. The first component is the -E[z\tr K z] and the second is -ln|K|
    :param lam_0: transformed time constant such that K(t,s) = exp(-lam / 2 * |t-s|^2)
    :param eps: noise parameter
    :param covX: list, each element is a trial
        contains the covariance in KT x KT formats (do not need the full posterior covariance since
    the RBF covariance factorizes across latent dimensions
    :param muX: list, each element is a trial
        contains the posterior mean KT x 1
    :param binSize: time sampling rate in ms
    :param T: number of time points of a trial
    :param isGrad: True: return gradient, False: return the function evaluation
    :return: 
    """
    if np.isscalar(eps):
        eps = np.array([eps]*len(lam_0))

    tau = np.exp(lam_0/2)*1000/binSize
    xDim = len(lam_0)
    _, _, K_big_inv, logDetK = makeK_big(xDim, tau, T*binSize, binSize, epsNoise=0.001, T=None, computeInv=True)

    K0 = len(tau)
    idx_rot = np.tile(np.arange(0, K0 * T, T), T) + np.arange(K0 * T) // K0
    K_big_inv = K_big_inv[idx_rot, :][:, idx_rot]
    mumuT = np.dot(muX, muX.T)
    dif = covX + mumuT
    if not isGrad:
        xtKinvx = np.dot(K_big_inv.flatten(), dif.flatten()) # trace(K_big_inv * covX); works since covX is sym, otherwise transpose
        return logDetK, xtKinvx
    else:
        # get the derivative of xT * Kinv * x in dK
        M = - np.dot(np.dot(K_big_inv, dif), K_big_inv)

        idx = np.arange(0, xDim * (T), xDim)

        Tvec = np.arange(T)#*(binSize/1000.)
        # compute dK_i / d\lam_i
        dKi_dLami = dK_dlamba_RBF(lam_0, Tvec, eps,binSize)

        # compute d \log | K | / dK_i
        dLogDetK_dLami = np.zeros((xDim, ))
        grad_xTKinvx = np.zeros((xDim, ))
        for i in range(xDim):
            xx,yy = np.meshgrid(i + idx, i + idx)
            KInvSub = K_big_inv[xx,yy] # T x T
            dLogDetK_dLami[i] = np.sum(KInvSub * np.squeeze(dKi_dLami[i,:,:]))
            grad_xTKinvx[i] = np.sum(M[xx,yy] * np.squeeze(dKi_dLami[i,:,:]))

        return dLogDetK_dLami, grad_xTKinvx

def dK_dlamba_RBF(lam_0, Tvec, eps, binSize):
    """
    Compute the derivative of the RBF covariance wrt the time constant
    :param lam_0: transformed time constant such that K(t,s) = exp(-lam / 2 * |t-s|^2)
    :param Tvec: vector of time points
    :param eps: noise parameter
    :param binSize: spacing between consecutive time points in MS
    :return: 
    """
    if np.isscalar(eps):
        eps = np.ones(len(lam_0))*eps
    xDim = len(lam_0)
    T = len(Tvec)
    tau = np.exp(lam_0 / 2) * 1000 / binSize
    Tdif = (np.repeat(Tvec,T).reshape(T,T) - np.repeat(Tvec.reshape(len(Tvec),1), T).reshape(T,T).T)**2
    dK = np.zeros((xDim, T, T))
    for k in range(xDim):
        # dK[k,:,:] = -(1 - eps[k]) * 0.5 * np.exp(lam_0[k]) * Tdif * np.exp(-np.exp(lam_0[k]) * Tdif * 0.5)
        dK[k,:,:] = (1 - eps[k]) * 0.5 * (binSize**2 / ((tau[k] * 1000) ** 2)) * Tdif * np.exp(-(binSize**2 / ((tau[k] * 1000) ** 2)) * Tdif * 0.5)
    return dK


def all_trial_GPLL(lam, data, idx_latent, block_trials=None, isGrad=False):
    if block_trials is None:
        block_trials = len(data.trialDur.keys())
    all_trials = list(data.trialDur.keys())
    trial_num = len(all_trials)

    trial_list = []
    nBlocks = int(np.ceil(len(all_trials) /block_trials))
    for k in range(nBlocks):
        trial_list.append(all_trials[k*block_trials:(k+1)*block_trials])
    f = 0
    for tl in trial_list:
        mu_list = []
        cov_list = []
        Tmax = 0
        for tr in tl:
            Tmax = max(Tmax,data.posterior_inf[tr].mean[idx_latent].shape[1])
            mu_list.append(data.posterior_inf[tr].mean[idx_latent])
            cov_list.append(data.posterior_inf[tr].cov_k[idx_latent])
            try:
                f = f + allTrial_grad_expectedLLGPPrior(lam , mu_list, cov_list, data.binSize,
                                                eps=data.epsNoise,Tmax=Tmax,isGrad=isGrad, trial_num=trial_num)
            except np.linalg.LinAlgError:
                if isGrad:
                    return -np.ones(lam.shape)*np.inf
                else:
                    return -np.inf

    # if isGrad:
    #     print( np.linalg.norm(f))
    return f


if __name__=='__main__':
    from gen_synthetic_data import *
    import matplotlib.pylab as plt
    from scipy.optimize import minimize

    # dat = dataGen(150,T=50)
    # eps = dat.cca_input.epsNoise
    # lam_0 = 2*np.log(
    #     ((dat.cca_input.priorPar[1]['tau']*dat.cca_input.binSize/1000))
    #                )
    #
    # mu_list = []
    # cov_list = []
    # for k in range(len(dat.cca_input.trialDur)):
    #     m,s = parse_fullCov_latDim(dat.cca_input, dat.meanPost[k], dat.covPost[k], dat.cca_input.trialDur[k])
    #     mu_list.append(m[1])
    #     cov_list.append(s[1])
    #
    #
    # func = lambda lam_0: -allTrial_grad_expectedLLGPPrior(lam_0, mu_list, cov_list, dat.cca_input.binSize,eps,
    #                                               max(dat.cca_input.trialDur), isGrad=False)
    # func_grad = lambda lam_0: -allTrial_grad_expectedLLGPPrior(lam_0, mu_list, cov_list, dat.cca_input.binSize,eps,
    #                                               max(dat.cca_input.trialDur)+1, isGrad=True)
    # res = minimize(func, np.zeros(len(lam_0)), jac=func_grad, method='L-BFGS-B',tol=10**-10)
    #
    #
    # tau_from_lam = lambda lam: np.exp(lam/2)*1000/dat.cca_input.binSize
    idx_latent = 1
    data = np.load('/Users/edoardo/Work/Code/P-GPCCA/inference_syntetic_data/sim_150Trials.npy',allow_pickle=True).all()
    np.random.rand(4)
    tau = np.random.uniform(0.2,1.2,size=data.priorPar[idx_latent]['tau'].shape[0])
    tau=np.array([0.61061895, 0.2510141, 0.57811751, 0.85722733, 0.5862096])
    lam0 = 2 * np.log(((tau * data.binSize/1000)))

    f = all_trial_GPLL(lam0, data, idx_latent, block_trials=40, isGrad=False)
    g = all_trial_GPLL(lam0, data, idx_latent, block_trials=40, isGrad=True)
    func = lambda lam0: -all_trial_GPLL(lam0, data, idx_latent, block_trials=1, isGrad=False)
    # ap_grad = -approx_grad(lam0,len(lam0),func,10**-5)
    gr_func = lambda lam0: -all_trial_GPLL(lam0, data, idx_latent, block_trials=1, isGrad=True)
    # res = minimize(func,lam0,jac=gr_func,method='L-BFGS-B')
    # tau_new = np.exp(res.x/2)*1000/data.binSize
