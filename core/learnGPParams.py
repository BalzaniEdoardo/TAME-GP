import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from data_processing_tools import makeK_big



def allTrial_loglike_RBF_wrt(lam , params, infRes, binSize,eps=0.001,Tmax=600,isGrad=False):
    C = params['C']
    trial_num = len(infRes['post_mean'])

    xDim = C.shape[1]

    idx_max = np.arange(0, (xDim) * (Tmax - 1), xDim)
    covX_max = np.zeros((Tmax * xDim, Tmax * xDim))
    muX_max = np.zeros((Tmax * xDim, 1))

    if not isGrad:
        f = 0
    else:
        f = np.zeros(xDim, )
    for i in range(trial_num):
        T = infRes['post_mean'][i].shape[1]
        #T = length(Tvec);
        idx = idx_max[:T]
        covX = covX_max[:T* xDim, :T * xDim]*0.
        muX = muX_max[:T * xDim]*0.
        for h in range(xDim):
            xx,yy= np.meshgrid(idx+h,idx+h)
            covX[xx,yy] = infRes['post_vsmGP'][i][:,:,h]
            muX[idx + h,:] = infRes['post_mean'][i][h].reshape((-1,1))
        f_tr = log_like_RBF_wrt_logTau(lam , C, eps, covX, muX, binSize,T,isGrad=isGrad)
        f = f + (f_tr / trial_num)
    return f

def log_like_RBF_wrt_logTau(lam , eps, covX, muX, binSize, T,isGrad=False):
    # dLogDetK_dLami, grad_xTKinvx, logDetK, xtKinvx = grad_log_det_K_RBF(lam , C, eps, covX, muX, binSize, T,isGrad=isGrad)
    term1, term2 = grad_log_det_K_RBF(lam , eps, covX, muX, binSize, T,isGrad=isGrad)

    f = -0.5 * (term1 + term2)

    return f


def grad_log_det_K_RBF(lam_0, eps, covX, muX, binSize, T,isGrad=False):
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

def  make_K_big_Edo(params, T_samp):
    T = len(T_samp)

    xDim = params['C'].shape[1]

    idx = np.arange(0, xDim * T, xDim)
    K_big = np.zeros((xDim * T,)*2)
    K_big_inv = np.zeros((xDim * T,)*2)
    Tdif = (np.repeat(T_samp,T).reshape(T,T) - np.repeat(T_samp.reshape(len(T_samp),1), T).reshape(T,T).T) # Tdif_ij = (t_i - t_j)^2
    logdet_K_big = 0

    for i in range(xDim):
        if params['covType'] == 'rbf':
            K = (1 - params['eps'][i]) * np.exp(-params['gamma'][i] / 2 * Tdif ** 2) + params['eps'][i] * np.eye(T)
        elif params['covType']:
            K = np.max(1 - params['eps'][i] - params['a'][i] * np.abs(Tdif), axis=0) + params['eps'][i] * np.eye(T)
        elif params['covType']:
            z = np.dot(params['gamma'], (1 - params['eps'][i] - params['a'][i] * np.abs(Tdif)))
            outUL = (z > 36)
            outLL = (z < -19)
            inLim = (~outUL) & (~outLL)

            hz = np.zeros(z.shape) * np.nan
            hz[outUL] = z[outUL]
            hz[outLL] = np.exp(z[outLL])
            hz[inLim] = np.log(1 + np.exp(z[inLim]))

            K = np.linalg.solve(params['gamma'].reshape(1, -1), hz.T).T + params['eps'][i] * np.eye(T)
        xx, yy = np.meshgrid(idx + i, idx + i)
        K_big[xx,yy] = K
        chl = np.linalg.cholesky(K)
        Linv = np.linalg.solve(chl, np.eye(chl.shape[0]))
        Kinv = np.dot(Linv.T, Linv)
        logdet_K = 2 * np.sum(np.log(np.diag(chl)))
        K_big_inv[xx,yy] = Kinv  # invToeplitz(K);
        logdet_K_big = logdet_K_big + logdet_K

    return K_big, K_big_inv, logdet_K_big

if __name__=='__main__':
    from gen_synthetic_data import *
    import matplotlib.pylab as plt

    dat = dataGen(1,T=250)
    C = dat.cca_input.stimPar['W0']
    lam_0 = 2*np.log(
        ((dat.cca_input.priorPar[0]['tau']*dat.cca_input.binSize/1000))
                   )
    eps = dat.cca_input.epsNoise
    covX = dat.covPost[0][:len(lam_0)*dat.cca_input.trialDur[0],:len(lam_0)*dat.cca_input.trialDur[0]]
    muX = dat.meanPost[0][:len(lam_0)*dat.cca_input.trialDur[0]]
    # mnt,covt,_ = parse_fullCov(dat.cca_input, muX[0], covX[0], dat.cca_input.trialDur[0])

    # allTrial_loglike_RBF_wrt(lam, params, infRes, binSize, eps=0.001, Tmax=600, isGrad=False)
    func = lambda lam_0: -log_like_RBF_wrt_logTau(lam_0 , eps, covX, muX, dat.cca_input.binSize, dat.cca_input.trialDur[0],isGrad=False)
    func_grad = lambda lam_0: -log_like_RBF_wrt_logTau(lam_0 , eps, covX, muX, dat.cca_input.binSize, dat.cca_input.trialDur[0],isGrad=True)
    res = minimize(func,np.zeros(len(lam_0)),jac=func_grad,method='L-BFGS-B')

    print('old tau:',dat.cca_input.priorPar[0]['tau'], ' - LL', -func(lam_0))

    tau_neu = 1000/dat.cca_input.binSize * np.exp(res.x/2.)
    print('new tau:', tau_neu, ' - LL', -func(res.x))
    dat.cca_input.priorPar[0]['tau'] = tau_neu


    muX1, covX1 = inferTrial(dat.cca_input, 0)
    muX1  = muX1[:len(lam_0) * dat.cca_input.trialDur[0]]
    covX1 = covX1[:len(lam_0) * dat.cca_input.trialDur[0],:len(lam_0) * dat.cca_input.trialDur[0]]
    plt.figure()
    ax1 = plt.subplot(121)
    ax1.plot(muX[::2])
    ax1.plot(muX[1::2])
    ax1.plot(dat.ground_truth_latent[0][:,0],'--')
    ax1.plot(dat.ground_truth_latent[0][:,1],'--')

    ax2 = plt.subplot(122)
    ax2.plot(muX1[::2])
    ax2.plot(muX1[1::2])
    ax2.plot(dat.ground_truth_latent[0][:, 0], '--')
    ax2.plot(dat.ground_truth_latent[0][:, 1], '--')
    xx =1
