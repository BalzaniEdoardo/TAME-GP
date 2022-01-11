function [f, g] = log_like_RBF_wrt_logTau(lambda, eps, Tvec, covX, muX)
    [dLogDetK_dLami, grad_xTKinvx, logDetK, xtKinvx] = grad_log_det_K_RBF(lambda, eps, Tvec, covX, muX);
    
    f = 0.5 * (logDetK + xtKinvx);
    g = 0.5 * (dLogDetK_dLami + grad_xTKinvx);
    
end