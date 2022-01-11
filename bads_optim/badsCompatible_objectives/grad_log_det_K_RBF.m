function [dLogDetK_dLami, grad_xTKinvx, logDetK, xtKinvx] = grad_log_det_K_RBF(lam_0, eps, Tvec, covX, muX)
    % This function should compute the gradient of log|K| as a function
    % of the time constant
    params.eps = eps;
    
    params.gamma = exp(lam_0);
    params.covType = 'rbf';
    
    % get the K inverse
    [~, K_big_inv, logDetK] = make_K_big_trialBasedDT(params, Tvec);
    mumuT = muX * muX';
    dif = covX + mumuT;
    xtKinvx = K_big_inv(:)' * (dif(:)); %trace(K_big_inv * covX); woorks since covX is sym, otherwise transpose
    
    % get the derivative of xT * Kinv * x  in dK
    M = - K_big_inv * (dif) * K_big_inv;
    
    xDim = length(eps);
    T = length(Tvec);
    idx = 0 : xDim : (xDim*(T-1));
    
    % compute dK_i/d\lam_i
    dKi_dLami = dK_dlamba_RBF(lam_0, Tvec, eps);
    
    % compute d \log |K| / dK_i 
    
    dLogDetK_dLami = zeros(xDim,1);
    grad_xTKinvx = zeros(xDim,1);
    for i = 1:xDim
        KInvSub = K_big_inv(i+idx, i+idx); % T x T
        dLogDetK_dLami(i) = sum(sum(KInvSub .* squeeze(dKi_dLami(i,:,:))));
        grad_xTKinvx(i) = sum(sum(M(i+idx, i+idx) .* squeeze(dKi_dLami(i,:,:))));
    end
end

