function [f,g] = allTrial_loglike_RBF_wrt(lambda, params, seq, Tmax)
    eps = params.eps;
    trial_num = length(seq.T);
    % this suppose same trial duration
    
    xDim = length(eps);
    
    
    idx_max = 0: xDim: (xDim)*(Tmax-1);
    covX_max = zeros(Tmax*xDim, Tmax*xDim);
    muX_max = zeros(Tmax*xDim, 1);
    
    f = 0;
    g = zeros(xDim,1);
    for i=1:trial_num
    %parfor i=1:trial_num
        Tvec = seq.T{i};
        T = length(Tvec);
        idx = idx_max(1:T);
        covX = covX_max(1:T*xDim, 1:T*xDim);
        muX = muX_max(1:T*xDim);
        for h = 1:xDim
            covX(idx+h,idx+h) = squeeze(seq.VsmGP{i}(:,:,h));
            muX(idx+h,1) = seq.xsm{i}(h,:);
        end
        [f_tr, g_tr] = log_like_RBF_wrt_logTau(lambda, eps, Tvec, covX, muX);
        %[i-1, f_tr]
        f = f + (f_tr/trial_num);
        g = g + (g_tr/trial_num);
         
    end
   
end