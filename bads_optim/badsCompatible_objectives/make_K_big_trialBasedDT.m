function [K_big, K_big_inv, logdet_K_big] = make_K_big_trialBasedDT(params, T_samp)
%
% [K_big, K_big_inv] = make_K_big(params, T)
%
% Constructs full GP covariance matrix across all state dimensions and timesteps.
%
% INPUTS:
%
% params       - GPFA model parameters
% T            - (1 x sampling tp) vector of timesteps
%
% OUTPUTS:
%
% K_big        - GP covariance matrix with dimensions (xDim * T) x (xDim * T).
%                The (t1, t2) block is diagonal, has dimensions xDim x xDim, and 
%                represents the covariance between the state vectors at
%                timesteps t1 and t2.  K_big is sparse and striped.
% K_big_inv    - inverse of K_big
% logdet_K_big - log determinant of K_big
%
% @ 2009 Byron Yu         byronyu@stanford.edu
%        John Cunningham  jcunnin@stanford.edu
  T = length(T_samp);
  
  xDim = length(params.eps);

  idx = 0 : xDim : (xDim*(T-1));
    
  K_big        = zeros(xDim*T);
  K_big_inv    = zeros(xDim*T);
  Tdif         = repmat(T_samp', 1, T) - repmat(T_samp, T, 1);
  logdet_K_big = 0;

  for i = 1:xDim
    
    K = (1 - params.eps{i}) * ...
        exp(-params.gamma(i) / 2 * Tdif.^2) +...
        params.eps{i} * eye(T);
    
    
    K_big(idx+i, idx+i) = K;
    L = chol(K);
    Linv = L \ eye(size(L,1));
    Kinv = Linv*Linv';
    logdet_K = 2*sum(log(diag(L)));
    K_big_inv(idx+i, idx+i) = Kinv;%invToeplitz(K);
    
    % check out the K_big determinant
    logdet_K_big = logdet_K_big + logdet_K;
    
  end  
end