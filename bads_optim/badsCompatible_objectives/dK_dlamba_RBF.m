function [dK] = dK_dlamba_RBF(lam_0, Tvec, eps)
    xDim = length(lam_0);
    T = length(Tvec);
    Tdif = (repmat(Tvec', 1, T) - repmat(Tvec, T, 1)).^2; % Tdif_ij = (t_i - t_j)^2
    dK = zeros(xDim, T, T);
    for k = 1:xDim
        dK(k,:,:) = -(1 - eps{k}) * 0.5 * exp(lam_0(k)) * (Tdif) .* exp(-exp(lam_0(k)) * Tdif * 0.5);
    end
end