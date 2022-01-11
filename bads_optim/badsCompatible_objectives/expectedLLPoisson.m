function [f] = expectedLLPoisson(x, yd, C, d, mean_post,cov_post, C1)
    
    %The Spike count observation expected likelihood
    sizeC = size(C);
    xdim = sizeC(2);
    mean0 = mean_post(:, 1:xdim);
    T = size(cov_post);
    T = T(1);
    if ~isempty(C1)
        xdim1 = size(C1);
        xdim1 = xdim1(2);
        mean1 = mean_post(:, xdim+1:end);
    else
        xdim1 = 0;
    end
    if ~isempty(C1)
        Cout = [C(1,:) C1(1,:)];
        CC = Cout(:) * Cout(:).'; %np.hstack((C[yd],C1[yd]))
        
    else
        Cout = C(1,:);
        CC = Cout(:) * Cout(:).';
    end
    yhat = sum( exp(0.5 * ...
           sum(reshape(cov_post, T, (xdim+xdim1)^2) * reshape(CC, (xdim+xdim1)^2,1),2)+...
           d(1) + einsum( Cout, mean_post, 'ij,tj->ti')));

    if ~isempty(C1)
        tmp1 = einsum(x(:,yd),mean1, 'tj,tk->jk');
        tmp0 =  einsum(x(:,yd),mean0, 'tj,tk->jk');
        hh = einsum(tmp1, C1(1,:), 'jk,jk->') +...
             einsum(tmp0, C(1,:), 'jk,jk->') +...
             sum(x(:,yd),1) * d(1);
    else
        tmp0 = einsum(x(:,yd),mean0, 'tj,tk->jk');
        hh = einsum(tmp0, C(1,:), 'jk,jk->') + sum(x(:,yd),1) * d(1);% np.einsum('tj,jk,tk->', x, C, mean0, optimize=True) + np.dot(x.sum(axis=0),d)
    end
    f = hh - yhat;
end