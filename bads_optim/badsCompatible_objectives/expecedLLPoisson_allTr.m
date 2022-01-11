function [f] = expecedLLPoisson_allTr(pars, yd, z0dim, z1dim, x, mean_post,cov_post, idxVar)
C = reshape(pars(1,1:z0dim),1, z0dim);
C1 = reshape(pars(1,z0dim+1:(z0dim+z1dim)),1, z1dim);
d = pars(1,z0dim+z1dim+1);
f = 0;    
for tr = 1:length(cov_post)
    f = f - expectedLLPoisson(double(x{1,idxVar}{1,tr}), yd, C, d, mean_post{1,tr},cov_post{1,tr}, C1);
end
end