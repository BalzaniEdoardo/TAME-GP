function [Copt,C1opt,dopt,feval] = bads_fitPoisson(C,C1,d,x,mean_post,cov_post, iXvar)
    

%     save('/Users/edoardo/Work/Code/bads/myObjectiveFun/inputPoiss.mat',...
%         'C','C1','d','x','mean_post','cov_post','iXvar');
sizeC = size(C);
sizeC1 = size(C1);
ydim = sizeC(1);
z0dim = sizeC(2);
z1dim = sizeC1(2);
lb = repmat(-15.,1, z0dim+z1dim+1);
ub = repmat(15.,1, z0dim+z1dim+1);
plb = repmat(-10.,1, z0dim+z1dim+1);
pub = repmat(10.,1, z0dim+z1dim+1);
Copt = zeros(size(C));
C1opt = zeros(size(C1));
dopt = zeros(size(d));
feval = 0;
for yd = 1:ydim
    fprintf('\n\nOPTIMIZING COORD %d\n', yd)
    func = @(pars)expecedLLPoisson_allTr(pars, yd, z0dim, z1dim, x, mean_post,cov_post, iXvar);
    x0 = [C(yd,:), C1(yd,:), d(yd)];
    [optPar, feval_yd] = bads(func,x0,lb,ub,plb,pub);
    Copt(yd,:) = reshape(optPar(1,1:z0dim),1, z0dim);
    C1opt(yd,:) = reshape(optPar(1,z0dim+1:(z0dim+z1dim)),1, z1dim);
    dopt(:,yd) = optPar(1,z0dim+z1dim+1);
    feval = feval + feval_yd;
end
%     func = @(x)allTrial_loglike_RBF_wrt(x, params, seq, Tmax);
%     lb = repmat(-15.,1, length(lambda));
%     ub = repmat(15.5,1 ,length(lambda));
%     plb = repmat(-10., 1,length(lambda));
%     pub = repmat(2.15, 1, length(lambda));
%     size(lambda),length(lambda),length(pub),length(plb)
%     [x, feval] = bads(func,lambda,lb,ub,plb,pub);

return