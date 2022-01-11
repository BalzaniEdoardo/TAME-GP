function [x,feval] = bads_fitGP(lambda, params, seq, Tmax)
    save('/Users/edoardo/Work/Code/bads/myObjectiveFun/input.mat','lambda','params','seq','Tmax');
    func = @(x)allTrial_loglike_RBF_wrt(x, params, seq, Tmax);
    lb = repmat(-15.,1, length(lambda));
    ub = repmat(15.5,1 ,length(lambda));
    plb = repmat(-10., 1,length(lambda));
    pub = repmat(2.15, 1, length(lambda));
    size(lambda),length(lambda),length(pub),length(plb)
    [x, feval] = bads(func,lambda,lb,ub,plb,pub);
return