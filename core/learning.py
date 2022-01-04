"""
Some of the code here is adapted from Machens et al. implementation of P-GPFA.
"""
import numpy as np
from scipy.optimize import minimize

def fun(xi,Ci,di,mu, cov):
    return xi*di + xi * np.dot(Ci,mu) - np.exp(di + np.dot(Ci, mu) + 0.5*np.dot(Ci, np.dot(cov, Ci)))

def gradFun(xi,Ci,di,mu, cov):
    return xi - np.exp(di + np.dot(Ci, mu) + 0.5 * np.dot(Ci, np.dot(cov, Ci)))

def slow_expectedLLPoisson_coord(xi, Ci, di, mean_post,cov_post):
    # LL = 0
    lin2 = xi.sum(axis=0)*di + np.einsum('t,j,tj->',xi,Ci,mean_post)
    exp2 = np.exp(di + np.einsum('j,tj->t',Ci,mean_post) + 0.5*np.einsum('i,tij,j->t',Ci,cov_post,Ci)).sum()
    # for t in range(xi.shape[0]):
    #     LL += fun(xi[t],Ci,di,mean_post[t],cov_post[t])
    return lin2-exp2


def grad_expectedLLPoisson_coord(xi, Ci, di, mean_post, cov_post):
    grad_Ci = np.zeros(Ci.shape[0])
    grad_di = 0
    for t in range(xi.shape[0]):
        grad_di += gradFun(xi[t], Ci, di, mean_post[t], cov_post[t])#xi[t] - np.exp(di+ np.dot(mean_post[t],Ci) + 0.5*np.dot(Ci, np.dot(cov_post[t], Ci)))
        grad_Ci = grad_Ci + xi[t]* mean_post[t] -\
                   np.exp(di+np.dot(mean_post[t],Ci)+ 0.5*np.dot(Ci, np.dot(cov_post[t], Ci))) *\
                   (mean_post[t] + np.dot(cov_post[t],Ci))
    return grad_di, grad_Ci

def slow_expectedLLPoisson(x,C,d,mean_post,cov_post):
    LL = 0
    for i in range(x.shape[1]):
        LL += slow_expectedLLPoisson_coord(x[:,i], C[i,:], d[i], mean_post,cov_post)
    return LL


def grad_slow_expectedLLPoisson(x, C, d, mean_post, cov_post):
    grad_C = np.zeros(C.shape)
    grad_d = np.zeros(d.shape)
    for i in range(x.shape[1]):
        grad_d[i], grad_C[i] = grad_expectedLLPoisson_coord(x[:,i], C[i,:], d[i], mean_post, cov_post)

    return grad_C, grad_d#np.hstack((grad_C.flatten(),grad_d))

def MStepPoisson_func(x, C, d, mean_post,cov_post):
    '''
    The Spike count observation expected likelihood
    '''
    ydim, xdim = C.shape
    yhat = 0
    for yd in range(ydim):
        CC = np.outer(C[yd, :], C[yd, :])
        yhat += np.exp(0.5*np.sum(cov_post.reshape(cov_post.shape[0],xdim**2) * CC.reshape(xdim**2),axis=1)+\
                d[yd] + np.einsum('j,tj->t', C[yd], mean_post))
    yhat = yhat.sum()
    hh = np.einsum('tj,jk,tk->', x, C, mean_post, optimize=True) + np.dot(x.sum(axis=0),d)
    return hh - yhat


def d_MStepPoisson_func_dC_dD(x, C, d, mean_post, cov_post):
    ydim, xdim = C.shape
    dyhat_C = np.zeros((ydim, xdim))
    dyhat_d = np.zeros((ydim,))
    for yd in range(ydim):
        CC = np.outer(C[yd, :], C[yd, :])
        EXP = np.exp(0.5 * np.sum(cov_post.reshape(cov_post.shape[0], xdim ** 2) * CC.reshape(xdim ** 2), axis=1) + \
                     d[yd] + np.einsum('j,tj->t', C[yd], mean_post))
        covC = np.einsum('tij,j->ti', cov_post, C[yd])
        dyhat_C[yd] = np.einsum('t,tj->j', EXP, mean_post + covC)
        dyhat_d[yd] = EXP.sum()
    dhh_d = x.sum(axis=0)
    dhh_C = np.einsum('ti,tj->ij', x, mean_post)
    return dhh_d - dyhat_d, dhh_C - dyhat_C#np.hstack(((dhh_C - dyhat_C).flatten(), dhh_d - dyhat_d))



def ECz(z,C):
    return np.exp(np.dot(C,z))

def grad_ECz(z,C):
    return np.exp(np.dot(C,z)) * z

def approx_grad(x0, dim, func, epsi):
    grad = np.zeros(shape=dim)
    for j in range(grad.shape[0]):
        if np.isscalar(x0):
            ej = epsi
        else:
            ej = np.zeros(x0.shape[0])
            ej[j] = epsi
        grad[j] = (func(x0 + ej) - func(x0 - ej)) / (2 * epsi)
    return grad

if __name__ == '__main__':
    from time import perf_counter
    np.random.seed(4)
    import  matplotlib.pylab as plt
    T = 100
    C = np.random.uniform(size=(100,7))*0.1
    cov = np.random.normal(size=(C.shape[1],)*2)
    cov = np.dot(cov,cov.T)
    mean_post = np.random.multivariate_normal(mean=np.zeros(7), cov=cov, size=(T,))
    cov_post = np.zeros((T,7,7))
    for t in range(T):
        cov_post[t] = cov
    d = np.ones(C.shape[0])
    d[:d.shape[0]//2] = -0.3
    d[d.shape[0]//2:] = -0.1
    x = np.random.poisson(lam=np.exp(np.dot(C,mean_post.T).T+d))


    ## check the fast implementation using the slow one
    #
    #
    # func = lambda xx: -MStepPoisson_func(x,xx[:np.prod(C.shape)].reshape(C.shape),
    #     xx[np.prod(C.shape):].reshape(d.shape),mean_post,cov_post)
    # grad_func = lambda xx: -d_MStepPoisson_func_dC_dD(x, xx[:np.prod(C.shape)].reshape(C.shape),
    #                                     xx[np.prod(C.shape):].reshape(d.shape), mean_post, cov_post)
    #
    # xxbar = np.hstack((C.flatten(),d))
    # grad = -d_MStepPoisson_func_dC_dD(x, C, d, mean_post, cov_post)
    # # app_grad = approx_grad(xxbar,xxbar.shape[0],func,10**-6)

    t0 = perf_counter()
    ll0 = slow_expectedLLPoisson(x, C, d, mean_post, cov_post)
    t1 = perf_counter()
    ll1 = MStepPoisson_func(x, C, d, mean_post, cov_post)
    t2 = perf_counter()
    print('slow',t1-t0)
    print('fast',t2-t1)
    print('difference',ll0-ll1)


    # slow_expectedLLPoisson(xi, Ci, di, mean_post,cov_post)
    # MStepPoisson_func
    gC,gd = grad_slow_expectedLLPoisson(x, C, d, mean_post, cov_post)
    func = lambda CC : MStepPoisson_func(x, CC.reshape(C.shape), d, mean_post, cov_post)
    agC = approx_grad(C.flatten(),C.flatten().shape,func,10**-5)

    gd_fast,gC_fast = d_MStepPoisson_func_dC_dD(x, C, d, mean_post, cov_post)
    err = (gC.flatten()-agC)
    err2 = (gC.flatten() - gC_fast.flatten())
    print('cum err grad and approx', np.abs(err).max())
    print('err grad and grad fast', np.abs(err2).max())

    # for tt in range(T):
    #     # func = lambda di:funLin(x[tt,7],C[7],di,mean_post[tt],cov_post[tt])
    #     # grad_func = lambda di:gradLin(x[tt,7],C[7],di,mean_post[tt],cov_post[tt])
    #
    #     func = lambda di: fun(x[tt, 7], C[7], di, mean_post[tt], cov_post[tt])
    #     grad_func = lambda di: gradFun(x[tt, 7], C[7], di, mean_post[tt], cov_post[tt])
    #
    #     x0 = np.array([0.2])
    #     grad = grad_func(x0)
    #     app_grad = approx_grad(x0,1,func,10**-5)
    #     print(grad-app_grad,grad)

    # t0 = perf_counter()
    # grad0 = grad_slow_expectedLLPoisson(x, C, d, mean_post, cov_post)
    # t1 = perf_counter()
    # grad1 = d_MStepPoisson_func_dC_dD(x, C, d, mean_post, cov_post)
    # t2 = perf_counter()
    # print('slow', t1 - t0)
    # print('fast', t2 - t1)
    # print('difference', np.abs(grad0 - grad1).max())



    #
    # func = lambda xx: -slow_expectedLLPoisson(x, xx[:np.prod(C.shape)].reshape(C.shape),
    #                                      xx[np.prod(C.shape):].reshape(d.shape), mean_post, cov_post)
    # grad_func = lambda xx: -grad_slow_expectedLLPoisson(x, xx[:np.prod(C.shape)].reshape(C.shape),
    #                                                   xx[np.prod(C.shape):].reshape(d.shape), mean_post, cov_post)
    #
    # xxbar = np.hstack((C.flatten(), d))
    # grad = grad_func(xxbar)
    # app_grad = approx_grad(xxbar,xxbar.shape[0],func,10**-6)


    # plt.figure()
    # plt.title('max err: %f'%np.max(np.abs(grad-app_grad)))
    # plt.scatter(app_grad, grad)
    #
    # xxbar0 = -0.5+np.random.normal(size=xxbar.shape)*0.001
    # res = minimize(func,xxbar0,jac=grad_func,method='L-BFGS-B')
    # Crec = res.x[:np.prod(C.shape)].reshape(C.shape)
    # drec = res.x[np.prod(C.shape):]