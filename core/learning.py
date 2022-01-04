"""
Some of the code here is adapted from Machens et al. implementation of P-GPFA.
"""
import numpy as np
from scipy.optimize import minimize


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
    return np.hstack(((dhh_C - dyhat_C).flatten(), dhh_d - dyhat_d))



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


z = np.random.normal(size=5)
C0 = np.random.uniform(size=(5))
func = lambda C: ECz(z,C)
grad = grad_ECz(z,C0)
app_grad = approx_grad(C0,5,func,10**-4)