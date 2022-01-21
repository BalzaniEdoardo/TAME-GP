import numpy as np
import csr
from inference_factorized import hess_factorized_logLike
import os,inspect,sys
basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(basedir,'core'))
from data_processing_tools import emptyStruct

def fast_stackCSRHes_memoryPreAllocation(vals, rowsptr, colindices, nnz, nrows, ncols, i0PTR, i0Val, newHes, sumK):
    """
    Fast stacking of csr matrix format hessian of different trials. Strongly uses the fact that across trials the
    structure of the matrix is the same. The important bit is to make sure that from one trial to another there are no
    additional zeroos in the block structure of the matrix; thi in the factorized model computation of the hessian is
    done by adding  (min + 1) to vector of values that go into the matrix.
    :param spHess: hessian for the log likelihood of multiple trials in csr.CSR format
    :param newHes: hessian for the log likelihood of a new trial in csr.CSR format that we want to stack as a new
    block-diagonal component
    :return:
    """
    # update indices
    new_indices = newHes.colinds + colindices[i0Val-1] + 1
    new_ptr = newHes.rowptrs + rowsptr[i0PTR]

    # create the concatenated hessian
    vals[i0Val: i0Val + newHes.values.shape[0]] = newHes.values
    rowsptr[i0PTR: i0PTR + new_ptr.shape[0]] = new_ptr
    colindices[i0Val: i0Val + newHes.values.shape[0]] = new_indices

    nnz = nnz + newHes.nnz
    nrows = nrows + newHes.nrows
    ncols = ncols + newHes.ncols

    i0Val = i0Val + newHes.values.shape[0]
    i0PTR = i0PTR + new_ptr.shape[0] - 1

    return nrows, ncols, nnz, rowsptr, colindices, vals, i0Val, i0PTR

def preproc_post_mean_factorizedModel(dat, returnDict=True):
    sumK = np.sum(dat.zdims)
    post_mean = {}
    if 'posterior_inf' not in dat.__dict__.keys():
        for tr in dat.trialDur.keys():
            T = dat.trialDur[tr]
            post_mean[tr] = np.zeros(T * sumK, dtype=np.float32, order='C')
    else:
        for tr in dat.trialDur.keys():
            T = dat.trialDur[tr]
            post_mean[tr] = np.zeros(T * sumK, dtype=np.float32, order='C')
            # start ordering it [K0*T, K1 *T, ... ]
            i0 = 0
            for j in range(len(dat.zdims)):
                post_mean[tr][i0: i0 + dat.zdims[j] * T] = dat.posterior_inf[tr].mean[j] .flatten()  # here you have stacked [z_{0,1:T},z_{1,1:T}, ...]
                i0 += dat.zdims[j] * T

    if not returnDict:
        totDur = np.sum(list(dat.trialDur.values()))
        zbar = np.zeros(totDur * sumK, dtype=np.float32, order='C')
        tr_dict = {}
        i0 = 0
        for tr in dat.trialDur.keys():
            zbar[i0: i0+dat.trialDur[tr]*sumK] = post_mean[tr]
            tr_dict[tr] = np.arange(i0, i0+dat.trialDur[tr]*sumK, dtype=np.int32)
            i0 += dat.trialDur[tr]*sumK
        return zbar, tr_dict
    return post_mean


def fast_stackCSRHes(spHess, newHes):
    """
    Fast stacking of csr matrix format hessian of different trials. Strongly uses the fact that across trials the
    structure of the matrix is the same. The important bit is to make sure that from one trial to another there are no
    additional zeroos in the block structure of the matrix; thi in the factorized model computation of the hessian is
    done by adding  (min + 1) to vector of values that go into the matrix.
    :param spHess: hessian for the log likelihood of multiple trials in csr.CSR format
    :param newHes: hessian for the log likelihood of a new trial in csr.CSR format that we want to stack as a new
    block-diagonal component
    :return:
    """
    # update indices
    new_indices = newHes.colinds + spHess.colinds[-1] + 1
    new_ptr = newHes.rowptrs + spHess.rowptrs[-1]

    # create the concatenated hessian
    values = np.zeros(spHess.values.shape[0] + newHes.values.shape[0], dtype=np.float64, order='C')
    indices = np.zeros(spHess.colinds.shape[0] + new_indices.shape[0], dtype=np.int32, order='C')
    indptr = np.zeros(spHess.rowptrs.shape[0] + new_ptr.shape[0] - 1, dtype=np.int32, order='C')

    values[:spHess.values.shape[0]] = spHess.values
    values[spHess.values.shape[0]:] = newHes.values

    indptr[:spHess.rowptrs.shape[0] - 1] = spHess.rowptrs[:-1]
    indptr[spHess.rowptrs.shape[0] - 1:] = new_ptr

    indices[:spHess.values.shape[0]] = spHess.colinds
    indices[spHess.values.shape[0]:] = new_indices

    nnz = spHess.nnz + newHes.nnz
    nrows = spHess.nrows + newHes.nrows
    ncols = spHess.ncols + newHes.ncols


    return csr.CSR(nrows, ncols, nnz, indptr, indices, values)

def reconstruct_post_mean_and_cov(dat, zbar, index_dict):
    """

    :param dat: P_GPCCA inputt
    :param zbar: the output of newton optim (which is the posterior mean for all trials)
    :param index_dict: index of zbar for each trial
    :return:
    """
    if 'posterior_inf' not in dat.__dict__.keys():
        dat.posterior_inf = {}

    for tr in index_dict.keys():
        stim, xList = dat.get_observations(tr)
        post_mean = zbar[index_dict[tr]]
        post_cov = -hess_factorized_logLike(dat, tr, stim, xList, zbar=post_mean, inverse=True,
                            return_tensor=True)
        T = dat.trialDur[tr]

        if tr not in dat.posterior_inf.keys():
            dat.posterior_inf[tr] = emptyStruct()
            dat.posterior_inf[tr].mean = {}
            dat.posterior_inf[tr].cov_t = {}
            dat.posterior_inf[tr].cross_cov_t = {}

        i0 = 0
        c0 = 0
        for j in range(len(dat.zdims)):
            dat.posterior_inf[tr].mean[j] = post_mean[i0: i0 + dat.zdims[j] * T].reshape(dat.zdims[j],T).T
            dat.posterior_inf[tr].cov_t[j] = post_cov[:, c0: c0 + dat.zdims[j], c0: c0 + dat.zdims[j]]
            if j > 0:
                dat.posterior_inf[tr].cross_cov_t[j] = post_cov[:, :dat.zdims[0], c0: c0 + dat.zdims[j]]

            i0 += dat.zdims[j] * T
            c0 += dat.zdims[j]
    return True