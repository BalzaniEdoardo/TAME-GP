import numpy as np
import csr
import os,inspect,sys
basedir = os.path.dirname(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.join(basedir,'core'))
from data_processing_tools import emptyStruct,sortGradient_idx
from numba import jit



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
                post_mean[tr][i0: i0 + dat.zdims[j] * T] = dat.posterior_inf[tr].mean[j].T.flatten() # here you have stacked [z_{0,1:T},z_{1,1:T}, ...]
                i0 += dat.zdims[j] * T


    if not returnDict:
        totDur = np.sum(list(dat.trialDur.values()))
        zbar = np.zeros(totDur * sumK, dtype=np.float32, order='C')
        tr_dict = {}
        i0 = 0
        for tr in dat.trialDur.keys():
            idx_sort = sortGradient_idx(dat.trialDur[tr], dat.zdims, isReverse=False)
            zbar[i0: i0+dat.trialDur[tr]*sumK] = post_mean[tr][idx_sort]
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



@jit(nopython=True)
def invertBlocks(*args):
    """
    Invert a list of T x Kj x Kj blocks of a block diagonal matri
    we want to invert
    :param args:
    :return:
    """
    Kmax = 0
    cc = 0
    for k in range(len(args)):
        blocks = args[k]
        Kmax = max(Kmax, blocks.shape[1])
        cc += blocks.shape[0]

    M = np.zeros((cc, Kmax, Kmax),dtype=np.float64)
    cc = 0
    for k in range(len(args)):
        blocks = args[k]
        K = blocks.shape[1]
        for t in range(blocks.shape[0]):
            B = blocks[t]
            M[cc, :K, :K] = np.linalg.inv(B)
            cc += 1
    return M
