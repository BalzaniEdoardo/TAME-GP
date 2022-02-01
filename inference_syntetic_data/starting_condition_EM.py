import numpy as np
import os,sys
basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(basedir,'core'))
sys.path.append(os.path.join(basedir,'initialization'))
from learnGaussianParam import learn_GaussianParams,full_GaussLL
from learnPoissonParam import all_trial_PoissonLL
from copy import deepcopy
from inference_factorized import newton_optim_map, reconstruct_post_mean_and_cov
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from gen_synthetic_data import dataGen
import matplotlib.pylab as plt
from expectation_maximization_factorized import *


mn_all = np.zeros((100,50))
std_all = np.zeros((100, 50))
latent = np.zeros((100, 50))
N = 500
lat = 2
if lat == 2:
    gen_dat = dataGen(100, N=200, N1=N,K2=2, K3=2, T=50,infer=False,setTruePar=True)
else:
    gen_dat = dataGen(100, N=N, N1=200, K2=2, K3=2, T=50, infer=False, setTruePar=True)

dat = gen_dat.cca_input
zmap, success, td, ll, ll_hist = newton_optim_map(dat, tol=10 ** -10, max_iter=100, max_having=20,
                                                  disp_ll=True, init_zeros=True, useNewton=True)
reconstruct_post_mean_and_cov(dat, zmap, td)


ii = dat.zdims[:lat].sum()
for tr in range(100):
    mn = dat.posterior_inf[tr].mean[lat][0]
    std = np.sqrt(dat.posterior_inf[tr].cov_t[lat][:, 0, 0])
    mn_all[tr] = mn
    std_all[tr] = std
    latent[tr] = dat.ground_truth_latent[tr][:,ii]




## fit em
dat2 = deepcopy(dat)
dat2.initializeParam(dat2.zdims,use_poissonPCA=True)
zmap, success, td, ll, ll_hist = newton_optim_map(dat2, tol=10 ** -10, max_iter=100, max_having=20,
                                                  disp_ll=True, init_zeros=True, useNewton=True)
reconstruct_post_mean_and_cov(dat2, zmap, td)


ll_list = expectation_mazimization_factorized(dat2, maxIter=4, boundsW0=[-3, 3], boundsD=[-10, 10])
# mn = dat2.posterior_inf[tr].mean[lat][0]
# std = np.sqrt(dat2.posterior_inf[tr].cov_t[lat][:, 0, 0])
# lt = dat2.ground_truth_latent[tr][:, ii]
# plt.figure()
# plt.title('EM reconstructed posterior: N %d'%N)
# p, = plt.plot(mn)
# plt.fill_between(np.arange(mn.shape[0]), mn - 1.96 * std, mn + 1.96 * std, color=p.get_color(), alpha=0.4)
# plt.plot(lt, color='k')
# model = LinearRegression()
# MN_post = dat2.posterior_inf[tr].mean[1]
# lat = dat2.ground_truth_latent[tr][:, ii:ii+dat2.zdims[lat]]
# res = model.fit(MN_post,lat.T)
#
# plt.figure(figsize=(12,4))
# lat = 1
# plt.subplot(121)
# plt.title('True parameter latent posterior:')
# tr = 13
# mn0,mn1 = dat.posterior_inf[tr].mean[0]
# p0, = plt.plot(mn0)
# p1, = plt.plot(mn1)
# std0 = np.sqrt(dat.posterior_inf[tr].cov_t[lat][:, 0, 0])
# std1 = np.sqrt(dat.posterior_inf[tr].cov_t[lat][:, 1, 1])
# plt.fill_between(range(mn0.shape[0]), mn0-std0, mn0+std0,alpha=0.4,color=p0.get_color())
# plt.fill_between(range(mn1.shape[0]), mn1-std1, mn1+std1,alpha=0.4,color=p1.get_color())
#
# plt.subplot(122)
# plt.title('EM parameter latent posterior:')
#
# mn0, mn1 = dat2.posterior_inf[tr].mean[0]*-1
# p1, = plt.plot(mn1)
# p0, = plt.plot(mn0)
#
# std1 = np.sqrt(dat2.posterior_inf[tr].cov_t[lat][:, 0, 0])
# std0 = np.sqrt(dat2.posterior_inf[tr].cov_t[lat][:, 1, 1])
# plt.fill_between(range(mn0.shape[0]), mn0 - std0, mn0 + std0, alpha=0.4, color=p0.get_color())
# plt.fill_between(range(mn1.shape[0]), mn1 - std1, mn1 + std1, alpha=0.4, color=p1.get_color())
#
#
#
#
#
