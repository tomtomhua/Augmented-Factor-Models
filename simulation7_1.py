#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas_datareader import data
from datetime import datetime
from numpy.linalg import inv
from scipy.stats import normaltest

from sklearn.linear_model import HuberRegressor,LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.cross_decomposition import CCA

import random
from scipy.linalg import sqrtm

import MyHuberLoss

from joblib import Parallel, delayed
import multiprocessing

from function_to_use import *

import warnings
warnings.filterwarnings("ignore")


# In[2]:


y_all,x_all,f_all = list(),list(),list()

omega_list = [10,1,0.1]
for oi in omega_list:
    K,N,T = 5,40,100+50
    gamma_t = scale(np.random.normal(0,1,(K,T)))
    u_t_normal = np.random.normal(0,1,(N,T))
    u_t_normal_log = np.random.lognormal(0,1,(N,T))
    x_t = np.random.normal(0,1,(K,T))

    D = np.random.uniform(1,2,(K,K))
    gx_t = scale(np.matmul(D,x_t),axis = 1)

    xs = np.linspace(0,1,T)
    theta_xs = np.sin(xs) + 2*np.exp(-30*(xs**2))
    sig = np.fft.rfft(theta_xs)/len(xs)
    a,b = np.real(sig[0:5]),np.imag(sig[0:5])
    x_ti = np.random.normal(0,1,T)
    gx = list()
    for i in range(5):
        gx.append(np.cos(x_ti*(i+1)**2*np.pi)*a[i] + np.sin(x_ti*(i+1)**2*np.pi)*b[i])
    gx_fb = scale(np.array(gx),axis=0)

    sigma_g = oi/(oi**2+1)**0.5
    sigma_gamma = 1/(oi**2+1)**0.5
    f_t = np.array(sigma_g * gx_t + sigma_gamma * gamma_t)
    f_t_fb = np.array(sigma_g * gx_fb + sigma_gamma * gamma_t)

    Lambda = np.random.normal(0,1,(N,K))
    y_t_normal = np.matmul(Lambda,f_t) + u_t_normal
    y_t_normal_log = np.matmul(Lambda,f_t) + u_t_normal_log
    y_t_normal_fb = np.matmul(Lambda,f_t_fb) + u_t_normal
    y_t_normal_log_fb = np.matmul(Lambda,f_t_fb) + u_t_normal_log

    y_list = [y_t_normal,y_t_normal_fb,y_t_normal_log,y_t_normal_log_fb]
    x_list = [x_t,x_ti,x_t,x_ti]
    f_list = [f_t,f_t_fb,f_t,f_t_fb]
    
    y_all.extend(y_list)
    x_all.extend(x_list)
    f_all.extend(f_list)


# In[3]:


C_list = [i/50 for i in range(1,5)] + [i/10 for i in range(1,11)] + [i for i in range(2,6)]
alpha_list = [np.sqrt(T/np.log(N*T))*C for C in C_list]
J_list = [i+1 for i in range(6)]


# In[4]:


spca_res = list()
spca_ls_res = list()

# for i in range(1):
for i in range(12):
    X_use,Y_use = x_all[i].transpose(),y_all[i].transpose()
    X_df,Y_df = pd.DataFrame(X_use[99:149]).reset_index().drop(columns='index'),pd.DataFrame(Y_use[99:149]).reset_index().drop(columns='index')
    par_out_spca = grid_cv(X_df,Y_df,out_sample_cv_parallel,MyHuberLoss.HuberRegressor,polynomial_basis,alpha_list,J_list)
    par_out_spca_ls = grid_cv(X_df,Y_df,out_sample_cv_parallel,LinearRegression,polynomial_basis,J_list = J_list)
    
    beta = np.random.uniform(0.5,1.5,K)
    z_t = np.matmul(beta,f_all[i]) + np.random.normal(0,1,150)
    z_ts = [np.mean(z_t[(i+50):(i+100)]) for i in range(50)]

    res_out_pca = augmented_factor_pca(Y_df,X_df)    
    pred_z_pca = LinearRegression(fit_intercept=False).fit(res_out_pca['f'],z_ts).predict(res_out_pca['f'])
    
    try:
        res_out_spca = augmented_factor_spca(Y_df,X_df,par_out_spca['alpha'],int(par_out_spca['J']))
        pred_z_spca = LinearRegression(fit_intercept=False).fit(res_out_spca['f'],z_ts).predict(res_out_spca['f'])
        spca_r2 = sum((pred_z_spca - z_ts)**2)/sum((pred_z_pca - z_ts)**2)
        spca_res.append(spca_r2)
    except:
        spca_res.append(-1)

    res_out_spca_ls = augmented_factor_spca_ls(Y_df,X_df,int(par_out_spca_ls['J']))
    pred_z_spca_ls = LinearRegression(fit_intercept=False).fit(res_out_spca_ls['f'],z_ts).predict(res_out_spca_ls['f'])
    spca_ls_r2 = sum((pred_z_spca_ls - z_ts)**2)/sum((pred_z_pca - z_ts)**2)
    spca_ls_res.append(spca_ls_r2)
    


# ## Form dataframe for output data
# order:
# omega(10,1,0.1) + \[(normal,I) - (normal,II) - (logN,I) - (logN,II)\]

# In[27]:


normal = list()
normal.append(spca_res[0:12:4])
normal.append(spca_ls_res[0:12:4])
normal.append(spca_res[1:12:4])
normal.append(spca_ls_res[1:12:4])

lognormal = list()
lognormal.append(spca_res[2:12:4])
lognormal.append(spca_ls_res[2:12:4])
lognormal.append(spca_res[3:12:4])
lognormal.append(spca_ls_res[3:12:4])


# In[28]:


ar1 = np.array(normal).transpose()
ar2 = np.array(lognormal).transpose()


# In[38]:


m_index=pd.MultiIndex.from_product([['Normal', 'LogN'], ['10', '1','0.1']],
                                     names=['ut', 'omega'])
m_columns = pd.MultiIndex.from_product([['Model I', 'Model II'], ['SPCA', 'SPCA-LS']],
                                     names=['', ''])
dat = np.vstack([ar1,ar2])
df1=pd.DataFrame(dat,index=m_index,columns=m_columns)
df1


# In[39]:


df1.to_csv('result/single_replication_table_7_1.csv')


# In[ ]:




