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


y_all,x_all = list(),list()
f_all = list()
Lambda_all = list()

omega_list = [10,1,0.1]

for oi in omega_list:
    K,N,T = 5,40,100
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
    f_t = sigma_g * gx_t + sigma_gamma * gamma_t
    f_t_fb = sigma_g * gx_fb + sigma_gamma * gamma_t

    Lambda = np.random.normal(0,1,(N,K))
    y_t_normal = np.matmul(Lambda,f_t) + u_t_normal
    y_t_normal_log = np.matmul(Lambda,f_t) + u_t_normal_log
    y_t_normal_fb = np.matmul(Lambda,f_t_fb) + u_t_normal
    y_t_normal_log_fb = np.matmul(Lambda,f_t_fb) + u_t_normal_log

    y_list = [y_t_normal,y_t_normal_fb,y_t_normal_log,y_t_normal_log_fb]
    x_list = [x_t,x_ti,x_t,x_ti]
    f_list = [f_t,f_t_fb,f_t,f_t_fb]
    Lambda_list = [Lambda,Lambda,Lambda,Lambda]
    
    y_all.extend(y_list)
    x_all.extend(x_list)
    f_all.extend(f_list)
    Lambda_all.extend(Lambda_list)


# In[3]:


C_list = [i/50 for i in range(1,5)] + [i/10 for i in range(1,11)] + [i for i in range(2,6)]
alpha_list = [np.sqrt(T/np.log(N*T))*C for C in C_list]
J_list = [i+1 for i in range(6)]


# In[4]:


def ccr_median(U,V):
    cca = CCA(n_components=5)
    U_c, V_c = cca.fit_transform(U, V)
    coef = np.abs(np.corrcoef(U_c.T,V_c.T).diagonal(offset = 5))
    return(np.median(coef))


# In[5]:


spca_res_f = list()
spca_ls_res_f = list()
pca_res_f = list()

spca_res_load = list()
spca_ls_res_load = list()
pca_res_load = list()

# for i in range(1):
for i in range(12):
    X_use,Y_use = x_all[i].transpose(),y_all[i].transpose()
    X_df,Y_df = pd.DataFrame(X_use),pd.DataFrame(Y_use)
    par_out_spca = grid_cv(X_df,Y_df,out_sample_cv_parallel,MyHuberLoss.HuberRegressor,polynomial_basis,alpha_list,J_list)
    par_out_spca_ls = grid_cv(X_df,Y_df,out_sample_cv_parallel,LinearRegression,polynomial_basis,J_list = J_list)
    
    try:
        res_out_spca = augmented_factor_spca(Y_df,X_df,par_out_spca['alpha'],int(par_out_spca['J']))        
        spca_res_f.append(ccr_median(f_all[i].T,res_out_spca['f']))
        spca_res_load.append(ccr_median(Lambda_all[i],res_out_spca['lambda'].T))
    except:
        spca_res_f.append(-1)
        spca_res_load.append(-1)

    res_out_spca_ls = augmented_factor_spca_ls(Y_df,X_df,int(par_out_spca_ls['J']))
    res_out_pca = augmented_factor_pca(Y_df,X_df)
    
    spca_ls_res_f.append(ccr_median(f_all[i].T,res_out_spca_ls['f']))
    pca_res_f.append(ccr_median(f_all[i].T,res_out_pca['f']))
    
    spca_ls_res_load.append(ccr_median(Lambda_all[i],res_out_spca_ls['lambda'].T))
    pca_res_load.append(ccr_median(Lambda_all[i],res_out_pca['lambda'].T))
    


# In[6]:


normal_f = list()
normal_f.append(spca_res_f[0:12:4])
normal_f.append(spca_ls_res_f[0:12:4])
normal_f.append(pca_res_f[0:12:4])
normal_f.append(spca_res_f[1:12:4])
normal_f.append(spca_ls_res_f[1:12:4])
normal_f.append(pca_res_f[1:12:4])

lognormal_f = list()
lognormal_f.append(spca_res_f[2:12:4])
lognormal_f.append(spca_ls_res_f[2:12:4])
lognormal_f.append(pca_res_f[2:12:4])
lognormal_f.append(spca_res_f[3:12:4])
lognormal_f.append(spca_ls_res_f[3:12:4])
lognormal_f.append(pca_res_f[3:12:4])

normal_l = list()
normal_l.append(spca_res_f[0:12:4])
normal_l.append(spca_ls_res_f[0:12:4])
normal_l.append(pca_res_f[0:12:4])
normal_l.append(spca_res_f[1:12:4])
normal_l.append(spca_ls_res_f[1:12:4])
normal_l.append(pca_res_f[1:12:4])

lognormal_l = list()
lognormal_l.append(spca_res_f[2:12:4])
lognormal_l.append(spca_ls_res_f[2:12:4])
lognormal_l.append(pca_res_f[2:12:4])
lognormal_l.append(spca_res_f[3:12:4])
lognormal_l.append(spca_ls_res_f[3:12:4])
lognormal_l.append(pca_res_f[3:12:4])


# In[7]:


ar1 = np.array(normal_l).transpose()
ar2 = np.array(lognormal_l).transpose()
ar3 = np.array(normal_f).transpose()
ar4 = np.array(lognormal_f).transpose()


# In[8]:


m_index=pd.MultiIndex.from_product([['Loadings','Factors'],['Normal', 'LogN'], ['10', '1','0.1']],
                                     names=['','ut', 'omega'])
m_columns = pd.MultiIndex.from_product([['Model I', 'Model II'], ['SPCA', 'SPCA-LS','PCA']],
                                     names=['', ''])
dat = np.vstack([ar1,ar2,ar3,ar4])
df1=pd.DataFrame(dat,index=m_index,columns=m_columns)
df1


# In[9]:


df1.to_csv('result/single_replication_table_7_2.csv')


# In[ ]:





# In[ ]:





# In[ ]:




