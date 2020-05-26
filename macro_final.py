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
from scipy.stats import kurtosis

from sklearn.linear_model import HuberRegressor,LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

from skpp import ProjectionPursuitRegressor

import MyHuberLoss

from joblib import Parallel, delayed
import multiprocessing

import random

import warnings
warnings.filterwarnings("ignore")


# In[2]:


x_month = pd.read_csv('8-agg-factors.csv').dropna().drop(columns='Data')
y_month = pd.read_csv('131-macrodat.csv',header = None)


# In[3]:


kurt_raw = y_month.apply(kurtosis,axis=0,fisher = False)


# ## Functions defined for use

# In[4]:


from sklearn.model_selection import KFold

def mse(data, pred):
    return(sum(data - pred)**2/len(data))

def in_sample_cv(X,y,model,cv = 5):
    kf = KFold(n_splits=cv,random_state=0)
    mse_cv = list()
    for train, _ in kf.split(X):
        mse_pred = mse(y[train],model.fit(X.iloc[train],y[train]).predict(X.iloc[train]))
        mse_cv.append(mse_pred)
    return(np.mean(mse_cv))

def out_sample_cv(X,y,model,cv = 5):
    kf = KFold(n_splits=cv,random_state=0)
    mse_cv = list()
    for train, test in kf.split(X):
        mse_pred = mse(y[test],model.fit(X.iloc[train],y[train]).predict(X.iloc[test]))
        mse_cv.append(mse_pred)
    return(np.mean(mse_cv))




# In[ ]:


def out_sample_cv_parallel(X,y,model,cv = 5):
    kf = KFold(n_splits=cv,random_state=0)
    
    def pred(train,test):
        return(mse(y[test],model.fit(X.iloc[train],y[train]).predict(X.iloc[test])))
    
    num_cores = multiprocessing.cpu_count()
    mse_cv = Parallel(n_jobs=5)(delayed(get_test_stat)(exc_return_df,x,window) for train,test in kf.split(X))
    return(np.mean(mse_cv))


# In[5]:


def fourier_basis(series,k):
    N = len(series)
    # t from 1 to N
    t = np.linspace(1,N,N)
    out_list = list()
    for index in range(1,k+1):
        # derieve numerial result of ai, bi (intergration) for fourier basis
        cosxfx = np.cos(2*index*t*np.pi/N)*np.array(series)
        sinxfx = np.sin(2*index*t*np.pi/N)*np.array(series)
        ai = 2/N*sum(cosxfx - cosxfx[0]/2 - cosxfx[-1]/2)
        bi = 2/N*sum(sinxfx - sinxfx[0]/2 - sinxfx[-1]/2)
        basis_i = ai*np.cos(2*np.pi*index*t/N)+bi*np.sin(2*np.pi*index*t/N)
        out_list.append(basis_i)
    return(out_list)

def polynomial_basis(series,k):
    out_list = list()
    for index in range(1,k+1):
        out_list.append(series**index)
    return(out_list)

def basis_transform(X,model = polynomial_basis,k = 5):
    temp = X.apply(model,axis = 0, k = k)
    out = np.array([i for i in temp]).reshape(k*X.shape[1],X.shape[0]).transpose()
    return(pd.DataFrame(out))


# In[43]:


def grid_cv(X, y, sample_method = out_sample_cv, model=None, basis = None, alpha_list = -1, J_list = -1, cv = 5):
    # the method is only use for SPCA model (alpha,J) and SPCA-LS model (J), not for PCA model and LS model
    if type(alpha_list) in {int, float}:
        alpha_list = [alpha_list]
    if type(J_list) in {int, float}:
        J_list = [J_list]
        
    if alpha_list != [-1] and J_list != [-1]:
        # case for scpa model
        res = pd.DataFrame([[a,J] for a in alpha_list for J in J_list],columns=['alpha','J'])
    
        final_par,final_loss = res.iloc[0,:],-1
        for item in range(res.shape[0]):
            X_expand = basis_transform(X, k = res.iloc[item,1])
            run_model = model(sigma = res.iloc[item,0],fit_intercept = False)

            loss = 0
            for i in range(y.shape[1]):
                yi = y[i]
                loss += sample_method(X_expand,yi,run_model,cv)
            if final_loss < 0 or final_loss > loss:
                final_par,final_loss = res.iloc[item,:],loss
    elif alpha_list == [-1]:
        # case for spca-ls model (model = sklearn.linearRegression)
        res = pd.DataFrame(J_list, columns=['J'])
        final_par,final_loss = res.iloc[0,:],-1
        for item in range(res.shape[0]):
            X_expand = basis_transform(X,k= res.iloc[item,0])
            run_model = model(fit_intercept = False)

            loss = 0
            for i in range(y.shape[1]):
                yi = y[i]
                loss += sample_method(X_expand,yi,run_model,cv)
            if final_loss < 0 or final_loss > loss:
                final_par,final_loss = res.iloc[item,:],loss
    else:
        final_par = -1
    return(final_par)


# In[7]:


N = 131
T = 480
C_list = [i/50 for i in range(1,5)] + [i/10 for i in range(1,11)] + [i for i in range(2,6)]
alpha_list = [np.sqrt(T/np.log(N*T))*C for C in C_list]
J_list = [i+1 for i in range(6)]


# In[8]:


x_month.shape
y_month.shape
pd.DataFrame(x_month[240:480]).reset_index()


# In[9]:


# par_out = grid_cv(x_month,y_month[1],out_sample_cv,MyHuberLoss.HuberRegressor,\
#         polynomial_basis,[0.1,0.2],3)
X_df,Y_df = pd.DataFrame(x_month[239:479]).reset_index().drop(columns='index'),pd.DataFrame(y_month[239:479]).reset_index().drop(columns='index')
par_out_spca = grid_cv(X_df,Y_df,out_sample_cv,MyHuberLoss.HuberRegressor,polynomial_basis,alpha_list,J_list)


# In[42]:


par_out_spca_ls = grid_cv(X_df,Y_df,out_sample_cv,LinearRegression,polynomial_basis,J_list = J_list)


# In[40]:


par_out_spca_ls


# ## Use parameters generated by cross validation to form model

# In[44]:


def augmented_factor_spca(y=None,obs_factors=None,alpha=1,J=1,start=0,end=None):
    # y is a pd.DataFrame with samples at row and features at column
    # obs_facotors is a pd.DataFrame with samples at row and features at column
    X = basis_transform(obs_factors,k=J)
    def robust_fit(y):
        return(MyHuberLoss.HuberRegressor(sigma=alpha,fit_intercept=False)               .fit(X.iloc[start:end,:], y).predict(X.iloc[start:end,:]))
    
    origin_y = y.iloc[start:end,:]
    pred_robust_df = origin_y.apply(robust_fit)
    pred_sigma = pred_robust_df.cov()

    K,N= obs_factors.shape[1] , origin_y.shape[1]
    T = origin_y.shape[0]
    
    pred_lambda= PCA(n_components=K,whiten=True).fit(pred_robust_df).components_*(N**0.5)
    pred_g = np.matmul(np.array(pred_robust_df),pred_lambda.transpose())/N
    pred_f = np.matmul(np.array(origin_y),pred_lambda.transpose())/N
    pred_gamma = pred_f - pred_g
    
    res_dict = {'lambda': pred_lambda, 'g':pred_g , 'f': pred_f, 'gamma': pred_gamma}
    return(res_dict)


# In[45]:


res_out_spca = augmented_factor_spca(Y_df,X_df,par_out_spca['alpha'],int(par_out_spca['J']))


# In[46]:


def augmented_factor_spca_ls(y=None,obs_factors=None,J=1,start=0,end=None):
    # y is a pd.DataFrame with samples at row and features at column
    # obs_facotors is a pd.DataFrame with samples at row and features at column
    X = basis_transform(obs_factors.iloc[start:end,:],k=J)
    
    origin_y = y.iloc[start:end,:]
    pred_df = pd.DataFrame(LinearRegression(fit_intercept=False).fit(X.iloc[start:end,:],y.iloc[start:end,:]).predict(X.iloc[start:end,:]))
    pred_sigma = pred_df.cov()

    K,N= obs_factors.shape[1] , origin_y.shape[1]
    T = origin_y.shape[0]
    
    pred_lambda= PCA(n_components=K,whiten=True).fit(pred_df).components_*(N**0.5)
    pred_g = np.matmul(np.array(pred_df),pred_lambda.transpose())/N
    pred_f = np.matmul(np.array(origin_y),pred_lambda.transpose())/N
    pred_gamma = pred_f - pred_g
    
    res_dict = {'lambda': pred_lambda, 'g':pred_g , 'f': pred_f, 'gamma': pred_gamma}
    return(res_dict)


# In[47]:


res_out_spca_ls = augmented_factor_spca_ls(Y_df,X_df,int(par_out_spca_ls['J']))


# In[48]:


def augmented_factor_pca(y=None,obs_factors=None,start=0,end=None):
    # y is a pd.DataFrame with samples at row and features at column
    # obs_facotors is a pd.DataFrame with samples at row and features at column
    
    origin_y = y.iloc[start:end,:]
    pred_df = pd.DataFrame(LinearRegression(fit_intercept = False).fit(obs_factors.iloc[start:end,:],y.iloc[start:end,:]).predict(obs_factors.iloc[start:end,:]))
    pred_sigma = pred_df.cov()

    K,N= obs_factors.shape[1] , origin_y.shape[1]
    T = origin_y.shape[0]
    
    pred_lambda= PCA(n_components=K,whiten=True).fit(pred_df).components_*(N**0.5)
    pred_g = np.matmul(np.array(pred_df),pred_lambda.transpose())/N
    pred_f = np.matmul(np.array(origin_y),pred_lambda.transpose())/N
    pred_gamma = pred_f - pred_g
    
    res_dict = {'lambda': pred_lambda, 'g':pred_g , 'f': pred_f, 'gamma': pred_gamma}
    return(res_dict)


# In[49]:


res_out_pca = augmented_factor_pca(Y_df,X_df)


# In[ ]:


pd.DataFrame(res_out_spca['f']).to_csv('temp/macro_res_out_spca_factor.csv')
pd.DataFrame(res_out_spca_ls['f']).to_csv('temp/macro_res_out_spca_ls_factor.csv')
pd.DataFrame(res_out_pca['f']).to_csv('temp/macro_res_out_pca_factor.csv')


# ## Notice on formulas
# 
# $z_s = \alpha + \beta'W_{s-1} + u_s, s= t-238,...,t$
# 
# We fit $\hat \alpha,\hat \beta$ based on $(z_{241},...,z_{480})$ and $(W_{240},...,W_{479})$ and we get prediction $(\hat z_{241|240},...,\hat z_{480|479})$

# In[51]:


def macro_R2(z,W,model_use):
    # z is a np.array with 480 values
    # W is a 240*colnum matrix with features at the column
    R2 = -1
    if model_use == 'linear':
        z_bar = [np.mean(z[(i-239):(i+1)]) for i in range(239,479)]
        z_raw = z[240:480]
        z_pred = LinearRegression().fit(W,z_raw).predict(W)
        R2 = 1 - sum((z_raw - z_pred)**2)/sum((z_bar - z_raw)**2)
    elif model_use == 'multi_index':
        value,_ = np.linalg.eig(np.cov(W.transpose()))
        L = int(np.argmax([value[i-1]/value[i] for i in range(1,8)]))+1
        model = ProjectionPursuitRegressor(r = L,fit_type='spline',degree=1)
        z_bar = [np.mean(z[(i-239):(i+1)]) for i in range(239,479)]
        z_raw = z[240:480]
        z_pred = model.fit(W,z_raw).predict(W)
        R2 = 1 - sum((z_raw - z_pred)**2)/sum((z_bar - z_raw)**2)
    return(R2)
    


# ## Recalculate $z$ for 4 different cases

# In[102]:


spca_factor = pd.read_csv('temp/macro_res_out_spca_factor.csv',index_col = 0)
spca_ls_factor = pd.read_csv('temp/macro_res_out_spca_ls_factor.csv',index_col = 0)
pca_factor = pd.read_csv('temp/macro_res_out_pca_factor.csv',index_col = 0)


# In[103]:


bond_price_all = pd.read_csv('data/bondprice.csv',skiprows = list(range(1,139)))
z2_all = np.array(np.log(bond_price_all.iloc[1:,1])) - np.array(np.log(bond_price_all.iloc[:-1,2])) + 1/2*np.array(np.log(bond_price_all.iloc[:-1,1]))
z3_all = np.array(np.log(bond_price_all.iloc[1:,2])) - np.array(np.log(bond_price_all.iloc[:-1,3])) + 1/3*np.array(np.log(bond_price_all.iloc[:-1,1]))
z4_all = np.array(np.log(bond_price_all.iloc[1:,3])) - np.array(np.log(bond_price_all.iloc[:-1,4])) + 1/4*np.array(np.log(bond_price_all.iloc[:-1,1]))
z5_all = np.array(np.log(bond_price_all.iloc[1:,4])) - np.array(np.log(bond_price_all.iloc[:-1,5])) + 1/5*np.array(np.log(bond_price_all.iloc[:-1,1]))


# In[104]:


z_list = [z2_all,z3_all,z4_all,z5_all]
z_name = ["z2","z3","z4","z5"]


# In[123]:


lin = list()
lin_fx = list()
mul = list()
mul_fx = list()

for z in z_list:
    lin.append(macro_R2(z,spca_factor,'linear'))
    lin_fx.append(macro_R2(z,spca_factor.join(X_df),'linear'))
    mul.append(macro_R2(z,spca_factor,'multi_index'))
    mul_fx.append(macro_R2(z,spca_factor.join(X_df),'multi_index'))

for z in z_list:
    lin.append(macro_R2(z,spca_ls_factor,'linear'))
    lin_fx.append(macro_R2(z,spca_ls_factor.join(X_df),'linear'))
    mul.append(macro_R2(z,spca_ls_factor,'multi_index'))
    mul_fx.append(macro_R2(z,spca_ls_factor.join(X_df),'multi_index'))

    
for z in z_list:
    lin.append(macro_R2(z,pca_factor,'linear'))
    lin_fx.append(macro_R2(z,pca_factor.join(X_df),'linear'))
    mul.append(macro_R2(z,pca_factor,'multi_index'))
    mul_fx.append(macro_R2(z,pca_factor.join(X_df),'multi_index'))


# In[126]:


m_index=pd.MultiIndex.from_product([['linear model', 'multi-index model'], ['ft,xt', '(ft,xt)']],
                                     names=['Model', ''])
m_columns = pd.MultiIndex.from_product([['SPCA', 'SPCA-LS','PCA'], ['2', '3','4','5']],
                                     names=['Wt', ''])
dat = np.array([lin,lin_fx,mul,mul_fx])
df1=pd.DataFrame(dat,index=m_index,columns=m_columns)
df1


# In[128]:


df1.to_csv('result/table_6_2.csv')


# In[ ]:




