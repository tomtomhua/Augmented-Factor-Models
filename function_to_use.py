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

from scipy.linalg import sqrtm

import MyHuberLoss

from joblib import Parallel, delayed
import multiprocessing

import warnings
warnings.filterwarnings("ignore")


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

def out_sample_cv_parallel(X,y,model,cv = 5):
    kf = KFold(n_splits=cv,random_state=0)
    
    def pred(train,test):
        return(mse(y[test],model.fit(X.iloc[train],y[train]).predict(X.iloc[test])))
    
    num_cores = multiprocessing.cpu_count()
    mse_cv = Parallel(n_jobs=5)(delayed(pred)(train,test) for train,test in kf.split(X))
    return(np.mean(mse_cv))

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
            try:
                X_expand = basis_transform(X, k = res.iloc[item,1])
                run_model = model(sigma = res.iloc[item,0],fit_intercept = False)

                loss = 0
                for i in range(y.shape[1]):
                    yi = y[i]
                    loss += sample_method(X_expand,yi,run_model,cv)
                if final_loss < 0 or final_loss > loss:
                    final_par,final_loss = res.iloc[item,:],loss
            except:
                continue
    elif alpha_list == [-1]:
        # case for spca-ls model (model = sklearn.linearRegression)
        res = pd.DataFrame(J_list, columns=['J'])
        
        final_par,final_loss = res.iloc[0,:],-1
        for item in range(res.shape[0]):
            try:
                X_expand = basis_transform(X, k=res.iloc[item,0])
                run_model = model(fit_intercept = False)

                loss = 0
                for i in range(y.shape[1]):
                    yi = y[i]
                    loss += sample_method(X_expand,yi,run_model,cv)
                if final_loss < 0 or final_loss > loss:
                    final_par,final_loss = res.iloc[item,:],loss
            except:
                continue
    else:
        final_par = -1
    return(final_par)

def augmented_factor_spca(y=None,obs_factors=None,alpha=1,J=1,start=0,end=None):
    # y is a pd.DataFrame with samples at row and features at column
    # obs_facotors is a pd.DataFrame with samples at row and features at column
    X = basis_transform(obs_factors,k=J)
    def robust_fit(y):
        return(MyHuberLoss.HuberRegressor(sigma=alpha,fit_intercept=False,tol=1e-3)\
               .fit(X.iloc[start:end,:], y).predict(X.iloc[start:end,:]))
    
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