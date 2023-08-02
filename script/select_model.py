import pandas as pd 
import numpy as np
import csv
import sys 
from os import sep
import warnings
warnings.filterwarnings("ignore")

from sklearn import linear_model
from sklearn import tree
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,AdaBoostRegressor,\
ExtraTreesClassifier,RandomForestClassifier,GradientBoostingRegressor,BaggingRegressor
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

models = [BaggingRegressor(n_jobs=60),
          tree.DecisionTreeRegressor(),ExtraTreesRegressor(n_jobs=60),GradientBoostingRegressor(),
          KNeighborsRegressor(),KernelRidge(),
          LinearSVR(),RandomForestRegressor(n_jobs=60),
          linear_model.Ridge(alpha=.5),SVR(), xgb.XGBRegressor(n_jobs=60)]
model_names = ['BG','DT','ET','GB','KNR','KRR','LSVR','RF','Ridge','SVR','XGB']
kfold = KFold(n_splits=10,shuffle=True)

def get_selected_model(input_index,des_std,yield_std):
    n = 20
    max_pearsr = -10
    tem_desc=des_std[input_index]
    tem_yield=yield_std
    for model,model_name in zip(models,model_names):
        all_pearsr = []
        repeat_pred = []
        repeat_test = []
        np.random.seed(36)
        for i in range(n):
            all_pred = []
            all_test = []
            for train_index_tep,test_index_tep in kfold.split(tem_desc):
                train_x,test_x = tem_desc[train_index_tep],tem_desc[test_index_tep]
                train_y,test_y = tem_yield[train_index_tep],tem_yield[test_index_tep]
                model.fit(train_x,train_y)
                test_pred = model.predict(test_x)
                all_pred.append(test_pred)
                all_test.append(test_y)
            all_pred = np.concatenate(all_pred)
            all_test = np.concatenate(all_test)
            repeat_pred.append(all_pred)
            repeat_test.append(all_test)
            pearsr = pearsonr(all_test,all_pred)
            all_pearsr.append(pearsr[0])
        print('Model: %5s, Pearson R: %.4f'%(model_name,np.mean(all_pearsr)))
        if np.mean(all_pearsr)>max_pearsr:
            max_pearsr = np.mean(all_pearsr)
            best_model = model_name
    print(best_model)
    return best_model