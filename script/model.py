import numpy as np
import sklearn
from sklearn import linear_model,tree
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,AdaBoostRegressor,\
GradientBoostingRegressor,BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb





AdaBoost = AdaBoostRegressor(base_estimator=sklearn.ensemble.ExtraTreesRegressor(n_jobs=-1))

Bagging = BaggingRegressor(n_jobs=-1,)

DecisionTree = tree.DecisionTreeRegressor()
  
    
ExtraTrees = ExtraTreesRegressor(n_jobs=-1, )    
GradientBoosting=GradientBoostingRegressor()                          
KNeighbors=KNeighborsRegressor() 

KernelRidge = KernelRidge()

LinearSVR=LinearSVR()

RandomForest =  RandomForestRegressor()

Ridge = linear_model.Ridge()

SVR = SVR()
XGB = xgb.XGBRegressor()                          
NeuralNetwork = MLPRegressor()                          
                                               
                          
models = [Bagging,
          DecisionTree,ExtraTrees,GradientBoosting,
          KNeighbors,KernelRidge,
          LinearSVR,RandomForest,
          Ridge,SVR, XGB]
                          
                          
'''models = [AdaBoostRegressor(ExtraTreesRegressor(n_jobs=60)),BaggingRegressor(n_jobs=60),
          tree.DecisionTreeRegressor(),ExtraTreesRegressor(n_jobs=60),GradientBoostingRegressor(),
          KNeighborsRegressor(),KernelRidge(),
          LinearSVR(),RandomForestRegressor(n_jobs=60,criterion='mae',n_estimators=10,max_depth=10),
          linear_model.Ridge(alpha=.5),SVR(), xgb.XGBRegressor(n_jobs=60),
          MLPRegressor(hidden_layer_sizes=(100,100))]'''