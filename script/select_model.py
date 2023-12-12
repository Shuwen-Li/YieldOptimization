import numpy as np
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore")
random_seed=2023
param_grid = {    'BG':{'n_estimators':[50,100,200,300,400]},
                  'DT':{'max_depth':[None,10,20,30]},
                  'ET':{'n_estimators':[50,100,200,300,400],'max_depth':[None,10,20,30]},
                  'GB':{'n_estimators':[50,100,200,300,400],'max_depth':[3,4,5]},
                  'KNR':{'n_neighbors':[2,4,6,8,10,12,14]},
                  'KRR':{'gamma':[None,0.01,0.001,0.0001]},
                  'LSVR':{'epsilon':[0.0,0.05,0.1],"C":[1,2,3,4,5,6,7,8,9,10]},
                  'RF':{'n_estimators':[50,100,200,300,400],'max_depth':[None,10,20,30]},
                  'Ridge':{'alpha':[0.5,1.0,1.5]},
                  'SVR':{'kernel':['rbf', 'linear', 'poly'],'gamma':['scale','auto']},
                  'XGB':{'max_depth':[None,10,20,30]},
                }

models = [BaggingRegressor(n_jobs=-1,random_state=random_seed),
          DecisionTreeRegressor(random_state=random_seed),                 
          ExtraTreesRegressor(n_jobs=-1,random_state=random_seed),
          GradientBoostingRegressor(random_state=random_seed),                   
          KNeighborsRegressor(n_jobs=-1),                    
          KernelRidge( ),                   
          LinearSVR( ),                   
          RandomForestRegressor(n_jobs=-1,random_state=random_seed),
          Ridge(random_state=random_seed),                      
          SVR(),                      
          XGBRegressor(n_jobs=-1,random_state=random_seed),            
            ]
model_names = ['BG','DT','ET','GB','KNR','KRR','LSVR','RF','Ridge','SVR','XGB']
def get_best_model_and_param(des_std,yield_std,input_index):
    best_params = {}
    performance_result = {}
    model2score={}
    kfold = KFold(n_splits=10,shuffle=True,random_state=random_seed)
    for model_name,model in zip(model_names,models):
        train_val_desc,train_val_target = des_std[input_index],yield_std
        GS = GridSearchCV(model,param_grid[model_name],cv=kfold,n_jobs=-1,scoring='neg_mean_absolute_error' )#neg_mean_absolute_error neg_mean_squared_error 
        GS.fit(train_val_desc,train_val_target)
        best_param = GS.best_params_
        best_score = GS.best_score_
        best_params[model_name] = best_param
        model2score[model_name] = best_score
        print('Model: %4s, Best Socre: %.4f, Best Param: '%(model_name,best_score),best_param)
    best_model_name=list(model2score.keys())[np.argmax(np.array(list((model2score.values()))))]
    best_model_params=best_params[best_model_name]
    print('Best Model: %4s, Best Param: '% best_model_name,best_model_params)
    return best_model_name,best_model_params,best_params