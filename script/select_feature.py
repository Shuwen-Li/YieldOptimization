import numpy as np
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

kfold = KFold(n_splits=10,shuffle=True)
def get_model(model_name,best_params,random_seed=2023):
    assert model_name == 'BG' or model_name == 'DT' or model_name == 'ET' or\
            model_name == 'GB' or model_name == 'KNR' or model_name == 'KRR' or\
            model_name == 'LSVR' or model_name == 'RF' or model_name == 'Ridge' or\
            model_name == 'SVR'or model_name == 'XGB' , 'Not support this ML model %s'%model_name
    
    if model_name=='BG':
        model = BaggingRegressor(n_jobs=-1,random_state=random_seed,
                                   n_estimators=best_params[model_name]['n_estimators'])
    elif model_name=='DT':
        model = tree.DecisionTreeRegressor(max_depth=best_params[model_name]['max_depth'])
    elif model_name=='ET':
        model = ExtraTreesRegressor(n_jobs=-1,max_depth=best_params[model_name]['max_depth'],
                                    n_estimators=best_params[model_name]['n_estimators'])
    elif model_name=='GB':
        model = GradientBoostingRegressor(n_estimators=best_params[model_name]['n_estimators'])
    elif model_name=='KNR':
        model = KNeighborsRegressor(n_neighbors=best_params[model_name]['n_neighbors'])
    elif model_name=='KRR':
        model = KernelRidge(gamma=best_params[model_name]['gamma'])
    elif model_name=='LSVR':
        model = LinearSVR(epsilon=best_params[model_name]['epsilon'])
    elif model_name=='RF':
        model = RandomForestRegressor(n_jobs=-1,max_depth=best_params[model_name]['max_depth'],
                                        n_estimators=best_params[model_name]['n_estimators'])
    elif model_name=='Ridge':
        model = linear_model.Ridge(alpha=best_params[model_name]['alpha'])
    elif model_name=='SVR':
        model = SVR(kernel=best_params[model_name]['kernel'],gamma=best_params[model_name]['gamma'])
    elif model_name=='XGB':
        model = xgb.XGBRegressor(n_jobs=-1,max_depth=best_params[model_name]['max_depth'])
    return model
    
def circle(des,lable,model,des_len,tem_des_sel,n):
    all_pearsr = []
    try_index = np.delete(np.array(range(des_len)),tem_des_sel)
    flag = True
    for tem_des_index in try_index:
        tem_des_sel_ = tem_des_sel+[tem_des_index]
        tem_des = des[:,tem_des_sel_]
        desc = tem_des[:]
        pearsr = []
        mae = []
        np.random.seed(2023)
        for i in range(n):
            all_pred = []
            all_test = []
            for train_index_tep,test_index_tep in kfold.split(desc):
                train_x,test_x = desc[train_index_tep],desc[test_index_tep]
                train_y,test_y = lable[train_index_tep],lable[test_index_tep]
                model.fit(train_x,train_y)
                test_pred = model.predict(test_x)
                all_pred.append(test_pred)
                all_test.append(test_y)
            all_pred = np.concatenate(all_pred)
            all_test = np.concatenate(all_test)
            pearsr.append(pearsonr(all_test,all_pred)[0])

        pearsr = np.mean(np.array(pearsr))
        all_pearsr.append([pearsr])
    
    best_sel = np.argmax(np.array(all_pearsr))
    max_pear = all_pearsr[best_sel]
    tem_des_sel_max = tem_des_sel+[try_index[np.argmax(all_pearsr)]]
    return max_pear,tem_des_sel_max

class feature_selection:
    def __init__(self,model_name,input_index,des_std,yield_std,best_params):
        des_sel_max=des_std.shape[1]
        tem_des_sel=[]
        selected_feature = []
        i = 1
        model = get_model(model_name,best_params,random_seed=2023)
        max_pear,tem_des_sel_max=circle(des_std[input_index],lable=yield_std,
                                        model=model,des_len=des_sel_max,
                                        tem_des_sel=tem_des_sel,n=20)
        pear = max_pear
        selected_feature = tem_des_sel_max

        i += 1
        while True:
            tem_des_sel=tem_des_sel_max
            max_pear,tem_des_sel_max=circle(des_std[input_index],lable=yield_std,
                                            model= model,des_len=des_sel_max,
                                            tem_des_sel=tem_des_sel,n=20)
            if max_pear[0]>pear:
                pear = max_pear
                selected_feature = tem_des_sel_max
            if len(tem_des_sel_max)==des_sel_max:
                break
            else:
                i += 1
                continue        
        print(selected_feature)
        self.selected_feature = selected_feature
        self.pear = pear[0]