import numpy as np
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
        np.random.seed(36)
        for i in range(n):
            all_pred = []
            all_test = []
            for train_index_tep,test_index_tep in kfold.split(desc):
                train_x,test_x = desc[train_index_tep],desc[test_index_tep]
                train_y,test_y = lable[train_index_tep],lable[test_index_tep]
                model = model
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
    def __init__(self,input_index,des_std,yield_std):
        des_sel_max=41
        tem_des_sel=[]
        pear = -10
        selected_feature = []
        i = 1
        model = SVR(C=0.13,epsilon=0.06)
        max_pear,tem_des_sel_max=circle(des_std[input_index],lable=yield_std,
                                        model=model,des_len=des_sel_max,
                                        tem_des_sel=tem_des_sel,n=20)
        if max_pear[0]>pear:
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
        self.selected_feature = selected_feature
        self.pear = pear[0]