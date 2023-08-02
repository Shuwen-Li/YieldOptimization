import numpy as np
import warnings
warnings.filterwarnings("ignore")
from script.load_dataset import chemical_space

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
Chemical_space = chemical_space()
all_data = Chemical_space.list_chemical_space

def get_sorted_pre_yield(model_name,input_index,des_std,yield_std,selected_feature):
    des_sel = des_std[:,selected_feature]
    test_index = np.delete(np.array(range(8640)),input_index)
    train_x,test_x = des_sel[input_index],des_sel[test_index]
    train_y = yield_std
    model = models[model_names.index(model_name)]
    model.fit(train_x,train_y)
    test_pred = model.predict(test_x)
    test_sort_index=sorted([[i,j] for i,j in zip(test_index,test_pred)],\
                           key=lambda x:x[1],reverse=True)
    result = []
    for index,i in enumerate(test_sort_index):
        result.append([index+1]+all_data[i[0]])
    return result
