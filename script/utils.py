import pandas as pd 
import numpy as np
import seaborn as sns
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
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

models = [BaggingRegressor(n_jobs=60),
          tree.DecisionTreeRegressor(),ExtraTreesRegressor(n_jobs=60),GradientBoostingRegressor(),
          KNeighborsRegressor(),KernelRidge(),
          LinearSVR(),RandomForestRegressor(n_jobs=60),
          linear_model.Ridge(alpha=.5),SVR(), xgb.XGBRegressor(n_jobs=60)]
model_names = ['BG','DT','ET','GB','KNR','KRR','LSVR','RF','Ridge','SVR','XGB']


def format_output(condition):
    condition = np.array(condition)
    dic_condition = {}
    dic_condition['rank']=list(condition[:,0])
    dic_condition['Anode/Cathode']=list(condition[:,1])
    dic_condition['Solvent']=list(condition[:,2])
    dic_condition['Electrolyte']=list(condition[:,3])
    dic_condition['Current/Potential']=list(condition[:,4])
    df = pd.DataFrame(dic_condition)
    return df


def get_ori_independence(des_dir,des_name):
    if des_name=='cp':
        array = np.array(des_dir)[:,1:]
        name=['0.3 mA','0.6 mA','0.9 mA','1.2 mA','1.0 V','1.5 V']
    else:
        array = np.array(pd.read_excel(des_dir))[:,1:]
        name=pd.read_excel(des_dir).columns[1:]
    array_std=f_des_std(array)
    
    relation=[]
    for i in range(array_std.shape[1]):
        tem_relation=[]
        for j in range(array_std.shape[1]):
            tem_relation.append(abs(pearsonr(array_std[:,i],array_std[:,j])[0]))
            #tem_relation.append(r2_score(array_std[:,i],array_std[:,j]))
        relation.append(tem_relation)
    relation=np.array(relation)
    sns.heatmap(relation,cmap='Blues', xticklabels=[], yticklabels=[])
    return relation,name

def f_des_std(des_array):
    react_feat_all = des_array[:,des_array.max(axis=0)!=des_array.min(axis=0)]
    react_feat_all = (react_feat_all-react_feat_all.min(axis=0))/\
    (react_feat_all.max(axis=0)-react_feat_all.min(axis=0))
    react_feat_all[np.where(np.isnan(react_feat_all)==True)]=0
    return react_feat_all

def get_mix_sol_dict(pure_des_dict,sol_set,name2smiles_dict):
    sol_all_dict_mix={}
    for i in sol_set:
        if ':' in i:
            name1 = i.split(':')[0]
            name2 = i.split(':')[1].split('(')[0].split(' ')[1]
            des1=pure_des_dict[name2smiles_dict[name1]]
            des2=pure_des_dict[name2smiles_dict[name2]]
            sol_all_dict_mix[i]=des1+des2
        else:
            sol_all_dict_mix[i]=pure_des_dict[name2smiles_dict[i]]+pure_des_dict[name2smiles_dict[i]]
    return sol_all_dict_mix

def plot_scatter(performance_dict,model_des_name):    
    y_val = performance_dict[model_des_name][-1]
    y_pred=performance_dict[model_des_name][-2]

    sns.set(style='darkgrid')
    fig = plt.figure(figsize=(11,11),facecolor='white',    
               edgecolor='black')
    plt.scatter(y_val,y_pred,s=250, c='royalblue', label="samples",alpha=0.6,edgecolors='navy')#royalblue
    plt.plot([-20,120],[-20,120],c='black')
    plt.xlim(-10,80)
    plt.ylim(-10,80)
    x_major_locator=MultipleLocator(20)
    y_major_locator=MultipleLocator(20)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlabel("Observed Yield(%)",fontsize=30)
    plt.ylabel("Predicted Yield(%)",fontsize=30)
    plt.tick_params(labelsize=30)
    plt.text(2,62,'Pearson R = %.3f'%pearsonr(y_val,y_pred)[0],fontsize=40)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1)
    plt.show()

def get_sorted_pre_yield_ep(ep_space,model_name,input_index,des_std,labels_std,selected_feature):

    des_sel = des_std[:,selected_feature]
    test_index = np.delete(np.array(range(len(labels_std)+ep_space.shape[0])),input_index)
    train_x,test_x = des_sel[input_index].reshape(len(labels_std),-1),des_sel[test_index].reshape(ep_space.shape[0],-1)
    train_y = labels_std
    model = models[model_names.index(model_name)]
    model.fit(train_x,train_y)
    test_pred = model.predict(test_x)
    test_sort_index=sorted([[i,j] for i,j in zip(test_index,test_pred)],\
                           key=lambda x:x[1],reverse=True)
    result = []
    for index,i in enumerate(test_sort_index):
        result.append([index+1]+list(ep_space[i[0]-len(labels_std)]))
    return result,test_sort_index
