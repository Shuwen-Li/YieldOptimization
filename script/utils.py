import pandas as pd 
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

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
    return react_feat_all