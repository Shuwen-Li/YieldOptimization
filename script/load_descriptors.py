import pandas as pd 
import numpy as np
import csv
import sys 
from os import sep
from script.load_dataset import chemical_space

def f_des_std(des_array):
    react_feat_all = des_array[:,des_array.max(axis=0)!=des_array.min(axis=0)]
    react_feat_all = (react_feat_all-react_feat_all.min(axis=0))/\
    (react_feat_all.max(axis=0)-react_feat_all.min(axis=0))
    return react_feat_all

def get_descriptors():
    Chemical_space = chemical_space()
    sta_em,all_em,all_sol,all_ele,all_cp = Chemical_space.sta_em,\
    Chemical_space.all_em,Chemical_space.all_sol,Chemical_space.all_ele,\
    Chemical_space.all_cp
    
    ele_df = pd.read_csv('descriptor'+sep+'descriptors of electrolyte.csv')
    ele_name = ele_df['Electrolyte'].to_list()
    op = ele_df['Onset potential/ V'].to_list()
    ts = ele_df['Tafel slope/ mV/dec'].to_list()

    ele2op = {tem_name:tem_op for tem_name,tem_op in zip(ele_name,op)}
    ele2ts = {tem_name:tem_ts for tem_name,tem_ts in zip(ele_name,ts)}
    ele_des1 = np.array([ele2op[i] for i in all_ele]).reshape(-1,1)
    ele_des2 = np.array([ele2ts[i] for i in all_ele]).reshape(-1,1)

    sol_name_des = {}
    with open('descriptor'+sep+'descriptors of solvents.csv','r') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            if i!=0:
                s = line.split(',')
                sol_name_des[s[0]] = list(map(float,s[1:]))
    sol_des = np.array([sol_name_des[i] for i in all_sol],
                       dtype='float64').reshape(-1,32)

    em2op = {'Pt':-0.02,'Fe':-0.26,'GF':-0.38,'BDD':-0.23}
    em2des = {}
    for i in sta_em:
        e1 = i.split('/')[0]
        e2 = i.split('/')[1]
        em2des[i] = [em2op[e1],em2op[e2],em2op[e2]-em2op[e1]]
    em_des = np.array([em2des[i] for i in all_em]).reshape(-1,3)

    cp2onehot = {'0.3 mA':[1,0,0,0,0,0],'0.6 mA':[0,1,0,0,0,0],
                 '0.9 mA':[0,0,1,0,0,0],'1.2 mA':[0,0,0,1,0,0],
                 '1.0 V':[0,0,0,0,1,0],'1.5 V':[0,0,0,0,0,1]}
    cp_des = np.array([cp2onehot[i] for i in all_cp]).reshape(-1,6)
    des = np.concatenate((ele_des1,ele_des2,sol_des,em_des,cp_des),axis=1)
    des_std = f_des_std(des)
    return des_std