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
    sta_em,em,sol,ele,cp = Chemical_space.sta_em,\
    Chemical_space.all_em,Chemical_space.all_sol,Chemical_space.all_ele,\
    Chemical_space.all_cp
    
    #electrolyte
    ele_df = np.array(pd.read_excel('descriptor/descriptors of electrolytes.xlsx'))

    ele_name_des={}
    for i in ele_df:
        ele_name_des[i[0]]=i[1:]
    ele_des = np.array([ele_name_des[i] for i in ele],dtype='float64')

    #solvents
    sol_df = np.array(pd.read_excel('descriptor/descriptors of solvents.xlsx'))
    sol_name_des={}
    for i in sol_df:
        sol_name_des[i[0]]=i[1:]
    sol_des = np.array([sol_name_des[i] for i in sol],dtype='float64')

    #em
    em_df = np.array(pd.read_excel('descriptor/descriptors of electrodes.xlsx'))
    em_name_des={}
    for i in em_df:
        em_name_des[i[0]]=i[1:]
    em_des = np.array([em_name_des[i] for i in em],dtype='float64')

    #cp
    cp2onehot = {'0.3 mA':[1,0,0,0,0,0],'0.6 mA':[0,1,0,0,0,0],
                 '0.9 mA':[0,0,1,0,0,0],'1.2 mA':[0,0,0,1,0,0],
                 '1.0 V':[0,0,0,0,1,0],'1.5 V':[0,0,0,0,0,1]}
    cp_des = np.array([cp2onehot[i] for i in cp]).reshape(-1,6)
    des = np.concatenate((ele_des,sol_des,em_des,cp_des),axis=1)
    des_std = f_des_std(des)
    return des_std