import pandas as pd 
import numpy as np
import csv
from os import sep
class input_dataset:
    def __init__(self,n_round):
        round_list = [16,20,23,27,31,35,39,43,47,51,54,74]
        #input experimential dataset
        df = pd.read_csv('dataset'+sep+'all_input_data.csv')[:round_list[n_round-1]]
        em = df['Anode/Cathode'].to_list()
        sol = df['Solvent'].to_list()
        ele = df['Electrolyte'].to_list()
        cp = df['Current/Potential'].to_list()
        yield_ = df['Yield (%)'].to_list()
        yield_std = np.array(yield_)/100

        #load chemicalspace
        Chemical_space = chemical_space()
        all_data = Chemical_space.list_chemical_space
        input_data = []
        for i in range(round_list[n_round-1]):
            input_data.append([em[i]]+[sol[i]]+[ele[i]]+[cp[i]])
        input_index = []
        for j in input_data:
            input_index.append(all_data.index(j))
            
        self.em = em
        self.sol = sol
        self.ele = ele
        self.cp = cp
        self.yield_std = yield_std
        self.input_data = input_data
        self.input_data_yield = df
        self.input_index = input_index
class chemical_space:
    def __init__(self):
        #input dataset of chemical space
        all_data_df = pd.read_csv('dataset'+sep+'chemical_space.csv')
        self.all_em = all_data_df['Anode/Cathode'].to_list()
        self.all_sol = all_data_df['Solvent'].to_list()
        self.all_ele = all_data_df['Electrolyte'].to_list()
        self.all_cp = all_data_df['Current/Potential'].to_list()
        self.sta_em = list(np.unique(self.all_em))
        self.sta_sol = list(np.unique(self.all_sol))
        self.sta_ele = list(np.unique(self.all_ele))
        self.sta_cp = list(np.unique(self.all_cp))
        list_chemical_space = []
        for i in range(len(self.all_em)):
            tmp_data = [self.all_em[i],self.all_sol[i],self.all_ele[i],
                        self.all_cp[i]]
            list_chemical_space.append(tmp_data)
        self.list_chemical_space = list_chemical_space