import pandas as pd 
import numpy as np
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