import pandas as pd
import numpy as np
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
import warnings
warnings.filterwarnings("ignore")
from itertools import product
def get_domain(defined_chemical_space):
    '''
    对定义的参数排列组合，生成化学空间
    '''
    domain_list = [tmp_combine for tmp_combine in product(*[defined_chemical_space[tmp_key] for tmp_key in defined_chemical_space])]
    domain = pd.DataFrame.from_dict({tmp_category:[domain_list[i][idx] for i in range(len(domain_list))] \
                                     for idx,tmp_category in enumerate(defined_chemical_space)})
    return domain
def getdescdomain(domain,desc_map,defined_chemical_space):
    '''
    将定义的化学空间转化为对应的描述符空间
    '''
    df=pd.DataFrame([np.concatenate([desc_map[domain.iloc[i][idx]] for idx,tmp_category \
                                       in enumerate(defined_chemical_space)]) for i in range(len(domain))])
    numeric_columns = df.select_dtypes(include='number')
    new_df = pd.DataFrame(numeric_columns)
    return new_df
def random_recom(batch_size,domain,desc_domain,init_pth,random_state=None,target = 'yield'):
    '''
    随机推荐初始batch_size个反应
    '''
    np.random.seed(random_state)          ## 为了演示，限制随机数
    exp_idx = np.random.randint(0,len(domain),batch_size)
    init_react = domain.iloc[exp_idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        init_react[target] = ['<Enter the result>']*batch_size
    init_desc = desc_domain.iloc[exp_idx]
    init_react.to_csv(init_pth)
    return init_react
def exe_exp(domain_sampled,result_pth='./Data/data_cn/experiment_index.csv'):
    '''
    This function is used to simulate human execute experiment, which is unavailable in reality.
    用于模拟试验化学家试验结果，在实际应用过程中并不存在这个函数
    '''
    exp_result = pd.read_csv(result_pth)
    arha_smi = domain_sampled['Aryl_halide_SMILES'].to_list()
    add_smi = domain_sampled['Additive_SMILES'].to_list()
    base_smi = domain_sampled['Base_SMILES'].to_list()
    lig_smi = domain_sampled['Ligand_SMILES'].to_list()
    result = []
    for i in range(len(arha_smi)):
        try:
            tmp_targ = float(exp_result[(exp_result['Aryl_halide_SMILES'] == arha_smi[i]) &\
                          (exp_result['Additive_SMILES'] == add_smi[i]) &\
                          (exp_result['Base_SMILES'] == base_smi[i]) &\
                          (exp_result['Ligand_SMILES'] == lig_smi[i])]['yield'])
        except:
            tmp_targ = np.nan
        result.append(tmp_targ)
    return np.array(result,dtype=np.float32)
def exe_exp_3(domain_sampled,result_pth):
    exp_result = pd.read_csv(result_pth)
    ele_smi = domain_sampled['Electrophile_SMILES'].to_list()
    nuc_smi = domain_sampled['Nucleophile_SMILES'].to_list()
    lig_smi = domain_sampled['Ligand_SMILES'].to_list()
    base_smi = domain_sampled['Base_SMILES'].to_list()
    sol_smi = domain_sampled['Solvent_SMILES'].to_list()
    result = []
    for i in range(len(ele_smi)):
        try:
            tmp_targ = float(exp_result[(exp_result['Electrophile_SMILES'] == ele_smi[i]) &\
                          (exp_result['Nucleophile_SMILES'] == nuc_smi[i]) &\
                          (exp_result['Ligand_SMILES'] == lig_smi[i]) &\
                          (exp_result['Base_SMILES'] == base_smi[i])&\
                           (exp_result['Solvent_SMILES'] == sol_smi[i])]['yield'])
        except:
            tmp_targ = np.nan
        result.append(tmp_targ)
    return np.array(result,dtype=np.float32)
def add_result(result,new_result_pth='',new_result_pd=None):
    '''
    向结果表格中添加新的结果
    '''
    if new_result_pth != '':
        new_result = pd.read_csv(new_result_pth,index_col=0)
        return result._append(new_result).dropna(axis=0)
    else:
        return result._append(new_result_pd).dropna(axis=0)
def result2xy(desc_domain,result_pth='',result=None,scale=0.01,target = 'yield'):
    '''
    将已经有的结果转化为描述符和标签，用于训练模型
    '''
    if result_pth != '':
        result = pd.read_csv(result_pth,index_col=0)
    
    exp_idx = [int(i) for i in result.index]
    train_x = torch.tensor(desc_domain.iloc[exp_idx].to_numpy(),dtype=torch.float32)
    train_y = torch.tensor(result[target].to_numpy(),dtype=torch.float32) * scale
    if torch.cuda.is_available():
        train_x = train_x.cuda()
        train_y = train_y.cuda()
    return train_x,train_y
class Top_k():
    def __init__(self,train_x,train_y,model='RF',n_jobs=1):
        self.train_x = train_x
        self.train_y = train_y
        model = model.lower()
        assert model == 'rf' or model == 'et' or model == 'xgb', 'Only support RandomForest (RF), ExtraTrees(ET) and XGBoost(XGB) currently.'
        if model == 'rf':
            self.model = RandomForestRegressor(n_jobs=n_jobs)
        elif model == 'et':
            self.model = ExtraTreesRegressor(n_jobs=n_jobs)
        elif model == 'xgb':
            self.train_x = self.train_x.numpy()
            self.train_y = self.train_y.numpy()
            self.model = XGBRegressor(n_jobs=n_jobs)
    def recommend(self,domain,desc_domain,result,batch_size=10,target = 'yield'):
        self.model.fit(self.train_x,self.train_y)
        desc_domain_np = desc_domain.to_numpy()
        pred = self.model.predict(desc_domain_np)
        sampled_idx = []
        known_idx = [int(tmp_item) for tmp_item in result.index]
        while len(sampled_idx) < 10:
            pot_idx = pred.argmax()
            pred[pot_idx] = -1
            if not pot_idx in known_idx:
                sampled_idx.append(pot_idx)
        domain_sampled = deepcopy(domain).iloc[sampled_idx]
        domain_sampled[target] = ['<Enter the result>'] * len(domain_sampled)
        return domain_sampled
class Top_k_add_or_total_cn():
    def __init__(self,train_x,train_y,random_state,model='RF',n_jobs=-1):
        self.random_state = random_state
        self.train_x = train_x
        self.train_y = train_y
        model = model.lower()
        assert model == 'rf' or model == 'et' or model == 'xgb', 'Only support RandomForest (RF), ExtraTrees(ET) and XGBoost(XGB) currently.'
        if model == 'rf':
            self.model = RandomForestRegressor(n_jobs=-1,random_state=random_state)
        elif model == 'et':
            self.model = ExtraTreesRegressor(n_jobs=-1,random_state=random_state)
        elif model == 'xgb':
            self.train_x = self.train_x.numpy()
            self.train_y = self.train_y.numpy()
            self.model = XGBRegressor(n_jobs=-1,random_state=random_state)
    def recommend(self,domain,desc_domain,result,batch_size=10,stage=1,cc1=3,cc2=2,cc1_num=100,cc2_num=50,space_num=3696,target = 'yield'):
        np.random.seed(self.random_state)
        self.model.fit(self.train_x,self.train_y)
        desc_domain_np = desc_domain.to_numpy()
        pred = self.model.predict(desc_domain_np)
        pred_ori=self.model.predict(desc_domain_np)
        pred_sort=sorted(pred_ori,reverse=True)
        sampled_idx = []
        known_idx = [int(tmp_item) for tmp_item in result.index]
        
        if stage==1:
            num=0
            while len(sampled_idx) < batch_size and num<space_num:
                num=num+1
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                difer_min=min([count_different_columns(domain,pot_idx,i) for i in known_idx])
                if not pot_idx in known_idx and difer_min==cc1:
                    sampled_idx.append(pot_idx)
                    known_idx.append(pot_idx)
            while len(sampled_idx) < batch_size:
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                if not pot_idx in known_idx:
                    sampled_idx.append(pot_idx) 
        elif stage==2:
            num=0
            while len(sampled_idx) < batch_size and num<space_num:
                num=num+1
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                difer_min=min([count_different_columns(domain,pot_idx,i) for i in known_idx])
                if not pot_idx in known_idx and difer_min==cc2:
                    sampled_idx.append(pot_idx) 
                    known_idx.append(pot_idx)
            while len(sampled_idx) < batch_size:
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                if not pot_idx in known_idx:
                    sampled_idx.append(pot_idx) 
        elif stage==3:
            num=0
            while len(sampled_idx) < batch_size and num<space_num:
                num=num+1
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                if not pot_idx in known_idx:
                    sampled_idx.append(pot_idx)
                    known_idx.append(pot_idx)
        tem_stage=1   
        rank_last_sample = pred_sort.index(pred_ori[sampled_idx[-1]]) 
        if stage==1 and rank_last_sample < cc1_num: 
            tem_stage=1
        elif stage==1 and rank_last_sample > cc1_num:  
            tem_stage=2
        elif stage==2 and rank_last_sample < cc2_num:
            tem_stage=2
        elif stage==2 and rank_last_sample > cc2_num:
            tem_stage=3
        elif stage==3:
            tem_stage=3
        domain_sampled = deepcopy(domain).iloc[sampled_idx]
        domain_sampled[target] = ['<Enter the result>'] * len(domain_sampled)
        return domain_sampled,tem_stage
class Top_k_add_or_total_cc():
    def __init__(self,train_x,train_y,random_state,model='RF',n_jobs=8):
        self.random_state = random_state
        self.train_x = train_x
        self.train_y = train_y
        model = model.lower()
        assert model == 'rf' or model == 'et' or model == 'xgb', 'Only support RandomForest (RF), ExtraTrees(ET) and XGBoost(XGB) currently.'
        if model == 'rf':
            self.model = RandomForestRegressor(n_jobs=n_jobs,random_state=random_state)
        elif model == 'et':
            self.model = ExtraTreesRegressor(n_jobs=n_jobs,random_state=random_state)
        elif model == 'xgb':
            self.train_x = self.train_x.numpy()
            self.train_y = self.train_y.numpy()
            self.model = XGBRegressor(n_jobs=n_jobs,random_state=random_state)
    def recommend(self,domain,desc_domain,result,batch_size=10,stage=1,cc1=4,cc2=3,cc3=2,cc1_num=100,cc2_num=50,cc3_num=50,space_num=3696,target = 'yield'):
        np.random.seed(self.random_state)
        self.model.fit(self.train_x.cpu(),self.train_y.cpu())
        desc_domain_np = desc_domain.to_numpy()
        pred = self.model.predict(desc_domain_np)
        pred_ori=self.model.predict(desc_domain_np)
        pred_sort=sorted(pred_ori,reverse=True)
        sampled_idx = []
        known_idx = [int(tmp_item) for tmp_item in result.index]
        
        if stage==1:
            num=0
            while len(sampled_idx) < batch_size and num<space_num:
                num=num+1
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                difer_min=min([count_different_columns(domain,pot_idx,i) for i in known_idx])
                if not pot_idx in known_idx and difer_min==cc1:
                    sampled_idx.append(pot_idx)
                    known_idx.append(pot_idx)
            while len(sampled_idx) < batch_size:
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                if not pot_idx in known_idx:
                    sampled_idx.append(pot_idx) 
        elif stage==2:
            num=0
            while len(sampled_idx) < batch_size and num<space_num:
                num=num+1
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                difer_min=min([count_different_columns(domain,pot_idx,i) for i in known_idx])
                if not pot_idx in known_idx and difer_min==cc2:
                    sampled_idx.append(pot_idx) 
                    known_idx.append(pot_idx)
            while len(sampled_idx) < batch_size:
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                if not pot_idx in known_idx:
                    sampled_idx.append(pot_idx) 
        elif stage==3:
            num=0
            while len(sampled_idx) < batch_size and num<space_num:
                num=num+1
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                difer_min=min([count_different_columns(domain,pot_idx,i) for i in known_idx])
                if not pot_idx in known_idx and difer_min==cc3:
                    sampled_idx.append(pot_idx) 
                    known_idx.append(pot_idx)
            while len(sampled_idx) < batch_size:
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                if not pot_idx in known_idx:
                    sampled_idx.append(pot_idx) 
        elif stage==4:
            num=0
            while len(sampled_idx) < batch_size and num<space_num:
                num=num+1
                pot_idx = pred.argmax()
                pred[pot_idx] = -1
                if not pot_idx in known_idx:
                    sampled_idx.append(pot_idx)
                    known_idx.append(pot_idx)
        tem_stage=1   
        rank_last_sample = pred_sort.index(pred_ori[sampled_idx[-1]]) 
        if stage==1 and rank_last_sample < cc1_num: 
            tem_stage=1
        elif stage==1 and rank_last_sample > cc1_num:  
            tem_stage=2
        elif stage==2 and rank_last_sample < cc2_num:
            tem_stage=2
        elif stage==2 and rank_last_sample > cc2_num:
            tem_stage=3
        elif stage==3 and rank_last_sample < cc3_num:
            tem_stage=3
        elif stage==3 and rank_last_sample > cc3_num:
            tem_stage=4
        elif stage==4:
            tem_stage=4
        domain_sampled = deepcopy(domain).iloc[sampled_idx]
        domain_sampled[target] = ['<Enter the result>'] * len(domain_sampled)
        return domain_sampled,tem_stage
def count_different_columns(df, index1, index2):
    row1 = df.iloc[index1]
    row2 = df.iloc[index2]
    different_columns = 0
    
    for column in df.columns:
        if row1[column] != row2[column]:
            different_columns += 1
    
    return different_columns

def plot_figure(all_max100,c1='lightblue',c2='blue',title=''):
    plt.title(title,fontsize=15)
    plt.xlabel('Experiment Number')
    plt.ylabel('Yield')
    plt.ylim([0,100])
    plt.xlim([0,50])
    for recorded_max in all_max100:
        plt.plot([tmp_item+1 for tmp_item in  list(range(len(recorded_max)))],recorded_max,linewidth=1,c=c1,alpha=0.8)
    recorded_max_mean = np.array(all_max100).mean(axis=0)
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean)))],recorded_max_mean,linewidth=3,c=c2)
    return recorded_max_mean
def get_all_max100(results_all_cycle):
    all_max100=[]
    for cycle in results_all_cycle:

        tem_max100=[]
        for exp in range(50):
            tem_max100.append(max(cycle[:exp+1]))
        if len(tem_max100)==50:
            all_max100.append(tem_max100)
        else:
            all_max100.append(tem_max100)
            all_max100.append([tem_max100[-1]]*(50-len(tem_max100)))
            print([tem_max100[-1]]*(50-len(tem_max100)))
    all_max100=np.array(all_max100)
    recorded_max_mean = np.array(all_max100).mean(axis=0)
    num_80 = np.where(recorded_max_mean>80)[0][0] 
    num_90 = np.where(recorded_max_mean>90)[0][0]
    yield_80 = recorded_max_mean[num_80]
    yield_90 = recorded_max_mean[num_90]
    return all_max100,num_80,num_90,yield_80,yield_90

def get_all_max100_2(results_all_cycle):
    all_max100=[]
    for cycle in results_all_cycle:

        tem_max100=[]
        for exp in range(50):
            tem_max100.append(max(cycle[:exp+1]))
        if len(tem_max100)==50:
            all_max100.append(tem_max100)
        else:
            all_max100.append(tem_max100)
            all_max100.append([tem_max100[-1]]*(50-len(tem_max100)))
            print([tem_max100[-1]]*(50-len(tem_max100)))
    all_max100=np.array(all_max100)
    recorded_max_mean = np.array(all_max100).mean(axis=0)
    num_80 = np.where(recorded_max_mean>80)[0][0] 
    num_85 = np.where(recorded_max_mean>85)[0][0]
    num_90 = np.where(recorded_max_mean>90)[0][0]
    yield_80 = recorded_max_mean[num_80]
    yield_85 = recorded_max_mean[num_85]
    yield_90 = recorded_max_mean[num_90]
    return all_max100,num_80,num_85,num_90,yield_80,yield_85,yield_90
def plt_figure3(recorded_max_mean1,recorded_max_mean3):
    plt.title('Comparation',fontsize=15)
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean1)))],recorded_max_mean1,linewidth=3,c='blue')
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean3)))],recorded_max_mean3,linewidth=3,c='red')
    plt.xlabel('Experiment Number')
    plt.ylabel('Yield')
def plot_figure4(all_max100,num_80,num_90,yield_80,yield_90,c1='lightblue',c2='blue',title=''):
    plt.title(title,fontsize=15)
    plt.xlabel('Experiment Number')
    plt.ylabel('Yield')
    plt.ylim([0,100])
    plt.xlim([0,50])
    for recorded_max in all_max100:
        plt.plot([tmp_item+1 for tmp_item in  list(range(len(recorded_max)))],recorded_max,linewidth=1,c=c1,alpha=0.8)
    recorded_max_mean = np.array(all_max100).mean(axis=0)
    plt.plot([tmp_item+1 for tmp_item in list(range(len(recorded_max_mean)))],recorded_max_mean,linewidth=3,c=c2)
    #plt.plot([num_80+1,num_80+1],[0,yield_80],c='black',ls='--')
    #plt.plot([0,num_80+1],[yield_80,yield_80],c='black',ls='--')
    
    #plt.plot([num_90+1,num_90+1],[0,yield_90],c='black',ls='--')
    #plt.plot([0,num_90+1],[yield_90,yield_90],c='black',ls='--')#lightslategrey brown peru sienna
    
    #plt.text(num_80+2,1.8,num_80+1,c='black',fontsize=13)
    #plt.text(num_90+2,1.8,num_90+1,c='black',fontsize=13)
    return recorded_max_mean