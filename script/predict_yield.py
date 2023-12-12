import numpy as np
import warnings
warnings.filterwarnings("ignore")
from script.load_dataset import chemical_space
from .select_feature import get_model

Chemical_space = chemical_space()
all_data = Chemical_space.list_chemical_space

def get_sorted_pre_yield(model_name,input_index,des_std,yield_std,selected_feature,best_params):
    des_sel = des_std[:,selected_feature]
    test_index = np.delete(np.array(range(8640)),input_index)
    train_x,test_x = des_sel[input_index],des_sel[test_index]
    train_y = yield_std
    model = get_model(model_name,best_params)
    model.fit(train_x,train_y)
    test_pred = model.predict(test_x)
    test_sort_index=sorted([[i,j] for i,j in zip(test_index,test_pred)],\
                           key=lambda x:x[1],reverse=True)
    result = []
    for index,i in enumerate(test_sort_index):
        result.append([index+1]+all_data[i[0]])
    return result