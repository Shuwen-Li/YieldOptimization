def round2stage(n_round):
    if n_round<3:
        stage = 1
    elif n_round<11:
        stage = 2
    else:
        stage = 3
    return stage

def get_orthogonal_selection(n_round,input_data,sorted_pre_yield):
    ort_data=[]
    tmp_data = input_data[:]
    stage = round2stage(n_round)
    if stage==1:
        experiment_num = 4
        threshold = 864
        ort_num = 1
    elif stage==2:
        experiment_num = 4
        threshold = 100
        ort_num = 2
    elif stage==3:
        experiment_num = 20
        threshold = 20
        ort_num = 3
    
    tmp_experiment_num = 0
    for i in sorted_pre_yield:
        m = 0
        for j in tmp_data:
            repeat_num=len(set(i[1:]) & set(j))
            if repeat_num>ort_num:
                m = 1
        if m==0:
            if i[0]<=threshold and tmp_experiment_num<experiment_num:
                ort_data.append(i)
                tmp_experiment_num += 1
                tmp_data += [i]
            else:
                break
    return ort_data