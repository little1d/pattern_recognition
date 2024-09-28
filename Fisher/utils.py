import numpy as np
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut

# 计算给定类别数据的协方差和平均向量
def cal_cov_and_avg(samples):
    mu = np.mean(samples, axis=0)
    cov_m = np.zeros((samples.shape[1], samples.shape[1]))
    for i in range(samples.shape[1]):
        s = samples[i,:]
        t = (s - mu)
        cov_m += t*t.T
    return cov_m, mu

# 定义 Fisher 判别分析函数
def fisher_discriminant_analysis(X, y):
    index1 = np.where(y == 0)[0] # 获取类别为'0'的 index
    index2 = np.where(y == 1)[0] # 获取类别为'1'的 index
    con_1, u1 = cal_cov_and_avg(X[index1])
    con_2, u2 = cal_cov_and_avg(X[index2])
    S_W = con_1 + con_2
    u, s, v = np.linalg.svd(S_W)
    # 投影方向向量wß
    S_W_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
    W = np.dot(S_W_inv,u1 - u2)
    return W

def judge_sample(sample, w, u1, u2):
    # sample 属于 c1 返回 False，属于 c2 返回 True
    center_1 = np.dot(w.T, u1)
    center_2 = np.dot(w.T, u2)
    pos = np.dot(w.T, sample)
    return abs(pos - center_1) > abs(pos - center_2)

def load_data(dataset):
    try:
        if dataset == 'sonar':
            data = fetch_ucirepo(id=151)
        elif dataset == 'iris':
            data = fetch_ucirepo(id=53)
    except Exception as e:
        raise ValueError(f"Error occurred while fetching data from ucimlrepo: {e}")
    X = data.data.features
    y = data.data.targets
    y_c = np.unique(y)
    for i in range(len(y_c)): 
        y[y == y_c[i]] = i
    y = y.astype(int) 
    return X, y

def random_split(X,y,test_size=0.3,random_state=42):
    X = X.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def k_fold_cross_validation(X, y, n_splits, random_state=42):
    X = X.values
    skf = StratifiedKFold(n_splits=n_splits,random_state=random_state,shuffle=True)
    skf.get_n_splits(X,y)
    for _, (train_index, test_index) in enumerate(skf.split(X,y)):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test

def leave_one_out_cross_validation(X,y):
    # dataframes to numpy arrays
    X = X.values
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    for _, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test