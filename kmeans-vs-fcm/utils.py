'''评价指标函数封装
1. 轮廓系数(Sihouette Coefficient)
2. Calinski-Harabasz 指数 
3. FMI 指数
4. 运行时间
5. 收敛速度
'''
from sklearn.metrics import silhouette_score, calinski_harabasz_score, fowlkes_mallows_score
import time

def evaluate_clusting(X, labels, y_true, model):
    metrics = {}

    # 1. 轮廓系数
    silhouette = silhouette_score(X, labels)

    # 2. Calinski-Harabasz 指数
    calinski_harabasz = calinski_harabasz_score(X, labels)

    # 3. FMI 指数
    fmi = fowlkes_mallows_score(y_true, labels)

    # 4. 运行时间
    start_time = time.time()
    model.fit(X)
    end_time = time.time()
    run_time = end_time - start_time

    # 5. 收敛速度
    num_iterations = model.max_iter

    metrics['silhouette_Coefficient'] = silhouette
    metrics['calinski_harabasz_score'] = calinski_harabasz
    metrics['fmi_score'] = fmi
    metrics['run_time'] = run_time
    metrics['num_iterations'] = num_iterations
    
    return metrics