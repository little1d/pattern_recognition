import numpy as np
import time
from sklearn.metrics import silhouette_score, calinski_harabasz_score, fowlkes_mallows_score
from scipy.spatial.distance import cdist

class FCM:
    def __init__(self, n_clusters=3, m=2.0, max_iter=300, tol=1e-4, random_state=None, init_center=None):
        self.n_clusters = n_clusters  # 聚类数量
        self.m = m  # 模糊指数
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛容差
        self.random_state = random_state  # 随机种子
        self.U = None  # 隶属度矩阵
        self.init_center = init_center  # 初始化簇中心
        self.cluster_centers_ = None  # 簇中心

    def _initialize_membership(self, X):
        np.random.seed(self.random_state)
        U = np.random.dirichlet(np.ones(self.n_clusters), size=len(X))
        return U

    def _update_cluster_centers(self, X):
        um = self.U ** self.m
        return (um.T @ X) / np.sum(um.T, axis=1, keepdims=True)

    def _update_membership(self, X, centers):
        dist = cdist(X, centers, metric='euclidean')
        dist = np.fmax(dist, np.finfo(np.float64).eps)  # 防止除以零
        inv_dist = 1.0 / dist
        U_new = inv_dist ** (2 / (self.m - 1))
        return U_new / np.sum(U_new, axis=1, keepdims=True)

    def fit(self, X):
        n_samples = len(X)
        if self.init_center is not None:
            # 使用 KMeans 生成的初始簇中心
            self.cluster_centers_ = self.init_center
            self.U = self._update_membership(X, self.cluster_centers_)
        else:
            # 随机初始化隶属度矩阵
            self.U = self._initialize_membership(X)

        for i in range(self.max_iter):
            centers = self._update_cluster_centers(X)
            U_new = self._update_membership(X, centers)
            if np.linalg.norm(self.U - U_new) < self.tol:
                break
            self.U = U_new
        self.cluster_centers_ = centers
        return self

    def predict(self, X):
        dist = cdist(X, self.cluster_centers_, metric='euclidean')
        return np.argmin(dist, axis=1)
