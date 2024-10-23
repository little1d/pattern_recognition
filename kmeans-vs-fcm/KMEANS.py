'''基于 scikit-learn 的 KMeans 算法，保持与 FCM 的接口操作一致'''
from sklearn.cluster import KMeans
import numpy as np

class KMEANS:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """
        KMEANS 类封装了 sklearn 的 KMeans 聚类算法，并提供与 FCM 类类似的接口。
        :param n_clusters: 簇的数量
        :param max_iter: 最大迭代次数
        :param tol: 收敛容差
        :param random_state: 随机种子
        """
        self.n_clusters = n_clusters  # 簇的数量
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛容差
        self.random_state = random_state  # 随机种子
        self.model = KMeans(n_clusters=n_clusters, max_iter=max_iter, tol=tol, random_state=random_state)
        self.cluster_centers_ = None  # 保存簇中心

    def fit(self, X):
        """
        拟合 K-means 模型，计算簇中心。
        :param X: 输入数据
        :return: self
        """
        self.model.fit(X)
        self.cluster_centers_ = self.model.cluster_centers_
        return self

    def predict(self, X):
        """
        对输入数据进行聚类预测。
        :param X: 输入数据
        :return: 聚类标签
        """
        return self.model.predict(X)

