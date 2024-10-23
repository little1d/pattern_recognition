import numpy
import time
import numpy as np
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

from KMEANS import KMEANS
from FCM import FCM
from utils import evaluate_clusting
import pandas as pd
from datasets import load_dataset

class ExperimentRunner:
    def __init__(self, X, y, n_clusters=3, random_state=42, dataset_name=None):
        """
        初始化实验运行类
        :param X: 输入数据
        :param y: 输入标签
        :param n_cluster: 聚类数
        :param random_state: 随机种子
        """
        self.X = X
        self.y = y
        self.n_clusters = n_clusters
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.results = []

    def run_kmeans(self):
        """直接运行 KMeans 实验"""
        print("Running KMeans...")
        kmeans = KMEANS(n_clusters=self.n_clusters, max_iter=300, random_state=self.random_state)
        start_time = time.time()
        kmeans.fit(self.X)
        end_time = time.time()

        labels = kmeans.predict(self.X)
        metrics = evaluate_clusting(self.X, labels, self.y, kmeans)
        metrics['run_time'] = end_time - start_time
        metrics['method'] = 'KMeans'
        self.results.append(metrics)
        return kmeans.cluster_centers_
    
    def run_fcm(self):
        """直接运行 FCM 实验"""
        print("Running FCM...")
        fcm = FCM(n_clusters=self.n_clusters, max_iter=300, random_state=self.random_state)
        start_time = time.time()
        fcm.fit(self.X)
        end_time = time.time()
        
        labels = fcm.predict(self.X)
        metrics = evaluate_clusting(self.X, labels, self.y, fcm)
        metrics['method'] = 'FCM'
        metrics['run_time'] = end_time - start_time
        self.results.append(metrics)
        return fcm.cluster_centers_
    
    def run_kmeans_then_fcm(self, kmeans_centers):
        """KMeans 初始中心，然后使用 FCM"""
        print("Running KMeans -> FCM...")
        fcm = FCM(n_clusters=self.n_clusters, max_iter=300, random_state=self.random_state, init_center=kmeans_centers)
        start_time = time.time()
        fcm.fit(self.X)
        end_time = time.time()

        labels = fcm.predict(self.X)
        metrics = evaluate_clusting(self.X, labels, self.y, fcm)
        metrics['method'] = 'KMeans -> FCM'
        metrics['run_time'] = end_time - start_time
        self.results.append(metrics)

    def run_fcm_then_kmeans(self, fcm_centers):
        """FCM 初始中心，然后使用 KMeans"""
        print("Running FCM -> KMeans...")
        kmeans = KMEANS(n_clusters=self.n_clusters, max_iter=300, random_state=self.random_state)
        kmeans.model.init = fcm_centers  # 使用 FCM 的簇中心作为 KMeans 初始中心
        start_time = time.time()
        kmeans.fit(self.X)
        end_time = time.time()

        labels = kmeans.predict(self.X)
        metrics = evaluate_clusting(self.X, labels, self.y, kmeans)
        metrics['method'] = 'FCM -> KMeans'
        metrics['run_time'] = end_time - start_time
        self.results.append(metrics)

    def run_fcm_then_fcm(self, fcm_centers):
        """FCM 初始中心，然后继续使用 FCM"""
        print("Running FCM -> FCM...")
        fcm = FCM(n_clusters=self.n_clusters, max_iter=300, random_state=self.random_state, init_center=fcm_centers)
        start_time = time.time()
        fcm.fit(self.X)
        end_time = time.time()

        labels = fcm.predict(self.X)
        metrics = evaluate_clusting(self.X, labels, self.y, fcm)
        metrics['method'] = 'FCM -> FCM'
        metrics['run_time'] = end_time - start_time
        self.results.append(metrics)

    def save_results(self):
        """保存结果到文件中"""
        filename = self.dataset_name + '_results.csv'
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

# iris
ds = load_dataset("scikit-learn/iris", split="train")
ds = ds.train_test_split(test_size=0.2, shuffle=True)
X = [(sample['SepalLengthCm'], sample['SepalWidthCm'], sample['PetalLengthCm'], sample['PetalWidthCm']) for sample in ds['train']]
y = [sample['Species'] for sample in ds['train']]
y = labelencoder.fit_transform(y)

experiment_runner = ExperimentRunner(X, y, n_clusters=3, random_state=42, dataset_name='iris')
kmeans_centers = experiment_runner.run_kmeans()
fcm_centers = experiment_runner.run_fcm()
experiment_runner.run_kmeans_then_fcm(kmeans_centers)
experiment_runner.run_fcm_then_kmeans(fcm_centers)
experiment_runner.run_fcm_then_fcm(fcm_centers)
experiment_runner.save_results()

# sonar
# sonar数据集的标签本就是 0-1 编码，无需 labelencoder 处理, mnist 同理
ds = load_dataset("mstz/sonar", split="train")
ds = ds.train_test_split(test_size=0.2, shuffle=True)
X = [list(sample.values())[:-1] for sample in ds['train']]
y = [list(sample.values())[-1] for sample in ds['train']] 
experiment_runner = ExperimentRunner(X, y, n_clusters=2, random_state=42, dataset_name='sonar')
kmeans_centers = experiment_runner.run_kmeans()
fcm_centers = experiment_runner.run_fcm()
experiment_runner.run_kmeans_then_fcm(kmeans_centers)
experiment_runner.run_fcm_then_kmeans(fcm_centers)
experiment_runner.run_fcm_then_fcm(fcm_centers)
experiment_runner.save_results()

# minist
ds = load_dataset("ylecun/mnist", split="train")
X = [np.array(sample['image']).flatten() for sample in ds]  # 展平图像为一维向量
y = [sample['label'] for sample in ds]
experiment_runner = ExperimentRunner(X, y, n_clusters=10, random_state=42, dataset_name='mnist')
kmeans_centers = experiment_runner.run_kmeans()
fcm_centers = experiment_runner.run_fcm()
experiment_runner.run_kmeans_then_fcm(kmeans_centers)
experiment_runner.run_fcm_then_kmeans(fcm_centers)
experiment_runner.run_fcm_then_fcm(fcm_centers)
experiment_runner.save_results()