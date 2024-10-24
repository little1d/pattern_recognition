import numpy as np
import matplotlib.pyplot as plt
import time
from datasets import load_dataset

ds = load_dataset("scikit-learn/iris", split="train")
ds = ds.train_test_split(test_size = 0.3, shuffle=True)

X_train = [(sample['SepalLengthCm'], sample['SepalWidthCm'], sample['PetalLengthCm'], sample['PetalWidthCm']) for sample in ds['train']]
y_train = [sample['Species'] for sample in ds['train']]

X_test = [(sample['SepalLengthCm'], sample['SepalWidthCm'], sample['PetalLengthCm'], sample['PetalWidthCm']) for sample in ds['test']]
y_test = [sample['Species'] for sample in ds['test']]

# KNN Algos
# 欧几里得距离的平方
def distance1(x,y):
    return np.sum(np.square(x-y))

# 标准欧几里得距离
def distance2(x,y):
    return np.sqrt(np.sum(np.square(x-y)))

# 欧几里得距离
def distance3(x,y):
    return np.linalg.norm(x-y)

# 曼哈顿距离/L1 范数/城市街区距离
def distance4(x,y):
    return np.sum(np.abs(x-y))

def kNN(x, k, data, label):
    data = np.asarray(data)
    label = np.asarray(label)
    x = np.asarray(x)
    distances =[distance1(x,data[i]) for i in range(len(data))]
    idx = np.argpartition(distances, k).astype(np.int64)
    clas, freq = np.unique(label[idx[:k]], return_counts=True)
    return clas[np.argmax(freq)]

# 定义函数计算准确率、误差方差
def accuracy_and_variance_set(data, label, train_data, train_label, k):
    cnt = 0
    errors = []
    for x, lab in zip(data, label):
        predicted_label = kNN(x, k, train_data, train_label)
        if predicted_label == lab:
            cnt += 1
            errors.append(0)  # 正确分类，误差为 0
        else:
            errors.append(1)  # 错误分类，误差为 1
    accuracy = cnt / len(label)
    variance = np.var(errors)  # 计算误差的方差
    return accuracy, variance

# 记录不同 k 值下的准确率和方差
k_values = range(1, 100)
accuracies = []
variances = []
time_start = time.time()

for k in k_values:
    acc, var = accuracy_and_variance_set(X_test, y_test, X_train, y_train, k)
    accuracies.append(acc)
    variances.append(var)

time_end = time.time()
print("Time cost:", time_end - time_start)

# 绘制准确率和方差图
plt.figure(figsize=(12, 6))

# 绘制准确率曲线
plt.subplot(1, 2, 1)
plt.plot(k_values, accuracies, label="Accuracy", color="blue")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.title("Accuracy vs k")

# 绘制方差曲线
plt.subplot(1, 2, 2)
plt.plot(k_values, variances, label="Variance", color="red")
plt.xlabel("k")
plt.ylabel("Variance")
plt.title("Variance vs k")

# 在图像上添加时间消耗信息
time_cost_text = f"Time cost: {time_end - time_start:.2f} seconds"
plt.gcf().text(0.5, 0.95, time_cost_text, fontsize=12, ha='center')

# 保存图像
plt.tight_layout()
plt.savefig("iris_knn_accuracy_variance.png")
plt.show()