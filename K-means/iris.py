from datasets import load_dataset
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


ds = load_dataset("scikit-learn/iris", split="train")

ds = ds.train_test_split(test_size=0.2, shuffle=True)

X_train = [(sample['SepalLengthCm'], sample['SepalWidthCm'], sample['PetalLengthCm'], sample['PetalWidthCm']) for sample in ds['train']]

X_test = [(sample['SepalLengthCm'], sample['SepalWidthCm'], sample['PetalLengthCm'], sample['PetalWidthCm']) for sample in ds['test']]


cluster_numbers = range(3, 10)

scores = []

for n_clusters in cluster_numbers:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_train)
    score = kmeans.score(X_test)
    print(f"KMeans with {n_clusters} clusters score: {score:.2f}")
    scores.append(score)

plt.figure(figsize=(12, 8))
plt.scatter(cluster_numbers, scores)
plt.xlabel("Number of clusters")
plt.ylabel("Score")
plt.title("KMeans Scores for Dataset iris")

plt.savefig("iris.png")

plt.show()