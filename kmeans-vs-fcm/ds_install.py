from datasets import load_dataset

add = ["scikit-learn/iris","mstz/sonar","ylecun/mnist"]


for ds in add:
    print(ds)
    data = load_dataset(ds, split="train")
    datasets = data.train_test_split(test_size=0.2, shuffle=True)