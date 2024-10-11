from datasets import load_dataset

add = ["scikit-learn/iris","mstz/sonar", "uoft-cs/cifar10","ylecun/mnist","torchgeo/ucmerced" ]


for ds in add:
    print(ds)
    data = load_dataset(ds, split="train")
    datasets = data.train_test_split(test_size=0.2, shuffle=True)