import numpy as np
import matplotlib.pyplot as plt
import time
from datasets import load_dataset

ds = load_dataset('ylecun/mnist', split = "train")
ds = ds.train_test_split(test_size = 0.2, shuffle=True)

train_data = []
train_label = []
test_data = []
test_label = []
for i in range(len(ds['train'])):
    train_data.append(ds['train'][i]['image'])
    train_label.append(ds['train'][i]['label'])

for i in range(len(ds['test'])):
    test_data.append(ds['test'][i]['image'])
    test_label.append(ds['test'][i]['label'])


def image_show(i, data, label):
    x = data[i] # get the vectorized image
    x = np.asarray(x)
    x = x.reshape((28,28)) # reshape it into 28x28 format
    print('The image label of index %d is %d.' %(i, label[i]))
    plt.imshow(x, cmap='gray') # show the image


    # KNN Algos
# L2 square distance between two vectorized images x and y
def distance1(x,y):
    return np.sum(np.square(x-y))
# L2 distance between two vectorized images x and y
def distance2(x,y):
    return np.sqrt(np.sum(np.square(x-y)))
# and can be coded as below
def distance3(x,y):
    return np.linalg.norm(x-y)

def kNN(x, k, data, label):
    data = np.asarray(data)
    label = np.asarray(label)
    x = np.asarray(x)
    #create a list of distances between the given image and the images of the training set
#     distances =[np.linalg.norm(x-data[i]) for i in range(len(data))]
    distances =[distance1(x,data[i]) for i in range(len(data))]
    #Use "np.argpartition". It does not sort the entire array. 
    #It only guarantees that the kth element is in sorted position 
    # and all smaller elements will be moved before it. 
    # Thus the first k elements will be the k-smallest elements.
    idx = np.argpartition(distances, k).astype(np.int64)
    clas, freq = np.unique(label[idx[:k]], return_counts=True)
    return clas[np.argmax(freq)]


def accuracy_set(data, label, train_data, train_label, k):
    cnt = 0
    for x, lab in zip(data,label):
        if kNN(x,k, train_data, train_label) == lab:
            cnt += 1
    return cnt/len(label)

time_start = time.time()
k_acc = [accuracy_set(test_data, test_label, train_data, train_label, k) for k in range(1,10)]
time_end = time.time()
print("Time cost:", time_end - time_start)
print("The accuracy of kNN with different k values is:", k_acc)

X = [k for k in range(1,10)]
plt.figure(figsize = (10,5))
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.plot(X,k_acc)
plt.savefig("k_acc.png")