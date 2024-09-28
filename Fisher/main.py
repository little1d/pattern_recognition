import argparse
from utils import *
import matplotlib.pyplot as plt 
import numpy as np 


def main():
    parser = argparse.ArgumentParser(description='Fisher')
    parser.add_argument('--dataset', choices=['iris','sonar'], help='Dataset to use: iris or sonar')
    parser.add_argument('--method', choices=['random', 'kfold', 'leaveoneout'], help='Method to use: random, kfold, or leaveoneout')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test size for random split (default: 0.3)')
    parser.add_argument('--k', type=int, default=10, help='Number of folds for kfold split (default: 10)')

    args = parser.parse_args()

    # iris
    if args.dataset == 'iris':
        X, y = load_data('iris')
        if args.method == 'random':
            X_train, X_test, y_train, y_test = random_split(X, y, test_size=args.test_size)
        elif args.method == 'kfold':
            X_train, X_test, y_train, y_test = k_fold_cross_validation(X, y, n_splits=args.k)
        elif args.method == 'leaveoneout':
            X_train, X_test, y_train, y_test = leave_one_out_cross_validation(X, y)
        else:
            raise ValueError('Invalid method under iris dataset')
        # 根据标签值将训练集和测试集分别划分为三个子集，分别表示三次实验的数据集
        X_train_1 = X_train[y_train != 2]
        y_train_1 = y_train[y_train != 2]
        X_test_1 = X_test[y_test != 2]
        y_test_1 = y_test[y_test != 2]
        X_train_2 = X_train[y_train != 1]
        y_train_2 = y_train[y_train != 1]
        X_test_2 = X_test[y_test!= 1]
        y_test_2 = y_test[y_test!= 1]
        X_train_3 = X_train[y_train != 0]
        y_train_3 = y_train[y_train != 0]
        X_test_3 = X_test[y_test != 0]
        y_test_3 = y_test[y_test != 0]

        # 第一类和第二类
        w1, u1, u2 = fisher_discriminant_analysis(X_train_1, y_train_1)
        y_pred = np.zeros(len(y_test_1))
        for i in range(len(y_test_1)):
            y_pred[i] = judge_sample(X_test_1[i,:], w1, u1, u2)
        count1=0
        for i in range(len(y_test_1)):
            if y_pred[i] == y_test_1[i]:
                count1+=1
        accuracy1 = count1 / len(y_test_1)
        print(f"accuracy for class 1 and 2 : {accuracy1 * 100:.2f}%")
        # 第一类和第三类
        w2, _, u3 = fisher_discriminant_analysis(X_train_2, y_train_2)
        y_pred = np.zeros(len(y_test_2))
        for i in range(len(y_test_2)):
            y_pred[i] = judge_sample(X_test_2[i,:], w2, u1, u3)
        count2=0
        for i in range(len(y_test_2)):
            if y_pred[i] == y_test_2[i]:
                count2+=1
        accuracy2 = count2 / len(y_test_2)
        print(f"accuracy for class 1 and 3 : {accuracy2 * 100:.2f}%")
        # 第二类和第三类
        w3, _, u4 = fisher_discriminant_analysis(X_train_3, y_train_3)
        y_pred = np.zeros(len(y_test_3))
        for i in range(len(y_test_3)):
            y_pred[i] = judge_sample(X_test_3[i,:], w3, u2, u4)
        count3=0
        for i in range(len(y_test_3)):
            if y_pred[i] == y_test_3[i]:
                count3+=1
        accuracy3 = count3 / len(y_test_3)
        print(f"accuracy for class 2 and 3 : {accuracy3 * 100:.2f}%")        
    # sonar
    elif args.dataset =='sonar':
        X, y = load_data('sonar')
        if args.method == 'random':
            X_train, X_test, y_train, y_test = random_split(X, y, test_size=args.test_size)
        elif args.method == 'kfold':
            X_train, X_test, y_train, y_test = k_fold_cross_validation(X, y, n_splits=args.k)
        elif args.method == 'leaveoneout':
            X_train, X_test, y_train, y_test = leave_one_out_cross_validation(X, y)
        else:
            raise ValueError('Invalid method under sonar dataset')
        # calculate fisher W
        W ,u1, u2 = fisher_discriminant_analysis(X_train, y_train)
        y_pred = np.zeros(len(y_test))
        for i in range(len(y_test)):
            y_pred[i] = judge_sample(X_test[i,:], W, u1, u2)
        # print("len(y_test):", len(y_test))
        count=0
        for i in range(len(y_test)):
            if y_pred[i] == y_test[i]:
                count+=1
        # print(count)
        accuracy = count / len(y_test)
        print(f"accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
