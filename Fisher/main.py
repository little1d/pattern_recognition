import argparse
from utils import *
import matplotlib.pyplot as plt 
import numpy as np 


def main():
    parser = argparse.ArgumentParser(description='Fisher')
    parser.add_argument('dataset', choices=['iris','sonar'], help='Dataset to use: iris or sonar')
    parser.add_argument('method', choices=['random', 'kfold', 'leaveoneout'], help='Method to use: random, kfold, or leaveoneout')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test size for random split (default: 0.3)')
    parser.add_argument('--k', type=int, default=10, help='Number of folds for kfold split (default: 10)')

    args = parser.parse_args()

    # iris
    if args.dataset == 'iris':
        pass
        
    # sonar
    elif args.dataset =='sonar':
        X, y = load_data('sonar')
        y_c = np.unique(y)
        y = np.array(y)
        for i in range(len(y_c)):
            y[y == y_c[i]] = i
        y = y.astype(int)
        if args.method == 'random':
            X_train, X_test, y_train, y_test = random_split(X, y, test_size=args.test_size)
        elif args.method == 'kfold':
            X_train, X_test, y_train, y_test = k_fold_cross_validation(X, y, n_splits=args.k)
        elif args.method == 'leaveoneout':
            X_train, X_test, y_train, y_test = leave_one_out_cross_validation(X, y)
        else:
            raise ValueError('Invalid method under sonar dataset')
        # calculate fisher W
        W = fisher_discriminant_analysis(X_train, y_train)
        index1 = np.where(y_train == 0)[0] # 获取类别为'0'的 index
        index2 = np.where(y_train == 1)[0] # 获取类别为'1'的 index
        _, u1 = cal_cov_and_avg(X_train[index1])
        _, u2 = cal_cov_and_avg(X_train[index2])
        y_pred = np.zeros(len(y_test))
        for i in range(len(y_test)):
            y_pred[i] = judge_sample(X_test[i,:], W, u1, u2)
        print("len(y_test):", len(y_test))
        count=0
        for i in range(len(y_test)):
            print(f"y_pred[i]:{y_pred[i]}, y_test[i]:{y_test[i]}")
            print(y_pred[i] == y_test[i])
            if y_pred[i] == y_test[i]:
                count+=1
        print("count:", count)
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        print("accuracy:", accuracy)

if __name__ == '__main__':
    main()