# -*- coding: utf-8 -*-

#######################
# Apply Data Shapely Methods to assign a value for each training datum
# And verify that Removing training data with low Shapley
# value improves the performance of the KNN regressor
# 
# The Implementation is taken from the following work: 
#   (ACL 2020) Beyond User Self-Reported Likert Scale Ratings: A Comparison Model for Automatic Dialog Evaluation 
#   Weixin Liang, James Zou and Zhou Yu
# 
#   (ACL 2021) HERALD: An Annotation Efficient Method to Train User Engagement Predictors in Dialogs 
#   Weixin Liang, Kai-Hui Liang and Zhou Yu
#######################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import os, shutil
import sys
import time
import sklearn
import pickle
import csv
from sklearn.metrics.cluster import completeness_score, homogeneity_score, v_measure_score

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score , roc_auc_score , roc_curve, classification_report, confusion_matrix
from scipy.stats import spearmanr, pearsonr


def detailed_do_knn_shapley(extract_prefix = './outputs/'):

    with open(extract_prefix + "feature_dump_train.pkl",'rb') as pkl_file:
        train_result_dict = pickle.load(pkl_file)
        X, y = train_result_dict['feature_all'], np.asarray(train_result_dict['target_all'], dtype=np.int)

    # X = np.concatenate((X, X), axis=0)
    # y = np.concatenate((y, 1-y), axis=0)

    with open(extract_prefix + "feature_dump_shapley_val.pkl",'rb') as pkl_file:
        testdev_result_dict = pickle.load(pkl_file)
        X_testdev, y_testdev = testdev_result_dict['feature_all'], np.asarray(testdev_result_dict['target_all'], dtype=np.int)
        print("X_testdev.shape", X_testdev.shape, type(X_testdev), "y_testdev.shape", y_testdev.shape, type(y_testdev))

    with open(extract_prefix + "feature_dump_val.pkl",'rb') as pkl_file:
        test_result_dict = pickle.load(pkl_file)
        X_test, y_test = test_result_dict['feature_all'], np.asarray(test_result_dict['target_all'], dtype=np.int)
        print("X_test.shape", X_test.shape, type(X_test), "y_test.shape", y_test.shape, type(y_test))

    N = X.shape[0]
    K = 10

    def single_point_shapley(xt_query, y_tdev_label):
        distance1 = np.sum(np.square(X-xt_query), axis=1)
        alpha = np.argsort(distance1)
        shapley_arr = np.zeros(N)
        for i in range(N-1, -1, -1): 
            if i == N-1:
                shapley_arr[alpha[i]] = int(y[alpha[i]] == y_tdev_label) /N
            else:
                shapley_arr[alpha[i]] = shapley_arr[alpha[i+1]] + ( int(y[alpha[i]]==y_tdev_label) - int(y[alpha[i+1]]==y_tdev_label) )/K * min(K,i+1)/(i+1)
        return shapley_arr


    global_shapley_arr = np.zeros(N)
    for x_tdev, y_tdev_label in zip(X_testdev, y_testdev):
        s1 = single_point_shapley(x_tdev, y_tdev_label)
        global_shapley_arr += s1
    global_shapley_arr /= y_testdev.shape[0]
    print("negative count:", np.sum(global_shapley_arr < 0), "all:", X.shape[0]  )

    shapley_out_dir = extract_prefix

    shapley_pkl_path = shapley_out_dir + '/shapley_value.pkl'
    with open(shapley_pkl_path, 'wb') as pkl_file:
        data_dict = {
            "global_shapley_arr": global_shapley_arr,
            "X": X,
            "y": y,
        }
        pickle.dump(data_dict, pkl_file)
    

    X_clened, y_cleaned = np.zeros((0, X.shape[1])), np.zeros(0, dtype=int)
    for i in range(y.shape[0]):
        if global_shapley_arr[i] > 0.:
            X_clened = np.concatenate([X_clened, X[[i]]])
            y_cleaned = np.concatenate([y_cleaned, y[[i]]])


    print("X_clened",X_clened.shape)
    for n_neighbors in [K]:
        neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
        neigh.fit(X_clened, y_cleaned)
        y_pred_testdev = neigh.predict(X_testdev)
        y_pred_pair = (y_pred_testdev>0.5).astype(int)
        print("n_neighbors",n_neighbors, "DEV accuracy_score", accuracy_score(y_testdev, y_pred_pair))
        print("classification_report", classification_report(y_testdev,y_pred_pair))

        y_pred_test = neigh.predict(X_test)
        y_pred_pair = (y_pred_test>0.5).astype(int)
        print("n_neighbors",n_neighbors, "TEST accuracy_score", accuracy_score(y_test, y_pred_pair))
        print("classification_report", classification_report(y_test,y_pred_pair))

if __name__ == "__main__":

    detailed_do_knn_shapley(extract_prefix = './outputs/')