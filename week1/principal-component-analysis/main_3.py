#!/usr/bin/env python3

# internal support
from ImportData import load_table_from_struct

# external support
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA


def pca_scratch(X_stand, k):
    """
    X_stand: standardized data
    k: number of principal components
    """
    # calculate covariance matrix C
    for i in range(len(X_stand)):
        c_mat_i = np.outer(X_stand[i], X_stand[i])
        if i == 0:
            sum_c_mat = c_mat_i
        else:
            sum_c_mat = sum_c_mat + c_mat_i
    c_mat = (1 / len(X_stand)) * sum_c_mat
    # find eigenvectors of covariance matrix and sort the eigenvalues in size
    (lamb, eigvec) = np.linalg.eig(c_mat)
    for i in range(0, len(lamb)):
        for j in range(i+1, len(lamb)):
            if (lamb[i] < lamb[j]):
                temp1 = lamb[i]
                temp2 = eigvec[:, i]
                lamb[i] = lamb[j]
                eigvec[:, i] = eigvec[:, j]
                lamb[j] = temp1
                eigvec[:, j] = temp2
    # eigenvectors are already normalized
    # construct transformation matrix u
    lamb_norm = lamb / np.sum(lamb)
    u_mat = eigvec[:, :k]
    # projected data matrix z and var
    X_pca = np.matmul(X_stand, u_mat)
    var = lamb_norm[:k]
    # var = X_pca.var(0)
    # all done
    return X_pca, var


if __name__ == '__main__':

    # dataset path
    dataset_path = 'data/raw_data/Features_LW500Int500Cycle.mat'

    # report
    print('Loading dataset ' + str(dataset_path))

    # load the dataset
    data = load_table_from_struct(dataset_path)

    # report
    print('> ReMAP (composite panels) is OK and loading process is done.')

    # show me
    # print(data)

    """ TODO:
    Part 2.2:
        - standardization of the data
        - apply PCA (eigenvectors & eigenvalues) to one sample
    """
    sample1 = data["Sample1"]
    sample1_mat = sample1.to_numpy()
    s = np.std(sample1_mat, 0)
    m = np.mean(sample1_mat, 0)
    # standardize matrix
    for i in range(len(sample1_mat)):
        for j in range(len(s)):
            sample1_mat[i, j] = (sample1_mat[i, j] - m[j]) / s[j]
    sample1_stand_mat = sample1_mat
    # use your function
    (z_mat, var_z) = pca_scratch(sample1_stand_mat, 4)
    print("--------------------------------------")
    print("own PCA method")
    print(z_mat)
    print(var_z)

    """ TODO:
    Part 2.3:
        - how many principal components are needed in order to cover 90% of the variance
            for the selected sample?
    """
    for i in range(100):
        (z_mat, var_z) = pca_scratch(sample1_stand_mat, i)
        sum_variance = np.sum(var_z)
        if sum_variance >= 0.90:
            print("--------------------------------------")
            print("components needed for 90\%")
            print(i)
            break

    """ TODO:
    Part 2.4:
        - compute the total variance covered with k components for each sample
    """
    N = len(data)
    var = np.empty(N)
    kappa = 10
    keys = ["Sample1", "Sample2", "Sample3", "Sample4", "Sample5", "Sample6",
            "Sample7", "Sample8", "Sample9", "Sample10", "Sample11", "Sample12"]
    for n in range(N):
        sample = data[keys[n]]
        sample_mat = sample.to_numpy()
        s = np.std(sample_mat, 0)
        m = np.mean(sample_mat, 0)
        # standardize matrix
        for i in range(len(sample_mat)):
            for j in range(len(s)):
                sample_mat[i, j] = (sample_mat[i, j] - m[j]) / s[j]
        stand_mat = sample_mat
        # use the function
        (z_mat, var_z) = pca_scratch(stand_mat, kappa)
        sum_variance_sample = np.sum(var_z)
        var[n] = sum_variance_sample
    print("--------------------------------------")
    print("Variance list")
    print(var)

    """ TODO:
    Part 2.5:
        - Use the pca function available in {sklearn} to validate your results
    """
    # # use sklearn to check covariance, covariance is accurate till 3 decimals
    # pca = PCA(n_components=4)
    # pca.fit(sample1_stand_mat)
    # z_mat_check = pca.fit_transform(sample1_stand_mat)
    # var_check = z_mat_check.var(0)
    # # c_mat_check = pca.get_covariance()
    # # print(c_mat_check)
    # # print(c_mat)
    # print("--------------------------------------")
    # print("PCA from sklearn")
    # print(z_mat_check)
    # print(var_check)
