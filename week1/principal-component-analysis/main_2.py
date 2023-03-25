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
    sample1 = data["Sample1"]
    sample1_mat = sample1.to_numpy()
    cov_x = 0

    """ TODO:
    Part 2.1:
        - implement PCA
    """

    # project the data onto the new feature space
    # X_pca =

    # variances
    # var =

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

    # tryout
    # data is the variable for the data structure
    # sample1 = data["Sample1"]
    keys = ["Sample1", "Sample2", "Sample3", "Sample4", "Sample5", "Sample6",
            "Sample7", "Sample8", "Sample9", "Sample10", "Sample11", "Sample12"]
    print(keys[0])
    # time = sample1[keys[0]]
    # amp = sample1[keys[1]]

    # plt.title("Random Plot")
    # plt.scatter(time, amp)
    # plt.savefig("Random_Plot.png")

    """ TODO:
    Part 2.2:
        - standardization of the data
        - apply PCA (eigenvectors & eigenvalues) to one sample
    """

    """ TODO:
    Part 2.3:
        - how many principal components are needed in order to cover 90% of the variance
            for the selected sample?
    """

    """ TODO:
    Part 2.4:
        - compute the total variance covered with k components for each sample
    """

    """ TODO:
    Part 2.5:
        - Use the pca function available in {sklearn} to validate your results
    """
