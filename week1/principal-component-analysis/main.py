#!/usr/bin/env python3

# internal support
from ImportData import load_table_from_struct

# external support
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA


def pca_scratch(X_stand, k):
    """
    X_stand: standardized data
    k: number of principal components
    """

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
    print(data)

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
