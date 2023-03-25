#!/usr/bin/env python3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# internal support
from project_directories import pickle_dir
from utils import pickle_file
from load_dataset import load_dataset_selection


def clean(X, y):
    """
    TODO:
    Part 0, Step 2: 
        - Use the pandas {isna} and {dropna} functions to remove from the dataset any corrupted samples
    """
    samples_b = list(X.index)
    X.dropna(inplace=True)
    samples_a = list(X.index)
    drop_list = []
    for ind in samples_b:
        if ind not in samples_a:
            drop_list.append(ind)
    y.drop(drop_list, inplace=True)
    # return the cleaned data
    return [X, y]


def train_test_validation_split(X, y, test_size, cv_size):
    """
    TODO:
    Part 0, Step 3: 
        - Use the sklearn {train_test_split} function to split the dataset (and the labels) into
            train, test and cross-validation sets
    """
    train_size = 1 - test_size - cv_size
    test_size2 = test_size / (test_size + cv_size)
    X_train, X_2, y_train, y_2 = train_test_split(
        X, y, train_size=train_size, shuffle=True, random_state=0)
    X_cv, X_test, y_cv, y_test = train_test_split(
        X_2, y_2, test_size=test_size2, shuffle=True, random_state=0)
    # return split data
    return [X_train, y_train, X_test, y_test, X_cv, y_cv]


def scale(X_train, X_test, X_cv):
    """
    TODO:
    Part 0, Step 4: 
        - Use the {preprocessing.StandardScaler} of sklearn to normalize the data
        - Scale the train, test and cross-validation sets accordingly
    """
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_cv = scaler.transform(X_cv)
    # return the normalized data and the scaler
    return [X_train, X_test, X_cv, scaler]


def clean_split_scale(X, y):

    # clean data (remove NaN data points)
    [X, y] = clean(X, y)

    # split data into 90% train, 10% test, 10% cross validation
    [X_train, y_train, X_test, y_test, X_cv, y_cv] = train_test_validation_split(
        X, y, test_size=0.1, cv_size=0.1)

    # convert data and labels to numpy arrays, ravel labels
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    X_cv = X_cv.to_numpy()
    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()
    y_cv = y_cv.to_numpy().ravel()

    # report
    print("Training set size: ", y_train.size)
    print("Test set size: ", y_test.size)
    print("Cross-validation set size: ", y_cv.size)

    # scale the data
    [X_train, X_test, X_cv, scaler] = scale(X_train, X_test, X_cv)

    # return cleaned, scaled and split data and the scaler
    return [X_train, y_train, X_test, y_test, X_cv, y_cv, scaler]


# main function
if __name__ == '__main__':

    # load a selection of labels {labels}, {n_files_per_label} files per label
    input_labels = ["one", "two"]
    [X, y] = load_dataset_selection(labels=input_labels, n_files_per_label=100)

    """
    TODO:
    Part 0, Step 2: 
        - Uncomment the following call to function {clean_split_scale} 
    """
    # cleanup data, split data in training set and test set, normalize data
    [X_train, y_train, X_test, y_test, X_cv, y_cv, scaler] = clean_split_scale(X, y)
