#!/usr/bin/env python3
import pickle

# internal support
from project_directories import pickle_dir
from utils import pickle_file


def train_binary_svm_classifier(X_train, y_train, C, gamma):
    """
    Train a binary Support Vector Machine classifier with sk-learn
    TODO:
    Part 1: 
        - Use the sklearn {svm.SVC} class to implement a binary classifier 
    """

    # get support
    from sklearn import svm
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), svm.SVC(C=C, gamma=gamma))
    clf.fit(X_train, y_train)

    return clf


def train_multi_class_svm_classifier(X_train, y_train, C, gamma):
    """
    Train a multi-class Support Vector Machine classifier with sk-learn
    TODO:
    Part 2: 
        - Use the sklearn {OneVsRestClassifier} class to implement a multi-class classifier 
    """

    # get support
    from sklearn import svm
    from sklearn.multiclass import OneVsRestClassifier

    clf = OneVsRestClassifier(svm.SVC(C=C, gamma=gamma))
    clf.fit(X_train, y_train)

    return clf


def train_multi_class_nn_classifier(
        X_train, y_train, hidden_layer_size, alpha, learning_rate, max_iter):
    """
    Train a Neural Network classifier with py-torch
    """

    # get support
    from MLPClassifier_torch import MLPClassifier

    # create a classifier
    clf = MLPClassifier(X_train, y_train, hidden_layer_size)

    # train the classifier
    clf.train(alpha=alpha, learning_rate=learning_rate, epochs=max_iter, verbose=True)

    return clf


# main function
if __name__ == '__main__':
    pass
