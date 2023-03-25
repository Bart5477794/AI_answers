#!/usr/bin/env python3

import pickle
import os
from sklearn import metrics

# internal support
from project_directories import pickle_dir, plot_dir


def evaluate(clf, y_true, y_predicted):

    # report
    print(f"Classification report for classifier {clf}:\n"
          f"{metrics.classification_report(y_true, y_predicted)}\n")


def f1score(y_true, y_predicted):

    # return the f1score of clf
    return metrics.f1_score(y_true, y_predicted, average='weighted')


def display_confusion_matrix(
        clf, X_test, y_test, display_labels, figurename='confusion_matrix.png'):

    # get support
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        display_labels=display_labels,
        cmap=plt.cm.Blues,
        normalize='true')

    # if it does not exist yet, create a directory for output files
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    filename = plot_dir + figurename
    plt.savefig(filename)


# main function
if __name__ == '__main__':

    # read preprocessed data from pickle
    X_test = pickle.load(open(pickle_dir + 'data-test.pkl', 'rb'))
    y_test = pickle.load(open(pickle_dir + 'labels-test.pkl', 'rb'))

    # load model from pickle file
    clf = pickle.load(open(pickle_dir + 'clf_weights.pkl', 'rb'))

    # predict the words on the test subset
    predicted = clf.predict(X_test)

    # print classification report
    evaluate(clf, y_test, predicted)
