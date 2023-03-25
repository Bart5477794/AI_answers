#!/usr/bin/env python3

# internal support
from load_dataset import load_dataset_selection
from clean_split_scale import clean_split_scale
from train import train_multi_class_svm_classifier
from evaluate import f1score
from evaluate import evaluate

# external support
import matplotlib.pyplot as plt


def plot_f1(f1_train, f1_cv):
    index_ = [1, 2, 3, 4, 5]
    plt.scatter(index_, f1_train, label="train")
    plt.scatter(index_, f1_cv, label="cv")
    plt.legend()
    plt.title("f1 vs gamma")
    plt.savefig('f1_vs_gamma.png')


if __name__ == '__main__':

    # load a selection of labels {labels}, {n_files_per_label} files per label
    input_labels = ["one", "two", "three", "go"]
    [X, y] = load_dataset_selection(labels=input_labels, n_files_per_label=1000)
    mode = 2

    if mode == 1:
        # initialize svm
        C = 10
        gamma = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        f1_train = []
        f1_cv = []
        for i in range(len(gamma)):  # best gamma is 0.001
            [X_train, y_train, X_test, y_test, X_cv, y_cv, scaler] = clean_split_scale(X, y)
            clf = train_multi_class_svm_classifier(X_train, y_train, C, gamma[i])
            y_train_pred = clf.predict(X_train)
            y_cv_pred = clf.predict(X_cv)
            f1_train.append(f1score(y_train, y_train_pred))
            f1_cv.append(f1score(y_cv, y_cv_pred))
        plot_f1(f1_train, f1_cv)
    if mode == 2:
        # initialize svm
        C = 10
        gamma = 1e-3
        [X_train, y_train, X_test, y_test, X_cv, y_cv, scaler] = clean_split_scale(X, y)
        clf = train_multi_class_svm_classifier(X_train, y_train, C, gamma)
        y_predicted = clf.predict(X_test)
        evaluate(clf, y_test, y_predicted)

        print("------------------")
        print(y_predicted[:10])
        print("------------------")
        print(y_test[:10])

    """ TODO: 
    Part 2:
        - call function {clean_split_scale} to cleanup data, split data in training 
            set and test set, normalize data
        - call function {train_multi_class_svm_classifier} to train the classifier that 
            you have implemented on the training data for several values of {gamma}
        - plot the f1 score obtained on the training set and on the cross-validation set
            as a function of {gamma}
        - use the previous plot to select the best value of {gamma} 
        - assess the performance of the trained model 
    """
