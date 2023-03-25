#!/usr/bin/env python3

# internal support
from load_dataset import load_dataset_selection
from clean_split_scale import clean_split_scale
from train import train_binary_svm_classifier
from evaluate import evaluate, display_confusion_matrix

if __name__ == '__main__':

    # load a selection of labels {labels}, {n_files_per_label} files per label
    input_labels = ["one", "two"]
    [X, y] = load_dataset_selection(labels=input_labels, n_files_per_label=1000)

    """ TODO: 
    Part 1:
        - call function {clean_split_scale} to cleanup data, split data in training 
            set and test set, normalize data
        - call function {train_binary_svm_classifier} to train the classifier that 
            you have implemented on the training data
        - use the trained model to make predictions on the test data 
    """

    # load a selection of labels {labels}, {n_files_per_label} files per label
    input_labels = ["one", "two"]
    [X, y] = load_dataset_selection(labels=input_labels, n_files_per_label=1000)
    # initialize svm
    C = 10
    gamma = 0.001
    [X_train, y_train, X_test, y_test, X_cv, y_cv, scaler] = clean_split_scale(X, y)
    clf = train_binary_svm_classifier(X_train, y_train, C, gamma)
    y_predicted = clf.predict(X_test)

    print("------------------")
    print(y_predicted)
    print("------------------")
    print(y_test)

    """ TODO: 
    Part 1:
        - use the following lines to assess the performance of the model implemented
    """

    # print classification report
    evaluate(clf, y_test, y_predicted)
    # display confusion matrix (figure saved in directory {plot})
    display_confusion_matrix(clf, X_test, y_test, display_labels=input_labels, figurename='confusion_matrix.png')
