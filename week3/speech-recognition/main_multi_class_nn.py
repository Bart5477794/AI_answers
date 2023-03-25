#!/usr/bin/env python3

# internal support
from utils import listdir
from load_dataset import load_dataset_selection
from clean_split_scale import clean_split_scale
from train import train_multi_class_nn_classifier
from evaluate import f1score
from project_directories import raw_data_dir

if __name__ == '__main__':

    # load a selection of labels {labels}, {n_files_per_label} files per label
    input_labels = ["one", "two", "three", "go"]
    [X, y] = load_dataset_selection(labels=input_labels, n_files_per_label=1000)

    """ TODO: 
    Part 3:
        - call function {clean_split_scale} to cleanup data, split data in training 
            set and test set, normalize data
        - call function {train_multi_class_nn_classifier} to train the classifier that 
            you have implemented on the training data
        - switch to the full dataset (load all labels as opposed to a selection of labels)
        - plot the f1 score obtained on the training set and on the cross-validation set
            as a function of the size of the hidden layer
        - use the previous plot to select a reasonable value for the size of the hidden layer 
        - assess the performance of the trained model 
        - plot the f1 score obtained on the training set and on the cross-validation set
            as a function of the number of samples per label
    """
