#!/usr/bin/env python3

import pandas as pd
import numpy as np

# internal support
from utils import listdir, dir_name, pickle_file
from project_directories import dataset_dir, pickle_dir


def populate_dataset(input_files):
    # report
    print("Loading dataset...")

    # create training data frame processing all input files
    X = pd.DataFrame([np.load(file) for file in input_files])
    # print(X)
    # get labels data frame processing all input labels
    y = pd.DataFrame([dir_name(file) for file in input_files])
    # print(y)

    # pickle the selected data and labels
    pickle_file(X, 'data.pkl')
    pickle_file(y, 'labels.pkl')

    return [X, y]


def load_dataset():
    # input files (full path, list data directory and subdirectories)
    input_files = [
        dataset_dir + label + '/' + file
        for label in listdir(dataset_dir) for file in listdir(dataset_dir + label)]
    # preprocess all listed files
    return populate_dataset(input_files)


def select_files(label, N):
    input_files = listdir(dataset_dir + label)
    first_N_files = [dataset_dir + label + '/' + next(input_files, None) for _ in range(N)]
    # return the extracted files (first N files)
    return first_N_files


def load_dataset_selection(labels, n_files_per_label):
    # get the first {n_files_per_label} files per each label (full path)
    input_files = [file for label in labels for file in select_files(label, n_files_per_label)]
    # preprocess the listed files
    return populate_dataset(input_files)


# main function
if __name__ == '__main__':
    # load whole dataset
    [X, y] = load_dataset()
