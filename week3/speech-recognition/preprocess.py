#!/usr/bin/env python3
import numpy as np
import librosa
import os

# internal support
from utils import listdir, filename_no_ext
from project_directories import raw_data_dir, dataset_dir


def preprocess_file(filename):
    # load dataset
    y, sr = librosa.load(filename)
    # compute spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20, fmax=8000)
    # convert to decibels
    S_dB = librosa.power_to_db(S, ref=np.max)
    # all done
    return [S_dB, sr]


def preprocess():
    # create directory for preprocessed dataset (fail immediately if directory exists)
    os.mkdir(dataset_dir)
    # fetch all labels
    labels = [label for label in listdir(raw_data_dir)]
    print("Preprocessing the data. This might take time. Please wait...")
    # loop on labels (i.e. words)
    for label in labels:
        # create a subdirectory for this word
        os.mkdir(dataset_dir + label)
        # loop on audio recordings for this work (raw data)
        for file in listdir(raw_data_dir + label):
            raw_filename = raw_data_dir + label + "/" + file
            processed_filename = dataset_dir + label + "/" + filename_no_ext(file) + ".npy"
            # process raw data file and save the result in binary format
            np.save(processed_filename, preprocess_file(raw_filename)[0].flatten())


# main function
if __name__ == '__main__':
    # preprocess all files
    preprocess()
