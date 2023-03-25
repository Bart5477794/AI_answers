#!/usr/bin/env python3

import wget
import os
import tarfile
import shutil

# internal support
from utils import listdir, filename_no_ext
from project_directories import data_dir, raw_data_dir


def fetch_data():
    # create data directory
    os.mkdir(data_dir)

    # download data data
    url = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
    tarball = wget.download(url, out=data_dir)

    # get file name from url
    filename = data_dir + os.path.basename(url)

    # open file
    file = tarfile.open(filename)

    # extracting file
    file.extractall(raw_data_dir)

    # close file
    file.close()

    # words to select
    words = ['down', 'five', 'go', 'nine', 'off', 'one', 'seven', 'stop', 'two', 'yes', 'eight',
             'four', 'left', 'no', 'on', 'right', 'six', 'three', 'up', 'zero']

    # loop on labels (i.e. words) and remove directories containing labels that are not in my list
    # of words
    for label in listdir(raw_data_dir):
        if label not in words and os.path.isdir(raw_data_dir + label):
            shutil.rmtree(raw_data_dir + label)

    # clean up compressed file
    os.remove(filename)

    # move everything that is not a directory to data directory
    for item in listdir(raw_data_dir):
        if not os.path.isdir(raw_data_dir + item):
            shutil.move(raw_data_dir + item, data_dir)

    # cleanup duplicates (all files that are not *0.wav)
    for label in listdir(raw_data_dir):
        for file in listdir(raw_data_dir + label):
            if filename_no_ext(file)[-1] != '0':
                os.remove(raw_data_dir + label + "/" + file)


# main function
if __name__ == '__main__':
    fetch_data()
