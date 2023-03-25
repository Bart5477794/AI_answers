import pathlib
import os
import pickle
import numpy as np

# internal support
from project_directories import pickle_dir

# generator to ignore hidden files


def listdir(path):
    for file in os.listdir(path):
        if not file.startswith('.'):
            yield file

# helper function to get file name without extension


def filename_no_ext(file_path):
    return pathlib.Path(file_path).resolve().stem


def dir_name(file_path):
    return os.path.basename(os.path.dirname(file_path))


def pickle_file(object, file_name):
    # if it does not exist yet, create a directory for pickle files
    if not os.path.exists(pickle_dir):
        os.mkdir(pickle_dir)

    # save the model to file
    pickle.dump(object, open(pickle_dir + file_name, 'wb'))


def encode_array(string_array):
    """
    Function to encode an array of string labels {string_array} into a numpy array of integral labels
    """

    # the encoded array to be filled and returned
    encoded_array = []
    # a dictionary to store the {string -> integer} encoding
    encoding_keys = {}
    # the next available spare key (start from 0)
    key = 0
    # for each string in the array of string labels {string_array}
    for string in string_array:
        # if you have not found the string in the string_array yet
        # (and therefore the string has not been encoded yet)
        if (not (string in encoding_keys)):
            # encode the string with the new spare key available
            encoding_keys[string] = key
            # compute the next available spare key
            key += 1
        # use the {encoding_keys} to compute the encoded value for {string}
        encoded_array.append(encoding_keys[string])

    # all done: return the {encoded_array} and the keys for encoding/decoding
    return np.array(encoded_array), encoding_keys


def decode_array(encoded_array, encoding_keys):
    """
    Function to decode a numpy array of integral labels {string_array} into an array of string labels
    based on the dictionary of {encoding_keys}, which stores the {string -> integer} encoding
    """

    # the decoded array to be filled and returned
    decoded_array = []
    # compute the decoding keys, which store the {integer -> string} decoding
    decoding_keys = {val: key for key, val in encoding_keys.items()}
    # for each {entry} of the the array of encoded labels
    for entry in encoded_array:
        # append the decoded entry to the {decoded_array}
        decoded_array.append(decoding_keys[entry])

    # all done: return the {decoded_array}
    return np.array(decoded_array)
