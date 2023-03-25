import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from typing import List, Tuple
import numpy as np
import torch

from project_directories import raw_data_dir


def read_file_to_np_array(filepath: str, dim_input:int=1) -> Tuple[np.array, np.array]:

    # get data from the filepath to a unique np array
    data = np.genfromtxt(filepath, dtype=np.float32,
                         delimiter=',', skip_header=1)

    dim_output = data.shape[1] - dim_input

    # reshape and return a tuple of input data and labels
    return data[:, :dim_input].reshape(-1, dim_input), data[:, dim_input:].reshape(-1, dim_output)


def read_file_to_torch_tensor(filepath: str, dim_input:int=1) -> torch.tensor:

    # read the data as np.array
    input_data, labels = read_file_to_np_array(filepath, dim_input=dim_input)

    # convert to toech.tensor and return
    return torch.tensor(input_data), torch.tensor(labels)


def get_composite_file_names(dir:str=raw_data_dir, shuffle:bool=True) -> List:
    """Returns all the file names corresponding to the open hole composite analyses 

    Args:
        dir (str): directory containing the files. Defaults to raw_dir (see project_directories.py)
        shuffle (bool): optional shuffle of the filename strings. Defaults to True

    Returns:
        List: list of file names
    """
    fnames =  [f for f in os.listdir(dir) if f[:2] == 'Ea']
    if shuffle:
        random.shuffle(fnames)
    return fnames


def generate_data_from_files(files_list:List[str], type:str="array"):
    """Reads in multiple files from the name in a list and returns either numpy arrays or torch tensors
    """
    if type not in ("array", "tensor"):
        return NotImplementedError("Data type not implemented.")

    X, y = read_file_to_np_array(raw_data_dir + files_list[0], dim_input=3)

    for f in files_list[1:]:
        X_i, y_i = read_file_to_np_array(raw_data_dir + f, dim_input=3)
        X = np.append(X, X_i, axis=0)
        y = np.append(y, y_i, axis=0)

    if type == "tensor":
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

    return X, y
