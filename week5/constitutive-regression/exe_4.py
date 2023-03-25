"""Batching: comparing training on a large dataset when using the entire dataset or mini-batches"""

#%% imports
import os
import random
import sys
import time
from typing import Callable, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from exe_4_utils import filter_plot_losses, predict, setup_predictions_plot, split_scale, weights_init
from read_data import get_composite_file_names, generate_data_from_files

def create_fixed_width_dnn(dim_hidden:int, n_hidden:int, dim_input:int=1, dim_output:int=1, act=nn.ReLU()):
    """Copy from previous exercises"""
    return model

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """Copy from previous exercies"""
        pass

class DatasetExe4(Dataset):
    """Implement this dataset as a standard Pytorch dataset. Make sure to include all the necessary methods. X and y are the PyTorch tensors containing your data.
    
    Ref: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files"""
    def __init__(self, X, y) -> None:
        super().__init__()
        self._X = X
        self._y = y
    
    ### YOUR CODE HERE ###
    pass

def train_model_early_stop(model:nn.Module, train_loader:DataLoader, X_val:torch.tensor, y_val:torch.tensor,loss_function: Callable, optimizer: torch.optim.Optimizer, n_epochs: int = 500,tol_train: float = 1e-5, es_patience=1, es_delta=0., verbose: bool = False):
    """Implement this training function by copying the earlier version and adapt it to process batches. Batches are iterated through a dataloader ('train_loader' in the input). The idea is that for every epoch, every batch should be used to compute the loss and update the model's parameter (how many 'for' loops will you need?).
    
    The function returns training and validation losses during the epochs and the mean wall-clock time elapsed per epoch."""
    return train_loss_history, val_loss_history, mean_epoch_time


if __name__ == '__main__':
     ### DO NOT CHANGE THE SEEDS! ###
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(0)
    
    #%% Preliminaries
    files = get_composite_file_names(shuffle=True)

    #%% 0.1: extract data and split
    X, y = generate_data_from_files(files, type="tensor")

    print(f"Test split:")
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_scale(X, y, test_size=.2, scale=True, verbose=True)
    print(f"\nValidation split:")
    X_train, X_val, y_train, y_val = split_scale(X_train, y_train, test_size=.2, scale=False, verbose=True)

    #%% 0.2: declare the training dataset
    trainset = DatasetExe4(X_train, y_train)

    #%% 0.3: build the model
    model = create_fixed_width_dnn(dim_hidden=64, n_hidden=5, dim_input=3, dim_output=3)


    #%% Part 1: compare performance and errors with and without batching (fixed batch size)
    loss_function = nn.MSELoss()
    lr = 3e-4

    # 1.1: train w/o mini-batches (single batch)
    train_loader = DataLoader(trainset, batch_size=len(y_train), shuffle=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    train_loss_history, val_loss_history, mean_epoch_time = train_model_early_stop(model, train_loader, X_val, y_val, loss_function, optimizer, n_epochs=1_000, tol_train=1e-4, es_patience=4, es_delta=1e-4, verbose=True)

    print(f"No batching: mean time per epoch {mean_epoch_time:.3f} s")
    filter_plot_losses(train_loss_history, val_loss_history, plot_title="Losses, no batching")

    #%% 1.2: train with mini-batches
    bs = 512

    model.apply(weights_init)

    train_loader = DataLoader(trainset, batch_size=bs, shuffle=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    train_loss_history, val_loss_history, mean_epoch_time =  train_model_early_stop(model, train_loader, X_val, y_val, loss_function, optimizer, n_epochs=1_000, tol_train=1e-4, es_patience=4, es_delta=1e-4, verbose=True)

    print(f"Batch size = {bs:d}: mean time per epoch {mean_epoch_time:.3f} s")
    filter_plot_losses(train_loss_history, val_loss_history, plot_title=f"Losses, batch size = {bs:d}")
    plt.savefig('exe4-losses-w-batch.png')

    #%% Part 2: compute the test error and plot predictions on the data from one file
    preds_test = predict(model, X_test)
    err_test = loss_function(preds_test, y_test)
    print(f"l2 error on test data: {err_test:.2E}")

    f = random.choice(files)
    X_f, y_f = generate_data_from_files([f], type="array")
    X_f = X_scaler.transform(X_f)
    y_f = y_scaler.transform(y_f)
    X_f = torch.tensor(X_f)
    y_f = torch.tensor(y_f)

    preds_f = predict(model, X_f)
    
    fig, axs = setup_predictions_plot(figtitle=f"Predictions for data on file {f}, after training with batch size = {bs}")
    for i in range(3):
        axs[i].scatter(X_f[:, i], y_f[:, i], label='true')
        axs[i].scatter(X_f[:, i], preds_f[:, i], label='predicted')
        axs[i].legend()
        axs[i].grid()
    fig.savefig('exe4-file-preds.png')