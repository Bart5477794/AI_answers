# %% imports
"""Hyperparameter optimization: finding the 'best' learning rate by random search."""

from read_data import read_file_to_torch_tensor
from project_directories import raw_data_dir
from exe_3_utils import setup_data_plot, split_scale, count_parameters, filter_plot_losses, predict, generate_prediction_grid, setup_predictions_plot, weights_init
import torch.nn as nn
import torch
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Callable


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def create_fixed_width_dnn(
        dim_hidden: int, n_hidden: int, dim_input: int = 1, dim_output: int = 1, act=nn.ReLU()):
    """Copy the same function from the previous exercise below"""
    dnn_list = [nn.Linear(dim_input, dim_hidden), act]
    for n in range(n_hidden):
        dnn_list = dnn_list + [nn.Linear(dim_hidden, dim_hidden), act]
    dnn_list = dnn_list + [nn.Linear(dim_hidden, dim_output), act]
    model = nn.Sequential(*dnn_list)
    return model


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """Copy the same function from the previous exercise below"""
        pass


def train_model_early_stop(
        model: nn.Module, X_train: torch.tensor, y_train: torch.tensor, X_val: torch.tensor,
        y_val: torch.tensor, loss_function: Callable, optimizer: torch.optim.Optimizer,
        n_epochs: int = 500, tol_train: float = 1e-5, es_patience=1, es_delta=0., verbose: bool = False):
    """Copy the same function from the previous exercise below"""
    return train_loss_history, val_loss_history


def lr_random_search(model, X_train, y_train, X_val, y_val, reps: int = 15):
    """Implement a random search of the learning rate. Use the log-uniform distribution to sample the different learning rate values (can be found in scipy, see ref below). 

    For training, use `train_model_early_stop` with the following arguments:
        - loss_function = nn.MSELoss()
        - optimizer = torch.optim.Adam()
            * what learning rate should you use?
        - n_epochs = 1_000
        - tol_train = 1e-4
        - es_patience = 4
        - es_delta = 1e-4

    Your function should return the learning rate value corresponding to the lowest final validation loss, among all the runs performed.
    The 'reps' parameter specifies how many runs to perform in the random search.

    Refs 
    1. (Scipy distributions): https://docs.scipy.org/doc/scipy/reference/stats.html
    """
    return lr_best


if __name__ == '__main__':
    ### DO NOT CHANGE THE SEEDS! ###
    np.random.seed(1)
    torch.manual_seed(0)

    # %% Preliminaries: load, plot, split and scale the data. Also allocate the model.
    data_file_path = raw_data_dir + 'Ea-5000_Eb-5000_Es10000.txt'
    X, y = read_file_to_torch_tensor(data_file_path, dim_input=3)
    fig, axs = setup_data_plot()
    for sig in range(3):
        for eps in range(3):
            axs[sig, eps].plot(X[:, eps], y[:, sig])
            axs[sig, eps].grid()
    fig.savefig('exe3-data.png')

    # %% 0.1: split and scale
    print(f"Test split:")
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_scale(
        X, y, test_size=.2, scale=True, verbose=True)
    print(f"\nValidation split:")
    X_train, X_val, y_train, y_val = split_scale(
        X_train, y_train, test_size=.2, scale=False, verbose=True)

    # %% 0.2: model creation
    model = create_fixed_width_dnn(dim_hidden=20, n_hidden=5,
                                   dim_input=3, dim_output=3, act=nn.ReLU())
    print(f"\nModel: fixed-width DNN with {count_parameters(model):d} parameters")

    # %% Part 1: randomly search the 'optimal' lr
    reps = 10

    lr_best = lr_random_search(model, X_train, y_train, X_val, y_val, reps=reps)
    print("Best learning rate found: ", lr_best)

    # %% Part 2: re-train with optimal learning rate and plot predictions
    model.apply(weights_init)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr_best)

    train_loss_history, val_loss_history = train_model_early_stop(
        model, X_train, y_train, X_val, y_val, loss_function, optimizer, n_epochs=10_000,
        tol_train=1e-4, es_patience=4, es_delta=1e-4, verbose=True)

    preds_train = predict(model, X_train)
    preds_test = predict(model, X_test)
    X_grid = generate_prediction_grid(n_gridpoints=1000)
    preds_grid = predict(model, X_grid)

    fig, axs = setup_predictions_plot(
        figtitle=f"Predictions after training with lr = {lr_best:.2E}")
    for i in range(3):
        axs[i, 0].scatter(X_train[:, i], y_train[:, i], label='true')
        axs[i, 0].scatter(X_train[:, i], preds_train[:, i], label='predicted')
        axs[i, 0].legend()
        axs[i, 0].grid()

        axs[i, 1].scatter(X_test[:, i], y_test[:, i], label='true')
        axs[i, 1].scatter(X_test[:, i], preds_test[:, i], label='predicted')
        axs[i, 1].legend()
        axs[i, 1].grid()

        axs[i, 2].plot(X_grid[:, i], preds_grid[:, i])
        axs[i, 2].grid()
    fig.savefig('exe3-data-predictions.png')
