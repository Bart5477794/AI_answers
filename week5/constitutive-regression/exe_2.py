# %% imports and functions
"""Training of a DNN on a single file of the composite dataset, with and without early stopping of the training."""

from read_data import read_file_to_torch_tensor
from project_directories import raw_data_dir
from exe_2_utils import setup_data_plot, split_scale, count_parameters, predict, setup_loss_plot, generate_prediction_grid, setup_predictions_plot, weights_init
import torch.nn as nn
import torch
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Callable
from collections import OrderedDict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def create_fixed_width_dnn(
        dim_hidden: int, n_hidden: int, dim_input: int = 1, dim_output: int = 1, act=nn.ReLU()):
    """Use the Sequential class from Pytorch to create a deep neural network with the following layers:
    1. input layer: linear with dimensions ('dim_input', 'dim_hidden')
    2. 'n_hidden' hidden layers: linear with dimensions ('dim_hidden', 'dim_hidden')
    3. output layers: linear with dimensions ('dim_hidden', 'dim_output')
    Don't forget to use the activation function 'act' in between layers.
    Return the model.

    Refs:
    1. (nn.Linear) https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    2. (nn.Sequential) https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html"""
    dnn_list = [nn.Linear(dim_input, dim_hidden), act]
    for n in range(n_hidden):
        dnn_list = dnn_list + [nn.Linear(dim_hidden, dim_hidden), act]
    dnn_list = dnn_list + [nn.Linear(dim_hidden, dim_output), act]
    model = nn.Sequential(*dnn_list)
    return model


def train_model(
        model: nn.Module, X_train: torch.tensor, y_train: torch.tensor, X_val: torch.tensor,
        y_val: torch.tensor, loss_function: Callable, optimizer: torch.optim.Optimizer,
        n_epochs: int = 500, tol_train: float = 1e-5, verbose: bool = False):
    """Train the model, while updating the lists 'train_loss_history' and 'val_loss_history' with the training and validation loss at each epoch.
    Notice that the 'verbose' argument is only there for your commodity. You can use it or not as a flag to execute some printing commands. All other arguments should be used in your implementation. 
    To implement this function, you can follow the steps in the reference below, keeping in mind that you will NOT use a dataloader (no loop on the batches), but feed 'X_train', 'y_train' and 'X_val', 'y_val' directly to the model.
    Return 'train_loss_history' and 'val_loss_history'.

    Ref: 
    1. (training loop) https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop
    2. (example of loss function) https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html"""

    # instantiate list for loss histories and include initial loss values
    with torch.no_grad():
        train_loss_history = [loss_function(model(X_train), y_train).item()]
        val_loss_history = [loss_function(model(X_val), y_val).item()]

    for n in range(n_epochs):
        # train loop
        y_tp = model(X_train)
        loss = loss_function(y_tp, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # update loss lists
        train_loss = loss_function(model(X_train), y_train).item()
        val_loss = loss_function(model(X_val), y_val).item()
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        # minimum error stop
        if loss_function(model(X_train), y_train).item() < tol_train:
            break
        # initiate early stopper

    return train_loss_history, val_loss_history


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        """Implement the early stopping criterion. The function has to return 'True' if the current validation loss (in the arguments) has increased with respect to the minimum value of more than 'min_delta' and for more than 'patience' steps. Otherwise the function returns 'False'."""
        stop_train = False
        if self.min_validation_loss > validation_loss:
            self.min_validation_loss = validation_loss
        else:
            if abs(validation_loss) - abs(self.min_validation_loss) > abs(self.min_delta):
                self.counter += 1
                if self.counter > self.patience:
                    stop_train = True
        return stop_train


def train_model_early_stop(
        model: nn.Module, X_train: torch.tensor, y_train: torch.tensor, X_val: torch.tensor,
        y_val: torch.tensor, loss_function: Callable, optimizer: torch.optim.Optimizer,
        n_epochs: int = 500, tol_train: float = 1e-5, es_patience=1, es_delta=0., verbose: bool = False):
    """Copy paste the code from the 'train_model' function above and edit it to include the early-stopping check. Use the EarlyStopper class implemented before. The new arguments es_patience and es_delta are the early stopping patience and threshold on the validation loss.
    Return 'train_loss_history' and 'val_loss_history'.

    Tip: you can exit any loop with the 'break' command"""

    # instantiate list for loss histories and include initial loss values
    with torch.no_grad():
        train_loss_history = [loss_function(model(X_train), y_train).item()]
        val_loss_history = [loss_function(model(X_val), y_val).item()]

    for n in range(n_epochs):
        # train loop
        y_tp = model(X_train)
        loss = loss_function(y_tp, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # update loss lists
        train_loss = loss_function(model(X_train), y_train).item()
        val_loss = loss_function(model(X_val), y_val).item()
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        # minimum error stop
        if loss_function(model(X_train), y_train).item() < tol_train:
            break
        # initiate early stopper
        es = EarlyStopper(patience=es_patience, min_delta=es_delta)
        es_boolean = es.early_stop(loss_function(model(X_val), y_val).item())
        if es_boolean == True:
            break
    return train_loss_history, val_loss_history


if __name__ == '__main__':
    ### DO NOT CHANGE THE SEEDS! ###
    np.random.seed(1)
    torch.manual_seed(0)

    # %% Preliminaries: load, plot, split and scale the data. Also allocate the model.
    data_file_path = raw_data_dir + 'Ea4999_Eb10000_Es10000.txt'
    X, y = read_file_to_torch_tensor(data_file_path, dim_input=3)
    fig, axs = setup_data_plot()
    for sig in range(3):
        for eps in range(3):
            axs[sig, eps].plot(X[:, eps], y[:, sig])
            axs[sig, eps].grid()
    fig.savefig('exe2-data.png')

    print(f"Test split:")
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_scale(
        X, y, test_size=.2, scale=True, verbose=True)
    print(f"\nValidation split:")
    X_train, X_val, y_train, y_val = split_scale(
        X_train, y_train, test_size=.2, scale=False, verbose=True)

    model = create_fixed_width_dnn(dim_hidden=20, n_hidden=5,
                                   dim_input=3, dim_output=3, act=nn.ReLU())
    print(f"\nModel: fixed-width DNN with {count_parameters(model):d} parameters")

    # %% Part 1: training of the DNN without early stopping
    loss_function = nn.MSELoss()
    lr = 1e-2
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    train_loss_history, val_loss_history = train_model(
        model, X_train, y_train, X_val, y_val, loss_function, optimizer, n_epochs=1000,
        tol_train=1e-3, verbose=True)

    fig, ax = setup_loss_plot(title=f"Training and validation loss.")
    ax.semilogy(train_loss_history, label='training loss')
    ax.semilogy(val_loss_history, label='validation loss')
    ax.legend()
    fig.savefig('exe2-loss-no-filter.png')

    # filter the loss values to have a reference for early stopping
    sigma_filter = 50
    train_loss_history = gaussian_filter1d(train_loss_history, sigma=sigma_filter)
    val_loss_history = gaussian_filter1d(val_loss_history, sigma=sigma_filter)

    fig, ax = setup_loss_plot(
        title=f"Training and validation loss with Gaussian filter for noise (sigma={sigma_filter:d}).")
    ax.semilogy(train_loss_history, label='training loss')
    ax.semilogy(val_loss_history, label='validation loss')
    ax.legend()
    fig.savefig('exe2-loss-w-filter.png')

    preds_train = predict(model, X_train)
    preds_test = predict(model, X_test)
    X_grid = generate_prediction_grid(n_gridpoints=1000, n_input=3)
    preds_grid = predict(model, X_grid)

    fig, axs = setup_predictions_plot()
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
    fig.savefig('exe2-predictions.png')

    # %% Part 2: use early stopping
    # Reinitialize the model
    model.apply(weights_init)

    # early stopping parameters
    patience = 5
    delta = 1e-2

    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    train_loss_history, val_loss_history = train_model_early_stop(
        model, X_train, y_train, X_val, y_val, loss_function, optimizer, n_epochs=1000,
        tol_train=1e-3, es_patience=patience, es_delta=delta, verbose=True)

    sigma_filter = 50
    train_loss_history = gaussian_filter1d(train_loss_history, sigma=sigma_filter)
    val_loss_history = gaussian_filter1d(val_loss_history, sigma=sigma_filter)

    fig, ax = setup_loss_plot(title=f"Losses during training (second time, with early stopping).")
    ax.semilogy(train_loss_history, label='training loss')
    ax.semilogy(val_loss_history, label='validation loss')
    ax.legend()
    fig.savefig('exe2-loss-es.png')

    preds_train = predict(model, X_train)
    preds_test = predict(model, X_test)
    preds_grid = predict(model, X_grid)

    fig, axs = setup_predictions_plot(
        figtitle="Predictions after training (second time, with early stopping)")
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
    fig.savefig('exe2-predictions-filter.png')
