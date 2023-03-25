import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn


def split_scale(X, y, test_size=.2, scale=True, verbose=False):
    original_data_type = type(X)

    # convert to numpy array, ravel labels 
    if original_data_type == torch.Tensor: 
        X = X.detach().numpy()
        y = y.detach().numpy()

    # split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    if verbose:
        # report
        print("Training set size: ", y_train.size)
        print("Test set size: ", y_test.size)

    # scale the data linearly between 0 and 1. Fit only on train dataset
    if scale:
        X_scaler = MinMaxScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)

        # scale the labels by the max of the training set
        y_scaler = MinMaxScaler()
        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)

        if original_data_type == torch.Tensor:
            X_train = torch.tensor(X_train)
            y_train = torch.tensor(y_train)
            X_test = torch.tensor(X_test)
            y_test = torch.tensor(y_test)

        # return scaled and split data and the scaler 
        return (X_train, X_test, y_train, y_test, X_scaler, y_scaler)

    if original_data_type == torch.Tensor:
            X_train = torch.tensor(X_train)
            y_train = torch.tensor(y_train)
            X_test = torch.tensor(X_test)
            y_test = torch.tensor(y_test)

    return (X_train, X_test, y_train, y_test)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def weights_init(layer: nn.Module) -> None:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

def predict(model, X):
    with torch.no_grad():
        preds = model(X)
    return preds

def generate_prediction_grid(n_gridpoints=100):
    X_grid = torch.linspace(1., 0., n_gridpoints).view(-1, 1)
    X_grid = X_grid.repeat(1, 2)
    X_grid = torch.cat((X_grid, torch.linspace(0., 1., n_gridpoints).view(-1, 1)), dim=1)
    return X_grid

def setup_data_plot(figtitle="Full-grid plot of plane strain/stresses in the composite OH plate"):
    fig, axs = plt.subplots(3, 3)
    axs[2, 0].set_xlabel('eps1')
    axs[2, 1].set_xlabel('eps2')
    axs[2, 2].set_xlabel('gamma12')
    axs[0, 0].set_ylabel('sig1 (MPa)')
    axs[1, 0].set_ylabel('sig2 (MPa)')
    axs[2, 0].set_ylabel('tau12 (MPa)')
    fig.suptitle(figtitle)
    fig.tight_layout()
    return fig, axs

def setup_loss_plot(title="Training and validation loss throughout epochs"):
    fig, ax = plt.subplots()
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE')
    ax.set_title(title)
    return fig, ax

def setup_predictions_plot(figtitle="Predictions after training"):
    fig, axs = plt.subplots(3, 3)
    axs[0, 0].set_xlabel('eps1')
    axs[0, 1].set_xlabel('eps1')
    axs[0, 2].set_xlabel('eps1')
    axs[1, 0].set_xlabel('eps1')
    axs[1, 1].set_xlabel('eps1')
    axs[1, 2].set_xlabel('eps1')
    axs[2, 0].set_xlabel('gamma12')
    axs[2, 1].set_xlabel('gamma12')
    axs[2, 2].set_xlabel('gamma12')
    axs[0, 0].set_ylabel('sig1')
    axs[1, 0].set_ylabel('sig2')
    axs[2, 0].set_ylabel('tau12')
    axs[0, 0].set_title('Train set')
    axs[0, 1].set_title('Test set')
    axs[0, 2].set_title('Grid')
    fig.suptitle(figtitle)
    fig.tight_layout()
    return fig, axs

def filter_plot_losses(train_loss_history, val_loss_history, sigma_filter=10, plot_title="Training and validation loss"):
    train_loss_history = gaussian_filter1d(train_loss_history, sigma=sigma_filter)
    val_loss_history = gaussian_filter1d(val_loss_history, sigma=sigma_filter)

    _, ax = setup_loss_plot(title=plot_title)
    ax.semilogy(train_loss_history, label='training loss')
    ax.semilogy(val_loss_history, label='validation loss')
    ax.legend()
    plt.draw()