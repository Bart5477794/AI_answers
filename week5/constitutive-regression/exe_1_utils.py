import os
import sys
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def split_scale(X, y, test_size=.2, verbose=False):
    # split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    if verbose:
        # report
        print("Training set size: ", y_train.size)
        print("Test set size: ", y_test.size)

    # scale the data linearly between 0 and 1. Fit only on train dataset
    X_scaler = MinMaxScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    # scale the labels by the max of the training set
    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test)

    # return scaled and split data and the scaler 
    return (X_train, X_test, y_train, y_test, X_scaler, y_scaler)

def setup_data_plot(title="Metal tensile plasticity data and polynomial predictions"):
    fig, ax = plt.subplots()
    ax.set_xlabel('d (mm)')
    ax.set_ylabel('F (N)')
    ax.set_title(title)
    ax.set_ylim([0., 1.2])
    return fig, ax

def setup_bar_plot(figtitle='MSE and R2 for different degrees of the fitting polynomial'):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].set_title('Train set')
    axs[0, 1].set_title('Test set')
    axs[0, 0].set_ylabel('MSE')
    axs[1, 0].set_ylabel('R2')
    axs[1, 0].set_xlabel('degree')
    axs[1, 1].set_xlabel('degree')
    axs[0, 0].set_ylim([0., 1.])
    axs[1, 0].set_ylim([-1., 1.])
    axs[0, 1].set_ylim([0., 1.])
    axs[1, 1].set_ylim([-1., 1.])
    fig.suptitle(figtitle)
    fig.tight_layout()
    return fig, axs

def setup_error_plot(title="MSE mean and standard deviation for increasing degree"):
    fig, ax = plt.subplots()
    ax.set_xlabel('degree')
    ax.set_ylabel('MSE')
    ax.set_title(title, fontsize=12)
    ax.set_ylim([0., 1.])
    return fig, ax
