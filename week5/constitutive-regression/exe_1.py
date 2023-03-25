# %% imports and function
"""Polynomial fit to the data from tensile testing of metal bar """

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from read_data import read_file_to_np_array
from project_directories import raw_data_dir
from exe_1_utils import split_scale, setup_data_plot, setup_bar_plot, setup_error_plot
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


def create_ls_regression_model(degree: int):
    """Return a least squares regression model over polynomial features up to the input degree using scikit-learn. Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html#sphx-glr-auto-examples-model-selection-plot-underfitting-overfitting-py"""
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    model = Pipeline([("polynomial_features", polynomial_features),
                      ("linear_regression", linear_regression)])
    return model


def fit(degree: int, X: np.ndarray, y: np.ndarray):
    """Use `create_ls_regression_model` to fit a polynomial of chosen degree over X and y. Return the model."""
    # Create the polynomial features matrix and a pipeline
    model = create_ls_regression_model(degree)
    model.fit(X, y)
    return model


def predict(model, X):
    """Return the model prediction on X. Be mindful of the data shape!"""
    X = X.reshape(-1, 1)
    y_predict = model.predict(X)
    return y_predict


def mse(y_true, y_pred):
    """Return the mse between labels and predictions"""
    error = mean_squared_error(y_true, y_pred)
    return error


def rsquared(y_true, y_pred):
    """Return the r2 score between labels and predictions"""
    score = r2_score(y_true, y_pred)
    return score


def kfold_cv(degree, X, y, K=5):
    """Perform cross validation on X, y and record the mean squared error (MSE) on the validation set at each split. Return a np.array of the errors at each split.
    You are NOT allowed to use the Scikit-Learn library to build this function."""
    for k in range(K):
        pass
    return np.array(errors)  # note 'errors' should be a List


if __name__ == '__main__':
    ### DO NOT CHANGE THE SEEDS! ###
    np.random.seed(1)

    # %% Part 0: preliminaries
    # Read data
    data_file_path = raw_data_dir + 'metal-tensile.txt'
    X, y = read_file_to_np_array(data_file_path)
    X = X[1:]
    y = y[1:]

    # %% 0.1 Split data into train and test datsets
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = split_scale(
        X, y, test_size=0.3, verbose=True)

    # %% Part 1: fitting with different polynomial degrees
    poly_degrees = [2, 3, 5, 13]
    X_plotgrid = np.linspace(0, 1., 100)

    f1, ax1 = setup_data_plot()
    f2, ax2 = setup_bar_plot()

    ax1.scatter(X_train, y_train, label='training data')

    for deg in poly_degrees:
        fitted_model = fit(deg, X_train, y_train)

        y_train_preds = predict(fitted_model, X_train)
        l2_err_train = mse(y_train, y_train_preds)
        r2_err_train = rsquared(y_train, y_train_preds)

        y_test_preds = predict(fitted_model, X_test)
        l2_err_test = mse(y_test, y_test_preds)
        r2_err_test = rsquared(y_test, y_test_preds)

        ax1.plot(X_plotgrid, predict(fitted_model, X_plotgrid), label=f'deg={deg:d}')
        ax2[0, 0].bar(deg, l2_err_train, color='C0')
        ax2[0, 1].bar(deg, l2_err_test, color='C0')
        ax2[1, 0].bar(deg, r2_err_train, color='C0')
        ax2[1, 1].bar(deg, r2_err_test, color='C0')

    ax1.legend()
    f1.savefig('exe1-data-predictions.png')
    f2.savefig('exe1-train-test-errors.png')

    # %% Part 2: k-fold cross validation to choose the polynomial degree
    # K = 4
    # poly_degrees = list(range(2, 11))
    # f1, ax1 = setup_error_plot(title=f"MSE mean and sdev for {K:d}-fold CV with increasing degree")

    # for deg in poly_degrees:
    #     val_scores = kfold_cv(deg, X_train, y_train, K=K)

    #     ax1.errorbar(deg, val_scores.mean(), val_scores.std(),
    #                  color='C0', marker='o', linestyle='None')

    # f1.savefig('exe1-cv-error.png')
