#!/usr/bin/env python3

# Multi-layer perceptron regression, lab session 2 of AE-2225-II:
# By Guido de Croon

import numpy as np
import torch
from matplotlib import pyplot as plt
import load_data
from scipy import signal

# Set the PyTorch and numpy random seeds for reproducibility:
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


def train_regressor_nn(n_features, n_hidden_neurons, learning_rate, n_epochs, X, Y):
    """ TODO:
    Complete the code below
    """

    # Define the model: defined as, linear model(input, output), function used, linear model(input output)
    # because it is in sequence the input of the first need to be te output of the second part
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, n_hidden_neurons),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_hidden_neurons, n_hidden_neurons),
        torch.nn.Sigmoid(),
        torch.nn.Linear(n_hidden_neurons, 1))

    # MSE loss function:
    loss_fn = torch.nn.MSELoss()

    # optimizer:
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Train the network:
    for t in range(n_epochs):
        # Forward pass
        y_pred = model(X)
        # Compute and print loss. We pass Tensors containing the predicted and
        # true values of y, and the loss function returns a Tensor containing
        # the loss.
        loss = loss_fn(Y, y_pred)
        if t % 100 == 0:
            print(t, loss.item())
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # return the trained model
    return model


# If sensor_data == True, real sensor data is used, otherwise a sine wave is used:
sensor_data = True
if (sensor_data):
    # Load the data:
    [accel_x, accel_y, accel_z, gyro_p, gyro_q, gyro_r, att_phi, att_theta, att_psi,
        cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw] = load_data.load_sensor_data()

    # Data:
    # Whether to include the gyros:
    include_gyros = False
    if (include_gyros):
        X = np.hstack([accel_x, accel_y, accel_z, gyro_p, gyro_q, gyro_r])
    else:
        X = np.hstack([accel_x, accel_y, accel_z])
    n_features = X.shape[1]
    Y = att_phi
    # data pre-processing
    X = X / 1024
    Y = (Y - np.mean(Y)) * 180 / np.pi
    # also introduce an estimation
    multiplication_factor = 180 / np.pi
    y_domain = -np.arctan2(X[:, 1], -X[:, 2]) * multiplication_factor
    y_domain = y_domain - y_domain.mean()
    y_domain = signal.savgol_filter(y_domain, 160, n_features)
else:
    # Load the sine data:
    [X, Y] = load_data.load_sine_data()
    n_features = X.shape[1]

# Total number of samples:
N = X.shape[0]

# Convert to torch tensors:
X = torch.from_numpy(X).float()
Y = torch.from_numpy(Y).float()

# Make a neural network model for the MLP with sigmoid activation functions in the hidden layer, and linear on the output
n_hidden_neurons = 30

""" TODO:
Complete the code below
"""

learning_rate = 0.2  # 0.231
n_epochs = 5000
model = train_regressor_nn(n_features, n_hidden_neurons, learning_rate, n_epochs, X, Y)

# plot the output of the network vs. the ground truth:
y_pred = model(X)
y_pred = y_pred.detach().numpy()
y_plot = Y.detach().numpy()
y_plot = y_plot.reshape(N, 1)

# plt.figure()
# plt.plot(y_plot, 'ko', label='Ground Truth')
# plt.plot(y_pred, label='Network Output', linewidth=2)
# plt.legend()
# plt.savefig('output_vs_ground_truth.png')

# plt.subplot(1, 4, 1)
# plt.hist(X[1], label='Acceleration 1')
# plt.legend()
# plt.subplot(1, 4, 2)
# plt.hist(X[2], label='Acceleration 2')
# plt.legend()
# plt.subplot(1, 4, 3)
# plt.hist(X[3], label='Acceleration 3')
# plt.legend()
# plt.subplot(1, 4, 4)
# plt.hist(y_plot, label='Model')
# plt.legend()
# plt.savefig('output_vs_ground_truth.png')

# plt.subplot(2, 2, 1)
# plt.plot(X[1], label='Acceleration 1', linewidth=2)
# plt.legend()
# plt.subplot(2, 2, 2)
# plt.plot(X[2], label='Acceleration 2', linewidth=2)
# plt.legend()
# plt.subplot(2, 2, 3)
# plt.plot(X[3], label='Acceleration 3', linewidth=2)
# plt.legend()
# plt.subplot(2, 2, 4)
# plt.plot(y_plot, label='Phi_Ground_Truth', linewidth=2)
# plt.legend()
# plt.savefig('output_vs_ground_truth.png')

plt.figure()
plt.plot(y_plot, 'ko', label='Ground Truth')
plt.plot(y_pred, label='Network Output', linewidth=2)
plt.plot(y_domain, label='Arctan Accelerometers', linewidth=2)
plt.legend()
plt.savefig('output_vs_ground_truth.png')

print('Root Mean Squared Error: ', np.sqrt(np.mean((y_pred - y_plot)**2)))
