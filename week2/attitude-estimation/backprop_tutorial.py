import torch
import torch.nn as nn
import numpy as np

'''
x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)

# forward pass
y_pred = w * x
loss = (y_pred - y)**2 
# backward pass
loss.backward()
# update weights 
'''

'''
# time for linear regression using ANN
# 1 input 1 output one neuron

# should get f = 2 * x
X = np.array([1, 2, 3, 4])
Y = np.array([2, 4, 6, 8])

w = 0.0


def forward(x):
    return w * x


def loss(y, y_pr):  # MSE
    return ((y_pr - y)**2).mean()


def gradient(x, y, y_pr):  # this gives total gradient
    return np.dot(2*x, y_pr - y).mean()


# training
learning_rate = 0.01
n_iterations = 10

for batch in range(n_iterations):
    # forward pass, calculate prediction
    y_pred = forward(X)
    # calculate loss
    l = loss(Y, y_pred)
    # calculate gradients (in this case total gradient)
    dw = gradient(X, Y, y_pred)
    # update weights
    w -= learning_rate * dw
    if batch % 1 == 0:
        print("Batch", str(batch+1), 'w =', round(w, 3), 'loss =', round(l, 8))
print("final prediction after training: f(12) =", str(forward(12)))
'''

'''
# time for linear regression using ANN
# 1 input 1 output one neuron
# using pytorch

# should get f = 2 * x
X = torch.tensor([1, 2, 3, 4])
Y = torch.tensor([2, 4, 6, 8])
# we want to know the gradient with respect to this one for backpropagation
w = torch.tensor([0.0], requires_grad=True)


def forward(x):
    return w * x


def loss(y, y_pr):  # MSE
    return ((y_pr - y)**2).mean()


# gradient is easily calculated know need for manual backprop, just use loss.backward()
# training
learning_rate = 0.01
n_iterations = 100

for batch in range(n_iterations):
    # forward pass, calculate prediction
    y_pred = forward(X)
    # calculate loss
    l = loss(Y, y_pred)
    # gradient
    l.backward()
    # change weights, since using pytorch we need to be careful with commands
    with torch.no_grad():  # get rid of the gradient boolean or operations go in error
        w -= learning_rate * w.grad  # use the gradient calculated with l.backward an initialized with the boolean
    w.grad.zero_()  # reset grad tensor, if we don't, we will add all gradients of every iteration
    if batch % 10 == 0:
        print("Batch", str(batch+1), 'w =', round(w.item(), 3), 'loss =', round(l.item(), 8))
print("final prediction after training: f(12) =", str(forward(12)))
# note that pytorch needs more iterations since the backward function is less accurate than the numerical answer
# backward function is an estimate and not an exact gradient
'''

'''
# now use everything with pytorch, using optimizers
# 1) Design the model
# 2) Construct loss and optimizer
# 3) training loop

# define input and weights
X = torch.tensor([1.0, 2.0, 3.0, 4.0])
Y = torch.tensor([2.0, 4.0, 6.0, 8.0])
w = torch.tensor([0.0], requires_grad=True)


def forward(x):
    return w * x


# training parameters
learning_rate = 0.01
n_iterations = 100

# define loss and optimizer, optimizer will do the weight gradient steps for you
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)
# optimizer is a class that needs a list op parameters to optimize, thus the weights
# and the learning rate so it can set steps
# optimizer has certain functions that make life easier

# training
for batch in range(n_iterations):
    # forward pass, calculate prediction
    y_pred = forward(X)
    # calculate loss
    l = loss(Y, y_pred)
    # gradient
    l.backward()
    # update weight with a simple function call from your optimizer
    optimizer.step()
    # zero out the gradients since we don't want to add all gradients from last iteration
    optimizer.zero_grad()
    if batch % 10 == 0:
        print("Batch", str(batch+1), 'w =', round(w.item(), 3), 'loss =', round(l.item(), 8))
print("final prediction after training: f(12) =", str(forward(12)))
'''

'''
# now use everything with pytorch, using optimizers and model
# 1) Design the model
# 2) Construct loss and optimizer
# 3) training loop

# define input and weights
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
# we use a model, this includes the weight tensor
# not the input and outputs are changed to since the model commands require
# specific formatting of the tensors
# first the sampels and features need to be specified
n_samples, n_features = X.shape
# note, shape command gives [rows, columns], thus [4, 1]
# then the model can be initialized, weights are randomly distributed
# model needs input, function, output look for the input things on internet, for MLPs its more complicated
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)

# training parameters

learning_rate = 0.01
n_iterations = 100
# define loss and optimizer, optimizer will do the weight gradient steps for you
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training
for batch in range(n_iterations):
    # forward pass, calculate prediction with calling the model
    y_pred = model(X)
    # calculate loss
    l = loss(Y, y_pred)
    # gradient
    l.backward()
    # update weight with a simple function call from your optimizer
    optimizer.step()
    # zero out the gradients since we don't want to add all gradients from last iteration
    optimizer.zero_grad()
    if batch % 10 == 0:
        # weight is now defined as model parameter with another bias variable
        [w, b] = model.parameters()
        # also the weight is a 2d tensor so if we want item we have to use the right indexes
        print("Batch", str(batch+1), 'w =', round(w[0][0].item(), 3), 'loss =', round(l.item(), 8))

# model input needs to be a 2d tensor, thus
X_test = torch.tensor([[12.0]])
print("final prediction after training: f(12) =", str(model(X_test).item()))
'''

# final code without explanation
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

# training parameters
learning_rate = 0.02
n_iterations = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training
for batch in range(n_iterations):
    y_pred = model(X)
    l = loss(Y, y_pred)
    l.backward()

    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
        [w, b] = model.parameters()
        print("Batch", str(batch+1), 'w =', round(w[0][0].item(), 3), 'loss =', round(l.item(), 8))

X_test = torch.tensor([[12.0]])
print("final prediction after training: f(12) =", str(model(X_test).item()))
