import torch
from matplotlib import pyplot as plt
import numpy as np

# RNN
# Set up a class for the recurrent neural network:


class RNN(torch.nn.Module):

    def __init__(self, n_inputs, n_hiddens, batch_size, n_outputs=1):
        super(RNN, self).__init__()
        self.n_hiddens = n_hiddens
        self.layer1 = torch.nn.Linear(n_inputs, n_hiddens)
        self.act1 = torch.nn.Sigmoid()
        self.layer2 = torch.nn.RNN(n_hiddens, n_hiddens)
        self.layer3 = torch.nn.Linear(n_hiddens, n_outputs)

        self.batch_size = batch_size
        self.hidden_activations = self.reset_hiddens()

    def forward(self, input, show_hiddens=False):

        # Reset the hidden activations:
        self.reset_hiddens()

        # How does this work?
        layer1_act = self.layer1(input)
        layer1_out = self.act1(layer1_act)
        layer2_out, self.hidden_activations = self.layer2(layer1_out, self.hidden_activations)
        if (show_hiddens):
            plt.figure()
            plt.plot(self.hidden_activations.view(self.batch_size, self.n_hiddens).detach().numpy())
            plt.savefig('hiddens.png')
        output = self.layer3(layer2_out)

        return output

    def reset_hiddens(self):
        # reset the hidden activations to zero
        self.hidden_activations = torch.zeros(1, self.batch_size, self.n_hiddens)
        return

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.reset_hiddens()
        return

    def set_n_hiddens(self, n_hiddens):
        self.n_hiddens = n_hiddens
        self.reset_hiddens()
        return
