import torch
import numpy as np
from utils import encode_array, decode_array

# Set the PyTorch and numpy random seeds for reproducibility:
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)


class MLPClassifier(torch.nn.Module):

    def __init__(self, X_train, y_train, hidden_layer_sizes):
        """
        Initialize the neural network classifier 
        """
        # initialize superclass
        super().__init__()
        # encode the {y_train} array of string labels into a numpy array of integral labels
        y_train_encoded, self.encoding_keys = encode_array(y_train)
        # convert data into appropriate format for {torch}
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.nn.functional.one_hot(
            torch.tensor(y_train_encoded),
            num_classes=len(self.encoding_keys)).to(dtype=torch.float32)
        # create the neural network
        self.network = self.create_network(hidden_layer_sizes)

    def create_network(self, hidden_layer_size):

        """
        TODO:
        Part 3:
            - Create a neural network classifier using {torch}
        """
        
        # network = torch....
        
        # ...
        
        # return network


    def train(self, alpha, learning_rate, epochs, verbose=False):
        """
        TODO:
        Part 3:
            - Train the neural network classifier 
        NOTE: You can fetch the network created in function {create_network} accessing the variable 
            {self.network}
        """


    def predict(self, X_test):
        """
        Use the trained neural network to predict the labels of the test set 
        """
        # convert input into appropriate format for {torch}
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        # evaluate the trained neural network on the test data
        y_test_encoded = torch.argmax(self.network(X_test_tensor), dim=1)
        # decode the numpy array of integral labels {y_test_encoded} into an array of string labels
        # based on the dictionary of {encoding_keys}, which stores the {string -> integer} encoding
        y_test_decoded = decode_array(y_test_encoded.numpy(), self.encoding_keys)
        return y_test_decoded
