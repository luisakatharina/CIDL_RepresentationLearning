"""
This file contains the model definitions for the autoencoder and the simple neural network.
"""
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    # This class defines the autoencoder model.
    
    def __init__(self, input_dim, hidden_dim):
        """
        @param input_dim: The input dimension of the autoencoder.
        @param hidden_dim: The hidden dimension of the autoencoder.
"""
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """
        @param x: The input to the autoencoder.
        @return: The output of the autoencoder.
        """
        encoded = torch.sigmoid(self.encoder(x)) # Apply the encoder and the sigmoid activation function.
        decoded = self.decoder(encoded) # Apply the decoder.
        return decoded, encoded

class SimpleNN(nn.Module):
    # Define the simple neural network model
    def __init__(self):
         # This function initializes the simple neural network model.
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        """
        This function defines the forward pass of the simple neural network model.
        @param x: The input to the simple neural network.
        @return: The output of the simple neural network.
        """
        x = torch.flatten(x, 1)  # Flatten the input. That is, convert the input to a 1D tensor.
        x = torch.sigmoid(self.layer1(x)) # Apply the first linear layer and the sigmoid activation function.
        x = torch.sigmoid(self.layer2(x)) # Apply the second linear layer and the sigmoid activation function.
        x = self.output(x) # Apply the output layer.
        return x