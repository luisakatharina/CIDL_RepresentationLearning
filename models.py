import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Should result in (batch_size, 784)
        print("Shape after flatten:", x.shape)
        x = torch.relu(self.layer1(x))
        print("Shape after layer1:", x.shape)
        x = torch.relu(self.layer2(x))
        print("Shape after layer2:", x.shape)
        x = self.output(x)
        print("Shape after output:", x.shape)
        return x