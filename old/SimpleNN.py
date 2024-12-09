import torch
import torch.nn as nn

# Define simple Autoencoder for unsupervised pretraining
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded

# Define a deep feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x

# Training function for autoencoder
def train_autoencoder(autoencoder, dataloader, epochs=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    for epoch in range(epochs):
        for data, _ in dataloader:
            data = data.view(data.size(0), -1)  # Flatten the images
            optimizer.zero_grad()
            decoded, _ = autoencoder(data)
            loss = criterion(decoded, data)
            loss.backward()
            optimizer.step()

# Function to extract pretrained weights to initialize layers of the main network
def greedy_layerwise_pretraining(train_loader):
    # Define the autoencoders for each pair of layers
    autoencoder1 = Autoencoder(28 * 28, 128)
    train_autoencoder(autoencoder1, train_loader)
    autoencoder2 = Autoencoder(128, 64)
    train_autoencoder(autoencoder2, train_loader)

    # Initialize the main model with pretrained weights
    pre_nn = SimpleNN()
    pre_nn.layer1.weight.data = autoencoder1.encoder.weight.data
    pre_nn.layer1.bias.data = autoencoder1.encoder.bias.data
    pre_nn.layer2.weight.data = autoencoder2.encoder.weight.data
    pre_nn.layer2.bias.data = autoencoder2.encoder.bias.data

    return pre_nn

# Supervised training function
def supervised_training(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    model.train()
    for epoch in range(5):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')

# Data preprocessing and loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Execute greedy layer-wise pretraining and supervised training
pretrained_model = greedy_layerwise_pretraining(train_loader)
supervised_training(pretrained_model, train_loader, test_loader)