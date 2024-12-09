import torch
import torch.optim as optim
import torch.nn as nn
from models import Autoencoder, SimpleNN

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


def greedy_layerwise_pretraining(train_loader):
    autoencoder1 = Autoencoder(28 * 28, 128)
    train_autoencoder(autoencoder1, train_loader, 28 * 28)  # Flattened input size for first autoencoder

    # Extract encoded representations from the first autoencoder to train the second one
    encoded_loader = [(autoencoder1.encoder(data.view(data.size(0), -1))[1].detach(), _) for data, _ in train_loader]
    encoded_loader = torch.DataLoader(encoded_loader, batch_size=64, shuffle=True)

    autoencoder2 = Autoencoder(128, 64)
    train_autoencoder(autoencoder2, encoded_loader, 128)  # Encoded size for second autoencoder

    pre_nn = SimpleNN()
    pre_nn.layer1.weight.data = autoencoder1.encoder.weight.data
    pre_nn.layer1.bias.data = autoencoder1.encoder.bias.data
    pre_nn.layer2.weight.data = autoencoder2.encoder.weight.data
    pre_nn.layer2.bias.data = autoencoder2.encoder.bias.data
    return pre_nn

def supervised_training(model, train_loader, test_loader):
    print("Starting supervised training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(5):
        print(f"Epoch {epoch+1} begins.")
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"Processing batch {batch_idx}...")
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    print("Training complete!")

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')