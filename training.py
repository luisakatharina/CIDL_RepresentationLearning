import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import Autoencoder, SimpleNN
from tqdm import tqdm

def train_autoencoder(autoencoder, dataloader, input_size, epochs=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    autoencoder.train()
    for epoch in range(epochs):
        print(f"Autoencoder training: Epoch {epoch+1}/{epochs}")
        for data, _ in tqdm(dataloader, desc="Training autoencoder batches"):
            data = data.view(data.size(0), input_size)
            optimizer.zero_grad()
            decoded, _ = autoencoder(data)
            loss = criterion(decoded, data)
            loss.backward()
            optimizer.step()


def greedy_layerwise_pretraining(train_loader):
    autoencoder1 = Autoencoder(28 * 28, 128)
    
    # Train the first autoencoder
    train_autoencoder(autoencoder1, train_loader, 28 * 28)

    # Collect encoded outputs
    all_encoded = []
    autoencoder1.eval()  # Evaluation mode for inference
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.view(data.size(0), -1)
            _, encoded = autoencoder1(data)
            all_encoded.append(encoded)
    
    # Stack and adjust dataset
    all_encoded = torch.cat(all_encoded, dim=0)
    dummy_targets = torch.zeros(all_encoded.size(0))
    encoded_dataset = TensorDataset(all_encoded, dummy_targets)
    encoded_loader = DataLoader(encoded_dataset, batch_size=64, shuffle=True)
    
    # Train the second autoencoder
    autoencoder2 = Autoencoder(128, 64)
    train_autoencoder(autoencoder2, encoded_loader, 128)
    
    # Initialize the SimpleNN with learned weights
    pre_nn = SimpleNN()
    pre_nn.layer1.weight.data = autoencoder1.encoder.weight.data
    pre_nn.layer1.bias.data = autoencoder1.encoder.bias.data
    pre_nn.layer2.weight.data = autoencoder2.encoder.weight.data
    pre_nn.layer2.bias.data = autoencoder2.encoder.bias.data

    return pre_nn

def supervised_training(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        print(f"Supervised training: Epoch {epoch+1}/5")
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training supervised batches")):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')


def train_baseline_model(train_loader, test_loader):
    # Instantiate a new model
    baseline_model = SimpleNN()  # Same architecture as pre-trained model

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(baseline_model.parameters(), lr=0.001)

    # Standard supervised training loop
    baseline_model.train()
    for epoch in range(5):
        print(f"Baseline training: Epoch {epoch+1}/5")
        for data, target in train_loader:
            optimizer.zero_grad()
            output = baseline_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Evaluate the baseline model
    baseline_model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = baseline_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    baseline_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Baseline model accuracy: {correct}/{len(test_loader.dataset)} ({baseline_accuracy:.2f}%)')
    
    return baseline_accuracy