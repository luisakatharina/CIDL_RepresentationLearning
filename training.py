import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import Autoencoder, SimpleNN
from tqdm import tqdm
from time import time

def train_autoencoder(autoencoder, dataloader, input_size, layer, epochs=5):
    """Train an individual autoencoder layer."""
    criterion = nn.MSELoss()
    optimizer = optim.SGD(autoencoder.parameters(), lr=0.01, momentum=0.9)
    
    autoencoder.train()
    layer_losses = []  # Store the loss for each batch
    layer_times = []   # Store time for each epoch
    layer_weights = [] # Store model weights after each epoch

    for epoch in range(epochs):
        start_time = time()  # Start time for epoch
        progress_bar = tqdm(dataloader, desc=f"  Autoencoder {layer} training: Epoch {epoch+1}/{epochs}", leave=True)

        for data, _ in progress_bar:
            data = data.view(data.size(0), input_size)
            optimizer.zero_grad()
            decoded, _ = autoencoder(data)
            loss = criterion(decoded, data)
            loss.backward()
            optimizer.step()
            
            # Track the loss for this batch
            layer_losses.append(loss.item())
            
            # Update the progress bar description to keep track of the current loss
            progress_bar.set_description(f"  Autoencoder {layer} training: Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        epoch_time = time() - start_time  # Calculate time for this epoch
        layer_times.append(epoch_time)  # Store training time for the epoch
        
        # Store the model weights after each epoch
        layer_weights.append({k: v.clone() for k, v in autoencoder.state_dict().items()})

    return layer_losses, layer_times, layer_weights


def greedy_layerwise_pretraining(train_loader, test_loader):
    """ Greedy layer-wise training of two autoencoders."""
    
    # Store overall metrics for pretraining
    pretrained_losses = []    # Losses for both autoencoders
    pretrained_times = []     # Training times for both autoencoders
    pretrained_weights = []   # Weights for both autoencoders
    pretrained_preds = []     # Encoded activations for each data point

    # First autoencoder (28*28 -> 128)
    autoencoder1 = Autoencoder(28 * 28, 128)
    print(" Training Autoencoder 1 (28*28 -> 128)")

    # Train first autoencoder
    layer1_losses, layer1_times, layer1_weights = train_autoencoder(autoencoder1, train_loader, 28 * 28, layer=1)
    pretrained_losses.extend(layer1_losses)
    pretrained_times.extend(layer1_times)
    pretrained_weights.append(layer1_weights)

    # Collect encoded outputs for the next layer
    all_encoded = []
    autoencoder1.eval()  # Evaluation mode for inference
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1)
            _, encoded = autoencoder1(data)
            all_encoded.append(encoded)
            #pretrained_preds.extend(encoded.cpu().numpy())  # Store activations for analysis
    
    # Stack and create new dataset
    all_encoded = torch.cat(all_encoded, dim=0)
    dummy_targets = torch.zeros(all_encoded.size(0))
    encoded_dataset = TensorDataset(all_encoded, dummy_targets)
    encoded_loader = DataLoader(encoded_dataset, batch_size=64, shuffle=True)

    # Second autoencoder (128 -> 64)
    autoencoder2 = Autoencoder(128, 64)
    print(" Training Autoencoder 2 (128 -> 64)")

    # Train second autoencoder
    layer2_losses, layer2_times, layer2_weights = train_autoencoder(autoencoder2, encoded_loader, 128, layer=2)
    pretrained_losses.extend(layer2_losses)
    pretrained_times.extend(layer2_times)
    pretrained_weights.append(layer2_weights)

    # Initialize the final SimpleNN model with weights from the autoencoders
    pre_nn = SimpleNN()
    pre_nn.layer1.weight.data = autoencoder1.encoder.weight.data
    pre_nn.layer1.bias.data = autoencoder1.encoder.bias.data
    pre_nn.layer2.weight.data = autoencoder2.encoder.weight.data
    pre_nn.layer2.bias.data = autoencoder2.encoder.bias.data

    # Evaluate pre_nn on the test set to collect predictions
    pre_nn.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = pre_nn(data)
            pred = output.argmax(dim=1, keepdim=True)
            pretrained_preds.extend(pred.view(-1).cpu().numpy())  # Store class indices for predictions

    return pre_nn, pretrained_losses, pretrained_times, pretrained_weights, pretrained_preds


def supervised_training(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Tracking metrics
    supervised_losses = []  # Store training loss for each batch
    supervised_times = []   # Store training time for each epoch
    supervised_weights = [] # Store model weights after each epoch
    supervised_preds = []   # Store predictions on the test set
    true_labels = []        # Store true labels on the test set

    # Training loop with progress bar
    model.train()
    for epoch in range(5):
        start_time = time()
        progress_bar = tqdm(train_loader, desc=f" Supervised training: Epoch {epoch+1}/5", leave=True)
        for batch_idx, (data, target) in enumerate(progress_bar):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Track the loss for this batch
            supervised_losses.append(loss.item())
            
            # Update the progress bar description to show current loss
            progress_bar.set_description(f" Supervised training: Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

        epoch_time = time() - start_time
        supervised_times.append(epoch_time)
        
        # Track model weights for this epoch (clone the weights to avoid references)
        supervised_weights.append({k: v.clone() for k, v in model.state_dict().items()})

    # Evaluation with progress bar
    model.eval()
    correct = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc=" Evaluating supervised model", leave=True)
        for data, target in progress_bar:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            supervised_preds.extend(pred.view(-1).cpu().numpy())  # Correctly storing class indices
            true_labels.extend(target.view(-1).cpu().numpy())            
            
            # Update the progress bar with the number of correct predictions
            progress_bar.set_description(f" Evaluating: {correct}/{len(test_loader.dataset)} correct")
    
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f' Test set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return model, accuracy, supervised_losses, supervised_times, supervised_weights, supervised_preds, true_labels



def train_baseline_model(train_loader, test_loader):
    # Instantiate a new model
    baseline_model = SimpleNN()  # Same architecture as pre-trained model

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(baseline_model.parameters(), lr=0.01, momentum=0.9)
    
    # Tracking metrics
    baseline_losses = []         # Store training loss for each batch
    baseline_times = []          # Store training time for each epoch
    baseline_weights = []        # Store model weights after each epoch
    baseline_preds = []          # Store predictions on the test set
    true_labels = []             # Store true labels on the test set

    # Standard supervised training loop
    baseline_model.train()
    for epoch in range(5):
        start_time = time()  # Start time for epoch
        progress_bar = tqdm(train_loader, desc=f" Baseline training: Epoch {epoch+1}/5", leave=True)
        # Use tqdm to display the progress of the batches
        for data, target in progress_bar:
            optimizer.zero_grad()
            output = baseline_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Track the loss for this batch
            baseline_losses.append(loss.item())
            progress_bar.set_description(f" Baseline training: Epoch {epoch+1}/5, Loss: {loss.item():.4f}")

        epoch_time = time() - start_time  # Calculate time for this epoch
        baseline_times.append(epoch_time)  # Store training time for the epoch
        
        # Track model weights for this epoch (clone the weights to avoid references)
        baseline_weights.append({k: v.clone() for k, v in baseline_model.state_dict().items()})

    # Evaluate the baseline model
    baseline_model.eval()
    correct = 0
    print(" Evaluating the model on the test set...")

    # Use tqdm for the progress bar during evaluation
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc=" Evaluating"):
            output = baseline_model(data)
            pred = output.argmax(dim=1, keepdim=True)  # Get predicted class
            baseline_preds.extend(pred.view(-1).cpu().numpy())  # Store predicted labels
            true_labels.extend(target.view(-1).cpu().numpy())  # Store true labels
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    baseline_accuracy = 100. * correct / len(test_loader.dataset)
    print(f' Baseline model accuracy: {correct}/{len(test_loader.dataset)} ({baseline_accuracy:.2f}%)')
    
    return baseline_accuracy, baseline_losses, baseline_times, baseline_weights, baseline_preds, true_labels