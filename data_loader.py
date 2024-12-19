"""
In this file we define the data loader for the MNIST dataset.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loaders(batch_size=64, test_batch_size=1000):
    """
    @param batch_size: The batch size for the training data loader.
    @param test_batch_size: The batch size for the test data loader.
    @return: The train and test data loaders for the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert the image to a pytorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and std dev values for MNIST dataset
    ])
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform) # Download the training data
    test_dataset = datasets.MNIST('.', train=False, download=True, transform=transform) # Download the test data
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Shuffle the training data
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False) # Do not shuffle the test data

    return train_loader, test_loader