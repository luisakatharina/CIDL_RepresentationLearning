import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix

def visualize_weights(model, layer_index=1, n_w=8, n_h=4):
    """Visualizes weights of the model for the specified layer."""
    if layer_index == 1:
        weights = model.layer1.weight.data.view(-1, 28, 28)
    elif layer_index == 2:
        weights = model.layer2.weight.data.view(-1, 128, 1)  # Adjust shape accordingly
    else:
        return

    fig, axes = plt.subplots(n_h, n_w, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < weights.shape[0]:
            ax.imshow(weights[i].cpu(), cmap='viridis')
        ax.axis('off')
    plt.show()

def plot_model_performance(final_accuracy, baseline_accuracy):
    models = ['Final', 'Baseline']
    accuracies = [final_accuracy, baseline_accuracy]
    plt.bar(models, accuracies, color=['skyblue', 'tomato'])
    plt.xlabel('Model Type')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 100)
    plt.show()

def plot_loss_curves(baseline_losses, pretrained_losses, final_losses):
    plt.plot(baseline_losses, label='Baseline Model Loss', color='red')
    plt.plot(pretrained_losses, label='Pretrained Model Loss', color='blue')
    plt.plot(final_losses, label='Final Model Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.show()

def plot_accuracy_curves(baseline_accuracy, pretrained_accuracy, final_accuracy):
    plt.plot(baseline_accuracy, label='Baseline Accuracy', color='red')
    plt.plot(pretrained_accuracy, label='Pretrained Accuracy', color='blue')
    plt.plot(final_accuracy, label='Final Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Set Accuracy Over Time')
    plt.legend()
    plt.show()

def plot_confusion_matrices(true_labels, baseline_preds, pretrained_preds, final_preds):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create a confusion matrix plot helper
    def create_cm(true, pred, ax, title):
        cm = confusion_matrix(true, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    
    create_cm(true_labels, baseline_preds, axes[0], 'Baseline Model')
    create_cm(true_labels, pretrained_preds, axes[1], 'Pretrained Model')
    create_cm(true_labels, final_preds, axes[2], 'Final Model')
    
    plt.show()

def plot_training_time(baseline_times, pretrained_times, final_times):
    epochs = range(1, len(baseline_times) + 1)
    layerOne_times = pretrained_times[:5]
    layerTwo_times = pretrained_times[5:]
    plt.plot(epochs, baseline_times, label='Baseline Model', color='red')
    plt.plot(epochs, layerOne_times, label='Pretrained Layer One', color='yellow')
    plt.plot(epochs, layerTwo_times, label='Pretrained Layer Two', color='violet')
    plt.plot(epochs, final_times, label='Final Model', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time Per Epoch')
    plt.legend()
    plt.show()

def visualize_weights_evolution(weights, epochs=[0, 5], layer_index='layer1.weight'):
    """
    Visualizes how the weights of the model evolve over epochs.
    
    :param weights: A list of dictionaries containing weight states across epochs.
    :param epochs: Epoch indices to visualize (e.g., [0, 5] for the first and last epoch).
    :param layer_index: String key to indicate which layer's weights to visualize (e.g., 'layer1.weight').
    """
    # Determine the layout for the plots.
    n_cols = min(len(weights[0][layer_index]), 8)  # Assume a max of 8 weights to visualize
    n_rows = min(len(epochs), 10)  # Max of 10 rows for chosen epochs

    # Set up the subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
    
    for i, epoch in enumerate(epochs):
        # Access the state dict from the specified epoch
        weight_tensor = weights[epoch][layer_index]
        
        # For this layer, assume a 2D weight visualization
        weight_images = weight_tensor.cpu().numpy().reshape(-1, 28, 28)

        for j in range(n_cols):
            ax = axes[i, j] if n_rows > 1 else axes[j]  # Adjust for single row of axes
            if j < len(weight_images):
                ax.imshow(weight_images[j], cmap='viridis', aspect='auto')
            ax.axis('off')
            if i == 0:  # Annotate the top row only
                ax.set_title(f'Neuron {j}')
        axes[i, 0].set_ylabel(f'Epoch {epoch}', size=12)
    
    plt.tight_layout()
    plt.show()

def plot_weight_differences(baseline_weights, final_weights):
    weight_diffs = {layer: (final_weights[layer] - baseline_weights[layer]).flatten().cpu().numpy() 
                    for layer in baseline_weights}
    for layer, diffs in weight_diffs.items():
        plt.hist(diffs, bins=50, color='purple', alpha=0.7)
        plt.xlabel('Weight Difference')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Weight Differences (Layer {layer})')
        plt.show()