# This file contains the visualization functions for the models and the training process.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def convert_seconds_to_mmss(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)  # Extract milliseconds
    return "{:02}:{:02}:{:03}".format(minutes, seconds, milliseconds)

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

def calculate_digit_accuracies(preds, true_labels):
    """
    Calculate per-digit accuracies for a given set of predictions and ground truth labels.
    
    Args:
    - preds (list or array): List of predicted labels.
    - true_labels (list or array): List of ground-truth labels.
    
    Returns:
    - List of accuracies (percentage) for each digit from 0 to 9.
    """
    correct_per_digit = {i: 0 for i in range(10)}
    total_per_digit = {i: 0 for i in range(10)}

    for pred, true in zip(preds, true_labels):
        total_per_digit[true] += 1
        if pred == true:
            correct_per_digit[true] += 1

    accuracies = {i: (correct_per_digit[i] / total_per_digit[i] * 100) if total_per_digit[i] > 0 else 0 
                  for i in range(10)}
    return list(accuracies.values())

def generate_comparison_table(
    baseline_accuracy, final_accuracy, 
    baseline_preds, final_preds, 
    true_labels, baseline_losses, final_losses, 
    baseline_training_time, supervised_training_time
    ):
    """
    Generate a comparison table between baseline and supervised models.
    
    Args:
    - baseline_accuracy (float): Final overall accuracy for the baseline model.
    - final_accuracy (float): Final overall accuracy for the supervised model.
    - baseline_preds (list or array): List of predictions for the baseline model.
    - final_preds (list or array): List of predictions for the supervised model.
    - true_labels (list or array): List of ground-truth labels.
    - baseline_losses (list): List of training losses for the baseline model.
    - final_losses (list): List of training losses for the supervised model.
    - baseline_training_time (float): Total training time for the baseline model.
    - supervised_training_time (float): Total training time for the supervised model.
    
    Returns:
    - DataFrame: Pandas DataFrame containing the comparison table.
    """
    
    # 1️⃣ Per-digit accuracy
    baseline_digit_accuracies = calculate_digit_accuracies(baseline_preds, true_labels)
    supervised_digit_accuracies = calculate_digit_accuracies(final_preds, true_labels)
    
    # 2️⃣ Accuracy statistics (mean, std, min, max)
    baseline_mean_accuracy = np.mean(baseline_digit_accuracies)
    supervised_mean_accuracy = np.mean(supervised_digit_accuracies)

    baseline_std = np.std(baseline_digit_accuracies)
    supervised_std = np.std(supervised_digit_accuracies)

    baseline_min, baseline_max = min(baseline_digit_accuracies), max(baseline_digit_accuracies)
    supervised_min, supervised_max = min(supervised_digit_accuracies), max(supervised_digit_accuracies)

    # 3️⃣ Final loss (last recorded loss for each model)
    baseline_final_loss = baseline_losses[-1] if baseline_losses else 0
    supervised_final_loss = final_losses[-1] if final_losses else 0

    # 4️⃣ Metrics and comparison table
    metrics = ['Overall Accuracy (%)', 'Accuracy Per Digit', 'Mean Accuracy (Digits)', 
               'Min/Max Accuracy', 'Standard Deviation', 'Variance', 
               'Training Time (s)', 'Loss (Final Value)']

    # 5️⃣ Metric values for each model
    baseline_values = [
        round(baseline_accuracy * 100, 2), 
        [round(acc, 2) for acc in baseline_digit_accuracies], 
        round(baseline_mean_accuracy, 2), 
        f'Min: {round(baseline_min, 2)}%, Max: {round(baseline_max, 2)}%', 
        round(baseline_std, 2), 
        round(baseline_std ** 2, 2), 
        round(baseline_training_time, 2), 
        round(baseline_final_loss, 4)
    ]

    supervised_values = [
        round(final_accuracy * 100, 2), 
        [round(acc, 2) for acc in supervised_digit_accuracies], 
        round(supervised_mean_accuracy, 2), 
        f'Min: {round(supervised_min, 2)}%, Max: {round(supervised_max, 2)}%', 
        round(supervised_std, 2), 
        round(supervised_std ** 2, 2), 
        round(supervised_training_time, 2), 
        round(supervised_final_loss, 4)
    ]

    # 6️⃣ Calculate the differences (Δ)
    differences = [
        round(supervised_values[0] - baseline_values[0], 2), 
        [s - b for s, b in zip(supervised_values[1], baseline_values[1])], 
        round(supervised_values[2] - baseline_values[2], 2), 
        f'ΔMin: {round(supervised_min - baseline_min, 2)}%, ΔMax: {round(supervised_max - baseline_max, 2)}%', 
        round(baseline_values[4] - supervised_values[4], 2), 
        round(baseline_values[5] - supervised_values[5], 2), 
        supervised_values[6] - baseline_values[6], 
        round(baseline_values[7] - supervised_values[7], 4)
    ]

    # 7️⃣ Create the DataFrame
    df = pd.DataFrame({
        'Metric': metrics,
        'Baseline Model': baseline_values,
        'Supervised Model': supervised_values,
        'Difference (Δ)': differences
    })

    # Display the table
    print(df.to_markdown(index=False))
    
    return df
