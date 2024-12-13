import matplotlib.pyplot as plt
import torch

def visualize_weights(model, layer_index=1, n_w=8, n_h=4):
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

def plot_model_performance(pretrained_accuracy, baseline_accuracy):
    models = ['Pretrained', 'Baseline']
    accuracies = [pretrained_accuracy, baseline_accuracy]
    
    plt.bar(models, accuracies, color=['skyblue', 'tomato'])
    plt.xlabel('Model Type')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 100)
    plt.show()
