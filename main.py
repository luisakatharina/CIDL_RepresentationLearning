from data_loader import get_mnist_loaders
from training import greedy_layerwise_pretraining, supervised_training, train_baseline_model
from time import time
from visualization import visualize_weights, plot_model_performance

def main():
    # Load data
    start_time = time()
    train_loader, test_loader = get_mnist_loaders()
    step_1_time = time()
    print("Step 1: Data loading time = {:.2f} seconds".format(step_1_time - start_time))

    # Train baseline model
    baseline_accuracy = train_baseline_model(train_loader, test_loader)
    step_2_time = time()
    print("Step 2: Baseline training time = {:.2f} seconds".format(step_2_time - step_1_time))

    # Greedy layer-wise unsupervised pretraining
    pretrained_model = greedy_layerwise_pretraining(train_loader)
    step_3_time = time()
    print("Step 3: Unsupervised pretraining time = {:.2f} seconds".format(step_3_time - step_2_time))

    # Supervised fine-tuning
    pretrained_accuracy = supervised_training(pretrained_model, train_loader, test_loader)
    step_4_time = time()
    print("Step 4: Supervised fine-tuning time = {:.2f} seconds".format(step_4_time - step_3_time))

    # Visualize weights
    visualize_weights(pretrained_model, layer_index=1)

    # Compare performances
    if pretrained_accuracy is None or baseline_accuracy is None:
        raise ValueError("Both pretrained_accuracy and baseline_accuracy must be provided and not None.")
    plot_model_performance(pretrained_accuracy, baseline_accuracy)

    # Total time
    print("Total execution time = {:.2f} seconds".format(step_4_time - start_time))

if __name__ == "__main__":
    main()
