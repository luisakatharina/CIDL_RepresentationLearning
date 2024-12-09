from data_loader import get_mnist_loaders
from training import greedy_layerwise_pretraining, supervised_training
from time import time

def main():
    start_time = time()
    train_loader, test_loader = get_mnist_loaders()
    step_1_time = time()
    print("step 1 time = " + str(step_1_time - start_time))


    # Greedy layer-wise unsupervised pretraining
    pretrained_model = greedy_layerwise_pretraining(train_loader)
    step_2_time = time()    
    print("step 2 time = " + str(step_2_time - step_1_time))

    # Supervised fine-tuning
    supervised_training(pretrained_model, train_loader, test_loader)

    step_3_time = time()    
    print("step 3 time = " + str(step_3_time - step_2_time))

    print("total time = " + str(step_3_time - start_time))
    

if __name__ == "__main__":
    main()