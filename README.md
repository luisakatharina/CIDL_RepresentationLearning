# Representation Learning: Greedy Layer-Wise Training vs Simple Neural Networks
## **Table of Contents**
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Sources](#sources)
---

## **Overview**
This repository serves as an **introduction to Representation Learning**, a crucial aspect of modern machine learning. The primary objective is to explore how machine learning models can automatically learn meaningful representations of data, which play a fundamental role in downstream tasks like classification, clustering, and more.

### **What is Representation Learning?**
**Representation Learning** allows machine learning models to automatically discover the best internal features to represent data. Instead of relying on manually engineered features, the models learn hierarchical representations from raw data — an essential capability behind modern deep learning systems like convolutional neural networks (CNNs) and transformers.

### **What does this repository do?**
This project compares two approaches to representation learning:
1. **Greedy Layer-Wise Training**: A step-by-step, unsupervised learning approach where each layer is trained independently before the entire model is fine-tuned.
2. **Simple Neural Network (NN)**: A traditional neural network trained using fully supervised backpropagation from the start.

The code in this repository enables you to train, visualize, and compare these two models on the **MNIST dataset**, which consists of handwritten digit images (0-9). You'll get insights into how greedy layer-wise pretraining affects the learning process, training time, and final performance.

## **Quick Start**
### ** Clone the Repository**
Run the following command in your terminal:
```bash
    git clone https://github.com/luisakatharina/CIDL_RepresentationLearning.git
    cd CIDL_RepresentationLearning
```

Install the necessary packages using the environment.yml file:
```bash
    conda env create -f environment.yml
    conda activate myenv
```


## **Usage**
This section explains how the code works and what you can expect from each step.
- Data Loading:
    Loads the MNIST dataset (or other datasets, if modified).
    The dataset is split into training and test sets.

- Training:
    Simple Neural Network: Fully trained with supervised learning using backpropagation.
    Greedy Layer-Wise Training: Each layer is trained one at a time in an unsupervised manner. After all layers are trained, supervised fine-tuning is applied to the whole network.

- Evaluation:
    Models are evaluated on the MNIST test set.
    Metrics like accuracy, per-digit accuracy, training time, and loss are computed.

- Visualization:
    Loss Curves: Shows how loss changes during training for both models.
    Accuracy Per Digit: Compares how each model classifies individual digits (0-9).
    Weight Visualizations: Displays how weights change over time.
    Confusion Matrices: Shows which digits are confused for each model.

## **Sources**

