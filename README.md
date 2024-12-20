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
[1] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA: MIT Press, 2016. [Online]. Available: http://www.deeplearningbook.org
[2] J. Devlin, M. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2019. Available: https://arxiv.org/abs/1810.04805.
[3] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A Simple Framework for Contrastive Learning of Visual Representations,” Proceedings of the 37th International Conference on Machine Learning (ICML), 2020. Available: https://arxiv.org/abs/2002.05709.
[4] A. Vaswani et al., “Attention is All You Need,” Advances in Neural Information Processing Systems (NeurIPS), 2017. Available: https://arxiv.org/abs/1706.03762.
[5] J. Ho, A. Jain, and P. Abbeel, “Denoising Diffusion Probabilistic Models,” Advances in Neural Information Processing Systems (NeurIPS), 2020. Available: https://arxiv.org/abs/2006.11239.
[6] S. J. Wang et al., “On the Efficiency of Large Language Models: Advances in Pruning, Distillation, and Quantization,” IEEE Transactions on Neural Networks and Learning Systems, 2024.
[7] J. Pearl, “Causality: Models, Reasoning, and Inference,” Cambridge University Press, 2009.
[8] S. Barocas, M. Hardt, and A. Narayanan, “Fairness and Machine Learning: Limitations and Opportunities,” MIT Press, 2024.
[9] A. Radford et al., “Learning Transferable Visual Models From Natural Language Supervision,” Proceedings of the International Conference on Machine Learning (ICML), 2021. Available: https://arxiv.org/abs/2103.00020.
[10] S. Hart et al., “Interactive AI: How AI is Augmenting Human Decision-Making,” Proceedings of the ACM Human-Computer Interaction Conference (CHI), 2024.


