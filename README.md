# Representation Learning: Greedy Layer-Wise Training vs Simple Neural Networks

## 📑 **Table of Contents**
- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Setup and Installation](#️-setup-and-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Results and Visualizations](#-results-and-visualizations)
- [Sources](#-Sources)
- [Contributing](#-contributing)
- [License](#-license)

---

## **Overview**
The goal of this repository is to provide an **introduction and overview of Representation Learning**. 

**Representation Learning** refers to the process of discovering the most useful representations of input data for machine learning tasks. Instead of relying on manually engineered features, representation learning allows models to learn meaningful features directly from raw data.

This repository demonstrates and compares two approaches to learning representations:
1. **Greedy Layer-Wise Unsupervised Training** – a first approach to unsupervised learning where each layer is trained sequentially without supervision. 
2. **Simple Neural Network (NN)** – a standard baseline network trained with traditional backpropagation and supervision.

By comparing these two approaches, you will gain insights into the evolution of representation learning techniques.

---

## **Repository Structure**

## ⚙️ **Setup and Installation**
To use the code in this repository, you need to clone the repo and install the necessary packages.

### **1️⃣ Clone the Repository**
Run the following command in your terminal:
```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

conda env create -f environment.yml
conda activate myenv
