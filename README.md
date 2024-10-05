# Model Architecture and Performance for MNIST and CIFAR-10 Datasets

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. MNIST Model Architecture](#2-mnist-model-architecture)
- [3. CIFAR-10 Model Architecture](#3-cifar-10-model-architecture)
- [4. Hyperparameters](#4-hyperparameters)
- [5. Results](#5-results)
- [6. Conclusion](#6-conclusion)

## 1. Introduction

This document provides an overview of the deep learning models developed for classifying the MNIST and CIFAR-10 datasets. The architectures and training processes are outlined, along with their respective performance metrics.

## 2. MNIST Model Architecture

The MNIST model is a convolutional neural network (CNN) designed for digit classification. The architecture consists of:

- **Input Layer**: Accepts grayscale images of size 28x28 pixels.
- **Convolutional Layer 1**:
  - Filters: 32
  - Kernel Size: 5x5
  - Activation: ReLU
- **Max Pooling Layer**: Reduces the spatial dimensions by half.
- **Convolutional Layer 2**:
  - Filters: 64
  - Kernel Size: 5x5
  - Activation: ReLU
- **Max Pooling Layer**: Further reduces the spatial dimensions.
- **Fully Connected Layer**: Flattens the output and maps to 10 output classes (digits 0-9).

This architecture allows the model to learn spatial hierarchies in the input images effectively.

### Performance

- **Accuracy**: 99.58%

---

## 3. CIFAR-10 Model Architecture

The CIFAR-10 model is a deeper convolutional neural network tailored for classifying 32x32 color images across 10 classes. The architecture includes:

- **Input Layer**: Accepts RGB images of size 32x32 pixels.
- **Convolutional Block 1**:
  - Two Convolutional Layers with 64 filters each
  - Kernel Size: 3x3
  - Activation: ReLU
  - Max Pooling Layer: Reduces the spatial dimensions by half.
- **Convolutional Block 2**:
  - Two Convolutional Layers with 128 filters each
  - Max Pooling Layer: Reduces the spatial dimensions by half.
- **Convolutional Block 3**:
  - Two Convolutional Layers with 256 filters each
  - Max Pooling Layer: Reduces the spatial dimensions by half.
- **Convolutional Block 4**:
  - Two Convolutional Layers with 512 filters each
  - Max Pooling Layer: Reduces the spatial dimensions by half.
- **Fully Connected Layers**:
  - A dropout layer to prevent overfitting followed by two fully connected layers to map to 10 output classes.

This architecture effectively captures complex patterns in the images due to its depth and use of multiple convolutional layers.

### Performance

- **Accuracy**: 90.21%

---

## 4. Hyperparameters

### CIFAR-10 Hyperparameters

```python
num_epochs = 100  # Number of epochs for training; CIFAR-10 often requires more epochs to converge
num_classes = 10  # Total number of classes in the CIFAR-10 dataset (0-9)
batch_size = 256  # Increased batch size for better training efficiency
learning_rate = 0.001  # Learning rate for the optimizer
```
