# CNN for MNIST and CIFAR-10 Dataset's

## Table of Contents
1. [Introduction](#introduction)
2. [MNIST Architecture](#mnist-architecture)
3. [CIFAR-10 Architecture](#cifar-10-architecture)
4. [Hyperparameters](#hyperparameters)
5. [Results](#results)

## Introduction
This repository outlines the convolutional neural network architectures implemented for the MNIST and CIFAR-10 datasets. The MNIST dataset consists of handwritten digits, while CIFAR-10 contains 32x32 color images across 10 classes. 

## MNIST Architecture
The MNIST model consists of a simple convolutional neural network with the following layers:

1. **Convolutional Layer**: 
   - 16 filters, 5x5 kernel, ReLU activation.
   - This layer extracts features from the input images.

2. **Max Pooling Layer**: 
   - Reduces spatial dimensions to half, helping to decrease computation and prevent overfitting.

3. **Second Convolutional Layer**: 
   - 32 filters, 5x5 kernel, ReLU activation.
   - This layer extracts more complex features.

4. **Second Max Pooling Layer**: 
   - Further reduces the dimensionality.

5. **Fully Connected Layer**: 
   - Converts 2D feature maps into a 1D feature vector.

6. **Output Layer**: 
   - Produces probabilities for each of the 10 classes using softmax.

### Model Summary
- Input Size: 28x28 grayscale images
- Output Classes: 10 (digits 0-9)

## CIFAR-10 Architecture
The CIFAR-10 model is a deeper convolutional neural network that is inspired by ResNet architectures:

1. **First Convolutional Block**:
   - Two convolutional layers with 64 filters each, 3x3 kernel, followed by batch normalization and ReLU activation.
   - Max pooling reduces the spatial dimensions.

2. **Second Convolutional Block**:
   - Similar structure with 128 filters, which helps to learn higher-level features.

3. **Third Convolutional Block**:
   - Again similar, with 256 filters to further extract complex features.

4. **Fourth Convolutional Block**:
   - Final block with 512 filters for advanced feature extraction.

5. **Fully Connected Layers**:
   - A dropout layer is used to reduce overfitting.
   - Two fully connected layers to produce the final output.

### Model Summary
- Input Size: 32x32 RGB images
- Output Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

## Hyperparameters

### CIFAR-10 Hyperparameters
| Parameter        | Value   | Descriptions                                        |
|------------------|---------|-----------------------------------------------------|
| Num Epochs       | 100     | Number of epochs for training; CIFAR-10 often requires more epochs to converge |
| Num Classes      | 10      | Total number of classes in the CIFAR-10 dataset (0-9) |
| Batch Size       | 256     | Increased batch size for better training efficiency |
| Learning Rate    | 0.001   | Learning rate for the optimizer                     |

### MNIST Hyperparameters
| Parameter        | Value   | Descriptions                                        |
|------------------|---------|-----------------------------------------------------|
| Num Epochs       | 100     | Number of complete passes through the dataset       |
| Num Classes      | 10      | Number of output classes (digits 0-9)               |
| Batch Size       | 256     | Number of samples processed before updating model parameters |
| Learning Rate    | 0.001   | Step size used by the optimizer to adjust the model's weights |

## Results
- **CIFAR-10 Accuracy**: 90.21%
- **MNIST Accuracy**: 99.58%
