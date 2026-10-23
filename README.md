> CS4287 - Assignment 1: Convolutional Neural Networks (CNNs)

# Semantic Image Segmentation on Cityscapes Dataset

![results/v1.2.2-512-1024/training.gif](results/v1.2.2-512-1024/training.gif)
GIF shows the training output of [v1.2.2-512-1024](results/v1.2.2-512-1024/README.md)

## Introduction

The objective of this assignment was to implement a **Convolutional Neural Network (CNN)** based on an existing architecture and classify a chosen dataset. 

**U-Net architecture** was chosen as the baseline because its encoder-decoder structure allows it to capture contextual information while preserving spatial details. 

**Cityscapes dataset** was selected because it contains diverse urban street scenes with pixel level annotations. It provides a suitable benchmark for training and evaluating the model.

> Get started with by following [`GUIDE.md`](GUIDE.md)

## Trained Models

Trained model weights are available in the [GitHub Releases](https://github.com/BaguetteJet/u-net-cityscapes/releases). Download the desired model checkpoint and place it in the `models/` folder. 

Use the corresponding source code from the same release, so that the model architecture and supporting code are compatible with the weights.