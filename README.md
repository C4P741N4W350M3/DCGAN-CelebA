# GAN for Image Generation with Limited Dataset

This project implements a Generative Adversarial Network (GAN) using Keras and TensorFlow to generate images from a limited dataset of 10,000 images. The GAN consists of a generator and a discriminator trained to produce realistic images from random noise.


<div style="text-align: center;">
  <img src="./images/output.gif" alt="Alt Text" />
</div>



## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Acknowledgements](#acknowledgements)

## Overview

This project showcases the capability of GANs to generate realistic images even with a relatively small dataset. By using a dataset of only 10,000 images, we demonstrate that it is possible to achieve significant results in image generation.

## Dataset

The dataset used in this project consists of 10,000 images from the CelebA dataset, resized to 64x64 pixels. The images are normalized to the range [-1, 1] to match the output range of the generator.

## Model Architecture

### Discriminator

The discriminator is a convolutional neural network (CNN) that classifies images as real or fake. It consists of:
- Four convolutional layers with LeakyReLU activations and batch normalization.
- A final dense layer with a sigmoid activation function.

### Generator

The generator is a CNN that transforms random noise vectors into images. It consists of:
- Four transposed convolutional layers with ReLU activations and batch normalization.
- A final transposed convolutional layer with a tanh activation function to produce images in the range [-1, 1].

## Training

The GAN is trained using the following setup:
- Optimizer: Adam with a learning rate of 0.0002 and beta values of 0.5 and 0.999.
- Loss Function: Binary Crossentropy.
- Epochs: 50 (for practice; adjust as needed for better results).

The training process involves alternating updates to the discriminator and generator to improve their performance iteratively.

## Results

The GAN produces realistic images after training on the limited dataset of 10,000 images. Sample images generated during the training process are saved and displayed to monitor progress.

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/gan-image-generation.git
    cd gan-image-generation
    ```
    
2. **Prepare the dataset**:
    - Ensure you have the CelebA dataset (or any other dataset) in a directory named `celeba_gan`.
    - The images should be in the format required by `keras.utils.image_dataset_from_directory`.

3. **Monitor training progress**:
    - The script saves generated images at the end of each epoch in the project directory.
    - Generated images are also displayed during training.

## Dependencies

- TensorFlow
- Keras
- NumPy
- Matplotlib

Install the required packages using:
```bash
pip install tensorflow keras numpy matplotlib
