# CycleGAN for Object-to-Object Translation (Apple to Orange and Orange to Apple)

This repository contains a PyTorch implementation of a CycleGAN model trained for object-to-object translation tasks. Specifically, this model translates images of apples to oranges and vice versa. The training was performed on a subset of ImageNet containing apple and orange images.

## Model Description

### Generator
The Generator model employs a U-Net-like architecture with several key components:
- **Initial Convolution:** A convolution layer with kernel size 7x7, followed by a downsampling layer.
- **Downsampling Layers:** Two convolutional blocks that progressively reduce the spatial dimensions while increasing the feature channels.
- **Residual Blocks:** A series of residual blocks to capture high-level features.
- **Upsampling Layers:** Transposed convolutional layers that increase the spatial dimensions back to the original size.
- **Final Convolution:** A convolution layer with a Tanh activation to output the final image.

### Discriminator
The Discriminator model uses a PatchGAN architecture:
- **Convolutional Layers:** Several convolutional layers with increasing feature channels to classify real vs. fake patches of the image.
- **Instance Normalization:** Applied to intermediate layers to stabilize training.

## Sample Results
Some generated samples from different epochs are displayed below:

- **Epoch 32:**  
  ![Epoch 32](Results/Epoch_32.png)

- **Epoch 33:**  
  ![Epoch 33](Results/Epoch_33.png)

- **Epoch 36:**  
  ![Epoch 36](Results/Epoch_36.png)

- **Epoch 41:**  
  ![Epoch 41](Results/Epoch_41.png)

## How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/CycleGAN.git
   cd CycleGAN
   ```
