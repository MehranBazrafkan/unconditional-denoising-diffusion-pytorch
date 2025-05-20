
# unconditional-denoising-diffusion-pytorch

A PyTorch implementation of an **unconditional denoising diffusion probabilistic model** for image generation. This repository features a UNet-based architecture and a complete training pipeline to synthesize images from noise without any conditional inputs.

----------

### Features

-   UNet backbone optimized for diffusion processes
    
-   Custom Gaussian diffusion scheduler with configurable timesteps
    
-   Training pipeline including dataset loading, augmentation, and checkpointing
    
-   Real-time image generation and sampling during training for visual monitoring
    
-   Modular, extensible PyTorch codebase ideal for research and experimentation
    

----------

### Use Cases

-   Unconditional image synthesis from noise
    
-   Exploration and study of diffusion probabilistic models in PyTorch
    
-   Foundation to extend towards conditional or guided diffusion models
    

----------

### Requirements

-   Python 3.13.2
    
-   PyTorch `2.6.0+cu126`
    
-   torchvision `0.21.0+cu126`
    
-   Pillow `11.1.0`
    
-   denoising-diffusion-pytorch `2.1.1`
    

Install dependencies via:

bash

CopyEdit

`pip install -r requirements.txt` 

⚠️ Ensure your Python environment supports CUDA 12.6 for the `+cu126` versions. Otherwise, install CPU-compatible packages.

----------

### Scripts

#### 🔁 `train_diffusion.py`

Main training script utilizing the UNet backbone.

bash

CopyEdit

`python train_diffusion.py` 

-   Loads images from the dataset folder
    
-   Trains the model with configurable parameters
    
-   Saves checkpoints and generated sample images during training
    

#### 🖼️ `augment_images.py`

Optional preprocessing script to augment the dataset.

bash

CopyEdit

`python augment_images.py` 

-   Performs image augmentations (flipping, resizing, normalization)
    
-   Saves augmented images to a specified output directory