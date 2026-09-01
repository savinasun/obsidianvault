### [[dl_moc]]  
# **U-Net**

**Tags**:: #generative #architecture
**Links**:: [[DDPM]] · [[LDM]] · [[Score-Based Generative Modeling Methods]]

## U-Net Architecture
---
The U-net architecture. 

![[unet_architecture.png|600]]

Each blue square is a feature map with the number of channels labeled on top and the height x width dimension labeled on the left bottom side. The gray arrows mark the shortcut connections. (Image source: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597), U-Net: Convolutional Networks for Biomedical Image Segmentation)

#### Downsampling
Each step consists of the repeated application of two 3x3 convolutions (unpadded convolutions), each followed by a ReLU and a 2x2 max pooling with stride 2. At each downsampling step, the number of feature channels is doubled.