### [[dl_moc]]  
# **Convolutional Neural Networks**

**Tags**:: #architecture #vision
**Links**:: [[ViT]] · [[U-Net]] · [[Neural Network Initialization]] · [[Normalization]] · [[Activation Functions]] · [[Regularization]] · [[ImageNet]]


**CNNs** (Convolutional Neural Networks) dominated computer vision from ~2012 to ~2020. Their core [[CNNs#Inductive Biases|inductive biases]] — **locality** and **translation equivariance** — are hard-coded into the architecture, making them highly data-efficient for image tasks. [[ViT|Vision Transformers]] have since matched or exceeded CNNs on large-scale datasets by learning these biases from data.

![[cnn_architecture_overview.png|700]]

## Core Operations
---
### Convolution
A filter $W \in \mathbb{R}^{f \times f \times C_\text{in}}$ slides across the input and computes dot products at each location:
$$
Z[i,j,k] = \sum_{u=0}^{f-1}\sum_{v=0}^{f-1}\sum_{c=0}^{C_\text{in}-1} X[si+u,\, sj+v,\, c] \cdot W_k[u,v,c]
$$

- **$f$**: filter/kernel size (e.g., 3×3, 5×5)
- **$s$**: stride (step size); larger stride → smaller output
- **$p$**: padding (zeros added to border); `same` padding: $p = (f-1)/2$ keeps spatial size constant
- **Output size**: $\lfloor(n + 2p - f)/s\rfloor + 1$

### Pooling
Reduces spatial resolution with no learned parameters:
- **Max pooling**: $\max$ over each $f \times f$ window — preserves strongest activations
- **Average pooling**: average over the window — used in global average pooling at the end of a network to collapse spatial dims

### $1 \times 1$ Convolution
A $1 \times 1$ filter applies a learned linear combination across channels at each spatial position — effectively a per-pixel fully-connected layer. Used to:
- **Change channel dimension** ($C_\text{in} \to C_\text{out}$) without touching spatial dimensions
- Add non-linearity between depthwise convolutions (MobileNet)
- Reduce computation (bottleneck in ResNet, Inception)

## Inductive Biases
---
CNNs hard-code two inductive biases:

**Locality**: each output unit depends on a small *receptive field* of the input. Filters learn local features (edges, textures) that are compositioned hierarchically into global features (objects).

The receptive field $r_0$ of the final layer feature map with respect to the input is computed as:
$$
r_0 = \sum\limits_{i=1}^L \left((k_i -1)\prod\limits_{j=1}^{i-1}s_j \right)+1
$$
![[cnn_receptive_field_diagram.png|186]]

**Translation equivariance**: applying the same filter at every position → a shifted input produces a shifted output. (Pooling breaks exact equivariance → approximate translation *invariance*.)

**Parameter sharing**: the same filter weights are applied at every spatial location — dramatically reduces parameters vs. a fully-connected layer.

These biases explain why CNNs work well with limited data. [[ViT|Vision Transformers]] lack them but learn analogous patterns given enough data.

## Classic Architectures
---
### LeNet-5 ([LeCun et al., 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf))
The original CNN. ~60K parameters. `Conv → Pool → Conv → Pool → FC → FC → output`. Demonstrated that learned features outperform hand-crafted features for digit recognition.

### AlexNet ([Krizhevsky et al., 2012](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf))
Sparked the deep learning revolution by winning ImageNet 2012 by a large margin. ~60M parameters.
- First large-scale use of **ReLU** (faster training than sigmoid/tanh)
- Trained on two GPUs (model parallelism)
- **Dropout** for regularization

### VGG-16/19 ([Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556))
~138M parameters. Used only **3×3 convolutions** throughout — two stacked 3×3 convs have the same receptive field as one 5×5 but fewer parameters and more non-linearities. Showed deeper is better if kept simple and uniform.

### ResNet ([He et al., 2015](https://arxiv.org/abs/1512.03385))
Introduced **skip connections** (residual connections) that let the gradient flow directly:
$$
y = F(x, W) + x
$$
The layer only needs to learn the **residual** $F(x) = y - x$ rather than the full transformation. This solves the **degradation problem**: adding more layers without skip connections hurts training (not overfitting — the training loss itself worsens). ResNets trained with 50, 101, and 152 layers.

![[cnn_residual_depth_accuracy.png|208]]

Skip connections are now universal: used in [[Transformer|Transformers]], [[U-Net|U-Nets]], [[DDPM|diffusion models]], etc.

### EfficientNet ([Tan & Le, 2019](https://arxiv.org/abs/1905.11946))
Jointly scales **depth** (layers), **width** (channels), and **resolution** (input size) via a **compound scaling** coefficient $\phi$:
$$
d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi, \quad \text{s.t.} \; \alpha\beta^2\gamma^2 \approx 2
$$
The constraint ensures compute doubles with each $\phi$ increment. Achieves better accuracy-efficiency tradeoff than scaling any single dimension.

ImageNet accuracy vs. receptive field size and accuracy vs. throughput across major architectures:
![[cnn_imagenet_receptive_field.png|328]]
![[cnn_architecture_throughput.png|450]]
## Object Detection
---
Detection extends classification to predicting **bounding boxes** and **class labels** for multiple objects.

**YOLO** ([Redmon et al., 2015](https://arxiv.org/abs/1506.02640)): divide image into $S \times S$ grid; each cell predicts $B$ bounding boxes + class probabilities in a **single forward pass** — enabling real-time detection.

**Non-Max Suppression (NMS)**: after prediction, discard overlapping boxes:
1. Keep box with highest confidence $p_c$
2. Discard remaining boxes with IoU ≥ 0.5 with the kept box
3. Repeat

**R-CNN family**: propose candidate regions first, then classify each region. Slower but more accurate for dense scenes.

## Neural Style Transfer
---
[Gatys et al., 2015](https://arxiv.org/abs/1508.06576). Uses a pretrained CNN (VGG) to transfer the *style* of one image onto the *content* of another by optimizing a generated image against two CNN-derived losses.

**Content loss**: measures how much the generated image $x$ differs from the content image $p$ at layer $l$ — directly compares feature activations.

**Style loss**: measures style similarity via the **Gram matrix** of feature maps $F^l \in \mathbb{R}^{N_l \times M_l}$ (channels × spatial positions):
$$
G_{ij}^l = \sum\limits_{k} F_{ik}^{\ell} F_{jk}^{\ell}.
$$
The Gram matrix $G^l_{ij}$ captures pairwise channel correlations — the "texture statistics" — independent of spatial layout. Style loss minimizes the difference between the Gram matrices of the generated and style images.

**Total loss** combines content and style terms:
$$
\mathcal{L}_{total}(\vec{p},\vec{a},\vec{x}) = \alpha \mathcal{L}_{content}(\vec{p},\vec{x})+\beta\mathcal{L}_{style}(\vec{a},\vec{x})
$$
where $\alpha$ controls content fidelity and $\beta$ controls style strength. Optimization is run on the generated image pixels via gradient descent.

## Transition to ViTs
---
[[ViT|Vision Transformers]] ([Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)) replaced the convolution inductive bias with global self-attention. Key tradeoffs:

| | CNN | ViT |
|--|-----|-----|
| Inductive bias | Strong (locality, equivariance) | Weak (learned from data) |
| Data efficiency | High (works on ImageNet-1k) | Low (needs JFT-300M or DINO pre-training) |
| Scaling | Limited | Excellent |
| Long-range | Limited (by receptive field) | Full (global attention from layer 1) |

Modern vision models like [[DINO]]/DINOv2 use ViT backbones. **CNNs remain competitive for small-data regimes**, mobile deployment (MobileNet), and medical imaging ([[U-Net]]).
