### [[dl_moc]] · [[vision_ssl_moc]]  
# **Vision Transformer (ViT)**

**Tags**:: #vision #architecture
**Links**:: [[Transformer]] · [[DINO]] · [[CLIP]] · [[Attention]] · [[Contrastive Learning]]

Prior to ViT, Convolutional Neural Networks (CNNs) were the undisputed standard for image recognition. CNNs use filters with local receptive fields to build up a hierarchical representation of an image.

The ViT paper ([Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)) challenged this convention by asking: Can we apply a standard Transformer directly to an image with minimal modifications? To do this, they had to solve one key problem: Transformers expect a 1D sequence of tokens (like words), but images are 2D grids of pixels.

The solution was to **reshape the image into a sequence of flattened patches**.

## Architecture
---
The ViT architecture can be broken down into a few key steps:
![[ViT.png]]
### A. Image Patching and Embedding

1.  **Split into Patches:** An input image is divided into a grid of fixed-size, non-overlapping patches. For example, a 224x224 image might be split into a 14x14 grid of 16x16 pixel patches.
2.  **Flatten and Project:** Each patch (e.g., 16x16x3 pixels) is flattened into a long vector (16 * 16 * 3 = 768 dimensions) and then linearly projected into a lower-dimensional embedding. This is the equivalent of a "word embedding" in NLP.

### B. Positional and `[CLS]` Token

1.  **Positional Embeddings:** Because the Transformer architecture is permutation-invariant (it doesn't inherently know the order of tokens), the model would have no sense of spatial layout. To fix this, **learnable positional embeddings** are added to the patch embeddings to retain positional information.
2.  **`[CLS]` Token:** Inspired by the BERT model in NLP, a special, learnable `[CLS]` (classification) token is prepended to the sequence of patch embeddings. The final output state of this token from the Transformer is used as the aggregate image representation for classification.

The final input sequence fed to the Transformer looks like this:
`[CLS_embedding], (patch_1_embedding + pos_1_embedding), (patch_2_embedding + pos_2_embedding), ...`

### C. Transformer Encoder

This sequence of vectors is then processed by a standard Transformer encoder, composed of alternating layers of:
* **Multi-Head Self-Attention:** This mechanism allows every patch to weigh the importance of and exchange information with every other patch in the image. This enables the model to capture **long-range, global dependencies**.
* **Feed-Forward Network (MLP):** A simple multi-layer perceptron applied independently to each token's representation.

Finally, a classification head (a simple MLP) is attached to the output `[CLS]` token to produce the final prediction.

---

## 3. Key Findings and Impact 🌍

* **Data Thirst and Inductive Bias:** The most critical finding was that ViTs lack the strong **inductive biases** inherent to CNNs (like translation equivariance and locality). As a result, when trained on smaller datasets like ImageNet-1k (~1M images), ViTs performed worse than comparable CNNs. However, when pre-trained on massive proprietary datasets like JFT-300M (~300M images), they **surpassed the performance of state-of-the-art CNNs**.
* **Scalability:** The performance of ViTs scaled more effectively with more data and larger model sizes compared to their CNN counterparts.
* **A New Foundation:** The ViT paper fundamentally changed the landscape of computer vision. It demonstrated that a general-purpose sequence processing architecture could outperform specialized vision models, paving the way for nearly all the follow-up research we've discussed, including [[DINO]], Registers, and countless other variants.



- Image Patch Tokenization
    
    - effective sequence length: $N = \frac{HW}{P^2}$
    - resolution of each image patch (P,P)
    - model dimension: D
- Pre-train on large image dataset (JFT300M), fine-tune to smaller downstream tasks
    
    - for mid-sized image dataset, inductive bias plays an important role, it is not the same case on large training datasets
    - better fine-tune on higher resolution than pre-training:
        - same patch size, larger effective sequence length
            
        - 2D interpolation of the pre-trained position embeddings, according to original position in image
            
            **← the only inductive bias injected in ViT**
            
- Learnable 1d position embeddings
    
    - not observe significant gains from using 2D-aware position embeddings




# Technical Deep Dive: "Vision Transformers Need Registers"

This note explores the findings of the paper ["Vision Transformers Need Registers"](https://arxiv.org/abs/2309.16588) by Darcet et al., 2023. The paper identifies a key issue with standard Vision Transformers (ViTs) — poor quality feature maps in early layers — and proposes a simple, effective solution: adding "registers" as extra input tokens.

---

## 1. The Core Problem: Noisy Feature Maps in ViTs

While models like [[DINO]] and DINOv2 produce excellent semantic features in their final layers, the feature maps from the *early and middle layers* of many ViTs are often noisy and lack clear spatial meaning.

* **Cause:** The authors discovered that a few background patches with high energy (high L2 norm) tend to dominate the self-attention mechanism. These "attention-hogging" patches capture the attention of many other tokens, preventing the model from building up meaningful, object-focused representations in its initial layers.
* **Effect:** This results in messy attention maps that don't correspond well to objects. For example, the `[CLS]` token, which is supposed to aggregate global information, ends up attending to these noisy background patches instead of the main subject of the image.



This is a significant issue for dense prediction tasks like semantic segmentation, which rely on high-quality, spatially precise feature maps from multiple layers of the network.

---

## 2. The Solution: Token Registers

The proposed solution is surprisingly simple: add a small number of extra, learnable tokens to the input sequence of the transformer. These are called **registers**.

* **What are Registers?** Registers are learnable vectors, just like the `[CLS]` token. However, unlike patch embeddings, they are not associated with any specific part of the image. They are simply appended to the sequence of patch embeddings before the first transformer block.

A standard ViT input sequence looks like this:
`[CLS], patch_1, patch_2, ..., patch_n`

With registers, the input sequence becomes:
`[CLS], reg_1, reg_2, reg_3, reg_4, patch_1, patch_2, ..., patch_n`

* **Function:** These registers act as a "scratchpad" or an information buffer for the model. Their purpose is to attract and store the noisy, global information that would otherwise be stored in the patch embeddings.

---

## 3. The Mechanism: How Registers Clean Up Features

The registers work by effectively "soaking up" the unwanted background attention.

1.  **Attracting Noise:** During training, the registers learn to attend to the high-energy background patches that were previously causing noise.
2.  **Freeing Up Patch Tokens:** By offloading this responsibility, the registers allow the actual patch tokens to focus their attention on other, more semantically related patches (i.e., other parts of the main object).
3.  **Cleaner Representations:** This leads to a cleaner flow of information through the network. The feature maps in the early and middle layers become much more semantically meaningful, with clear object boundaries and less background noise.

The key insight is that the model *needs* a place to store this global, non-localized information. Without registers, it's forced to use the patch tokens, corrupting their spatial features. Registers provide a dedicated space for this information, preserving the integrity of the patch representations.

---

## 4. Key Results and Impact 🚀

Adding just a few registers (e.g., 4) to a ViT architecture leads to significant improvements with negligible computational cost.

* **Improved Downstream Performance:** Models trained with registers show notably better performance on dense tasks like semantic segmentation.
* **Better `[CLS]` Token Attention:** The attention maps of the `[CLS]` token become much cleaner, resembling the high-quality segmentation seen in DINO, but now present from the very first layer.
* **Broad Applicability:** This technique is effective across different ViT sizes, training paradigms (supervised, self-supervised), and datasets.

In conclusion, registers provide a simple, low-cost, and powerful modification to the ViT architecture that directly addresses a fundamental flaw in how the model processes spatial information, leading to more robust and useful features.