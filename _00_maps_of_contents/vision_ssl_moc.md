### [[dl_moc]]
# **Vision & Self-Supervised Learning**

**Tags**:: #vision #learning
**Links**:: [[math_moc]] · [[learning_methods_moc]] · [[sequence_llms_moc]] · [[generative_moc]] · [[rl_embodied_moc]]

Visual representation learning — from convolutional networks to vision transformers — and the self-supervised objectives (contrastive, masked, distillation) that have made large-scale label-free pre-training the standard for vision.

## Visual Architectures
---
- **[[CNN]]** — Convolutional Neural Networks: convolution math, locality and translation equivariance, pooling, 1×1 convolutions; LeNet → AlexNet → VGG → ResNet (skip connections) → EfficientNet (compound scaling); object detection (YOLO, NMS)
- **[[ViT]]** — Vision Transformer: patch embedding, CLS token, position encodings; inductive bias tradeoffs vs. CNN; scaling behavior; ViT Registers; DINO pre-training unlocks emergent segmentation
- **[[U-Net]]** — Encoder-decoder with skip connections; originally for biomedical segmentation; backbone for diffusion model denoisers (DDPM, LDM); see also [[generative_moc]]

## Self-Supervised Learning Objectives
---
- **[[Contrastive Learning]]** — InfoNCE / NT-Xent loss; SimCLR, MoCo, BYOL; positive pairs via augmentation; negative sampling; the augmentation set encodes target invariances
- **[[DINO]]** — Self-distillation with no labels: student-teacher with EMA weights, centering + sharpening to avoid collapse; emergent semantic segmentation in attention maps; DINOv2 at scale
- **[[MAE]]** — Masked Autoencoder: 75% patch masking, asymmetric encoder-decoder (encoder sees visible patches only), MSE reconstruction loss; 3–4× faster than BEiT; strong fine-tuning backbone

## Multimodal Vision-Language
---
- **[[CLIP]]** — Contrastive Language-Image Pre-Training: dual encoder (ViT + text Transformer), InfoNCE over image-text pairs; zero-shot classification; foundation for VLMs and text-to-image retrieval; see also [[sequence_llms_moc]]

## Transfer & Low-Data Regimes
---
- **[[FSL]]** — Few-Shot Learning: meta-learning, prototypical networks, Siamese networks; in-context learning as emergent few-shot; cross-reference [[learning_methods_moc]] for the general framework

## Dataview
---
```dataview
table tags as "Tags"
from "_01_notes"
WHERE contains(tags, "vision")
SORT file.name ASC
```
