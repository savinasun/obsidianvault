### [[dl_moc]]  
# **Vision Language Models**

**Tags**::  
**Links**::

## Architecture
--- 
The Qwen2.5-VL model [@bai2025] builds upon the foundation laid by the Qwen series. The model has three components:
![[vlm_qwen_01.png]]
- **Large Language Model:** the 1D ROPE is replaced with **MROPE(Multimodal Rotary Position Embedding)** aligned to absolute time.
- **Vision Encoder:** redesigned ViT with 2D-RoPE and window attention to support variable input resolutions and improve computational efficiency. During the training and inference the height and width of the images are resized to multiples of 28, and the image patches are generated with a stride of 14.
- **MLP-based Vision Language Merger:** compresses the feature sequences before feeding them into the LLM by grouping spatially adjacent sets of patch features, concatenating them, and projecting them into a dimension aligned with the LLM’s text embeddings using a two-layer MLP.





+ Visual encoder
+ Cross-Modal Projector
+ LLM
+ Omni & MoE
+ Fine-grained perception models
+ Resolution scaling




![[vlm_qwen_02.png]]



## Architecture Innovations
---
**Window Attention:** A primary enhancement is the implementation of window attention in the visual encoder, optimizing inference efficiency by linearly scaling computational costs with the number of patches rather than quadratically. While Qwen2 VL used full attention, Qwen2.5-VL uses a combination of full attention and window attention, leading to better performance and efficiency.

**Dynamic FPS Sampling:** Qwen2.5-VL extends dynamic resolution to the temporal dimension with dynamic FPS sampling. This enables comprehensive video understanding across varying sampling rates by aligning MROPE IDs directly with timestamps, thus understanding temporal dynamics without additional computational overhead. The former model lacked dynamic FPS sampling and did not handle videos as efficiently.

**Multimodal Rotary Position Embedding (MROPE) Alignment:** The multimodal rotary position embedding (MROPE) is also aligned to the absolute time with the help of a novel strategy that allows the model to better comprehend temporal dynamics without additional computational overhead.


## Downstream Tasks
---
+ Image-Text Retrieval
+ **Visual QA (VQA)**: Enhance AI systems to answer questions about image content with grounding.
+ Visual Reasoning
+ Visual Entailment
+ **Image Captioning**: Automatically describe images for accessibility tools or content generation.
+ **Scene Analysis**: Analyze spatial relationships between objects in visual datasets.


+ **General Vision-Language Task:** assesses comprehensive multimodal capabilities of LMMs
	+ **VQA:** Visual question answering, to enhance AI systems to answer questions about image content with grounding.
	+ **GQA:** Question answering on image scene graphs.
	+ **SQA:** ScienceQA is a dataset collected from elementary and high school science curricula, consisting of 21,208 multimodal multiple-choice science questions. Out of these, 10,332 questions include image context, 10,220 include text context, and 6,532 include both. Most questions are annotated with lectures (17,795) and detailed explanations (19,184) to provide general knowledge and specific reasoning for the correct answers. The dataset spans three subjects: natural science, language science, and social science, and is organized into 26 topics, 127 categories, and 379 skills.
	+ **SEED-Bench-image**
	+ **LLaVA-Bench (In-the-Wild)**
	+ **MM-Vet:** MM-Vet assesses a model’s ability to engage in visual conversations across various tasks, evaluating both the accuracy and helpfulness of responses using GPT-4. The dataset includes 200 images and 218 questions, each paired with corresponding ground truth answers.
	+ VQAv2 GQA VizWiz SQAI VQAT MME MMBen SEEDI LLaVAW MM-Vet
+ **Visual Hallucination Task:** evaluates whether the response of the model is consistent with the image content to ensure the trustworthiness and reliability of the model
	+ Example benchmarks: CHAIR (Rohrbach et al., 2018), POPE (Li et al., 2023c), and MMHal-Bench (Sun et al., 2023)
+ **Vision-Centric Task:** evaluates visual understanding capabilities, such as determining the spatial relationship between objects in the image
	+ Example benchmarks: MMVP (Tong et al., 2024b), CV-Bench2D, and CV-Bench3D (Tong et al., 2024a).