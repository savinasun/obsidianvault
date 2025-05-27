### [[dl_moc]]  
### Vision Transformer
**Tags**::  #transformer 
**Links**::

- [[Dosovitskiy, et al., 2021](https://arxiv.org/abs/2010.11929) An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale]

![[Pasted image 20250513120824.png]]

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
