### [[dl_moc]]  
## **Neural Network Initialization**
**Tags**:: #architecture 
**Links**::

- **Cons of Zero Initialization**
    
    - [**Symmetry Breaking versus Zero Initialization**](https://community.deeplearning.ai/t/symmetry-breaking-versus-zero-initialization/16061)
        
    - Gradient vanishing/explosion → a problem for deep neural network
        
        ![[Pasted image 20250513105524.png]]
        
        gradient explosion: gradient clipping ??
        
- **Random Initialization**
    
    - break the symmetry
        
    - $W \sim N(0, 1)$ or $W \sim U(-0.1, 0.1)$
        
    - set the scaling factor to be 1 for each layer / set the
        
        → solve the vanishing gradient problem
    
- **Xavier Initialization**: torch::nn::init::xavier_normal_
    
    ![[xavier_initialization.png]]
    ![[Pasted image 20250513105559.png]]
    
- **Kaiming Initialization**: torch::nn::init::kaiming_normal_
    
    ![[Pasted image 20250513105622.png]]
    helps VGG nets train from scratch without stage-wise training
    
