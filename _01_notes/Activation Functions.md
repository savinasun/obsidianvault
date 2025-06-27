### [[dl_moc]]  
## **Activation Functions**

**Tags**::  #dl_theo
**Links**::

---

Activation functions are used to introduce non-linearities in models. Without non-linearities, models can only map linear relationships in data, which severely limits their the expressive powers.

1. Sigmoid
    
2. Softmax: a function converts a set of numbers to a set of probabilities sum up to 1
    
    Softmax tends to amplify the variances of high variance vectors
    
3. Tanh
    - Tanh is superior than Sigmoid in near all aspects, so don’t use sigmoid unless it is the output layer
    - $tanh’(x) = 1 - tanh^2(x)$

4. ReLu
    - The most commonly used activation function, if not sure what to use, just use it; it is fast for learning since there is no gradient vanish as the output closes to 1
    - LeNet uses sigmoid, AlexNet uses ReLu
