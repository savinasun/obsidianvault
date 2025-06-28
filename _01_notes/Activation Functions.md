### [[dl_moc]]  
## **Activation Functions**

**Tags**::  #dl_theo
**Links**::

---

Activation functions are used to introduce non-linearities in models. Without non-linearities, models can only map linear relationships in data, which severely limits their the expressive powers.

### Sigmoid
$$h(x) = \sigma (x) = \frac{1}{1+e^{-x}}$$
+ another name is called logistic function
+ $\sigma'(x) = \sigma(x)(1-\sigma(x))$
+ $\sigma(x) = 1-\sigma(-x)$
+ Image of sigmoid function and it's derivative function
	
	 ![[Pasted image 20250628001017.png]]

### Tanh
$$h(x)=tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$$
+ $tanh'(x) = 1-tanh^2(x)$
+ Tanh is superior than Sigmoid in near all aspects, so don’t use sigmoid unless it is the output layer
+ Image of tanh function and it's derivative function
	
	 ![Better Activation functions: (tanh > sigmoid) - MLS Resources -  DeepLearning.AI](https://global.discourse-cdn.com/dlai/original/3X/8/d/8d35a615be329bcf868b187fc8c43e4cd7e80fc9.png)

### Softmax
$$\sigma(\mathbf{z})_i =\frac{e^{z_i}}{\Sigma_{j=1}^{n} e^{z_j}}, \quad i=1, \cdots, n.$$
+ Softmax is commonly used to convert logits to probability distribution, for example in classification with [[Cross entropy]] or to determine weights in [Attention Mechanism].
+ Softmax tends to amplify the variances of high variance vectors.

### ReLu
$$\mathrm{ReLu}(x) = max(0, x)$$
+ ReLu is the most commonly used activation function, if not sure what to use, just use it; 
+ it is fast for learning since there is no gradient vanish as the output closes to 1
    - LeNet uses sigmoid, AlexNet uses ReLu
- Image of ReLu and Swish function and their derivative functions 
	![[Pasted image 20250628004816.png|800]]

