### [[dl_moc]]  
## **Activation Functions**

**Tags**::  #dl_theo
**Links**:: 

---

Activation functions are used to introduce non-linearities in models. Without non-linearities, models can only map linear relationships in data, which severely limits their the expressive powers. We list a few popular activation functions here and there are more could be seen at [@du2024].

### Sigmoid
$$h(x) = \sigma (x) = \frac{1}{1+e^{-x}}$$
+ another name is called logistic function
+ $\sigma'(x) = \sigma(x)(1-\sigma(x))$
+ $\sigma(x) = 1-\sigma(-x)$
+ Image of sigmoid function and it's derivative function
	
	 ![[activation_functions_sigmoid.png]]

### Softmax
$$\sigma(\mathbf{z})_i =\frac{e^{z_i}}{\Sigma_{j=1}^{n} e^{z_j}}, \quad i=1, \cdots, n.$$
+ Softmax is commonly used to convert logits to probability distribution, for example in classification with [[Cross entropy]] or to determine weights in [Attention Mechanism].
+ Softmax tends to amplify the variances of high variance vectors.

### Tanh (Hyperbolic Tangent)
$$h(x)=tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$$
+ $tanh'(x) = 1-tanh^2(x)$
+ Tanh is superior than Sigmoid in near all aspects, so don’t use sigmoid unless it is the output layer
+ Image of tanh function and it's derivative function
	
	 ![Better Activation functions: (tanh > sigmoid) - MLS Resources -  DeepLearning.AI](https://global.discourse-cdn.com/dlai/original/3X/8/d/8d35a615be329bcf868b187fc8c43e4cd7e80fc9.png)

### ReLU (Rectified Linear Unit)
$$\mathrm{ReLu}(x) = max(0, x)$$
+ ReLU is the most commonly used activation function, if not sure what to use, just use it; 
+ it is fast for learning since there is no gradient vanish as the output closes to 1
    - LeNet uses sigmoid, AlexNet uses ReLu
- Image of ReLU and Swish function and their derivative functions 
	![[activation_functions_relu.png|800]]
+ **Swish**:  A derived modification from SiLU$$Swish_{\beta}(x) = x\cdot \sigma(\beta x) = \frac{x}{1+e^{-\beta x}}$$
### GELU (Gaussian Error Linear Unit)
$$
GELU(x) = x\cdot\Phi(x)
$$
where $\Phi(x)$ is the c.d.f. of standard Gaussian distribution.


### SwiGLU (Swish Gated Linear Unit)

$$
SwiGLU(x, W, V, b, c\beta) = Swish_{\beta}(xW+b) \otimes (xV+c)
$$

SwiGLU together with GeGLU were first developed as variants of GLU in [@shazeer2020]. The basic idea is to simply combine GLU with Swish or GELU, replacing the sigmoid gate with either a Swish gate or a GELU gate.

![[Pasted image 20250628231848.png]]

SwiGLU has since been adopted in a number of prominent LLMs including Google’s PaLM [22](https://dublog.net/blog/all-the-activations/#fn:22) and Meta’s Llama2 [23](https://dublog.net/blog/all-the-activations/#fn:23),
