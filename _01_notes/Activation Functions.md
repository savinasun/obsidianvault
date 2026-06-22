### [[dl_moc]]
# **Activation Functions**

**Tags**:: #architecture #dl_theo
**Links**:: [[Transformer]] · [[LLaMA]] · [[Neural Network Initialization]] · [[LLMs]]

Activation functions are used to introduce non-linearities in models. Without non-linearities, models can only map linear relationships in data, which severely limits their the expressive powers. We list a few popular activation functions here and there are more could be seen at [@du2024].

## Sigmoid
---
$$h(x) = \sigma (x) = \frac{1}{1+e^{-x}}$$
+ Another name is called **logistic function**
+ $\sigma'(x) = \sigma(x)(1-\sigma(x))$
+ $\sigma(x) = 1-\sigma(-x)$
+ Image of sigmoid function and it's derivative function
	
	 ![[activation_functions_02.png]]


## Softmax
---
$$\sigma(\mathbf{z})_i =\frac{e^{z_i}}{\Sigma_{j=1}^{n} e^{z_j}}, \quad i=1, \cdots, n.$$
+ Softmax is commonly used to convert logits to probability distribution, for example in classification with [[Entropy#Cross Entropy|cross entropy]] or to determine weights in [[Attention]].
+ Softmax tends to amplify the variances of high variance vectors.


## Hyperbolic Tangent (Tanh)
---
$$h(x)=\tanh(x)=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$$
+ $\tanh'(x) = 1-tanh^2(x)$
+ Tanh is superior than Sigmoid in near all aspects, so don’t use sigmoid unless it is the output layer
+ Image of tanh function and it's derivative function
	![[activation_functions_04.png]]


## Rectified Linear Unit (ReLU)
---
$$\mathrm{ReLU}(x) = \max(0, x),$$
$$\textrm{FF}(x) = \max(0, xW_1)W_2$$
+ ReLU is the most commonly used activation function, if not sure what to use, just use it; 
+ it is fast for learning since there is no gradient vanish as the output closes to 1
    - LeNet uses sigmoid, AlexNet uses ReLu
- Image of ReLU and Swish function and their derivative functions 
	![[activation_functions_03.png]]
+ Notable models using ReLU: [[Transformer|original transformer]], T5, Chinchilla, Gopher
+ **Swish**:  A derived modification from SiLU$$\textrm{Swish}_{\beta}(x) = x\cdot \sigma(\beta x) = \frac{x}{1+e^{-\beta x}}$$

## Gaussian Error Linear Unit (GELU)
---
$$
\textrm{GELU}(x) = x\cdot\Phi(x),
$$
$$
\textrm{FF}(x) = \textrm{GELU}(xW_1) W_2,
$$where $\Phi(x)$ is the c.d.f. of standard Gaussian distribution.
+ Notable models using GELU: [[GPTs|GPTs]], Bloom
+ **GELU has a significantly smoother gradient transition than the sharp and abrupt ReLU.**
+ For negative values, GELU function is not monotonic, instead, it decreases at first then increases as it becomes close to zero.
![[activation_functions_06.png|800]]



## Swish Gated Linear Unit (SwiGLU)
---
 + SwiGLU together with GeGLU were first developed as variants of Gated Linear Unit (GLU) in [Shazeer, 2020](https://arxiv.org/abs/2002.05202). The basic idea is to simply combine GLU with Swish or GELU, replacing the sigmoid gate with either a Swish gate or a GELU gate
$$
\textrm{SwiGLU}(x, W, V, \beta) = \textrm{Swish}_{\beta}(xW) \otimes (xV),
$$
$$
\textrm{FF}_{\textrm{SwiGLU}}(x, W, V, \beta, W_2) = \textrm{Swish}_{\beta}(xW) \otimes (xV)W_2.
$$
+ SwiGLU has since been adopted in a number of prominent LLMs including Google’s PaLM ([Chowdhery et al., 2022](https://arxiv.org/abs/2204.02311)) and Meta’s Llama 2 ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)) — see also [[LLaMA]]
![[activation_functions_05.png]]
+ Notable models using SwiGLU: LLaMA, PaLM, Mistral, most models post 2023



## ReGLU
---
+ In ReGLU, it modify the first part of a FF layer $$
\textrm{FF}(x) =  \textcolor{red}{\max(0, xW_1)}W2.
$$Instead of simply using a linear projection together with [[Activation Functions#Rectified Linear Unit (ReLU)|ReLU]], it augments the above with an entry-wise linear term$$
	\textcolor{red}{\max(0, xW_1)}\to\max(0, xW_1)\otimes(xV).
	$$This gives the gated variant ReGLU, which gates the vanilla ReLU using an extra parameter $V$$$
\textrm{FF}_{\textrm{ReGLU}}(x) = (\max(0, xW_1)\otimes xV)W_2.
$$
+ Notable models using ReGLU: T5 v1.1, mT5, Phi3, Gemma2, Gemma3

