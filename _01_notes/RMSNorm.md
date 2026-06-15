### [[dl_moc]] · [[learning_methods_moc]]  
# **Root Mean Square Layer Normalization**

**Tags**:: #dl_theo #llm
**Links**:: [[Layer Normalization]] · [[Normalization]] · [[LLMs]] · [[LLaMA]] · [[Transformer]]

### Definition
---
From [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467), we know that RMS Norm applies Root Mean Square [[Layer Normalization]] over a mini-batch of inputs.
$$
y_i = \frac{x_i}{\text{RMS}_{\epsilon}(x)}\cdot \gamma_i, \quad i=1,2,\dots,d
$$
where
$$
\text{RMS}_{\epsilon}(x) = \sqrt{\epsilon + \frac{1}{d} \sum\limits_{i=1}^d x_i^2},\ d=d_{\text{model}}.
$$
In matrix notation, the transformation could be represented as
$$
X \to \tilde{X}\cdot\Gamma,
$$
where
$$
\tilde{X} = X \oslash \sqrt{\frac 1 d H\odot H \mathbf{1}^d + \epsilon}
$$
and$$
\Gamma = \text{diag} (\gamma_1, \gamma_2, \dots, \gamma_d).
$$
### Parameters
---
1. Just like LayerNorm, we also have **learnable scaling parameters** $\gamma_i$, which is multiplied with the normalized input here in RMSNorm, it allows the network to learn the optimal scaling for the activations, adding flexibility to the normalization process.

2. It also helps with stabilizing training and accelerating convergence, similar to LayerNorm.


### Implementation
---
From [@zotero-item-191], a small constant value of $\epsilon=1e-6$ is used for numerical stability.

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```



