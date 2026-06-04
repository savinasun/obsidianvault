### [[dl_moc]]  
# **Evidence Lower Bound (ELBO)**

**Tags**::  
**Links**::

## Variational Lower Bound
---
Let $X$ and $Z$ be random variables jointly distributed with distribution $p_{\theta}(X,Z)$, denote $p_{\theta}(X)$ as the marginal distribution of $X$ and $p_{\theta}(Z|X)$ be the conditional distribution of $Z$ given $X$. Then for any distribution $q_{\phi}$ and a sample $x$ drawn from $p_{data}$, the ELBO is defined as
$$
\begin{aligned}
L(\phi, \theta; x) &\coloneqq \mathbb{E}_{z\sim q_{\phi}(\cdot|x)} \left[ \ln \frac{ p_{\theta}(x,z) }{q_{\phi}(z|x)} \right] \\
&= \ln p_{\theta}(x) - D_{KL}(q_{\phi}(z|x)\|p_{\theta}(x,z)).
\end{aligned}
$$
