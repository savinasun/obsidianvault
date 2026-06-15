### [[dl_moc]]  
# **Recurrent Neural Networks**

**Tags**:: #architecture
**Links**:: [[Transformer]] · [[Mamba]] · [[Attention]] · [[Optimization Algorithms]] · [[Activation Functions]]


**RNNs** process sequences by maintaining a **hidden state** that is updated at each step. They were the dominant architecture for sequence modeling from ~2014 until the [[Transformer]] (2017) replaced them. Understanding RNNs is the foundation for understanding *why* Transformers (full attention) and [[Mamba]] (selective state spaces) were developed.

## Vanilla RNN
---
At each timestep $t$, the hidden state $h_t$ is updated from the previous state $h_{t-1}$ and current input $x_t$:
$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$
$$
y_t = W_{hy} h_t + b_y
$$

Weights $W_{hh}$, $W_{xh}$, $W_{hy}$ are **shared across all timesteps** — the same transformation is applied at every position (temporal weight sharing).

**Types of RNN architectures**:
- **One-to-many**: one input → sequence output (image captioning)
- **Many-to-one**: sequence input → one output (sentiment classification)
- **Many-to-many (same length)**: e.g., POS tagging
- **Many-to-many (different length)**: encoder-decoder for machine translation (Cho et al., 2014; Sutskever et al., 2014)

### Vanishing Gradient Problem
Backpropagating through $T$ timesteps involves products of $T$ Jacobians:
$$
\frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}
$$
Each factor is $W_{hh}^\top \cdot \text{diag}(\tanh'(h_{t-1}))$. If the spectral radius of $W_{hh}$ is $< 1$, gradients **vanish exponentially** → the model fails to learn long-range dependencies. If $> 1$, gradients **explode** → gradient clipping is required.

This is why vanilla RNNs practically only capture **local context** effectively.

## LSTM (Long Short-Term Memory)
---
LSTMs ([Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)) address vanishing gradients by introducing a **cell state** $c_t$ that flows through time with only element-wise multiplications — a "highway" for gradients.

Four gates (all using sigmoid $\sigma$ or tanh):

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) && \text{(forget gate: what to erase from cell state)} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) && \text{(input/update gate: what new info to write)} \\
\tilde{c}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) && \text{(candidate value to add)} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) && \text{(output gate: what to expose as hidden state)}
\end{aligned}
$$

**Cell state update** (the key innovation):
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

**Hidden state** (what gets passed to the next cell and used for prediction):
$$
h_t = o_t \odot \tanh(c_t)
$$

The additive update to $c_t$ (via $i_t \odot \tilde{c}_t$) creates an **uninterrupted gradient path** — the gradient of $c_t$ w.r.t. $c_{t-1}$ is just $f_t$ (element-wise), making it much easier to propagate over long distances.

**Intuition for gates**:
- **Forget gate** $f_t$: 0 = erase cell state completely, 1 = keep it unchanged
- **Input gate** $i_t$: controls how much of the candidate $\tilde{c}_t$ to add
- **Output gate** $o_t$: controls how much of the cell state to expose as $h_t$

## GRU (Gated Recurrent Unit)
---
GRU (Cho et al., 2014) is a simplified LSTM with **two gates** and no separate cell state:

$$
\begin{aligned}
z_t &= \sigma(W_z [h_{t-1}, x_t]) && \text{(update gate: blend old and new)} \\
r_t &= \sigma(W_r [h_{t-1}, x_t]) && \text{(reset gate: how much of past to forget)} \\
\tilde{h}_t &= \tanh(W [r_t \odot h_{t-1}, x_t]) && \text{(candidate hidden state)} \\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t && \text{(interpolate old and new)}
\end{aligned}
$$

GRU is computationally cheaper (fewer parameters) and empirically performs similarly to LSTM. Preferred when compute is limited.

## Why Transformers Replaced RNNs
---
| | RNN/LSTM | Transformer |
|--|----------|-------------|
| **Sequential dependency** | Each step depends on previous → hard to parallelize | All positions computed simultaneously |
| **Long-range modeling** | Exponential decay of gradient signal | Direct attention between any two positions |
| **Training speed** | Slow (sequential backprop) | Fast (highly parallelizable) |
| **Memory** | Fixed-size hidden state | KV cache grows with context |
| **Inductive bias** | Recency bias (local context easier) | Uniform distance bias (global context) |

The fundamental bottleneck: RNNs must **compress** all past context into a fixed-size hidden state. The [[Attention|attention mechanism]] eliminates this bottleneck by directly attending to all past tokens.

## Sequence Decoding
---
RNN-based seq2seq models (encoder-decoder for machine translation) need to **decode** a sequence token-by-token from the output distribution. Two main strategies:

**Greedy decoding**: at each step $t$, pick the most likely token:
$$
\hat{y}_t = \arg\max_{w} P(y_t = w \mid y_{<t}, x)
$$
Fast but suboptimal — a locally best token may lead to a poor overall sequence.

**Beam search**: maintain the top-$k$ (beam width) most probable partial sequences simultaneously. At each step, expand each beam, compute joint probabilities, and prune back to $k$ candidates:
$$
\text{score}(y_{1:t}) = \log P(y_1) + \log P(y_2 \mid y_1) + \cdots + \log P(y_t \mid y_{<t})
$$
Length normalization: $\text{score}_\text{norm} = \text{score}(y_{1:T}) / T^\alpha$ (typically $\alpha = 0.6$–$0.7$) prevents beam search from preferring shorter sequences.

**Beam search tradeoffs**:
- Larger $k$ → more exploration, higher quality, quadratic cost
- $k = 1$ = greedy decoding; $k \to \infty$ = approximate exhaustive search
- Works poorly for very long sequences (combinatorial growth)

For open-ended generation (LLMs), beam search is rarely used — **sampling** with temperature, top-$k$, or nucleus sampling (top-$p$) is preferred to increase diversity.

## Beyond RNNs: Mamba / SSMs
---
[[Mamba]] (Gu & Dao, 2023) revives the recurrent idea with a key improvement: **selective state spaces** where the state transition parameters $(B, C, \Delta)$ depend on the input content. This gives:
- $O(1)$ inference memory (fixed state, like an RNN)
- $O(L)$ training via parallel scan (not sequential like an RNN)
- Content-based selection (unlike fixed-transition SSMs like S4)

SSMs occupy the middle ground between RNNs (efficient inference, lossy compression) and Transformers (exact attention, $O(L^2)$ complexity).
