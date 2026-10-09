### [[dl_moc]]
# **LLaMA (Large Language Model Meta AI)**

**Tags**:: #llm #llama #architecture
**Links**:: [[LLMs]] · [[Attention]] · [[RoPE]] · [[RMSNorm]] · [[Activation Functions]] · [[Transformer]]

## Architecture
---

+ [[RMSNorm]] & [[LLMs#Pre-Layer Normalization V.S. Post-Layer Normalization|Pre-Layer Normalization]]
+ [[RoPE|Rotary Positional Encodings]]
+ [[Attention#Multi-Head Attention (MHA)|MHA]] with [[LLaMA#KV Cache|KV Cache]]
+ [[Attention#Grouped-Query Attention|Grouped Multi-Query Attention]]
+ [[Activation Functions#Swish Gated Linear Unit|SwiGLU]] in FFC

There are four submodules forming the blocks of LLaMa decoding layer:
![[llama_01.png]]

![5552e5181b504239b9ffcefe2da0bf39](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/llamadecoderlayer.svg)


## Hyperparameters
---
 Llama 1 & 2 Params
	![[llama_02.png]]


## MLP Layer
---
The MLP layer in LLaMa uses [[Activation Functions#Swish Gated Linear Unit (SwiGLU)|SwiGLU]] activation functions
$$
FFN_{SwiGLU}(x, W_1, W_2, W_3) = (Swish_1(xW_1)\otimes xW_3)W_2
$$
The following figure, which is quoted from [@zotero-item-132], shows the SwiGLU architecture in the FFN layer,
![fabef926d9d446079bd05a7619455342|500](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/swiglu.svg)

The following is quoted from [@zotero-item-184],

```python
class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
```



## KV Cache
---
![[llama_03.png|500]]
+ Number of entries in caching: 
$$
2\cdot n\cdot d_h \cdot n_h \cdot \ell,
$$
+ $n$ is the number of tokens in the input sequence
+ $d_h$ is dimension of key/value matrices
+ $n_h$ is number of attention heads per layer
+ $\ell$ is number of layers

For DeepSeek R1-V3 architecture, 
$$d_h=128, n_h=128, \ell=61, n=10000$$
the caching size is around $4MB$ per token


## Grouped-Query Attention (GQA)
---
+ **Grouped-Query attention (GQA)**, is an evolution of MQA and strikes a balance by using an intermediate number of key-value heads (more than one but fewer than the query heads). The GQA model efficiently breaks the **query** into `n_heads` segments like the original **multi-head attention**, and the key and value are divided into `n_kv_heads`groups, ==enabling multiple query heads to share the same key-value head.==
![[llama_mha_variants.png]]
![[llama_gqa_comparison.png]]
DeepSeek R1 uses Multi-Head Latent Attention (MLA)

## Llama 3
---
[Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783) — **The Llama 3 Herd of Models** — scales to 405B parameters with a 128K token context window, natively supports multilinguality, coding, reasoning, and tool use. Llama 3 retains the core architectural choices of Llama 2 (GQA, RoPE, SwiGLU) while scaling pre-training data to ~15T tokens and adopting a more carefully curated post-training pipeline (SFT + rejection sampling + [[DPO]]). Multimodal (image, video, speech) variants were trained via a compositional approach, integrating modality-specific encoders into the frozen language backbone.