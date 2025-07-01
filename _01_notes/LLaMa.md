### [[dl_moc]]  
## **LLaMa**

**Tags**::  
**Links**::

---

### LLaMa Architecture

![[llama_encoder_architecture.png | 500]]
+ [[RMSNorm]] & Pre-Norm
+ [[RoPE|Rotary Positional Encodings]]
+ Self-Attention with KV Cache
+ Grouped Multi-Query Attention
+ [[Activation Functions#SwiGLU (Swish Gated Linear Unit)|SwiGLU]] in FFC


 Llama 1 & 2 Params
	![[llama12_params.png]]

There are four submodules forming the blocks of LLaMa decoding layer:
![5552e5181b504239b9ffcefe2da0bf39](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/llamadecoderlayer.svg)

### FeedForward Layer (MLP Layer)
The MLP layer in LLaMa2 uses [[Activation Functions#SwiGLU (Swish Gated Linear Unit)|SwiGLU]] activation functions
$$
FFN_{SwiGLU}(x, W_1, W_2, W_3) = (Swish_1(xW_1)\otimes xW_3)W_2
$$
The following figure, which is quoted from [@zotero-item-132], shows the SwiGLU architecture in the FFN layer,
![fabef926d9d446079bd05a7619455342|500](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/_images/swiglu.svg)

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
