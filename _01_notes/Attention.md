### [[dl_moc]]  
## **Attention**

**Tags**::  
**Links**::

---
### Self-Attention
$$
\textrm{Attention}(Q,K,V) = \textrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)\cdot V
$$
where $d_k$ represents the dimension of matrix $K$.
#### Interpretations of Scaled Dot-Product Attention Function
calculate the **scaled dot-product similarities** among each input tokens
→ convert these similarities into percentages by softmax function
→ calculate the self-attention score by using these percentages to weight the values embedding

### Masked Self-Attention



### Cross-Attention (Encoder-Decoder Attention)
+ K,V from encoder
+ Q from decoder    
→ calculate encoder-decoder attention scores to generate text (these are from the Seq2Seq models)

**Multi-Modal Models**: encoder trained on images, speech → context aware embeddings → text based decoder to generate captions or audio-prompt responses

### Multi-Head Attention

$$
\textrm{MultiHead}(Q,K,V) = \textrm{Concat}(h_1, \dots, h_H)W^O
$$
where
$$
h_i = \textrm{Attention}(QW_i^Q, KW_i)
$$
![[attention_multihead.png]]

- $d_\textrm{model}$: dimension of the embedding vector
- $d_k=d_v$: dimension of each head
- $H= d_{\textrm{model}} / d_k$ is number of heads

```python
query_states = self.q_proj(hidden_states)
key_states = self.k_proj(hidden_states)
value_states = self.v_proj(hidden_states)

query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

# apply rotary positional embedding
kv_seq_len = key_states.shape[-2]
cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# repeat k/v heads if n_kv_heads < n_heads
key_states = repeat_kv(key_states, self.num_key_value_groups)
value_states = repeat_kv(value_states, self.num_key_value_groups)

attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
attn_output = torch.matmul(attn_weights, value_states)

attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

attn_output = self.o_proj(attn_output)

```

### Grouped-Query Attention
