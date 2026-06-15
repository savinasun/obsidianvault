### [[dl_moc]]  
# **Mixture of Experts**

**Tags**:: #architecture #llm
**Links**:: [[Transformer]] · [[LLMs]] · [[Scaling Laws]] · [[FLOPs]]

## Architecture
---
**Mixture of Experts (MoE)** is a neural network architecture that leverages conditional 
computation to scale models efficiently. Instead of activating all parameters for every input, MoE activates only a subset of **"expert" networks** per example, allowing the model to scale capacity while keeping computation per input relatively low. See [[FLOPs]] for compute accounting.

Introduced by Jacobs et al. (1991, [Neural Computation](https://www.sciencedirect.com/science/article/pii/S0893608005800231)), MoE has seen a resurgence with large-scale models like **GShard**, **Switch Transformer**, and **GLaM**, which use sparse routing and efficient parallelization.

An MoE layer consists of:
- A **gating network** $G(x)$ that decides which experts to activate for a given input $x$.
- A set of **experts** (usually feedforward networks or Transformer blocks) $\{E_1, \dots, E_K\}$.

The output is typically a weighted sum of the top-k selected experts:
$$\text{MoE}(x) = \sum_{i \in \text{TopK}(G(x))} G_i(x) \cdot E_i(x)$$


## Types of Gating Functions
---
1. **Soft Gating (original MoE)**  
   - Uses softmax over all experts  
   - High compute cost, used mainly in early works

2. **Sparse Gating (modern MoE)**  
   - Activates only top-k experts  
   - Example: **Switch Transformer** activates just 1 expert per token (top-1)

3. **Noisy Top-k Gating** (used in GShard)  
   - Adds Gaussian noise during training to encourage load balancing


## Variants and Implementations
---

| Model                                       | Features                                                      |
| ------------------------------------------- | ------------------------------------------------------------- |
| **GShard** ([Shazeer et al., 2020](https://arxiv.org/abs/2006.16668))  | Large-scale Transformer with top-2 gating                     |
| **Switch Transformer** ([Fedus et al., 2021](https://arxiv.org/abs/2101.03961)) | Simplified MoE with top-1 gating, very scalable               |
| **GLaM** ([Du et al., 2022](https://arxiv.org/abs/2112.06905))         | Top-2 experts, trained with mixture-of-experts at large scale |
| **BASE Layers** ([Lewis et al., 2021](https://arxiv.org/abs/2103.16716)) | Balanced training objective for stable MoE                    |

### Pros
- **High parameter count** without increasing FLOPs significantly.
- **Conditional computation** leads to better efficiency on large-scale workloads.
- Easily parallelizable across GPUs/TPUs via expert parallelism.
### Cons
- **Load balancing**: Uneven expert usage can lead to underutilization or overload.
- **Routing instability**: Noisy gating and sparse selection can cause training instability.
- **Memory and communication overhead**: Especially in distributed training setups.


To ensure even distribution of input tokens across experts, common techniques include:
- **Auxiliary loss** for load balancing (used in GShard/Switch)
- **Token dropping/clipping** to avoid expert overload
- **Noise injection** in gating scores


## Use Cases
---
- **Language models** (e.g., GLaM, Switch Transformer) — see [[LLMs]]
- **Multimodal models** (e.g., Flamingo, CoMoE)
- **Recommendation systems**
- **Translation and multilingual modeling**


## References
---
- Jacobs et al. (1991): [Adaptive Mixtures of Local Experts](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)
- Shazeer et al. (2017): [Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538)
- Shazeer et al. (2020): [GShard: Scaling Giant Models with Conditional Computation](https://arxiv.org/abs/2006.16668)
- Fedus et al. (2021): [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- Du et al. (2022): [GLaM: Efficient Scaling with Mixture-of-Experts](https://arxiv.org/abs/2112.06905)
