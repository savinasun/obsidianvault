### [[dl_moc]]  
# **Scaling Laws**

**Tags**:: #dl_theo #llm
**Links**:: [[LLMs]] · [[Optimization Algorithms]] · [[MoE]] · [[Double Descent]]


**Scaling laws** describe how model performance (typically measured by loss) changes as a function of model size $N$, training data $D$, and compute budget $C$. Understanding these relationships guides decisions about model architecture, training duration, and resource allocation.

## OpenAI Scaling Laws (Kaplan et al., 2020)
---
Kaplan et al. found that language model loss follows clean **power laws** in $N$, $D$, and $C$:

$$
L(N) \sim N^{-\alpha_N}, \quad L(D) \sim D^{-\alpha_D}, \quad L(C) \sim C^{-\alpha_C}
$$

with $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$.

**Key findings**:
- Performance improves smoothly and predictably with scale
- Model size $N$ has the largest impact; dataset size $D$ matters less than model size (given compute)
- Given a fixed compute budget, it's better to train a **larger model for fewer steps** than a smaller model to convergence

This last point was later revised by Chinchilla.

## Chinchilla Scaling Laws (Hoffmann et al., 2022)
---
DeepMind's **Chinchilla** paper corrected the earlier conclusion: prior large models (GPT-3, Gopher) were **significantly undertrained** — too large for their dataset size.

**Compute-optimal training**: for a given compute budget $C \approx 6ND$ (in FLOPs, where 6 accounts for the forward + backward pass):
$$
N_\text{opt} \propto C^{0.5}, \quad D_\text{opt} \propto C^{0.5}
$$

The optimal strategy: **scale $N$ and $D$ in equal proportion**. The rule of thumb: **train on ~20 tokens per parameter**.

| Model | Parameters | Training tokens | Tokens/param |
|-------|-----------|----------------|-------------|
| GPT-3 | 175B | 300B | ~1.7 |
| Gopher | 280B | 300B | ~1.1 |
| **Chinchilla** | 70B | 1.4T | **20** |
| LLaMA-2 70B | 70B | 2T | ~29 |
| LLaMA-3 70B | 70B | 15T | ~214 |

**Chinchilla** (70B, 1.4T tokens) outperforms Gopher (280B) on all benchmarks, using the same compute — validating the compute-optimal hypothesis.

**Modern practice**: LLaMA-3 and similar models overtrain beyond Chinchilla optimal (more tokens per parameter) because inference is expensive — a smaller, overtrained model is cheaper to deploy than a larger, compute-optimal model.

## Loss as a Function of N and D
---
Hoffman et al. model loss as:
$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

where:
- $E$: irreducible loss (entropy of natural language $\approx$ 1.69 nats)
- $A/N^\alpha$: contribution from limited model capacity ($\alpha \approx 0.34$)
- $B/D^\beta$: contribution from limited data ($\beta \approx 0.28$)

This allows extrapolating the loss of a larger model before training it — enabling principled model design at scale.

## Emergent Abilities
---
**Emergent abilities** (Wei et al., 2022) are capabilities that appear abruptly at a certain scale — not present in smaller models, but suddenly appearing in larger ones. Examples:
- Multi-step arithmetic
- Chain-of-thought reasoning (see [[CoT Prompting]])
- Few-shot translation to rare languages
- Code generation

**Controversy**: Schaeffer et al. (2023) argue that emergent abilities are an artifact of **discontinuous evaluation metrics** (e.g., exact-match accuracy on multi-step tasks). Using continuous metrics (log-likelihood), they find smooth scaling curves. This suggests "emergence" may be a measurement artifact rather than a fundamental phenomenon.

## Beyond Language Models
---
Scaling laws have been found to hold across modalities:

| Domain | Scaling finding |
|--------|----------------|
| Vision (ViT) | Loss scales with model size and image resolution |
| Code generation | Similar power law structure to language |
| Multimodal | Separate scaling for vision encoder vs. language decoder |
| RL | Less clean — environment complexity breaks simple laws |

## Compute Frontiers
---
**FLOPs for a forward pass**: $\approx 2N$ multiply-adds (each weight used once in a matmul, each matmul contributes $2\times$ for multiply + add). Training a token costs $\approx 6N$ FLOPs (forward + backward, with 1:2 ratio). See [[FLOPs]] for details.

**The scaling wall**: there are potential limits to data scaling (internet-scale data is finite), but synthetic data generation (from models themselves) and multimodal data may extend the frontier.

## Implications for Architecture Design
---
Scaling laws suggest that **architecture details matter less than scale**:
- Transformers with many architectural variants (MoE, different attention patterns) all follow similar scaling laws
- [[MoE]] models scale compute-efficiently: by activating only a fraction of parameters per token, they offer better loss-per-FLOP than dense models
- The scaling exponent determines whether it's worth investing in architectural improvements vs. simply scaling up
