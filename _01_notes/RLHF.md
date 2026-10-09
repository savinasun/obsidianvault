### [[dl_moc]]  
# **Reinforcement Learning from Human Feedback (RLHF)**

**Tags**:: #rl #llm
**Links**:: [[Policy Gradient Methods]] · [[PPO]] · [[DPO]] · [[GRPO]] · [[Reinforcement Learning]] · [[Behavior Cloning]] · [[LLMs]] · [[Elo Score]]


**RLHF** is the technique used to align language models with human intent by incorporating human preference judgments into the training loop. It is the method behind InstructGPT, ChatGPT, Claude, and most modern instruction-following LLMs.

## Motivation
---
Pre-trained LLMs are trained to predict the next token — not to follow instructions, be helpful, or avoid harmful outputs. RLHF bridges this gap by optimizing for **human preference** rather than next-token likelihood.

The core challenge: human preferences are hard to specify as a reward function (what makes a response "good"?), but humans can reliably *compare* two responses and say which is better.

## Three-Stage Pipeline
---

### Stage 1: Supervised Fine-Tuning (SFT)
Train the base LLM on a curated dataset of high-quality (prompt, response) pairs collected from human demonstrators. This is pure [[Behavior Cloning]]:
$$
\mathcal{L}_\text{SFT}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_\text{demo}}[\log \pi_\theta(y \mid x)]
$$
The SFT model $\pi_\text{SFT}$ is the starting point — it can follow instructions but isn't yet optimized for quality.

### Stage 2: Reward Model Training
Collect a **preference dataset**: for each prompt $x$, show humans two responses $(y_w, y_l)$ and ask which is better. This produces pairs where $y_w \succ y_l$ (preferred over loser).

Train a **reward model** $r_\phi(x, y)$ using the **Bradley-Terry preference model**:
$$
\mathcal{L}_\text{RM}(\phi) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]
$$
The reward model is typically initialized from the SFT model with the final token prediction head replaced by a scalar reward head.

Bradley-Terry is the same paired-comparison model that underlies the [[Elo Score]] — the reward model effectively learns a per-response Elo rating (an analogy made explicitly by Christiano et al., 2017).

### Stage 3: RL Fine-Tuning with PPO
Use [[PPO]] to maximize expected reward while staying close to the SFT model:
$$
\mathcal{L}_\text{RL}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\left[r_\phi(x, y) - \beta \log\frac{\pi_\theta(y\mid x)}{\pi_\text{SFT}(y\mid x)}\right]
$$

The **KL penalty** $\beta \log(\pi_\theta / \pi_\text{SFT})$ prevents the RL policy from exploiting the reward model (reward hacking) by drifting too far from the SFT distribution.

## Reward Hacking
---
A critical failure mode: the policy finds ways to maximize $r_\phi$ that don't reflect true human preferences. For example, generating very long responses (if annotators rate length favorably), using flattering phrases, or producing confident-sounding but incorrect text.

The KL penalty $\beta$ controls the tradeoff between reward optimization and staying in-distribution. Higher $\beta$ → safer but less optimized; lower $\beta$ → more reward hacking risk.

## Constitutional AI (CAI)
---
Anthropic's approach ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)): instead of human preference labels at scale, use a set of **principles** (a "constitution") to have the model critique and revise its own outputs (RLAIF — RL from AI Feedback):

1. Generate initial response
2. Critique using a principle from the constitution
3. Revise based on the critique
4. Use the revised responses as SFT data or as preference pairs for RM training

Reduces reliance on human labeling while maintaining alignment.

## Direct Preference Optimization (DPO)
---
DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)) eliminates the explicit reward model and PPO training entirely. It derives that the optimal RLHF policy satisfies:
$$
\pi^*(y \mid x) \propto \pi_\text{ref}(y \mid x) \exp\!\left(\frac{1}{\beta} r^*(x,y)\right)
$$
Rearranging, the **implicit reward** can be expressed in terms of the policy ratio:
$$
r^*(x,y) = \beta \log \frac{\pi^*(y\mid x)}{\pi_\text{ref}(y\mid x)} + \beta \log Z(x)
$$
Substituting into the Bradley-Terry loss, the normalizing constant $Z(x)$ cancels, giving the DPO loss:
$$
\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w\mid x)}{\pi_\text{ref}(y_w\mid x)} - \beta \log \frac{\pi_\theta(y_l\mid x)}{\pi_\text{ref}(y_l\mid x)}\right)\right]
$$

**Benefits**: simpler, more stable, no RL machinery needed, same preference data as RLHF.

**Limitations**: sensitive to the reference model quality; can be brittle if the preference data distribution doesn't cover the policy's output distribution well.

See [[DPO]] for the full derivation, the gradient, and adoption in Zephyr and Llama 3.

## GRPO (Group Relative Policy Optimization)
---
Used in [DeepSeek-R1](https://arxiv.org/abs/2501.12948) and other reasoning models. Rather than a separate reward model, GRPO uses **verifiable rewards** (e.g., math answer correctness) and groups $G$ responses to the same prompt:
$$
\hat{A}_i = \frac{r_i - \text{mean}(r_{1:G})}{\text{std}(r_{1:G})}
$$
This normalizes advantages within the group, removing the need for a critic network. See [[GRPO]] for the full GRPO loss (GRPO itself is agnostic to the reward source: DeepSeekMath used a trained reward model, while DeepSeek-R1-Zero used rule-based verifiable rewards) and [[Policy Gradient Methods]] for the comparison with PPO and DPO.

## Comparison

| Method | Reward signal | Policy training | Complexity |
|--------|--------------|----------------|-----------|
| **RLHF (PPO)** | Learned reward model | PPO | High |
| **DPO** | Implicit (from preferences) | Supervised loss | Low |
| **GRPO** | Verifiable (rule-based) | Policy gradient | Medium |
| **CAI / RLAIF** | AI-generated preferences | PPO or DPO | Medium |
