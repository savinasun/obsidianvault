### [[dl_moc]]  
# **Direct Preference Optimization (DPO)**

**Tags**:: #rl #llm
**Links**:: [[Policy Gradient Methods]] · [[RLHF]] · [[PPO]] · [[GRPO]] · [[KL Divergence]] · [[Behavior Cloning]] · [[Elo Score]] · [[LLMs]] · [[LLaMA]]

**Direct Preference Optimization (DPO)** optimizes the same KL-regularized reward-maximization objective as RLHF but needs no reward model, no critic, and no sampling from the policy during training: a change of variables turns the reward-learning loss into a classification loss directly on the policy ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290), "Your Language Model Is Secretly a Reward Model"). Its only inputs are a fixed dataset of preference pairs $(x, y_w, y_l)$ (winner, loser) and a frozen reference model $\pi_\text{ref}$ — the SFT model, or a model fit to the preferred completions when no SFT model is available. In LLM practice it replaced the PPO stage in recipes such as Zephyr and Llama 3. See [[RLHF]] for the SFT → reward model → policy pipeline that DPO collapses into one step.

Primary sources: [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290) for the algorithm and results; [Tunstall et al., 2023](https://arxiv.org/abs/2310.16944) (Zephyr) and [Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783) (Llama 3) for LLM adoption.

![[policy_gradient_methods_01.png]]

*The slide above is titled "Direct Policy Optimization"; the method in the paper is Direct **Preference** Optimization.*

## Derivation
---
DPO starts from the KL-regularized objective of RLHF (*DPO Eq. 3*), whose optimum has a closed form (*Eq. 4*):
$$\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal D,\,y\sim\pi_\theta(\cdot|x)}\big[r(x,y)\big]-\beta\,D_\text{KL}\big[\pi_\theta(y|x)\,\|\,\pi_\text{ref}(y|x)\big]\quad\Longrightarrow\quad \pi_r(y|x)=\frac{1}{Z(x)}\,\pi_\text{ref}(y|x)\exp\!\Big(\tfrac1\beta r(x,y)\Big)$$
where $Z(x)=\sum_y \pi_\text{ref}(y|x)\exp(\tfrac1\beta r(x,y))$ is intractable. Inverting gives the reward as a function of the policy (*Eq. 5*): $r(x,y)=\beta\log\frac{\pi_r(y|x)}{\pi_\text{ref}(y|x)}+\beta\log Z(x)$. The Bradley–Terry probability $p(y_w\succ y_l|x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$ depends only on a reward *difference*, so $\beta\log Z(x)$ cancels, and maximum-likelihood fitting of $\pi_\theta$ to the preference data gives (*Eq. 7*)
$$\mathcal{L}_\text{DPO}(\theta)=-\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\Big)\Big]$$
The policy *is* the reward model: $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$ is the implicit reward, and $\beta$ plays the role of RLHF's KL coefficient.

```python
def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta=0.1):    # sequence log-probs (summed over tokens)
    margin = beta * ((logp_w - ref_w) - (logp_l - ref_l))   # implicit-reward gap r_hat(y_w) - r_hat(y_l)
    return -F.logsigmoid(margin).mean()
```

## What the Gradient Does
---
The gradient of the loss (*DPO §4*):
$$\nabla_\theta\mathcal{L}_\text{DPO}=-\beta\,\mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)}_{\text{larger when the implicit reward mis-orders the pair}}\Big(\nabla_\theta\log\pi_\theta(y_w|x)-\nabla_\theta\log\pi_\theta(y_l|x)\Big)\Big]$$
It raises the likelihood of $y_w$ and lowers that of $y_l$, weighted by how badly the implicit reward currently ranks the pair; the paper reports that dropping this weight makes the model degenerate (App. Table 3). It has the same $\nabla\log\pi$ form as a policy gradient (see [[Policy Gradient Methods#PPO vs DPO vs GRPO]]), but over a fixed pair of sequences instead of sampled rollouts.

## Offline vs On-Policy
---
[[PPO]] samples fresh responses from the current policy and scores them; DPO trains on pairs collected once (the paper samples $y_1,y_2\sim\pi_\text{ref}$ and has humans label them, and in practice often reuses public preference datasets), so it is off-policy and the pairs go stale as $\pi_\theta$ drifts from the model that wrote them. Llama 3 counters this by training each round on preference data collected from the most recent models, so that "our training data conforms better to the distribution of the policy model" ([Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783)).

## Results in the Paper
---
With models up to 6B parameters, DPO exceeded PPO-based RLHF at controlling the sentiment of generations (IMDb). On Reddit TL;DR summarization (GPT-J SFT model, GPT-4 as judge) it reached a ≈61% win rate against the reference summaries versus 57% for PPO at its best temperature, and in human evaluation DPO samples (temperature 0.25) were preferred over PPO samples (temperature 0) 58% of the time. On single-turn Anthropic-HH dialogue (Pythia-2.8B) it matched or beat the best-of-128 baseline. The authors caution that GPT-4 win rates depend on the judging prompt and that scaling beyond 6B was untested.

## Adoption in LLMs
---

+ **Zephyr-7B** ([Tunstall et al., 2023](https://arxiv.org/abs/2310.16944)) applies "distilled DPO" to preference pairs from AI feedback (UltraFeedback): no human annotation, no sampling during fine-tuning, a few hours of training, and it surpasses the RLHF-trained Llama 2-Chat-70B on MT-Bench.
+ **Llama 3** ([Grattafiori et al., 2024](https://arxiv.org/abs/2407.21783)) post-trains with SFT, rejection sampling, and DPO instead of PPO. The authors explored PPO but found that DPO "required less compute for large-scale models and performed better, especially on instruction following benchmarks like IFEval", using $\beta=0.1$ and learning rate $10^{-5}$. Two stabilizers: header and termination tokens are masked out of the loss (they appear in both the chosen and rejected response, so the contrastive loss gives them conflicting gradients), and an extra NLL loss with coefficient 0.2 on the chosen response keeps the output format and stops the chosen log-probability from falling.

## Relationship to Other Notes
---
- [[RLHF]] — DPO collapses the reward-model and PPO stages into one supervised step on the same Bradley–Terry preference data; RLHF covers the full three-stage pipeline DPO shortcuts.
- [[PPO]] — the method DPO replaces: PPO samples on-policy and needs a reward model and critic, while DPO trains offline on fixed pairs, with $\beta$ playing the role of the KL coefficient.
- [[GRPO]] — the other response to PPO's cost: GRPO stays on-policy and drops only the critic, whereas DPO drops the reward model and the sampling loop.
- [[Policy Gradient Methods]] — the unified-gradient table shows DPO's gradient coefficient next to PPO's and GRPO's, all of the form $GC\cdot\nabla\log\pi_\theta$.
- [[KL Divergence]] — the closed-form optimum of the KL-regularized objective is what lets the reward be rewritten in terms of the policy.
- [[Elo Score]] — the Bradley–Terry model whose reward *difference* the DPO loss reparameterizes, which is why the partition function cancels.
- [[Behavior Cloning]] — the SFT model is the reference $\pi_\text{ref}$, and Llama 3 adds an SFT-style NLL term on the chosen response to stabilize DPO.
- [[LLaMA]] — Llama 3's post-training recipe (SFT, rejection sampling, DPO) is the main large-scale use of DPO.
