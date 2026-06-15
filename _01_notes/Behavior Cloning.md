### [[dl_moc]] · [[rl_embodied_moc]]  
# **Behavior Cloning**

**Tags**:: #rl
**Links**:: [[Reinforcement Learning]] · [[Deep Reinforcement Learning]] · [[Policy Gradient Methods]] · [[RLHF]]


**Behavior Cloning (BC)** is the simplest form of imitation learning: treat expert demonstrations as supervised learning data and train a policy $\pi_\theta$ to mimic observed actions.

## Learning from Demonstration
---
At each time step $t$, our goal is to learn a policy $\pi$ that uses the given observation $o_t$ to predict an optimal action:
$$
a_t \sim \pi_\theta(\cdot \mid o_t)
$$
where actions $a_t$ are commands sent to a controller and observations $o_t = [i_t, j_t]$ may include image and robot joint state.

The policy is learned from a dataset of expert demonstrations $\mathcal{D} = \{\tau_1, \ldots, \tau_n\}$ where each trajectory $\tau_i = [(o_0, a_0), \ldots, (o_T, a_T)]$ consists of observation–action pairs collected by a proficient agent.

**Training objective** (for continuous actions):
$$
\mathcal{L}(\theta) = \mathbb{E}_{(o,a) \sim \mathcal{D}}\left[\|a - \pi_\theta(o)\|^2\right]
$$

## The Covariate Shift Problem
---
BC works well in training but **fails at test time** due to **distributional shift** (also called **covariate shift**):

1. The expert demonstrations cover states the expert visits: $p_{\text{expert}}(s)$
2. The trained policy makes small errors → drifts into states not seen in training
3. At those out-of-distribution states, errors compound → **cascading failures**

The error accumulates **quadratically**: a policy with per-step error $\epsilon$ over a horizon $T$ incurs $O(\epsilon T^2)$ total error. This makes BC unreliable for long-horizon tasks.

**Intuition**: training with an expert driver, the student never sees what to do after a mistake, because the expert never makes that mistake.

## DAgger: Dataset Aggregation
---
**DAgger** (Ross & Bagnell, 2011) fixes covariate shift by iteratively querying the expert at states the *learner* visits:

1. Train initial policy $\pi_1$ on demonstration data $\mathcal{D}_1$
2. **Repeat** for iteration $i$:
   a. Roll out policy $\pi_i$ to collect states $s_1, \ldots, s_T$
   b. Query the expert to label those states: $(s_t, a^*_t = \pi_\text{expert}(s_t))$
   c. Aggregate: $\mathcal{D}_{i+1} = \mathcal{D}_i \cup \{(s_t, a^*_t)\}$
   d. Retrain policy $\pi_{i+1}$ on $\mathcal{D}_{i+1}$

**Key result**: DAgger reduces compounding error from $O(\epsilon T^2)$ to $O(\epsilon T)$ — linear rather than quadratic.

**Limitation**: requires an interactive expert (human or oracle) that can label states on demand, which is expensive.

## Imitation Learning Spectrum
---

| Method | Data source | Expert interaction | Error bound |
|--------|------------|-------------------|-------------|
| **Behavior Cloning** | Fixed offline demos | None after data collection | $O(\epsilon T^2)$ |
| **DAgger** | Online rollouts + expert labels | Online labeling required | $O(\epsilon T)$ |
| **GAIL** | Offline demos | None | Adversarial (GAN) |
| **Offline RL** | Fixed offline dataset | None | Depends on coverage |
| **RLHF** | Human preference labels | Pairwise comparisons | — |

## Connection to Offline RL
---
BC is a special case of **offline RL** where the reward signal is not used at all — we only copy the behavior. Offline RL methods (e.g., IQL, TD3+BC, Decision Transformer) extend this by leveraging reward structure while staying within the behavioral support of the dataset.

**Conservative approaches** penalize out-of-distribution actions that weren't in the dataset, preventing the policy from exploiting distribution shift in the value function.

## Relationship to RLHF
---
In [[RLHF]], BC is used as the **initialization step**: a language model is first fine-tuned via supervised learning on demonstration data (high-quality human-written completions), then refined with RL using a learned reward model. Starting from a well-behaved BC policy makes RL more stable and reduces the risk of reward hacking.

## Practical Considerations for Robotics
---
- **Action chunking**: predicting a sequence of $k$ future actions rather than one step reduces compounding error by reducing the effective horizon
- **Diffusion policies**: model $\pi_\theta(a \mid o)$ as a denoising diffusion process over action space, enabling multimodal action distributions
- **Data quality over quantity**: BC is highly sensitive to demonstration quality; noisy or inconsistent demonstrators hurt more than helping
- **Observation design**: the robot's observation $o_t$ must contain enough information for the task — partial observability causes systematic BC failures even with perfect demonstrations
