### [[dl_moc]]  
# **Deep Reinforcement Learning**

**Tags**:: #rl
**Links**:: [[Reinforcement Learning]] · [[Policy Gradient Methods]] · [[Behavior Cloning]] · [[World Models]]


Deep RL combines the RL framework with deep neural networks as function approximators for policies and value functions — enabling RL to operate on raw observations (images, sensor data) at scale.

## The Bitter Lesson
---
> "We have to learn the bitter lesson that building in how we think we think does not work in the long run. The two methods that seem to scale arbitrarily… are **learning** and **search**."
> — [Rich Sutton, 2019](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)

Hand-crafted features and domain knowledge have been repeatedly outcompeted by general-purpose methods that leverage computation. This applies directly to RL: learned representations beat engineered state features.

## Deep Q-Network (DQN)
---
DQN ([Mnih et al., 2015](https://arxiv.org/abs/1312.5602)) extends Q-learning to work with neural networks by addressing two key instabilities: **correlation between consecutive samples** and **non-stationary targets**.

**Core components**:

1. **Experience Replay Buffer** — Store transitions $(s, a, r, s')$ in a replay buffer $\mathcal{D}$. Sample random mini-batches for training, breaking temporal correlations.

2. **Target Network** — A separate, slowly-updated copy $Q_{\bar\theta}$ of the main network $Q_\theta$. Targets are:
$$
y_t = r_t + \gamma \max_{a'} Q_{\bar\theta}(s_{t+1}, a')
$$
The target network is frozen for $C$ steps then copied from the main network.

3. **$\epsilon$-greedy exploration** — With probability $\epsilon$ (annealed from 1 to 0.1), select a random action; otherwise $a = \arg\max_a Q_\theta(s,a)$.

**Loss**:
$$
\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\left(y_t - Q_\theta(s,a)\right)^2\right]
$$

**DQN variants**: Double DQN (decouple action selection from evaluation), Dueling DQN (separate advantage and value streams), Prioritized Experience Replay.

## Actor-Critic Methods
---
Actor-Critic separates the policy (actor) and value function (critic) into two networks:

- **Actor** $\pi_\theta(a \mid s)$: learns to select actions
- **Critic** $V_\phi(s)$ or $Q_\phi(s,a)$: estimates return to reduce variance in policy gradient

The **advantage function** $A(s,a) = Q(s,a) - V(s)$ measures how much better action $a$ is than the average:
$$
\nabla_\theta J(\theta) \approx \mathbb{E}\left[A(s,a) \nabla_\theta \log \pi_\theta(a\mid s)\right]
$$

**A2C / A3C**: Synchronous / asynchronous workers each interact with their own environment copy and compute gradients, which are aggregated centrally.

## On-Policy: PPO
---
**Proximal Policy Optimization** ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) is the workhorse of modern deep RL. It constrains how much the policy can change per update by clipping the probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_\text{old}}(a_t\mid s_t)}$:

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\;\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

- $\hat{A}_t$: generalized advantage estimate (GAE)
- $\epsilon \approx 0.2$: clip range

Benefits: stable, simple to tune, works with continuous and discrete actions. See [[Policy Gradient Methods]] for the full PPO derivation and comparison with GRPO/DPO.

## Off-Policy: SAC (Soft Actor-Critic)
---
SAC ([Haarnoja et al., 2018](https://arxiv.org/abs/1801.01290)) maximizes a **maximum entropy** objective — reward *plus* entropy of the policy:
$$
J(\pi) = \sum_t \mathbb{E}\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t))\right]
$$
where $\alpha$ is the temperature parameter (can be auto-tuned).

**Benefits**:
- Off-policy → high sample efficiency via replay buffer
- Entropy bonus encourages exploration and prevents premature convergence
- Robust to hyperparameters

**Comparison**:

| | PPO | SAC |
|---|---|---|
| Policy type | On-policy | Off-policy |
| Action space | Discrete/continuous | Continuous |
| Sample efficiency | Lower | Higher |
| Stability | High | High |
| Key use case | Games, robotics sim | Continuous control |

## Reward Shaping & Sparse Rewards
---
Sparse rewards (only signal at task completion) make learning extremely slow. Common solutions:

- **Reward shaping**: add intermediate dense rewards $F(s,s')$ that guide the agent without changing the optimal policy (potential-based shaping guarantees this)
- **Curiosity / intrinsic motivation**: reward novelty via prediction error (ICM) or count-based exploration
- **Hindsight Experience Replay (HER)**: re-label failed trajectories with the goal that *was* achieved, turning failures into useful data
- **Imitation from demonstration**: initialize from expert data via [[Behavior Cloning]] before fine-tuning with RL

## From Simulation to Reality
---
Deep RL is often trained in simulation (faster, cheaper, safer) and deployed in the real world. Key challenges:
- **Sim-to-real gap**: physics and visual differences cause policy failure
- **Domain randomization**: randomize sim parameters (friction, lighting, mass) so the policy learns to be robust
- **Privileged information**: train with extra state info available only in sim, distill to observation-based policy at deploy time
