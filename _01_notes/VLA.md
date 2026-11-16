### [[dl_moc]]  
# **Vision-Language-Action Models (VLA)**

**Tags**:: #rl #vision #llm #architecture
**Links**:: [[VLM]] · [[LLaVA]] · [[Behavior Cloning]] · [[Flow Matching]] · [[DiT]] · [[World Models]] · [[Deep Reinforcement Learning]] · [[RLHF]] · [[Robot Kinematics]] · [[Camera Calibration]]


A **Vision-Language-Action model (VLA)** is a robot policy built by adapting a pretrained [[VLM]] to output **actions**, conditioned on camera observations and a natural-language instruction. The term was coined by **RT-2** ([Brohan et al., 2023](https://arxiv.org/abs/2307.15818)). Mechanically, a VLA is [[Behavior Cloning]] at foundation-model scale: supervised learning on demonstration trajectories, with the VLM backbone contributing web-scale semantic priors that pure from-scratch imitation lacks.

## Why Build Policies from VLMs
---
- **Semantic generalization**: the VLM already knows what a "cactus" or "Taylor Swift" is — RT-2 showed policies inherit this, manipulating novel objects and following symbolic instructions never seen in robot data
- **Language as the task interface**: one policy, thousands of tasks, composable instructions
- **Actions as just another modality**: the same [[Transformer]] sequence model handles images, text, and (tokenized or embedded) actions — no bespoke architecture per task
- **Data leverage**: robot demonstrations are scarce and expensive; web-scale pre-training amortizes most of the representation learning

## Problem Setup
---
Learn $\pi_\theta(A_t \mid o_t, \ell)$ where:
- $o_t$ = one or more camera images $I_t^1, \dots, I_t^k$ (each camera with its own intrinsics and extrinsics — see [[Camera Calibration]]) + proprioceptive state $q_t$
- $\ell$ = language instruction
- $A_t = [a_t, a_{t+1}, \dots, a_{t+H-1}]$ = an **action chunk** of $H$ future actions (joint targets, end-effector deltas, gripper commands — see [[Robot Kinematics]] for the joint-space vs. task-space distinction)

Training maximizes likelihood of demonstrated chunks — standard [[Behavior Cloning]], inheriting its compounding-error analysis ($O(\epsilon T^2)$; see DAgger). **Action chunking** (from ACT, [Zhao et al., 2023](https://arxiv.org/abs/2304.13705)) mitigates this by shrinking the effective decision horizon by $H\times$, improving temporal consistency, and enabling high-frequency control from a slow model.

## The Central Design Axis: Action Representation
---
| Representation | Examples | Pros | Cons |
|---|---|---|---|
| **Per-dim binned tokens** (e.g. 256 bins) | RT-2, OpenVLA | reuses the LM head unchanged | slow AR decoding; discretization artifacts |
| **Compressed tokens** (DCT + BPE over the chunk) | π0-FAST | ~an order fewer tokens; AR training ~5× cheaper than flow | still autoregressive at inference |
| **Continuous regression** (L1/MSE head, parallel decode) | OpenVLA-OFT | single forward pass; fast | unimodal — averages over multimodal demos |
| **Diffusion head** | Octo, RDT-1B (cf. Diffusion Policy, [Chi et al., 2023](https://arxiv.org/abs/2303.04137)) | captures multimodal action distributions | iterative denoising at inference |
| **Flow-matching action expert** | π0, GR00T N1 | multimodal + few (~10) integration steps | extra expert weights; training interference with backbone |
| **Discrete diffusion / FM over tokens** (masked, parallel refinement) | Discrete Diffusion VLA ([2508.20072](https://arxiv.org/abs/2508.20072)) | backbone-native CE loss; any-order parallel decoding + re-masking error correction | quantization; multi-round inference — see [[Discrete Flow Matching]] |

RT-2 mapped bins to number strings (PaLI-X) or overwrote rarely-used vocabulary tokens; OpenVLA likewise overwrites the 256 least-used Llama tokens. OpenVLA-OFT ([Kim et al., 2025](https://arxiv.org/abs/2502.19645)) showed that swapping AR discrete decoding for **parallel decoding + action chunking + continuous L1 regression** raises LIBERO success from 76.5% → **97.1%** with **26×** inference throughput — evidence that the action-representation choice, not backbone quality, was the bottleneck in early VLAs.

## Landmark Models
---
| Model | Year | Backbone → Actions | Key fact |
|---|---|---|---|
| [RT-1](https://arxiv.org/abs/2212.06817) | 2022 | 35M EfficientNet+[[FiLM]], TokenLearner → binned tokens | pre-VLM precursor; 130k episodes, 700+ tasks, 13 robots, 3 Hz |
| [PaLM-E](https://arxiv.org/abs/2303.03378) | 2023 | 562B embodied multimodal LM → text plans | showed VLM scale transfers to embodied reasoning |
| [RT-2](https://arxiv.org/abs/2307.15818) | 2023 | PaLI-X 5B/55B, PaLM-E 12B → actions-as-text | coined "VLA"; co-fine-tuning with web data → emergent semantic skills; 1–3 Hz |
| [Open X-Embodiment](https://arxiv.org/abs/2310.08864) | 2023 | (dataset + RT-X models) | 1M+ trajectories, 22 embodiments; positive cross-embodiment transfer |
| [Octo](https://arxiv.org/abs/2405.12213) | 2024 | 27M/93M transformer → diffusion head | 800k OXE trajectories; readout tokens; open recipe |
| [OpenVLA](https://arxiv.org/abs/2406.09246) | 2024 | Prismatic 7B (SigLIP+DINOv2 fused → Llama-2) → binned tokens | 970k episodes; beats RT-2-X (55B) by 16.5% abs. at 7× fewer params |
| [π0](https://arxiv.org/abs/2410.24164) | 2024 | PaliGemma 3B + 300M action expert → flow matching, $H{=}50$ | ~10k hours, 7 robot configs, 68 tasks (+OXE/Bridge/DROID); up to 50 Hz; laundry folding |
| [π0-FAST](https://arxiv.org/abs/2501.09747) | 2025 | π0 backbone → FAST tokens | DCT+BPE action tokenizer; matches flow π0 with up to 5× less training compute |
| [GR00T N1](https://arxiv.org/abs/2503.14734) | 2025 | Eagle-2 VLM 1.34B + flow-matching [[DiT]], $H{=}16$ (2.2B total) | humanoid foundation model; System 2 @ 10 Hz, System 1 @ 120 Hz; data pyramid |
| [Gemini Robotics](https://arxiv.org/abs/2503.20020) | 2025 | Gemini 2.0 → VLA + ER variant | "embodied reasoning" model family; later on-device variant |
| [π0.5](https://arxiv.org/abs/2504.16054) | 2025 | π0 + hierarchical inference | open-world generalization: cleans kitchens/bedrooms in entirely unseen homes |
| [SmolVLA](https://arxiv.org/abs/2506.01844) | 2025 | ~450M compact VLA | trained on community LeRobot data; async inference; consumer-GPU deployable |

Industry landscape: Physical Intelligence (π), Google DeepMind, NVIDIA, Figure (**Helix**: 7B System 2 @ 7–9 Hz + 80M System 1 @ 200 Hz for humanoid upper-body control — blog only), Skild AI, Tesla — most converging on VLM-backbone + fast action head designs.

## π0's Flow-Matching Action Expert
---
π0 attaches a separate **action expert** (~300M params) to the frozen-vocabulary VLM: action tokens attend to the VLM's keys/values but use their own weights (a two-expert mixture, similar in spirit to [[MoE]] routing by modality). The expert is trained with conditional [[Flow Matching]] over the action chunk:

$$
A_t^\tau = \tau A_t + (1-\tau)\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
$$
\mathcal{L}(\theta) = \mathbb{E}\left[\left\| v_\theta(A_t^\tau, o_t) - (A_t - \epsilon) \right\|^2\right]
$$

i.e. regress the straight-line velocity from noise ($\tau{=}0$) to data ($\tau{=}1$) — the same convention as [[Flow Matching]]. π0 samples $\tau$ from a shifted Beta distribution emphasizing noisier timesteps. Inference integrates ~**10 Euler steps** ($\delta = 0.1$) — far fewer than DDPM-style heads — producing a 50-step action chunk in one shot.

**Knowledge insulation** ([Driess et al., 2025](https://arxiv.org/abs/2505.23705)): gradients from the freshly-initialized action expert degrade the pretrained VLM's semantics early in training. Fix: stop-gradient from the expert into the backbone, and keep the backbone trained with discrete FAST action tokens + web-data co-training — train fast *and* generalize.

## Dual-System Architectures
---
A recurring pattern (π0, GR00T N1, Helix): a large **System 2** VLM runs slowly (≈10 Hz or less) to interpret scene + instruction, while a small **System 1** action module runs fast (50–200 Hz) for closed-loop control, conditioned on the VLM's latents. This resolves the core tension: the most capable VLMs are too large for real-time control loops, but control only needs the semantics to update at task speed, not servo speed.

## Training Recipe
---
1. **Pre-training**: large cross-embodiment mixture — [Open X-Embodiment](https://arxiv.org/abs/2310.08864), [DROID](https://arxiv.org/abs/2403.12945) (76k trajectories, 564 scenes; per-scene camera extrinsics, re-audited post hoc — see [[Hand-Eye Calibration]]), proprietary teleoperation. GR00T's **data pyramid**: web data + human video (base) → simulation + synthetic augmentation (middle) → real robot teleop (peak, scarcest).
2. **Co-training with web data**: retain VLM abilities by mixing vision-language batches with robot batches (RT-2's co-fine-tuning; π0.5 adds subtask prediction and verbal-instruction data). Without it, action fine-tuning catastrophically forgets semantics.
3. **Post-training**: fine-tune on curated, task- or robot-specific demonstrations for dexterous skills. LoRA fine-tuning makes 7B-scale VLAs adaptable on a single consumer GPU (OpenVLA).
4. **RL fine-tuning** (frontier): improving beyond the demonstrator with online interaction or preference signals — the robotics analogue of [[RLHF]] post-training; still much less mature than the BC stage.

## Evaluation
---
- **Real-robot evals** are the ground truth but expensive, slow, and high-variance (tens of trials per task)
- **LIBERO** ([Liu et al., 2023](https://arxiv.org/abs/2306.03310)): simulated lifelong-learning manipulation suite; now near-saturated (97%+) for top recipes
- **SIMPLER** ([Li et al., 2024](https://arxiv.org/abs/2405.05941)): real-to-sim evaluation environments designed to correlate with real-robot rankings
- The actual test is **open-world generalization** — π0.5's unseen-home cleaning evals are the current high-water mark

## Open Problems
---
- **Data**: there is no web of robot actions; scaling depends on the data pyramid — human video, simulation, synthetic generation, cross-embodiment transfer
- **Latency**: autoregressive decoding conflicts with real-time control → parallel decoding, action experts, asynchronous execution of chunks while the next is computed
- **Long horizon**: single-chunk policies don't plan; hierarchy (π0.5's language subtasks, embodied-reasoning models) reintroduces it
- **Cross-embodiment**: one set of weights across grippers, arms, humanoids — positive transfer is demonstrated (RT-X) but morphology-general control is unsolved
- **Relation to [[World Models]]**: VLAs are model-free imitation; they don't predict consequences. Combining VLA policies with learned world models (planning, imagined rollouts, verification) is a natural convergence point

## References
---
- [Brohan et al., 2022 — RT-1: Robotics Transformer](https://arxiv.org/abs/2212.06817)
- [Brohan et al., 2023 — RT-2: Vision-Language-Action Models](https://arxiv.org/abs/2307.15818)
- [Open X-Embodiment Collaboration, 2023](https://arxiv.org/abs/2310.08864)
- [Kim et al., 2024 — OpenVLA](https://arxiv.org/abs/2406.09246)
- [Black et al., 2024 — π0: A Vision-Language-Action Flow Model](https://arxiv.org/abs/2410.24164)
- [Pertsch et al., 2025 — FAST: Efficient Action Tokenization](https://arxiv.org/abs/2501.09747)
- [Kim et al., 2025 — OpenVLA-OFT: Optimized Fine-Tuning](https://arxiv.org/abs/2502.19645)
- [NVIDIA, 2025 — GR00T N1](https://arxiv.org/abs/2503.14734)
- [Intelligence et al., 2025 — π0.5: Open-World Generalization](https://arxiv.org/abs/2504.16054)
- [Driess et al., 2025 — Knowledge Insulating VLAs](https://arxiv.org/abs/2505.23705)
- [Zhao et al., 2023 — ACT: Action Chunking with Transformers](https://arxiv.org/abs/2304.13705)
- [Chi et al., 2023 — Diffusion Policy](https://arxiv.org/abs/2303.04137)
