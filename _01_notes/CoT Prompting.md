### [[dl_moc]]  
# **Chain-Of-Thought Prompting**

**Tags**:: #llm
**Links**:: [[FSL]] · [[LLMs]] · [[Scaling Laws]] · [[GRPO]]


CoT Prompting = [[FSL|Few-Shot Learning]] + Reasoning

**Chain-of-Thought (CoT)** prompting elicits multi-step reasoning from language models by including example reasoning traces in the prompt. Instead of jumping from question to answer, the model produces intermediate steps — a scratchpad — before committing to the final answer.

## Standard CoT (Few-Shot)
---
[Wei et al., 2022](https://arxiv.org/abs/2201.11903) showed that providing exemplars with reasoning steps dramatically improves performance on arithmetic, commonsense, and symbolic reasoning tasks.

**Standard prompting**:
> Q: If John has 5 apples and gives 2 away, how many does he have? A: 3.

**CoT prompting**:
> Q: If John has 5 apples and gives 2 away, how many does he have?
> A: John starts with 5 apples. He gives 2 away. 5 − 2 = 3. The answer is 3.

![[cot_example.png]]

**Key finding**: CoT is an **emergent ability** — it only helps for models above ~100B parameters. Smaller models are not improved (and sometimes hurt) by CoT exemplars.

## Zero-Shot CoT
---
[Kojima et al., 2022](https://arxiv.org/abs/2205.11916) showed that simply appending **"Let's think step by step."** to the question triggers CoT reasoning without any exemplars:

> Q: Roger has 5 tennis balls. He buys 2 more cans of 3 tennis balls each. How many tennis balls does he now have?
> A: **Let's think step by step.** Roger starts with 5 balls. He buys 2 cans × 3 = 6 more balls. 5 + 6 = 11. The answer is 11.

This works because the model has seen similar reasoning patterns during pre-training. The phrase acts as a **soft prompt** that shifts the model into a reasoning mode.

![[cot_benchmark.png | 500]]

## Self-Consistency
---
[Wang et al., 2022](https://arxiv.org/abs/2203.11171) improves CoT by sampling **multiple diverse reasoning paths** and taking a **majority vote** on the final answer:

1. Sample $k$ independent CoT completions: $\{c_1, \ldots, c_k\}$
2. Extract the final answer from each: $\{a_1, \ldots, a_k\}$
3. Return $\hat{a} = \text{majority}(a_1, \ldots, a_k)$

This exploits the intuition that correct answers are robust — many valid reasoning paths lead to the same answer, while errors are more idiosyncratic. Self-consistency consistently improves over greedy CoT by 5–10 percentage points on math benchmarks.

## Tree of Thought (ToT)
---
[Yao et al., 2023](https://arxiv.org/abs/2305.10601) extends CoT from a linear chain to a **tree search** over reasoning steps:

- The model proposes multiple **thought candidates** at each step
- A **value estimator** (the model itself, prompted) scores each candidate
- **BFS or DFS** explores the tree, backtracking when paths look unpromising

This is useful for tasks requiring exploration (e.g., creative writing, multi-step planning) where greedy decoding gets stuck.

## Program of Thought (PoT)
---
Rather than natural language reasoning, the model writes **executable code** (Python) as the chain of thought. The code is then run, and the output is the answer. This offloads arithmetic and combinatorial reasoning to a reliable interpreter, dramatically improving accuracy on math and data analysis tasks.

## Why CoT Works
---
Several mechanisms have been proposed:

1. **Scratchpad hypothesis**: intermediate steps expand the model's "working memory" beyond what fits in the next-token prediction
2. **Decomposition**: complex problems are broken into sub-problems that the model has seen in training
3. **Calibration**: generating steps before the answer allows the model to revise; it's less likely to commit to a wrong answer prematurely
4. **Length bias**: longer outputs correlate with more careful reasoning (but also with overthinking in some settings)

## Connection to Reasoning Models (o1, DeepSeek-R1)
---
Modern reasoning models like OpenAI o1 and DeepSeek-R1 internalize CoT by training the model to generate long **"thinking" traces** before producing the final answer:

- **DeepSeek-R1**: uses GRPO (see [[GRPO]]) to reinforce reasoning traces that lead to correct answers, without human-labeled thought chains
- **o1**: uses RL over verifiable tasks (math, code) to learn to think before answering

This represents a shift from CoT as an inference-time trick to CoT as a **trained behavior** — the model learns *when* to think deeply and *how* to structure its reasoning.

## Limitations
---
- **Hallucinated steps**: the model may generate plausible-sounding but wrong intermediate steps
- **Verbosity without depth**: more tokens doesn't guarantee better reasoning
- **Task-specific**: CoT helps most on multi-step reasoning; it can hurt on simple factual recall (introduces noise)
- **Prompt sensitivity**: slight rewording of exemplars can significantly change performance
