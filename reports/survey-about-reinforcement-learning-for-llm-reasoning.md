# Reinforcement learning for LLM reasoning: methods, rewards, and evidence

## TL;DR

- Recent methods use verifiable outcomes and group-relative updates; DAPO proposes additional changes to sampling and training stability. The evidence reviewed here is concentrated in math and code tasks. [1][2]
- GRPO removes the separate learned critic used in PPO-style training and estimates relative advantages from multiple answers to one prompt. This can reduce training components, though results reported for math tasks do not establish broad reasoning gains by themselves. [2]
- Process reward models can give feedback at intermediate steps, but their measured advantage over outcome rewards depends on label quality and evaluation design. Newer work tries to generate denser credit from outcomes rather than requiring human labels for every step. [3][4][5][6]
- RL gains are not uniform across tasks or metrics. Studies report stronger pass@1 or math scores in some setups, alongside weak transfer, response-length issues, and sensitivity to experimental choices. [7][8][9][10]
- I read the most important open issue as measurement: pass@1, pass@k, multi-seed replication, matched compute and tests of compositional reasoning answer different questions and should be reported together. [9][10][11]

## Background

Reinforcement learning (RL) trains a model by sampling outputs, scoring them with a reward signal, and adjusting the policy so that higher-scoring outputs become more likely. For reasoning, a key design choice is whether the reward checks only the final answer or gives feedback on intermediate steps. Outcome reward models (ORMs) score a completed response, while process reward models (PRMs) assign scores or labels to steps. These signals affect both what models learn and how results should be evaluated. [3]

A foundational policy method is proximal policy optimization (PPO), which typically uses a learned value model, or critic, to estimate the expected return. DeepSeekMath introduced group relative policy optimization (GRPO), a PPO variant that samples several responses for each prompt and uses their relative rewards as a baseline, rather than training a separate critic. The paper reported improvements on mathematical benchmarks in its own setup. This is useful context for later reasoning systems, though those results are not a direct comparison of all current RL pipelines. [2]

OpenAI's 2024 o1 launch connected its reported performance to compute at both training and inference. The announcement reported improved AIME performance with more training RL and more time spent generating a response, including higher results when sampling and selecting multiple answers. It also noted that established math benchmarks were becoming less discriminative among strong models. As a company announcement, this is evidence of the evaluation approach and company-reported results, not an independent controlled comparison. [12]

The research reviewed here mainly concerns mathematical and code reasoning, where final answers or program behavior can sometimes be checked automatically. Most papers and benchmarks are not evidence about every kind of reasoning, such as open-ended planning or explanation quality. I therefore treat reported scores as results for their stated models, training recipe and evaluation, rather than as a general measure of reasoning ability.

## Policy optimization and training pipelines

R1-Zero illustrates a direct RL proposal. DeepSeek's paper abstract says pure RL can incentivize reasoning without human-labeled reasoning trajectories. I treat this as the authors' stated approach, not evidence that every observed benchmark gain came from RL alone. [13]

DAPO modifies the sampling and optimization details around GRPO. Its authors propose asymmetric clipping to support exploration, dynamic sampling that drops prompt groups with uniformly correct or incorrect answers, token-level policy loss, and reward shaping for overly long responses. They report 50 points on AIME 2024 with Qwen2.5-32B, compared with a cited 47-point result for DeepSeek-R1-Zero-Qwen-32B using half as many training steps. This is an author-reported comparison across their stated setup, not independent confirmation or a clean estimate of each component's effect. [1]

These methods differ in how they handle prompt sampling, exploration, response length and long-run stability. DAPO makes sampling and rollout changes explicit, while prolonged-RL work studies stability interventions. For practitioners, the papers suggest that the optimizer alone is not the full recipe. Reward reliability, sample diversity and stability during long training also need attention. The available comparisons do not establish a universal best method. [1][14]

A particularly striking data-efficiency result comes from a study that reports training Qwen2.5-Math-1.5B with one example. The authors report MATH500 accuracy increasing from 36.0% to 73.6%, while noting that format correction accounts for part of the gain. After adjusting for that effect, they report an 8.6 percentage-point improvement on MATH500. Their results also cover other math benchmarks and configurations, but they do not show that one example is enough for coding or general reasoning tasks. [7]

Training for longer introduces stability concerns. A prolonged-RL study reports gains in math, coding and logic tasks using a small model, and identifies KL regularization, clipping settings and periodic reference-policy resets as important parts of its recipe. The reported improvements are specific to the study's baselines and training setup; the available abstract does not clarify every benchmark aggregation detail. I would read this as evidence that long-run stability is a concrete engineering problem, not as proof that the named interventions transfer unchanged to other models. [14]

## Verifiable rewards and what they teach

In RL with verifiable rewards, correctness feedback scores the outcome, but a correct final answer does not certify that every intermediate step was sound. Process-reward research motivates measuring intermediate steps as a separate target. DeepSeek's R1-Zero abstract presents pure RL as a way to incentivize reasoning without human-labeled reasoning trajectories. [3][7][13]

Results from verifiable-reward RL (RLVR) suggest both promise and qualification. One-example RLVR reports gains on math benchmarks, but explicitly separates format correction from improvements it attributes to reasoning. Another study of R1-like training finds that pretraining characteristics influence RL performance and that standard GRPO can favor unnecessarily long incorrect responses. Its authors report that Dr. GRPO improves token efficiency while maintaining reasoning performance in their experiments. Together, these findings caution against attributing all observed behavior to RL or assuming that more generated tokens indicate better reasoning. [7][8]

Coding can also be evaluated with executable tests, though performance depends on the task and evaluation set. A prolonged-RL study reports coding improvements in its particular setup. That result does not establish whether a model generalizes to code requirements that are absent from the tests. [14]

## Outcome and process rewards

Outcome-based feedback is relatively simple when the answer can be checked, but it gives sparse credit: an entire sequence may receive one score. Process reward models attempt to localize errors and reward useful intermediate steps. In a foundational comparison on mathematical reasoning, a PRM achieved 78.2% on a representative MATH subset in best-of-1860 selection, compared with 72.4% for an ORM and 69.6% for majority voting. The paper also reports that active learning improved process-supervision data efficiency. These results support process feedback in that math search setting, but the many-candidate selection protocol and the cost of human step labels constrain the conclusion. [3]

The later lessons from PRM development make the comparison less straightforward. That study reports that automatically synthesized Monte Carlo process labels often lagged human or LLM-judge annotations, and warns that best-of-N answer selection alone can hide weak step verification. It recommends combining response-level scores with direct measures of step-level error detection. Thus, process labels add useful detail only if they actually identify valid steps. A PRM trained on noisy labels can inherit outcome-level shortcuts while appearing to provide finer feedback. [4]

Recent work explores ways to improve credit assignment without labeling each step by hand. PRIME updates an implicit process reward from policy rollouts and outcome labels, with the aim of deriving denser reward information online. Its authors report gains over an SFT starting point across reasoning benchmarks, but the abstract-level evidence available here is not a common head-to-head trial against every ORM or PRM. [5] CAPO asks a general-purpose language model to generate step-wise critiques and uses them to redistribute token-level credit alongside outcome feedback. The method is promising as a hybrid design, but depends on the critique model's judgment, and the retrieved evidence does not independently verify critique quality. [6]

Formal verification offers a different way to create step labels. The FoVer paper's HF-search abstract describes using formal verification tools to annotate step-level errors for PRM training, and reports cross-task gains over human-annotated approaches. Because the retrieved evidence is limited to the abstract, I treat that comparison as provisional. Taken together, the work points to a tradeoff: sparse outcome rewards are easier to validate, while denser process signals may improve credit assignment but introduce their own labeling or judge errors. No common comparison in these sources establishes one reward type as best across reasoning domains. [15]

## Evaluation, generalization and efficiency

The metric changes the interpretation of RL results. Pass@1 estimates the chance that one sampled answer is correct; pass@k asks whether at least one of k samples succeeds. A NeurIPS 2025 study reports that RLVR models performed better at small k in its tests, while base models sometimes performed better at large k. The authors interpret the pattern as improved sampling efficiency without clear evidence of broader reasoning coverage in their setups. This is a contested empirical interpretation, but it shows why a higher pass@1 alone cannot demonstrate that training expands the set of solvable problems. [10]

Reproducibility is another issue. A standardized evaluation study reports sensitivity to random seeds, prompts, decoding and hardware or software settings. It also reports modest RL gains and overfitting on small contests such as AIME 2024 in its math experiments. This is a warning about evaluation practice, not a universal negative finding about RL. [9]

Aggregate accuracy can also obscure which reasoning components improved. SPARKLE evaluates plan following, knowledge integration and chains of subproblems. Its authors report improved use of supplied knowledge and stronger following of explicit plans in some settings, alongside continued weakness in compositional subproblem solving. They propose partial scaffolding for hard problems. Because the study is math-focused and uses its own task construction, it should be read as a diagnostic example rather than a comprehensive measure of reasoning. [11]

These evaluation studies suggest reporting results at matched training and inference compute, along with pass@1 and pass@k, repeated seeds, prompt and decoding details, and newer held-out tasks. Reasoning traces may help diagnose a failure, but fluent traces alone do not prove that intermediate steps caused the final answer. For applied work, it is also useful to report token use, verifier coverage and failure cases, since accuracy can improve while inference costs or response lengths grow. [8][9][10][11]

## Trends and open problems

The research is moving from basic outcome-reward training toward more explicit control of the complete learning loop: which prompts are sampled, how trajectories are selected, how rewards are normalized, and how updates remain stable. DAPO and prolonged-RL work treat sampling and long-run stability as design targets, while PRIME, CAPO and FoVer seek finer credit assignment from online outcomes, model-generated critiques or formal tools. These are complementary directions, but their reported results come from different tasks and evaluation protocols, so they are not a ranked comparison. [1][14][5][6][15]

Several issues remain open. First, the field needs more independent replications and multi-seed evaluations on larger, less familiar task sets. Second, tests should separate sampling efficiency from expanded problem coverage by including multiple k values at matched compute. Third, reward design needs better checks for verifier gaps, incorrect process labels, verbosity incentives and judge-model errors. Finally, diagnostic evaluation should test whether models can combine steps and transfer to new task structures, not only reproduce performance on familiar contest benchmarks. [4][9][10][11]

My overall reading is that RL can make a model more likely to produce a correct answer in some verifiable reasoning settings, and some papers report sizable gains. The evidence is more conditional on whether this improves general reasoning capacity: pretraining, extra sampling, response length and experimental choices can all affect the measured result. Progress depends on making those factors visible, rather than treating one benchmark score as a complete account of reasoning. [13][7][8][9][10]

## References
[1] DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arxiv. https://arxiv.org/abs/2503.14476 (2025-03-18)
[2] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. web. https://arxiv.org/abs/2402.03300 (2024-02-05)
[3] Let's Verify Step by Step. web. https://arxiv.org/abs/2305.20050 (2023-05-31)
[4] The Lessons of Developing Process Reward Models in Mathematical Reasoning. web. https://aclanthology.org/2025.findings-acl.547/ (2025-07)
[5] Process Reinforcement through Implicit Rewards. web. https://arxiv.org/abs/2502.01456 (2025-02-03)
[6] CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment. web. https://arxiv.org/abs/2508.02298 (2025-08-04)
[7] Reinforcement Learning for Reasoning in Large Language Models with One Training Example. arxiv. https://arxiv.org/abs/2504.20571 (2025-04-29)
[8] Understanding R1-Zero-Like Training: A Critical Perspective. hf-search. https://huggingface.co/papers/2503.20783 (2025-03-26)
[9] A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibility. hf-search. https://huggingface.co/papers/2504.07086 (2025-04-09)
[10] Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?. web. https://papers.nips.cc/paper_files/paper/2025/file/537d5aa768c2d534016a4d06f87bc8fb-Paper-Conference.pdf (2025)
[11] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning. web. https://proceedings.neurips.cc/paper_files/paper/2025/file/6058d0c628a03fd95dfe5c72cbdf9e64-Paper-Conference.pdf (2025)
[12] Learning to reason with LLMs. web. https://openai.com/index/learning-to-reason-with-llms/ (2024-09-12)
[13] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. web. https://arxiv.org/abs/2501.12948 (2025-01-22)
[14] Scaling Up RL: Unlocking Diverse Reasoning in LLMs via Prolonged Training. arxiv. https://arxiv.org/abs/2507.12507 (2025-07-16)
[15] Training Step-Level Reasoning Verifiers with Formal Verification Tools. hf-search. https://huggingface.co/papers/2505.15960 (2025-05-21)
