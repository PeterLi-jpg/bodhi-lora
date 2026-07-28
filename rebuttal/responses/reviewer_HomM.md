Thank you for your thorough and generous review. We are grateful you find the application relevant,
the framing compelling in separating clinical generation from communication behavior, the recipe
simple and practical, the paper clearly organized and conceptually clean, and the limitations candidly
stated. You suspected we were already pursuing the obvious extensions across more models and
benchmarks; that was correct, and several completed during the discussion period.

**Why the evaluation protocol was limited**

Short answer: we traded breadth for depth, and for a paper claiming generality that was the wrong
trade.

The 27B TPU training runs were expensive, so we prioritized depth on one setting (5 seeds, bootstrap
intervals, an integrity-checked holdout with a contamination probe) over coverage across settings. We
have begun correcting it, as below.

**Results on additional base models and evaluation sets**

As you requested at least one or two additional base models or one additional clinical evaluation set,
we re-ran the pipeline unchanged, with all hyperparameters fixed (LoRA $r=16$, $\alpha=32$, effective batch 16,
3 epochs, $\tau=0.4$, Qwen-14B filter, Llama-3.1-8B evaluator), on two further model families and two
further benchmarks, 5 seeds per cell and ~200 prompts per seed per condition.

**Table R1.** Transfer across model families and benchmarks. Every cell reads Base → LoRA.
Mistral-Small-24B is general-purpose, Med42-8B is clinically tuned (Llama-3 family), BioMistral-7B is
the scope condition; the submitted result used MedGemma-27B (Gemma).

| Benchmark | Base model | Active inquiry | Context-seek |
|---|---|---|---|
| HealthBench | **MedGemma-27B (submitted)** | **17.5 → 45.6%** | **1.40 → 1.75** |
| HealthBench | Mistral-Small-24B | 25.6 → 56.6% | 1.05 → 1.76 |
| HealthBench | Med42-8B | 11.8 → 52.5% | 0.52 → 1.40 |
| ChatDoctor | Mistral-Small-24B | 7.8 → 58.6% | 0.83 → 1.49 |
| ChatDoctor | Med42-8B | 2.9 → 86.4% | 0.71 → 1.81 |
| MedQuAD | Mistral-Small-24B | 6.5 → 26.5% | 0.32 → 1.23 |
| HealthBench | BioMistral-7B | 11.7 → 10.1% (null) | 0.27 → 0.28 |

In four of the five working cells the adapter matches or exceeds its own teacher while requiring no
CoT at inference. A sixth cell, MedQA-USMLE reframed as open-ended questions, is excluded: our
reframing preamble cued the behavior being scored, driving base-model inquiry to 99%, so it measured
instruction-following rather than calibration. We will rebuild that prompt neutrally for the
camera-ready.

Two results are not merely confirmatory:

- **The student does not copy the teacher indiscriminately.** On MedQuAD the wrapper *degrades* scope
  bounding (1.85 → 1.61) and hedging quality (1.45 → 1.40) while the adapter *improves* both (1.89 and
  1.78). The quality-filtering step removes the teacher's failures before they reach the weights.
- **A scope condition.** BioMistral-7B shows no effect, and the diagnostic is that the teacher fails
  there too (wrapper reaches only 14.5%). The failure is visible upstream of training: only 20% of its
  traces cleared the quality filter, versus 62% for Mistral-Small-24B, leaving 733 training rows. We
  therefore cannot fully separate teacher incapacity from the smaller surviving training set, though
  both follow from the same cause. The recipe requires a base model capable of following the protocol,
  which we now state as a precondition rather than leaving a reader to discover it.

We note the third element of this weakness, a single CoT protocol, remains unaddressed; we vary the
model and the benchmark, not the teacher.

**What the behavior gains cost in aggregate quality**

We report this plainly, including where it is unfavourable. On HealthBench the adapter's aggregate
rubric score is essentially unchanged (Mistral-Small-24B 0.425 → 0.411; Med42-8B 0.405 → 0.392),
consistent with the submitted non-inferiority result. On the new benchmarks there are modest
decreases, largest on MedQuAD (0.672 → 0.586).

We read the MedQuAD case partly as a property of the rubric rather than of the model: it rewards
agreement with a reference answer, so a response that asks a clarifying question instead of answering
scores lower by construction. Consistent with that reading, the wrapper, which asks most, drops
furthest of all (0.672 → 0.542), further than the adapter. This is the measurement problem the paper
is about, and it is why we report the behavioral decomposition alongside the aggregate rather than in
place of it.

**Claims depend heavily on automatic grading**

We share this concern and have not resolved it. The physician validation remains partial: one of three
raters has returned grades, giving $\kappa = 0.35$ against the Llama-3.1-8B grader, below our pre-registered
target of 0.6. We report this as a limitation on the aggregate-quality claims and do not present LLM
grading as settled clinical validation.

**Why the evaluation models are not clinically fine-tuned**

The reason is structural rather than incidental. The evaluator must sit in a different model family
from the Qwen-14B filter, because that separation is what prevents filter-grader circularity in a
self-distillation pipeline. Meditron-3, Med42-v2 and Aloe are Llama-derived, so promoting one to
evaluator would place filter and grader in adjacent families and weaken that property. We plan to add
a clinical judge as an additional robustness panel alongside the primary grader rather than replacing
it, and will state this rationale explicitly in the revision.

**Comparison to prior work is absent**

This is a fair criticism that our additional runs do not address, and we will not pretend otherwise.
The revision adds a Discussion subsection, "Comparison to Prior Calibration Approaches," positioning
the method against uncertainty prompting [1,2], which calibrates at inference time whereas we
internalize the behavior into weights; STaR-style self-improvement [3], whose filter-then-finetune
logic we apply to behavioral rather than factual demonstrations; and Constitutional AI [4], which uses
self-critique where we use a structured CoT protocol as a behavioral teacher. We include a summary
table comparing inference-time cost, need for a teacher model, behavioral versus factual focus, and
demonstrated generality. A direct empirical head-to-head against an inference-time calibration
baseline is the single most valuable experiment we have not run, and we name it as the immediate next
step.

**On workshop versus conference scope**

We understand the assessment and think it was fair to the submitted version. The gap you identified
has been narrowed for two of its three parts: 5-seed evidence across three model families and three
benchmarks, plus a scope condition showing where the recipe fails and why. The two mechanistic
analyses requested by Reviewer LxMF were also completed, distinguishing calibration from surface
mimicry and identifying output-format leakage as the cause of the CoT/LoRA interference. Still
outstanding: the prior-work comparison, the completed physician validation, and a second CoT protocol.

We would ask you to reconsider in light of the added evidence, with those gaps visible, and we will
incorporate all feedback into the paper.

[1] Lin, Hilton, Evans. "Teaching Models to Express Their Uncertainty in Words." TMLR 2022.
[2] Tian et al. "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from
Language Models Fine-Tuned with Human Feedback." EMNLP 2023.
[3] Zelikman et al. "STaR: Bootstrapping Reasoning With Reasoning." NeurIPS 2022.
[4] Bai et al. "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073, 2022.
