We thank the reviewer for detailed comments and suggestions. We are grateful the reviewer finds the
application relevant, the framing compelling in separating clinical generation from communication
behavior, the recipe simple and practical, the paper clearly organized and conceptually clean, and
the limitations candidly stated. The reviewer suspected we were already pursuing the obvious
extensions across more models and benchmarks. That was correct, and several completed during the
discussion period.

**Why the evaluation protocol was limited**

The 27B TPU runs were expensive and we prioritized depth (5 seeds, bootstrap intervals, an
integrity-checked holdout) over breadth. For a paper claiming generality that was the wrong trade,
and we have begun correcting it.

**Results on additional base models and evaluation sets**

As the reviewer requested at least one or two additional base models or one additional clinical
evaluation set, we re-ran the pipeline unchanged, with all hyperparameters fixed (LoRA r=16, effective
batch 16, 3 epochs, τ=0.4, Qwen-14B filter, Llama-3.1-8B evaluator), on two further model families
and two further benchmarks, 5 seeds per cell:

| Benchmark | Base model (family) | Active inquiry | Context-seek |
|---|---|---|---|
| HealthBench | Mistral-Small-24B (Mistral) | 25.6% → 56.6% | 1.05 → 1.76 |
| HealthBench | Med42-8B (Llama-3) | 11.8% → 52.5% | 0.52 → 1.40 |
| ChatDoctor | Mistral-Small-24B (Mistral) | 7.8% → 58.6% | 0.83 → 1.49 |
| ChatDoctor | Med42-8B (Llama-3) | 2.9% → 86.4% | 0.71 → 1.81 |
| MedQuAD | Mistral-Small-24B (Mistral) | 6.5% → 26.5% | 0.32 → 1.23 |
| HealthBench | BioMistral-7B | 11.7% → 10.1% (null) | 0.27 → 0.28 |

In four of the five working cells the adapter matches or exceeds its own teacher while requiring no
CoT at inference. A clinician-facing benchmark (MedQA-USMLE, open-ended) is in progress.

Two results are not merely confirmatory. First, on MedQuAD the CoT wrapper *degrades* scope bounding
(1.85 to 1.61) and hedging (1.45 to 1.40) while the adapter *improves* both (1.89 and 1.78),
suggesting the filtering step removes the teacher's failures rather than copying them. Second,
BioMistral-7B shows no effect, and the diagnostic is that the teacher fails there too (wrapper
reaches only 14.5%). The recipe requires a base capable of following the protocol, and we now state
that as a precondition. We note the third element of this weakness, a single CoT protocol, remains
unaddressed.

**What the behavior gains cost in aggregate quality**

Reported plainly, including where unfavourable. On HealthBench the adapter's aggregate score is
essentially unchanged (Mistral-Small-24B 0.425 to 0.411; Med42-8B 0.405 to 0.392), consistent with
the submitted non-inferiority result. On the new benchmarks there are modest decreases, largest on
MedQuAD (0.672 to 0.586). We read this partly as a property of the rubric: MedQuAD's synthesized
rubric rewards agreement with a reference answer, so a response that asks a clarifying question
instead of answering scores lower by construction. Consistent with that, the wrapper, which asks
most, drops furthest (0.672 to 0.542). This is the measurement problem the paper is about, and it is
why we report the decomposition alongside the aggregate rather than instead of it.

**Claims depend heavily on automatic grading**

We share this concern and have not resolved it. The physician validation remains partial: one of
three raters has returned grades, giving κ = 0.35 against the Llama-3.1-8B grader, below our
pre-registered target of 0.6. We report this as a limitation on the aggregate-quality claims and do
not present LLM grading as settled clinical validation.

**Why the evaluation models are not clinically fine-tuned**

The reason is structural rather than incidental. The evaluator must sit in a different model family
from the Qwen-14B filter, because that separation is what prevents filter-grader circularity in a
self-distillation pipeline. Meditron-3, Med42-v2 and Aloe are Llama-derived, so promoting one to
evaluator would place filter and grader in adjacent families and weaken that property. We plan to add
a clinical judge as an additional robustness panel alongside the primary grader rather than replacing
it, and will state this rationale explicitly in the revision.

**Comparison to prior work is absent**

This is a fair criticism that our additional runs do not address, and we will not pretend otherwise.
The revision adds a Discussion subsection positioning the method against uncertainty prompting (Lin
et al., Tian et al.), which calibrates at inference time whereas we internalize into weights;
STaR-style self-improvement (Zelikman et al.), whose filter-then-finetune logic we apply to
behavioral rather than factual demonstrations; and Constitutional AI (Bai et al.), which uses
self-critique where we use a structured CoT protocol as a behavioral teacher. We include a summary
table comparing inference-time cost, need for a teacher model, behavioral versus factual focus, and
demonstrated generality. A direct empirical head-to-head against an inference-time calibration
baseline is the single most valuable experiment we have not run, and we name it as the immediate next
step.

**On workshop versus conference scope**

We understand the assessment and think it was fair to the submitted version. The specific gap
identified has been narrowed for two of its three parts: 5-seed evidence across three model families
and three benchmarks, plus a scope condition showing where the recipe fails and why. The two
mechanistic analyses requested by Reviewer LxMF were also completed, distinguishing calibration from
surface mimicry and identifying output-format leakage as the cause of the CoT/LoRA interference.
Still outstanding: the prior-work comparison, the completed physician validation, and a second CoT
protocol. We would ask the reviewer to reconsider in that light, and we will incorporate all feedback
into the paper.
