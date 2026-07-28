Thank you for this careful and generous review. We are grateful that you find the application relevant,
the framing compelling in separating clinical generation from communication behavior, the recipe simple
and scalable, the paper clearly organized, and the limitations candidly stated. You also suspected we
were already pursuing the obvious extensions across more models and benchmarks. That was correct, and
several completed during the discussion period; they are reported below, together with the parts of your
critique they do not answer.

**W1 and Q1. Why the evaluation protocol was limited**

The honest answer is that we traded breadth for depth, and for a paper claiming generality that was the
wrong trade. Training the 27B adapters on TPU v6e-8 was the dominant cost, so we spent the budget on
depth in one setting, 5 seeds with bootstrap intervals, an integrity-checked holdout and a contamination
probe, rather than on coverage across settings. The runs below are the correction.

**Q2. Results on additional base models and evaluation sets**

You asked for at least one or two additional base models, or one additional clinical evaluation set. We
re-ran the pipeline unchanged, with every hyperparameter fixed (LoRA $r=16$, $\alpha=32$, effective batch
16, 3 epochs, $\tau=0.4$, Qwen-14B filter, Llama-3.1-8B evaluator), on two further model families and
two further benchmarks, 5 seeds per cell and ~200 prompts per seed per condition.

**Table R1.** Transfer across model families and benchmarks. Every cell reads Base → LoRA.
Mistral-Small-24B is general-purpose, Med42-8B is clinically tuned (Llama-3 family), BioMistral-7B is a
scope condition. Row 1 repeats the submitted result for comparison.

| Benchmark | Base model | Active inquiry | Context-seek |
|---|---|---|---|
| HealthBench | **MedGemma-27B (submitted)** | **17.5 → 45.6%** | **1.40 → 1.75** |
| HealthBench | Mistral-Small-24B | 25.6 → 56.6% | 1.05 → 1.76 |
| HealthBench | Med42-8B | 11.8 → 52.5% | 0.52 → 1.40 |
| ChatDoctor | Mistral-Small-24B | 7.8 → 58.6% | 0.83 → 1.49 |
| ChatDoctor | Med42-8B | 2.9 → 86.4% | 0.71 → 1.81 |
| MedQuAD | Mistral-Small-24B | 6.5 → 26.5% | 0.32 → 1.23 |
| HealthBench | BioMistral-7B | 11.7 → 10.1% (null) | 0.27 → 0.28 |

In four of the five working cells the adapter matches or exceeds its own teacher while requiring no CoT
at inference. Base rates vary by more than a factor of two across models (11.8%, 17.5%, 25.6%) and
converge into a narrow band after adaptation (45.6%, 52.5%, 56.6%), which is what one would expect if
the endpoint is a property of the training signal rather than of the starting model.

Two findings are not merely confirmatory. First, the student does not copy the teacher indiscriminately:
on MedQuAD the CoT wrapper *degrades* scope bounding (1.85 → 1.61) and hedging quality (1.45 → 1.40)
while the distilled adapter *improves* both (1.89 and 1.78), because the quality filter removes the
teacher's failures before they reach the weights. Second, there is a precondition we can now state rather
than leave a reader to discover: on BioMistral-7B the recipe produces no effect, and the teacher fails
there too, reaching only 14.5% active inquiry, with 20% of its traces clearing the filter against 62%
for Mistral-Small-24B. Teacher incapacity and the smaller surviving training set are not fully separable
there, though both follow from the same cause.

**One part of W1 we have not addressed.** Your weakness named three things: a single model, a single
benchmark, and a single CoT protocol. We have varied the first two. We have not varied the protocol, so
the claim that the recipe is protocol-agnostic remains untested, and we do not assert it.

**What the behavior gains cost in aggregate quality**

Since this bears on whether the trade is worth making, we report it including where it is unfavourable.
On HealthBench the adapter's aggregate rubric score is essentially unchanged (Mistral-Small-24B
0.425 → 0.411; Med42-8B 0.405 → 0.392), consistent with the submitted non-inferiority result. On the new
benchmarks there are modest decreases, largest on MedQuAD (0.672 → 0.586).

We read the MedQuAD case partly as a property of the rubric rather than of the model: it rewards
agreement with a reference answer, so a response that asks a clarifying question instead of answering
scores lower by construction. Consistent with that reading, the inference-time wrapper drops further
than the adapter does (0.542 against 0.586) despite changing no weights. This is the measurement problem
the paper is about, and why we report the behavioral decomposition alongside the aggregate rather than in
place of it.

**W3 and Q4. Automatic grading, and why the graders are not clinically fine-tuned**

We share this concern and have not resolved it.

On the choice of graders, the reason is structural rather than incidental. The evaluator must sit in a
different model family from the Qwen-14B filter, because that separation is what prevents filter-grader
circularity in a self-distillation pipeline: if one family both selects training traces and scores the
result, the reported metric partly measures agreement with the selector. Meditron-3, Med42-v2 and the
Aloe family are all Llama-derived, so promoting one of them to evaluator would place filter and grader
in adjacent families and weaken exactly that property. If accepted we will state this rationale
explicitly and add a clinical judge as an additional robustness panel reported alongside the primary
grader rather than replacing it, so both the cross-family guarantee and the clinical-specificity check
are available to the reader.

On the strength of the evidence, the physician validation remains partial: one of three raters has
returned grades, giving $\kappa = 0.35$ against the Llama-3.1-8B grader, below our pre-registered target
of 0.6. We report that as a limitation on the aggregate-quality claims and do not present LLM grading as
settled clinical validation.

One piece of evidence does not depend on the grader's own scores. Using labels it never sees,
HealthBench's `context_seeking` theme tag and whether a prompt's rubric rewards asking, the adapter
distinguishes information-withholding prompts from self-contained ones (+10.5pp, bootstrap CI excluding
zero) while the base model does not (+2.1pp, CI including zero). That speaks to whether the behavior is
targeted, which is the part of the claim most exposed to grader bias.

**W4 and Q3. Comparison with prior work**

This is a fair criticism, it is the one our additional runs do not address, and we will not pretend
otherwise. If accepted we will add a Discussion subsection, "Comparison to Prior Calibration
Approaches," positioning the method against uncertainty prompting (Lin et al., 2022; Tian et al., 2023),
which calibrates at inference time whereas we internalize the behavior into weights; STaR-style
self-improvement (Zelikman et al., 2022), whose filter-then-finetune logic we apply to behavioral rather
than factual demonstrations; and Constitutional AI (Bai et al., 2022), which uses self-critique where we
use a structured CoT protocol as a behavioral teacher. It will include a summary table comparing
inference-time cost, need for a teacher model, behavioral versus factual focus, and demonstrated
generality.

We should be clear about what that does and does not give you. It is a positioning argument, not a
measurement. A direct empirical head-to-head against an inference-time calibration baseline is the single
most valuable experiment we have not run, and we name it as the immediate next step rather than folding
it into a claim.

**On workshop versus conference scope**

We understand the assessment and think it was a fair reading of the submitted version. The gap you
identified has narrowed for two of its three parts: there is now 5-seed evidence across three model
families and three benchmarks, with a scope condition showing where the recipe fails and why. Two
mechanistic analyses requested by another reviewer were also completed, distinguishing calibration from
surface mimicry and identifying output-format leakage rather than capacity exhaustion as the cause of the
CoT/LoRA interference we reported.

Three things remain outstanding, and we would rather name them than have them found: the empirical
prior-work comparison, the completed three-physician validation, and a second CoT protocol.

We would be grateful if you would reconsider the rating in light of the added evidence, with those gaps
visible. Thank you again for a review specific enough to act on. Please do let us know if questions
remain; we would be glad to run further analyses while the discussion period is open.
