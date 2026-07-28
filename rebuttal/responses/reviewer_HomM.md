Thank you for this careful and generous review, and for setting out what does work: the separation of
clinical generation from communication behavior, and the simplicity of the recipe. On W2, you suspected we were already pursuing the obvious extensions across more models,
benchmarks and human evaluation. That was correct on all three counts: the first two completed during
the discussion period, and the human evaluation is partial, so we report its current state rather than
waiting. Below are the results, with the parts of your critique they do not answer.

**W1, Q1 and Q2. The evaluation is too narrow**

The constraint was compute access rather than a judgment about what mattered. Training one 27B adapter on
TPU v6e-8 consumed most of what we had, so we spent it on depth in one setting: 5 seeds with bootstrap
intervals, an integrity-checked holdout and a contamination probe. The paper says as much, listing
component ablations as requiring compute beyond the budget. Further compute became available only after
the paper was finished, which is why the runs below exist and why the remaining items are commitments we
can keep rather than aspirations.

What was ours to fix is the mismatch between that evidence and the language around it. The contribution
will be described as a recipe plus an evaluation decomposition for open-ended, information-seeking
clinical Q&A, not as a validated general framework, and we will state what it does not cover.

*Results on additional base models and evaluation sets.* You asked for one or two additional base
models, or one additional clinical evaluation set. We re-ran the
pipeline unchanged, every hyperparameter fixed (LoRA $r=16$, $\alpha=32$, effective batch 16, 3 epochs,
$\tau=0.4$, Qwen-14B filter, Llama-3.1-8B evaluator), on two further model families and two further
benchmarks, 5 seeds per cell and ~200 prompts per seed per condition.

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
at inference; the exception is Med42-8B on HealthBench, where the adapter reaches 52.5% against the
wrapper's 57.1%. Base rates vary by more than a factor of two across models (11.8%, 17.5%, 25.6%) and
converge into a narrow band after adaptation (45.6%, 52.5%, 56.6%), which is what one would expect if
the endpoint is a property of the training signal rather than of the starting model.

Two findings are not merely confirmatory. The student does not copy the teacher indiscriminately: on
MedQuAD the wrapper *degrades* scope bounding (1.85 → 1.61) and hedging quality (1.45 → 1.40) while the
adapter *improves* both (1.89 and 1.78), because the filter removes the teacher's failures before they
reach the weights. And there is a precondition: on BioMistral-7B the recipe produces no effect and the
teacher fails there too (14.5% active inquiry, 20% of traces clearing the filter against 62% for
Mistral-24B).

*One part of W1 we have not addressed.* Your weakness named three things: a single model, a single
benchmark, and a single CoT protocol. We have varied the first two, not the protocol, so the claim that
the recipe is protocol-agnostic remains untested and we do not assert it.

*What the behavior gains cost in aggregate quality.* We report this including where it is unfavourable.
On HealthBench the aggregate rubric score is
essentially unchanged (Mistral-Small-24B 0.425 → 0.411; Med42-8B 0.405 → 0.392), consistent with the
submitted non-inferiority result. On the new benchmarks there are modest decreases, largest on MedQuAD
(0.672 → 0.586). We read that partly as a rubric property: it rewards agreement with a reference answer,
so asking instead of answering scores lower by construction, and consistent with this the wrapper drops
further than the adapter (0.542 against 0.586) despite changing no weights.

**W3 and Q4. Automatic grading, and why the graders are not clinically fine-tuned**

We share this concern and have not resolved it.

On the choice of graders the reason is structural. The evaluator must sit in a different family from the
Qwen-14B filter, because that separation is what prevents filter-grader circularity: if one family both
selects training traces and scores the result, the metric partly measures agreement with the selector.
Meditron-3, Med42-v2 and the Aloe family are all Llama-derived, so promoting one to evaluator would put
filter and grader in adjacent families and weaken that property. If accepted we will state this and add
a clinical judge as an additional robustness panel alongside the primary grader rather than replacing
it.

On the strength of the evidence, the physician validation remains partial: one of three raters has
returned grades, giving $\kappa = 0.35$ against the Llama-3.1-8B grader, below our pre-registered target
of 0.6. We report that as a limitation on the aggregate-quality claims and do not present LLM grading as
settled clinical validation.

One piece of evidence does not depend on the grader's own scores, and it covers one of the three
dimensions you singled out. Using labels the grader never sees, HealthBench's `context_seeking` theme
tag and whether a prompt's rubric rewards asking, the adapter distinguishes information-withholding
prompts from self-contained ones (+10.5pp, bootstrap CI excluding zero) while the base model does not
(+2.1pp, CI including zero). Context-seeking is therefore supported by a signal external to the grader.
Hedging quality and red-flag identification are not, and we do not claim otherwise; those two rest on
the automatic grader until the physician panel is complete.

It is worth separating which claims this touches. Trace filtering and the aggregate-quality comparison
use HealthBench's expert-authored rubrics, so the quality-preservation result rests on clinician-written
criteria applied automatically. It is the seven epistemic dimensions that use our own anchors, so the
concern applies to the behavioral claims rather than to the non-inferiority one.

**W4 and Q3. Comparison with prior work**

This is a fair criticism and the one our additional runs do not address. We would gently push back on
one word, though. Section 2 does position the method against STaR, Constitutional AI and
chain-of-thought prompting, and states how our filter-then-finetune step differs from each, so the
comparison is not wholly absent. What is absent is an *empirical* one, which is a narrower gap than the
sentence implies, and a real one we do not dispute.

If accepted we will expand that into a Discussion subsection, "Comparison to Prior Calibration
Approaches," covering uncertainty prompting (Lin et al., 2022; Tian et al., 2023), which calibrates at
inference time whereas we internalize into weights; STaR (Zelikman et al., 2022), whose
filter-then-finetune logic we apply to behavioral rather than factual demonstrations; and Constitutional
AI (Bai et al., 2022), which uses self-critique where we use a CoT protocol as a behavioral teacher, with
a table comparing inference cost, teacher requirement, behavioral versus factual focus, and generality.

That is a positioning argument, not a measurement. A direct head-to-head against an inference-time
calibration baseline is the single most valuable experiment we have not run, and we name it as the
immediate next step rather than folding it into a claim.

**On workshop versus conference scope**

We understand the assessment and think it was a fair reading of the submitted version. The gap has
narrowed for two of its three parts: 5-seed evidence across three model families and three benchmarks,
with a scope condition showing where the recipe fails and why. Two mechanistic analyses requested by
another reviewer were also completed, distinguishing calibration from surface mimicry and identifying
output-format leakage rather than capacity exhaustion as the cause of the interference we reported.

We would also argue the evidence changed in kind, not only in quantity: a single positive instantiation
cannot separate "the recipe works" from "this pairing happens to work," whereas several cells, a
documented null with a diagnosis, and a mechanistic account of the interference effect can.

On the rating, which we raise only because the category is broad. Reject covers technical flaws, weak
evaluation, inadequate reproducibility and incompletely addressed ethics. Your review identifies the
second of these; on the others it records clarity 4, no ethical concerns, and no technical flaw, and the
submitted version already carries 5 independent seeds, bootstrap intervals, a contamination probe,
specified hardware and released code and adapters. If evaluation breadth is the operative concern, that
is the ground we have been able to move.

Three things remain outstanding, and if accepted we commit to all three: the empirical prior-work
comparison, completion of the physician validation, and a second CoT protocol.

We would be grateful if you would reconsider the rating in light of the added evidence, with those gaps
visible. Thank you again for a review specific enough to act on. Please let us know if questions remain;
we would gladly run further analyses while the discussion period is open.
