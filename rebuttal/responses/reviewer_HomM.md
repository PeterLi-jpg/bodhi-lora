# Response to Reviewer HomM

We thank the reviewer for an assessment that was both generous about the framing and unambiguous
about the problem: the evaluation was too narrow to support the claims. The reviewer also suspected
we were already pursuing the obvious extensions across more models, benchmarks and human evaluation.
We were, and we used the discussion period to finish them rather than to promise them. Below we
report what the new runs show, and we are equally explicit about the two requests we have not met.

## W1, Q1, Q2: Single model, single benchmark, single CoT protocol

We have applied the identical pipeline — unchanged in its four steps of generate, grade, filter,
distill, with the paper's hyperparameters held fixed (LoRA r=16, effective batch 16, three epochs,
Qwen-14B filter, Llama-3.1-8B evaluator) — to three model families and three benchmarks, with five
independent seeds per cell and approximately 200 prompts per seed per condition.

The added base models are Mistral-Small-24B-Instruct (general-purpose, Mistral family) and Med42-8B
(clinically tuned, Llama-3 family), joining the original MedGemma-27B (Gemma family). The added
benchmarks are ChatDoctor/HealthCareMagic (unedited questions real patients asked physicians online)
and MedQuAD (NIH consumer-health question answering), joining HealthBench. A fourth benchmark,
MedQA-USMLE reframed as open-ended clinician-facing questions, is still running and will appear in
the camera-ready.

| Cell | Model family | Active inquiry (Base → LoRA) | Context-seeking |
|---|---|---|---|
| HealthBench × Mistral-Small-24B | Mistral, general | 25.6% → 56.6% | 1.05 → 1.76 |
| HealthBench × Med42-8B | Llama-3, clinical | 11.8% → 52.5% | 0.52 → 1.40 |
| ChatDoctor × Mistral-Small-24B | Mistral, general | 7.8% → 58.6% | 0.83 → 1.49 |
| ChatDoctor × Med42-8B | Llama-3, clinical | 2.9% → 86.4% | 0.71 → 1.81 |
| MedQuAD × Mistral-Small-24B | Mistral, general | 6.5% → 26.5% | 0.32 → 1.23 |
| HealthBench × BioMistral-7B | scope condition | 11.7% → 10.1% (null) | 0.27 → 0.28 |

This answers Q2 directly — the reviewer asked for at least one or two additional base models, or one
additional clinical evaluation set, and we have both, across families rather than within one. In
four of the five working cells the adapter matches or exceeds its own teacher, the inference-time
CoT protocol, while requiring no chain-of-thought at deployment. That is the internalization claim,
now demonstrated beyond the original model and benchmark.

On Q1, the reason for the original narrow protocol was straightforward: the 27B TPU training runs
were expensive and we prioritized depth (five seeds, bootstrap intervals, an integrity-checked
holdout) over breadth. We think that was the wrong trade for a paper claiming generality, and the
new runs correct it.

We would highlight one unanticipated result, because it speaks to whether the recipe merely copies a
teacher. On MedQuAD the CoT wrapper *degrades* two dimensions relative to the base model — scope
bounding falls from 1.85 to 1.61 and hedging quality from 1.45 to 1.40 — while the distilled adapter
*improves* both (1.89 and 1.78). The student is not reproducing the teacher indiscriminately; the
quality-filtering step removes the teacher's failures before they reach the weights.

We also report a boundary condition rather than only successes. On BioMistral-7B the recipe produces
nothing (11.7% to 10.1%), and the diagnostic is that the teacher fails there too: the CoT wrapper
reaches only 14.5% on that base. Where the protocol cannot elicit the behavior, there is nothing to
distill. We now state this as a precondition of the recipe — it requires a base model capable of
following the structured protocol — rather than leaving a reader to discover it.

## Aggregate quality: what the behavior gains cost

Because the reviewer will reasonably ask whether these gains come at the expense of clinical
quality, we report the aggregate rubric scores plainly, including where they are unfavourable. On
HealthBench the adapter's aggregate score is essentially unchanged (Mistral-Small-24B 0.425 to
0.411; Med42-8B 0.405 to 0.392), consistent with the paper's original non-inferiority result. On the
new benchmarks there are modest decreases, the largest on MedQuAD (0.672 to 0.586).

We think the MedQuAD case is informative rather than merely negative. Its rubric rewards agreement
with a reference answer, so a response that asks a clarifying question instead of answering
immediately scores lower by construction. Consistent with that reading, the CoT wrapper — which asks
most — drops furthest of all (0.672 to 0.542), further than the adapter does. This is precisely the
measurement problem the paper is about: aggregate rubric scores can penalize the epistemic behavior
that makes a response safer, which is why we argue for reporting the decomposition alongside the
aggregate rather than instead of it. We state the decreases in the camera-ready rather than
reporting only the dimensions that improved.

## W3, Q4: Dependence on automated grading, and why the graders are not clinical models

We share this concern and we have not resolved it. The physician validation described in Appendix B
is still partial: one of three physicians has returned grades, giving κ = 0.35 against the
Llama-3.1-8B grader, below our pre-registered target of 0.6. We report this as a limitation on the
aggregate-quality claims and we do not present LLM grading as settled clinical validation.

On why the graders are general-purpose rather than clinically tuned: the reason is structural rather
than incidental. The evaluator must sit in a different model family from the Qwen-14B quality
filter, because that separation is what prevents filter-grader circularity in a self-distillation
pipeline. The clinically tuned open-weight models most often suggested — Meditron-3, Med42-v2, the
Aloe family — are Llama-derived, so promoting one to evaluator would place filter and grader in
adjacent families and weaken the property the asymmetric design exists to guarantee. Our plan is to
add a clinical judge as an additional robustness panel reported alongside the primary grader rather
than replacing it, so that both the cross-family guarantee and the clinical-specificity check are
available. We list this as outstanding.

## W4, Q3: No comparison to prior work

This is a fair criticism that the new runs do not address, and we will not pretend otherwise. The
camera-ready adds a Discussion subsection positioning the method against uncertainty prompting
(Lin et al., Tian et al.), which achieves calibration at inference time whereas we internalize it
into weights; STaR-style self-improvement (Zelikman et al.), whose filter-then-finetune logic we
apply to behavioral rather than factual demonstrations; and Constitutional AI (Bai et al.), which
uses self-critique where we use a structured CoT protocol as a behavioral teacher. We will include a
summary table comparing these along inference-time cost, need for a teacher model, behavioral versus
factual focus, and demonstrated generality.

A direct empirical head-to-head against an inference-time calibration baseline is, in our view, the
single most valuable experiment we have not run. We name it as the next step rather than folding it
into a claim.

## On workshop versus conference scope

We understand the assessment as originally given, and we think it was correct about the submitted
version. Our position now is that the specific gap identified — one model, one benchmark, one CoT
protocol — has been closed for the first two: five-seed evidence across three model families and
three benchmark families, plus a boundary condition showing where the recipe fails and why. In
addition, two mechanistic analyses requested by Reviewer LxMF have been completed: a test
distinguishing genuine calibration from surface mimicry, and a test showing that the CoT/LoRA
interference is caused by output-format leakage rather than attention dilution.

We would ask the reviewer to reconsider in light of the added evidence, while acknowledging plainly
that the prior-work comparison and the completed three-physician validation remain outstanding, and
that a third CoT protocol has not been tested.
