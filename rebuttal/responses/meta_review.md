One comment per listed weakness, each self-contained for its own box. The three Strengths boxes need
no reply. Scope note: NeurIPS permits additional results that reviewers specifically requested, so
each new experiment below is attributed to the reviewer who asked for it.

---

## W1. Use of MedGemma does not align with HealthBench use cases

We agree this was a fair concern about the submitted version. As Reviewer FzpA suggested a more widely
used model would be more appropriate, we re-ran the identical pipeline with all hyperparameters fixed
on **Mistral-Small-24B-Instruct**, a general-purpose non-clinical open-weight model. The effect
reproduces on HealthBench: active inquiry 25.6% → 56.6%, context-seeking 1.05 → 1.76 (5 seeds). We
also ran **Med42-8B**, clinically tuned from a third model family (Llama-3). The submitted finding
therefore does not appear to be an artifact of pairing a clinical model with consumer-style questions.
Full results are in Table R1 under W2.

---

## W2. One modeling pipeline, though the contribution is stated as a general framework

We have narrowed the claim and widened the evidence.

**On the claim:** the contribution is now described as a recipe plus an evaluation decomposition for
open-ended, information-seeking clinical Q&A. We state explicitly what is not covered, namely task
types with no missing information to seek, such as closed-form multiple choice or summarization.

**On the evidence:** per Reviewer HomM's Q2, we re-ran the pipeline unchanged across three model
families and three benchmarks, 5 seeds per cell.

**Table R1.** Transfer across model families and benchmarks, all hyperparameters fixed.

| Benchmark | Base model (family) | Active inquiry, Base → LoRA | Context-seek, Base → LoRA |
|---|---|---|---|
| HealthBench | Mistral-Small-24B (Mistral) | 25.6% → 56.6% | 1.05 → 1.76 |
| HealthBench | Med42-8B (Llama-3) | 11.8% → 52.5% | 0.52 → 1.40 |
| ChatDoctor | Mistral-Small-24B (Mistral) | 7.8% → 58.6% | 0.83 → 1.49 |
| ChatDoctor | Med42-8B (Llama-3) | 2.9% → 86.4% | 0.71 → 1.81 |
| MedQuAD | Mistral-Small-24B (Mistral) | 6.5% → 26.5% | 0.32 → 1.23 |
| HealthBench | BioMistral-7B (scope cond.) | 11.7% → 10.1% (null) | 0.27 → 0.28 |

In four of the five working cells the adapter matches or exceeds its own teacher while requiring no
CoT at inference. We also report a precondition: on BioMistral-7B there is no effect, and the
diagnostic is that the teacher fails there too (the CoT wrapper reaches only 14.5%), so there is
nothing to distill. A clinician-facing benchmark (MedQA-USMLE, open-ended) is in progress. One element
of this weakness remains unaddressed: we vary the model and the benchmark, not the CoT protocol.

---

## W3. Automated grading is limited, so robustness is unclear

We share this concern and have not fully resolved it.

**Physician validation remains partial:** one of three raters has returned grades, giving $\kappa = 0.35$
against the Llama-3.1-8B grader, below our pre-registered target of 0.6. We report this as a
limitation on the aggregate-quality claims and do not present LLM grading as settled clinical
validation.

**On why the graders are general-purpose** (Reviewer HomM's Q4, Reviewer LxMF's Q3): the evaluator
must sit in a different model family from the Qwen-14B filter, since that separation is what prevents
filter-grader circularity in a self-distillation pipeline. Meditron-3, Med42-v2 and Aloe are all
Llama-derived, so promoting one to evaluator would place filter and grader in adjacent families and
weaken exactly that property. We plan to add a clinical judge as an additional robustness panel
reported alongside the primary grader rather than replacing it.

---

## W4. Unclear whether SFT distills calibration or enables mimicry

Reviewer LxMF proposed a specific test for this, and we ran it on the submitted evaluation data. Using
labels independent of the grader's own scores (HealthBench's `context_seeking` theme tag, and whether
a prompt's rubric rewards asking for clarification), we computed a discrimination index per condition,
defined as $P(\text{ask} \mid \text{info missing}) - P(\text{ask} \mid \text{self-contained})$.

| Condition | Discrimination | Bootstrap 95% CI |
|---|---|---|
| Base | +2.1pp | contains zero |
| Wrapper (CoT) | +11.0pp | excludes zero |
| LoRA | +10.5pp | excludes zero |

On a stricter labelling the adapter reaches +13.2pp, CI [+4.1, +21.7]. The base model does not
reliably discriminate; the wrapper and the adapter both do. The adapter therefore does not ask more
indiscriminately, it acquires targeting the base model lacks, which is the opposite of what mimicry
predicts.

A second line of evidence: on MedQuAD the CoT wrapper *degrades* scope bounding (1.85 → 1.61) and
hedging quality (1.45 → 1.40) while the distilled adapter *improves* both (1.89 and 1.78). The student
does not copy the teacher indiscriminately; the quality-filtering step removes the teacher's failures
before they reach the weights.

---

## W5. Discussion and contextualization lack clarity

We accept this and have made the changes rather than describing them:

- **Vague "not statements" removed,** replaced with positive claims stating the per-dimension numbers,
  including the accuracy (0.117 → 0.121) and completeness (0.168 → 0.169) figures that support the
  communication-not-knowledge reading.
- **Table 2** carries a direction arrow on every row, with a caption note that red-flag identification
  is higher-is-better and does not capture false positives.
- **Table 1's caption** now explains the differing sample sizes at the point of confusion: the
  two-pass CoT protocol exceeds the context window on ~4–5% of prompts, while LoRA has a 100% response
  rate.
- **Figure 2 split into two panels** so 0–2 scores and percentage rates no longer share an axis.
- **A clinical implications paragraph added,** stating what each behavior means in practice.

---

## W6. No comparison with prior work

This is a fair criticism, and it is the one our additional runs do not address; we will not imply
otherwise. The revision adds a Discussion subsection, "Comparison to Prior Calibration Approaches,"
positioning the method against uncertainty prompting [1,2], which calibrates at inference time whereas
we internalize the behavior into weights; STaR-style self-improvement [3], whose filter-then-finetune
logic we apply to behavioral rather than factual demonstrations; and Constitutional AI [4], which uses
self-critique where we use a structured CoT protocol as a behavioral teacher. It includes a summary
table comparing inference-time cost, need for a teacher model, behavioral versus factual focus, and
demonstrated generality.

A direct empirical head-to-head against an inference-time calibration baseline is the single most
valuable experiment we have not run, and we name it as the immediate next step rather than folding it
into a claim.

[1] Lin, Hilton, Evans. "Teaching Models to Express Their Uncertainty in Words." TMLR 2022.
[2] Tian et al. "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from
Fine-Tuned Language Models with Human Feedback." EMNLP 2023.
[3] Zelikman et al. "STaR: Bootstrapping Reasoning With Reasoning." NeurIPS 2022.
[4] Bai et al. "Constitutional AI: Harmlessness from AI Feedback." arXiv:2212.08073, 2022.

---

# BEFORE SUBMITTING: replace the stale drafts

Several "Authors comment" boxes already contain drafts written before these runs existed. Submitted
alongside the responses above, they would contradict them:

- **FzpA W2 draft** says commercial models cannot be LoRA-adapted and lists "extending to Llama,
  Mistral" as future work. We have now done exactly that; as written the draft hides the strongest
  answer to that reviewer.
- **FzpA W1 draft** promises to reword the abstract to "a demonstration in one concrete setting." Keep
  the narrowing of the claim, but it should no longer describe the evidence as a single setting.
- **HomM W1 draft** frames the work as "proof-of-concept rather than comprehensive validation" and
  lists testing other models and domains as next steps. Two of three are now done, so it concedes more
  than warranted, to the reviewer who rated Reject.
- **HomM W3 draft** describes physician validation as a plan; it should state the current partial
  result (1 of 3 raters, $\kappa = 0.35$).

Still accurate and can stand: the prior-work draft (meta W6) and the Table 1 sample-size draft
(LxMF Q1).
