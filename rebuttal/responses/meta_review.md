# Response to the Area Chair (Meta Review)

One comment per listed weakness, each self-contained for pasting into its own box.
The three Strengths boxes need no reply; leave them empty or write a one-line thanks.

---

## W1. "Use of MedGemma does not align with HealthBench use cases"

We agree this was a fair concern about the submitted version, and we tested it directly rather than
arguing the point. Running the identical pipeline, hyperparameters fixed, on
**Mistral-Small-24B-Instruct** (a general-purpose, non-clinical, widely used open-weight model)
reproduces the result on HealthBench: active inquiry 25.6% to 56.6%, context-seeking 1.05 to 1.76
across five seeds. We also ran **Med42-8B**, a clinically tuned model from a third family (Llama-3).
The submitted MedGemma finding therefore does not appear to be an artifact of pairing a clinical
model with consumer-style questions. Details are in the new generality appendix.

---

## W2. "One modeling pipeline is used, though the contribution is stated as a general framework"

We have narrowed the claim and widened the evidence.

On the claim: the contribution is now described as a recipe plus an evaluation decomposition for
open-ended, information-seeking clinical Q&A, and we state explicitly what is not covered (task
types with no missing information to seek, such as closed-form multiple choice or summarization).

On the evidence: the pipeline was re-run unchanged across **three model families** (Gemma, Mistral,
Llama-3) and **three benchmarks** (HealthBench, ChatDoctor, MedQuAD), five seeds per cell. Active
inquiry and context-seeking rise in every capable configuration, and in four of five cells the
adapter matches or exceeds its own teacher without requiring CoT at inference. We also report a
precondition: on BioMistral-7B there is no effect, and the diagnostic is that the teacher fails
there too (wrapper reaches only 14.5%), so there is nothing to distill.

---

## W3. "Automated grading is limited and therefore robustness of results is unclear"

We share this concern and have not fully resolved it. The physician validation remains partial: one
of three raters has returned grades, giving kappa = 0.35 against the Llama-3.1-8B grader, below our
pre-registered target of 0.6. We report this as a limitation on the aggregate-quality claims and do
not present LLM grading as settled clinical validation.

On why the graders are general-purpose: the evaluator must sit in a different model family from the
Qwen-14B filter, since that separation is what prevents filter-grader circularity in a
self-distillation pipeline. The clinically tuned open-weight models usually suggested (Meditron-3,
Med42-v2, Aloe) are Llama-derived, so promoting one to evaluator would place filter and grader in
adjacent families and weaken exactly that property. We plan to add a clinical judge as an additional
robustness panel alongside the primary grader rather than replacing it.

---

## W4. "Unclear whether SFT distills calibration or enables mimicry of favorable responses"

Reviewer LxMF proposed a specific test for this, and we ran it on the submitted evaluation data.
Using labels independent of the grader's own scores, we computed a discrimination index for each
condition, defined as P(ask | information genuinely missing) minus P(ask | self-contained).

The base model does not reliably discriminate (+2.1pp, bootstrap 95% CI contains zero). The CoT
wrapper does (+11.0pp, CI excludes zero). The LoRA adapter does (+10.5pp, CI excludes zero; +13.2pp
with CI [+4.1, +21.7] on a stricter labelling). The adapter therefore does not ask more
indiscriminately; it acquires targeting the base model lacks, which is the opposite of what surface
mimicry predicts.

A second piece of evidence comes from the new benchmarks: on MedQuAD the CoT wrapper *degrades*
scope bounding (1.85 to 1.61) and hedging (1.45 to 1.40) while the distilled adapter *improves* both
(1.89 and 1.78). The student is not copying the teacher indiscriminately; the filtering step removes
the teacher's failures before they reach the weights.

---

## W5. "Discussion and contextualization of results lack clarity"

We accept this and have made the specific changes rather than describing them:

- The vague "not statements" are replaced with positive claims that state the per-dimension numbers,
  including the accuracy and completeness figures that support the "communication, not knowledge"
  reading.
- Table 2 now carries an explicit direction arrow on every row, with a caption note that red-flag
  identification is higher-is-better and does not capture false positives.
- Table 1's caption explains the differing sample sizes (200 vs 191 vs 192) at the point of
  confusion: the two-pass CoT protocol exceeds the context window on roughly 4 to 5% of prompts,
  while LoRA has a 100% response rate.
- Figure 2 is split into two panels so 0 to 2 scores and percentage rates no longer share an axis.
- A new paragraph states what each behavior means clinically.

---

## W6. "There is no comparison with prior work"

This is a fair criticism, and it is the one our additional runs do not address. We will not imply
otherwise. The revision adds a Discussion subsection, "Comparison to Prior Calibration Approaches,"
positioning the method against uncertainty prompting (Lin et al., Tian et al.), which calibrates at
inference time whereas we internalize into weights; STaR-style self-improvement (Zelikman et al.),
whose filter-then-finetune logic we apply to behavioral rather than factual demonstrations; and
Constitutional AI (Bai et al.), which uses self-critique where we use a structured CoT protocol as a
behavioral teacher. It includes a summary table comparing inference-time cost, need for a teacher
model, behavioral versus factual focus, and demonstrated generality.

A direct empirical head-to-head against an inference-time calibration baseline is the single most
valuable experiment we have not run, and we name it as the immediate next step rather than folding
it into a claim.

---

# IMPORTANT: replace the drafts already in some boxes

Several "Authors comment" boxes already contain earlier drafts written before these runs existed.
If those are submitted alongside the responses above, the overall reply will contradict itself.
Specifically:

- **FzpA W2 draft** says commercial models cannot be LoRA-adapted and lists "extending to other
  open-weight models (e.g. Llama, Mistral)" as future work. We have now done exactly that, so this
  draft understates the response and reads as if we did not.
- **FzpA W1 draft** promises to reword the abstract to a "demonstration in one concrete setting."
  Keep the narrowing, but it should no longer say the empirical basis is a single setting.
- **HomM W1 draft** frames the work as "a proof-of-concept rather than a comprehensive validation"
  and says "the important next steps are testing other models, domains, and CoT protocols." Two of
  those three are now done, so this draft concedes more than is warranted.
- **HomM W3 draft** says physician validation will be prioritized; it should state the current
  partial result (1 of 3 raters, kappa = 0.35) rather than describing the plan only.

The prior-work draft (meta-review W6) and the Table 1 sample-size draft (LxMF Q1) are still accurate
and can stand.
