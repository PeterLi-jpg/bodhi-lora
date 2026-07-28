One comment per listed weakness, each self-contained for its own box. The three Strengths boxes need
no reply. Note on scope: NeurIPS permits additional results that reviewers specifically requested;
each new experiment below is attributed to the reviewer who asked for it.

---

## W1. Use of MedGemma does not align with HealthBench use cases

We agree this was a fair concern. As Reviewer FzpA suggested a more widely used model would be more
appropriate, we re-ran the identical pipeline with all hyperparameters fixed on
**Mistral-Small-24B-Instruct**, a general-purpose non-clinical open-weight model. The effect
reproduces on HealthBench: active inquiry 25.6% to 56.6%, context-seeking 1.05 to 1.76 (5 seeds). We
also ran **Med42-8B**, clinically tuned from a third family (Llama-3). The submitted finding
therefore does not appear to be an artifact of pairing a clinical model with consumer-style
questions.

---

## W2. One modeling pipeline, though the contribution is stated as a general framework

We have narrowed the claim and widened the evidence. On the claim: the contribution is now described
as a recipe plus an evaluation decomposition for open-ended, information-seeking clinical Q&A, and we
state what is not covered (task types with no missing information to seek, such as closed-form
multiple choice or summarization).

On the evidence, per Reviewer HomM's Q2, the pipeline was re-run unchanged across three model
families (Gemma, Mistral, Llama-3) and three benchmarks (HealthBench, ChatDoctor, MedQuAD), 5 seeds
per cell. Active inquiry and context-seeking rise in every capable configuration, and in four of five
cells the adapter matches or exceeds its own teacher without CoT at inference. We also report a
precondition: on BioMistral-7B there is no effect, and the diagnostic is that the teacher fails there
too (wrapper reaches only 14.5%), so there is nothing to distill.

---

## W3. Automated grading is limited, so robustness is unclear

We share this concern and have not fully resolved it. The physician validation remains partial: one
of three raters has returned grades, giving κ = 0.35 against the Llama-3.1-8B grader, below our
pre-registered target of 0.6. We report this as a limitation on the aggregate-quality claims and do
not present LLM grading as settled clinical validation.

On why the graders are general-purpose (Reviewer HomM's Q4, Reviewer LxMF's Q3): the evaluator must
sit in a different model family from the Qwen-14B filter, since that separation prevents
filter-grader circularity in a self-distillation pipeline. Meditron-3, Med42-v2 and Aloe are all
Llama-derived, so promoting one to evaluator would place filter and grader in adjacent families and
weaken exactly that property. We plan to add a clinical judge as an additional robustness panel
alongside the primary grader rather than replacing it.

---

## W4. Unclear whether SFT distills calibration or enables mimicry

Reviewer LxMF proposed a specific test, and we ran it on the submitted evaluation data. Using labels
independent of the grader's own scores, we computed a discrimination index per condition, defined as
P(ask | information missing) minus P(ask | self-contained). The base model does not reliably
discriminate (+2.1pp, bootstrap 95% CI contains zero); the CoT wrapper does (+11.0pp, CI excludes
zero); the LoRA adapter does (+10.5pp, CI excludes zero; +13.2pp, CI [+4.1, +21.7] on a stricter
labelling). The adapter does not ask more indiscriminately, it acquires targeting the base model
lacks, which is the opposite of what mimicry predicts.

A second piece of evidence: on MedQuAD the CoT wrapper *degrades* scope bounding (1.85 to 1.61) and
hedging (1.45 to 1.40) while the distilled adapter *improves* both (1.89 and 1.78). The student does
not copy the teacher indiscriminately; the filtering step removes the teacher's failures before they
reach the weights.

---

## W5. Discussion and contextualization lack clarity

We accept this and have made the changes rather than describing them. The vague "not statements" are
replaced with positive claims stating the per-dimension numbers, including the accuracy (0.117 to
0.121) and completeness (0.168 to 0.169) figures supporting the communication-not-knowledge reading.
Table 2 carries a direction arrow on every row, with a caption note that red-flag identification is
higher-is-better and does not capture false positives. Table 1's caption now explains the differing
sample sizes (the two-pass CoT protocol exceeds the context window on ~4-5% of prompts; LoRA has a
100% response rate). Figure 2 is split into two panels so 0-2 scores and percentage rates no longer
share an axis. A new paragraph states what each behavior means clinically.

---

## W6. No comparison with prior work

This is a fair criticism, and it is the one our additional runs do not address; we will not imply
otherwise. The revision adds a Discussion subsection, "Comparison to Prior Calibration Approaches,"
positioning the method against uncertainty prompting (Lin et al., Tian et al.), which calibrates at
inference time whereas we internalize into weights; STaR-style self-improvement (Zelikman et al.),
whose filter-then-finetune logic we apply to behavioral rather than factual demonstrations; and
Constitutional AI (Bai et al.), which uses self-critique where we use a structured CoT protocol as a
behavioral teacher. It includes a summary table comparing inference-time cost, need for a teacher
model, behavioral versus factual focus, and demonstrated generality. A direct empirical head-to-head
against an inference-time calibration baseline is the single most valuable experiment we have not
run, and we name it as the immediate next step.

---

# BEFORE SUBMITTING: replace the stale drafts

Several "Authors comment" boxes already contain drafts written before these runs existed. Submitted
alongside the responses above, they would contradict them:

- **FzpA W2 draft** says commercial models cannot be LoRA-adapted and lists "extending to Llama,
  Mistral" as future work. We have now done exactly that; as written the draft hides the strongest
  answer to that reviewer.
- **FzpA W1 draft** promises to reword the abstract to "a demonstration in one concrete setting."
  Keep the narrowing, but it should no longer describe the evidence as a single setting.
- **HomM W1 draft** frames the work as "proof-of-concept rather than comprehensive validation" and
  lists testing other models and domains as next steps. Two of three are now done, so it concedes
  more than warranted, to the reviewer who rated Reject.
- **HomM W3 draft** describes physician validation as a plan; it should state the current partial
  result (1 of 3 raters, κ = 0.35).

Still accurate and can stand: the prior-work draft (meta W6) and the Table 1 sample-size draft
(LxMF Q1).
