# Response to Reviewer HomM

We thank the reviewer for an assessment that was generous about the framing and unambiguous about
the problem. The reviewer also suspected we were already pursuing extensions across more models and
benchmarks. That was correct, and several completed during the discussion period.

### Q1: Why was the evaluation protocol so limited?

The 27B TPU runs were expensive, and we prioritized depth (five seeds, bootstrap intervals, an
integrity-checked holdout) over breadth. For a paper claiming generality that was the wrong trade,
and we have begun correcting it.

### W1, Q2: Additional base models and evaluation sets

We applied the pipeline unchanged, with the paper's hyperparameters fixed, to two further model
families and two further benchmarks, five seeds per cell:

| Cell | Model family | Active inquiry (Base to LoRA) |
|---|---|---|
| HealthBench x Mistral-Small-24B | Mistral, general | 25.6% to 56.6% |
| HealthBench x Med42-8B | Llama-3, clinical | 11.8% to 52.5% |
| ChatDoctor x Mistral-Small-24B | Mistral, general | 7.8% to 58.6% |
| ChatDoctor x Med42-8B | Llama-3, clinical | 2.9% to 86.4% |
| MedQuAD x Mistral-Small-24B | Mistral, general | 6.5% to 26.5% |
| HealthBench x BioMistral-7B | scope condition | 11.7% to 10.1% (null) |

Context-seeking moves the same way in each cell. Q2 asked for one or two additional base models or
one additional evaluation set; we have both, across model families rather than within one. A
clinician-facing benchmark (MedQA-USMLE, open-ended) is in progress.

Two results are not simply confirmatory:

- **The student does not copy the teacher indiscriminately.** On MedQuAD the CoT wrapper *degrades*
  scope bounding (1.85 to 1.61) and hedging (1.45 to 1.40), while the adapter *improves* both (1.89
  and 1.78). The filtering step removes the teacher's failures before they reach the weights.
- **A boundary condition.** BioMistral-7B shows no effect, and the diagnostic is that the teacher
  fails there too (wrapper 14.5%). The recipe requires a base capable of following the protocol, and
  we now state that as a precondition rather than leaving a reader to find it.

We note that the third element of this weakness, a single CoT protocol, is not yet addressed. We
list it as outstanding.

### Aggregate quality: what the behavior gains cost

Reported plainly, including where unfavourable:

- **HealthBench:** essentially unchanged (Mistral-Small-24B 0.425 to 0.411; Med42-8B 0.405 to
  0.392), consistent with the submitted non-inferiority result.
- **New benchmarks:** modest decreases, largest on MedQuAD (0.672 to 0.586).

We think the MedQuAD case is informative. Its rubric rewards agreement with a reference answer, so a
response that asks a clarifying question instead of answering scores lower by construction.
Consistent with that, the wrapper, which asks most, drops furthest (0.672 to 0.542). This is the
measurement problem the paper is about, and it is why we report the decomposition alongside the
aggregate rather than instead of it.

### W3, Q4: Automated grading, and why the graders are not clinical models

We share this concern and have not resolved it.

- **Physician validation remains partial:** one of three physicians has returned grades, giving
  kappa = 0.35 against the Llama-3.1-8B grader, below our pre-registered target of 0.6. We report
  this as a limitation on the aggregate-quality claims and do not present LLM grading as settled
  clinical validation.
- **On grader choice (Q4):** the evaluator must sit in a different model family from the Qwen-14B
  filter, because that separation prevents filter-grader circularity. Meditron-3, Med42-v2 and Aloe
  are Llama-derived, so promoting one to evaluator would place filter and grader in adjacent
  families and weaken that property. We plan to add a clinical judge as an additional robustness
  panel alongside the primary grader rather than replacing it.

### W4, Q3: No comparison to prior work

This is a fair criticism the new runs do not address, and we will not pretend otherwise. The
revision adds a Discussion subsection positioning the method against:

- **Uncertainty prompting** (Lin et al., Tian et al.): calibrates at inference time, whereas we
  internalize into weights.
- **STaR-style self-improvement** (Zelikman et al.): same filter-then-finetune logic, applied to
  behavioral rather than factual demonstrations.
- **Constitutional AI** (Bai et al.): self-critique, where we use a structured CoT protocol as a
  behavioral teacher.

We will include a summary table comparing inference-time cost, need for a teacher model, behavioral
versus factual focus, and demonstrated generality. A direct empirical head-to-head against an
inference-time calibration baseline is the single most valuable experiment we have not run, and we
name it as the next step.

### On workshop versus conference scope

We understand the assessment as given, and think it was fair to the submitted version. The specific
gap identified has been narrowed: five-seed evidence across three model families and three
benchmarks, plus a boundary condition showing where the recipe fails and why. Two mechanistic
analyses requested by Reviewer LxMF were also completed, distinguishing calibration from surface
mimicry and identifying output-format leakage as the cause of the CoT/LoRA interference.

Still outstanding, and we do not claim otherwise: the prior-work comparison, the completed
three-physician validation, and a second CoT protocol. We would ask the reviewer to reconsider in
light of the added evidence, with those gaps visible.
