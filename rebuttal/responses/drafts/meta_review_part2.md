Continued from the previous comment.

**W5. Discussion and contextualization lack clarity.**

Agreed, and these are the priority fixes. Figure 2 is split so graded dimensions and percentage rates have separate axes, Table 2 gets a direction arrow on every row, and the vague formulations Reviewer FzpA quoted are replaced with per-dimension numbers and a verbatim example each.

The behavioural gain and the aggregate cost will be reported together for every cell. Aggregate rubric score falls from base to adapter in all six reported cells, by 0.013 to 0.015 on HealthBench and 0.016 to 0.086 on ChatDoctor and MedQuAD, with paired t-tests over five seeds putting all six below p = 0.02 and all six surviving Holm correction. The submitted cell rose, 0.459 to 0.463, and stands apart from the six. Measured against the wrapper, the inference-time alternative for obtaining this behaviour, the adapter is lower in three cells, level in two and higher in one, and the wrapper is itself below base in all six. Against that cost, the adapter carries the behaviour without the protocol: one generation call where the protocol takes two, roughly 58% fewer output tokens, and a response on every prompt, where the two-pass protocol returned nothing on about 5% of them. On active inquiry we did not detect a difference between the two, bootstrap CI [-0.069, +0.021]. The abstract will carry that trade-off alongside the gain, replacing the submitted claim that aggregate clinical quality was preserved. The Discussion will give a criterion for when the gain is worth the cost, framed by deployment risk.

Four statements we withdraw or scope. We first attributed part of the aggregate decrease to a rubric artefact penalising asking; we tested it and it failed. A convergence claim was computed with the null excluded. "Scope condition" was the wrong label for the BioMistral cell. And a count of cells where the adapter matched or exceeded its teacher was stated without saying what it counted; scoped, it is four of the five added cells that transfer, and four of seven once the submitted cell and the null are included.

The interference mechanism also changed. The paper attributes the stacking interference to competition for behavioural capacity. Testing Reviewer LxMF's alternatives, truncation and attention dilution, pointed to control of the output format. Format leakage reaches 54.6% of responses under LoRA plus CoT against 0.2 to 3.0% in every other condition, red-flag identification scores 1.33 in the leaked subset against 1.73 in the non-leaked one on a 0 to 2 scale, above LoRA alone at 1.67, and within non-leaked responses the score rises across length quartiles, where dilution predicts a fall. The non-leaked subset is identified by a format property after the fact, so we cannot fully exclude leakage correlating with prompt difficulty.

**W6. No comparison with prior work.**

This remains open, and it is the one item the added runs leave untouched. Section 2 positions the method against STaR, Constitutional AI and chain-of-thought prompting conceptually. An empirical head-to-head against a published inference-time calibration method, Tian et al. (2023), is the next experiment.

**Two items for the record.**

Reviewer LxMF proposed both analyses above, on surface mimicry and on the cause of the stacking interference. We ran both and reported them in full, and the second replaced a mechanism we had asserted with a better-supported one. If the per-cell outputs and code would help you weigh any of this, the rules allow us to send you an anonymized link on request. Reviewer LxMF has since confirmed that their concerns are sufficiently addressed.

On the consolidation Reviewer HomM raised: if accepted, the camera-ready carries all seven added cells in the main evaluation, the BioMistral null and the MedQA cell among them; the behavioural gain and the aggregate cost stated together for every cell, with the trade-off in the abstract; a Discussion criterion framed by deployment risk; the scope named as patient-facing clinical Q&A; and the interpretations we corrected in this discussion appearing as corrected. That list is on the record in our HomM reply.

This discussion asked a great deal of the reviewers. Reviewer LxMF proposed two tests that changed what we could claim. Reviewer HomM pressed for the full tables and for the precision that came with them. Reviewer FzpA's questions sharpened how we describe the setting and what the rubric measures. We are grateful to all three, and to you for coordinating a process that has made the paper more accurate.
