A companion post to the previous comment, since the rules allow several comments and the full breakdown did not fit alongside the tables you asked for.

You asked, in effect, what the movement is being measured against, and that is a fair thing to want explicit. Our earlier tables reported base to LoRA, which shows the adapter differs from the unadapted model but not what it buys over simply running the protocol at inference. Below is every condition for every cell, so the comparison you prefer is available rather than asserted. Means over five seeds, 200 prompts per seed per condition. BioMistral is included as the documented null.

**Active inquiry (%)**

| Benchmark | Base model | Base | Base+CoT | LoRA | LoRA+CoT |
|---|---|---|---|---|---|
| HealthBench | Mistral-Small-24B | 25.6 | 54.2 | 56.6 | 72.4 |
| HealthBench | Med42-8B | 11.8 | 57.1 | 52.5 | 84.3 |
| HealthBench | BioMistral-7B | 11.7 | 14.5 | 10.1 | 10.5 |
| ChatDoctor | Mistral-Small-24B | 7.8 | 54.5 | 58.6 | 48.8 |
| ChatDoctor | Med42-8B | 2.9 | 83.4 | 86.4 | 77.6 |
| MedQuAD | Mistral-Small-24B | 6.5 | 23.1 | 26.5 | 42.2 |

**Context-seeking**

| Benchmark | Base model | Base | Base+CoT | LoRA | LoRA+CoT |
|---|---|---|---|---|---|
| HealthBench | Mistral-Small-24B | 1.05 | 1.80 | 1.76 | 1.77 |
| HealthBench | Med42-8B | 0.52 | 1.47 | 1.40 | 1.88 |
| HealthBench | BioMistral-7B | 0.27 | 0.40 | 0.28 | 0.28 |
| ChatDoctor | Mistral-Small-24B | 0.83 | 1.59 | 1.49 | 1.17 |
| ChatDoctor | Med42-8B | 0.71 | 1.82 | 1.81 | 1.83 |
| MedQuAD | Mistral-Small-24B | 0.32 | 0.87 | 1.23 | 1.68 |

**Red-flag identification**

| Benchmark | Base model | Base | Base+CoT | LoRA | LoRA+CoT |
|---|---|---|---|---|---|
| HealthBench | Mistral-Small-24B | 1.03 | 1.73 | 1.68 | 1.65 |
| HealthBench | Med42-8B | 0.81 | 1.10 | 1.03 | 1.09 |
| HealthBench | BioMistral-7B | 0.29 | 0.32 | 0.37 | 0.45 |
| ChatDoctor | Mistral-Small-24B | 1.44 | 1.92 | 1.88 | 1.85 |
| ChatDoctor | Med42-8B | 1.15 | 1.37 | 1.19 | 1.35 |
| MedQuAD | Mistral-Small-24B | 0.57 | 0.74 | 0.68 | 1.33 |

**Scope bounding**

| Benchmark | Base model | Base | Base+CoT | LoRA | LoRA+CoT |
|---|---|---|---|---|---|
| HealthBench | Mistral-Small-24B | 1.77 | 1.96 | 1.97 | 1.91 |
| HealthBench | Med42-8B | 1.50 | 1.82 | 1.79 | 1.87 |
| HealthBench | BioMistral-7B | 0.75 | 0.62 | 0.70 | 0.69 |
| ChatDoctor | Mistral-Small-24B | 1.89 | 1.95 | 1.94 | 1.67 |
| ChatDoctor | Med42-8B | 1.56 | 1.95 | 1.87 | 1.87 |
| MedQuAD | Mistral-Small-24B | 1.85 | 1.61 | 1.89 | 1.95 |

**Hedging quality**

| Benchmark | Base model | Base | Base+CoT | LoRA | LoRA+CoT |
|---|---|---|---|---|---|
| HealthBench | Mistral-Small-24B | 1.68 | 1.89 | 1.90 | 1.83 |
| HealthBench | Med42-8B | 1.38 | 1.67 | 1.64 | 1.79 |
| HealthBench | BioMistral-7B | 0.67 | 0.58 | 0.57 | 0.59 |
| ChatDoctor | Mistral-Small-24B | 1.64 | 1.82 | 1.78 | 1.41 |
| ChatDoctor | Med42-8B | 1.51 | 1.74 | 1.59 | 1.67 |
| MedQuAD | Mistral-Small-24B | 1.46 | 1.40 | 1.78 | 1.92 |
Three readings we would draw, and one we would not.

Against the base, the adapter moves every dimension in every transferring cell, which is the paper's headline claim and the one these tables support most cleanly.

Against the teacher, the adapter is close but generally slightly behind on the graded dimensions, within 0.10 in fourteen of twenty cases and behind by at most 0.19, while being ahead on active inquiry in four of five cells. The honest summary is near-parity at one forward pass rather than the protocol's roughly two times inference cost and about 5% response loss, not superiority. Our earlier reply said the adapter matches or exceeds its teacher in four of five cells; that was computed on active inquiry, and we should have scoped it.

Against LoRA+CoT, stacking is better than the adapter alone about as often as it is worse, which is consistent with what we said about the interference being benchmark-dependent rather than a general property.

What we would not read from these tables is a claim about which pipeline component causes any of it. Without an unfiltered-SFT arm we cannot separate the filter's contribution from the demonstrations themselves, and we have said so.
