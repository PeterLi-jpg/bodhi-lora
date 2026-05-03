# Tunix migration (Stage 3 LoRA SFT)

Stage 3 (TPU LoRA fine-tune of MedGemma-27B) was rebuilt on top of Google's
[tunix](https://tunix.readthedocs.io/) + [qwix](https://qwix.readthedocs.io/)
stack. This page documents why the migration happened, what changed, how to
run the new path, and how the old path is being retired.

## Why we migrated

The previous Stage 3 entry (`scripts/train_lora_maxtext.py` + `scripts/maxtext_lora/*`)
was roughly 1500 lines of custom JAX/Optax/Flax glue layered on top of
MaxText's trainer. Every TPU-side bug we hit between smoke runs v18 and v27
sat at an integration seam we authored ourselves: the LoRA injector's `ToLinen`
wrapping, FSDP assertion mismatches against MaxText's mesh assumptions, the
init-batch shape we passed to MaxText, HBM OOM at the smoke step batch, and so
on. None of those were bugs in MaxText or in JAX. They were bugs in the seam.

`tunix.sft.peft_trainer.PeftTrainer` + `qwix.apply_lora_to_model` is Google's
official, supported path for SFT + LoRA on Gemma-family models on TPU. The
original reason we bypassed it (PR #169) was that `google-tunix` requires
Python 3.11, while v6e TPU VMs ship with Python 3.10 by default. PRs #181-#188
closed that gap: `tpu/setup_tpu.sh` provisions Python 3.11 into
`~/.venv-py311`, and every TPU launcher's `python` invocation now routes
through that venv. With the py3.11 wall down, tunix is usable, and there is no
longer a reason to maintain the bespoke MaxText glue.

## What changed

| Old MaxText path | New tunix path |
|---|---|
| `scripts/train_lora_maxtext.py` | `scripts/train_lora_tunix.py` |
| `scripts/maxtext_lora/{injector,layer}.py` | replaced by `qwix.apply_lora_to_model` |
| `scripts/convert_medgemma_to_maxtext.py` | DELETED — tunix loads safetensors directly |
| `scripts/export_maxtext_lora_to_peft.py` | `scripts/export_tunix_lora_to_peft.py` |
| `tpu/launch_5seeds_maxtext.sh` | `tpu/launch_5seeds_tunix.sh` |
| `configs/lora_medgemma27b_maxtext*.yaml` | `configs/lora_medgemma27b_tunix_smoke.yaml` |

Stages 1, 2, 4, and 5 are unaffected. The exported PEFT adapter contract
(what Stage 4 consumes) is unchanged.

## How to use the new path

```bash
export GCS_OUTPUT_PATH=gs://...
export GCS_DATA_PATH=gs://.../seed_42
export TRAIN_CONFIG=configs/lora_medgemma27b_tunix_smoke.yaml
bash tpu/launch_5seeds_tunix.sh
```

Set `HF_TOKEN` first for gated MedGemma access. The launcher provisions the
TPU VM, runs `tpu/setup_tpu.sh` (which builds the `~/.venv-py311` env with
tunix + qwix already installed), then drives `scripts/train_lora_tunix.py`
across the configured seeds.

## Deprecation timeline

The old MaxText path is **not** removed yet. It stays in the repo until the
tunix path is verified end-to-end on a real TPU smoke run. Once that's green,
a single follow-up cleanup PR removes the obsolete files in one shot.

Until that cleanup lands:

- New work goes to the tunix path.
- The MaxText path remains runnable so existing evidence/runs can be
  reproduced if needed.
- Bug fixes against the MaxText glue are no longer in scope unless they
  unblock an in-flight reproduction.

## References

- tunix docs: https://tunix.readthedocs.io/
- qwix docs: https://qwix.readthedocs.io/
- Migration PRs: #196-#203 (8-unit batch)
- Earlier blocker context: `KNOWN_ISSUES.md` (Stage 3b section), PR #169,
  PRs #181-#188 (py3.11 enablement)
