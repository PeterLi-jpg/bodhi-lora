# MaxText migration for Stage 3 (LoRA fine-tune)

> **Note (2026-05):** The MaxText path described here is being migrated to tunix.
> See [contributions/tunix-migration.md](../contributions/tunix-migration.md) for
> the current path. Content below remains as historical context for the
> PyTorch+torch_xla → MaxText migration that preceded the tunix migration.

This doc records why Stage 3 of the BOHDI-LoRA pipeline gained a JAX-native
training path on top of a forked MaxText, and what every other stage needs
to keep doing for that path to slot in cleanly.

The PyTorch path stays in tree as the fallback. Stages 1, 2, 4, and 5 are
unchanged.

## Why we forked MaxText

Stage 3 on TPU under PyTorch + `torch_xla` 2.7 + FSDPv2 + Gemma-3-27B
hangs on the first `xm.mark_step()` of the step graph for over an hour.
We verified this live on v6e-8: the trace shows the XLA compiler stuck
inside `torch_xla`'s SPMD partitioner, no progress, no useful stderr.
Lowering rank, disabling activation checkpointing, and shrinking the
sequence length all reproduced the hang.

There is no free workaround. The XLA pathology is specific to the
`torch_xla` FSDPv2 sharding rules colliding with Gemma-3's 27B step
graph. JAX-native MaxText does not exhibit this hang because it never
goes through the `torch_xla` partitioner.

We forked MaxText (rather than using upstream) because MaxText doesn't
ship LoRA. The fork adds a small `scripts/maxtext_lora/` package
(LoRA `Linear` layer, injector, and a thin training entry point) on
top of MaxText's existing Gemma-3 modeling code.

## What changed

- **Stage 3 (training)** gained a JAX-native path:
  `scripts/train_lora_maxtext.py` (Unit 7) loads MedGemma weights via
  `scripts/convert_medgemma_to_maxtext.py` (Unit 4) and traces via
  `scripts/convert_traces_to_maxtext.py` (Unit 5), runs LoRA fine-tune
  through the forked MaxText, then exports a HuggingFace PEFT adapter
  via `scripts/export_maxtext_lora_to_peft.py` (Unit 6).
- **Stages 1, 2, 4, 5 are unchanged.** Stage 4 (`eval_healthbench.py`)
  and `_xla_lora_inference.py` still load `--lora-path
  checkpoints/seed_<N>/best/` and call
  `peft.PeftModel.from_pretrained` exactly as before.
- **PyTorch fallback stays in tree.** `scripts/train_lora.py` and
  `tpu/launch_5seeds.sh` are untouched. If the JAX path hits an issue
  we can fall back without touching Stages 1, 2, 4, or 5.

## Run plan

The three Stage 3 scripts and how they chain together. Flag names below
match `--help` for each script as of this writing; re-run `--help` if
in doubt.

1. **Convert MedGemma weights once** to MaxText's orbax format:
   ```
   python scripts/convert_medgemma_to_maxtext.py \
       --hf-path google/medgemma-27b-text-it \
       --output gs://bohdi-cache/maxtext/medgemma27b/
   ```
   The converter writes the actual checkpoint under `{output}/0/items`.
   This is a one-time cost; the converted checkpoint is reused across
   every seed and every retry. The base-checkpoint path is read by the
   trainer through the YAML config (`load_parameters_path`), not via a
   CLI flag.
2. **Train one seed** end to end on a v6e-8. The trainer's CLI is
   intentionally tiny; everything else (base checkpoint path, LoRA
   rank, dataset paths, optimizer) lives in the YAML:
   ```
   python scripts/train_lora_maxtext.py \
       --config configs/lora_medgemma27b_maxtext.yaml \
       --seed 42 \
       --output-dir checkpoints/seed_42
   ```
   `--output-dir` produces two subdirs: `<output-dir>/orbax/` (periodic
   LoRA orbax checkpoints) and `<output-dir>/best/` (the final HF/PEFT
   adapter the eval consumes).
3. **Export the orbax LoRA to a PEFT adapter** if you need to invoke
   the exporter manually (the trainer already writes `<output-dir>/best/`
   on its own; this step is for re-exporting from a saved orbax step):
   ```
   python scripts/export_maxtext_lora_to_peft.py \
       --orbax-path checkpoints/seed_42/orbax/<step> \
       --output-dir checkpoints/seed_42/best \
       --base-model google/medgemma-27b-text-it \
       --config configs/lora_medgemma27b_maxtext.yaml
   ```
   `--orbax-path` is the leaf step directory, not the CheckpointManager
   root. `--config` lets the exporter read `r`, `lora_alpha`,
   `target_modules`, and `variant` from the same YAML the trainer used;
   individual CLI flags (`--lora-r`, `--lora-alpha`, `--target-modules`,
   etc.) override YAML values when present.
4. **Verify Stage 4 round-trips** the exported adapter with the
   existing eval script, no code changes:
   ```
   python scripts/eval_healthbench.py \
       --model google/medgemma-27b-text-it \
       --lora-path checkpoints/seed_42/best \
       --sample-ids data/raw/hard_200_sample_ids.json \
       --output eval/seed_42/lora_no_wrapper.json
   ```
   If this works, the contract holds.

## How to run a smoke

End-to-end smoke is one command via `tpu/launch_5seeds_maxtext.sh`,
which fans out to TPU spot VMs and runs the full Stage 1 -> 2 -> 3 -> 4
chain per seed. The launcher consumes a few env vars to scope the run
down to a fast end-to-end check on one seed:

```
GCS_OUTPUT_PATH=gs://your-bucket/bohdi-runs \
SEEDS="42" \
MAX_EXAMPLES=100 \
EVAL_MAX=10 \
TRAIN_CONFIG=configs/lora_medgemma27b_maxtext_smoke.yaml \
bash tpu/launch_5seeds_maxtext.sh
```

- `GCS_OUTPUT_PATH` is strongly recommended: every preempt without it
  loses local-disk progress. With it, traces, checkpoints, and eval
  JSONs are mirrored to `${GCS_OUTPUT_PATH}/seed_<N>/`.
- `SEEDS` defaults to `42 7 13 99 101`; override to a single seed for
  the smoke.
- `MAX_EXAMPLES` caps Stage 1 trace generation; `EVAL_MAX` caps Stage 4
  eval prompts. Both are passed through to the underlying scripts as
  `--max-examples`.
- `TRAIN_CONFIG` swaps the train YAML; the smoke variant
  (`configs/lora_medgemma27b_maxtext_smoke.yaml`) shrinks steps and
  batch size for a fast end-to-end check.

For the full 5-seed production run, drop the smoke env-vars and let the
defaults hold (`SEEDS="42 7 13 99 101"`, `TRAIN_CONFIG=configs/lora_medgemma27b_maxtext.yaml`).

## Checkpoint format (contract for the exporter)

`checkpoints/seed_<N>/best/` is the only thing Stage 4 sees. It MUST be
a valid HuggingFace PEFT adapter directory. The exporter (Unit 6) is
the seam that converts MaxText params back into this layout, and it
MUST satisfy these contracts:

- `adapter_config.json`: `peft.LoraConfig` serialized via
  `LoraConfig.save_pretrained`. Required keys (verified from the
  PyTorch path in `scripts/train_lora.py`): `peft_type=LORA`,
  `target_modules`, `r`, `lora_alpha`, `bias`, `task_type=CAUSAL_LM`.
- `adapter_model.safetensors`: only `requires_grad=True` LoRA delta
  params, gathered to CPU. Float32 or bfloat16 is fine; PEFT casts on
  load. The PyTorch path explicitly gathers only LoRA params (~8.4 M,
  ~16 MB on disk) into the state dict; the exporter must match this
  scope or `peft.PeftModel.from_pretrained` will fail with key-mismatch
  errors.
- Tokenizer files: `tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, plus `tokenizer.model` if the base uses
  SentencePiece. `_xla_lora_inference.py:172` calls
  `AutoTokenizer.from_pretrained(self.lora_path)`, so the tokenizer
  used during MaxText training must be saved alongside the adapter.

How Stage 4 reads it (read-only verified):

- `scripts/eval_healthbench.py` (line 183-225): accepts
  `--lora-path` and forwards it to `VLLMEngine(lora_path=...)`.
- `scripts/_xla_lora_inference.py` (line 161): calls
  `PeftModel.from_pretrained(base, self.lora_path)`, then
  `merge_and_unload()`, then `save_pretrained(merged_dir)` for vLLM
  to pick up. The adapter directory is a **plain HuggingFace PEFT
  adapter directory**; Stage 4 makes zero assumptions about which
  trainer wrote it.

If the Unit 6 exporter produces a directory that
`PeftModel.from_pretrained` can load against the MedGemma base, Stage 4
round-trips with no changes.

## Acceptance criterion

The point of the migration is this: under the JAX path, the first
`mark_step` (or the JAX equivalent of "compile + first step") must
land in **under 30 minutes**, vs the PyTorch path where it hangs for
over an hour and never completes. If Phase 2's smoke seed clears the
first step in well under 30 minutes, the migration paid for itself.
If it doesn't, the migration didn't fix the underlying problem and
we should keep the PyTorch fallback as the primary path.
