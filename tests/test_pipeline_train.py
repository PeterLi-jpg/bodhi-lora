"""The train stage (bodhi_lora/pipeline/train.py): recipe, manifest and template helpers.
No training runs here; the smoke cell trains for real (tests/test_pipeline_runner.py)."""

import json
from dataclasses import replace

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
pytest.importorskip("pydantic")

from bodhi_lora import manifest as mf  # noqa: E402
from bodhi_lora.paths import CONFIGS_DIR, project  # noqa: E402
from bodhi_lora.pipeline.context import StageContext  # noqa: E402
from bodhi_lora.registry import load_registry  # noqa: E402


def _ctx(tmp_path, cell_id):
    reg = load_registry(CONFIGS_DIR)
    proj = replace(project(CONFIGS_DIR), results_root=tmp_path)
    return StageContext(reg=reg, project=proj, cell=reg.cells[cell_id], paths=proj.cell(cell_id))


def test_recipe_comes_from_the_cell_and_the_model_record(tmp_path):
    from bodhi_lora.pipeline import train as train_stage

    cfg = train_stage.recipe(_ctx(tmp_path, "healthbench__mistral-small-24b__bodhi__2026-07"))
    assert cfg["model"]["quantization"] == "4bit"
    assert cfg["model"]["response_template"] == "[/INST]"
    assert cfg["model"]["tokenizer_kwargs"] == {"fix_mistral_regex": True}
    assert cfg["model"]["pad_token"] == "<pad>"
    assert cfg["lora"]["r"] == 16 and len(cfg["lora"]["target_modules"]) == 7
    assert train_stage.effective_batch(cfg["training"], world_size=1) == 16
    # The validation loss runs at the training batch (R7's out-of-memory at batch 8).
    assert cfg["training"]["per_device_eval_batch_size"] == cfg["training"]["per_device_train_batch_size"] == 2


def test_train_manifest_records_recipe_batch_adapter_and_losses(tmp_path, monkeypatch):
    from bodhi_lora.pipeline import train as train_stage

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    ctx = _ctx(tmp_path, "healthbench__mistral-small-24b__bodhi__2026-07")
    cfg = train_stage.recipe(ctx)
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text('{"messages": [], "response": "a"}\n' * 3)
    val.write_text('{"messages": [], "response": "b"}\n')
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"]}))
    (adapter / "adapter_model.safetensors").write_bytes(b"w")
    state = {"global_step": 525, "epoch": 3.0, "max_steps": 525, "num_train_epochs": 3,
             "log_history": [{"loss": 0.9, "step": 10}, {"loss": 0.5, "step": 520},
                             {"eval_loss": 0.6, "step": 175}]}
    m = train_stage.build_train_manifest(ctx, 7, cfg, train, val, adapter, state, mf.Timer())
    assert (m["stage"], m["status"], m["seed"]) == ("train", "complete", 7)
    assert m["counts"]["effective_batch"] == 16          # 2 x 8 x 1 process
    assert m["counts"]["global_step"] == 525
    assert (m["counts"]["first_train_loss"], m["counts"]["last_train_loss"]) == (0.9, 0.5)
    assert m["counts"]["eval_loss_by_step"] == [[175, 0.6]]
    assert m["adapter"]["r"] == 16 and len(m["adapter"]["sha256"]) == 64
    assert m["models"]["base"]["revision"] == "9527884be6e5616bdd54de542f9ae13384489724"
    assert m["quantization"] == "4bit"
    assert [i["n_lines"] for i in m["inputs"]] == [3, 1]


def test_effective_batch_counts_processes():
    from bodhi_lora.pipeline import train as train_lora
    cfg = {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 8}
    assert train_lora.effective_batch(cfg, world_size=1) == 16
    assert train_lora.effective_batch(cfg, world_size=4) == 64


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = " | ".join(f"{msg['role']}:{msg['content']}" for msg in messages)
        if add_generation_prompt:
            rendered += " | assistant:"
        return rendered


def test_format_example_single_example(monkeypatch):
    from bodhi_lora.pipeline import train as train_lora
    monkeypatch.setattr(train_lora, "_tokenizer", FakeTokenizer())

    result = train_lora.format_example(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "response": "hello",
        }
    )

    assert result == "user:hi | assistant:hello"


def test_format_example_batched_examples(monkeypatch):
    from bodhi_lora.pipeline import train as train_lora
    monkeypatch.setattr(train_lora, "_tokenizer", FakeTokenizer())

    result = train_lora.format_example(
        {
            "messages": [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ],
            "response": ["one", "two"],
        }
    )

    assert result == [
        "user:first | assistant:one",
        "user:second | assistant:two",
    ]


def _tiny_tokenizer():
    """The smoke model's tokenizer from the local cache (the smoke run downloads it)."""
    from transformers import AutoTokenizer
    reg = load_registry(CONFIGS_DIR)
    m = reg.models["tiny-random-llama"]
    try:
        return AutoTokenizer.from_pretrained(m.hf_id, revision=m.revision, local_files_only=True)
    except OSError:
        pytest.skip("the tiny-random-llama tokenizer is not in the local cache")


def _collated_rows(tok):
    from trl import DataCollatorForCompletionOnlyLM
    tok.padding_side = "right"
    texts = [tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                                     tokenize=False)
             for q, a in (("hi", "Hello there."), ("Is it bad?", "Tell me more about the pain."))]
    enc = [tok(t, add_special_tokens=False) for t in texts]
    dataset = [{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in enc]
    return dataset, DataCollatorForCompletionOnlyLM(response_template="[/INST]", tokenizer=tok)


def test_pad_equal_to_eos_takes_the_end_of_the_reply_out_of_the_loss():
    """The bug of every Mistral and Llama adapter before 2026-10-17: with pad = EOS the
    collator masks the EOS that ends the reply. The check stops such a run before a step."""
    from bodhi_lora.pipeline import train as train_stage
    tok = _tiny_tokenizer()
    tok.pad_token = tok.eos_token
    dataset, collator = _collated_rows(tok)
    labels = collator(dataset)["labels"][0][:len(dataset[0]["input_ids"])].tolist()
    eos_at = dataset[0]["input_ids"].index(tok.eos_token_id)
    assert labels[eos_at] == -100
    with pytest.raises(SystemExit, match="out of the loss"):
        train_stage.check_replies_in_loss(dataset, collator, tok)


def test_with_a_pad_token_that_is_not_eos_the_whole_reply_is_trained():
    from bodhi_lora.pipeline import train as train_stage
    tok = _tiny_tokenizer()
    train_stage.set_pad_token(tok, "<unk>")
    dataset, collator = _collated_rows(tok)
    assert train_stage.check_replies_in_loss(dataset, collator, tok) == 2
    labels = collator(dataset)["labels"][0][:len(dataset[0]["input_ids"])].tolist()
    assert labels[dataset[0]["input_ids"].index(tok.eos_token_id)] == tok.eos_token_id


def test_set_pad_token_refuses_eos_and_unknown_tokens():
    from bodhi_lora.pipeline import train as train_stage
    tok = _tiny_tokenizer()
    tok.pad_token = tok.eos_token
    with pytest.raises(SystemExit, match="pads with its EOS"):
        train_stage.set_pad_token(tok, None)
    with pytest.raises(SystemExit, match="not in the vocabulary"):
        train_stage.set_pad_token(tok, "<no-such-token>")




def test_a_half_written_checkpoint_is_not_resumed(tmp_path):
    """Trainer writes trainer_state.json last; a folder without it is one a killed run left."""
    from bodhi_lora.pipeline import train as train_stage
    for step, complete in ((118, True), (236, False)):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if complete:
            (d / "trainer_state.json").write_text("{}")
    assert train_stage.latest_checkpoint(tmp_path) == str(tmp_path / "checkpoint-118")
    assert train_stage.latest_checkpoint(tmp_path / "none") is None
