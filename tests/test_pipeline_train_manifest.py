"""Tests for the manifest that scripts/train_lora.py writes. No training runs."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")
yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_train_manifest_records_config_batch_adapter_and_losses(tmp_path, monkeypatch):
    from scripts import train_lora

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    cfg_path = REPO / "rebuttal" / "configs" / "lora_mistral_small_24b_qlora.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["model"]["revision"] = "rev123"
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    train.write_text('{"messages": [], "response": "a"}\n' * 3)
    val.write_text('{"messages": [], "response": "b"}\n')
    best = tmp_path / "ckpt" / "best"
    best.mkdir(parents=True)
    (best / "adapter_config.json").write_text(json.dumps(
        {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj"]}))
    (best / "adapter_model.safetensors").write_bytes(b"w")
    state = {"global_step": 525, "epoch": 3.0, "max_steps": 525, "num_train_epochs": 3,
             "log_history": [{"loss": 0.9, "step": 10}, {"loss": 0.5, "step": 520},
                             {"eval_loss": 0.6, "step": 175}]}

    m = train_lora.build_train_manifest(cfg, cfg_path, 7, train, val,
                                        tmp_path / "ckpt", best, state)
    assert m["stage"] == "train"
    assert m["counts"]["effective_batch"] == 16          # 2 x 8 x 1 process
    assert m["counts"]["global_step"] == 525 and m["counts"]["seed"] == 7
    assert (m["counts"]["first_train_loss"], m["counts"]["last_train_loss"]) == (0.9, 0.5)
    assert m["counts"]["eval_loss_by_step"] == [[175, 0.6]]
    assert m["adapter"]["r"] == 16 and len(m["adapter"]["sha256"]) == 64
    assert m["models"]["base"]["revision"] == "rev123"
    assert m["quantization"] == "4bit"
    assert [i["n_lines"] for i in m["inputs"]] == [3, 1]
    assert m["config_file"]["sha256"]


def test_effective_batch_counts_processes():
    from scripts import train_lora
    cfg = {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 8}
    assert train_lora.effective_batch(cfg, world_size=1) == 16
    assert train_lora.effective_batch(cfg, world_size=4) == 64
