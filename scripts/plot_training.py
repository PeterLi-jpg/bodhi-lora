"""Plot training and evaluation loss curves from a trainer_state.json file."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path):
    with open(path) as handle:
        payload = json.load(handle)
    return payload.get("log_history", [])


def series_from_history(history, key):
    xs = []
    ys = []
    for entry in history:
        if key not in entry:
            continue
        step = entry.get("step")
        if step is None:
            continue
        xs.append(step)
        ys.append(entry[key])
    return xs, ys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainer-state",
        default="checkpoints/best/trainer_state.json",
        help="path to trainer_state.json",
    )
    parser.add_argument(
        "--output",
        default="eval/figures/training_loss.png",
        help="where to write the PNG",
    )
    args = parser.parse_args()

    history = load_history(args.trainer_state)
    train_steps, train_loss = series_from_history(history, "loss")
    eval_steps, eval_loss = series_from_history(history, "eval_loss")

    if not train_loss and not eval_loss:
        raise ValueError(f"no loss metrics found in {args.trainer_state}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    if train_loss:
        plt.plot(train_steps, train_loss, label="train_loss", linewidth=2)
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="eval_loss", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Dynamics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
