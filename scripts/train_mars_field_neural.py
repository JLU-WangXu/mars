from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marsstack.field_network.neural_dataset import load_neural_corpus
from marsstack.field_network.neural_training import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=str, default="")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "neural_field_training")
    args = parser.parse_args()

    target_list = [item for item in args.targets.split(",") if item]
    corpus = load_neural_corpus(ROOT / "outputs", include_targets=target_list or None)
    if not corpus:
        raise SystemExit("No neural corpus targets found.")

    model, history = train_model(
        train_batches=corpus,
        epochs=int(args.epochs),
        lr=float(args.lr),
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.outdir / "mars_field_neural.pt")
    (args.outdir / "model_config.json").write_text(json.dumps(model.export_config(), indent=2), encoding="utf-8")
    (args.outdir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved neural model to {args.outdir / 'mars_field_neural.pt'}")


if __name__ == "__main__":
    main()
