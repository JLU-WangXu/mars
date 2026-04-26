from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marsstack.field_network.neural_dataset import NeuralTargetBatch, load_neural_corpus
from marsstack.field_network.neural_training import score_batch, train_model


def resolve_target(corpus: list[NeuralTargetBatch], target_name: str) -> NeuralTargetBatch:
    target_name_l = target_name.lower()
    for batch in corpus:
        if batch.target.lower() == target_name_l:
            return batch
        if batch.pipeline_dir.name.lower() == target_name_l:
            return batch
        if batch.pipeline_dir.name.replace("_pipeline", "").lower() == target_name_l:
            return batch
    raise SystemExit(f"Target not found in neural corpus: {target_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Target name or pipeline directory stem, e.g. 1LBT or tem1_1btl")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    corpus = load_neural_corpus(ROOT / "outputs")
    if not corpus:
        raise SystemExit("No neural corpus targets found.")

    holdout = resolve_target(corpus, args.target)
    train_batches = [batch for batch in corpus if batch.pipeline_dir != holdout.pipeline_dir]
    if not train_batches:
        raise SystemExit("Need at least one non-holdout target to train the neural reranker.")

    outdir = args.outdir or (holdout.pipeline_dir / "neural_field_rerank")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"holdout={holdout.target}")
    print(f"train_batches={len(train_batches)}")
    model, history = train_model(train_batches=train_batches, epochs=int(args.epochs), lr=float(args.lr))
    outputs = score_batch(model, holdout)
    energies = outputs["energies"]
    selection_pred = outputs["selection_pred"]
    engineering_pred = outputs["engineering_pred"]
    policy_pred = outputs["policy_pred"]
    gates = outputs["gates"]
    energy_z = (energies - energies.mean()) / (energies.std() + 1e-6)
    selection_z = (selection_pred - selection_pred.mean()) / (selection_pred.std() + 1e-6)
    engineering_z = (engineering_pred - engineering_pred.mean()) / (engineering_pred.std() + 1e-6)
    policy_z = (policy_pred - policy_pred.mean()) / (policy_pred.std() + 1e-6)

    ranked_path = holdout.pipeline_dir / "combined_ranked_candidates.csv"
    ranked_df = pd.read_csv(ranked_path).copy()
    selector_prior = ranked_df.get("selection_score", ranked_df.get("ranking_score", ranked_df.get("mars_score", pd.Series(np.zeros(len(ranked_df)))))).astype(float).to_numpy()
    engineering_prior = ranked_df.get("engineering_score", ranked_df.get("mars_score", pd.Series(np.zeros(len(ranked_df))))).astype(float).to_numpy()
    selector_prior_z = (selector_prior - selector_prior.mean()) / (selector_prior.std() + 1e-6)
    engineering_prior_z = (engineering_prior - engineering_prior.mean()) / (engineering_prior.std() + 1e-6)
    ranked_df["neural_energy"] = energies
    ranked_df["neural_energy_z"] = energy_z
    ranked_df["neural_selection_pred"] = selection_pred
    ranked_df["neural_selection_z"] = selection_z
    ranked_df["neural_engineering_pred"] = engineering_pred
    ranked_df["neural_engineering_z"] = engineering_z
    ranked_df["neural_policy_pred"] = policy_pred
    ranked_df["neural_policy_z"] = policy_z
    ranked_df["selector_prior_z"] = selector_prior_z
    ranked_df["engineering_prior_z"] = engineering_prior_z
    ranked_df["neural_policy_score"] = (
        0.45 * ranked_df["neural_policy_z"]
        + 0.15 * ranked_df["neural_engineering_z"]
        + 0.10 * ranked_df["neural_selection_z"]
        + 0.15 * ranked_df["selector_prior_z"]
        + 0.15 * ranked_df["engineering_prior_z"]
    )
    ranked_df["neural_rank"] = ranked_df["neural_policy_score"].rank(method="first", ascending=False).astype(int)
    ranked_df = ranked_df.sort_values(["neural_policy_score", "ranking_score"], ascending=[False, False]).reset_index(drop=True)
    ranked_df.to_csv(outdir / "neural_reranked_candidates.csv", index=False)

    top_row = ranked_df.iloc[0]
    gate_means = gates.mean(dim=0).numpy().tolist()
    gate_by_position = {
        int(pos): {
            "geom": float(gates[idx, 0]),
            "phylo": float(gates[idx, 1]),
            "asr": float(gates[idx, 2]),
            "retrieval": float(gates[idx, 3]),
            "environment": float(gates[idx, 4]),
        }
        for idx, pos in enumerate(holdout.positions)
    }
    summary = {
        "target": holdout.target,
        "pipeline_dir": str(holdout.pipeline_dir),
        "training_targets": [batch.target for batch in train_batches],
        "training_target_count": len(train_batches),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "top_neural_candidate_id": str(top_row["candidate_id"]),
        "top_neural_source": str(top_row["source"]),
        "top_neural_mutations": str(top_row["mutations"]),
        "top_neural_energy": float(top_row["neural_energy"]),
        "top_neural_selection_pred": float(top_row["neural_selection_pred"]),
        "top_neural_engineering_pred": float(top_row["neural_engineering_pred"]),
        "top_neural_policy_pred": float(top_row["neural_policy_pred"]),
        "top_neural_policy_z": float(top_row["neural_policy_z"]),
        "top_neural_policy_score": float(top_row["neural_policy_score"]),
        "top_neural_mars_score": float(top_row.get("mars_score", 0.0)),
        "model_config": model.export_config(),
        "candidate_feature_names": holdout.candidate_feature_names,
        "policy_score_formula": {
            "policy_z": 0.45,
            "engineering_z": 0.15,
            "selection_z": 0.10,
            "selector_prior_z": 0.15,
            "engineering_prior_z": 0.15,
        },
        "gate_means": {
            "geom": float(gate_means[0]),
            "phylo": float(gate_means[1]),
            "asr": float(gate_means[2]),
            "retrieval": float(gate_means[3]),
            "environment": float(gate_means[4]),
        },
        "history": history,
    }
    (outdir / "neural_rerank_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), outdir / "mars_field_neural_reranker.pt")
    (outdir / "neural_site_gates.json").write_text(json.dumps(gate_by_position, indent=2), encoding="utf-8")

    report_lines = [
        f"# Neural reranker summary for {holdout.target}",
        "",
        f"- training target count: {len(train_batches)}",
        f"- epochs: {int(args.epochs)}",
        f"- learning rate: {float(args.lr)}",
        f"- top neural candidate: `{top_row['mutations']}` from {top_row['source']}",
        f"- top neural energy: {float(top_row['neural_energy']):.6f}",
        f"- top neural selection prediction: {float(top_row['neural_selection_pred']):.6f}",
        f"- top neural engineering prediction: {float(top_row['neural_engineering_pred']):.6f}",
        f"- top neural policy prediction: {float(top_row['neural_policy_pred']):.6f}",
        f"- top neural policy z: {float(top_row['neural_policy_z']):.6f}",
        f"- top neural policy score: {float(top_row['neural_policy_score']):.6f}",
        f"- top neural candidate MARS score: {float(top_row.get('mars_score', 0.0)):.3f}",
        f"- mean gate weights: geom={gate_means[0]:.3f}, phylo={gate_means[1]:.3f}, asr={gate_means[2]:.3f}, retrieval={gate_means[3]:.3f}, env={gate_means[4]:.3f}",
        "",
        "## Top 5 neural candidates",
        "",
    ]
    for _, row in ranked_df.head(5).iterrows():
        report_lines.append(
            f"- `{row['mutations']}` | source={row['source']} | policy={float(row['neural_policy_score']):.4f} | neural={float(row['neural_energy']):.4f} | mars={float(row.get('mars_score', 0.0)):.3f}"
        )
    (outdir / "neural_rerank_summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved neural rerank outputs to {outdir}")


if __name__ == "__main__":
    main()
