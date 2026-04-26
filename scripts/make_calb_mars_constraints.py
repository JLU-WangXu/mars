from __future__ import annotations

import argparse
import json
from pathlib import Path


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def zero_bias(length: int) -> list[list[float]]:
    return [[0.0 for _ in ALPHABET] for _ in range(length)]


def set_bias(row: list[float], aa: str, value: float) -> None:
    row[ALPHABET.index(aa)] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="1LBT")
    parser.add_argument("--chain", default="A")
    parser.add_argument("--length", type=int, default=317)
    parser.add_argument(
        "--bias-out",
        type=Path,
        default=Path(r"D:\Codex\Work\4-12 Mars protein\mars_stack\outputs\calb_mpnn_seed\calb_bias_by_res.jsonl"),
    )
    parser.add_argument(
        "--omit-out",
        type=Path,
        default=Path(r"D:\Codex\Work\4-12 Mars protein\mars_stack\outputs\calb_mpnn_seed\calb_omit_aa.jsonl"),
    )
    args = parser.parse_args()

    bias = zero_bias(args.length)

    # Position 249: allow modest exploration but avoid obvious oxidation/stickiness.
    for aa, value in {"L": 0.8, "Q": 0.6, "N": 0.4, "V": 0.5, "E": 0.2, "D": 0.2}.items():
        set_bias(bias[249 - 1], aa, value)
    for aa, value in {"W": -1.5, "Y": -1.2, "C": -2.0, "M": -2.0, "R": -0.3, "K": -0.3}.items():
        set_bias(bias[249 - 1], aa, value)

    # Position 251: favor hydration-friendly or acidic surface substitutions.
    for aa, value in {"E": 1.6, "D": 1.3, "Q": 0.8, "N": 0.7, "S": 0.6, "T": 0.6}.items():
        set_bias(bias[251 - 1], aa, value)
    for aa, value in {"W": -1.4, "Y": -1.2, "C": -1.8, "M": -1.5, "F": -0.8}.items():
        set_bias(bias[251 - 1], aa, value)

    # Position 298: the main antiradiation hotspot.
    for aa, value in {"L": 2.4, "I": 2.0, "V": 1.2}.items():
        set_bias(bias[298 - 1], aa, value)
    for aa, value in {"M": -2.5, "R": -2.5, "K": -2.2, "H": -1.8, "C": -2.2, "W": -2.0, "Y": -1.8, "P": -1.2}.items():
        set_bias(bias[298 - 1], aa, value)

    bias_obj = {args.name: {args.chain: bias}}

    # Hard mask at position 298: only allow L/I/V to test the CALB antiradiation hypothesis cleanly.
    omit_letters = "".join([aa for aa in ALPHABET[:-1] if aa not in {"L", "I", "V"}])
    omit_obj = {
        args.name: {
            args.chain: [
                [[298], omit_letters],
            ]
        }
    }

    args.bias_out.parent.mkdir(parents=True, exist_ok=True)
    args.omit_out.parent.mkdir(parents=True, exist_ok=True)
    args.bias_out.write_text(json.dumps(bias_obj) + "\n", encoding="utf-8")
    args.omit_out.write_text(json.dumps(omit_obj) + "\n", encoding="utf-8")

    print(f"Wrote bias file to {args.bias_out}")
    print(f"Wrote omit-AA file to {args.omit_out}")


if __name__ == "__main__":
    main()
