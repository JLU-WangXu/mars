from __future__ import annotations

import json
from pathlib import Path


def preprocess_pdb(
    src_path: Path,
    dst_path: Path,
    residue_renames: list[dict[str, object]] | None = None,
) -> Path:
    """Copy ``src_path`` to ``dst_path``, optionally rewriting residue names.

    When ``residue_renames`` is provided, each ATOM/HETATM line whose chain and
    residue number match an entry has its three-letter residue name replaced
    with ``to_name``. The line is also normalized to start with ``ATOM  `` even
    when the original was ``HETATM``, to keep downstream parsers happy.
    """

    if not residue_renames:
        dst_path.write_bytes(src_path.read_bytes())
        return dst_path

    rename_map: dict[tuple[str, int, str | None], str] = {}
    for item in residue_renames:
        chain = str(item["chain"])
        residue_number = int(item["residue_number"])
        from_name = str(item["from_name"]).upper() if item.get("from_name") else None
        to_name = str(item["to_name"]).upper()
        rename_map[(chain, residue_number, from_name)] = to_name

    out_lines: list[str] = []
    for line in src_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            chain = line[21].strip()
            try:
                residue_number = int(line[22:26].strip())
            except ValueError:
                residue_number = None
            resname = line[17:20].strip().upper()
            replacement = None
            if residue_number is not None:
                replacement = rename_map.get((chain, residue_number, resname))
                if replacement is None:
                    replacement = rename_map.get((chain, residue_number, None))
            if replacement is not None:
                line = "ATOM  " + line[6:]
                line = line[:17] + f"{replacement:>3}" + line[20:]
        out_lines.append(line)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return dst_path


def normalize_parsed_names(parsed_jsonl: Path) -> None:
    """Strip directory + extension from ``name`` fields written by ProteinMPNN's parser."""
    lines = []
    with parsed_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            obj["name"] = Path(obj["name"]).stem
            lines.append(obj)
    with parsed_jsonl.open("w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")


def load_parsed_chain_sequence(parsed_jsonl: Path, chain: str) -> str:
    """Read the first entry of ``parsed_jsonl`` and return ``seq_chain_<chain>``."""
    with parsed_jsonl.open("r", encoding="utf-8") as fh:
        first = json.loads(next(fh))
    return first[f"seq_chain_{chain}"]
