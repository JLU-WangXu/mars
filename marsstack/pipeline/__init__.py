"""MARS-FIELD pipeline helpers, used by ``scripts/run_mars_pipeline.py``.

Submodules group the script's stateless utilities so the entry script stays
focused on orchestration:

* :mod:`marsstack.pipeline.io`         — path resolution
* :mod:`marsstack.pipeline.pdb`        — PDB preprocessing and parsing
* :mod:`marsstack.pipeline.mpnn`       — ProteinMPNN integration glue
* :mod:`marsstack.pipeline.aggregation`— alignment + recommendation merging
* :mod:`marsstack.pipeline.decoding`   — decoded-candidate materialization
"""

from .aggregation import merge_recommendation_maps, summarize_aligned_entries
from .decoding import materialize_decoded_candidate_rows
from .io import resolve_project_path
from .mpnn import (
    ALPHABET,
    build_bias_and_omit,
    build_parsed_index_maps,
    collapse_mpnn_sequence,
    project_to_design_positions,
    restore_template_mismatches,
)
from .pdb import (
    load_parsed_chain_sequence,
    normalize_parsed_names,
    preprocess_pdb,
)

__all__ = [
    "ALPHABET",
    "build_bias_and_omit",
    "build_parsed_index_maps",
    "collapse_mpnn_sequence",
    "load_parsed_chain_sequence",
    "materialize_decoded_candidate_rows",
    "merge_recommendation_maps",
    "normalize_parsed_names",
    "preprocess_pdb",
    "project_to_design_positions",
    "resolve_project_path",
    "restore_template_mismatches",
    "summarize_aligned_entries",
]
