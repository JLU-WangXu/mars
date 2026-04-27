"""High-performance serialization layer for Mars.

This module provides a unified serialization interface that can use multiple backends:
- JSON: Human-readable, compatible, for metadata and small objects
- MessagePack: Binary, ~30% smaller, ~2-5x faster, for numeric-heavy data
- NumPy binary: Fastest, for large arrays

Usage:
    from marsstack.serialization import Serializer, SerialFormat

    # Auto-detect format from file extension
    serializer = Serializer()

    # Save with specified format
    serializer.dump(data, path, format=SerialFormat.MSGPACK)

    # Load with auto-detection
    data = serializer.load(path)

    # Check file format
    fmt = serializer.detect_format(path)
"""

from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Union

import numpy as np

# Optional: msgpack for binary serialization
try:
    import msgpack

    _MSGPACK_AVAILABLE = True
except ImportError:
    _MSGPACK_AVAILABLE = False


T = TypeVar("T")


class SerialFormat(Enum):
    """Supported serialization formats."""

    JSON = "json"
    MSGPACK = "msgpack"
    NUMPY = "numpy"
    JSONL = "jsonl"
    AUTO = "auto"


# File extensions for each format
_FORMAT_EXTENSIONS = {
    SerialFormat.JSON: [".json"],
    SerialFormat.MSGPACK: [".msgpack", ".mp"],
    SerialFormat.NUMPY: [".npy", ".npz"],
    SerialFormat.JSONL: [".jsonl"],
}


@dataclass
class SerialStats:
    """Statistics about serialization operations."""

    format: SerialFormat
    original_size: int = 0
    serialized_size: int = 0
    load_time_ms: float = 0.0
    dump_time_ms: float = 0.0

    @property
    def compression_ratio(self) -> float:
        if self.original_size == 0:
            return 1.0
        return self.serialized_size / self.original_size


class SerializationError(Exception):
    """Raised when serialization/deserialization fails."""

    pass


def _is_msgpack_available() -> bool:
    return _MSGPACK_AVAILABLE


def _guess_format_from_path(path: Path) -> SerialFormat:
    """Guess format from file extension."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return SerialFormat.JSON
    elif suffix in (".msgpack", ".mp"):
        return SerialFormat.MSGPACK
    elif suffix in (".npy", ".npz"):
        return SerialFormat.NUMPY
    elif suffix == ".jsonl":
        return SerialFormat.JSONL
    return SerialFormat.JSON  # Default


def _serialize_msgpack_fallback(obj: Any) -> Any:
    """Convert objects to msgpack-serializable types."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist(), "__dtype__": str(obj.dtype)}
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    elif isinstance(obj, set):
        return {"__set__": list(obj)}
    elif isinstance(obj, tuple):
        return {"__tuple__": list(obj)}
    elif isinstance(obj, (pathlib_Path := __import__("pathlib").Path)):
        return {"__path__": str(obj)}
    elif hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    elif isinstance(obj, dict):
        return {k: _serialize_msgpack_fallback(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_msgpack_fallback(i) for i in obj]
    return obj


def _deserialize_msgpack_fallback(obj: dict) -> Any:
    """Reconstruct objects from msgpack-serialized dicts."""
    if "__ndarray__" in obj:
        return np.array(obj["__ndarray__"], dtype=obj.get("__dtype__", "float64"))
    elif "__set__" in obj:
        return set(obj["__set__"])
    elif "__tuple__" in obj:
        return tuple(obj["__tuple__"])
    elif "__path__" in obj:
        return Path(obj["__path__"])
    return obj


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert dataclass to dict, handling nested dataclasses."""
    if not hasattr(obj, "__dataclass_fields__"):
        return obj

    result = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        if value is None:
            continue
        if hasattr(value, "__dataclass_fields__"):
            result[f.name] = _dataclass_to_dict(value)
        elif isinstance(value, list) and value and hasattr(value[0] if value else None, "__dataclass_fields__"):
            result[f.name] = [_dataclass_to_dict(v) for v in value]
        elif isinstance(value, np.ndarray):
            result[f.name] = value.tolist()
        elif isinstance(value, (set, frozenset)):
            result[f.name] = list(value)
        elif isinstance(value, tuple):
            result[f.name] = list(value)
        else:
            result[f.name] = value
    return result


def _ndarray_to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_ndarray_to_list(i) for i in obj]
    elif isinstance(obj, tuple):
        return [_ndarray_to_list(i) for i in obj]
    return obj


class Serializer:
    """High-performance serialization with multiple backend support.

    The serializer automatically handles:
    - NumPy arrays (converted to/from lists for JSON)
    - Dataclasses (converted to/from dicts)
    - Sets and tuples (converted to/from lists)

    Example:
        >>> from marsstack.serialization import Serializer, SerialFormat
        >>> import numpy as np
        >>> data = {"features": np.array([1.0, 2.0, 3.0]), "label": "test"}
        >>> path = Path("data.msgpack")
        >>> Serializer().dump(data, path)
        >>> loaded = Serializer().load(path)
    """

    def __init__(self, default_format: SerialFormat = SerialFormat.JSON):
        """Initialize serializer.

        Args:
            default_format: Default format to use when not specified and cannot auto-detect.
        """
        self.default_format = default_format

    def detect_format(self, path: Path) -> SerialFormat:
        """Detect serialization format from file header or extension.

        Args:
            path: Path to the file.

        Returns:
            Detected format or default.
        """
        if not path.exists():
            return self.default_format

        # Try to detect from magic bytes
        try:
            with open(path, "rb") as f:
                header = f.read(16)

            # NumPy .npy files start with magic bytes
            if header[:6] == b"\x93NUMPY":
                return SerialFormat.NUMPY

            # MessagePack has no standard magic bytes but msgpack_ext uses 0xD7 or 0xD8
            # Try msgpack unpacking on small binary files
            if _MSGPACK_AVAILABLE and not path.suffix.lower() == ".json":
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                    msgpack.unpackb(data, raw=False)
                    return SerialFormat.MSGPACK
                except Exception:
                    pass

        except Exception:
            pass

        # Fall back to extension-based detection
        return _guess_format_from_path(path)

    def dump(
        self,
        obj: Any,
        path: Path,
        format: SerialFormat | None = None,
        *,
        indent: int | None = 2,
        **kwargs: Any,
    ) -> SerialStats:
        """Serialize object to file.

        Args:
            obj: Object to serialize (dataclass, dict, list, numpy array, etc.)
            path: Output file path.
            format: Serialization format (auto-detected if None).
            indent: JSON indentation (None for compact output).
            **kwargs: Additional format-specific arguments.

        Returns:
            SerialStats with performance metrics.
        """
        import time

        if format is None:
            format = _guess_format_from_path(path)

        start = time.perf_counter()

        # Convert dataclasses and numpy arrays
        if hasattr(obj, "__dataclass_fields__"):
            obj = _dataclass_to_dict(obj)
        obj = _ndarray_to_list(obj)

        if format == SerialFormat.JSON:
            self._dump_json(obj, path, indent=indent, **kwargs)
        elif format == SerialFormat.MSGPACK:
            self._dump_msgpack(obj, path, **kwargs)
        elif format == SerialFormat.NUMPY:
            self._dump_numpy(obj, path, **kwargs)
        elif format == SerialFormat.JSONL:
            self._dump_jsonl(obj, path, indent=indent, **kwargs)
        else:
            raise SerializationError(f"Unsupported format: {format}")

        dump_time = (time.perf_counter() - start) * 1000

        # Calculate stats
        serialized_size = path.stat().st_size
        return SerialStats(
            format=format,
            serialized_size=serialized_size,
            dump_time_ms=dump_time,
        )

    def load(
        self,
        path: Path,
        format: SerialFormat | None = None,
        cls: type[T] | None = None,
    ) -> Any:
        """Deserialize object from file.

        Args:
            path: Input file path.
            format: Serialization format (auto-detected if None).
            cls: Target dataclass type for reconstruction.

        Returns:
            Deserialized object.
        """
        import time

        if format is None:
            format = self.detect_format(path)

        start = time.perf_counter()

        if format == SerialFormat.JSON:
            data = self._load_json(path)
        elif format == SerialFormat.MSGPACK:
            data = self._load_msgpack(path)
        elif format == SerialFormat.NUMPY:
            data = self._load_numpy(path)
        elif format == SerialFormat.JSONL:
            data = self._load_jsonl(path)
        else:
            raise SerializationError(f"Unsupported format: {format}")

        load_time = (time.perf_counter() - start) * 1000

        # Optionally reconstruct dataclass
        if cls is not None and isinstance(data, dict) and hasattr(cls, "__dataclass_fields__"):
            data = self._dict_to_dataclass(data, cls)

        # Attach load time for profiling
        if hasattr(data, "__dict__"):
            pass  # Cannot easily attach timing

        return data

    def _dump_json(self, obj: Any, path: Path, indent: int | None = 2, **kwargs: Any) -> None:
        """Write JSON to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False, **kwargs)

    def _load_json(self, path: Path) -> Any:
        """Read JSON from file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _dump_msgpack(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Write MessagePack binary to file."""
        if not _MSGPACK_AVAILABLE:
            raise SerializationError("MessagePack not available. Install with: pip install msgpack")
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = _serialize_msgpack_fallback(obj)
        with open(path, "wb") as f:
            msgpack.packb(obj, f, **kwargs)

    def _load_msgpack(self, path: Path) -> Any:
        """Read MessagePack binary from file."""
        if not _MSGPACK_AVAILABLE:
            raise SerializationError("MessagePack not available. Install with: pip install msgpack")
        with open(path, "rb") as f:
            data = msgpack.unpackb(f.read(), raw=False, strict_map_key=False)
        return _deserialize_msgpack_fallback(data)

    def _dump_numpy(self, obj: Any, path: Path, **kwargs: Any) -> None:
        """Write NumPy array(s) to .npy or .npz file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(obj, np.ndarray):
            np.save(path, obj, **kwargs)
        elif isinstance(obj, dict) and all(isinstance(v, np.ndarray) for v in obj.values()):
            np.savez(path, **obj, **kwargs)
        elif isinstance(obj, (list, tuple)) and all(isinstance(x, np.ndarray) for x in obj):
            np.savez(path, **{f"arr_{i}": x for i, x in enumerate(obj)}, **kwargs)
        else:
            raise SerializationError("NumPy format requires ndarray, dict of ndarrays, or list of ndarrays")

    def _load_numpy(self, path: Path) -> Any:
        """Read NumPy .npy or .npz file."""
        suffix = path.suffix.lower()
        if suffix == ".npz":
            return dict(np.load(path, strict=False))
        return np.load(path, strict=False)

    def _dump_jsonl(self, obj: Any, path: Path, indent: int | None = None, **kwargs: Any) -> None:
        """Write JSONL (JSON Lines) to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    f.write(json.dumps(item, indent=indent, ensure_ascii=False, **kwargs) + "\n")
            else:
                f.write(json.dumps(obj, indent=indent, ensure_ascii=False, **kwargs) + "\n")

    def _load_jsonl(self, path: Path) -> list[Any]:
        """Read JSONL (JSON Lines) from file."""
        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results

    def _dict_to_dataclass(self, data: dict[str, Any], cls: type[T]) -> T:
        """Convert dict to dataclass instance."""
        if not hasattr(cls, "__dataclass_fields__"):
            return data

        # Filter to only known fields
        init_fields = {f.name for f in fields(cls) if f.init}
        kwargs = {}
        for key, value in data.items():
            if key in init_fields:
                field_type = next((f.type for f in fields(cls) if f.name == key), None)
                if field_type and hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
                    kwargs[key] = self._dict_to_dataclass(value, field_type)
                elif isinstance(value, list) and value:
                    # Try to convert list items if field type suggests
                    kwargs[key] = value
                else:
                    kwargs[key] = value
        return cls(**kwargs)

    def dumps(
        self,
        obj: Any,
        format: SerialFormat = SerialFormat.JSON,
        **kwargs: Any,
    ) -> str | bytes:
        """Serialize object to string/bytes.

        Args:
            obj: Object to serialize.
            format: Serialization format.
            **kwargs: Additional arguments.

        Returns:
            Serialized string (JSON/JSONL) or bytes (MessagePack).
        """
        if hasattr(obj, "__dataclass_fields__"):
            obj = _dataclass_to_dict(obj)
        obj = _ndarray_to_list(obj)

        if format == SerialFormat.JSON:
            return json.dumps(obj, ensure_ascii=False, **kwargs)
        elif format == SerialFormat.MSGPACK:
            if not _MSGPACK_AVAILABLE:
                raise SerializationError("MessagePack not available")
            obj = _serialize_msgpack_fallback(obj)
            return msgpack.packb(obj, **kwargs)
        elif format == SerialFormat.JSONL:
            if isinstance(obj, (list, tuple)):
                return "\n".join(json.dumps(item, ensure_ascii=False, **kwargs) for item in obj)
            return json.dumps(obj, ensure_ascii=False, **kwargs)
        else:
            raise SerializationError(f"Unsupported format for dumps: {format}")

    def loads(
        self,
        data: str | bytes,
        format: SerialFormat = SerialFormat.JSON,
    ) -> Any:
        """Deserialize object from string/bytes.

        Args:
            data: Serialized data.
            format: Serialization format.

        Returns:
            Deserialized object.
        """
        if format == SerialFormat.JSON:
            return json.loads(data)
        elif format == SerialFormat.MSGPACK:
            if not _MSGPACK_AVAILABLE:
                raise SerializationError("MessagePack not available")
            result = msgpack.unpackb(data, raw=False, strict_map_key=False)
            return _deserialize_msgpack_fallback(result)
        elif format == SerialFormat.JSONL:
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return [json.loads(line) for line in data.strip().split("\n") if line.strip()]
        else:
            raise SerializationError(f"Unsupported format for loads: {format}")

    def bulk_convert(
        self,
        input_paths: list[Path],
        output_format: SerialFormat,
        output_suffix: str | None = None,
    ) -> list[tuple[Path, Path, SerialStats]]:
        """Convert multiple files to a new format.

        Args:
            input_paths: List of input file paths.
            output_format: Target format.
            output_suffix: Suffix to add to output filenames (e.g., ".msgpack").

        Returns:
            List of (input_path, output_path, stats) tuples.
        """
        results = []
        for input_path in input_paths:
            if output_suffix is None:
                suffix_map = {
                    SerialFormat.JSON: ".json",
                    SerialFormat.MSGPACK: ".msgpack",
                    SerialFormat.JSONL: ".jsonl",
                }
                output_suffix = suffix_map.get(output_format, ".bin")

            output_path = input_path.with_suffix(output_suffix)
            if output_path == input_path:
                output_path = input_path.with_name(f"{input_path.stem}{output_suffix}")

            data = self.load(input_path)
            stats = self.dump(data, output_path, format=output_format)
            results.append((input_path, output_path, stats))

        return results


# Convenience functions for common operations
def dump_msgpack(obj: Any, path: Path, **kwargs: Any) -> None:
    """Quick dump to MessagePack format."""
    Serializer().dump(obj, path, format=SerialFormat.MSGPACK, **kwargs)


def load_msgpack(path: Path) -> Any:
    """Quick load from MessagePack format."""
    return Serializer().load(path, format=SerialFormat.MSGPACK)


def dump_json(obj: Any, path: Path, **kwargs: Any) -> None:
    """Quick dump to JSON format."""
    Serializer().dump(obj, path, format=SerialFormat.JSON, **kwargs)


def load_json(path: Path) -> Any:
    """Quick load from JSON format."""
    return Serializer().load(path, format=SerialFormat.JSON)


def save_candidates_jsonl(candidates: list[dict[str, Any]], path: Path) -> None:
    """Save ranked candidates in JSONL format (one JSON object per line).

    This is the standard format used by MPNN and other tools.

    Args:
        candidates: List of candidate dictionaries.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for candidate in candidates:
            f.write(json.dumps(candidate, ensure_ascii=False) + "\n")


def load_candidates_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load candidates from JSONL format.

    Args:
        path: Input file path.

    Returns:
        List of candidate dictionaries.
    """
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


# =============================================================================
# Optimized Data Transfer Objects
# =============================================================================


@dataclass
class CompactFeatureVector:
    """Memory-efficient feature vector container for serialization.

    Uses fixed-size numpy arrays internally for better performance
    compared to nested Python lists.
    """

    position: int
    wt_residue: str
    values: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values, dtype=np.float32)

    def to_list(self) -> list[Any]:
        return {"position": self.position, "wt_residue": self.wt_residue, "values": self.values.tolist()}

    @classmethod
    def from_list(cls, data: dict[str, Any]) -> CompactFeatureVector:
        return cls(
            position=int(data["position"]),
            wt_residue=str(data["wt_residue"]),
            values=np.array(data["values"], dtype=np.float32),
        )


@dataclass
class SerializedPositionFields:
    """Optimized container for position_fields data.

    Stores all residue distributions as a single 2D array instead of
    nested dictionaries for faster batch processing.
    """

    positions: np.ndarray  # Shape: (N,)
    wt_residues: np.ndarray  # Shape: (N,) - amino acid indices
    options_per_position: np.ndarray  # Shape: (N,) - number of options
    residues: np.ndarray  # Flat array of amino acid indices
    scores: np.ndarray  # Flat array of scores, indexed by options

    # Metadata for reconstruction
    num_amino_acids: int = 20

    def to_dict(self) -> dict[str, Any]:
        """Convert back to nested dict format for compatibility."""
        result = {}
        idx = 0
        for i, pos in enumerate(self.positions):
            num_opts = int(self.options_per_position[i])
            pos_residues = self.residues[idx : idx + num_opts].tolist()
            pos_scores = self.scores[idx : idx + num_opts].tolist()
            options = [{"residue": self._idx_to_aa(r), "score": s} for r, s in zip(pos_residues, pos_scores)]
            result[int(pos)] = {"position": int(pos), "wt_residue": self._idx_to_aa(int(self.wt_residues[i])), "options": options}
            idx += num_opts
        return result

    def _idx_to_aa(self, idx: int) -> str:
        return "ACDEFGHIKLMNPQRSTVWY"[idx] if idx < 20 else "X"

    @classmethod
    def from_position_fields(cls, position_fields: list[dict[str, Any]]) -> SerializedPositionFields:
        """Create from standard position_fields format."""
        positions = []
        wt_residues = []
        all_residues = []
        all_scores = []
        options_per_position = []

        AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

        for field in position_fields:
            positions.append(int(field["position"]))
            wt_residues.append(AA_TO_IDX.get(field["wt_residue"], 20))
            num_opts = len(field.get("options", []))
            options_per_position.append(num_opts)
            for opt in field.get("options", []):
                all_residues.append(AA_TO_IDX.get(opt["residue"], 20))
                all_scores.append(float(opt["score"]))

        return cls(
            positions=np.array(positions, dtype=np.int32),
            wt_residues=np.array(wt_residues, dtype=np.int8),
            options_per_position=np.array(options_per_position, dtype=np.int16),
            residues=np.array(all_residues, dtype=np.int8),
            scores=np.array(all_scores, dtype=np.float32),
        )


# =============================================================================
# Performance benchmarks
# =============================================================================


def benchmark_serialization(
    data: dict[str, Any],
    iterations: int = 100,
) -> dict[str, dict[str, float]]:
    """Benchmark different serialization formats on given data.

    Args:
        data: Data to serialize.
        iterations: Number of iterations for timing.

    Returns:
        Dictionary with timing and size metrics for each format.
    """
    import tempfile
    import time

    results = {}
    formats = [SerialFormat.JSON]
    if _MSGPACK_AVAILABLE:
        formats.append(SerialFormat.MSGPACK)

    serializer = Serializer()

    for fmt in formats:
        with tempfile.NamedTemporaryFile(suffix=f".{fmt.value}", delete=False) as f:
            path = Path(f.name)

        try:
            times_dump = []
            times_load = []
            sizes = []

            for _ in range(iterations):
                # Measure dump
                start = time.perf_counter()
                serializer.dump(data, path, format=fmt)
                times_dump.append((time.perf_counter() - start) * 1000)
                sizes.append(path.stat().st_size)

                # Measure load
                start = time.perf_counter()
                _ = serializer.load(path, format=fmt)
                times_load.append((time.perf_counter() - start) * 1000)

            results[fmt.value] = {
                "dump_mean_ms": sum(times_dump) / len(times_dump),
                "load_mean_ms": sum(times_load) / len(times_load),
                "size_bytes": sizes[-1],
                "size_ratio": sizes[-1] / max(sizes[0], 1) if fmt == SerialFormat.MSGPACK else 1.0,
            }
        finally:
            path.unlink(missing_ok=True)

    return results
