"""Experiment tracking and reproducibility support for MARS-FIELD.

Features:
    - Configuration snapshots
    - Run history management
    - Results comparison
    - Parameter search tracking
    - MLflow/Weights & Biases integration interfaces
    - Optional local tracking storage
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class RunStatus(Enum):
    """Status of an experiment run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrackerBackend(Enum):
    """Available tracking backends."""

    LOCAL = "local"
    MLFLOW = "mlflow"
    WANDB = "wandb"


@dataclass
class ConfigSnapshot:
    """Immutable snapshot of experiment configuration."""

    config_id: str
    config_dict: dict[str, Any]
    config_hash: str
    created_at: str
    platform: str
    python_version: str

    @classmethod
    def create(cls, config_dict: dict[str, Any]) -> ConfigSnapshot:
        """Create a new configuration snapshot with hash."""
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
        config_id = f"cfg_{config_hash}_{int(time.time())}"
        return cls(
            config_id=config_id,
            config_dict=config_dict,
            config_hash=config_hash,
            created_at=datetime.now().isoformat(),
            platform=platform.platform(),
            python_version=platform.python_version(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigSnapshot:
        """Restore from dictionary."""
        return cls(**data)


@dataclass
class MetricRecord:
    """Single metric record for a run."""

    name: str
    value: float | int | str
    step: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RunRecord:
    """Record of a single experiment run."""

    run_id: str
    experiment_name: str
    status: RunStatus
    config_snapshot: ConfigSnapshot
    metrics: list[MetricRecord] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        experiment_name: str,
        config_dict: dict[str, Any],
        tags: dict[str, str] | None = None,
    ) -> RunRecord:
        """Create a new run record."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        config_snapshot = ConfigSnapshot.create(config_dict)
        return cls(
            run_id=run_id,
            experiment_name=experiment_name,
            status=RunStatus.RUNNING,
            config_snapshot=config_snapshot,
            tags=tags or {},
        )

    def log_metric(
        self,
        name: str,
        value: float | int | str,
        step: int = 0,
    ) -> None:
        """Log a metric for this run."""
        self.metrics.append(MetricRecord(name=name, value=value, step=step))

    def log_metrics(self, metrics: dict[str, float | int], step: int = 0) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_artifact(self, name: str, path: str) -> None:
        """Log an artifact path."""
        self.artifacts[name] = path

    def set_status(self, status: RunStatus) -> None:
        """Update run status."""
        self.status = status
        if status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            self.completed_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "status": self.status.value,
            "config_snapshot": self.config_snapshot.to_dict(),
            "metrics": [m.to_dict() for m in self.metrics],
            "tags": self.tags,
            "artifacts": self.artifacts,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        """Restore from dictionary."""
        data = dict(data)
        data["status"] = RunStatus(data["status"])
        data["config_snapshot"] = ConfigSnapshot.from_dict(data["config_snapshot"])
        data["metrics"] = [MetricRecord(**m) for m in data.get("metrics", [])]
        return cls(**data)


@dataclass
class SearchTrial:
    """Single trial in a parameter search."""

    trial_id: str
    search_id: str
    params: dict[str, Any]
    metrics: dict[str, float]
    run_id: str | None = None
    status: RunStatus = RunStatus.RUNNING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["status"] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchTrial:
        """Restore from dictionary."""
        data = dict(data)
        data["status"] = RunStatus(data["status"])
        return cls(**data)


class LocalTrackerStore:
    """Local JSON-based storage for experiment tracking."""

    def __init__(self, root_dir: str | Path = "./outputs/tracker"):
        """Initialize local tracker storage."""
        self.root_dir = Path(root_dir)
        self.runs_dir = self.root_dir / "runs"
        self.experiments_dir = self.root_dir / "experiments"
        self.searches_dir = self.root_dir / "searches"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create necessary directories."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.searches_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, run: RunRecord) -> str:
        """Save a run record."""
        path = self.runs_dir / f"{run.run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)
        return str(path)

    def load_run(self, run_id: str) -> RunRecord | None:
        """Load a run record by ID."""
        path = self.runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return RunRecord.from_dict(json.load(f))

    def list_runs(
        self,
        experiment_name: str | None = None,
        status: RunStatus | None = None,
    ) -> list[RunRecord]:
        """List runs, optionally filtered by experiment or status."""
        runs = []
        for path in self.runs_dir.glob("*.json"):
            with open(path, encoding="utf-8") as f:
                run = RunRecord.from_dict(json.load(f))
            if experiment_name and run.experiment_name != experiment_name:
                continue
            if status and run.status != status:
                continue
            runs.append(run)
        return sorted(runs, key=lambda r: r.started_at, reverse=True)

    def save_search(self, search_id: str, trials: list[SearchTrial]) -> str:
        """Save parameter search trials."""
        path = self.searches_dir / f"{search_id}.json"
        data = {
            "search_id": search_id,
            "trials": [t.to_dict() for t in trials],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return str(path)

    def load_search(self, search_id: str) -> list[SearchTrial] | None:
        """Load parameter search trials."""
        path = self.searches_dir / f"{search_id}.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [SearchTrial.from_dict(t) for t in data["trials"]]


class TrackingBackend(Protocol):
    """Protocol for tracking backends."""

    def init(self, experiment_name: str, **kwargs: Any) -> None:
        """Initialize tracking for an experiment."""
        ...

    def log_run(self, run: RunRecord) -> None:
        """Log a run to the backend."""
        ...

    def log_metrics(self, metrics: dict[str, float | int], step: int = 0) -> None:
        """Log metrics."""
        ...

    def log_artifact(self, path: str, name: str | None = None) -> None:
        """Log an artifact."""
        ...

    def finish(self) -> None:
        """Finish the current run."""
        ...


class MLflowTracker:
    """MLflow integration for experiment tracking."""

    def __init__(self):
        """Initialize MLflow tracker."""
        self._client = None
        self._run_id = None
        self._experiment_name = None

    def _is_available(self) -> bool:
        """Check if MLflow is available."""
        try:
            import mlflow
            return True
        except ImportError:
            return False

    def init(self, experiment_name: str, **kwargs: Any) -> None:
        """Initialize MLflow tracking."""
        if not self._is_available():
            raise ImportError("mlflow is not installed. Install with: pip install mlflow")

        import mlflow

        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run()
        self._run_id = self._run.info.run_id
        self._experiment_name = experiment_name

    def log_run(self, run: RunRecord) -> None:
        """Log a complete run to MLflow."""
        if not self._is_available():
            raise ImportError("mlflow is not installed")

        import mlflow

        with mlflow.start_run(run_id=self._run_id, nested=True):
            mlflow.set_tag("experiment_name", run.experiment_name)
            mlflow.set_tag("run_id", run.run_id)
            for key, value in run.tags.items():
                mlflow.set_tag(key, str(value))

            for metric in run.metrics:
                mlflow.log_metric(metric.name, float(metric.value), step=metric.step)

            mlflow.log_params(run.config_snapshot.config_dict)

            for name, path in run.artifacts.items():
                mlflow.log_artifact(path, name)

    def log_metrics(self, metrics: dict[str, float | int], step: int = 0) -> None:
        """Log metrics to MLflow."""
        if not self._is_available():
            raise ImportError("mlflow is not installed")

        import mlflow

        for name, value in metrics.items():
            mlflow.log_metric(name, float(value), step=step)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        """Log an artifact to MLflow."""
        if not self._is_available():
            raise ImportError("mlflow is not installed")

        import mlflow

        mlflow.log_artifact(path, name)

    def finish(self) -> None:
        """Finish MLflow run."""
        if not self._is_available():
            return

        import mlflow

        mlflow.end_run()


class WandbTracker:
    """Weights & Biases integration for experiment tracking."""

    def __init__(self):
        """Initialize W&B tracker."""
        self._run = None

    def _is_available(self) -> bool:
        """Check if wandb is available."""
        try:
            import wandb
            return True
        except ImportError:
            return False

    def init(self, experiment_name: str, **kwargs: Any) -> None:
        """Initialize W&B tracking."""
        if not self._is_available():
            raise ImportError("wandb is not installed. Install with: pip install wandb")

        import wandb

        wandb.init(project=experiment_name, **kwargs)
        self._run = wandb.run

    def log_run(self, run: RunRecord) -> None:
        """Log a complete run to W&B."""
        if not self._is_available():
            raise ImportError("wandb is not installed")

        import wandb

        with wandb.init(project=run.experiment_name, id=run.run_id, resume=True):
            wandb.config.update(run.config_snapshot.config_dict, allow_val_change=True)

            for metric in run.metrics:
                wandb.log({metric.name: metric.value, "step": metric.step})

            for name, path in run.artifacts.items():
                wandb.save(path, base_path=name)

    def log_metrics(self, metrics: dict[str, float | int], step: int = 0) -> None:
        """Log metrics to W&B."""
        if not self._is_available():
            raise ImportError("wandb is not installed")

        import wandb

        wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        """Log an artifact to W&B."""
        if not self._is_available():
            raise ImportError("wandb is not installed")

        import wandb

        artifact = wandb.Artifact(name or Path(path).name, type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Finish W&B run."""
        if not self._is_available():
            return

        import wandb

        wandb.finish()


class ExperimentTracker:
    """Main experiment tracking interface.

    Supports local storage, MLflow, and Weights & Biases backends.

    Example:
        >>> tracker = ExperimentTracker("my-experiment", backend=TrackerBackend.LOCAL)
        >>> tracker.log_params({"learning_rate": 0.001, "batch_size": 32})
        >>> tracker.log_metrics({"loss": 0.5, "accuracy": 0.9})
        >>> tracker.finish()
    """

    def __init__(
        self,
        experiment_name: str,
        backend: TrackerBackend = TrackerBackend.LOCAL,
        local_dir: str | Path | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment.
            backend: Tracking backend to use.
            local_dir: Directory for local storage (if using LOCAL backend).
            run_name: Optional name for this specific run.
            tags: Tags to attach to the run.
            config: Configuration parameters for the experiment.
            **kwargs: Additional arguments passed to the backend.
        """
        self.experiment_name = experiment_name
        self.backend_type = backend

        if backend == TrackerBackend.LOCAL:
            self._backend: TrackingBackend = LocalTrackerStore(
                root_dir=local_dir or "./outputs/tracker"
            )
        elif backend == TrackerBackend.MLFLOW:
            self._backend = MLflowTracker()
            self._backend.init(experiment_name, **kwargs)
        elif backend == TrackerBackend.WANDB:
            self._backend = WandbTracker()
            self._backend.init(experiment_name, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._config = config or {}
        self._current_run = RunRecord.create(
            experiment_name=experiment_name,
            config_dict=self._config,
            tags=tags,
        )
        if run_name:
            self._current_run.tags["run_name"] = run_name

    @property
    def run_id(self) -> str:
        """Get current run ID."""
        return self._current_run.run_id

    @property
    def config_hash(self) -> str:
        """Get configuration hash."""
        return self._current_run.config_snapshot.config_hash

    def log_params(self, params: dict[str, Any]) -> None:
        """Log experiment parameters."""
        self._config.update(params)
        self._current_run.config_snapshot = ConfigSnapshot.create(self._config)

    def log_metrics(self, metrics: dict[str, float | int], step: int = 0) -> None:
        """Log metrics for the current step."""
        self._current_run.log_metrics(metrics, step)
        if self.backend_type != TrackerBackend.LOCAL:
            self._backend.log_metrics(metrics, step)

    def log_metric(self, name: str, value: float | int, step: int = 0) -> None:
        """Log a single metric."""
        self.log_metrics({name: value}, step)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        """Log an artifact path."""
        self._current_run.log_artifact(name or Path(path).name, path)
        if self.backend_type != TrackerBackend.LOCAL:
            self._backend.log_artifact(path, name)

    def set_tags(self, tags: dict[str, str]) -> None:
        """Update run tags."""
        self._current_run.tags.update(tags)

    def set_status(self, status: RunStatus) -> None:
        """Update run status."""
        self._current_run.set_status(status)

    def finish(self, status: RunStatus = RunStatus.COMPLETED) -> None:
        """Finish the experiment run.

        Args:
            status: Final status of the run.
        """
        self._current_run.set_status(status)

        if self.backend_type == TrackerBackend.LOCAL:
            self._backend.save_run(self._current_run)
        else:
            self._backend.log_run(self._current_run)
            self._backend.finish()

    def __enter__(self) -> ExperimentTracker:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if exc_type is not None:
            self.set_status(RunStatus.FAILED)
        else:
            self.set_status(RunStatus.COMPLETED)
        self.finish()


class ParameterSearchTracker:
    """Tracker for hyperparameter search experiments.

    Manages multiple trials and provides methods to compare results.

    Example:
        >>> search = ParameterSearchTracker("lr-search")
        >>> search.define_search_space({
        ...     "lr": [0.001, 0.01, 0.1],
        ...     "batch_size": [16, 32, 64],
        ... })
        >>> for trial in search.trials():
        ...     # run experiment with trial.params
        ...     search.log_trial_result(trial.trial_id, {"val_loss": 0.3})
        >>> best = search.get_best_trial("val_loss", mode="min")
    """

    def __init__(
        self,
        search_name: str,
        backend: TrackerBackend = TrackerBackend.LOCAL,
        local_dir: str | Path | None = None,
    ):
        """Initialize parameter search tracker.

        Args:
            search_name: Name of the search experiment.
            backend: Tracking backend to use.
            local_dir: Directory for local storage.
        """
        self.search_name = search_name
        self.search_id = f"search_{uuid.uuid4().hex[:8]}"
        self.backend_type = backend
        self._local_store = LocalTrackerStore(root_dir=local_dir or "./outputs/tracker")
        self._trials: list[SearchTrial] = []
        self._search_space: dict[str, list[Any]] = {}
        self._loaded = False

    def define_search_space(self, space: dict[str, list[Any]]) -> None:
        """Define the parameter search space.

        Args:
            space: Dictionary mapping parameter names to lists of values.
        """
        self._search_space = space
        self._trials = self._generate_trials(space)
        self._save()

    def _generate_trials(self, space: dict[str, list[Any]]) -> list[SearchTrial]:
        """Generate trials from search space (grid search)."""
        import itertools

        keys = list(space.keys())
        values = list(space.values())
        trials = []

        for i, combination in enumerate(itertools.product(*values)):
            params = dict(zip(keys, combination))
            trial = SearchTrial(
                trial_id=f"{self.search_id}_{i:03d}",
                search_id=self.search_id,
                params=params,
                metrics={},
            )
            trials.append(trial)

        return trials

    def trials(self) -> list[SearchTrial]:
        """Get all trials for this search."""
        if not self._trials and not self._loaded:
            self._trials = self._local_store.load_search(self.search_id) or []
            self._loaded = True
        return self._trials

    def get_trial(self, trial_id: str) -> SearchTrial | None:
        """Get a specific trial by ID."""
        for trial in self.trials():
            if trial.trial_id == trial_id:
                return trial
        return None

    def log_trial_result(
        self,
        trial_id: str,
        metrics: dict[str, float],
        status: RunStatus = RunStatus.COMPLETED,
    ) -> None:
        """Log results for a completed trial.

        Args:
            trial_id: ID of the trial.
            metrics: Dictionary of metric names to values.
            status: Final status of the trial.
        """
        trial = self.get_trial(trial_id)
        if trial is None:
            raise ValueError(f"Unknown trial ID: {trial_id}")

        trial.metrics = metrics
        trial.status = status
        trial.completed_at = datetime.now().isoformat()
        self._save()

    def get_best_trial(
        self,
        metric: str,
        mode: str = "max",
    ) -> SearchTrial | None:
        """Get the best trial by a specific metric.

        Args:
            metric: Name of the metric to optimize.
            mode: "min" or "max".

        Returns:
            Trial with the best metric value, or None if no completed trials.
        """
        completed = [t for t in self.trials() if metric in t.metrics]
        if not completed:
            return None

        if mode == "max":
            return max(completed, key=lambda t: t.metrics[metric])
        else:
            return min(completed, key=lambda t: t.metrics[metric])

    def get_trials_dataframe(self) -> pd.DataFrame:
        """Get all trials as a pandas DataFrame.

        Returns:
            DataFrame with trial parameters and metrics.
        """
        records = []
        for trial in self.trials():
            record = {"trial_id": trial.trial_id, **trial.params}
            record.update(trial.metrics)
            record["status"] = trial.status.value
            records.append(record)
        return pd.DataFrame(records)

    def compare_trials(self, metric: str) -> pd.DataFrame:
        """Compare trials by a specific metric.

        Args:
            metric: Name of the metric to compare.

        Returns:
            DataFrame sorted by the specified metric.
        """
        df = self.get_trials_dataframe()
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not found in trial results")
        return df.sort_values(metric, ascending=False)

    def _save(self) -> None:
        """Save trials to storage."""
        self._local_store.save_search(self.search_id, self._trials)


class ExperimentComparator:
    """Compare results across multiple experiment runs."""

    def __init__(self, store: LocalTrackerStore | None = None):
        """Initialize comparator.

        Args:
            store: Local tracker store to load runs from.
        """
        self.store = store or LocalTrackerStore()

    def load_runs(
        self,
        experiment_name: str | None = None,
        limit: int | None = None,
    ) -> list[RunRecord]:
        """Load runs for comparison.

        Args:
            experiment_name: Filter by experiment name.
            limit: Maximum number of runs to load.

        Returns:
            List of run records sorted by start time.
        """
        runs = self.store.list_runs(experiment_name=experiment_name)
        if limit:
            runs = runs[:limit]
        return runs

    def compare_metrics(
        self,
        runs: list[RunRecord],
        metric_names: list[str],
    ) -> pd.DataFrame:
        """Compare specific metrics across runs.

        Args:
            runs: List of runs to compare.
            metric_names: Names of metrics to compare.

        Returns:
            DataFrame with run_id, config_hash, and specified metrics.
        """
        records = []
        for run in runs:
            record = {
                "run_id": run.run_id,
                "config_hash": run.config_snapshot.config_hash,
                "status": run.status.value,
                "started_at": run.started_at,
            }
            for metric_name in metric_names:
                for m in run.metrics:
                    if m.name == metric_name:
                        record[metric_name] = m.value
                        break
            records.append(record)

        return pd.DataFrame(records)

    def compare_configs(
        self,
        runs: list[RunRecord],
    ) -> pd.DataFrame:
        """Compare configurations across runs.

        Args:
            runs: List of runs to compare.

        Returns:
            DataFrame with config parameters across runs.
        """
        records = []
        all_keys = set()
        for run in runs:
            all_keys.update(run.config_snapshot.config_dict.keys())

        for run in runs:
            record = {
                "run_id": run.run_id,
                "config_hash": run.config_snapshot.config_hash,
            }
            for key in all_keys:
                record[key] = run.config_snapshot.config_dict.get(key)
            records.append(record)

        return pd.DataFrame(records)

    def find_similar_configs(
        self,
        run: RunRecord,
        max_hash_distance: int = 2,
    ) -> list[RunRecord]:
        """Find runs with similar configurations.

        Args:
            run: Reference run to compare against.
            max_hash_distance: Maximum hash character distance for similarity.

        Returns:
            List of runs with similar configurations.
        """
        runs = self.store.list_runs(experiment_name=run.experiment_name)
        similar = []

        for other in runs:
            if other.run_id == run.run_id:
                continue

            hash1 = run.config_snapshot.config_hash
            hash2 = other.config_snapshot.config_hash
            distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

            if distance <= max_hash_distance:
                similar.append(other)

        return similar
