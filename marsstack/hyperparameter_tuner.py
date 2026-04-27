"""Hyperparameter tuning module with Optuna, GridSearch, RandomSearch, and Bayesian optimization."""

from __future__ import annotations

import json
import logging
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class OptimizerType(Enum):
    """Available hyperparameter optimization strategies."""
    OPTUNA = "optuna"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"


@dataclass
class SearchSpace:
    """Definition of hyperparameter search space."""

    name: str
    params: dict[str, Any]

    def to_optuna_space(self) -> dict[str, Any]:
        """Convert to Optuna search space format."""
        space = {}
        for param_name, param_config in self.params.items():
            if isinstance(param_config, dict):
                param_type = param_config.get("type", "float")
                if param_type == "int":
                    space[param_name] = {
                        "type": "int",
                        "low": param_config.get("low", 1),
                        "high": param_config.get("high", 100),
                        "step": param_config.get("step", 1),
                        "log": param_config.get("log", False),
                    }
                elif param_type == "categorical":
                    space[param_name] = {
                        "type": "categorical",
                        "choices": param_config.get("choices", []),
                    }
                else:
                    space[param_name] = {
                        "type": "float",
                        "low": param_config.get("low", 0.0),
                        "high": param_config.get("high", 1.0),
                        "log": param_config.get("log", False),
                    }
            else:
                space[param_name] = {"type": "categorical", "choices": [param_config]}
        return space


# Predefined search spaces for Mars algorithms
class PredefinedSpaces:
    """Predefined search spaces for common Mars algorithm parameters."""

    @staticmethod
    def mars_score_weights() -> SearchSpace:
        """Search space for MARS score weighting parameters."""
        return SearchSpace(
            name="mars_score_weights",
            params={
                "evolution_weight": {"type": "float", "low": 0.0, "high": 2.0, "log": False},
                "structure_weight": {"type": "float", "low": 0.0, "high": 2.0, "log": False},
                "energy_weight": {"type": "float", "low": 0.0, "high": 2.0, "log": False},
                "conservation_weight": {"type": "float", "low": 0.0, "high": 2.0, "log": False},
                "diversity_weight": {"type": "float", "low": 0.0, "high": 2.0, "log": False},
            },
        )

    @staticmethod
    def ensemble_ranker() -> SearchSpace:
        """Search space for ensemble ranker parameters."""
        return SearchSpace(
            name="ensemble_ranker",
            params={
                "n_estimators": {"type": "int", "low": 10, "high": 200, "step": 10},
                "max_depth": {"type": "int", "low": 3, "high": 15, "step": 1},
                "learning_rate": {"type": "float", "low": 0.001, "high": 0.3, "log": True},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
                "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
            },
        )

    @staticmethod
    def evolution_params() -> SearchSpace:
        """Search space for evolution algorithm parameters."""
        return SearchSpace(
            name="evolution_params",
            params={
                "population_size": {"type": "int", "low": 20, "high": 200, "step": 20},
                "generations": {"type": "int", "low": 50, "high": 500, "step": 50},
                "mutation_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": False},
                "crossover_rate": {"type": "float", "low": 0.5, "high": 0.95, "log": False},
                "tournament_size": {"type": "int", "low": 2, "high": 10, "step": 1},
            },
        )

    @staticmethod
    def fusion_ranker() -> SearchSpace:
        """Search space for fusion ranker parameters."""
        return SearchSpace(
            name="fusion_ranker",
            params={
                "alpha": {"type": "float", "low": 0.0, "high": 1.0, "log": False},
                "beta": {"type": "float", "low": 0.0, "high": 1.0, "log": False},
                "temperature": {"type": "float", "low": 0.01, "high": 2.0, "log": True},
                "use_uncertainty": {"type": "categorical", "choices": [True, False]},
            },
        )

    @staticmethod
    def cache_manager() -> SearchSpace:
        """Search space for cache manager parameters."""
        return SearchSpace(
            name="cache_manager",
            params={
                "max_size": {"type": "int", "low": 100, "high": 10000, "step": 100, "log": True},
                "ttl": {"type": "int", "low": 60, "high": 3600, "step": 60, "log": True},
                "eviction_threshold": {"type": "float", "low": 0.7, "high": 0.99, "log": False},
            },
        )


@dataclass
class TuningResult:
    """Results from hyperparameter tuning."""

    best_params: dict[str, Any]
    best_score: float
    n_trials: int
    duration_seconds: float
    study_name: str
    optimizer_type: OptimizerType
    trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "duration_seconds": self.duration_seconds,
            "study_name": self.study_name,
            "optimizer_type": self.optimizer_type.value,
            "trials": self.trials,
        }


class ObjectiveFunction(Protocol):
    """Protocol for objective functions used in hyperparameter tuning."""

    def __call__(self, params: dict[str, Any]) -> float:
        """Evaluate params and return score."""
        ...


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking."""

    @abstractmethod
    def log_trial(self, trial_number: int, params: dict[str, Any], score: float) -> None:
        """Log a single trial."""
        pass

    @abstractmethod
    def log_study(self, study_name: str, search_space: SearchSpace) -> None:
        """Log the study configuration."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finalize tracking."""
        pass


class SimpleTracker(ExperimentTracker):
    """Simple in-memory experiment tracker."""

    def __init__(self) -> None:
        self.trials: list[dict[str, Any]] = []
        self.studies: dict[str, Any] = {}

    def log_trial(self, trial_number: int, params: dict[str, Any], score: float) -> None:
        self.trials.append({"trial": trial_number, "params": params, "score": score})

    def log_study(self, study_name: str, search_space: SearchSpace) -> None:
        self.studies[study_name] = {"search_space": search_space.params, "started_at": time.time()}

    def finish(self) -> None:
        for study in self.studies.values():
            study["finished_at"] = time.time()
            if "started_at" in study:
                study["duration"] = study["finished_at"] - study["started_at"]

    def get_best_trial(self) -> dict[str, Any] | None:
        if not self.trials:
            return None
        return max(self.trials, key=lambda t: t["score"])


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker integration."""

    def __init__(self, experiment_name: str, run_name: str | None = None) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._run = None
        self._mlflow_available = self._check_mlflow()

    def _check_mlflow(self) -> bool:
        try:
            import mlflow
            return True
        except ImportError:
            logger.warning("MLflow not available. Install with: pip install mlflow")
            return False

    def log_trial(self, trial_number: int, params: dict[str, Any], score: float) -> None:
        if not self._mlflow_available or self._run is None:
            return
        import mlflow
        mlflow.log_metrics({"score": score, "trial": trial_number}, step=trial_number)
        mlflow.log_params(params)

    def log_study(self, study_name: str, search_space: SearchSpace) -> None:
        if not self._mlflow_available:
            return
        import mlflow
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=self.run_name or study_name)
        mlflow.log_param("search_space", json.dumps(search_space.params))

    def finish(self) -> None:
        if self._mlflow_available and self._run is not None:
            import mlflow
            mlflow.end_run()


class HyperparameterTuner:
    """Main hyperparameter tuning orchestrator."""

    def __init__(
        self,
        objective_fn: ObjectiveFunction,
        search_space: SearchSpace,
        optimizer: OptimizerType = OptimizerType.OPTUNA,
        tracker: ExperimentTracker | None = None,
        study_name: str = "mars_tuning",
        direction: str = "maximize",
    ) -> None:
        self.objective_fn = objective_fn
        self.search_space = search_space
        self.optimizer = optimizer
        self.tracker = tracker or SimpleTracker()
        self.study_name = study_name
        self.direction = direction

    def tune(self, n_trials: int = 100, timeout: int | None = None) -> TuningResult:
        """Execute hyperparameter tuning."""
        start_time = time.time()
        self.tracker.log_study(self.study_name, self.search_space)

        if self.optimizer == OptimizerType.OPTUNA:
            result = self._tune_optuna(n_trials, timeout)
        elif self.optimizer == OptimizerType.GRID_SEARCH:
            result = self._tune_grid_search(n_trials)
        elif self.optimizer == OptimizerType.RANDOM_SEARCH:
            result = self._tune_random_search(n_trials)
        elif self.optimizer == OptimizerType.BAYESIAN:
            result = self._tune_bayesian(n_trials)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer}")

        result.duration_seconds = time.time() - start_time
        self.tracker.finish()
        return result

    def _tune_optuna(self, n_trials: int, timeout: int | None) -> TuningResult:
        """Run Optuna optimization."""
        try:
            import optuna
            from optuna.distributions import CategoricalDistribution, IntUniformDistribution, LogUniformDistribution, UniformDistribution
        except ImportError:
            logger.error("Optuna not installed. Install with: pip install optuna")
            raise ImportError("Optuna is required for OPTUNA optimizer")

        def optuna_objective(trial: optuna.Trial) -> float:
            params = {}
            for param_name, param_config in self.search_space.params.items():
                param_type = param_config.get("type", "float")
                if param_type == "int":
                    if param_config.get("log", False):
                        params[param_name] = trial.suggest_int(
                            param_name,
                            int(param_config["low"]),
                            int(param_config["high"]),
                            log=True,
                        )
                    else:
                        step = param_config.get("step", 1)
                        params[param_name] = trial.suggest_int(
                            param_name,
                            int(param_config["low"]),
                            int(param_config["high"]),
                            step=step,
                        )
                elif param_type == "categorical":
                    choices = param_config.get("choices", [])
                    params[param_name] = trial.suggest_categorical(param_name, choices)
                else:
                    if param_config.get("log", False):
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                            log=True,
                        )
                    else:
                        params[param_name] = trial.suggest_float(
                            param_name,
                            param_config["low"],
                            param_config["high"],
                        )

            score = self.objective_fn(params)
            self.tracker.log_trial(trial.number, params, score)
            return score

        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize" if self.direction == "maximize" else "minimize",
            sampler=optuna.samplers.TPESampler(),
        )
        study.optimize(optuna_objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        trials = [
            {"trial": i, "params": t.params, "score": t.value}
            for i, t in enumerate(study.trials)
            if t.value is not None
        ]

        return TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            n_trials=len(study.trials),
            duration_seconds=0.0,
            study_name=self.study_name,
            optimizer_type=OptimizerType.OPTUNA,
            trials=trials,
        )

    def _tune_grid_search(self, n_points: int) -> TuningResult:
        """Run grid search optimization."""
        param_names = list(self.search_space.params.keys())
        param_configs = list(self.search_space.params.values())

        param_values = []
        for config in param_configs:
            param_type = config.get("type", "float")
            low = config.get("low", 0)
            high = config.get("high", 1)
            n = min(n_points, 10)

            if param_type == "int":
                values = np.linspace(int(low), int(high), n, dtype=int).tolist()
            elif param_type == "categorical":
                values = config.get("choices", [])
            else:
                values = np.linspace(low, high, n).tolist()
            param_values.append(values)

        from itertools import product
        combinations = list(product(*param_values))

        best_params = {}
        best_score = float("-inf") if self.direction == "maximize" else float("inf")
        trials = []

        for idx, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            score = self.objective_fn(params)
            self.tracker.log_trial(idx, params, score)
            trials.append({"trial": idx, "params": params, "score": score})

            if self.direction == "maximize" and score > best_score:
                best_score = score
                best_params = params.copy()
            elif self.direction == "minimize" and score < best_score:
                best_score = score
                best_params = params.copy()

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(trials),
            duration_seconds=0.0,
            study_name=self.study_name,
            optimizer_type=OptimizerType.GRID_SEARCH,
            trials=trials,
        )

    def _tune_random_search(self, n_trials: int) -> TuningResult:
        """Run random search optimization."""
        best_params = {}
        best_score = float("-inf") if self.direction == "maximize" else float("inf")
        trials = []

        rng = np.random.default_rng()

        for i in range(n_trials):
            params = {}
            for param_name, config in self.search_space.params.items():
                param_type = config.get("type", "float")
                low = config.get("low", 0)
                high = config.get("high", 1)

                if param_type == "int":
                    if config.get("log", False):
                        params[param_name] = int(rng.uniform(
                            math.log(low), math.log(high)
                        ))
                        params[param_name] = int(math.exp(params[param_name]))
                    else:
                        params[param_name] = int(rng.integers(int(low), int(high) + 1))
                elif param_type == "categorical":
                    choices = config.get("choices", [])
                    params[param_name] = rng.choice(choices)
                else:
                    if config.get("log", False):
                        params[param_name] = math.exp(rng.uniform(math.log(low), math.log(high)))
                    else:
                        params[param_name] = rng.uniform(low, high)

            score = self.objective_fn(params)
            self.tracker.log_trial(i, params, score)
            trials.append({"trial": i, "params": params, "score": score})

            if self.direction == "maximize" and score > best_score:
                best_score = score
                best_params = params.copy()
            elif self.direction == "minimize" and score < best_score:
                best_score = score
                best_params = params.copy()

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=n_trials,
            duration_seconds=0.0,
            study_name=self.study_name,
            optimizer_type=OptimizerType.RANDOM_SEARCH,
            trials=trials,
        )

    def _tune_bayesian(self, n_trials: int) -> TuningResult:
        """Run Bayesian optimization using a Gaussian Process surrogate."""
        try:
            from scipy.optimize import gp_minimize
        except ImportError:
            logger.error("scipy not installed. Install with: pip install scipy")
            raise ImportError("scipy is required for BAYESIAN optimizer")

        param_names = list(self.search_space.params.keys())
        bounds = []

        for config in self.search_space.params.values():
            param_type = config.get("type", "float")
            low = config.get("low", 0.0)
            high = config.get("high", 1.0)
            bounds.append((float(low), float(high)))

        X_observed: list[list[float]] = []
        y_observed: list[float] = []

        def scipy_objective(x: np.ndarray) -> float:
            params = dict(zip(param_names, x.tolist()))
            score = self.objective_fn(params)
            self.tracker.log_trial(len(X_observed), params, score)
            X_observed.append(x.tolist())
            y_observed.append(score)
            return -score if self.direction == "maximize" else score

        n_initial = min(5, n_trials // 4)
        rng = np.random.default_rng()
        X_init = rng.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            size=(n_initial, len(bounds)),
        )

        for x in X_init:
            scipy_objective(x)

        result = gp_minimize(
            scipy_objective,
            bounds,
            n_calls=n_trials - n_initial,
            random_state=int(time.time()),
            noise="gaussian",
            n_restarts_optimizer=5,
        )

        best_idx = np.argmin(y_observed) if self.direction == "minimize" else np.argmax(y_observed)
        best_params = dict(zip(param_names, X_observed[best_idx]))
        best_score = y_observed[best_idx]

        trials = [
            {"trial": i, "params": dict(zip(param_names, x)), "score": y}
            for i, (x, y) in enumerate(zip(X_observed, y_observed))
        ]

        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(trials),
            duration_seconds=0.0,
            study_name=self.study_name,
            optimizer_type=OptimizerType.BAYESIAN,
            trials=trials,
        )


def create_tuner(
    objective_fn: ObjectiveFunction,
    search_space: SearchSpace | str = "mars_score_weights",
    optimizer: OptimizerType = OptimizerType.OPTUNA,
    use_mlflow: bool = False,
    study_name: str = "mars_tuning",
) -> HyperparameterTuner:
    """Factory function to create a configured hyperparameter tuner.

    Args:
        objective_fn: Function that takes params dict and returns score.
        search_space: SearchSpace instance or name of predefined space.
        optimizer: Optimization strategy to use.
        use_mlflow: Whether to use MLflow tracking.
        study_name: Name for the tuning study.

    Returns:
        Configured HyperparameterTuner instance.
    """
    if isinstance(search_space, str):
        space_map = {
            "mars_score_weights": PredefinedSpaces.mars_score_weights,
            "ensemble_ranker": PredefinedSpaces.ensemble_ranker,
            "evolution_params": PredefinedSpaces.evolution_params,
            "fusion_ranker": PredefinedSpaces.fusion_ranker,
            "cache_manager": PredefinedSpaces.cache_manager,
        }
        if search_space not in space_map:
            raise ValueError(f"Unknown search space: {search_space}. Available: {list(space_map.keys())}")
        search_space = space_map[search_space]()

    tracker: ExperimentTracker
    if use_mlflow:
        tracker = MLflowTracker(experiment_name=study_name)
    else:
        tracker = SimpleTracker()

    return HyperparameterTuner(
        objective_fn=objective_fn,
        search_space=search_space,
        optimizer=optimizer,
        tracker=tracker,
        study_name=study_name,
    )
