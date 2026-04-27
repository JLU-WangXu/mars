"""Thermostability profile configuration system.

Provides validation and merge mechanisms for thermostability-specific
configuration profiles used in MARS-FIELD design.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# Valid thermostability profile types
VALID_PROFILE_TYPES = {"high_temperature", "thermostable", "cryoprotection"}

# Required top-level keys for thermostability configs
REQUIRED_KEYS = {"thermostability", "generation", "method", "validation"}


@dataclass
class TemperatureRange:
    """Temperature range specification."""

    min_celsius: float
    max_celsius: float
    optimal_celsius: float

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "TemperatureRange":
        return cls(
            min_celsius=data.get("min_celsius", 0),
            max_celsius=data.get("max_celsius", 100),
            optimal_celsius=data.get("optimal_celsius", 37),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "min_celsius": self.min_celsius,
            "max_celsius": self.max_celsius,
            "optimal_celsius": self.optimal_celsius,
        }

    def validate(self) -> list[str]:
        """Validate temperature range consistency."""
        errors = []
        if self.min_celsius > self.max_celsius:
            errors.append(
                f"min_celsius ({self.min_celsius}) > max_celsius ({self.max_celsius})"
            )
        if not (self.min_celsius <= self.optimal_celsius <= self.max_celsius):
            errors.append(
                f"optimal_celsius ({self.optimal_celsius}) outside range "
                f"[{self.min_celsius}, {self.max_celsius}]"
            )
        return errors


@dataclass
class ThermostabilityProfile:
    """Thermostability profile configuration container."""

    profile_type: str
    target_temp_range: TemperatureRange
    design_strategy: dict[str, bool]
    aa_preferences: dict[str, list[str]]
    structural_constraints: dict[str, Any]
    score_modifiers: dict[str, float]
    generation: dict[str, Any]
    method: dict[str, Any]
    evolution: dict[str, Any]
    validation: dict[str, Any]
    raw_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThermostabilityProfile":
        """Create profile from dictionary."""
        thermo = data.get("thermostability", {})
        return cls(
            profile_type=thermo.get("profile_type", "unknown"),
            target_temp_range=TemperatureRange.from_dict(
                thermo.get("target_temp_range", {})
            ),
            design_strategy=thermo.get("design_strategy", {}),
            aa_preferences=thermo.get("aa_preferences", {}),
            structural_constraints=thermo.get("structural_constraints", {}),
            score_modifiers=thermo.get("score_modifiers", {}),
            generation=data.get("generation", {}),
            method=data.get("method", {}),
            evolution=data.get("evolution", {}),
            validation=data.get("validation", {}),
            raw_config=data,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ThermostabilityProfile":
        """Load profile from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"Empty configuration file: {path}")

        return cls.from_dict(data)

    def validate(self) -> list[str]:
        """Validate profile configuration."""
        errors = []

        # Check profile type
        if self.profile_type not in VALID_PROFILE_TYPES:
            errors.append(
                f"Invalid profile_type '{self.profile_type}'. "
                f"Must be one of: {VALID_PROFILE_TYPES}"
            )

        # Validate temperature range
        errors.extend(self.target_temp_range.validate())

        # Check required keys
        for key in REQUIRED_KEYS:
            if key not in self.raw_config:
                errors.append(f"Missing required key: {thermostability}")

        # Validate amino acid preferences
        if "increase" not in self.aa_preferences and "decrease" not in self.aa_preferences:
            errors.append("Must specify at least one of 'increase' or 'decrease' amino acids")

        # Validate score weights
        method = self.raw_config.get("method", {})
        weights = method.get("score_weights", {})
        if "thermostability" not in weights and "cryostability" not in weights:
            errors.append("method.score_weights must include thermostability or cryostability")

        # Validate thresholds
        validation = self.raw_config.get("validation", {})
        thresholds = validation.get("thresholds", {})
        if not thresholds:
            errors.append("validation.thresholds must be specified")

        return errors

    def merge_with_base(self, base_config: dict[str, Any]) -> dict[str, Any]:
        """Merge profile with base configuration.

        Args:
            base_config: Base configuration dictionary to merge with.

        Returns:
            Merged configuration dictionary with profile overrides applied.
        """
        merged = copy.deepcopy(base_config)

        # Deep merge strategy:
        # Profile settings override base settings
        # Nested dicts are merged recursively

        # Merge generation settings
        if "generation" in self.raw_config:
            merged.setdefault("generation", {})
            merged["generation"] = self._deep_merge(
                merged["generation"], self.raw_config["generation"]
            )

        # Merge method settings (especially score_weights)
        if "method" in self.raw_config:
            merged.setdefault("method", {})
            merged["method"] = self._deep_merge(
                merged["method"], self.raw_config["method"]
            )

        # Merge evolution settings
        if "evolution" in self.raw_config:
            merged.setdefault("evolution", {})
            merged["evolution"] = self._deep_merge(
                merged["evolution"], self.raw_config["evolution"]
            )

        # Add thermostability profile metadata
        merged["thermostability_profile"] = {
            "profile_type": self.profile_type,
            "source": "thermostability_profiles",
            "temp_range": self.target_temp_range.to_dict(),
        }

        return merged

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries.

        Override values take precedence. Lists are replaced, not concatenated.
        """
        result = copy.deepcopy(base)

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ThermostabilityProfile._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result


def load_thermostability_profile(
    profile_name: str,
    config_dir: str | Path | None = None,
) -> ThermostabilityProfile:
    """Load a thermostability profile by name.

    Args:
        profile_name: Name of the profile (e.g., 'high_temp', 'thermostable').
        config_dir: Directory containing thermostability profiles.
                   If None, uses default 'configs/thermostability_profiles'.

    Returns:
        Loaded and validated ThermostabilityProfile.

    Raises:
        FileNotFoundError: If profile file not found.
        ValueError: If profile validation fails.
    """
    if config_dir is None:
        # Infer from project structure
        config_dir = Path(__file__).parent.parent.parent / "configs" / "thermostability_profiles"
    else:
        config_dir = Path(config_dir)

    # Map profile names to file names
    profile_map = {
        "high_temp": "high_temp.yaml",
        "thermostable": "thermostable.yaml",
        "cryoprotection": "cryoprotection.yaml",
    }

    if profile_name not in profile_map:
        available = list(profile_map.keys())
        raise ValueError(
            f"Unknown profile '{profile_name}'. Available profiles: {available}"
        )

    profile_path = config_dir / profile_map[profile_name]

    if not profile_path.exists():
        raise FileNotFoundError(
            f"Profile file not found: {profile_path}. "
            f"Available profiles: {list(profile_map.keys())}"
        )

    profile = ThermostabilityProfile.from_yaml(profile_path)

    # Validate on load
    errors = profile.validate()
    if errors:
        raise ValueError(
            f"Profile validation failed for '{profile_name}':\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return profile


def merge_profile_with_config(
    profile_name: str,
    base_config: dict[str, Any],
    config_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load and merge a thermostability profile with a base configuration.

    Args:
        profile_name: Name of the thermostability profile to apply.
        base_config: Base configuration dictionary.
        config_dir: Directory containing thermostability profiles.

    Returns:
        Merged configuration with profile applied.

    Raises:
        ValueError: If profile not found or validation fails.
    """
    profile = load_thermostability_profile(profile_name, config_dir)
    return profile.merge_with_base(base_config)


def list_available_profiles(config_dir: str | Path | None = None) -> list[str]:
    """List all available thermostability profile names.

    Args:
        config_dir: Directory to search for profiles.

    Returns:
        List of available profile names.
    """
    if config_dir is None:
        config_dir = Path(__file__).parent.parent.parent / "configs" / "thermostability_profiles"
    else:
        config_dir = Path(config_dir)

    if not config_dir.exists():
        return []

    return [p.stem for p in config_dir.glob("*.yaml")]


__all__ = [
    "ThermostabilityProfile",
    "TemperatureRange",
    "VALID_PROFILE_TYPES",
    "load_thermostability_profile",
    "merge_profile_with_config",
    "list_available_profiles",
]
