"""External API integration for enhanced protein stability prediction.

This module provides integrations with external APIs for:
- ThermoNet: Thermostability prediction using deep learning
- DDGun: Thermostability change prediction for point mutations
- Molecular dynamics pre-heating suggestions
- Protein expression prediction API

All APIs support:
- Async/await for non-blocking calls
- Automatic retry with exponential backoff
- Response caching to reduce redundant requests
- Graceful fallback when APIs are unavailable
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Workaround for Python 3.13 dataclass module registration issue
_module_name = __name__
if _module_name not in sys.modules:
    sys.modules[_module_name] = sys.modules.get("__main__", None) or type(sys)("module")

import aiohttp

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data classes                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class ThermoNetResult:
    """ThermoNet thermostability prediction result."""

    sequence: str
    wildtype_stability: float
    mutant_stability: float | None
    mutation_position: int | None
    mutation_aa: str | None
    confidence: float
    thermal_tolerance_estimate: float  # Celsius
    api_version: str
    cached: bool = False


@dataclass
class DDGunResult:
    """DDGun thermostability change prediction result."""

    sequence: str
    mutation_position: int
    wildtype_aa: str
    mutant_aa: str
    ddg: float  # Delta Delta G in kcal/mol (negative = stabilizing)
    confidence: float
    method: str  # "deep" or "thermo"
    cached: bool = False


@dataclass
class MDPreHeatSuggestion:
    """Molecular dynamics pre-heating protocol suggestion."""

    protein_length: int
    initial_temperature_k: float
    target_temperature_k: float
    heating_rate_k_per_ns: float
    equilibration_steps: int
    ensemble: str  # "NVT" or "NPT"
    box_padding_angstrom: float
    ionic_concentration_mm: float
    salt_type: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExpressionPrediction:
    """Protein expression level prediction result."""

    sequence: str
    predicted_expression_level: float  # Normalized 0-1
    expression_category: str  # "high", "medium", "low"
    codon_usage_score: float
    mrna_stability_score: float
    secretory_signal_probability: float
    membrane_protein: bool
    solubility_score: float
    confidence: float
    recommendations: list[str] = field(default_factory=list)
    cached: bool = False


@dataclass
class APICallResult:
    """Generic API call result with metadata."""

    success: bool
    data: Any = None
    error: str | None = None
    cached: bool = False
    latency_ms: float = 0.0
    api_name: str = ""


# --------------------------------------------------------------------------- #
# Cache implementation                                                         #
# --------------------------------------------------------------------------- #


class APICache:
    """Disk-backed API response cache with TTL support."""

    def __init__(self, cache_dir: Path | None = None, default_ttl_seconds: int = 86400):
        if cache_dir is None:
            cache_dir = Path.home() / ".mars" / "api_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl_seconds
        self._memory_cache: dict[str, tuple[Any, datetime]] = {}
        self._memory_ttl_seconds = 3600  # 1 hour for in-memory

    def _make_key(self, api_name: str, *args: Any, **kwargs: Any) -> str:
        """Generate a cache key from API name and parameters."""
        key_data = {
            "api": api_name,
            "args": args,
            "kwargs": {k: v for k, v in sorted(kwargs.items()) if v is not None},
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, api_name: str, *args: Any, **kwargs: Any) -> Any | None:
        """Retrieve cached value if available and not expired."""
        key = self._make_key(api_name, *args, **kwargs)

        # Check memory cache first
        if key in self._memory_cache:
            value, expires_at = self._memory_cache[key]
            if datetime.now() < expires_at:
                logger.debug(f"Cache hit (memory): {api_name}")
                return value
            else:
                del self._memory_cache[key]

        # Check disk cache
        cache_path = self._cache_path(key)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                expires_at = datetime.fromisoformat(data["_expires_at"])
                if datetime.now() < expires_at:
                    logger.debug(f"Cache hit (disk): {api_name}")
                    # Promote to memory cache
                    self._memory_cache[key] = (data["value"], expires_at)
                    return data["value"]
                else:
                    cache_path.unlink(missing_ok=True)
            except (json.JSONDecodeError, KeyError, ValueError):
                cache_path.unlink(missing_ok=True)

        return None

    def set(
        self,
        value: Any,
        api_name: str,
        *args: Any,
        ttl_seconds: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Store value in cache with TTL."""
        key = self._make_key(api_name, *args, **kwargs)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)

        # Store in memory
        self._memory_cache[key] = (value, expires_at)

        # Store on disk
        data = {"value": value, "_expires_at": expires_at.isoformat()}
        try:
            self._cache_path(key).write_text(json.dumps(data, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to write cache to disk: {e}")

    def clear_expired(self) -> int:
        """Remove expired cache entries. Returns count of removed entries."""
        removed = 0
        now = datetime.now()

        # Clear memory cache
        expired_keys = [k for k, (_, exp) in self._memory_cache.items() if now >= exp]
        for k in expired_keys:
            del self._memory_cache[k]
            removed += 1

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                expires_at = datetime.fromisoformat(data["_expires_at"])
                if now >= expires_at:
                    cache_file.unlink()
                    removed += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                cache_file.unlink(missing_ok=True)
                removed += 1

        return removed


# Global cache instance
_api_cache: APICache | None = None


def get_cache() -> APICache:
    """Get or create the global API cache instance."""
    global _api_cache
    if _api_cache is None:
        _api_cache = APICache()
    return _api_cache


# --------------------------------------------------------------------------- #
# HTTP client with retry and timeout                                           #
# --------------------------------------------------------------------------- #


@dataclass
class HTTPClientConfig:
    """Configuration for async HTTP client."""

    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    retry_backoff_max: float = 30.0
    connection_limit: int = 10


class HTTPClient:
    """Async HTTP client with automatic retry and timeout."""

    def __init__(self, config: HTTPClientConfig | None = None):
        self.config = config or HTTPClientConfig()
        self._session: aiohttp.ClientSession | None = None
        self._semaphore: asyncio.Semaphore | None = None

    async def __aenter__(self) -> HTTPClient:
        connector = aiohttp.TCPConnector(limit=self.config.connection_limit)
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        self._semaphore = asyncio.Semaphore(self.config.connection_limit)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform GET request with retry logic."""
        return await self._request("GET", url, params=params, headers=headers)

    async def post(
        self,
        url: str,
        json_data: dict[str, Any] | None = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform POST request with retry logic."""
        return await self._request("POST", url, json_data=json_data, data=data, headers=headers)

    async def _request(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not self._session:
            raise RuntimeError("HTTPClient must be used as async context manager")

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                async with self._semaphore:  # type: ignore
                    start_time = time.perf_counter()
                    async with self._session.request(
                        method=method,
                        url=url,
                        params=params,
                        json=json_data,
                        data=data,
                        headers=headers,
                    ) as response:
                        elapsed_ms = (time.perf_counter() - start_time) * 1000
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:
                            # Rate limited - wait longer
                            wait_time = min(
                                self.config.retry_backoff_base ** (attempt + 3),
                                self.config.retry_backoff_max,
                            )
                            logger.warning(f"Rate limited, waiting {wait_time:.1f}s")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            text = await response.text()
                            raise aiohttp.ClientResponseError(
                                response.request_info,
                                response.history,
                                status=response.status,
                                message=text,
                            )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    wait_time = min(
                        self.config.retry_backoff_base ** attempt,
                        self.config.retry_backoff_max,
                    )
                    logger.warning(f"Request failed (attempt {attempt + 1}): {e}. Retrying in {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)

        raise RuntimeError(f"Request failed after {self.config.max_retries} attempts: {last_error}")


# --------------------------------------------------------------------------- #
# ThermoNet API integration                                                   #
# --------------------------------------------------------------------------- #


class ThermoNetAPI:
    """ThermoNet API client for thermostability prediction.

    ThermoNet is a deep learning model for predicting protein thermostability.
    API documentation: https://protdata.swmt.de/thermonet/

    Note: This integration uses the ThermoNet web API. If the API is unavailable,
    the client falls back to a rule-based estimation using sequence properties.
    """

    BASE_URL = "https://protdata.swmt.de/thermonet/api"
    TIMEOUT = 60.0

    def __init__(
        self,
        cache: APICache | None = None,
        http_client: HTTPClient | None = None,
    ):
        self.cache = cache or get_cache()
        self.http_client = http_client

    async def predict_thermostability(
        self,
        sequence: str,
        temperature_celsius: float = 50.0,
    ) -> APICallResult[ThermoNetResult]:
        """Predict thermostability for a protein sequence.

        Args:
            sequence: Protein amino acid sequence (single-letter code)
            temperature_celsius: Target temperature for prediction

        Returns:
            APICallResult containing ThermoNetResult with stability predictions
        """
        start_time = time.perf_counter()

        # Check cache first
        cached_result = self.cache.get("thermonet", sequence=sequence, temp=temperature_celsius)
        if cached_result is not None:
            cached_result["cached"] = True
            return APICallResult(
                success=True,
                data=ThermoNetResult(**cached_result),
                cached=True,
                latency_ms=0.0,
                api_name="thermonet",
            )

        try:
            result = await self._call_api(sequence, temperature_celsius)
            result["cached"] = False

            # Cache successful result
            self.cache.set(result, "thermonet", sequence=sequence, temp=temperature_celsius)

            latency_ms = (time.perf_counter() - start_time) * 1000
            return APICallResult(
                success=True,
                data=ThermoNetResult(**result),
                cached=False,
                latency_ms=latency_ms,
                api_name="thermonet",
            )
        except Exception as e:
            logger.error(f"ThermoNet API failed: {e}")
            # Fallback to rule-based estimation
            fallback_result = self._rule_based_fallback(sequence, temperature_celsius)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return APICallResult(
                success=True,
                data=fallback_result,
                error=str(e),
                cached=False,
                latency_ms=latency_ms,
                api_name="thermonet",
            )

    async def predict_mutation_effect(
        self,
        sequence: str,
        mutation_position: int,
        mutant_aa: str,
        temperature_celsius: float = 50.0,
    ) -> APICallResult[ThermoNetResult]:
        """Predict the effect of a mutation on thermostability.

        Args:
            sequence: Wild-type protein sequence
            mutation_position: Position of mutation (1-indexed)
            mutant_aa: Mutant amino acid (single-letter code)
            temperature_celsius: Target temperature

        Returns:
            APICallResult containing ThermoNetResult with mutation effect prediction
        """
        start_time = time.perf_counter()

        # Check cache
        cached_result = self.cache.get(
            "thermonet_mutation",
            sequence=sequence,
            pos=mutation_position,
            mutant=mutant_aa,
            temp=temperature_celsius,
        )
        if cached_result is not None:
            cached_result["cached"] = True
            return APICallResult(
                success=True,
                data=ThermoNetResult(**cached_result),
                cached=True,
                latency_ms=0.0,
                api_name="thermonet",
            )

        try:
            result = await self._call_mutation_api(sequence, mutation_position, mutant_aa, temperature_celsius)
            result["cached"] = False

            self.cache.set(result, "thermonet_mutation", sequence=sequence, pos=mutation_position, mutant=mutant_aa, temp=temperature_celsius)

            latency_ms = (time.perf_counter() - start_time) * 1000
            return APICallResult(
                success=True,
                data=ThermoNetResult(**result),
                cached=False,
                latency_ms=latency_ms,
                api_name="thermonet",
            )
        except Exception as e:
            logger.error(f"ThermoNet mutation API failed: {e}")
            fallback_result = self._rule_based_mutation_fallback(sequence, mutation_position, mutant_aa, temperature_celsius)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return APICallResult(
                success=True,
                data=fallback_result,
                error=str(e),
                cached=False,
                latency_ms=latency_ms,
                api_name="thermonet",
            )

    async def _call_api(self, sequence: str, temperature: float) -> dict[str, Any]:
        """Call ThermoNet prediction API."""
        if self.http_client is None:
            async with HTTPClient() as client:
                return await client.post(
                    f"{self.BASE_URL}/predict",
                    json_data={"sequence": sequence, "temperature": temperature},
                )
        else:
            return await self.http_client.post(
                f"{self.BASE_URL}/predict",
                json_data={"sequence": sequence, "temperature": temperature},
            )

    async def _call_mutation_api(
        self,
        sequence: str,
        position: int,
        mutant: str,
        temperature: float,
    ) -> dict[str, Any]:
        """Call ThermoNet mutation effect API."""
        if self.http_client is None:
            async with HTTPClient() as client:
                return await client.post(
                    f"{self.BASE_URL}/mutate",
                    json_data={
                        "sequence": sequence,
                        "position": position,
                        "mutant": mutant,
                        "temperature": temperature,
                    },
                )
        else:
            return await self.http_client.post(
                f"{self.BASE_URL}/mutate",
                json_data={
                    "sequence": sequence,
                    "position": position,
                    "mutant": mutant,
                    "temperature": temperature,
                },
            )

    def _rule_based_fallback(self, sequence: str, temperature: float) -> ThermoNetResult:
        """Rule-based fallback when API is unavailable.

        Estimates thermostability based on amino acid composition and
        sequence properties. This is less accurate than the deep learning
        model but provides reasonable estimates for screening.
        """
        # Thermophilic amino acid preferences (from literature)
        thermophilic_favored = {"E", "L", "R", "K", "A", "I", "V", "M"}
        thermophilic_avoided = {"T", "S", "N", "Y", "C", "D", "P", "G"}

        count = len(sequence)
        if count == 0:
            return ThermoNetResult(
                sequence=sequence,
                wildtype_stability=0.5,
                mutant_stability=None,
                mutation_position=None,
                mutation_aa=None,
                confidence=0.1,
                thermal_tolerance_estimate=temperature,
                api_version="fallback-v1.0",
                cached=False,
            )

        favored_ratio = sum(1 for aa in sequence if aa in thermophilic_favored) / count
        avoided_ratio = sum(1 for aa in sequence if aa in thermophilic_avoided) / count

        # Estimate thermal tolerance based on composition
        composition_score = favored_ratio - avoided_ratio
        # Map score to temperature range (mesophilic: 25-50C, thermophilic: 50-80C, hyperthermophilic: 80-120C)
        est_temp = temperature + composition_score * 30.0
        est_temp = max(25.0, min(120.0, est_temp))

        # Stability score: higher is more stable
        stability = 0.5 + composition_score * 0.4
        stability = max(0.0, min(1.0, stability))

        return ThermoNetResult(
            sequence=sequence,
            wildtype_stability=stability,
            mutant_stability=None,
            mutation_position=None,
            mutation_aa=None,
            confidence=0.3,  # Low confidence for fallback
            thermal_tolerance_estimate=round(est_temp, 1),
            api_version="fallback-v1.0",
            cached=False,
        )

    def _rule_based_mutation_fallback(
        self,
        sequence: str,
        position: int,
        mutant: str,
        temperature: float,
    ) -> ThermoNetResult:
        """Rule-based mutation effect prediction fallback."""
        wildtype_aa = sequence[position - 1] if 1 <= position <= len(sequence) else "X"

        # Simple stability impact scoring
        thermostability_effects = {
            ("G", "A"): 0.05, ("A", "G"): -0.02,
            ("V", "I"): 0.08, ("I", "V"): -0.05,
            ("L", "I"): 0.04, ("I", "L"): -0.03,
            ("M", "L"): 0.02, ("L", "M"): -0.01,
            ("P", "A"): -0.1, ("A", "P"): 0.05,
            ("S", "T"): 0.03, ("T", "S"): -0.02,
            ("D", "E"): 0.08, ("E", "D"): -0.06,
            ("N", "Q"): 0.02, ("Q", "N"): -0.01,
            ("K", "R"): 0.06, ("R", "K"): -0.04,
        }

        effect = thermostability_effects.get((wildtype_aa, mutant), 0.0)

        base_stability = 0.5 + effect
        mutant_stability = base_stability + effect

        return ThermoNetResult(
            sequence=sequence,
            wildtype_stability=round(base_stability, 3),
            mutant_stability=round(mutant_stability, 3),
            mutation_position=position,
            mutation_aa=mutant,
            confidence=0.25,
            thermal_tolerance_estimate=temperature + effect * 20.0,
            api_version="fallback-v1.0",
            cached=False,
        )


# --------------------------------------------------------------------------- #
# DDGun API integration                                                        #
# --------------------------------------------------------------------------- #


class DDGunAPI:
    """DDGun API client for thermostability change prediction.

    DDGun predicts the change in thermostability (DDG) caused by point mutations.
    API documentation: http://biologicalpy.pythonanywhere.com/

    Falls back to empirical calculations when API is unavailable.
    """

    BASE_URL = "https://ddgun.biocomp.unibo.it/api"
    BACKUP_URL = "http://biologicalpy.pythonanywhere.com/ddgun"

    def __init__(
        self,
        cache: APICache | None = None,
        http_client: HTTPClient | None = None,
    ):
        self.cache = cache or get_cache()
        self.http_client = http_client
        self._use_backup = False

    async def predict_ddg(
        self,
        sequence: str,
        mutation_position: int,
        mutant_aa: str,
        method: str = "auto",
    ) -> APICallResult[DDGunResult]:
        """Predict thermostability change (DDG) for a mutation.

        Args:
            sequence: Protein sequence (single-letter code)
            mutation_position: Position of mutation (1-indexed)
            mutant_aa: Mutant amino acid (single-letter code)
            method: Prediction method - "deep" (deep learning), "thermo" (thermodynamic), or "auto"

        Returns:
            APICallResult containing DDGunResult with DDG prediction
        """
        start_time = time.perf_counter()

        if not (1 <= mutation_position <= len(sequence)):
            return APICallResult(
                success=False,
                error=f"Invalid mutation position: {mutation_position}",
                latency_ms=0.0,
                api_name="ddgun",
            )

        wildtype_aa = sequence[mutation_position - 1]

        # Check cache
        cached_result = self.cache.get(
            "ddgun",
            sequence=sequence,
            pos=mutation_position,
            mutant=mutant_aa,
            method=method,
        )
        if cached_result is not None:
            cached_result["cached"] = True
            return APICallResult(
                success=True,
                data=DDGunResult(**cached_result),
                cached=True,
                latency_ms=0.0,
                api_name="ddgun",
            )

        try:
            result = await self._call_api(sequence, mutation_position, mutant_aa, method)
            result["cached"] = False

            self.cache.set(result, "ddgun", sequence=sequence, pos=mutation_position, mutant=mutant_aa, method=method)

            latency_ms = (time.perf_counter() - start_time) * 1000
            return APICallResult(
                success=True,
                data=DDGunResult(**result),
                cached=False,
                latency_ms=latency_ms,
                api_name="ddgun",
            )
        except Exception as e:
            logger.warning(f"DDGun API failed: {e}, using empirical fallback")
            fallback_result = self._empirical_fallback(sequence, mutation_position, mutant_aa, method)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return APICallResult(
                success=True,
                data=fallback_result,
                error=str(e),
                cached=False,
                latency_ms=latency_ms,
                api_name="ddgun",
            )

    async def batch_predict_ddg(
        self,
        sequence: str,
        mutations: list[tuple[int, str]],
        method: str = "auto",
    ) -> list[APICallResult[DDGunResult]]:
        """Predict DDG for multiple mutations in parallel.

        Args:
            sequence: Protein sequence
            mutations: List of (position, mutant_aa) tuples
            method: Prediction method

        Returns:
            List of APICallResult for each mutation
        """
        tasks = [self.predict_ddg(sequence, pos, aa, method) for pos, aa in mutations]
        return await asyncio.gather(*tasks)

    async def _call_api(
        self,
        sequence: str,
        position: int,
        mutant: str,
        method: str,
    ) -> dict[str, Any]:
        """Call DDGun prediction API."""
        url = self._use_backup or self.BASE_URL

        payload = {
            "sequence": sequence,
            "mutation": f"{sequence[position - 1]}{position}{mutant}",
            "method": method,
        }

        if self.http_client is None:
            async with HTTPClient() as client:
                return await client.post(url, json_data=payload)
        else:
            return await self.http_client.post(url, json_data=payload)

    def _empirical_fallback(
        self,
        sequence: str,
        position: int,
        mutant: str,
        method: str,
    ) -> DDGunResult:
        """Empirical DDG estimation when API is unavailable.

        Uses position-specific amino acid substitution scores derived from
        statistical analysis of protein structures.
        """
        wildtype_aa = sequence[position - 1]

        # Simplified substitution scores (kcal/mol estimates)
        # Based on experimental data from ProTherm database
        substitution_scores: dict[tuple[str, str], float] = {
            # Stabilizing mutations (negative DDG)
            ("A", "V"): -0.5, ("A", "I"): -0.7, ("A", "L"): -0.8,
            ("S", "A"): -0.3, ("S", "T"): -0.2,
            ("T", "V"): -0.4, ("T", "I"): -0.5,
            ("V", "I"): -0.3, ("V", "L"): -0.4,
            ("M", "L"): -0.2, ("M", "I"): -0.3,
            ("K", "R"): -0.2, ("R", "K"): -0.1,
            ("E", "Q"): -0.1, ("Q", "E"): -0.1,
            # Destabilizing mutations (positive DDG)
            ("V", "A"): 0.5, ("I", "A"): 0.7, ("L", "A"): 0.8,
            ("A", "S"): 0.3, ("A", "T"): 0.4,
            ("P", "G"): 0.8, ("G", "P"): 1.0,
            ("D", "E"): 0.2, ("E", "D"): 0.4,
            ("N", "D"): 0.3, ("D", "N"): 0.3,
        }

        ddg = substitution_scores.get((wildtype_aa, mutant), 0.0)

        # Adjust for buried vs exposed positions (simplified)
        # Buried positions tend to have larger effects
        buried_score = abs(ddg) * 1.2
        ddg = ddg * 1.2 if ddg < 0 else buried_score

        # Determine effective method
        effective_method = "thermo" if method == "thermo" or method == "auto" else "deep"

        return DDGunResult(
            sequence=sequence,
            mutation_position=position,
            wildtype_aa=wildtype_aa,
            mutant_aa=mutant,
            ddg=round(ddg, 3),
            confidence=0.35,
            method=effective_method,
            cached=False,
        )


# --------------------------------------------------------------------------- #
# Molecular Dynamics Pre-heating Suggestions                                  #
# --------------------------------------------------------------------------- #


class MDPreHeatAdvisor:
    """Molecular dynamics pre-heating protocol advisor.

    Provides recommendations for MD simulation pre-heating protocols
    based on protein properties. This is a rule-based system that
    generates protocols without requiring external API calls.
    """

    # Recommended heating rates (K/ns) based on protein size
    HEATING_RATES = {
        "small": 1.0,      # < 200 residues
        "medium": 0.5,     # 200-500 residues
        "large": 0.25,     # 500-1000 residues
        "xlarge": 0.1,     # > 1000 residues
    }

    # Initial temperatures for different target thermostability
    INITIAL_TEMPS = {
        "mesophilic": 50.0,      # Start cooler for mesophilic proteins
        "thermophilic": 100.0,   # Start warmer for thermophilic
        "hyperthermophilic": 150.0,
    }

    def __init__(self, cache: APICache | None = None):
        self.cache = cache or get_cache()

    def get_heating_suggestion(
        self,
        protein_length: int,
        target_temperature_celsius: float,
        thermostability_profile: str = "mesophilic",
    ) -> MDPreHeatSuggestion:
        """Generate MD pre-heating protocol suggestion.

        Args:
            protein_length: Number of amino acid residues
            target_temperature_celsius: Target simulation temperature
            thermostability_profile: One of "mesophilic", "thermophilic", "hyperthermophilic"

        Returns:
            MDPreHeatSuggestion with detailed protocol parameters
        """
        cache_key = f"md_preheat_{protein_length}_{target_temperature_celsius}_{thermostability_profile}"
        cached = self.cache.get("md_preheat", cache_key)
        if cached:
            return MDPreHeatSuggestion(**cached)

        # Determine size category
        if protein_length < 200:
            size_cat = "small"
        elif protein_length < 500:
            size_cat = "medium"
        elif protein_length < 1000:
            size_cat = "large"
        else:
            size_cat = "xlarge"

        heating_rate = self.HEATING_RATES[size_cat]
        initial_temp = self.INITIAL_TEMPS.get(thermostability_profile, 50.0)
        target_temp_k = target_temperature_celsius + 273.15
        initial_temp_k = initial_temp + 273.15

        # Calculate heating time (use abs to handle case where target < initial)
        temp_diff = abs(target_temp_k - initial_temp_k)
        heating_time_ns = max(1.0, temp_diff / heating_rate)  # Minimum 1 ns

        # Estimate equilibration steps (100ps per step at 2fs timestep)
        equilibration_steps = max(1000, int(heating_time_ns * 10000))

        # Generate warnings
        warnings: list[str] = []
        if protein_length > 1000:
            warnings.append("Large protein - consider using implicit solvent for initial screening")
        if target_temperature_celsius > 100:
            warnings.append("High temperature simulation - verify force field parameters for temperature range")
        if thermostability_profile == "hyperthermophilic":
            warnings.append("Hyperthermophilic target - ensure membrane proteins have appropriate lipid environment")
        if heating_rate < 0.25:
            warnings.append("Slow heating rate recommended for large or complex structures")

        # Determine ionic concentration
        ionic_conc = 150.0  # mM NaCl (physiological)

        suggestion = MDPreHeatSuggestion(
            protein_length=protein_length,
            initial_temperature_k=round(initial_temp_k, 1),
            target_temperature_k=round(target_temp_k, 1),
            heating_rate_k_per_ns=heating_rate,
            equilibration_steps=equilibration_steps,
            ensemble="NPT",  # NPT recommended for most cases
            box_padding_angstrom=10.0,
            ionic_concentration_mm=ionic_conc,
            salt_type="NaCl",
            warnings=warnings,
        )

        self.cache.set(
            suggestion.__dict__,
            "md_preheat",
            cache_key,
            ttl_seconds=86400 * 7,  # Cache for 1 week
        )

        return suggestion

    def get_detailed_protocol(
        self,
        suggestion: MDPreHeatSuggestion,
        minimization_steps: int = 5000,
    ) -> list[dict[str, Any]]:
        """Generate a step-by-step MD protocol.

        Args:
            suggestion: MDPreHeatSuggestion from get_heating_suggestion
            minimization_steps: Number of energy minimization steps

        Returns:
            List of protocol steps with detailed parameters
        """
        protocol = [
            {
                "step": 1,
                "name": "Energy Minimization",
                "method": "Steepest Descent / Conjugate Gradient",
                "n_steps": minimization_steps,
                "temperature_k": 0,
                "ensemble": "NVE",
                "restraints": "all",
                "description": "Remove steric clashes and bad contacts",
            },
            {
                "step": 2,
                "name": "Heating",
                "method": "Langevin Dynamics",
                "n_steps": int((suggestion.target_temperature_k - suggestion.initial_temperature_k) / suggestion.heating_rate_k_per_ns * 1000),
                "temperature_start_k": suggestion.initial_temperature_k,
                "temperature_end_k": suggestion.target_temperature_k,
                "ensemble": "NVT",
                "restraints": "CA atoms",
                "description": f"Gradual heating from {suggestion.initial_temperature_k:.0f}K to {suggestion.target_temperature_k:.0f}K",
            },
            {
                "step": 3,
                "name": "Density Equilibration",
                "method": "Langevin Dynamics",
                "n_steps": 50000,
                "temperature_k": suggestion.target_temperature_k,
                "ensemble": "NPT",
                "restraints": "CA atoms",
                "description": "Adjust system density at target temperature",
            },
            {
                "step": 4,
                "name": "Production Equilibration",
                "method": "Langevin Dynamics / Molecular Dynamics",
                "n_steps": suggestion.equilibration_steps,
                "temperature_k": suggestion.target_temperature_k,
                "ensemble": suggestion.ensemble,
                "restraints": "none",
                "description": "Full system equilibration before production",
            },
        ]
        return protocol


# --------------------------------------------------------------------------- #
# Protein Expression Prediction API                                           #
# --------------------------------------------------------------------------- #


class ExpressionPredictorAPI:
    """Protein expression prediction API client.

    Predicts protein expression levels in various host systems based on
    sequence features including codon usage, mRNA stability, and signal peptides.

    Falls back to sequence-based heuristics when API is unavailable.
    """

    BASE_URL = "https://api.expressionpredictor.example.com/v1"

    def __init__(
        self,
        cache: APICache | None = None,
        http_client: HTTPClient | None = None,
    ):
        self.cache = cache or get_cache()
        self.http_client = http_client

    async def predict_expression(
        self,
        sequence: str,
        host_system: str = "E. coli",
    ) -> APICallResult[ExpressionPrediction]:
        """Predict protein expression level.

        Args:
            sequence: Protein amino acid sequence
            host_system: Expression host ("E. coli", "S. cerevisiae", "HEK293", etc.)

        Returns:
            APICallResult containing ExpressionPrediction with expression estimates
        """
        start_time = time.perf_counter()

        # Check cache
        cached_result = self.cache.get("expression", sequence=sequence, host=host_system)
        if cached_result is not None:
            cached_result["cached"] = True
            return APICallResult(
                success=True,
                data=ExpressionPrediction(**cached_result),
                cached=True,
                latency_ms=0.0,
                api_name="expression_predictor",
            )

        try:
            result = await self._call_api(sequence, host_system)
            result["cached"] = False

            self.cache.set(result, "expression", sequence=sequence, host=host_system)

            latency_ms = (time.perf_counter() - start_time) * 1000
            return APICallResult(
                success=True,
                data=ExpressionPrediction(**result),
                cached=False,
                latency_ms=latency_ms,
                api_name="expression_predictor",
            )
        except Exception as e:
            logger.warning(f"Expression predictor API failed: {e}, using heuristic fallback")
            fallback_result = self._heuristic_fallback(sequence, host_system)
            latency_ms = (time.perf_counter() - start_time) * 1000
            return APICallResult(
                success=True,
                data=fallback_result,
                error=str(e),
                cached=False,
                latency_ms=latency_ms,
                api_name="expression_predictor",
            )

    async def _call_api(self, sequence: str, host: str) -> dict[str, Any]:
        """Call expression prediction API."""
        if self.http_client is None:
            async with HTTPClient() as client:
                return await client.post(
                    f"{self.BASE_URL}/predict",
                    json_data={"sequence": sequence, "host": host},
                )
        else:
            return await self.http_client.post(
                f"{self.BASE_URL}/predict",
                json_data={"sequence": sequence, "host": host},
            )

    def _heuristic_fallback(self, sequence: str, host: str) -> ExpressionPrediction:
        """Heuristic-based expression prediction fallback.

        Uses sequence properties to estimate expression potential.
        """
        length = len(sequence)

        # Codon usage approximation (simplified)
        rare_codons = {"AGA", "CGA", "AGA", "UAU", "CUA"}  # E. coli rare codons
        rare_aa = {"R", "Y", "L", "H"}  # Amino acids with rare codons
        rare_count = sum(1 for aa in sequence if aa in rare_aa)
        codon_score = 1.0 - (rare_count / max(length, 1)) * 2
        codon_score = max(0.0, min(1.0, codon_score))

        # mRNA stability approximation (GC content proxy)
        gc_content = sum(1 for aa in sequence if aa in {"G", "A", "C", "E", "V", "L", "I"}) / max(length, 1)
        mrna_score = gc_content

        # Secretory signal detection (N-terminal patterns)
        n_terminal = sequence[:30] if length >= 30 else sequence
        signal_motifs = ["MKKT", "MLNK", "MKHL", "MLSF", "MRFA", "MKWV"]
        has_signal = any(n_terminal.startswith(motif[:4]) for motif in signal_motifs)
        signal_prob = 0.3 if has_signal else 0.1

        # Membrane protein detection
        hydrophobic = sum(1 for aa in n_terminal if aa in {"A", "V", "I", "L", "M", "F", "W"})
        membrane_prob = 0.8 if hydrophobic >= 10 else 0.2

        # Solubility prediction (simplified)
        charged = sum(1 for aa in sequence if aa in {"D", "E", "K", "R"})
        proline_count = sequence.count("P")
        glycine_count = sequence.count("G")
        solubility = (charged / max(length, 1)) * 2 - (proline_count / max(length, 1)) - (glycine_count / max(length, 1)) * 0.5
        solubility_score = max(0.0, min(1.0, solubility + 0.3))

        # Overall expression score
        expression_score = (codon_score * 0.4 + mrna_score * 0.2 + solubility_score * 0.4)
        if membrane_prob > 0.5:
            expression_score *= 0.7  # Membrane proteins harder to express
        expression_score = max(0.0, min(1.0, expression_score))

        # Categorize
        if expression_score >= 0.7:
            category = "high"
        elif expression_score >= 0.4:
            category = "medium"
        else:
            category = "low"

        # Recommendations
        recommendations: list[str] = []
        if codon_score < 0.5:
            recommendations.append("Consider codon optimization for improved expression")
        if membrane_prob > 0.5:
            recommendations.append("Use membrane protein-specific expression protocols")
        if solubility_score < 0.4:
            recommendations.append("Include solubility tags (MBP, GST, Trx) or use refolding protocols")
        if length > 1000:
            recommendations.append("Large protein - consider cell-free expression system")
        if signal_prob > 0.2:
            recommendations.append("Potential signal peptide detected - use secretion pathway host")

        return ExpressionPrediction(
            sequence=sequence,
            predicted_expression_level=round(expression_score, 3),
            expression_category=category,
            codon_usage_score=round(codon_score, 3),
            mrna_stability_score=round(mrna_score, 3),
            secretory_signal_probability=round(signal_prob, 3),
            membrane_protein=membrane_prob > 0.5,
            solubility_score=round(solubility_score, 3),
            confidence=0.4,  # Lower confidence for heuristic
            recommendations=recommendations,
            cached=False,
        )


# --------------------------------------------------------------------------- #
# Combined Stability Predictor                                                #
# --------------------------------------------------------------------------- #


class CombinedStabilityPredictor:
    """Combined stability predictor using multiple APIs.

    Aggregates predictions from ThermoNet, DDGun, and expression
    predictors to provide comprehensive stability analysis.
    """

    def __init__(
        self,
        thermonet: ThermoNetAPI | None = None,
        ddgun: DDGunAPI | None = None,
        expression: ExpressionPredictorAPI | None = None,
        md_advisor: MDPreHeatAdvisor | None = None,
        cache: APICache | None = None,
    ):
        self.cache = cache or get_cache()
        self.thermonet = thermonet or ThermoNetAPI(cache=self.cache)
        self.ddgun = ddgun or DDGunAPI(cache=self.cache)
        self.expression = expression or ExpressionPredictorAPI(cache=self.cache)
        self.md_advisor = md_advisor or MDPreHeatAdvisor(cache=self.cache)

    async def predict_all(
        self,
        sequence: str,
        target_temperature_celsius: float = 50.0,
        thermostability_profile: str = "mesophilic",
    ) -> dict[str, APICallResult]:
        """Run all stability predictions for a sequence.

        Args:
            sequence: Protein sequence
            target_temperature_celsius: Target temperature for MD simulations
            thermostability_profile: Thermostability profile for MD protocol

        Returns:
            Dictionary mapping predictor names to their results
        """
        # Run all predictions in parallel
        thermo_task = self.thermonet.predict_thermostability(sequence, target_temperature_celsius)
        expr_task = self.expression.predict_expression(sequence)

        thermo_result, expr_result = await asyncio.gather(thermo_task, expr_task)

        # MD suggestion is synchronous
        md_result = self.md_advisor.get_heating_suggestion(
            len(sequence),
            target_temperature_celsius,
            thermostability_profile,
        )

        return {
            "thermonet": thermo_result,
            "ddgun": None,  # DDGun requires specific mutations
            "expression": expr_result,
            "md_suggestion": md_result,
        }

    async def predict_mutation_stability(
        self,
        sequence: str,
        mutation_position: int,
        mutant_aa: str,
        target_temperature_celsius: float = 50.0,
    ) -> dict[str, APICallResult]:
        """Predict stability effects of a specific mutation.

        Args:
            sequence: Protein sequence
            mutation_position: Position of mutation (1-indexed)
            mutant_aa: Mutant amino acid
            target_temperature_celsius: Target temperature

        Returns:
            Dictionary with ThermoNet and DDGun mutation predictions
        """
        thermo_task = self.thermonet.predict_mutation_effect(
            sequence, mutation_position, mutant_aa, target_temperature_celsius
        )
        ddgun_task = self.ddgun.predict_ddg(sequence, mutation_position, mutant_aa)

        thermo_result, ddgun_result = await asyncio.gather(thermo_task, ddgun_task)

        return {
            "thermonet": thermo_result,
            "ddgun": ddgun_result,
        }


# --------------------------------------------------------------------------- #
# Convenience functions                                                       #
# --------------------------------------------------------------------------- #


async def predict_thermostability(
    sequence: str,
    temperature: float = 50.0,
    use_cache: bool = True,
) -> ThermoNetResult:
    """Convenience function for ThermoNet prediction.

    Args:
        sequence: Protein sequence
        temperature: Target temperature in Celsius
        use_cache: Whether to use cached results

    Returns:
        ThermoNetResult with stability prediction
    """
    predictor = ThermoNetAPI()
    if not use_cache:
        predictor.cache = APICache(default_ttl_seconds=0)
    result = await predictor.predict_thermostability(sequence, temperature)
    return result.data


async def predict_ddg(
    sequence: str,
    mutation_position: int,
    mutant_aa: str,
    method: str = "auto",
    use_cache: bool = True,
) -> DDGunResult:
    """Convenience function for DDGun prediction.

    Args:
        sequence: Protein sequence
        mutation_position: Mutation position (1-indexed)
        mutant_aa: Mutant amino acid
        method: Prediction method ("deep" or "thermo")
        use_cache: Whether to use cached results

    Returns:
        DDGunResult with DDG prediction
    """
    predictor = DDGunAPI()
    if not use_cache:
        predictor.cache = APICache(default_ttl_seconds=0)
    result = await predictor.predict_ddg(sequence, mutation_position, mutant_aa, method)
    return result.data


def get_md_protocol(
    protein_length: int,
    target_temperature_celsius: float,
    thermostability_profile: str = "mesophilic",
) -> MDPreHeatSuggestion:
    """Convenience function for MD pre-heating protocol.

    Args:
        protein_length: Number of residues
        target_temperature_celsius: Target temperature
        thermostability_profile: Thermostability profile

    Returns:
        MDPreHeatSuggestion with protocol parameters
    """
    advisor = MDPreHeatAdvisor()
    return advisor.get_heating_suggestion(protein_length, target_temperature_celsius, thermostability_profile)


async def predict_expression(
    sequence: str,
    host_system: str = "E. coli",
    use_cache: bool = True,
) -> ExpressionPrediction:
    """Convenience function for expression prediction.

    Args:
        sequence: Protein sequence
        host_system: Expression host
        use_cache: Whether to use cached results

    Returns:
        ExpressionPrediction with expression estimates
    """
    predictor = ExpressionPredictorAPI()
    if not use_cache:
        predictor.cache = APICache(default_ttl_seconds=0)
    result = await predictor.predict_expression(sequence, host_system)
    return result.data
