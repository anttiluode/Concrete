from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import nn


@dataclass
class CacheConfig:
    """Policy for one adaptive module cache.

    ``output_tolerance`` is relative RMS drift on a small deterministic tensor
    sketch. The cache learns a conservative local gain from input drift to output
    drift and periodically audits predicted-safe reuses.
    """

    output_tolerance: float = 0.01
    min_observations: int = 6
    history: int = 64
    quantile: float = 0.90
    safety_margin: float = 1.5
    audit_rate: float = 0.10
    min_audit_rate: float = 0.02
    failure_boost: float = 0.25
    sketch_size: int = 512
    seed: int = 0
    eps: float = 1e-8


@dataclass
class CacheStats:
    calls: int = 0
    executed: int = 0
    reused: int = 0
    audits: int = 0
    unsafe_audits: int = 0
    observed_seconds: float = 0.0
    estimated_seconds_saved: float = 0.0
    last_input_drift: float = math.inf
    last_output_drift: float = math.inf
    learned_gain: float = math.inf

    @property
    def reuse_rate(self) -> float:
        return self.reused / self.calls if self.calls else 0.0

    @property
    def unsafe_audit_rate(self) -> float:
        return self.unsafe_audits / self.audits if self.audits else 0.0

    @property
    def mean_execution_seconds(self) -> float:
        return self.observed_seconds / self.executed if self.executed else 0.0

    def to_dict(self) -> dict[str, float | int]:
        out = asdict(self)
        out.update(
            reuse_rate=self.reuse_rate,
            unsafe_audit_rate=self.unsafe_audit_rate,
            mean_execution_seconds=self.mean_execution_seconds,
        )
        return out


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _iter_tensors(value[key])


def _structure(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), str(value.device))
    if isinstance(value, tuple):
        return ("tuple", tuple(_structure(v) for v in value))
    if isinstance(value, list):
        return ("list", tuple(_structure(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple((k, _structure(value[k])) for k in sorted(value)))
    if isinstance(value, (str, int, float, bool, type(None))):
        return ("scalar", value)
    return ("object", type(value).__qualname__, repr(value))


def _clone_detached(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_detached(v) for v in value)
    if isinstance(value, list):
        return [_clone_detached(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_detached(v) for k, v in value.items()}
    return value


def _sample_tensor(x: torch.Tensor, limit: int) -> torch.Tensor:
    flat = x.detach().reshape(-1)
    if flat.numel() <= limit:
        return flat.float().clone()
    step = max(1, flat.numel() // limit)
    return flat[::step][:limit].float().clone()


def _sketch(value: Any, limit: int) -> tuple[torch.Tensor, ...]:
    tensors = list(_iter_tensors(value))
    if not tensors:
        return ()
    per_tensor = max(8, limit // len(tensors))
    return tuple(_sample_tensor(t, per_tensor) for t in tensors)


def _relative_rms_sketch(
    a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...], eps: float
) -> float:
    if len(a) != len(b):
        return math.inf
    if not a:
        return 0.0

    num = 0.0
    den = 0.0
    count = 0
    for x, y in zip(a, b):
        if x.shape != y.shape or x.device != y.device:
            return math.inf
        diff = x - y
        num += float(torch.sum(diff * diff).item())
        den += float(torch.sum(y * y).item())
        count += y.numel()
    if count == 0:
        return 0.0
    rms_diff = math.sqrt(num / count)
    rms_ref = math.sqrt(den / count)
    return rms_diff / (rms_ref + eps)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.inf
    xs = sorted(values)
    q = min(1.0, max(0.0, q))
    idx = q * (len(xs) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return xs[lo]
    frac = idx - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


class AdaptiveModuleCache(nn.Module):
    """Inference-time wrapper that learns when a module can be reused.

    This is empirical approximate computation, not a correctness proof. A fraction
    of predicted-safe cache hits are audited by executing the wrapped module anyway.
    Unsafe audits become evidence that makes later reuse more cautious.
    """

    def __init__(self, module: nn.Module, config: CacheConfig | None = None, name: str = ""):
        super().__init__()
        self.module = module
        self.config = config or CacheConfig()
        self.name = name or module.__class__.__name__
        self.stats = CacheStats()
        self._rng = random.Random(self.config.seed)
        self._gains: deque[float] = deque(maxlen=self.config.history)
        self._cached_input_sketch: tuple[torch.Tensor, ...] | None = None
        self._cached_output: Any = None
        self._cached_output_sketch: tuple[torch.Tensor, ...] | None = None
        self._cached_structure: Any = None
        self._has_cache = False

    def reset(self, keep_learning: bool = True) -> None:
        self._cached_input_sketch = None
        self._cached_output = None
        self._cached_output_sketch = None
        self._cached_structure = None
        self._has_cache = False
        if not keep_learning:
            self._gains.clear()
            self.stats = CacheStats()

    def learned_gain(self) -> float:
        if len(self._gains) < self.config.min_observations:
            return math.inf
        gain = _quantile(list(self._gains), self.config.quantile) * self.config.safety_margin
        return max(0.0, gain)

    def current_audit_rate(self) -> float:
        base = max(self.config.min_audit_rate, self.config.audit_rate)
        return min(1.0, base + self.config.failure_boost * self.stats.unsafe_audit_rate)

    def _record_observation(self, input_drift: float, output_drift: float) -> None:
        self.stats.last_input_drift = input_drift
        self.stats.last_output_drift = output_drift
        if math.isfinite(input_drift) and math.isfinite(output_drift):
            if input_drift <= self.config.eps:
                gain = 0.0 if output_drift <= self.config.output_tolerance else math.inf
            else:
                gain = output_drift / input_drift
            if math.isfinite(gain):
                self._gains.append(gain)
        self.stats.learned_gain = self.learned_gain()

    def _execute(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, float]:
        start = time.perf_counter()
        output = self.module(*args, **kwargs)
        tensors = list(_iter_tensors(output))
        if not tensors:
            raise RuntimeError(f"ConcreteVideo currently requires tensor output from {self.name}")
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            raise RuntimeError(
                f"ConcreteVideo cache {self.name!r} is inference-only. "
                "Run under torch.inference_mode() or torch.no_grad()."
            )
        elapsed = time.perf_counter() - start
        self.stats.executed += 1
        self.stats.observed_seconds += elapsed
        return output, elapsed

    def _refresh_cache(
        self,
        inputs: Any,
        structure: Any,
        output: Any,
        input_sketch: tuple[torch.Tensor, ...] | None = None,
        output_sketch: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        self._cached_input_sketch = input_sketch or _sketch(inputs, self.config.sketch_size)
        self._cached_output = _clone_detached(output)
        self._cached_output_sketch = output_sketch or _sketch(output, self.config.sketch_size)
        self._cached_structure = structure
        self._has_cache = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.stats.calls += 1
        inputs = (args, kwargs)
        structure = _structure(inputs)

        if not self._has_cache or structure != self._cached_structure:
            output, _ = self._execute(args, kwargs)
            self._refresh_cache(inputs, structure, output)
            return output

        current_input_sketch = _sketch(inputs, self.config.sketch_size)
        input_drift = _relative_rms_sketch(
            current_input_sketch, self._cached_input_sketch or (), self.config.eps
        )
        gain = self.learned_gain()
        predicted_output_drift = input_drift * gain
        eligible = math.isfinite(predicted_output_drift) and (
            predicted_output_drift <= self.config.output_tolerance
        )

        if eligible and self._rng.random() >= self.current_audit_rate():
            self.stats.reused += 1
            self.stats.last_input_drift = input_drift
            self.stats.last_output_drift = predicted_output_drift
            self.stats.learned_gain = gain
            self.stats.estimated_seconds_saved += self.stats.mean_execution_seconds
            return self._cached_output

        if eligible:
            self.stats.audits += 1

        old_output_sketch = self._cached_output_sketch or ()
        output, _ = self._execute(args, kwargs)
        current_output_sketch = _sketch(output, self.config.sketch_size)
        output_drift = _relative_rms_sketch(
            current_output_sketch, old_output_sketch, self.config.eps
        )
        self._record_observation(input_drift, output_drift)

        if eligible and output_drift > self.config.output_tolerance:
            self.stats.unsafe_audits += 1

        # We paid the compute, so use the fresh result and make it the new reference.
        self._refresh_cache(
            inputs,
            structure,
            output,
            input_sketch=current_input_sketch,
            output_sketch=current_output_sketch,
        )
        return output


class ConcreteVideoController:
    """Replace named submodules of an existing model with adaptive caches."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.wrappers: dict[str, AdaptiveModuleCache] = {}

    def wrap(
        self, names: Iterable[str], config: CacheConfig | None = None
    ) -> "ConcreteVideoController":
        for i, name in enumerate(names):
            if name in self.wrappers:
                continue
            parent, leaf = self._parent_and_leaf(name)
            original = getattr(parent, leaf)
            if not isinstance(original, nn.Module):
                raise TypeError(f"{name!r} is not an nn.Module")
            cfg = config or CacheConfig()
            # Decorrelation keeps block audits reproducible without synchronizing them.
            cfg = CacheConfig(**{**asdict(cfg), "seed": cfg.seed + i})
            wrapper = AdaptiveModuleCache(original, config=cfg, name=name)
            setattr(parent, leaf, wrapper)
            self.wrappers[name] = wrapper
        return self

    def _parent_and_leaf(self, dotted: str) -> tuple[nn.Module, str]:
        parts = dotted.split(".")
        if not parts or not all(parts):
            raise ValueError("invalid module name")
        parent: nn.Module = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
            if not isinstance(parent, nn.Module):
                raise TypeError(f"{'.'.join(parts[:-1])!r} is not an nn.Module")
        return parent, parts[-1]

    def reset(self, keep_learning: bool = True) -> None:
        for wrapper in self.wrappers.values():
            wrapper.reset(keep_learning=keep_learning)

    def report(self) -> dict[str, dict[str, float | int]]:
        return {name: wrapper.stats.to_dict() for name, wrapper in self.wrappers.items()}

    def print_report(self) -> None:
        total_calls = total_exec = total_reuse = total_audits = total_unsafe = 0
        total_observed = total_saved = 0.0
        print("ConcreteVideo report")
        for name, wrapper in self.wrappers.items():
            s = wrapper.stats
            total_calls += s.calls
            total_exec += s.executed
            total_reuse += s.reused
            total_audits += s.audits
            total_unsafe += s.unsafe_audits
            total_observed += s.observed_seconds
            total_saved += s.estimated_seconds_saved
            print(
                f"  {name}: calls={s.calls} exec={s.executed} reuse={s.reused} "
                f"reuse={s.reuse_rate:.1%} audits={s.audits} unsafe={s.unsafe_audits} "
                f"gain={s.learned_gain:.4g} est_saved={s.estimated_seconds_saved:.3f}s"
            )
        print(
            f"  TOTAL: calls={total_calls} exec={total_exec} reuse={total_reuse} "
            f"audits={total_audits} unsafe={total_unsafe} "
            f"observed_exec={total_observed:.3f}s est_saved={total_saved:.3f}s"
        )
