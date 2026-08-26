import torch
from torch import nn

from concrete_video import AdaptiveModuleCache, CacheConfig, ConcreteVideoController


class Scale(nn.Module):
    def __init__(self, factor=2.0):
        super().__init__()
        self.factor = factor
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x * self.factor


def test_cache_learns_and_reuses_slow_change():
    inner = Scale()
    cache = AdaptiveModuleCache(
        inner,
        CacheConfig(
            output_tolerance=0.05,
            min_observations=2,
            audit_rate=0.0,
            min_audit_rate=0.0,
            safety_margin=1.0,
            quantile=1.0,
        ),
    )
    xs = [torch.ones(8) * (1.0 + i * 0.005) for i in range(8)]
    with torch.inference_mode():
        for x in xs:
            cache(x)
    assert cache.stats.reused > 0
    assert inner.calls < len(xs)


def test_structure_change_forces_execution():
    inner = Scale()
    cache = AdaptiveModuleCache(inner, CacheConfig(min_observations=1))
    with torch.inference_mode():
        cache(torch.ones(4))
        cache(torch.ones(5))
    assert inner.calls == 2


def test_unsafe_audit_is_counted_and_actual_output_returned():
    class Jump(nn.Module):
        def forward(self, x):
            # Tiny input changes near 1 cross a sharp output boundary.
            return (x > 1.0).float() * 10.0

    cache = AdaptiveModuleCache(
        Jump(),
        CacheConfig(
            output_tolerance=0.1,
            min_observations=1,
            audit_rate=1.0,
            min_audit_rate=1.0,
            safety_margin=0.0,
            quantile=1.0,
        ),
    )
    with torch.inference_mode():
        cache(torch.tensor([0.99]))
        # Teach zero gain below threshold.
        cache(torch.tensor([0.995]))
        y = cache(torch.tensor([1.005]))
    assert float(y.item()) == 10.0
    assert cache.stats.audits >= 1
    assert cache.stats.unsafe_audits >= 1


def test_controller_wraps_named_submodule():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
            self.b = nn.Linear(4, 1)

        def forward(self, x):
            return self.b(self.a(x))

    m = Model().eval()
    c = ConcreteVideoController(m).wrap(["a"])
    assert isinstance(m.a, AdaptiveModuleCache)
    assert "a" in c.report()
