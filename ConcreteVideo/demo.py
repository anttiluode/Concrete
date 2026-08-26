from __future__ import annotations

import argparse
import copy
import time

import torch
from torch import nn

from concrete_video import CacheConfig, ConcreteVideoController


class ExpensiveBlock(nn.Module):
    def __init__(self, channels: int, repeats: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1) for _ in range(repeats)]
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.act(layer(x))
        return x


class TinyVideoNet(nn.Module):
    def __init__(self, channels: int = 24):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.block1 = ExpensiveBlock(channels)
        self.block2 = ExpensiveBlock(channels)
        self.head = nn.Conv2d(channels, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


def frames(n: int, size: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, 3, size, size, generator=g) * 0.15
    for i in range(n):
        frame = base.clone()
        # Most pixels remain stable. A small patch moves and brightness drifts slowly.
        y = (3 * i) % max(1, size - 12)
        x = (2 * i) % max(1, size - 12)
        frame[:, :, y : y + 12, x : x + 12] += 0.7
        frame += 0.002 * i
        yield frame


def timed_run(model: nn.Module, xs: list[torch.Tensor]) -> tuple[list[torch.Tensor], float]:
    outs = []
    start = time.perf_counter()
    with torch.inference_mode():
        for x in xs:
            outs.append(model(x))
    return outs, time.perf_counter() - start


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm((a - b).float())
        / (torch.linalg.vector_norm(a.float()) + 1e-8)
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("--audit", type=float, default=0.10)
    args = p.parse_args()

    torch.manual_seed(0)
    baseline = TinyVideoNet().eval()
    cached = copy.deepcopy(baseline).eval()
    xs = list(frames(args.frames, args.size))

    # Warm CPU kernels before measuring.
    with torch.inference_mode():
        baseline(xs[0])
        cached(xs[0])

    truth, baseline_s = timed_run(baseline, xs)

    cfg = CacheConfig(
        output_tolerance=args.tolerance,
        min_observations=5,
        audit_rate=args.audit,
        safety_margin=1.35,
        quantile=0.90,
        seed=7,
    )
    controller = ConcreteVideoController(cached).wrap(["block1", "block2"], cfg)
    approx, cached_s = timed_run(cached, xs)

    errors = [rel_error(t, a) for t, a in zip(truth, approx)]
    print(f"baseline wall time: {baseline_s:.3f}s")
    print(f"cached wall time:   {cached_s:.3f}s")
    print(f"wall speedup:       {baseline_s / cached_s:.2f}x")
    print(f"mean output error:  {sum(errors)/len(errors):.5f}")
    print(f"max output error:   {max(errors):.5f}")
    controller.print_report()


if __name__ == "__main__":
    main()
