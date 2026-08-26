# ConcreteVideo

**Spend GPU work only where the current frame has made old computation invalid. Audit yourself often enough to notice when that belief is wrong.**

`ConcreteVideo` is an inference-time PyTorch experiment for adaptive computation reuse across slowly changing inputs such as video frames, iterative image pipelines, and repeated AI calls over nearby states.

It is not a model and it does not retrain one. It wraps expensive `nn.Module`s inside an existing model.

For every wrapped block it learns an empirical relationship:

```text
input drift  --->  output drift
```

If the current input changed so little that the learned envelope predicts the block output will remain within a requested tolerance, the block can reuse its previous output instead of executing.

Crucially, reuse is never granted permanent authority. A fraction of predicted-safe calls are **audited** by recomputing the block anyway. If an audit finds that the cached answer would have crossed the tolerance, that miss becomes new evidence and future reuse becomes more cautious.

```text
new frame
   |
   v
input changed how much?
   |
   +-- learned envelope says output can matter --> execute block --> refresh cache
   |
   `-- predicted below tolerance
             |
             +-- most calls: reuse cached block output
             |
             `-- audit calls: execute anyway
                                |
                                +-- safe --> refresh evidence/cache
                                `-- unsafe --> record miss, become more cautious
```

This is the Concrete idea applied directly to AI compute: **compile repeated validity into cheap persistent structure, but leave enough exploration for the world to prove the structure wrong.**

## What exists now

The first version is intentionally model-agnostic:

- `AdaptiveModuleCache`: wraps one PyTorch module;
- `ConcreteVideoController`: replaces named submodules in an existing model;
- cheap sampled relative-RMS sketches for input and output drift;
- online learned input→output gain envelope;
- configurable quality tolerance and safety margin;
- random safety audits with increased audit pressure after misses;
- per-block counters for executions, reuses, audits, unsafe audits, observed execution time and estimated saved time;
- a toy video benchmark that compares exact and cached execution frame by frame.

It is **inference only**. Use `torch.inference_mode()` or `torch.no_grad()`.

## Run the benchmark

From this directory:

```bash
python -m pip install -r requirements.txt
python demo.py
```

Try a more permissive tolerance:

```bash
python demo.py --frames 120 --size 128 --tolerance 0.03 --audit 0.10
```

The output reports wall time, output error, reuse rate and unsafe audits. A useful result is not merely a high reuse percentage. The actual target is:

```text
large wall/GPU compute reduction
subject to
small end-to-end output error
and
few unsafe audits
```

## Wrap part of a real model

```python
import torch
from concrete_video import CacheConfig, ConcreteVideoController

model = load_your_model().eval()

controller = ConcreteVideoController(model).wrap(
    [
        "transformer.blocks.4",
        "transformer.blocks.5",
        "transformer.blocks.6",
    ],
    CacheConfig(
        output_tolerance=0.01,
        audit_rate=0.10,
        safety_margin=1.5,
    ),
)

with torch.inference_mode():
    for frame in frames:
        output = model(frame)

controller.print_report()
```

The dotted names are ordinary PyTorch submodule paths. Start with a few genuinely expensive blocks rather than wrapping every tiny operation; measuring/caching cheap blocks can cost more than simply executing them.

## How the current learner works

For the previous cached input `x0` and current input `x1`, ConcreteVideo measures relative RMS drift `dx` on a small deterministic activation sketch rather than the full tensor. When it executes the module, it also measures output drift `dy` relative to the previous cached output.

Each observation supplies a local gain estimate:

```text
g = dy / dx
```

The cache keeps a bounded history of gains. Its predictor uses a high quantile multiplied by a safety margin:

```text
predicted_dy = dx * quantile(g_history) * safety_margin
```

Reuse is eligible only when:

```text
predicted_dy <= output_tolerance
```

This is deliberately simple and attackable. It is not claiming neural activations are globally Lipschitz with one scalar gain. The audit loop exists precisely because that approximation will sometimes be wrong.

## Where the compute can disappear

The sweet spot is a repeated expensive computation whose input changes slowly:

- webcam/video-to-video pipelines;
- diffusion/video transformer blocks across nearby frames;
- iterative denoising blocks across nearby states;
- repeated feature extraction from mostly stable scenes;
- agents/models that repeatedly process a stable large representation plus a small changing frontier.

The bad targets are cheap layers, highly discontinuous blocks, training passes, and calls whose tensor shapes/control arguments change constantly.

## Current boundaries

This is a first concrete instrument, not a production accelerator.

1. **No correctness guarantee.** The learned envelope is empirical. Always measure end-to-end quality.
2. **Cache memory can be large.** Cached activations live on the tensor's device. Wrapping many high-resolution blocks can consume VRAM quickly.
3. **Drift measurement itself costs compute.** The current detector samples a bounded number of activation values (`sketch_size`, default 512) rather than scanning full tensors, but GPU synchronization and sketch traffic are still real overhead.
4. **Single previous state.** It caches one prior input/output per block, not a bank keyed by scene/state.
5. **No diffusion-specific adapter yet.** Time-step, conditioning and attention-cache semantics need explicit treatment before claiming useful Wan/SD speedups.
6. **No CUDA event accounting yet.** `perf_counter` is sufficient for the CPU toy, but serious GPU benchmarking needs synchronization/CUDA events.
7. **Audits spend the compute they audit.** That is intentional; their value is epistemic, not immediate speed.

## The next real test

The benchmark only proves the mechanism behaves. The first valuable experiment is to wrap a real local image/video network and measure:

```text
baseline GPU-seconds / useful frame
ConcreteVideo GPU-seconds / useful frame
VRAM overhead
end-to-end perceptual/output error
unsafe audit rate
```

If it cannot beat ordinary full execution after counting drift checks, cache memory and audits, it crumbles.

## Reference toy result

A local CPU sanity run used 20 slowly changing 48×48 frames and two wrapped convolutional blocks:

```text
baseline wall time: 0.087s
cached wall time:   0.051s
wall speedup:       1.70x
mean output error:  ~1e-5
wrapped calls:      40
executed:           21
reused:             19
unsafe audits:      0
```

That is only a mechanism check, not evidence of diffusion-model acceleration. GPU/video-model results are the real target.
