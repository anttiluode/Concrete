# Concrete

**Run the tests that can still matter. Keep a few that supposedly cannot. Learn when you were wrong.**

Concrete is a small adaptive test selector for Python/pytest projects.

## ConcreteVideo: the AI-compute branch

`ConcreteVideo/` applies the same rule directly inside PyTorch inference: wrap expensive modules, learn how input drift predicts output drift, reuse cached activations when the predicted consequence is below a tolerance, and deliberately audit some predicted-safe reuses so the model can discover when its validity map is wrong.

The first toy video benchmark is already executable. It removed 19 of 40 wrapped block executions and measured about 1.70x CPU wall-clock speedup at roughly `1e-5` mean end-to-end output error. That is only a mechanism check; the real target is a local diffusion/video model where skipped blocks mean saved GPU-seconds.

See [`ConcreteVideo/README.md`](ConcreteVideo/README.md).

A normal test selector needs a dependency graph supplied by the build system or a static guess from file names. Concrete learns a receiver-relative graph from actual executions:

```text
changed file
     |
     v
learned file -> test execution edges
     |
     +---- tests whose observed computation touched the change
     |
     `---- small exploration reserve outside the known frontier
```

The first full test run is expensive on purpose. It compiles experience into a persistent local map. Later changes can use that map to avoid waking the whole suite.

The exploration reserve is load-bearing. A learned dependency map can be wrong because code paths change, imports are dynamic, tests are flaky, or a new dependency did not exist when the map was learned. Concrete always has the option to spend part of its finite budget outside the current map and then update the map from what actually ran.

This is a useful tool, not a research gate sequence.

## Install

From a checkout:

```bash
python -m pip install -e '.[dev]'

# or directly from GitHub in another project
python -m pip install 'git+https://github.com/anttiluode/Concrete.git'
```

Concrete itself has no runtime dependency beyond Python's standard library. `pytest` is required for the current adapter.

## 1. Teach it once

Run your normal pytest suite through Concrete:

```bash
concrete learn -- pytest -q
```

Concrete instruments each test while it runs and stores a compact map in:

```text
.concrete/state.sqlite3
```

It records:

- which project Python files each test actually executed;
- pass/fail/skip observations;
- mean observed duration;
- outcome flips, used as a simple test-reliability signal.

The map is local state. Commit it only if that makes sense for your workflow; `.concrete/` is ignored by default.

## 2. Change code

Edit your project normally. Concrete reads the working-tree/staged/untracked file set from Git.

Ask what it would run under a 12-test budget:

```bash
concrete plan --budget 12
```

Example shape:

```text
Changed: pkg/parser.py
Selected 5 test(s):
  [RUN    ] tests/test_parser.py::test_nested  score=14.2 trust=1.00 mean=0.031s
             executed pkg/parser.py (3 observed runs)
  [RUN    ] tests/test_api.py::test_upload     score=13.4 trust=1.00 mean=0.052s
             executed pkg/parser.py (2 observed runs)
  [EXPLORE] tests/test_export.py::test_csv     score= 0.0 trust=1.00 mean=0.010s
             exploration reserve: tests the learned map's blind spots
```

You can also declare changed files explicitly:

```bash
concrete plan --changed pkg/parser.py --changed pkg/model.py --budget 12
```

Or compare a branch/base against `HEAD`:

```bash
concrete plan --base origin/main --budget 20
```

## 3. Run only that frontier

```bash
concrete run --budget 12 -- -q
```

Concrete selects the tests, runs them through pytest, traces them again, and updates the learned graph from the new execution.

If there is no learned state yet, `concrete run` fails safe: it runs the full suite and learns it.

## Why not trust the learned map completely?

Because old structure makes execution cheap **and** can make a system blind.

Concrete therefore splits a finite test budget into:

```text
exploitation
    run tests with learned or path-based evidence that they can be affected

exploration
    run a few tests outside the current causal frontier
```

Default exploration is 15% of the budget:

```bash
concrete run --budget 20 --exploration 0.15 -- -q
```

Set it to zero if your build graph is already trustworthy:

```bash
concrete run --budget 20 --exploration 0 -- -q
```

Set `--budget 0` to run all learned tests while still using Concrete's tracing/state bookkeeping.

## Reliability instead of blind surprise

A failing test is evidence, but intermittent evidence should not be allowed to dominate a learned selector forever.

Concrete tracks whether a test's outcome flips across observations:

```bash
concrete status
concrete explain tests/test_parser.py::test_nested
```

`explain` also prints the test's learned execution footprint, so selection is inspectable rather than magical.

The current reliability model is deliberately small: `1 - outcome_flip_rate`. It is not a full flaky-test classifier. The point is to keep source trust separate from dependency evidence from the beginning.

## What this actually buys

Concrete is aimed at repositories where:

- the test suite is materially larger than the causal footprint of a typical change;
- dependencies are dynamic enough that a hand-maintained graph is annoying;
- repeated CI history is available to amortize future work;
- running a small safety reserve is cheaper than running everything.

The useful invariant is:

> **A large codebase can change globally while most individual tests remain valid. The scalable problem is to find the few invalidated receivers without scanning or executing all of them.**

A full suite is the boring attacker and the safety ceiling. Concrete only earns its existence if the saved test time is larger than tracing, planning, misses, and exploration.

## Current boundaries

Version 0.1.1 is intentionally narrow:

- Python project files only;
- pytest adapter only;
- execution tracing uses Python's tracing hook, so mapping runs are slower than ordinary pytest;
- native-extension, subprocess, external-service and generated-file dependencies are not yet observed directly;
- per-test execution edges are evidence of participation, not proof of semantic causality;
- pytest-xdist/parallel mapping is not supported yet; learn with ordinary serial pytest;
- selection is local-state based; there is no hosted service or shared fleet model.

That is enough for the first real use: **learn on a full local/CI run, then cheaply select a bounded regression frontier for subsequent changes.**

## Commands

```text
concrete learn [--] pytest ...        full run + learn execution map
concrete plan [options]               inspect selected frontier
concrete run [options] -- pytestargs  select + execute + relearn
concrete status                       summarize learned state
concrete explain NODEID               inspect one test's learned footprint
```

## Design rule

Concrete is built around one rule:

> **Compile repeated causal relevance into persistent local structure, but reserve enough work for the world to prove that structure wrong.**

If that rule does not reduce real CI cost on real repositories, Concrete crumbles.
