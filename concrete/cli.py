from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

from .select import Selection, plan_tests
from .state import StateDB


def _strip_separator(parts: list[str]) -> list[str]:
    return parts[1:] if parts and parts[0] == "--" else parts


def _pytest_command(command: list[str], plugin: bool) -> list[str]:
    if not command:
        command = [sys.executable, "-m", "pytest", "-q"]
    command = list(command)
    if not plugin:
        return command

    base = Path(command[0]).name.lower()

    # A bare `pytest` command is a console-script launcher. On Windows in
    # particular it can have different sys.path/import behaviour, or even
    # belong to a different Python installation than the one running
    # Concrete. Normalize it to this interpreter.
    if "pytest" in base or base.startswith("py.test"):
        return [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "concrete.pytest_plugin",
            *command[1:],
        ]

    if base.startswith("python") and len(command) >= 3 and command[1:3] == ["-m", "pytest"]:
        return [command[0], "-m", "pytest", "-p", "concrete.pytest_plugin", *command[3:]]

    raise SystemExit(
        "Concrete's learner currently instruments pytest. Use e.g. "
        "`concrete learn -- pytest -q` or `concrete learn -- python -m pytest -q`."
    )


def _run_traced(command: list[str], db_path: Path) -> tuple[int, int]:
    root = Path.cwd().resolve()
    trace_dir = db_path.parent
    trace_dir.mkdir(parents=True, exist_ok=True)
    fd, trace_name = tempfile.mkstemp(prefix="trace-", suffix=".jsonl", dir=trace_dir)
    os.close(fd)
    trace_path = Path(trace_name)
    env = os.environ.copy()
    env["CONCRETE_ROOT"] = str(root)
    env["CONCRETE_TRACE_PATH"] = str(trace_path)

    # Preserve ordinary "run from the repository root" import semantics even
    # for projects that import root-level modules directly from tests.
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(root) if not old_pythonpath else str(root) + os.pathsep + old_pythonpath
    )

    cmd = _pytest_command(command, plugin=True)
    print("Concrete >", shlex.join(cmd))
    result = subprocess.run(cmd, env=env)
    with StateDB(db_path) as db:
        ingested = db.ingest_jsonl(trace_path)
    try:
        trace_path.unlink()
    except OSError:
        pass
    return result.returncode, ingested


def _git_lines(args: list[str]) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]


def discover_changed_files(base: str | None = None) -> list[str]:
    changed: set[str] = set()
    if base:
        changed.update(_git_lines(["diff", "--name-only", f"{base}...HEAD"]))
    else:
        changed.update(_git_lines(["diff", "--name-only", "HEAD"]))
        changed.update(_git_lines(["diff", "--cached", "--name-only"]))
        changed.update(_git_lines(["ls-files", "--others", "--exclude-standard"]))
    return sorted(changed)


def _resolve_changed(explicit: list[str], base: str | None) -> list[str]:
    if explicit:
        return sorted({x.replace("\\", "/") for x in explicit})
    return discover_changed_files(base)


def _render_plan(plan: list[Selection], changed: list[str], suite_mean: float | None = None) -> None:
    print("Changed:", ", ".join(changed) if changed else "(none detected)")
    if not plan:
        print("No learned tests yet. Run `concrete learn -- pytest -q` first.")
        return
    selected_mean = sum((s.stats.mean_duration if s.stats else 0.0) for s in plan)
    if suite_mean and suite_mean > 0:
        fraction = selected_mean / suite_mean
        print(
            f"Selected {len(plan)} test(s); observed-time estimate "
            f"{selected_mean:.3f}s / {suite_mean:.3f}s ({fraction:.1%})."
        )
    else:
        print(f"Selected {len(plan)} test(s):")
    for sel in plan:
        tag = "EXPLORE" if sel.exploratory else "RUN"
        reliability = sel.stats.reliability if sel.stats else 1.0
        duration = sel.stats.mean_duration if sel.stats else 0.0
        reason = "; ".join(sel.reasons[:3]) or "highest remaining learned risk"
        print(
            f"  [{tag:7}] {sel.nodeid}  score={sel.score:5.1f} "
            f"trust={reliability:0.2f} mean={duration:0.3f}s\n"
            f"           {reason}"
        )


def cmd_learn(args: argparse.Namespace) -> int:
    command = _strip_separator(args.command)
    code, ingested = _run_traced(command, Path(args.state))
    if code == 0:
        print(f"Concrete learned {ingested} test observation(s).")
    elif ingested:
        print(
            f"Concrete: pytest exited with code {code}; retained {ingested} "
            "observation(s) from tests that actually ran.",
            file=sys.stderr,
        )
    else:
        print(
            f"Concrete: pytest exited with code {code} before any tests ran; learned nothing.",
            file=sys.stderr,
        )
    return code


def _make_plan(args: argparse.Namespace) -> tuple[list[Selection], list[str], float]:
    changed = _resolve_changed(args.changed, args.base)
    with StateDB(args.state) as db:
        plan = plan_tests(
            db,
            changed_files=changed,
            budget=args.budget,
            exploration=args.exploration,
            seed=args.seed,
        )
        suite_mean = float(db.counts()["suite_mean"])
    return plan, changed, suite_mean


def cmd_plan(args: argparse.Namespace) -> int:
    plan, changed, suite_mean = _make_plan(args)
    if args.json:
        print(
            json.dumps(
                {
                    "changed": changed,
                    "estimated_selected_seconds": round(
                        sum((s.stats.mean_duration if s.stats else 0.0) for s in plan), 6
                    ),
                    "estimated_full_seconds": round(suite_mean, 6),
                    "tests": [
                        {
                            "nodeid": s.nodeid,
                            "score": round(s.score, 6),
                            "exploratory": s.exploratory,
                            "reasons": s.reasons,
                            "reliability": round(s.stats.reliability, 6) if s.stats else None,
                        }
                        for s in plan
                    ],
                },
                indent=2,
            )
        )
    else:
        _render_plan(plan, changed, suite_mean)
    return 0 if plan else 2


def cmd_run(args: argparse.Namespace) -> int:
    plan, changed, suite_mean = _make_plan(args)
    _render_plan(plan, changed, suite_mean)
    extra = _strip_separator(args.pytest_args)
    if not plan:
        print("Concrete has no learned map; running the full suite and learning it now.")
        return cmd_learn(
            argparse.Namespace(
                command=[sys.executable, "-m", "pytest", *extra], state=args.state
            )
        )
    nodeids = [s.nodeid for s in plan]
    command = [sys.executable, "-m", "pytest", *nodeids, *extra]
    code, ingested = _run_traced(command, Path(args.state))
    if code == 0:
        print(f"Concrete refreshed {ingested} selected-test observation(s).")
    else:
        print(
            f"Concrete recorded {ingested} selected-test observation(s); "
            f"pytest exited with code {code}.",
            file=sys.stderr,
        )
    return code


def cmd_status(args: argparse.Namespace) -> int:
    with StateDB(args.state) as db:
        c = db.counts()
    print("Concrete state")
    print(f"  tests              {c['tests']}")
    print(f"  observed files     {c['files']}")
    print(f"  learned edges      {c['edges']}")
    print(f"  test observations  {c['observations']}")
    print(f"  accumulated traced time {c['duration']:.3f}s")
    print(f"  estimated full suite {c['suite_mean']:.3f}s from per-test means")
    print(f"  tests that flipped {c['flaky']}")
    print(f"  failing last run   {c['currently_failing']}")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    with StateDB(args.state) as db:
        stats = db.test_stats(args.nodeid)
        files = db.files_for_test(args.nodeid)
    if not stats:
        print(f"Unknown test: {args.nodeid}", file=sys.stderr)
        return 2
    print(args.nodeid)
    print(
        f"  runs={stats.runs} pass={stats.passes} fail={stats.failures} skip={stats.skips} "
        f"flip_rate={stats.flip_rate:.3f} reliability={stats.reliability:.3f} "
        f"mean={stats.mean_duration:.4f}s"
    )
    print("  learned execution footprint:")
    for file, hits in files:
        print(f"    {hits:4d}  {file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="concrete",
        description=(
            "Learn which tests actually depend on changed code and run only "
            "the relevant frontier, plus exploration."
        ),
    )
    p.add_argument("--state", default=".concrete/state.sqlite3", help="state database path")
    sub = p.add_subparsers(dest="subcommand", required=True)

    learn = sub.add_parser(
        "learn", help="run a pytest suite once and learn per-test execution footprints"
    )
    learn.add_argument("command", nargs=argparse.REMAINDER, help="pytest command after --")
    learn.set_defaults(func=cmd_learn)

    def selection_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--budget",
            type=int,
            default=20,
            help="maximum number of tests to run (0 = all learned tests)",
        )
        sp.add_argument(
            "--exploration",
            type=float,
            default=0.15,
            help="fraction of budget reserved for tests outside the learned frontier",
        )
        sp.add_argument("--seed", default="concrete", help="stable exploration seed")
        sp.add_argument("--base", help="compare BASE...HEAD instead of working-tree changes")
        sp.add_argument(
            "--changed", action="append", default=[], help="explicit changed file; repeatable"
        )

    plan = sub.add_parser("plan", help="show which tests Concrete would run for the current change")
    selection_args(plan)
    plan.add_argument("--json", action="store_true", help="machine-readable plan")
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="select, run, and relearn the relevant pytest frontier")
    selection_args(run)
    run.add_argument("pytest_args", nargs=argparse.REMAINDER, help="extra pytest args after --")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="summarize the learned map and test reliability")
    status.set_defaults(func=cmd_status)

    explain = sub.add_parser("explain", help="show why a learned test is connected to project files")
    explain.add_argument("nodeid")
    explain.set_defaults(func=cmd_explain)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
