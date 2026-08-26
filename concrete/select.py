from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from .state import StateDB, TestStats


@dataclass
class Selection:
    nodeid: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    exploratory: bool = False
    stats: TestStats | None = None


def _norm(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _affinity(changed: str, nodeid: str) -> float:
    """Cheap prior for never-before-seen files.

    This is intentionally weak: learned execution edges dominate it.
    """
    changed_path = PurePosixPath(_norm(changed))
    test_file = PurePosixPath(nodeid.split("::", 1)[0])
    score = 0.0

    if changed_path == test_file:
        return 8.0

    cstem = changed_path.stem.removeprefix("test_")
    tstem = test_file.stem.removeprefix("test_")
    if cstem and cstem == tstem:
        score += 3.0
    elif cstem and cstem in tstem:
        score += 1.5

    common = 0
    for a, b in zip(changed_path.parts[:-1], test_file.parts[:-1]):
        if a != b:
            break
        common += 1
    score += min(1.5, common * 0.35)
    return score


def _stable_noise(seed: str, nodeid: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{nodeid}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def plan_tests(
    db: StateDB,
    changed_files: list[str],
    budget: int,
    exploration: float = 0.15,
    seed: str = "concrete",
) -> list[Selection]:
    tests = db.all_tests()
    if not tests:
        return []

    changed = [_norm(x) for x in changed_files]
    edge_map = db.edges_for_files(changed)
    selections: dict[str, Selection] = {
        t.nodeid: Selection(nodeid=t.nodeid, stats=t) for t in tests
    }

    for file, neighbors in edge_map.items():
        for nodeid, hits in neighbors.items():
            if nodeid not in selections:
                continue
            contribution = 12.0 + 2.0 * math.log1p(hits)
            selections[nodeid].score += contribution
            selections[nodeid].reasons.append(
                f"executed {file} ({hits} observed run{'s' if hits != 1 else ''})"
            )

    for sel in selections.values():
        for file in changed:
            a = _affinity(file, sel.nodeid)
            if a:
                sel.score += a
                if a >= 3.0:
                    sel.reasons.append(f"path affinity with {file}")
        if sel.stats:
            if sel.stats.last_outcome == "failed":
                sel.score += 2.5
                sel.reasons.append("failed on its latest observed run")
            elif sel.stats.failure_rate > 0:
                sel.score += min(1.5, 2.0 * sel.stats.failure_rate)
            # Flaky evidence still matters, but it should not dominate a stable direct dependency.
            if sel.stats.flip_rate > 0.25:
                sel.score -= min(1.0, sel.stats.flip_rate)

    n_total = len(tests)
    if budget <= 0:
        budget = n_total
    budget = min(budget, n_total)
    reserve = 0
    if budget > 1 and exploration > 0 and n_total > 1:
        reserve = max(1, int(round(budget * min(max(exploration, 0.0), 0.8))))
        reserve = min(reserve, budget - 1)

    ranked = sorted(
        selections.values(),
        key=lambda s: (s.score, -(s.stats.mean_duration if s.stats else 0.0), s.nodeid),
        reverse=True,
    )

    exploitation_budget = budget - reserve
    chosen = ranked[:exploitation_budget]
    chosen_ids = {x.nodeid for x in chosen}

    remaining = [x for x in ranked if x.nodeid not in chosen_ids]
    if reserve and remaining:
        exploratory = sorted(
            remaining,
            key=lambda s: _stable_noise(seed + "|" + "|".join(changed), s.nodeid),
            reverse=True,
        )[:reserve]
        for sel in exploratory:
            sel.exploratory = True
            sel.reasons.append("exploration reserve: tests the learned map's blind spots")
        chosen.extend(exploratory)

    return sorted(chosen, key=lambda s: (s.exploratory, -s.score, s.nodeid))
