from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS tests (
    nodeid TEXT PRIMARY KEY,
    runs INTEGER NOT NULL DEFAULT 0,
    passes INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    skips INTEGER NOT NULL DEFAULT 0,
    flips INTEGER NOT NULL DEFAULT 0,
    last_outcome TEXT,
    duration_sum REAL NOT NULL DEFAULT 0,
    last_seen INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS edges (
    file TEXT NOT NULL,
    nodeid TEXT NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0,
    last_seen INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (file, nodeid)
);
CREATE INDEX IF NOT EXISTS edges_file_idx ON edges(file);
CREATE INDEX IF NOT EXISTS edges_nodeid_idx ON edges(nodeid);
"""


@dataclass(frozen=True)
class TestStats:
    nodeid: str
    runs: int
    passes: int
    failures: int
    skips: int
    flips: int
    last_outcome: str | None
    duration_sum: float
    last_seen: int

    @property
    def mean_duration(self) -> float:
        return self.duration_sum / self.runs if self.runs else 0.0

    @property
    def failure_rate(self) -> float:
        return self.failures / self.runs if self.runs else 0.0

    @property
    def flip_rate(self) -> float:
        return self.flips / max(1, self.runs - 1)

    @property
    def reliability(self) -> float:
        return 1.0 - self.flip_rate


class StateDB:
    def __init__(self, path: str | Path = ".concrete/state.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "StateDB":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def ingest_record(self, record: dict) -> None:
        nodeid = str(record["nodeid"])
        outcome = str(record.get("outcome", "unknown"))
        duration = float(record.get("duration", 0.0) or 0.0)
        files = sorted({str(x).replace("\\", "/") for x in record.get("files", [])})
        now = int(record.get("timestamp", time.time()))

        old = self.conn.execute(
            "SELECT last_outcome FROM tests WHERE nodeid = ?", (nodeid,)
        ).fetchone()
        flip = int(bool(old and old["last_outcome"] and old["last_outcome"] != outcome))
        passed = int(outcome == "passed")
        failed = int(outcome == "failed")
        skipped = int(outcome == "skipped")

        self.conn.execute(
            """
            INSERT INTO tests(nodeid, runs, passes, failures, skips, flips,
                              last_outcome, duration_sum, last_seen)
            VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(nodeid) DO UPDATE SET
                runs = runs + 1,
                passes = passes + excluded.passes,
                failures = failures + excluded.failures,
                skips = skips + excluded.skips,
                flips = flips + excluded.flips,
                last_outcome = excluded.last_outcome,
                duration_sum = duration_sum + excluded.duration_sum,
                last_seen = excluded.last_seen
            """,
            (nodeid, passed, failed, skipped, flip, outcome, duration, now),
        )

        for file in files:
            self.conn.execute(
                """
                INSERT INTO edges(file, nodeid, hits, last_seen)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(file, nodeid) DO UPDATE SET
                    hits = hits + 1,
                    last_seen = excluded.last_seen
                """,
                (file, nodeid, now),
            )
        self.conn.commit()

    def ingest_jsonl(self, path: str | Path) -> int:
        count = 0
        p = Path(path)
        if not p.exists():
            return 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.ingest_record(json.loads(line))
                count += 1
        return count

    def all_tests(self) -> list[TestStats]:
        rows = self.conn.execute("SELECT * FROM tests ORDER BY nodeid").fetchall()
        return [TestStats(**dict(row)) for row in rows]

    def test_stats(self, nodeid: str) -> TestStats | None:
        row = self.conn.execute("SELECT * FROM tests WHERE nodeid = ?", (nodeid,)).fetchone()
        return TestStats(**dict(row)) if row else None

    def edges_for_files(self, files: Iterable[str]) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for file in files:
            norm = str(file).replace("\\", "/")
            rows = self.conn.execute(
                "SELECT nodeid, hits FROM edges WHERE file = ?", (norm,)
            ).fetchall()
            out[norm] = {row["nodeid"]: int(row["hits"]) for row in rows}
        return out

    def files_for_test(self, nodeid: str) -> list[tuple[str, int]]:
        rows = self.conn.execute(
            "SELECT file, hits FROM edges WHERE nodeid = ? ORDER BY hits DESC, file",
            (nodeid,),
        ).fetchall()
        return [(str(row["file"]), int(row["hits"])) for row in rows]

    def counts(self) -> dict[str, int | float]:
        tests = self.conn.execute("SELECT COUNT(*) AS n FROM tests").fetchone()["n"]
        files = self.conn.execute("SELECT COUNT(DISTINCT file) AS n FROM edges").fetchone()["n"]
        edges = self.conn.execute("SELECT COUNT(*) AS n FROM edges").fetchone()["n"]
        runs = self.conn.execute("SELECT COALESCE(SUM(runs),0) AS n FROM tests").fetchone()["n"]
        duration = self.conn.execute(
            "SELECT COALESCE(SUM(duration_sum),0) AS n FROM tests"
        ).fetchone()["n"]
        suite_mean = self.conn.execute(
            "SELECT COALESCE(SUM(CASE WHEN runs > 0 THEN duration_sum * 1.0 / runs ELSE 0 END),0) AS n FROM tests"
        ).fetchone()["n"]
        flaky = self.conn.execute("SELECT COUNT(*) AS n FROM tests WHERE flips > 0").fetchone()["n"]
        failing = self.conn.execute(
            "SELECT COUNT(*) AS n FROM tests WHERE last_outcome = 'failed'"
        ).fetchone()["n"]
        return {
            "tests": int(tests),
            "files": int(files),
            "edges": int(edges),
            "observations": int(runs),
            "duration": float(duration),
            "suite_mean": float(suite_mean),
            "flaky": int(flaky),
            "currently_failing": int(failing),
        }
