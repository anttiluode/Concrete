from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

_ROOT = Path(os.environ.get("CONCRETE_ROOT", os.getcwd())).resolve()
_TRACE_PATH = Path(os.environ.get("CONCRETE_TRACE_PATH", ".concrete/trace.jsonl"))
_current_files: set[str] | None = None
_current_nodeid: str | None = None
_current_reports: list[object] = []
_previous_sys_trace = None
_previous_thread_trace = None
_filename_cache: dict[str, str | None] = {}


def _relative_project_file(filename: str) -> str | None:
    cached = _filename_cache.get(filename, ...)
    if cached is not ...:
        return cached
    try:
        p = Path(filename)
        if not p.is_absolute():
            p = (_ROOT / p).resolve()
        else:
            p = p.resolve()
        rel = p.relative_to(_ROOT)
    except (OSError, ValueError):
        _filename_cache[filename] = None
        return None
    if any(
        part in {".git", ".concrete", ".venv", "venv", "site-packages", "__pycache__"}
        for part in rel.parts
    ):
        _filename_cache[filename] = None
        return None
    if p.suffix not in {".py", ".pyi"}:
        _filename_cache[filename] = None
        return None
    text = rel.as_posix()
    _filename_cache[filename] = text
    return text


def _trace(frame, event, arg):
    global _current_files
    if _current_files is not None and event == "call":
        rel = _relative_project_file(frame.f_code.co_filename)
        if rel:
            _current_files.add(rel)
    return _trace


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _current_files, _current_nodeid, _current_reports
    global _previous_sys_trace, _previous_thread_trace

    _current_files = set()
    item_file = _relative_project_file(str(item.path))
    if item_file:
        _current_files.add(item_file)
    _current_nodeid = item.nodeid
    _current_reports = []
    _previous_sys_trace = sys.gettrace()
    _previous_thread_trace = threading.gettrace()
    sys.settrace(_trace)
    threading.settrace(_trace)
    try:
        yield
    finally:
        sys.settrace(_previous_sys_trace)
        threading.settrace(_previous_thread_trace)
        reports = list(_current_reports)
        if any(getattr(r, "failed", False) for r in reports):
            outcome = "failed"
        elif any(
            getattr(r, "when", None) == "call" and getattr(r, "skipped", False)
            for r in reports
        ):
            outcome = "skipped"
        else:
            outcome = "passed"
        duration = sum(float(getattr(r, "duration", 0.0) or 0.0) for r in reports)
        record = {
            "nodeid": _current_nodeid,
            "outcome": outcome,
            "duration": duration,
            "files": sorted(_current_files or ()),
            "timestamp": int(time.time()),
        }
        _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        _current_files = None
        _current_nodeid = None
        _current_reports = []


def pytest_runtest_logreport(report):
    if _current_nodeid and report.nodeid == _current_nodeid:
        _current_reports.append(report)
