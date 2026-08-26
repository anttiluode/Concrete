from concrete.state import StateDB


def test_ingest_tracks_edges_and_flips(tmp_path):
    path = tmp_path / "state.sqlite3"
    with StateDB(path) as db:
        db.ingest_record({
            "nodeid": "tests/test_a.py::test_a",
            "outcome": "passed",
            "duration": 0.1,
            "files": ["pkg/a.py", "tests/test_a.py"],
        })
        db.ingest_record({
            "nodeid": "tests/test_a.py::test_a",
            "outcome": "failed",
            "duration": 0.2,
            "files": ["pkg/a.py"],
        })
        stats = db.test_stats("tests/test_a.py::test_a")
        assert stats is not None
        assert stats.runs == 2
        assert stats.failures == 1
        assert stats.flips == 1
        assert stats.flip_rate == 1.0
        assert db.edges_for_files(["pkg/a.py"])["pkg/a.py"]["tests/test_a.py::test_a"] == 2
