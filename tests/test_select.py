from concrete.select import plan_tests
from concrete.state import StateDB


def _record(nodeid, files, outcome="passed"):
    return {"nodeid": nodeid, "outcome": outcome, "duration": 0.01, "files": files}


def test_direct_execution_edge_beats_unrelated_test(tmp_path):
    with StateDB(tmp_path / "s.db") as db:
        db.ingest_record(_record("tests/test_alpha.py::test_alpha", ["pkg/alpha.py"]))
        db.ingest_record(_record("tests/test_beta.py::test_beta", ["pkg/beta.py"]))
        plan = plan_tests(db, ["pkg/alpha.py"], budget=1, exploration=0)
        assert [x.nodeid for x in plan] == ["tests/test_alpha.py::test_alpha"]
        assert "executed pkg/alpha.py" in plan[0].reasons[0]


def test_exploration_reserve_runs_outside_frontier(tmp_path):
    with StateDB(tmp_path / "s.db") as db:
        for name in "abcd":
            db.ingest_record(_record(f"tests/test_{name}.py::test_{name}", [f"pkg/{name}.py"]))
        plan = plan_tests(db, ["pkg/a.py"], budget=3, exploration=0.34, seed="x")
        assert len(plan) == 3
        assert any(x.nodeid == "tests/test_a.py::test_a" for x in plan)
        assert sum(x.exploratory for x in plan) == 1


def test_path_affinity_can_cover_new_test_file(tmp_path):
    with StateDB(tmp_path / "s.db") as db:
        db.ingest_record(_record("tests/test_widget.py::test_widget", ["pkg/old.py"]))
        db.ingest_record(_record("tests/test_other.py::test_other", ["pkg/other.py"]))
        plan = plan_tests(db, ["pkg/widget.py"], budget=1, exploration=0)
        assert plan[0].nodeid == "tests/test_widget.py::test_widget"
