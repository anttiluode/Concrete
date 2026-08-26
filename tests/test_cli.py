from concrete.cli import _pytest_command


def test_injects_plugin_into_python_m_pytest():
    cmd = _pytest_command(["python", "-m", "pytest", "-q"], plugin=True)
    assert cmd == ["python", "-m", "pytest", "-p", "concrete.pytest_plugin", "-q"]


def test_injects_plugin_into_pytest_binary():
    cmd = _pytest_command(["pytest", "-q"], plugin=True)
    assert cmd == ["pytest", "-p", "concrete.pytest_plugin", "-q"]
