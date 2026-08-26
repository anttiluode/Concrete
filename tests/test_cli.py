import sys

from concrete.cli import _pytest_command


def test_bare_pytest_uses_concretes_python():
    cmd = _pytest_command(["pytest", "-q"], plugin=True)
    assert cmd == [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "concrete.pytest_plugin",
        "-q",
    ]


def test_windows_pytest_exe_uses_concretes_python():
    cmd = _pytest_command(["pytest.exe", "tests/test_x.py"], plugin=True)
    assert cmd[:5] == [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "concrete.pytest_plugin",
    ]
    assert cmd[5:] == ["tests/test_x.py"]


def test_explicit_python_m_pytest_is_preserved():
    cmd = _pytest_command([sys.executable, "-m", "pytest", "-q"], plugin=True)
    assert cmd == [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "concrete.pytest_plugin",
        "-q",
    ]
