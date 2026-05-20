import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-perf",
        action="store_true",
        default=False,
        help="Run wall-clock performance regression tests",
    )
