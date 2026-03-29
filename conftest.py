"""Root pytest configuration.

Fast-test filtering
-------------------
By default **all** tests are run.

Pass ``--run-fast`` to skip slow tests and get a quick feedback loop::

    pytest --run-fast

Tests are considered slow if they carry ``@pytest.mark.slow`` **or** live
under ``anuga/parallel/tests/`` (MPI-based tests that spawn subprocesses).

To run *only* slow tests::

    pytest -m slow
"""

import pytest


# ---------------------------------------------------------------------------
# CLI option
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        '--run-fast',
        action='store_true',
        default=False,
        help='Skip slow tests (parallel + individually marked) for a quick run.',
    )


# ---------------------------------------------------------------------------
# Auto-mark and conditional skip
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    # Auto-mark every test under anuga/parallel/tests/ as slow.
    for item in items:
        if 'parallel/tests/' in str(item.fspath).replace('\\', '/'):
            item.add_marker(pytest.mark.slow)

    # Only skip slow tests when --run-fast is explicitly requested.
    if not config.getoption('--run-fast'):
        return

    skip_slow = pytest.mark.skip(reason='slow test — omit --run-fast to include')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip_slow)
