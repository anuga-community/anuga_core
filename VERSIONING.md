# ANUGA Versioning

## Overview

ANUGA's version is derived automatically from git tags at build time. There is no manually maintained version string — the single source of truth is a git tag on the repository.

## Version format

Tags follow plain `MAJOR.MINOR.PATCH` — no `v` prefix:

```
3.3.0        # release tag
3.4.0        # next release tag
```

Between releases, `git describe` produces a string that is normalised to [PEP 440](https://peps.python.org/pep-0440/) by `_git_version.py`:

| `git describe` output | Installed `__version__` |
|---|---|
| `3.3.0` | `3.3.0` |
| `3.3.0-4-gabcdef` | `3.3.0.dev4+gabcdef` |
| `3.3.0-4-gabcdef-dirty` | `3.3.0.dev4+gabcdef.dirty` |
| `3.3.0-dirty` | `3.3.0+dirty` |
| `abcdef` (no tag reachable) | `0.0.0+gabcdef` |

## How it works — step by step

**1. `_git_version.py` (repo root)**

Called by meson at configure time. Runs `git describe --tags --dirty --always` and normalises the output to a PEP 440 string. This is the only place the version string is computed.

**2. `meson.build` (repo root)**

Calls `_git_version.py` to populate the meson `project()` version, then fetches the full commit SHA and commit date for embedding in `_version.py`:

```meson
project('anuga', 'c', 'cpp', 'cython',
  version: run_command('python', ['_git_version.py'], check: false).stdout().strip(),
  ...)

git = find_program('git', ...)
_git_sha  = run_command(git, ['rev-parse', 'HEAD'], ...).stdout().strip()
_git_date = run_command(git, ['log', '-1', '--format=%ci', 'HEAD'], ...).stdout().strip()
```

**3. `anuga/meson.build`**

Uses meson's `configure_file()` to generate `anuga/_version.py` from the template `anuga/_version.py.in`, substituting the version, SHA and date:

```meson
configure_file(
  input:       '_version.py.in',
  output:      '_version.py',
  configuration: _version_conf,
  install:     true,
  install_dir: py3.get_install_dir() / 'anuga',
)
```

**4. `anuga/_version.py.in` (template)**

```python
__version__ = "@VERSION@"
__git_sha__ = "@GIT_SHA@"
__git_committed_datetime__ = "@GIT_DATE@"
```

**5. `anuga/_version.py` (generated, not committed)**

The generated file is listed in `.gitignore`. It is created fresh every time `pip install --no-build-isolation .` is run.

**6. `anuga/__init__.py`**

Imports from the generated file:

```python
from ._version import __version__
from ._version import __git_sha__
from ._version import __git_committed_datetime__
```

**7. `pyproject.toml`**

Uses `dynamic = ['version']` so meson-python reads the version from `project()` in `meson.build` rather than a hard-coded string.

## Making a release

1. Ensure all changes are committed and pushed to `main`.
2. Tag the commit — **no `v` prefix**:
   ```bash
   git tag 3.3.0
   git push origin 3.3.0
   ```
3. Create a GitHub release from that tag. The `python-publish-pypi.yml` workflow triggers on `release: published` and builds wheels for all supported platforms and Python versions, then uploads to PyPI.

## GitHub Actions requirement

Both CI workflows (`conda-setup.yml` and `python-publish-pypi.yml`) use `fetch-depth: 0` in the `actions/checkout` step. This is required so that `git describe` can walk back to find the most recent tag — a shallow clone (the default) would cause `git describe` to fail and produce a `0.0.0+unknown` version.

## Checking the version

```python
import anuga
print(anuga.__version__)                  # e.g. '3.3.0.dev274+gc733e0ff'
print(anuga.__git_sha__)                  # full commit SHA
print(anuga.__git_committed_datetime__)   # e.g. '2026-03-17 10:23:45 +1100'
```

## Files involved

| File | Status | Purpose |
|---|---|---|
| `_git_version.py` | committed | Computes PEP 440 version from `git describe` |
| `meson.build` | committed | Calls `_git_version.py`; fetches SHA/date |
| `anuga/meson.build` | committed | Generates `_version.py` via `configure_file()` |
| `anuga/_version.py.in` | committed | Template for generated file |
| `anuga/_version.py` | **gitignored** | Generated at build time — do not edit |
| `pyproject.toml` | committed | `dynamic = ['version']` — no hard-coded version |
