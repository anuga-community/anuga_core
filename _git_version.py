"""
Helper script called by meson.build to produce a PEP 440-compliant version
string from `git describe` output.

git describe format:  TAG[-N-gSHA][-dirty]
PEP 440 mapping:
  3.2.0                          -> 3.2.0
  3.2.0-4-gabcdef                -> 3.2.0.dev4+gabcdef
  3.2.0-4-gabcdef-dirty          -> 3.2.0.dev4+gabcdef.dirty
  3.2.0-dirty                    -> 3.2.0+dirty
  abcdef  (no tag reachable)     -> 0.0.0+gabcdef
"""

import re
import subprocess
import sys


def git_version():
    result = subprocess.run(
        ['git', 'describe', '--tags', '--dirty', '--always'],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return '0.0.0+unknown'

    raw = result.stdout.strip()

    # Full describe: TAG-N-gSHA[-dirty]
    m = re.match(
        r'^(\d+\.\d+(?:\.\d+)?(?:[a-z]\w*)?)'  # base tag
        r'(?:-(\d+)-g([0-9a-f]+))?'             # optional distance + sha
        r'(-dirty)?$',                           # optional dirty flag
        raw
    )
    if not m:
        # No reachable tag — just a bare SHA (possibly dirty)
        sha = re.sub(r'-dirty$', '', raw)
        dirty = raw.endswith('-dirty')
        return f'0.0.0+g{sha}' + ('.dirty' if dirty else '')

    base, distance, sha, dirty = m.groups()
    if distance:
        version = f'{base}.dev{distance}+g{sha}'
        if dirty:
            version += '.dirty'
    elif dirty:
        version = f'{base}+dirty'
    else:
        version = base

    return version


if __name__ == '__main__':
    print(git_version())
