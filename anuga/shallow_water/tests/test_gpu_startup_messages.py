"""The unified (mode-2) start-up lines all print before the first yield.

The C layer names its flux kernel ("flux path : cell" and so on). It used to
do so at the first flux call, which comes after the t=0 yield, so the line
landed in the middle of a script's own progress output. evolve() now asks for
it during start-up.

The line prints once per process, so each case runs in its own interpreter.
"""
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent("""
    import anuga
    d = anuga.rectangular_cross_domain(6, 6, len1=6.0, len2=6.0)
    d.set_store(False)
    d.set_quantity('elevation', lambda x, y: 0.1 * x)
    d.set_quantity('stage', 1.0)
    Br = anuga.Reflective_boundary(d)
    d.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})
    d.set_compute_mode('unified')
    %s
    for t in d.evolve(yieldstep=0.1, finaltime=0.2):
        print('YIELD %%.1f' %% t, flush=True)
""")


def run(extra=''):
    out = subprocess.run([sys.executable, '-c', SCRIPT % extra],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    return out.stdout


@pytest.mark.parametrize('extra, kernel', [
    ('', 'cell'),
    ('d.set_use_active_set(True)', 'scatter'),
])
def test_the_flux_path_line_comes_before_the_first_yield(extra, kernel):
    lines = [line.strip() for line in run(extra).splitlines()]
    flux = [i for i, line in enumerate(lines) if line.startswith('flux path')]
    assert len(flux) == 1, lines
    assert lines[flux[0]] == 'flux path : %s' % kernel
    assert flux[0] < lines.index('YIELD 0.0')
