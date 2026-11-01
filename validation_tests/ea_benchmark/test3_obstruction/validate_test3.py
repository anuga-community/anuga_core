"""Validate EA benchmark Test 3 (momentum over a small obstruction).

No exact solution exists. Until the full shallow-water packages' results
are digitised from the report (Neelz and Pender 2013, Figure 4.8), the
case is checked against

* the inflow volume (closed domain, so all of it must still be there);
* the report's summary of the full SWE packages (Section 4.4.3): point 1
  rises from 9.75 m to ~9.98-9.99 m by t ~ 120 s, and after t ~ 150 s the
  level at point 2 rises by ~5-6 cm as water crosses the crest;
* a stored baseline per flow algorithm (baseline_test3_<alg>.csv), so that
  a change that moves the answer is caught. Set EA_REGENERATE_BASELINE=1
  to write or rewrite the baseline for the algorithm being run.
"""

import os
import sys
import unittest

import numpy as np
import anuga

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import ea_common as ea  # noqa: E402

args = anuga.get_args()
indent = anuga.indent
verbose = args.verbose

BED = 9.75                 # bed level at both points


class Test_results(unittest.TestCase):
    def setUp(self):
        for f in os.listdir('.'):
            if f.endswith(('.sww', '.msh', '.stdout', '.png')) or f.startswith('test3_'):
                os.remove(f)

    def test_test3_obstruction(self):
        res = anuga.run_anuga_script('run_test3.py', args=args, allow_parallel=False)
        assert res == 0, 'run_test3.py failed with return code %d' % res

        g = ea.load_gauges('test3_gauges.csv')
        t = g['time']
        with open('test3_volume.txt') as f:
            vol = {k: float(v) for k, v in (line.split() for line in f)}

        vol_err = abs(vol['final'] - vol['inflow']) / vol['inflow']
        level_1_120 = np.interp(120.0, t, g['stage_1'])
        level_1_end = g['stage_1'][-1]
        rise_2 = g['stage_2'][-1] - BED
        wet_2 = t[np.argmax(g['depth_2'] > 0.01)] if np.any(g['depth_2'] > 0.01) else np.inf

        print()
        print(indent + 'Volume: inflow %.1f m^3, final %.1f m^3, error %.2f%%'
              % (vol['inflow'], vol['final'], 100 * vol_err))
        print(indent + 'Point 1 level at 120 s: %.3f m, at 900 s: %.3f m'
              % (level_1_120, level_1_end))
        print(indent + 'Point 2 wet from %.0f s, final rise %.3f m' % (wet_2, rise_2))

        ea.plot_gauges(g, ['1', '2'], 'stage', 'stage_plot.png',
                       'EA Test 3: water level', 'Level (m)')
        ea.plot_gauges(g, ['1'], 'speed', 'speed_plot.png',
                       'EA Test 3: speed', 'Speed (m/s)')

        assert vol_err < 0.01, 'Volume error %.2f%% exceeds 1%%' % (100 * vol_err)
        assert 9.94 < level_1_120 < 10.01, \
            'Point 1 level %.3f m at 120 s, report gives ~9.98-9.99 m' % level_1_120
        assert 9.95 < level_1_end < 10.0, \
            'Point 1 final level %.3f m outside 9.95-10.0 m' % level_1_end
        assert 0.03 < rise_2 < 0.09, \
            'Point 2 rise %.3f m, report gives ~0.05-0.06 m' % rise_2
        assert 100.0 < wet_2 < 250.0, \
            'Point 2 wets at %.0f s, report gives ~150 s' % wet_2

        failed = ea.check_baseline('test3_gauges.csv', 'test3', args.alg,
                                   ['stage_1', 'stage_2'], {'stage': 0.01}, indent)
        assert not failed, 'Gauges moved from baseline: %s' % failed


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_results)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
