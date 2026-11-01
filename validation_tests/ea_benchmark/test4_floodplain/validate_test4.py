"""Validate EA benchmark Test 4 (flood propagation over an extended floodplain).

No exact solution exists. Until the full shallow-water packages' results
are digitised from the report (Neelz and Pender 2013, Figures 4.10-4.11),
the case is checked against

* the inflow volume (closed domain, so all of it must still be there);
* the report's summary (Section 4.5.4): the full SWE packages agree within
  a few per cent, with a travel time of ~1 h to points 5 and 6;
* the geometry: the front reaches points 1-5 in order of distance from
  the breach, and point 6 (424 m away) shortly after point 5 (400 m);
* a stored baseline per flow algorithm (baseline_test4_<alg>.csv). Set
  EA_REGENERATE_BASELINE=1 to write or rewrite it.
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

NAMES = ['1', '2', '3', '4', '5', '6']
WET = 0.01                 # depth marking the front's arrival [m]


class Test_results(unittest.TestCase):
    def setUp(self):
        for f in os.listdir('.'):
            if f.endswith(('.sww', '.msh', '.stdout', '.png')) or f.startswith('test4_'):
                os.remove(f)

    def test_test4_floodplain(self):
        res = anuga.run_anuga_script('run_test4.py', args=args, allow_parallel=False)
        assert res == 0, 'run_test4.py failed with return code %d' % res

        g = ea.load_gauges('test4_gauges.csv')
        t = g['time']
        with open('test4_volume.txt') as f:
            vol = {k: float(v) for k, v in (line.split() for line in f)}

        vol_err = abs(vol['final'] - vol['inflow']) / vol['inflow']
        arrival = {}
        for n in NAMES:
            wet = g['depth_' + n] > WET
            arrival[n] = t[np.argmax(wet)] / 60.0 if wet.any() else np.inf

        print()
        print(indent + 'Volume: inflow %.0f m^3, final %.0f m^3, error %.3f%%'
              % (vol['inflow'], vol['final'], 100 * vol_err))
        for n in NAMES:
            print(indent + 'Point %s: front at %5.1f min, peak depth %.3f m, peak speed %.3f m/s'
                  % (n, arrival[n], g['depth_' + n].max(), g['speed_' + n].max()))

        ea.plot_gauges(g, NAMES, 'stage', 'stage_plot.png',
                       'EA Test 4: water level', 'Level (m)',
                       time_scale=3600.0, time_label='Time (h)')
        ea.plot_gauges(g, NAMES, 'speed', 'speed_plot.png',
                       'EA Test 4: speed', 'Speed (m/s)',
                       time_scale=3600.0, time_label='Time (h)')

        assert vol_err < 0.01, 'Volume error %.2f%% exceeds 1%%' % (100 * vol_err)
        order = [arrival[n] for n in NAMES[:5]]
        assert all(a < b for a, b in zip(order, order[1:])), \
            'Front does not reach points 1-5 in order: %s' % order
        assert 45.0 < arrival['5'] < 80.0, \
            'Front reaches point 5 at %.1f min, report gives ~1 h' % arrival['5']
        assert arrival['5'] < arrival['6'] < 1.25 * arrival['5'], \
            'Point 6 front at %.1f min against point 5 at %.1f min' % (arrival['6'], arrival['5'])

        failed = ea.check_baseline('test4_gauges.csv', 'test4', args.alg,
                                   ['stage_' + n for n in NAMES], {'stage': 0.01}, indent)
        assert not failed, 'Gauges moved from baseline: %s' % failed


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_results)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
