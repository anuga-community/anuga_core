"""Validate EA benchmark Test 6B (dam break past a building, field scale).

No exact solution exists. Until the full shallow-water packages' results
are digitised from the report (Neelz and Pender 2013, Figures 4.27-4.28),
the case is checked against

* volume: the channel is closed, so the initial volume must be kept;
* the geometry: the bore reaches G1/G2 (x = 53 m) before G3/G4 (80 m)
  and G5 (104 m), and the reservoir gauge G6 starts to fall;
* the report's 2013 concern about ANUGA's high peak velocities (Tests 5
  and 6B): the largest speed anywhere in water deeper than 5 cm must stay
  below the dam-break bound 2 sqrt(g h0) = 17.7 m/s;
* the end state: by 30 min the channel is close to level everywhere;
* a stored baseline per flow algorithm (baseline_test6b_<alg>.csv). Set
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

NAMES = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6']
H_DOWN = 0.4
SPEED_BOUND = 2.0 * np.sqrt(9.8 * 8.0)


class Test_results(unittest.TestCase):
    def setUp(self):
        for f in os.listdir('.'):
            if f.endswith(('.sww', '.msh', '.stdout', '.png')) or f.startswith('test6b_'):
                os.remove(f)

    def test_test6b_dambreak(self):
        res = anuga.run_anuga_script('run_test6b.py', args=args, allow_parallel=False)
        assert res == 0, 'run_test6b.py failed with return code %d' % res

        g = ea.load_gauges('test6b_gauges.csv')
        t = g['time']
        with open('test6b_volume.txt') as f:
            vol = {k: float(v) for k, v in (line.split() for line in f)}

        vol_err = abs(vol['final'] - vol['initial']) / vol['initial']
        arrival = {}
        for n in NAMES[:5]:
            risen = g['stage_' + n] > H_DOWN + 0.05
            arrival[n] = t[np.argmax(risen)] if risen.any() else np.inf
        final = np.array([g['stage_' + n][-1] for n in NAMES])

        print()
        print(indent + 'Volume: initial %.1f m^3, final %.1f m^3, error %.2e'
              % (vol['initial'], vol['final'], vol_err))
        print(indent + 'Largest speed in the domain: %.2f m/s' % vol['max_speed'])
        for n in NAMES:
            print(indent + '%s: peak level %.3f m, peak speed %.2f m/s, final level %.3f m'
                  % (n, g['stage_' + n].max(), g['speed_' + n].max(), g['stage_' + n][-1]))
        print(indent + 'Bore arrival (s): ' +
              ', '.join('%s %.0f' % (n, arrival[n]) for n in NAMES[:5]))

        ea.plot_gauges(g, NAMES, 'stage', 'stage_plot.png',
                       'EA Test 6B: water level', 'Level (m)',
                       time_scale=60.0, time_label='Time (min)')
        ea.plot_gauges(g, NAMES, 'speed', 'speed_plot.png',
                       'EA Test 6B: speed', 'Speed (m/s)',
                       time_scale=60.0, time_label='Time (min)')

        assert vol_err < 1.0e-3, 'Volume error %.2e exceeds 1e-3' % vol_err
        assert max(arrival['G1'], arrival['G2']) < min(arrival['G3'], arrival['G4']) \
            < arrival['G5'], 'Bore does not reach the gauges in order: %s' % arrival
        assert g['stage_G6'][-1] < 0.5 * g['stage_G6'][0], 'Reservoir (G6) did not drain'
        assert vol['max_speed'] < SPEED_BOUND, \
            'Largest speed %.2f m/s exceeds 2 sqrt(g h0) = %.1f m/s' % (vol['max_speed'], SPEED_BOUND)
        assert final.max() - final.min() < 0.05, \
            'Channel not level at 30 min: final levels %s' % np.round(final, 3)

        failed = ea.check_baseline('test6b_gauges.csv', 'test6b', args.alg,
                                   ['stage_' + n for n in NAMES], {'stage': 0.02}, indent)
        assert not failed, 'Gauges moved from baseline: %s' % failed


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_results)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
