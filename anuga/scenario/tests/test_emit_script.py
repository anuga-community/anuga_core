"""Tests for anuga.scenario.emit_script (the `anuga_toml_run --emit-script`
generator)."""

import ast
import os
import tempfile
import unittest

from anuga.scenario.emit_script import build_run_script, write_run_script


class Test_emit_script(unittest.TestCase):

    def test_generated_script_is_valid_python(self):
        """The emitted script must at least parse as Python."""
        text = build_run_script('scenario.toml')
        ast.parse(text)   # raises SyntaxError on failure

    def test_config_name_is_substituted(self):
        text = build_run_script('my_flood.toml')
        # Passed to PrepareData and shown in the docstring/example.
        self.assertIn("PrepareData('my_flood.toml'", text)
        self.assertIn('anuga_toml_run my_flood.toml', text)

    def test_drives_the_standard_phases(self):
        """All the runner's phases appear in the generated script, in order."""
        text = build_run_script('c.toml')
        phases = [
            'setup_mesh.setup_mesh(project)',
            'setup_initial_conditions.setup_initial_conditions(domain, project)',
            'setup_riverwalls.setup_riverwalls(domain, project)',
            'setup_rainfall.setup_rainfall(domain, project)',
            'setup_inlets.setup_inlets(domain, project)',
            'setup_boundary_conditions.setup_boundary_conditions(domain, project)',
            'domain.evolve(',
        ]
        last = -1
        for p in phases:
            idx = text.find(p)
            self.assertNotEqual(idx, -1, 'missing phase: %s' % p)
            self.assertGreater(idx, last, 'phase out of order: %s' % p)
            last = idx

    def test_evolve_uses_config_timestepping(self):
        text = build_run_script('c.toml')
        self.assertIn('yieldstep=project.yieldstep', text)
        self.assertIn('finaltime=project.finaltime', text)

    def test_parallel_aware(self):
        """MPI-safe: barrier + rank-guarded output + parallel sww merge."""
        text = build_run_script('c.toml')
        self.assertIn('finalize()', text)
        self.assertIn('if numprocs > 1:', text)
        self.assertIn('domain.sww_merge(delete_old=True)', text)

    def test_script_example_uses_given_path(self):
        text = build_run_script('c.toml', script_path='/some/dir/my_run.py')
        self.assertIn('python my_run.py', text)

    def test_write_run_script_writes_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'run.py')
            returned = write_run_script('c.toml', path)
            self.assertEqual(returned, path)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding='utf-8') as fh:
                ast.parse(fh.read())


if __name__ == '__main__':
    unittest.main()
