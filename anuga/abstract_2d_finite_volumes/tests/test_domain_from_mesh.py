"""Tests for creating a Domain from a Mesh.

Covers the three construction paths:
  1. Domain(mesh)                     — from a Mesh object directly
  2. Domain(coordinates, triangles, boundary) — from raw arrays
  3. Domain(mesh) via pmesh_to_mesh   — from a Pmesh/tsh file

For each path the tests verify:
  - Domain is the correct type
  - Element and node counts are consistent with the input mesh
  - boundary dict keys and tags are preserved
  - Geometric attributes (areas, normals, edgelengths, radii) have
    correct shape and physically sensible values
  - Conserved quantities (stage, xmomentum, ymomentum) and other
    quantities (elevation, friction) are initialised and settable
  - set_boundary accepts Reflective_boundary without error
"""

import os
import unittest
import tempfile
import numpy as num

from anuga.abstract_2d_finite_volumes.neighbour_mesh import Mesh
from anuga.shallow_water.shallow_water_domain import Domain
from anuga.shallow_water.boundaries import Reflective_boundary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mesh_cross(m=3, n=4, len1=1.0, len2=1.0):
    """Return a Mesh built from rectangular_cross_with_neighbours."""
    from anuga.abstract_2d_finite_volumes.mesh_factory import \
        rectangular_cross_with_neighbours
    points, triangles, boundary, neighbours = \
        rectangular_cross_with_neighbours(m, n, len1, len2)
    return Mesh(points, triangles, boundary, triangle_neighbours=neighbours)


def _make_mesh_rect(m=3, n=4, len1=1.0, len2=1.0):
    """Return a Mesh built from rectangular_with_neighbours."""
    from anuga.abstract_2d_finite_volumes.mesh_factory import \
        rectangular_with_neighbours
    points, triangles, boundary, neighbours = \
        rectangular_with_neighbours(m, n, len1, len2)
    return Mesh(points, triangles, boundary, triangle_neighbours=neighbours)


_TSH_CONTENT = """\
4 3 # <vertex #> <x> <y> [attributes]
0 0.0 0.0 0.0 0.0 0.01
1 1.0 0.0 10.0 10.0 0.02
2 0.0 1.0 0.0 10.0 0.03
3 0.5 0.25 8.0 12.0 0.04
# Vert att title
elevation
stage
friction
2 # <triangle #> [<vertex #>] [<neigbouring triangle #>]
0 0 3 2 -1  -1  1 dsg
1 0 1 3 -1  0 -1   ole nielsen
4 # <segment #> <vertex #>  <vertex #> [boundary tag]
0 1 0 2
1 0 2 3
2 2 3
3 3 1 1
3 0 # <x> <y> [attributes] ...Mesh Vertices...
0 216.0 -86.0
1 160.0 -167.0
2 114.0 -91.0
3 # <vertex #>  <vertex #> [boundary tag] ...Mesh Segments...
0 0 1 0
1 1 2 0
2 2 0 0
0
0
0
#Geo reference
56
140
120
"""


def _make_domain_from_pmesh():
    """Return (domain, tsh_path) created via pmesh_to_mesh → Domain."""
    from anuga.pmesh.mesh import importMeshFromFile
    from anuga.abstract_2d_finite_volumes.pmesh2domain import pmesh_to_mesh
    fd, path = tempfile.mkstemp(suffix='.tsh')
    with os.fdopen(fd, 'w') as f:
        f.write(_TSH_CONTENT)
    pmesh = importMeshFromFile(path)
    mesh = pmesh_to_mesh(pmesh)
    return Domain(mesh), path


# ---------------------------------------------------------------------------
# Tests: Domain from a cross-pattern Mesh
# ---------------------------------------------------------------------------

class Test_Domain_from_cross_Mesh(unittest.TestCase):

    def setUp(self):
        self.m, self.n = 3, 4
        self.mesh = _make_mesh_cross(self.m, self.n)
        self.domain = Domain(self.mesh)

    def test_returns_domain_instance(self):
        self.assertIsInstance(self.domain, Domain)

    def test_number_of_elements(self):
        expected = 4 * self.m * self.n
        self.assertEqual(self.domain.number_of_elements, expected)

    def test_number_of_nodes(self):
        expected = (self.m + 1) * (self.n + 1) + self.m * self.n
        self.assertEqual(self.domain.number_of_nodes, expected)

    def test_mesh_is_same_object(self):
        self.assertIs(self.domain.mesh, self.mesh)

    def test_boundary_preserved(self):
        self.assertEqual(self.domain.boundary, self.mesh.boundary)

    def test_boundary_tags_are_compass(self):
        tags = set(self.domain.boundary.values())
        self.assertTrue({'left', 'right', 'top', 'bottom'}.issubset(tags))

    def test_areas_shape_and_sign(self):
        areas = self.domain.areas
        self.assertEqual(areas.shape, (self.domain.number_of_elements,))
        self.assertTrue(num.all(areas > 0), "All triangle areas must be positive")

    def test_total_area(self):
        num.testing.assert_allclose(self.domain.areas.sum(), 1.0, rtol=1e-10)

    def test_normals_shape(self):
        # normals: (N, 6) — two components per edge, three edges per triangle
        N = self.domain.number_of_elements
        self.assertEqual(self.domain.normals.shape, (N, 6))

    def test_edgelengths_shape_and_sign(self):
        N = self.domain.number_of_elements
        self.assertEqual(self.domain.edgelengths.shape, (N, 3))
        self.assertTrue(num.all(self.domain.edgelengths > 0))

    def test_radii_shape_and_sign(self):
        N = self.domain.number_of_elements
        self.assertEqual(self.domain.radii.shape, (N,))
        self.assertTrue(num.all(self.domain.radii > 0))

    def test_centroid_coordinates_shape(self):
        N = self.domain.number_of_elements
        self.assertEqual(self.domain.centroid_coordinates.shape, (N, 2))

    def test_centroid_coordinates_finite(self):
        self.assertTrue(num.all(num.isfinite(self.domain.centroid_coordinates)))

    def test_vertex_coordinates_shape(self):
        N = self.domain.number_of_elements
        self.assertEqual(self.domain.vertex_coordinates.shape, (3 * N, 2))

    def test_conserved_quantities_exist(self):
        for qty in ('stage', 'xmomentum', 'ymomentum'):
            self.assertIn(qty, self.domain.quantities)

    def test_other_quantities_exist(self):
        for qty in ('elevation', 'friction'):
            self.assertIn(qty, self.domain.quantities)

    def test_quantities_initialise_to_zero(self):
        for qty in ('stage', 'xmomentum', 'ymomentum'):
            vals = self.domain.quantities[qty].centroid_values
            num.testing.assert_array_equal(vals, 0.0)

    def test_set_quantity_scalar(self):
        self.domain.set_quantity('stage', 1.5)
        vals = self.domain.quantities['stage'].centroid_values
        num.testing.assert_allclose(vals, 1.5)

    def test_set_quantity_array(self):
        N = self.domain.number_of_elements
        arr = num.linspace(0.0, 1.0, N)
        self.domain.set_quantity('elevation', arr, location='centroids')
        num.testing.assert_allclose(
            self.domain.quantities['elevation'].centroid_values, arr)

    def test_set_quantity_function(self):
        self.domain.set_quantity('elevation', lambda x, y: x + y)
        vals = self.domain.quantities['elevation'].centroid_values
        cx = self.domain.centroid_coordinates[:, 0]
        cy = self.domain.centroid_coordinates[:, 1]
        num.testing.assert_allclose(vals, cx + cy, rtol=1e-10)

    def test_set_boundary_reflective(self):
        tags = set(self.domain.boundary.values())
        bmap = {tag: Reflective_boundary(self.domain) for tag in tags}
        self.domain.set_boundary(bmap)   # must not raise

    def test_get_quantity_returns_array(self):
        vals = self.domain.get_quantity('stage').centroid_values
        self.assertEqual(vals.shape, (self.domain.number_of_elements,))


# ---------------------------------------------------------------------------
# Tests: Domain from a rectangular (2-triangle-per-cell) Mesh
# ---------------------------------------------------------------------------

class Test_Domain_from_rect_Mesh(unittest.TestCase):

    def setUp(self):
        self.m, self.n = 4, 3
        self.mesh = _make_mesh_rect(self.m, self.n)
        self.domain = Domain(self.mesh)

    def test_returns_domain_instance(self):
        self.assertIsInstance(self.domain, Domain)

    def test_number_of_elements(self):
        self.assertEqual(self.domain.number_of_elements, 2 * self.m * self.n)

    def test_number_of_nodes(self):
        self.assertEqual(self.domain.number_of_nodes,
                         (self.m + 1) * (self.n + 1))

    def test_boundary_tags_are_compass(self):
        tags = set(self.domain.boundary.values())
        self.assertTrue({'left', 'right', 'top', 'bottom'}.issubset(tags))

    def test_areas_positive(self):
        self.assertTrue(num.all(self.domain.areas > 0))

    def test_total_area(self):
        num.testing.assert_allclose(self.domain.areas.sum(), 1.0, rtol=1e-10)

    def test_conserved_quantities_exist(self):
        for qty in ('stage', 'xmomentum', 'ymomentum'):
            self.assertIn(qty, self.domain.quantities)

    def test_set_quantity_and_retrieve(self):
        self.domain.set_quantity('stage', 2.0)
        num.testing.assert_allclose(
            self.domain.quantities['stage'].centroid_values, 2.0)


# ---------------------------------------------------------------------------
# Tests: Domain from raw coordinate/triangle arrays (no Mesh object)
# ---------------------------------------------------------------------------

class Test_Domain_from_raw_arrays(unittest.TestCase):

    def setUp(self):
        from anuga.abstract_2d_finite_volumes.mesh_factory import \
            rectangular_cross_with_neighbours
        m, n = 2, 2
        self.points, self.triangles, self.boundary, _ = \
            rectangular_cross_with_neighbours(m, n)
        self.m, self.n = m, n
        self.domain = Domain(self.points, self.triangles, self.boundary)

    def test_returns_domain_instance(self):
        self.assertIsInstance(self.domain, Domain)

    def test_number_of_elements(self):
        self.assertEqual(self.domain.number_of_elements,
                         4 * self.m * self.n)

    def test_boundary_is_dict(self):
        self.assertIsInstance(self.domain.boundary, dict)

    def test_areas_positive(self):
        self.assertTrue(num.all(self.domain.areas > 0))

    def test_total_area(self):
        num.testing.assert_allclose(self.domain.areas.sum(), 1.0, rtol=1e-10)

    def test_conserved_quantities_initialised(self):
        for qty in ('stage', 'xmomentum', 'ymomentum'):
            self.assertIn(qty, self.domain.quantities)


# ---------------------------------------------------------------------------
# Tests: Domain from a Pmesh-derived Mesh
# ---------------------------------------------------------------------------

class Test_Domain_from_pmesh_Mesh(unittest.TestCase):

    def setUp(self):
        self.domain, self.tsh = _make_domain_from_pmesh()

    def tearDown(self):
        os.remove(self.tsh)

    def test_returns_domain_instance(self):
        self.assertIsInstance(self.domain, Domain)

    def test_number_of_elements(self):
        self.assertEqual(self.domain.number_of_elements, 2)

    def test_number_of_nodes(self):
        self.assertEqual(self.domain.number_of_nodes, 4)

    def test_boundary_is_dict(self):
        self.assertIsInstance(self.domain.boundary, dict)

    def test_boundary_has_entries(self):
        self.assertGreater(len(self.domain.boundary), 0)

    def test_areas_positive(self):
        self.assertTrue(num.all(self.domain.areas > 0))

    def test_geo_reference_preserved(self):
        self.assertIsNotNone(self.domain.geo_reference)
        self.assertAlmostEqual(self.domain.geo_reference.xllcorner, 140.0)
        self.assertAlmostEqual(self.domain.geo_reference.yllcorner, 120.0)

    def test_conserved_quantities_exist(self):
        for qty in ('stage', 'xmomentum', 'ymomentum'):
            self.assertIn(qty, self.domain.quantities)

    def test_set_quantity_scalar(self):
        self.domain.set_quantity('friction', 0.03)
        num.testing.assert_allclose(
            self.domain.quantities['friction'].centroid_values, 0.03)


if __name__ == '__main__':
    unittest.main()
