"""Tests for mesh_factory.py and mesh_factory_ext.pyx

Covers rectangular_with_neighbours and rectangular_cross_with_neighbours,
verifying point coordinates, element connectivity, boundary tags,
and neighbour arrays produced by the Cython extensions.
"""

import unittest
import numpy as num


class Test_rectangular_with_neighbours(unittest.TestCase):

    def _call(self, m, n, len1=1.0, len2=1.0, origin=(0.0, 0.0)):
        from anuga.abstract_2d_finite_volumes.mesh_factory import rectangular_with_neighbours
        return rectangular_with_neighbours(m, n, len1, len2, origin)

    # ------------------------------------------------------------------
    # Basic shape checks
    # ------------------------------------------------------------------

    def test_array_shapes(self):
        m, n = 3, 4
        points, elements, boundary, neighbours = self._call(m, n)
        self.assertEqual(points.shape,     ((m+1)*(n+1), 2))
        self.assertEqual(elements.shape,   (2*m*n, 3))
        self.assertEqual(neighbours.shape, (2*m*n, 3))

    def test_1x1_mesh(self):
        points, elements, boundary, neighbours = self._call(1, 1)
        self.assertEqual(points.shape,   (4, 2))
        self.assertEqual(elements.shape, (2, 3))

    # ------------------------------------------------------------------
    # Point coordinates
    # ------------------------------------------------------------------

    def test_unit_square_corners(self):
        points, _, _, _ = self._call(2, 2)
        # index(i,j) = j + i*(n+1), so corner indices are:
        # (0,0)->0, (0,2)->2, (2,0)->6, (2,2)->8
        num.testing.assert_allclose(points[0],  [0.0, 0.0])
        num.testing.assert_allclose(points[2],  [0.0, 1.0])
        num.testing.assert_allclose(points[6],  [1.0, 0.0])
        num.testing.assert_allclose(points[8],  [1.0, 1.0])

    def test_origin_offset(self):
        points, _, _, _ = self._call(2, 2, origin=(5.0, 10.0))
        num.testing.assert_allclose(points[0], [5.0, 10.0])
        num.testing.assert_allclose(points[8], [6.0, 11.0])

    def test_non_unit_lengths(self):
        points, _, _, _ = self._call(2, 3, len1=4.0, len2=6.0)
        # point at index(2,3) = 3 + 2*4 = 11
        num.testing.assert_allclose(points[11], [4.0, 6.0])

    def test_point_spacing(self):
        m, n = 4, 3
        len1, len2 = 2.0, 3.0
        points, _, _, _ = self._call(m, n, len1=len1, len2=len2)
        delta1 = len1 / m
        delta2 = len2 / n
        for i in range(m+1):
            for j in range(n+1):
                idx = j + i*(n+1)
                num.testing.assert_allclose(points[idx, 0], i*delta1, rtol=1e-12)
                num.testing.assert_allclose(points[idx, 1], j*delta2, rtol=1e-12)

    # ------------------------------------------------------------------
    # Element vertex indices
    # ------------------------------------------------------------------

    def test_element_vertices_in_range(self):
        m, n = 3, 4
        points, elements, _, _ = self._call(m, n)
        np_pts = (m+1)*(n+1)
        self.assertTrue(num.all(elements >= 0))
        self.assertTrue(num.all(elements < np_pts))

    def test_element_vertices_1x1(self):
        # index(i,j)=j+i*2: a=0,b=1,c=2,d=3
        # lower [c,d,a]=[2,3,0], upper [b,a,d]=[1,0,3]
        _, elements, _, _ = self._call(1, 1)
        num.testing.assert_array_equal(elements[0], [2, 3, 0])  # lower
        num.testing.assert_array_equal(elements[1], [1, 0, 3])  # upper

    def test_no_degenerate_elements(self):
        """All three vertices of every element must be distinct."""
        m, n = 5, 5
        _, elements, _, _ = self._call(m, n)
        for k, tri in enumerate(elements):
            self.assertEqual(len(set(tri)), 3, f"Degenerate element {k}: {tri}")

    # ------------------------------------------------------------------
    # Boundary tags
    # ------------------------------------------------------------------

    def test_boundary_tag_counts(self):
        m, n = 3, 4
        _, _, boundary, _ = self._call(m, n)
        bottom = [v for v in boundary.values() if v == 'bottom']
        top    = [v for v in boundary.values() if v == 'top']
        left   = [v for v in boundary.values() if v == 'left']
        right  = [v for v in boundary.values() if v == 'right']
        self.assertEqual(len(bottom), m)
        self.assertEqual(len(top),    m)
        self.assertEqual(len(left),   n)
        self.assertEqual(len(right),  n)

    def test_boundary_edge_indices(self):
        """Boundary edges must use edge index 1 (bottom/top) or 2 (left/right)."""
        m, n = 2, 2
        _, _, boundary, _ = self._call(m, n)
        for (elem, edge), tag in boundary.items():
            if tag in ('bottom', 'top'):
                self.assertEqual(edge, 1, f"Expected edge 1 for {tag}, got {edge}")
            else:
                self.assertEqual(edge, 2, f"Expected edge 2 for {tag}, got {edge}")

    def test_boundary_element_indices_in_range(self):
        m, n = 3, 3
        _, elements, boundary, _ = self._call(m, n)
        for (elem, edge), tag in boundary.items():
            self.assertGreaterEqual(elem, 0)
            self.assertLess(elem, len(elements))

    # ------------------------------------------------------------------
    # Neighbour array
    # ------------------------------------------------------------------

    def test_neighbours_shape(self):
        m, n = 3, 4
        _, _, _, neighbours = self._call(m, n)
        self.assertEqual(neighbours.shape, (2*m*n, 3))

    def test_neighbours_values_in_range(self):
        """All neighbour entries are either -1 or a valid element index."""
        m, n = 4, 5
        _, _, _, neighbours = self._call(m, n)
        Nt = 2*m*n
        for val in neighbours.flat:
            self.assertTrue(val == -1 or (0 <= val < Nt),
                            f"Neighbour value {val} out of range")

    def test_neighbours_symmetry(self):
        """If element A lists B as a neighbour on some edge, B must list A."""
        m, n = 4, 5
        _, _, _, neighbours = self._call(m, n)
        Nt = 2*m*n
        for a in range(Nt):
            for edge in range(3):
                b = neighbours[a, edge]
                if b != -1:
                    self.assertIn(a, neighbours[b],
                                  f"Asymmetric: {a}->edge{edge}->{b} but {b} doesn't list {a}")

    def test_boundary_edges_are_minus_one(self):
        """Edges tagged as boundary must have neighbour -1."""
        m, n = 3, 4
        _, _, boundary, neighbours = self._call(m, n)
        for (elem, edge), tag in boundary.items():
            self.assertEqual(neighbours[elem, edge], -1,
                             f"Boundary edge ({elem},{edge}) tag={tag} "
                             f"has neighbour {neighbours[elem, edge]}, expected -1")

    def test_internal_edges_not_minus_one(self):
        """Internal edges (not in boundary dict) must have a valid neighbour."""
        m, n = 3, 4
        _, _, boundary, neighbours = self._call(m, n)
        Nt = 2*m*n
        for elem in range(Nt):
            for edge in range(3):
                if (elem, edge) not in boundary:
                    nb = neighbours[elem, edge]
                    self.assertNotEqual(nb, -1,
                                        f"Internal edge ({elem},{edge}) has neighbour -1")
                    self.assertGreaterEqual(nb, 0)
                    self.assertLess(nb, Nt)

    def test_1x1_neighbours(self):
        # m=n=1: lower=0, upper=1
        # lower [c,d,a]: edge0->upper(1), edge1->-1(bottom), edge2->-1(right)
        # upper [b,a,d]: edge0->lower(0), edge1->-1(top),    edge2->-1(left)
        _, _, _, neighbours = self._call(1, 1)
        num.testing.assert_array_equal(neighbours[0], [ 1, -1, -1])
        num.testing.assert_array_equal(neighbours[1], [ 0, -1, -1])

    def test_2x2_diagonal_shared(self):
        """In every cell the two triangles must be mutual edge-0 neighbours."""
        m, n = 2, 2
        _, _, _, neighbours = self._call(m, n)
        for i in range(m):
            for j in range(n):
                lower = 2*(i*n+j)
                upper = lower + 1
                self.assertEqual(neighbours[lower, 0], upper)
                self.assertEqual(neighbours[upper, 0], lower)


class Test_rectangular_cross_with_neighbours(unittest.TestCase):

    def _call(self, m, n, len1=1.0, len2=1.0, origin=(0.0, 0.0)):
        from anuga.abstract_2d_finite_volumes.mesh_factory import rectangular_cross_with_neighbours
        return rectangular_cross_with_neighbours(m, n, len1, len2, origin)

    # ------------------------------------------------------------------
    # Basic shape checks
    # ------------------------------------------------------------------

    def test_array_shapes(self):
        m, n = 3, 4
        points, elements, boundary, neighbours = self._call(m, n)
        self.assertEqual(points.shape,     ((m+1)*(n+1) + m*n, 2))
        self.assertEqual(elements.shape,   (4*m*n, 3))
        self.assertEqual(neighbours.shape, (4*m*n, 3))

    def test_1x1_mesh(self):
        points, elements, boundary, neighbours = self._call(1, 1)
        self.assertEqual(points.shape,   (5, 2))
        self.assertEqual(elements.shape, (4, 3))

    # ------------------------------------------------------------------
    # Point coordinates
    # ------------------------------------------------------------------

    def test_unit_square_corners(self):
        points, _, _, _ = self._call(2, 2)
        # Corner grid points: vertices[i,j] = j + i*(n+1) (as in rectangular_cross_construct)
        # For m=n=2 corners: (0,0)->0, (0,2)->2, (2,0)->6, (2,2)->8
        num.testing.assert_allclose(points[0], [0.0, 0.0], atol=1e-12)
        num.testing.assert_allclose(points[2], [0.0, 1.0], atol=1e-12)
        num.testing.assert_allclose(points[6], [1.0, 0.0], atol=1e-12)
        num.testing.assert_allclose(points[8], [1.0, 1.0], atol=1e-12)

    def test_centre_points_are_cell_centroids(self):
        m, n = 2, 3
        len1, len2 = 2.0, 3.0
        points, _, _, _ = self._call(m, n, len1=len1, len2=len2)
        delta1 = len1 / m
        delta2 = len2 / n
        # Centre points start at index (m+1)*(n+1)
        cp_start = (m+1)*(n+1)
        for i in range(m):
            for j in range(n):
                cx = (i + 0.5) * delta1
                cy = (j + 0.5) * delta2
                idx = cp_start + i*n + j
                num.testing.assert_allclose(points[idx, 0], cx, rtol=1e-12,
                                            err_msg=f"Centre x wrong for cell ({i},{j})")
                num.testing.assert_allclose(points[idx, 1], cy, rtol=1e-12,
                                            err_msg=f"Centre y wrong for cell ({i},{j})")

    def test_origin_offset(self):
        points, _, _, _ = self._call(1, 1, origin=(3.0, 7.0))
        num.testing.assert_allclose(points[0], [3.0, 7.0], atol=1e-12)

    # ------------------------------------------------------------------
    # Element vertices
    # ------------------------------------------------------------------

    def test_element_vertices_in_range(self):
        m, n = 3, 4
        points, elements, _, _ = self._call(m, n)
        np_pts = (m+1)*(n+1) + m*n
        self.assertTrue(num.all(elements >= 0))
        self.assertTrue(num.all(elements < np_pts))

    def test_no_degenerate_elements(self):
        m, n = 4, 5
        _, elements, _, _ = self._call(m, n)
        for k, tri in enumerate(elements):
            self.assertEqual(len(set(tri)), 3, f"Degenerate element {k}: {tri}")

    # ------------------------------------------------------------------
    # Boundary tags
    # ------------------------------------------------------------------

    def test_boundary_tag_counts(self):
        m, n = 3, 4
        _, _, boundary, _ = self._call(m, n)
        bottom = [v for v in boundary.values() if v == 'bottom']
        top    = [v for v in boundary.values() if v == 'top']
        left   = [v for v in boundary.values() if v == 'left']
        right  = [v for v in boundary.values() if v == 'right']
        self.assertEqual(len(bottom), m)
        self.assertEqual(len(top),    m)
        self.assertEqual(len(left),   n)
        self.assertEqual(len(right),  n)

    def test_all_boundary_edges_use_edge_1(self):
        m, n = 2, 3
        _, _, boundary, _ = self._call(m, n)
        for (elem, edge), tag in boundary.items():
            self.assertEqual(edge, 1,
                             f"Expected all boundary edges to be index 1, "
                             f"got {edge} for ({elem},{edge})={tag}")

    def test_boundary_element_indices_in_range(self):
        m, n = 3, 3
        _, elements, boundary, _ = self._call(m, n)
        for (elem, edge), tag in boundary.items():
            self.assertGreaterEqual(elem, 0)
            self.assertLess(elem, len(elements))

    # ------------------------------------------------------------------
    # Neighbour array
    # ------------------------------------------------------------------

    def test_neighbours_values_in_range(self):
        m, n = 4, 5
        _, _, _, neighbours = self._call(m, n)
        Nt = 4*m*n
        for val in neighbours.flat:
            self.assertTrue(val == -1 or (0 <= val < Nt),
                            f"Neighbour value {val} out of range")

    def test_neighbours_symmetry(self):
        m, n = 4, 5
        _, _, _, neighbours = self._call(m, n)
        Nt = 4*m*n
        for a in range(Nt):
            for edge in range(3):
                b = neighbours[a, edge]
                if b != -1:
                    self.assertIn(a, neighbours[b],
                                  f"Asymmetric: {a}->edge{edge}->{b} but {b} doesn't list {a}")

    def test_boundary_edges_are_minus_one(self):
        m, n = 3, 4
        _, _, boundary, neighbours = self._call(m, n)
        for (elem, edge), tag in boundary.items():
            self.assertEqual(neighbours[elem, edge], -1,
                             f"Boundary edge ({elem},{edge}) tag={tag} "
                             f"has neighbour {neighbours[elem, edge]}, expected -1")

    def test_internal_edges_not_minus_one(self):
        m, n = 3, 4
        _, _, boundary, neighbours = self._call(m, n)
        Nt = 4*m*n
        for elem in range(Nt):
            for edge in range(3):
                if (elem, edge) not in boundary:
                    nb = neighbours[elem, edge]
                    self.assertNotEqual(nb, -1,
                                        f"Internal edge ({elem},{edge}) has neighbour -1")
                    self.assertGreaterEqual(nb, 0)
                    self.assertLess(nb, Nt)

    def test_1x1_neighbours(self):
        # 4 triangles: base+0=left, base+1=bottom, base+2=right, base+3=top
        # All external edges (edge 1) should be -1
        # Internal edges connect within the cell
        _, _, _, neighbours = self._call(1, 1)
        # left (0): edge0->top(3), edge1->-1, edge2->bottom(1)
        num.testing.assert_array_equal(neighbours[0], [3, -1, 1])
        # bottom (1): edge0->left(0), edge1->-1, edge2->right(2)
        num.testing.assert_array_equal(neighbours[1], [0, -1, 2])
        # right (2): edge0->bottom(1), edge1->-1, edge2->top(3)
        num.testing.assert_array_equal(neighbours[2], [1, -1, 3])
        # top (3): edge0->right(2), edge1->-1, edge2->left(0)
        num.testing.assert_array_equal(neighbours[3], [2, -1, 0])

    def test_internal_cell_neighbours_symmetric(self):
        """Within every cell the 4 triangles form a consistent ring."""
        m, n = 3, 4
        _, _, _, neighbours = self._call(m, n)
        for i in range(m):
            for j in range(n):
                base = 4*(i*n+j)
                # left(0) edge0 -> top(3)
                self.assertEqual(neighbours[base+0, 0], base+3)
                # left(0) edge2 -> bottom(1)
                self.assertEqual(neighbours[base+0, 2], base+1)
                # bottom(1) edge0 -> left(0)
                self.assertEqual(neighbours[base+1, 0], base+0)
                # bottom(1) edge2 -> right(2)
                self.assertEqual(neighbours[base+1, 2], base+2)
                # right(2) edge0 -> bottom(1)
                self.assertEqual(neighbours[base+2, 0], base+1)
                # right(2) edge2 -> top(3)
                self.assertEqual(neighbours[base+2, 2], base+3)
                # top(3) edge0 -> right(2)
                self.assertEqual(neighbours[base+3, 0], base+2)
                # top(3) edge2 -> left(0)
                self.assertEqual(neighbours[base+3, 2], base+0)

    def test_cross_cell_neighbours(self):
        """Check cross-cell neighbour links for an interior mesh."""
        m, n = 3, 4
        _, _, _, neighbours = self._call(m, n)
        # Pick an interior cell (1,1): base=4*(1*4+1)=20
        i, j = 1, 1
        base       = 4*(i*n+j)
        base_left  = 4*((i-1)*n+j)   # cell (0,1)
        base_right = 4*((i+1)*n+j)   # cell (2,1)
        base_below = 4*(i*n+(j-1))   # cell (1,0)
        base_above = 4*(i*n+(j+1))   # cell (1,2)

        # left tri (base+0) edge1 -> right tri (base_left+2)
        self.assertEqual(neighbours[base+0, 1], base_left+2)
        # right tri (base+2) edge1 -> left tri (base_right+0)
        self.assertEqual(neighbours[base+2, 1], base_right+0)
        # bottom tri (base+1) edge1 -> top tri (base_below+3)
        self.assertEqual(neighbours[base+1, 1], base_below+3)
        # top tri (base+3) edge1 -> bottom tri (base_above+1)
        self.assertEqual(neighbours[base+3, 1], base_above+1)


class Test_rectangular_cross_matches_python(unittest.TestCase):
    """Verify the Cython rectangular_cross output matches the pure-Python version."""

    def test_points_match(self):
        from anuga.abstract_2d_finite_volumes.mesh_factory import (
            rectangular_cross_python, rectangular_cross_with_neighbours)
        m, n = 3, 4
        pts_py, _, _, = rectangular_cross_python(m, n)
        pts_cy, _, _, _ = rectangular_cross_with_neighbours(m, n)
        pts_py = num.array(pts_py)
        num.testing.assert_allclose(pts_cy, pts_py, rtol=1e-12)

    def test_elements_match(self):
        from anuga.abstract_2d_finite_volumes.mesh_factory import (
            rectangular_cross_python, rectangular_cross_with_neighbours)
        m, n = 3, 4
        _, elems_py, _ = rectangular_cross_python(m, n)
        _, elems_cy, _, _ = rectangular_cross_with_neighbours(m, n)
        elems_py = num.array(elems_py)
        num.testing.assert_array_equal(elems_cy, elems_py)

    def test_boundary_match(self):
        from anuga.abstract_2d_finite_volumes.mesh_factory import (
            rectangular_cross_python, rectangular_cross_with_neighbours)
        m, n = 3, 4
        _, _, bnd_py = rectangular_cross_python(m, n)
        _, _, bnd_cy, _ = rectangular_cross_with_neighbours(m, n)
        self.assertEqual(bnd_cy, bnd_py)


if __name__ == '__main__':
    unittest.main()
