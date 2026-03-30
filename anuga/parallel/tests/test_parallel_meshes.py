"""Tests for parallel_meshes strip-decomposition mesh builders.

These tests run in serial (single process) and validate the geometry,
boundary tags, neighbour connectivity, and communication dictionaries
produced by the parallel mesh functions by simulating multiple ranks
through the _column_distribution helper.
"""

import unittest
import numpy as num
import anuga.parallel.parallel_meshes as pm
import anuga.parallel.parallel_api as api


def _run_as_rank(func, myid, numprocs, *args, **kwargs):
    """Temporarily set myid/numprocs and call func(*args, **kwargs)."""
    old_myid, old_np = pm.myid, pm.numprocs
    api.myid, api.numprocs = myid, numprocs
    pm.myid,  pm.numprocs  = myid, numprocs
    try:
        return func(*args, **kwargs)
    finally:
        pm.myid,  pm.numprocs  = old_myid, old_np
        api.myid, api.numprocs = old_myid, old_np


class TestColumnDistribution(unittest.TestCase):

    def test_even_distribution(self):
        # 8 columns, 4 ranks → 2 each
        starts = []
        mms = []
        for rank in range(4):
            i_start, mm = pm._column_distribution(8, 4, rank)
            starts.append(i_start)
            mms.append(mm)
        self.assertEqual(starts, [0, 2, 4, 6])
        self.assertEqual(mms,    [2, 2, 2, 2])

    def test_remainder_distribution(self):
        # 10 columns, 4 ranks → ranks 0,1 get 3, ranks 2,3 get 2
        starts, mms = [], []
        for rank in range(4):
            i_start, mm = pm._column_distribution(10, 4, rank)
            starts.append(i_start)
            mms.append(mm)
        self.assertEqual(starts, [0, 3, 6, 8])
        self.assertEqual(mms,    [3, 3, 2, 2])
        # All columns covered
        self.assertEqual(sum(mms), 10)

    def test_single_rank(self):
        i_start, mm = pm._column_distribution(5, 1, 0)
        self.assertEqual(i_start, 0)
        self.assertEqual(mm, 5)


class TestParallelRectangleSerial(unittest.TestCase):
    """Serial (1-rank) checks."""

    def setUp(self):
        p, e, b, nb, ne, fsd, grd = _run_as_rank(
            pm.parallel_rectangular_with_neighbours, 0, 1, 3, 2)
        self.p, self.e, self.b = p, e, b
        self.nb, self.ne = nb, ne
        self.fsd, self.grd = fsd, grd

    def test_point_count(self):
        self.assertEqual(len(self.p), 4 * 3)   # (3+1)*(2+1)

    def test_triangle_count(self):
        self.assertEqual(len(self.e), 2 * 3 * 2)

    def test_boundary_tags(self):
        tags = set(self.b.values())
        self.assertEqual(tags, {'left', 'right', 'top', 'bottom'})

    def test_no_ghost_dicts(self):
        self.assertEqual(self.fsd, {})
        self.assertEqual(self.grd, {})

    def test_neighbour_shape(self):
        self.assertEqual(self.nb.shape, (12, 3))
        self.assertEqual(self.ne.shape, (12, 3))

    def test_all_internal_neighbours_valid(self):
        # Boundary edges have -1; all others should be valid triangle indices
        Nt = len(self.e)
        for tri in range(Nt):
            for edge in range(3):
                n = self.nb[tri, edge]
                self.assertTrue(n == -1 or (0 <= n < Nt),
                                f'tri={tri} edge={edge} has invalid neighbour {n}')

    def test_neighbour_reciprocity(self):
        Nt = len(self.e)
        for tri in range(Nt):
            for edge in range(3):
                nb_tri = self.nb[tri, edge]
                if nb_tri < 0:
                    continue
                nb_edge = self.ne[tri, edge]
                self.assertEqual(self.nb[nb_tri, nb_edge], tri,
                                 f'Reciprocity failed: tri={tri} edge={edge}')

    def test_points_in_unit_square(self):
        self.assertTrue(num.all(self.p[:, 0] >= 0))
        self.assertTrue(num.all(self.p[:, 0] <= 1))
        self.assertTrue(num.all(self.p[:, 1] >= 0))
        self.assertTrue(num.all(self.p[:, 1] <= 1))


class TestParallelCrossSerial(unittest.TestCase):
    """Serial (1-rank) cross mesh checks."""

    def setUp(self):
        p, e, b, nb, ne, fsd, grd = _run_as_rank(
            pm.parallel_rectangular_cross_with_neighbours, 0, 1, 3, 2)
        self.p, self.e, self.b = p, e, b
        self.nb, self.ne = nb, ne

    def test_point_count(self):
        self.assertEqual(len(self.p), 4 * 3 + 3 * 2)  # corners + centres

    def test_triangle_count(self):
        self.assertEqual(len(self.e), 4 * 3 * 2)

    def test_boundary_tags(self):
        self.assertEqual(set(self.b.values()), {'left', 'right', 'top', 'bottom'})

    def test_neighbour_shape(self):
        Nt = 4 * 3 * 2
        self.assertEqual(self.nb.shape, (Nt, 3))

    def test_neighbour_reciprocity(self):
        Nt = len(self.e)
        for tri in range(Nt):
            for edge in range(3):
                nb_tri = self.nb[tri, edge]
                if nb_tri < 0:
                    continue
                nb_edge = self.ne[tri, edge]
                self.assertEqual(self.nb[nb_tri, nb_edge], tri,
                                 f'Reciprocity failed: tri={tri} edge={edge}')


def _check_full_first(test_obj, nb, ne, ghost_recv_dict, Nt, n_full):
    """Assert full triangles occupy indices 0..n_full-1."""
    ghost_ids = set()
    for arr, _ in ghost_recv_dict.values():
        ghost_ids.update(arr.tolist())
    for idx in ghost_ids:
        test_obj.assertGreaterEqual(idx, n_full,
            f'Ghost triangle {idx} appears before full section (n_full={n_full})')
    for idx in range(n_full):
        test_obj.assertNotIn(idx, ghost_ids,
            f'Full triangle {idx} listed as ghost')


class TestParallelRectangle2Ranks(unittest.TestCase):
    """Two-rank decomposition of a 4×3 rectangular mesh."""

    M, N = 4, 3   # global cell counts

    def _mesh(self, rank):
        return _run_as_rank(
            pm.parallel_rectangular_with_neighbours, rank, 2,
            self.M, self.N)

    def test_triangle_counts(self):
        for rank in range(2):
            p, e, b, nb, ne, fsd, grd = self._mesh(rank)
            # each rank owns 2 full cols + 1 ghost col = 3 cols total
            self.assertEqual(len(e), 2 * 3 * self.N)

    def test_ghost_boundary_present(self):
        for rank in range(2):
            _, _, b, _, _, _, _ = self._mesh(rank)
            self.assertIn('ghost', b.values())

    def test_full_send_recv_symmetry(self):
        _, _, _, _, _, fsd0, grd0 = self._mesh(0)
        _, _, _, _, _, fsd1, grd1 = self._mesh(1)
        # rank 0 sends to rank 1; rank 1 receives from rank 0
        self.assertIn(1, fsd0)
        self.assertIn(0, grd1)
        # same triangle count in send/recv
        self.assertEqual(len(fsd0[1][0]), len(grd1[0][0]))

    def test_send_count(self):
        # full-right column = 1 column × N rows × 2 tri/cell
        _, _, _, _, _, fsd0, _ = self._mesh(0)
        self.assertEqual(len(fsd0[1][0]), 2 * self.N)

    def test_ghost_count(self):
        _, _, _, _, _, _, grd0 = self._mesh(0)
        # ghost-right: 1 col × N rows × 2 tri/cell
        self.assertEqual(len(grd0[1][0]), 2 * self.N)

    def test_no_left_ghost_rank0(self):
        _, _, _, _, _, fsd0, grd0 = self._mesh(0)
        self.assertNotIn(0 - 1, fsd0)  # no left neighbour
        self.assertNotIn(0 - 1, grd0)

    def test_no_right_ghost_last_rank(self):
        _, _, _, _, _, fsd1, grd1 = self._mesh(1)
        self.assertNotIn(2, fsd1)   # no right neighbour
        self.assertNotIn(2, grd1)

    def test_neighbour_reciprocity_rank0(self):
        _, e, _, nb, ne, _, _ = self._mesh(0)
        Nt = len(e)
        for tri in range(Nt):
            for edge in range(3):
                n = nb[tri, edge]
                if n < 0:
                    continue
                self.assertEqual(nb[n, ne[tri, edge]], tri)

    def test_full_send_matches_ghost_recv_indices(self):
        _, e0, _, _, _, fsd0, _ = self._mesh(0)
        _, e1, _, _, _, _, grd1 = self._mesh(1)
        send_count = len(fsd0[1][0])
        recv_count = len(grd1[0][0])
        self.assertEqual(send_count, recv_count)
        for idx in grd1[0][0]:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, len(e1))

    def test_full_triangles_first(self):
        for rank in range(2):
            _, e, _, nb, ne, fsd, grd = self._mesh(rank)
            _, mm = pm._column_distribution(self.M, 2, rank)
            n_full = 2 * mm * self.N
            _check_full_first(self, nb, ne, grd, len(e), n_full)


class TestParallelCross2Ranks(unittest.TestCase):
    """Two-rank decomposition of a 4×3 cross mesh."""

    M, N = 4, 3

    def _mesh(self, rank):
        return _run_as_rank(
            pm.parallel_rectangular_cross_with_neighbours, rank, 2,
            self.M, self.N)

    def test_triangle_counts(self):
        for rank in range(2):
            p, e, b, nb, ne, fsd, grd = self._mesh(rank)
            # 3 local cols (2 full + 1 ghost) × N rows × 4 tri/cell
            self.assertEqual(len(e), 4 * 3 * self.N)

    def test_send_count(self):
        _, _, _, _, _, fsd0, _ = self._mesh(0)
        self.assertEqual(len(fsd0[1][0]), 4 * self.N)

    def test_ghost_count(self):
        _, _, _, _, _, _, grd0 = self._mesh(0)
        self.assertEqual(len(grd0[1][0]), 4 * self.N)

    def test_ghost_boundary_present(self):
        for rank in range(2):
            _, _, b, _, _, _, _ = self._mesh(rank)
            self.assertIn('ghost', b.values())

    def test_neighbour_reciprocity_rank1(self):
        _, e, _, nb, ne, _, _ = self._mesh(1)
        Nt = len(e)
        for tri in range(Nt):
            for edge in range(3):
                n = nb[tri, edge]
                if n < 0:
                    continue
                self.assertEqual(nb[n, ne[tri, edge]], tri)

    def test_full_triangles_first(self):
        for rank in range(2):
            _, e, _, nb, ne, fsd, grd = self._mesh(rank)
            _, mm = pm._column_distribution(self.M, 2, rank)
            n_full = 4 * mm * self.N
            _check_full_first(self, nb, ne, grd, len(e), n_full)


class TestParallelRectangle3Ranks(unittest.TestCase):
    """Three-rank decomposition of an 8×4 rectangular mesh."""

    M, N = 8, 4

    def _mesh(self, rank):
        return _run_as_rank(
            pm.parallel_rectangular_with_neighbours, rank, 3,
            self.M, self.N)

    def test_total_full_triangles(self):
        total = 0
        for rank in range(3):
            _, mm = pm._column_distribution(self.M, 3, rank)
            total += 2 * mm * self.N
        self.assertEqual(total, 2 * self.M * self.N)

    def test_interior_rank_has_two_ghosts(self):
        p, e, b, nb, ne, fsd, grd = self._mesh(1)
        # rank 1 (interior): 2 ghost cols + full cols = M//3+1 full + 2 ghosts
        _, mm = pm._column_distribution(self.M, 3, 1)
        expected_m_local = mm + 2   # left + right ghost
        self.assertEqual(len(e), 2 * expected_m_local * self.N)
        self.assertIn(0, fsd)
        self.assertIn(2, fsd)

    def test_edge_ranks_have_one_ghost(self):
        for rank in (0, 2):
            p, e, b, nb, ne, fsd, grd = self._mesh(rank)
            _, mm = pm._column_distribution(self.M, 3, rank)
            expected_m_local = mm + 1   # only one ghost col
            self.assertEqual(len(e), 2 * expected_m_local * self.N)

    def test_neighbour_reciprocity_all_ranks(self):
        for rank in range(3):
            _, e, _, nb, ne, _, _ = self._mesh(rank)
            Nt = len(e)
            for tri in range(Nt):
                for edge in range(3):
                    n = nb[tri, edge]
                    if n < 0:
                        continue
                    self.assertEqual(nb[n, ne[tri, edge]], tri,
                                     f'rank={rank} tri={tri} edge={edge}')

    def test_full_triangles_first_all_ranks(self):
        for rank in range(3):
            _, e, _, nb, ne, fsd, grd = self._mesh(rank)
            _, mm = pm._column_distribution(self.M, 3, rank)
            n_full = 2 * mm * self.N
            _check_full_first(self, nb, ne, grd, len(e), n_full)


class TestDomainWrappers(unittest.TestCase):

    def test_rectangular_domain_triangle_count(self):
        from anuga.parallel.parallel_meshes import parallel_rectangular_domain
        d = parallel_rectangular_domain(4, 3)
        self.assertEqual(d.get_number_of_triangles(), 2 * 4 * 3)

    def test_cross_domain_triangle_count(self):
        from anuga.parallel.parallel_meshes import parallel_rectangular_cross_domain
        d = parallel_rectangular_cross_domain(4, 3)
        self.assertEqual(d.get_number_of_triangles(), 4 * 4 * 3)

    def test_rectangular_domain_geometry(self):
        from anuga.parallel.parallel_meshes import parallel_rectangular_domain
        d = parallel_rectangular_domain(4, 3, len1_g=2.0, len2_g=1.5,
                                         origin_g=(1.0, 0.5))
        verts = d.mesh.get_vertex_coordinates()
        self.assertAlmostEqual(verts[:, 0].min(), 1.0)
        self.assertAlmostEqual(verts[:, 0].max(), 3.0)
        self.assertAlmostEqual(verts[:, 1].min(), 0.5)
        self.assertAlmostEqual(verts[:, 1].max(), 2.0)


if __name__ == '__main__':
    unittest.main()
