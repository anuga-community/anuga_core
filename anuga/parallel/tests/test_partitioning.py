import pytest
import numpy as np

from anuga.parallel.partitioning import morton_order_from_points
from anuga.parallel.partitioning import hilbert_order_from_points
from anuga.parallel.partitioning import bfs_order_from_neighbours
from anuga.parallel.partitioning import bfs_partition

def test_morton_order_from_points_basic():
    """Test basic Morton ordering with simple points."""
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    order = morton_order_from_points(points)
    assert len(order) == 3
    assert np.all(np.isin(order, [0, 1, 2]))


def test_morton_order_from_points_sorted():
    """Test that returned order is valid permutation."""
    points = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
    order = morton_order_from_points(points)
    assert np.array_equal(np.sort(order), np.arange(len(points)))


def test_morton_order_from_points_single_point():
    """Test with single point."""
    points = np.array([[0.5, 0.5]])
    order = morton_order_from_points(points)
    assert len(order) == 1
    assert order[0] == 0



def test_morton_order_from_points_invalid_shape():
    """Test error handling for invalid point shapes."""
    with pytest.raises(ValueError, match="points must have shape"):
        morton_order_from_points(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="points must have shape"):
        morton_order_from_points(np.array([[0.0, 1.0, 0.5]]))


def test_morton_order_from_points_degenerate():
    """Test with degenerate points (zero span in one direction)."""
    points = np.array([[0.5, 0.0], [0.5, 1.0], [0.5, 0.5]])
    order = morton_order_from_points(points)
    assert len(order) == 3
    assert np.array_equal(np.sort(order), np.arange(3))


def test_morton_order_from_points_large_array():
    """Test with larger array of points."""
    rng = np.random.default_rng(42)
    points = rng.random((100, 2))
    order = morton_order_from_points(points)
    assert len(order) == 100
    assert np.array_equal(np.sort(order), np.arange(100))


def test_morton_order_from_points_type_conversion():
    """Test that function converts input to float64."""
    points = np.array([[0, 0], [1, 1]], dtype=np.int32)
    order = morton_order_from_points(points)
    assert len(order) == 2
    assert np.array_equal(np.sort(order), np.arange(2))


def test_hilbert_order_from_points_basic():
    """Test basic Hilbert ordering with simple points."""
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    order = hilbert_order_from_points(points)
    assert len(order) == 3
    assert np.all(np.isin(order, [0, 1, 2]))


def test_hilbert_order_from_points_sorted():
    """Test that returned order is valid permutation."""
    points = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
    order = hilbert_order_from_points(points)
    assert np.array_equal(np.sort(order), np.arange(len(points)))


def test_hilbert_order_from_points_single_point():
    """Test with single point."""
    points = np.array([[0.5, 0.5]])
    order = hilbert_order_from_points(points)
    assert len(order) == 1
    assert order[0] == 0


def test_hilbert_order_from_points_invalid_shape():
    """Test error handling for invalid point shapes."""
    with pytest.raises(ValueError, match="points must have shape"):
        hilbert_order_from_points(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="points must have shape"):
        hilbert_order_from_points(np.array([[0.0, 1.0, 0.5]]))


def test_hilbert_order_from_points_degenerate():
    """Test with degenerate points (zero span in one direction)."""
    points = np.array([[0.5, 0.0], [0.5, 1.0], [0.5, 0.5]])
    order = hilbert_order_from_points(points)
    assert len(order) == 3
    assert np.array_equal(np.sort(order), np.arange(3))


def test_hilbert_order_from_points_large_array():
    """Test with larger array of points."""
    rng = np.random.default_rng(42)
    points = rng.random((100, 2))
    order = hilbert_order_from_points(points)
    assert len(order) == 100
    assert np.array_equal(np.sort(order), np.arange(100))


def test_hilbert_order_from_points_type_conversion():
    """Test that function converts input to float64."""
    points = np.array([[0, 0], [1, 1]], dtype=np.int32)
    order = hilbert_order_from_points(points)
    assert len(order) == 2
    assert np.array_equal(np.sort(order), np.arange(2))


def test_hilbert_order_from_points_custom_p():
    """Test with custom precision parameter."""
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    order = hilbert_order_from_points(points, p=8)
    assert len(order) == 3
    assert np.array_equal(np.sort(order), np.arange(3))


def test_hilbert_order_from_points_all_same():
    """Test with all identical points."""
    points = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    order = hilbert_order_from_points(points)
    assert len(order) == 3
    assert np.array_equal(np.sort(order), np.arange(3))


def test_bfs_order_from_neighbours_chain():
    """BFS order follows neighbour adjacency and skips boundary markers."""
    neighbours = np.array([
        [1, -1, -1],
        [0, 2, -1],
        [1, 3, -1],
        [2, -1, -1],
    ])
    order = bfs_order_from_neighbours(neighbours)
    assert np.array_equal(order, np.arange(4))


def test_bfs_order_from_neighbours_disconnected_seed_order():
    """Disconnected components are started in seed_order order."""
    neighbours = np.array([
        [1, -1, -1],
        [0, -1, -1],
        [3, -1, -1],
        [2, -1, -1],
    ])
    order = bfs_order_from_neighbours(neighbours, seed_order=[2, 3, 0, 1])
    assert np.array_equal(order, np.array([2, 3, 0, 1]))


def test_bfs_order_from_neighbours_invalid_inputs():
    """BFS ordering validates neighbour and seed array shapes."""
    with pytest.raises(ValueError, match="neighbours must be a 2D array"):
        bfs_order_from_neighbours(np.array([0, 1, 2]))
    with pytest.raises(ValueError, match="seed_order must have length"):
        bfs_order_from_neighbours(np.array([[1], [0]]), seed_order=[0])
    with pytest.raises(ValueError, match="seed_order must be a permutation"):
        bfs_order_from_neighbours(np.array([[1], [0]]), seed_order=[0, 0])


def test_bfs_partition_rectangular_basic_mesh():
    """BFS partition returns a complete ordering and balanced partitions."""
    from anuga.abstract_2d_finite_volumes.basic_mesh import (
        rectangular_cross_basic_mesh)

    bm = rectangular_cross_basic_mesh(4, 3)
    order, triangles_per_proc = bfs_partition(bm, 5)

    assert np.array_equal(np.sort(order), np.arange(bm.number_of_triangles))
    assert triangles_per_proc.sum() == bm.number_of_triangles
    assert np.max(triangles_per_proc) - np.min(triangles_per_proc) <= 1


def test_bfs_partition_rejects_too_many_processors():
    """BFS partition follows the existing partition size contract."""
    from anuga.abstract_2d_finite_volumes.basic_mesh import (
        rectangular_basic_mesh)

    bm = rectangular_basic_mesh(1, 1)
    with pytest.raises(ValueError, match="Number of processors"):
        bfs_partition(bm, bm.number_of_triangles + 1)
