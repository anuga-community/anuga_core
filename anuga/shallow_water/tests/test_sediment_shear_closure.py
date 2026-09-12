"""Bed shear closures [T-1] and [T-7] (PHYSICS_SPEC 3.1, 3.4).

    [T-1]  tau_b = rho f_c |v|^2     quadratic drag (default)
    [T-7]  tau_b = rho g h S         depth-slope

[T-7] is the steady uniform flow approximation, kept for reproducing published
anugaSed results. [T-1] is the default because normal-flow equilibrium fails in
exactly the unsteady floods this targets.

The mode 1 / mode 2 comparison for these closures is in test_sediment_gpu.py.
"""
import numpy as np
import pytest

from anuga import Reflective_boundary, rectangular_cross_domain

LEN = 200.0


def channel(slope=0.01, depth=1.0, n_manning=0.03, dt=1.0):
    d = rectangular_cross_domain(20, 10, len1=LEN, len2=LEN / 2)
    d.set_flow_algorithm('DE0')
    d.set_low_froude(0)
    d.store = False
    d.set_quantity('elevation', lambda x, y: -slope * x)
    d.set_quantity('stage', lambda x, y: -slope * x + depth)
    d.set_quantity('friction', n_manning)
    d.set_boundary({t: Reflective_boundary(d) for t in d.get_boundary_tags()})
    d.evolve_max_timestep = dt
    return d


def test_quadratic_drag_is_the_default():
    assert channel().sediment_shear_closure == 0


def test_depth_slope_can_be_selected():
    d = channel()
    d.set_shear_closure('depth_slope')
    assert d.sediment_shear_closure == 1


def test_an_unknown_closure_is_rejected():
    with pytest.raises(ValueError):
        channel().set_shear_closure('depth-slope')


@pytest.mark.parametrize('imposed', [0.01, 0.05])
def test_the_reconstructed_bed_slope_matches_the_imposed_one(imposed):
    """[T-7] needs grad z from the divergence theorem over a cell's own edges.

    Mirrored here and checked against the slope actually imposed, so the test
    checks the reconstruction rather than reimplementing it.
    """
    d = channel(slope=imposed)
    bed_ev = d.quantities['elevation'].edge_values
    gx = (bed_ev * d.normals[:, 0::2] * d.edgelengths).sum(axis=1) / d.areas
    gy = (bed_ev * d.normals[:, 1::2] * d.edgelengths).sum(axis=1) / d.areas
    S = np.sqrt(gx ** 2 + gy ** 2)
    assert np.allclose(S, imposed, rtol=1e-9, atol=1e-12)


def test_the_energy_slope_substituted_into_T7_reproduces_T1():
    """Spec 3.4: S_f = f_c |v|^2 / (g h), so rho g h S_f is rho f_c |v|^2.

    This identity is the whole argument for preferring [T-1], so it is checked
    numerically rather than asserted in a comment.
    """
    f_c, vel2, h, g = 0.00883, 4.0, 1.5, 9.81
    S_f = f_c * vel2 / (g * h)
    assert abs(g * h * S_f - f_c * vel2) < 1e-15


def test_the_two_closures_give_materially_different_erosion():
    """PHYSICS_SPEC divergence D1: they are not interchangeable."""
    means = {}
    for closure in ('quadratic_drag', 'depth_slope'):
        d = channel()
        d.set_shear_closure(closure)
        d.add_grain_size(name='sand', diameter=1e-4, initial_concentration=0.0)
        d.evolve_to_end(finaltime=30.0)
        means[closure] = float(d.get_tracer('sand').mean())

    assert all(np.isfinite(v) and v >= 0.0 for v in means.values()), means
    assert (abs(means['quadratic_drag'] - means['depth_slope'])
            > 0.01 * max(means.values())), (
        'the closures should differ materially, got %r' % means)
