"""Bedload transport [K-1]-[K-4] and its bed evolution [G-5] (PHYSICS_SPEC 6).

Bedload is a DIVERGENCE, not a source: it moves sediment ALONG the bed rather
than between bed and water column. So its defining property is that in a closed
domain it redistributes bed material and conserves total bed volume EXACTLY.
That is what separates a correct divergence from a plausible one, and it is why
the flux is computed per shared edge rather than from a reconstructed gradient.

The mode 1 / mode 2 comparison is in test_sediment_gpu.py.
"""
import numpy as np
import pytest

from anuga import Reflective_boundary, rectangular_cross_domain

LEN = 100.0


def channel(slope=0.02, depth=0.5, n_manning=0.025, dt=0.5):
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


def _gravel_channel(formula='wong_parker_eq24', concentration=0.0):
    d = channel()
    d.add_grain_size(name='gravel', diameter=5e-3, tau_c_star=0.0,
                     initial_concentration=concentration)
    d.set_bedload(formula)
    return d


@pytest.mark.parametrize('formula,expected', [
    ('wong_parker_eq24', (3.97, 1.5, 0.0495)),
    ('wong_parker_eq23', (4.93, 1.60, 0.0470)),
])
def test_the_published_parameter_sets(formula, expected):
    """Wong & Parker's corrected Meyer-Peter-Muller fits. Which of the two FG21
    used is not settled, so both are available."""
    d = _gravel_channel(formula)
    assert (d.sediment_bedload_K, d.sediment_bedload_m,
            d.sediment_bedload_tau_c_star) == expected


def test_an_unknown_formula_is_rejected():
    with pytest.raises(ValueError):
        channel().set_bedload('mpm')


@pytest.fixture(scope='module')
def transported():
    d = _gravel_channel()
    z0 = d.quantities['elevation'].centroid_values.copy()
    vol0 = float((z0 * d.areas).sum())
    d.evolve_to_end(finaltime=60.0)
    z1 = d.quantities['elevation'].centroid_values
    return d, z0, z1, vol0, float((z1 * d.areas).sum())


def test_bedload_moves_the_bed(transported):
    _, z0, z1, _, _ = transported
    assert np.abs(z1 - z0).max() > 0.0


def test_total_bed_volume_is_conserved_exactly(transported):
    """The property that defines a divergence. FG21 report ~1% loss in their
    bedload operator, which they could not correct; computing the flux as a
    difference across shared edges makes conservation structural instead."""
    _, _, _, vol0, vol1 = transported
    assert abs(vol1 - vol0) <= 1e-10 * max(abs(vol0), 1.0)


def test_it_redistributes_rather_than_only_eroding(transported):
    _, z0, z1, _, _ = transported
    dz = z1 - z0
    assert dz.min() < 0.0 < dz.max()


def test_it_erodes_upstream_and_builds_downstream(transported):
    d, z0, z1, _, _ = transported
    dz = z1 - z0
    x = d.centroid_coordinates[:, 0]
    up, dn = x < LEN * 0.25, x > LEN * 0.75
    assert dz[up].sum() < 0.0 < dz[dn].sum()


def test_engelund_hansen_disables_the_suspended_source():
    """[K-5] is TOTAL load: it already contains suspension, so running the
    suspended source alongside it double counts (spec 6's usage rule)."""
    d = _gravel_channel('engelund_hansen', concentration=0.01)
    assert d._sediment_suspended_enabled is False
    assert d.sediment_bedload_tau_c_star == 0.0, 'total load has no threshold'


def test_under_engelund_hansen_suspended_mass_is_advected_not_exchanged():
    """With the source off, m is advected but never exchanged, so in a closed
    domain its total is conserved. Were the suspended operator still running
    alongside [K-5] -- the double counting spec 6 forbids -- this would change.
    """
    d = _gravel_channel('engelund_hansen', concentration=0.01)
    m0 = float((d.tracer_conserved_values[0] * d.areas).sum())
    d.evolve_to_end(finaltime=20.0)
    m1 = float((d.tracer_conserved_values[0] * d.areas).sum())
    assert abs(m1 - m0) <= 1e-9 * max(abs(m0), 1.0)
    assert np.isfinite(d.quantities['elevation'].centroid_values).all()
