"""Tests of Normal_depth_boundary: outflow as uniform flow on a given slope."""

import warnings

import numpy as np
import pytest

import anuga

SLOPE = 0.001
MANNING = 0.03
DEPTH = 1.0
LENGTH = 400.0
WIDTH = 20.0
ALGORITHMS = ['DE0', 'DE1', 'DE_ader2']


def uniform_discharge(depth=DEPTH, slope=SLOPE, manning=MANNING):
    """Discharge per unit width of uniform flow (Manning)."""
    return depth**(5.0/3.0)*np.sqrt(slope)/manning


def channel(algorithm='DE1', boundary_slope=SLOPE, friction=MANNING,
            mode=None, **boundary_kwargs):
    """A sloping channel in uniform flow, with the boundary at its outlet."""

    domain = anuga.rectangular_cross_domain(40, 2, len1=LENGTH, len2=WIDTH)
    if mode is not None:
        domain.set_compute_mode(mode)
    domain.set_flow_algorithm(algorithm)
    domain.set_store(False)
    domain.set_quantity('elevation', lambda x, y: -SLOPE*x)
    domain.set_quantity('friction', friction)
    domain.set_quantity('stage', lambda x, y: -SLOPE*x + DEPTH)
    domain.set_quantity('xmomentum', uniform_discharge())

    Bi = anuga.Dirichlet_boundary([DEPTH, uniform_discharge(), 0.0])
    Br = anuga.Reflective_boundary(domain)
    Bn = anuga.Normal_depth_boundary(domain, slope=boundary_slope,
                                     **boundary_kwargs)
    domain.set_boundary({'left': Bi, 'right': Bn, 'top': Br, 'bottom': Br})
    return domain


def depth_between(domain, x0, x1):
    """Mean depth of the cells with centroid between x0 and x1."""
    x = domain.centroid_coordinates[:, 0]
    depth = (domain.quantities['stage'].centroid_values
             - domain.quantities['elevation'].centroid_values)
    return depth[(x > x0) & (x < x1)].mean()


def run(domain, finaltime, yieldstep=None):
    yieldstep = finaltime if yieldstep is None else yieldstep
    for _ in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
        pass


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_uniform_flow_leaves_at_normal_depth(algorithm):
    """Uniform flow down a channel stays uniform up to the outlet."""

    domain = channel(algorithm)
    run(domain, 1200.0)

    assert depth_between(domain, 150.0, 250.0) == pytest.approx(DEPTH, rel=2e-3)
    assert depth_between(domain, 380.0, 400.0) == pytest.approx(DEPTH, rel=2e-3)


def test_a_transmissive_outlet_does_not_hold_normal_depth():
    """The case the boundary is for: a transmissive outlet drifts deep."""

    domain = channel('DE1')
    Bi = anuga.Dirichlet_boundary([DEPTH, uniform_discharge(), 0.0])
    Br = anuga.Reflective_boundary(domain)
    Bt = anuga.Transmissive_boundary(domain)
    domain.set_boundary({'left': Bi, 'right': Bt, 'top': Br, 'bottom': Br})
    run(domain, 1200.0)

    assert depth_between(domain, 380.0, 400.0) > 1.2*DEPTH


@pytest.mark.parametrize('boundary_slope', [0.0005, 0.004])
def test_outlet_depth_follows_the_boundary_slope(boundary_slope):
    """With another slope the outlet moves to that slope's normal depth."""

    domain = channel('DE1', boundary_slope=boundary_slope)
    run(domain, 3000.0)

    # Depth of uniform flow at the channel's discharge on the boundary slope
    expected = (uniform_discharge()*MANNING/np.sqrt(boundary_slope))**0.6
    outlet = depth_between(domain, 390.0, 400.0)

    assert outlet == pytest.approx(expected, rel=0.06)
    # Backwater for a flatter slope, drawdown for a steeper one
    assert (outlet > DEPTH) == (boundary_slope < SLOPE)


def test_manning_argument_replaces_the_cell_friction():
    """manning= sets the roughness of the uniform flow at the outlet."""

    smooth = 0.02
    domain = channel('DE1', manning=smooth)
    run(domain, 3000.0)

    expected = (uniform_discharge()*smooth/np.sqrt(SLOPE))**0.6
    outlet = depth_between(domain, 390.0, 400.0)
    assert outlet == pytest.approx(expected, rel=0.06)
    assert outlet < 0.9*DEPTH


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_outflow_is_in_the_boundary_flux_integral(algorithm):
    """The water that leaves is what the boundary flux integral reports."""

    domain = channel(algorithm, boundary_slope=0.004)   # draws the channel down
    volume0 = domain.get_water_volume()
    run(domain, 600.0)

    change = domain.get_water_volume() - volume0
    assert change < -1.0
    assert domain.get_boundary_flux_integral() == pytest.approx(change, rel=1e-8)


def pond(inflow, momentum):
    """A level pond with the boundary on the right and walls elsewhere."""

    domain = anuga.rectangular_cross_domain(30, 3, len1=150.0, len2=15.0)
    domain.set_flow_algorithm('DE1')
    domain.set_store(False)
    domain.set_quantity('elevation', 0.0)
    domain.set_quantity('friction', MANNING)
    domain.set_quantity('stage', 0.5)
    domain.set_quantity('xmomentum', momentum)

    Br = anuga.Reflective_boundary(domain)
    Bn = anuga.Normal_depth_boundary(domain, slope=SLOPE, inflow=inflow)
    domain.set_boundary({'left': Br, 'right': Bn, 'top': Br, 'bottom': Br})
    return domain


def volume_entered(domain, finaltime=10.0):
    """Total of the increases of the boundary flux integral."""
    entered = 0.0
    previous = 0.0
    for _ in domain.evolve(yieldstep=0.1, finaltime=finaltime):
        integral = domain.get_boundary_flux_integral()
        entered += max(integral - previous, 0.0)
        previous = integral
    return entered


def test_no_water_enters_when_the_flow_is_into_the_domain():
    """Flow directed away from the boundary must not draw water in."""

    assert volume_entered(pond('wall', momentum=-2.0)) == 0.0

    # What the wall rule prevents
    assert volume_entered(pond('transmissive', momentum=-2.0)) > 0.1


def test_a_still_pond_drains():
    """Still water at the boundary leaves as uniform flow at its depth."""

    domain = pond('wall', momentum=0.0)
    volume0 = domain.get_water_volume()
    run(domain, 60.0)

    assert domain.get_water_volume() < volume0
    assert domain.get_boundary_flux_integral() < 0.0


def test_a_dry_edge_passes_nothing():
    domain = pond('wall', momentum=0.0)
    domain.set_quantity('stage', 0.0)   # dry bed
    run(domain, 10.0)

    assert domain.get_water_volume() == 0.0
    assert domain.get_boundary_flux_integral() == 0.0


def test_zero_friction_warns_once_and_is_transmissive():
    """Uniform flow needs friction; ANUGA's default friction is zero."""

    domain = channel('DE1', friction=0.0)

    with pytest.warns(UserWarning, match='Manning friction is zero') as record:
        run(domain, 60.0)

    ours = [w for w in record if 'Manning friction is zero' in str(w.message)]
    assert len(ours) == 1

    stage = domain.quantities['stage'].centroid_values
    assert np.all(np.isfinite(stage))
    assert np.all(np.isfinite(domain.quantities['xmomentum'].centroid_values))
    # Water left through the boundary
    assert domain.get_boundary_flux_integral() < 0.0


def test_thin_flow_does_not_collapse_the_timestep():
    """Rain on a steep dry slope: the film at the boundary is thinner than
    the fall of the surface across half a cell."""

    domain = anuga.rectangular_cross_domain(10, 10, len1=100.0, len2=100.0)
    domain.set_flow_algorithm('DE0')
    domain.set_store(False)
    domain.set_quantity('elevation', lambda x, y: 0.02*np.abs(x - 50.0) + 0.01*y)
    domain.set_quantity('friction', MANNING)
    domain.set_quantity('stage', expression='elevation')

    rain = anuga.Rate_operator(domain, rate=50.0e-3/3600.0)
    Bn = anuga.Normal_depth_boundary(domain, slope=0.01)
    domain.set_boundary({'left': Bn, 'right': Bn, 'top': Bn, 'bottom': Bn})

    # A dry domain takes its first step to the first yield, so yield often
    run(domain, 600.0, yieldstep=50.0)

    volume = domain.get_water_volume()
    outflow = -domain.get_boundary_flux_integral()
    assert outflow > 0.0
    assert volume + outflow == pytest.approx(rain.cumulative_influx, rel=1e-8)


def test_evaluate_agrees_with_evaluate_segment():
    """The edge-by-edge form and the vectorised form give the same values."""

    domain = channel('DE1', boundary_slope=0.004)
    run(domain, 100.0)

    Bn = domain.boundary_map['right']
    ids = np.array([k for k, (_, B) in enumerate(domain.boundary_objects)
                    if B is Bn])
    assert len(ids) > 0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Bn.evaluate_segment(domain, ids)

    for k in ids:
        vol_id = domain.boundary_cells[k]
        edge_id = domain.boundary_edges[k]
        q = Bn.evaluate(vol_id, edge_id)
        assert q[0] == pytest.approx(domain.quantities['stage'].boundary_values[k])
        assert q[1] == pytest.approx(domain.quantities['xmomentum'].boundary_values[k])
        assert q[2] == pytest.approx(domain.quantities['ymomentum'].boundary_values[k])
        assert q[1] > 0.0   # out of the right-hand boundary


def centroid_values(domain):
    return [domain.quantities[name].centroid_values.copy()
            for name in ('stage', 'xmomentum', 'ymomentum')]


@pytest.mark.parametrize('algorithm', ALGORITHMS + ['DE2'])
def test_unified_mode_matches_legacy(algorithm):
    """The device kernel gives what the host evaluation gives."""

    results = {}
    for mode in ('legacy', 'unified'):
        domain = channel(algorithm, boundary_slope=0.004, mode=mode)
        run(domain, 600.0, yieldstep=300.0)
        results[mode] = (centroid_values(domain),
                         domain.get_boundary_flux_integral())

    for legacy, unified in zip(results['legacy'][0], results['unified'][0]):
        assert np.max(np.abs(legacy - unified)) < 1e-11
    assert results['unified'][1] == pytest.approx(results['legacy'][1], rel=1e-11)
    assert results['legacy'][1] < -1.0


def pond_with_two_outlets(mode):
    """Two boundaries that differ in slope, roughness and inflow rule."""

    domain = anuga.rectangular_cross_domain(12, 12, len1=60.0, len2=60.0)
    domain.set_compute_mode(mode)
    domain.set_flow_algorithm('DE1')
    domain.set_store(False)
    domain.set_quantity('elevation', 0.0)
    domain.set_quantity('friction', MANNING)
    domain.set_quantity('stage', 0.5)

    Br = anuga.Reflective_boundary(domain)
    Bn = anuga.Normal_depth_boundary(domain, slope=0.001)
    Bm = anuga.Normal_depth_boundary(domain, slope=0.004, manning=0.05,
                                     inflow='transmissive')
    domain.set_boundary({'left': Br, 'right': Bn, 'top': Bm, 'bottom': Br})
    return domain


def test_two_boundaries_keep_their_own_settings_on_the_device():
    """Slope, manning= and the inflow rule are held per edge in the kernel."""

    results = {}
    for mode in ('legacy', 'unified'):
        domain = pond_with_two_outlets(mode)
        run(domain, 60.0, yieldstep=20.0)
        results[mode] = (centroid_values(domain),
                         domain.get_boundary_flux_integral())

    for legacy, unified in zip(results['legacy'][0], results['unified'][0]):
        assert np.max(np.abs(legacy - unified)) < 1e-11
    assert results['unified'][1] == pytest.approx(results['legacy'][1], rel=1e-11)

    # The two outlets do differ: the pond is not symmetric about its diagonal
    stage = results['legacy'][0][0]
    x, y = pond_with_two_outlets('legacy').centroid_coordinates.T
    right = stage[(x > 50.0) & (y < 30.0)].mean()
    top = stage[(y > 50.0) & (x < 30.0)].mean()
    assert abs(right - top) > 1e-3


def test_unified_mode_evaluates_it_on_the_device():
    """No host fallback: the class is one the device path evaluates."""

    from anuga.shallow_water.shallow_water_domain import GPU_BOUNDARY_TYPES
    assert 'Normal_depth_boundary' in GPU_BOUNDARY_TYPES

    domain = channel('DE1', mode='unified')
    with warnings.catch_warnings():
        # The host fallback announces itself with a UserWarning
        warnings.simplefilter('error', UserWarning)
        run(domain, 60.0)


def test_arguments_are_checked():
    domain = anuga.rectangular_cross_domain(2, 2)

    with pytest.raises(Exception, match='Domain'):
        anuga.Normal_depth_boundary(slope=SLOPE)
    with pytest.raises(Exception, match='Slope'):
        anuga.Normal_depth_boundary(domain)
    with pytest.raises(ValueError, match='slope must be positive'):
        anuga.Normal_depth_boundary(domain, slope=0.0)
    with pytest.raises(ValueError, match='manning must be positive'):
        anuga.Normal_depth_boundary(domain, slope=SLOPE, manning=-0.03)
    with pytest.raises(ValueError, match='inflow'):
        anuga.Normal_depth_boundary(domain, slope=SLOPE, inflow='open')
