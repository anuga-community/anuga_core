"""Barometric pressure operator — fractional-step operator for pressure-gradient momentum forcing."""

import numpy as np
from anuga.config import rho_w
from anuga.operators.base_operator import Operator
from anuga.utilities.numerical_tools import gradient


class Barometric_pressure_operator(Operator):
    """Apply barometric pressure gradient forcing to water momentum each timestep.

    For each triangle the horizontal pressure gradient ∇p is computed from
    pressure values at the three vertices and the momentum is updated by::

        d(xmom)/dt += h * dp/dx / rho_w
        d(ymom)/dt += h * dp/dy / rho_w

    where *h* is the local water depth.

    Parameters
    ----------
    domain : anuga.Domain
    pressure : float or callable
        Barometric pressure in Pa.  If callable, signature must be
        ``pressure(t, x, y)`` where *x*, *y* are the global node (vertex)
        coordinate arrays, and the return value must be broadcastable to
        shape ``(number_of_nodes,)``.
    use_coordinates : bool, optional
        Must be ``True`` (default).  ``False`` is not yet supported; use
        :class:`~anuga.shallow_water.forcing.Barometric_pressure` with a
        ``file_function`` for that case.
    description, label, logging, verbose : passed to :class:`Operator`.

    Examples
    --------
    Uniform pressure::

        P = Barometric_pressure_operator(domain, pressure=101325.0)

    Spatially and temporally varying pressure function::

        def storm_pressure(t, x, y):
            r2 = (x - x0)**2 + (y - y0)**2
            return (p_max - (p_max - p_min) * np.exp(-r2 / R**2)).reshape(1, -1)

        P = Barometric_pressure_operator(domain, storm_pressure, use_coordinates=True)
    """

    def __init__(self, domain, pressure=101325.0, use_coordinates=True,
                 description=None, label=None, logging=False, verbose=False):

        Operator.__init__(self, domain,
                          description=description, label=label,
                          logging=logging, verbose=verbose)

        if not use_coordinates:
            raise NotImplementedError(
                'use_coordinates=False is not supported in Barometric_pressure_operator. '
                'Use anuga.shallow_water.forcing.Barometric_pressure with a '
                'file_function for node-indexed input.')

        self.pressure = pressure

    def __call__(self):
        domain = self.domain
        t = domain.get_time()
        dt = domain.get_timestep()
        N = domain.number_of_elements

        # Evaluate pressure at each mesh node (unique vertices)
        if callable(self.pressure):
            node_coords = domain.get_nodes()
            p_nodes = np.asarray(
                self.pressure(t, node_coords[:, 0], node_coords[:, 1]),
                dtype=float).ravel()
        else:
            p_nodes = np.full(domain.get_number_of_nodes(), float(self.pressure))

        # Per-triangle vertex coordinates (3N × 2) and node indices (N × 3)
        xv = domain.get_vertex_coordinates()
        triangles = domain.triangles
        height = self.stage_c - self.elev_c

        for k in range(N):
            p0 = p_nodes[triangles[k, 0]]
            p1 = p_nodes[triangles[k, 1]]
            p2 = p_nodes[triangles[k, 2]]

            k3 = 3 * k
            x0, y0 = xv[k3]
            x1, y1 = xv[k3 + 1]
            x2, y2 = xv[k3 + 2]

            px, py = gradient(x0, y0, x1, y1, x2, y2, p0, p1, p2)

            self.xmom_c[k] += dt * height[k] * px / rho_w
            self.ymom_c[k] += dt * height[k] * py / rho_w

    def parallel_safe(self):
        return True
