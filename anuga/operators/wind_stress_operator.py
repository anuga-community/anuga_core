"""Wind stress operator — fractional-step operator for wind-driven momentum forcing."""

import numpy as np
from math import pi
from anuga.config import rho_a, rho_w, eta_w
from anuga.operators.base_operator import Operator


class Wind_stress_operator(Operator):
    """Apply wind stress to water momentum each timestep.

    Wind speed *s* [m/s] and direction *phi* [degrees] drive a surface stress
    that is added to xmomentum and ymomentum.  Both can be scalars or
    functions ``f(t, x, y)`` returning arrays aligned with the centroid
    coordinates.

    The stress formula follows Large & Pond (1981) via::

        S = (eta_w * rho_a / rho_w) * |wind|
        d(xmom)/dt += S * u_wind
        d(ymom)/dt += S * v_wind

    Parameters
    ----------
    domain : anuga.Domain
    speed : float or callable
        Wind speed in m/s.  If callable, signature must be
        ``speed(t, x, y)`` where *x*, *y* are centroid coordinate arrays.
    phi : float or callable
        Wind direction in degrees (standard mathematical convention:
        0° = east, 90° = north).  If callable, same signature as *speed*.
    description, label, logging, verbose : passed to :class:`Operator`.

    Examples
    --------
    Uniform wind::

        W = Wind_stress_operator(domain, speed=10.0, phi=180.0)

    Spatially varying wind functions::

        def my_speed(t, x, y):
            return 20.0 * np.ones_like(x)

        def my_phi(t, x, y):
            return 270.0 * np.ones_like(x)

        W = Wind_stress_operator(domain, my_speed, my_phi)

    Two positional arguments are accepted for compatibility with the
    legacy ``Wind_stress`` calling convention::

        W = Wind_stress_operator(domain, my_speed, my_phi)
    """

    def __init__(self, domain, speed=0.0, phi=0.0,
                 description=None, label=None, logging=False, verbose=False):

        Operator.__init__(self, domain,
                          description=description, label=label,
                          logging=logging, verbose=verbose)

        self.speed = speed
        self.phi = phi
        self.const = eta_w * rho_a / rho_w

    def __call__(self):
        domain = self.domain
        t = domain.get_time()
        dt = domain.get_timestep()
        xc = domain.centroid_coordinates
        N = domain.number_of_elements

        if callable(self.speed):
            s_vec = np.asarray(
                self.speed(t, xc[:, 0], xc[:, 1]), dtype=float).ravel()
        else:
            s_vec = np.full(N, float(self.speed))

        if callable(self.phi):
            phi_vec = np.asarray(
                self.phi(t, xc[:, 0], xc[:, 1]), dtype=float).ravel()
        else:
            phi_vec = np.full(N, float(self.phi))

        phi_rad = phi_vec * (pi / 180.0)
        u = s_vec * np.cos(phi_rad)
        v = s_vec * np.sin(phi_rad)
        S = self.const * np.sqrt(u**2 + v**2)

        self.xmom_c += dt * S * u
        self.ymom_c += dt * S * v

    def parallel_safe(self):
        return True
