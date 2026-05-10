"""

Integrate the fluxes through the boundaries

Relies on boundary_flux_sum being computed in
compute_fluxes.



"""

__author__="gareth"
__date__ ="$10/07/2014 $"



import numpy as num
from anuga.operators.base_operator import Operator

class boundary_flux_integral_operator(Operator):
    """
    Simple operator to collect the integral of the boundary fluxes during a run
    """

    def __init__(self,
                 domain,
                 description = None,
                 label = None,
                 logging = False,
                 verbose = False):


        Operator.__init__(self, domain, description, label, logging, verbose)

        self.boundary_flux_integral = num.array([0.])
        self.domain = domain

        # Cache the RK coefficient once — timestepping_method never changes during a run.
        # Eliminates a per-step attribute lookup + string comparison (called 36002×).
        _ts = domain.timestepping_method
        if _ts == 'euler':
            self._bfs_coeff = (1, False)   # (substeps to use, rk3_special)
        elif _ts == 'rk2':
            self._bfs_coeff = (2, False)
        elif _ts == 'rk3':
            self._bfs_coeff = (3, True)
        else:
            self._bfs_coeff = None  # fallback to original code path


    def __call__(self):
        """Accumulate boundary flux for each timestep."""
        dt = self.domain.timestep
        raw = self.domain.boundary_flux_sum
        # Strip masked-array wrapper without touching .data (which returns a
        # raw memoryview/bytes buffer in some numpy versions, not an ndarray).
        # numpy.asarray() always returns a plain writable ndarray view.
        bfs = num.asarray(raw)

        if self._bfs_coeff is not None:
            n, rk3 = self._bfs_coeff
            if n == 1:
                self.boundary_flux_integral += dt * bfs[0]
            elif n == 2:
                self.boundary_flux_integral += 0.5 * dt * (bfs[0] + bfs[1])
            else:  # rk3
                self.boundary_flux_integral += (dt / 6.0) * (bfs[0] + bfs[1] + 4.0 * bfs[2])
        else:
            ts_method = self.domain.timestepping_method
            if ts_method == 'euler':
                self.boundary_flux_integral += dt * bfs[0]
            elif ts_method == 'rk2':
                self.boundary_flux_integral += 0.5 * dt * (bfs[0] + bfs[1])
            elif ts_method == 'rk3':
                self.boundary_flux_integral += (dt / 6.0) * (bfs[0] + bfs[1] + 4.0 * bfs[2])
            else:
                raise Exception('Cannot compute boundary flux integral with this timestepping method')

        bfs[:] = 0.0

    def parallel_safe(self):
        """Operator is applied independently on each parallel domain

        """
        return True

    def statistics(self):

        message = self.label + ': Boundary_flux_integral operator'
        return message


    def timestepping_statistics(self):
        from anuga import indent

        message  = indent + self.label + ': Integrating the boundary flux'
        return message

