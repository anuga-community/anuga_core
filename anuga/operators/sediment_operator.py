"""Fractional-step operator for suspended sediment transport and bed evolution.

Applies, once per timestep and after the hydrodynamic step:

    m_s  <-  m_s + dt (E_s - D_s)          [G-3] source term
    z    <-  z   + dt (D - E)/(1 - lambda) [G-4] Exner, suspended contribution

Both use the same limited source, so the sediment volume leaving suspension is
exactly `(1 - lambda) dz` and the mass budget closes by construction, whatever
timestepping method the flow algorithm uses.

The advective transport of `m` is NOT here: that lives in the flux kernel and
runs every substep with the rest of the shallow-water system. This operator
carries only the bed exchange.

GPU safety
----------
The work is done by `core_apply_sediment_source`, the same kernel in both
compute modes. In mode 2 it is called through the device path, so the tracer and
bed arrays are updated in place on the GPU with no host round trip. That matters
more than it looks: a fractional-step operator that writes on the host forces
`_gpu_host_writes_suppressed` on, which reactivates the sync bracket and drops
the whole run onto the host path -- a large, silent performance loss. This
operator must therefore stay on the GPU-safe list in
`Domain._has_cpu_only_fractional_operators`.
"""

import math

import anuga.utilities.log as log
from anuga.operators.base_operator import Operator


class Sediment_operator(Operator):
    """Apply sediment bed exchange and Exner bed evolution as a fractional step.

    Parameters
    ----------
    domain : Domain
        Must already have at least one class registered via
        `domain.add_sediment_class()`.
    """

    def __init__(self, domain, description=None, label=None, logging=False,
                 verbose=False):
        Operator.__init__(self, domain, description, label, logging, verbose)

        if getattr(domain, 'n_sediment_classes', 0) == 0:
            raise ValueError(
                'Sediment_operator requires at least one sediment class; '
                'call domain.add_sediment_class(...) first')

        # Reporting for the repose relaxation (spec 7).
        self.repose_sweeps = 0
        self.repose_sweeps_total = 0
        self.repose_cap_hits = 0

    def __call__(self):
        timestep = self.domain.get_timestep()
        if timestep <= 0.0:
            return

        domain = self.domain
        suspended = getattr(domain, '_sediment_suspended_enabled', True)
        # _gpu_host_writes_suppressed marks a HOST-COHERENT window: because a
        # CPU-only operator is registered, apply_fractional_steps has already
        # pulled the device state to the host and will push the host state back
        # when the loop finishes. Running on the device inside that window is
        # not merely redundant -- the trailing push overwrites whatever the
        # device computed with the host copy, so the work is silently lost.
        # That is what made an external source contribute nothing in mode 2
        # whenever any CPU-only operator was present (#288).
        on_gpu = (domain.multiprocessor_mode == 2
                  and domain.gpu_interface is not None
                  and not getattr(domain, '_gpu_host_writes_suppressed', False))

        if on_gpu:
            from anuga.shallow_water.sw_domain_gpu_ext import (
                apply_sediment_source_gpu, apply_bedload_gpu)
            if suspended:
                apply_sediment_source_gpu(domain.gpu_interface.gpu_dom, timestep)
            if domain.sediment_bedload_mode:
                apply_bedload_gpu(domain.gpu_interface.gpu_dom, timestep)
        else:
            from anuga.shallow_water.sw_domain_openmp_ext import (
                apply_sediment_source, apply_bedload)
            if suspended:
                apply_sediment_source(domain, timestep)
            if domain.sediment_bedload_mode:
                apply_bedload(domain, timestep)

        # Angle-of-repose relaxation LAST, so it relaxes the bed this step
        # actually produced rather than the one it started from. It is the only
        # non-cell-local sediment kernel, and the only one that can iterate.
        if domain.sediment_repose_tan > 0.0:
            if on_gpu:
                from anuga.shallow_water.sw_domain_gpu_ext import (
                    apply_repose_gpu)
                sweeps = apply_repose_gpu(domain.gpu_interface.gpu_dom)
            else:
                from anuga.shallow_water.sw_domain_openmp_ext import (
                    apply_repose)
                sweeps = apply_repose(domain)

            self.repose_sweeps = sweeps
            self.repose_sweeps_total += sweeps
            if sweeps >= domain.sediment_repose_max_sweeps:
                # Spec 7 requires this be reported, not swallowed: hitting the
                # cap means the bed may still be over-steep where the whole
                # point of the kernel was that it should not be.
                self.repose_cap_hits += 1
                if self.repose_cap_hits == 1 or self.verbose:
                    log.critical(
                        '%s: angle-of-repose relaxation hit its %d-sweep cap '
                        'at t = %g s; the bed may still exceed %.1f degrees. '
                        'Raise max_sweeps, or relax less aggressively.'
                        % (self.label, domain.sediment_repose_max_sweeps,
                           domain.get_time(),
                           math.degrees(math.atan(domain.sediment_repose_tan))))

            # Spec 7 requires a halo exchange per sweep. This exchanges once
            # per timestep instead: the sweep loop lives inside the kernel so
            # that it stays on the device, and pulling it into Python to
            # exchange between sweeps would put a host round trip in the middle
            # of every sweep and drop the run off the GPU path. In parallel the
            # consequence is that relaxation propagates across a subdomain
            # boundary one sweep per TIMESTEP rather than one per sweep, so a
            # slump spanning a boundary relaxes more slowly there. Recorded in
            # PHYSICS_SPEC 7.1; serial results are unaffected.
            if domain.parallel:
                domain.update_ghosts(['elevation'])

    def parallel_safe(self):
        """Safe in parallel.

        The suspended exchange is cell-local. The bedload divergence DOES read
        neighbours, but only their transport vector, which is computed from
        stage, momentum and friction -- all quantities the halo exchange already
        keeps current -- so ghost cells carry a valid q_b and the divergence is
        correct on owned cells.

        Angle-of-repose relaxation also reads neighbours, and exchanges
        elevation after its sweeps. See the note at the call site for what it
        does NOT do, which is exchange between them.
        """
        return True

    def statistics(self):
        return '%s: %d sediment class(es)' % (self.label,
                                              self.domain.n_sediment_classes)

    def timestepping_statistics(self):
        if self.domain.sediment_repose_tan > 0.0:
            return ', repose sweeps %d' % self.repose_sweeps
        return ''
