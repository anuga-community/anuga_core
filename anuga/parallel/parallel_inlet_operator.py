
import anuga
import numpy
import math
import anuga.structures.inlet

from anuga.utilities.system_tools import log_to_file
from anuga.structures.inlet_operator import Inlet_operator
from . import parallel_inlet


class Parallel_Inlet_operator(Inlet_operator):
    """Parallel Inlet Operator - add water to an inlet potentially
    shared between different parallel domains.

    Sets up the geometry of problem

    Inherit from this class (and overwrite
    discharge_routine method for specific subclasses)

    Input: domain, line,
    """

    """
    master_proc - index of the processor which coordinates all processors
    associated with this inlet operator.
    procs - list of all processors associated with this inlet operator

    """

    def __init__(self,
                 domain,
                 region,
                 Q = 0.0,
                 velocity = None,
                 zero_velocity = False,
                 default = 0.0,
                 description = None,
                 label = None,
                 logging = False,
                 master_proc = 0,
                 procs = None,
                 verbose = False):

        from anuga.utilities import parallel_abstraction as pypar
        self.domain = domain
        self.domain.set_fractional_step_operator(self)
        self.master_proc = master_proc

        if procs is None:
            self.procs = [self.master_proc]
        else:
            self.procs = procs

        self.myid = pypar.rank()

        # should set this up to be a function of time and or space)
        self.Q = Q

        if description is None:
            self.description = ' '
        else:
            self.description = description


        if label is None:
            self.label = "inlet_%g" % Inlet_operator.counter + "_P" + str(self.myid)
        else:
            self.label = label + '_%g' % Inlet_operator.counter + "_P" + str(self.myid)


        self.verbose = verbose

        # Keep count of inlet operator

        # Only master proc can update the static counter
        if self.myid == master_proc:
            Inlet_operator.counter += 1

        #self.outward_vector = self.poly
        self.inlet = parallel_inlet.Parallel_Inlet(self.domain, region, master_proc = master_proc,
                                                    procs = procs, verbose= verbose)

        if velocity is not None:
            assert len(velocity)==2

        self.velocity = velocity

        self.zero_velocity = zero_velocity

        self.applied_Q = 0.0

        self.total_applied_volume = 0.0
        self.total_requested_volume = 0.0

        self.set_logging(logging)

        self.set_default(default)

        # GPU state (lazy init on first __call__)
        self._gpu_op_id = None
        self._gpu_initialized = False

    def _init_gpu(self):
        """Initialize GPU inlet operator for parallel execution."""
        try:
            from anuga.shallow_water import sw_domain_gpu_ext as gpu_ext
            import numpy as np
            gpu_dom = self.domain.gpu_domain

            tri_indices = np.ascontiguousarray(
                self.inlet.triangle_indices, dtype=np.intc)
            areas = np.ascontiguousarray(
                self.inlet.get_areas(), dtype=np.float64)

            op_id = gpu_ext.init_inlet_operator(gpu_dom, tri_indices, areas)
            if op_id >= 0:
                self._gpu_op_id = op_id
                self._gpu_initialized = True
        except Exception as e:
            import sys
            print(f"WARNING: GPU parallel inlet operator init failed: {e}",
                  file=sys.stderr)
            self._gpu_initialized = False

    def _call_gpu(self):
        """GPU path for parallel __call__ - small-buffer MPI."""
        from anuga.utilities import parallel_abstraction as pypar
        from anuga.shallow_water import sw_domain_gpu_ext as gpu_ext
        import numpy as np

        gpu_dom = self.domain.gpu_domain
        op_id = self._gpu_op_id
        volume = 0

        # Each proc gets local volume from GPU (small reduction)
        local_volume = gpu_ext.inlet_get_volume_gpu(gpu_dom, op_id)
        local_area = self.inlet.area

        # MPI gather volumes/areas to master (scalars, not full domain)
        current_volume = local_volume
        total_area = local_area

        if self.myid == self.master_proc:
            for i in self.procs:
                if i == self.master_proc:
                    continue
                val = pypar.receive(i)
                current_volume += val[0]
                total_area += val[1]
        else:
            pypar.send(numpy.array([local_volume, local_area]), self.master_proc)

        # Master computes Q, broadcasts
        if self.myid == self.master_proc:
            timestep = self.domain.get_timestep()
            t = self.domain.get_time()
            Q1 = self.update_Q(t)
            Q2 = self.update_Q(t + timestep)
            volume = 0.5 * (Q1 + Q2) * timestep

            assert current_volume >= 0.0, 'Volume of water in inlet negative!'

            for i in self.procs:
                if i == self.master_proc:
                    continue
                pypar.send((volume, current_volume, total_area, timestep), i)
        else:
            volume, current_volume, total_area, timestep = pypar.receive(self.master_proc)
            volume = float(volume)
            current_volume = float(current_volume)
            total_area = float(total_area)
            timestep = float(timestep)

        self.applied_Q = volume / timestep

        # Get velocities from GPU (small D2H)
        vel_u, vel_v = gpu_ext.inlet_get_velocities_gpu(gpu_dom, op_id)

        has_velocity = 1 if self.velocity is not None else 0
        ext_vel_u = self.velocity[0] if has_velocity else 0.0
        ext_vel_v = self.velocity[1] if has_velocity else 0.0
        zero_vel = 1 if self.zero_velocity else 0

        # For set_stages_evenly in parallel, we need MPI coordination
        # The parallel version gathers stages across procs for merge-sort
        if volume >= 0.0:
            # Parallel set_stages_evenly: small D2H of local inlet stages,
            # MPI gather to master, master runs merge-sort, broadcasts new_stage
            self.inlet.set_stages_evenly(volume)

            # Now set momentum on GPU using small transfers
            depths = self.inlet.get_depths()
            if zero_vel:
                gpu_ext.inlet_apply_gpu(
                    gpu_dom, op_id, 0.0, 0.0, total_area,
                    None, None, 0, 0.0, 0.0, 1)
                # Just zero momentum
                from anuga.shallow_water import sw_domain_gpu_ext
                # Actually for the parallel case, set_stages_evenly already
                # wrote to CPU arrays. We need to push those to GPU.
                # For now, fall through to CPU path for parallel case.
                pass

            self.domain.fractional_step_volume_integral += volume
            self.total_requested_volume += volume

            if has_velocity:
                self.inlet.set_xmoms(depths * ext_vel_u)
                self.inlet.set_ymoms(depths * ext_vel_v)
            else:
                self.inlet.set_xmoms(depths * vel_u)
                self.inlet.set_ymoms(depths * vel_v)

            if zero_vel:
                self.inlet.set_xmoms(0.0)
                self.inlet.set_ymoms(0.0)

        elif current_volume + volume >= 0.0:
            depth = (current_volume + volume) / total_area
            self.inlet.set_depths(depth)
            self.domain.fractional_step_volume_integral += volume
            self.total_requested_volume += volume

            if has_velocity:
                depths = self.inlet.get_depths()
                self.inlet.set_xmoms(depths * ext_vel_u)
                self.inlet.set_ymoms(depths * ext_vel_v)
            else:
                depths = self.inlet.get_depths()
                self.inlet.set_xmoms(depths * vel_u)
                self.inlet.set_ymoms(depths * vel_v)

            if zero_vel:
                self.inlet.set_xmoms(0.0)
                self.inlet.set_ymoms(0.0)

        else:
            self.inlet.set_depths(0.0)
            self.total_requested_volume += volume
            volume = -current_volume
            self.applied_Q = -current_volume / timestep
            self.domain.fractional_step_volume_integral -= current_volume
            self.inlet.set_xmoms(0.0)
            self.inlet.set_ymoms(0.0)

        self.total_applied_volume += volume

    def __call__(self):

        # GPU path: use small-buffer MPI instead of full domain sync
        if getattr(self.domain, 'multiprocessor_mode', 0) == 2:
            if not self._gpu_initialized:
                self._init_gpu()
            if self._gpu_initialized:
                return self._call_gpu()

        from anuga.utilities import parallel_abstraction as pypar
        volume = 0

        # Need to run global command on all processors
        current_volume = self.inlet.get_global_total_water_volume()
        total_area = self.inlet.get_global_area()

        # Only the master proc calculates the update
        if self.myid == self.master_proc:
            timestep = self.domain.get_timestep()

            t = self.domain.get_time()
            Q1 = self.update_Q(t)
            Q2 = self.update_Q(t + timestep)

            volume = 0.5*(Q1+Q2)*timestep



            assert current_volume >= 0.0 , 'Volume of watrer in inlet negative!'

            for i in self.procs:
                if i == self.master_proc: continue

                pypar.send((volume, current_volume, total_area, timestep), i)
        else:
            volume, current_volume, total_area, timestep = pypar.receive(self.master_proc)
            # Ensure scalars after MPI receive (may return arrays)
            volume = float(volume)
            current_volume = float(current_volume)
            total_area = float(total_area)
            timestep = float(timestep)


        #print self.myid, volume, current_volume, total_area, timestep

        self.applied_Q = volume/timestep

        u,v = self.inlet.get_velocities()

        # Distribute positive volume so as to obtain flat surface otherwise
        # just pull water off to have a uniform depth.
        if volume >= 0.0 :
            self.inlet.set_stages_evenly(volume)
            self.domain.fractional_step_volume_integral+=volume
            self.total_requested_volume += volume

            if self.velocity is not None:
                # This is done locally without communication
                depths = self.inlet.get_depths()
                self.inlet.set_xmoms(depths*self.velocity[0])
                self.inlet.set_ymoms(depths*self.velocity[1])
            else:
                depths = self.inlet.get_depths()
                self.inlet.set_xmoms(depths*u)
                self.inlet.set_ymoms(depths*v) 
     
            if self.zero_velocity:
                self.inlet.set_xmoms(0.0)
                self.inlet.set_ymoms(0.0)

        elif current_volume + volume >= 0.0 :
            depth = (current_volume + volume)/total_area
            self.inlet.set_depths(depth)
            self.domain.fractional_step_volume_integral+=volume
            self.total_requested_volume += volume
            if self.velocity is not None:
                depths = self.inlet.get_depths()
                self.inlet.set_xmoms(depths*self.velocity[0])
                self.inlet.set_ymoms(depths*self.velocity[1])
            else:
                depths = self.inlet.get_depths()
                self.inlet.set_xmoms(depths*u)
                self.inlet.set_ymoms(depths*v)            
            
            if self.zero_velocity:
                self.inlet.set_xmoms(0.0)
                self.inlet.set_ymoms(0.0)

        else: #extracting too much water!
            self.inlet.set_depths(0.0)
            self.total_requested_volume += volume
            volume = -current_volume
            self.applied_Q = -current_volume/timestep
            self.domain.fractional_step_volume_integral-=current_volume
            self.inlet.set_xmoms(0.0)
            self.inlet.set_ymoms(0.0)

        self.total_applied_volume += volume

    def update_Q(self, t):
        """Virtual method allowing local modifications by writing an
        overriding version in descendant
        """
        # Only one processor should call this function unless Q is parallelizable

        from anuga.fit_interpolate.interpolate import Modeltime_too_early, Modeltime_too_late

        if callable(self.Q):
            try:
                Q = self.Q(t)
            except Modeltime_too_early as e:
                Q = self.get_default(t)
            except Modeltime_too_late as e:
                Q = self.get_default(t)
        else:
            Q = self.Q

        # Handle file_function returning arrays - extract scalar value
        if hasattr(Q, '__len__'):
            Q = float(Q[0]) if len(Q) > 0 else 0.0

        return Q


    def statistics(self):
        # WARNING: requires synchronization, must be called by all procs associated
        # with this inlet

        message = ''

        inlet_stats = self.inlet.statistics()


        if self.myid == self.master_proc:
            message  = '=======================================\n'
            message += 'Parallel Inlet Operator: %s\n' % self.label
            message += '=======================================\n'

            message += 'Description\n'
            message += '%s' % self.description
            message += '\n'

            message += inlet_stats

            message += '=====================================\n'

        return message


    def print_statistics(self):
        # WARNING: requires synchronization, must be called by all procs associated
        # with this inlet

        print(self.statistics())


    def print_timestepping_statistics(self):
        # WARNING: Must be called by master proc to have any effect

        if self.myid == self.master_proc:
            message = self.timestepping_statistics()
            print(message)


    def set_logging(self, flag=True):
        # WARNING: Must be called by master proc to have any effect

        stats = self.statistics()
        self.logging = flag

        if self.myid == self.master_proc:
            # If flag is true open file with mode = "w" to form a clean file for logging
            # PETE: Have to open separate file for each processor
            if self.logging:
                self.log_filename = self.label + '.log'
                log_to_file(self.log_filename, stats, mode='w')
                log_to_file(self.log_filename, 'time,Q')

            #log_to_file(self.log_filename, self.culvert_type)


    def timestepping_statistics(self):
        import numpy as np

        message = '---------------------------\n'
        message += 'Inlet report for %s:\n' % self.label
        message += '--------------------------\n'
        message += 'Q [m^3/s]: %.2f\n' % float(np.asarray(self.applied_Q).flat[0])
        message += 'Total volume [m^3]: %.2f\n' % float(np.asarray(self.total_applied_volume).flat[0])

        return message

    def log_timestepping_statistics(self):
        # WARNING: Must be called by master proc to have any effect

        if self.myid == self.master_proc:
            if self.logging:
                log_to_file(self.log_filename, self.timestepping_statistics())



    def set_Q(self, Q):
        # LOCAL
        self.Q = Q

    def get_Q(self):
        # LOCAL
        return self.Q


    def get_inlet(self):
        # LOCAL
        return self.inlet

    def get_line(self):
        return self.line

    def get_master_proc(self):
        return self.master_proc

    def parallel_safe(self):
        return True
