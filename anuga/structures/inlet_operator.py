
import anuga
import numpy
from . import inlet

import warnings

class Inlet_operator(anuga.Operator):
    """Inlet Operator - add water to an inlet.
    Sets up the geometry of problem

    Inherit from this class (and overwrite
    discharge_routine method for specific subclasses)

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
                 verbose = False):
        """Inlet Operator - add water to a domain via an inlet.

        :param domain: Specify domain
        :param region: Apply Inlet flow over a region (which can be a Region, Polygon or line)
        :param Q: function(t) or scalar discharge (m^3/s)
        :param velocity: Optional [u,v] to set velocity of applied discharge
        :param zero_velocity: If set to True, velocity of inlet region set to 0
        :param default: If outside time domain of the Q function, use this default discharge
        :param description: Describe the Inlet_operator
        :param label: Give Inlet_operator a label (name)
        :param verbose: Provide verbose output



        Example:

        >>> inflow_region  = anuga.Region(domain, center=[0.0,0.0], radius=1.0)
        >>> inflow = anuga.Inlet_operator(domain, inflow_region, Q = lambda t : 1 + 0.5*math.sin(t/60))

        """

        anuga.Operator.__init__(self, domain, description, label, logging, verbose)

        self.inlet = inlet.Inlet(self.domain, region, verbose= verbose)

        # constant or function of time, m^3/s
        self.Q = Q

        if velocity is not None:
            assert len(velocity)==2

        self.velocity = velocity

        self.zero_velocity = zero_velocity

        self.applied_Q = 0.0

        self.total_applied_volume = 0.0
        self.total_requested_volume = 0.0

        self.set_default(default)

        self.activate_logging()

        # GPU state (lazy init on first __call__)
        self._gpu_op_id = None
        self._gpu_initialized = False

    def _init_gpu(self):
        """Initialize GPU inlet operator (lazy, called on first __call__ in GPU mode)."""
        try:
            from anuga.shallow_water import sw_domain_gpu_ext as gpu_ext
            gpu_dom = self.domain.gpu_interface.gpu_dom

            tri_indices = numpy.ascontiguousarray(
                self.inlet.triangle_indices, dtype=numpy.intc)
            areas = numpy.ascontiguousarray(
                self.inlet.get_areas(), dtype=numpy.float64)

            op_id = gpu_ext.init_inlet_operator(gpu_dom, tri_indices, areas)
            if op_id >= 0:
                self._gpu_op_id = op_id
                self._gpu_initialized = True
        except Exception as e:
            import sys
            print(f"WARNING: GPU inlet operator init failed: {e}", file=sys.stderr)
            self._gpu_initialized = False

    def _call_gpu(self):
        """GPU path for __call__ - transfers only inlet data (~6KB)."""
        from anuga.shallow_water import sw_domain_gpu_ext as gpu_ext

        gpu_dom = self.domain.gpu_interface.gpu_dom
        op_id = self._gpu_op_id
        timestep = self.domain.get_timestep()
        t = self.domain.get_time()

        # Get current volume from GPU (small reduction, returns scalar)
        current_volume = gpu_ext.inlet_get_volume_gpu(gpu_dom, op_id)
        total_area = self.inlet.area

        assert current_volume >= 0.0

        # Compute Q on CPU (scalar, cheap)
        Q1 = self.update_Q(t)
        Q2 = self.update_Q(t + timestep)
        Q = 0.5 * (Q1 + Q2)
        volume = Q * timestep

        self.applied_Q = Q

        # Get velocities from GPU (small D2H)
        vel_u, vel_v = gpu_ext.inlet_get_velocities_gpu(gpu_dom, op_id)

        has_velocity = 1 if self.velocity is not None else 0
        ext_vel_u = self.velocity[0] if has_velocity else 0.0
        ext_vel_v = self.velocity[1] if has_velocity else 0.0
        zero_vel = 1 if self.zero_velocity else 0

        # Apply on GPU (handles all 3 cases)
        actual_volume = gpu_ext.inlet_apply_gpu(
            gpu_dom, op_id, volume, current_volume, total_area,
            vel_u, vel_v, has_velocity, ext_vel_u, ext_vel_v, zero_vel)

        # Update tracking variables
        self.total_requested_volume += volume
        if volume >= 0.0:
            self.domain.fractional_step_volume_integral += volume
        elif current_volume + volume >= 0.0:
            self.domain.fractional_step_volume_integral += volume
        else:
            self.applied_Q = -current_volume / timestep
            self.domain.fractional_step_volume_integral -= current_volume

        self.total_applied_volume += actual_volume

    def __call__(self):

        # GPU path: skip full domain sync, transfer only inlet data
        if getattr(self.domain, 'multiprocessor_mode', 0) == 2:
            if not self._gpu_initialized:
                self._init_gpu()
            if self._gpu_initialized:
                return self._call_gpu()

        timestep = self.domain.get_timestep()

        #print('Timestep', timestep)

        t = self.domain.get_time()


        # Need to run global command on all processors
        current_volume = self.inlet.get_total_water_volume()
        total_area = self.inlet.get_area()


        #print(current_volume)
        #print(total_area)

        assert current_volume >= 0.0

        Q1 = self.update_Q(t)
        Q2 = self.update_Q(t + timestep)


        #print Q1,Q2
        Q = 0.5*(Q1+Q2)
        volume = Q*timestep

        #print(volume)
        #print volume

        #print Q, volume

        # store last discharge
        self.applied_Q = Q

        #print(self.domain.fractional_step_volume_integral)

        u,v = self.inlet.get_velocities()

        # Distribute positive volume so as to obtain flat surface otherwise
        # just pull water off to have a uniform depth.
        if volume >= 0.0 :
            #print('volume>=0.0')
            self.inlet.set_stages_evenly(volume)
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

        elif current_volume + volume >= 0.0 :
            depth = (current_volume + volume)/total_area
            self.inlet.set_depths(depth)
            self.total_requested_volume += volume
            self.domain.fractional_step_volume_integral+=volume

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
            self.applied_Q = - current_volume/timestep
            self.inlet.set_xmoms(0.0)
            self.inlet.set_ymoms(0.0)


        self.total_applied_volume += volume


    def update_Q(self, t):
        """Allowing local modifications of Q
        """
        from anuga.fit_interpolate.interpolate import Modeltime_too_early, Modeltime_too_late

        if callable(self.Q):
            try:
                Q = self.Q(t)
            except Modeltime_too_early as e:
                Q = self.get_default(t,err_msg=e)
            except Modeltime_too_late as e:
                Q = self.get_default(t,err_msg=e)
        else:
            Q = self.Q

        return Q

    def statistics(self):


        message  = '=====================================\n'
        message += 'Inlet Operator: %s\n' % self.label
        message += '=====================================\n'

        message += 'Description\n'
        message += '%s' % self.description
        message += '\n'

        inlet = self.inlet

        message += '-------------------------------------\n'
        message +=  'Inlet\n'
        message += '-------------------------------------\n'

        message += 'inlet triangle indices and centres\n'
        message += '%s' % inlet.triangle_indices
        message += '\n'

        message += '%s' % self.domain.get_centroid_coordinates()[inlet.triangle_indices]
        message += '\n'

        message += 'region\n'
        message += '%s' % inlet
        message += '\n'

        message += '=====================================\n'

        return message


    def timestepping_statistics(self):

        message = '---------------------------\n'
        message += 'Inlet report for %s:\n' % self.label
        message += '--------------------------\n'
        message += 'Q [m^3/s]: %.2f\n' % self.applied_Q
        message += 'Total volume [m^3]: %.2f\n' % self.total_applied_volume

        return message

    def print_timestepping_statisitics(self):

        message = self.timestepping_statistics()
        print(message)


    def set_default(self, default=None):

        """ Either leave default as None or change it into a function"""

        if default is not None:
            # If it is a constant, make it a function
            if not callable(default):
                tmp = default
                default = lambda t: tmp

            # Check that default_rate is a function of one argument
            try:
                default(0.0)
            except TypeError:
                msg = "could not call default"
                raise Exception(msg)

        self.default = default
        self.default_invoked = False



    def get_default(self,t, err_msg=' '):
        """ Call get_default only if exception
        Modeltime_too_late(msg) has been raised
        """


        # Pass control to default rate function
        value = self.default(t)

        if self.default_invoked is False:
            # Issue warning the first time
            msg = ('%s\n'
                   'Instead I will use the default rate: %s\n'
                   'Note: Further warnings will be suppressed'
                   % (str(err_msg), str(self.default(t))))

            warnings.warn(msg)

            # FIXME (Ole): Replace this crude flag with
            # Python's ability to print warnings only once.
            # See http://docs.python.org/lib/warning-filter.html
            self.default_invoked = True

        return value

    def set_Q(self, Q):

        self.Q = Q

    def get_Q(self):

        return self.Q

    def get_inlet(self):

        return self.inlet

    def get_applied_Q(self):

        return self.applied_Q

    def get_total_applied_volume(self):

        return self.total_applied_volume
