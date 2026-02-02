"""
Rate operators (such as rain)

Constraints: See GPL license in the user guide
Version: 1.0 ($Revision: 7731 $)
"""

from builtins import str
__author__="steve"
__date__ ="$09/03/2012 4:46:39 PM$"



from anuga.config import indent
import numpy as num
import anuga.utilities.log as log
from anuga.utilities.function_utils import evaluate_temporal_function


from anuga import Quantity
from anuga.operators.base_operator import Operator
from anuga import Region

class Rate_operator(Operator):


    def __init__(self,
                 domain,
                 rate=0.0,
                 factor=1.0,
                 region=None,
                 indices=None,
                 polygon=None,
                 center=None,
                 radius=None,
                 default_rate=0.0,
                 description = None,
                 label = None,
                 logging = False,
                 verbose = False,
                 monitor = False):
        """
Create a Rate_operator that adds water over a region at a specified
rate (ms^{-1} = vol/Area/sec)

Parameters specifying locaton of operator

:param region: Region object where water applied 
:param indices: List of triangles where water applied
:param polygon: List of [x,y] points specifying a polygon where water applied
:param center: [x.y] point of circle where water applied
:param radius: radius of circle where water applied

Parameters specifying rate

:param rate: scalar, function of (t), (x,y), or (x,y,t), or a Quantity, 
                a numpy array of size (number_of_triangles), 
                or an xarray with rate at points and time
:param factor: scalar, function of t, or 2 by n numpy array time sequence, 
                used to specify conversion from rate argument to m/s
:param default_rate: use this rate if outside time interval of rate function or xarray

Parameters involving communication

:param description:
:param label:
:param logging:
:param verbose:
:param monitor:
    """



        Operator.__init__(self, domain, description, label, logging, verbose)

        #-----------------------------------------------------
        # Make sure region is actually an instance of a region
        # Otherwise create a new region based on the other 
        # input arguments
        #-----------------------------------------------------
        if isinstance(region,Region):
            region.set_verbose(verbose)
            self.region = region

        else:
            self.region = Region(domain,
                        indices=indices,
                        polygon=polygon,
                        center=center,
                        radius=radius,
                        verbose=verbose)


        #------------------------------------------
        # Local variables
        #------------------------------------------
        self.indices = self.region.indices
        self.set_areas()
        self.set_full_indices()        

        #--------------------------------
        # Setting up rate
        #--------------------------------
        self.rate_callable = False
        self.rate_spatial = False
        self.rate_xarray = False


        #-------------------------------
        # Check if rate is actually an xarray. 
        # Need xarray package installed
        #-------------------------------
        try:
            import xarray
        except:
            pass
        else:
            if type(rate) is xarray.core.dataarray.DataArray:
                self.rate_xarray = True
                xa = rate
                rate = 0.0
                self._prepare_xarray_rate(xa)


        self.set_rate(rate)
        self.set_default_rate(default_rate)
        self.default_rate_invoked = False    # Flag

        #------------------------------
        # Setting up factor, can be a scalar
        # or a function of time
        #------------------------------
        # FIXME SR: maybe also allow time, factor file or array

        self.factor_callable = False
        self.set_factor(factor)



        # ----------------
        # Mass tracking
        #-----------------
        self.monitor = monitor
        self.cumulative_influx = 0.0
        self.local_influx=0.0
        self.local_max = 0.0
        self.local_min = 0.0

        # ----------------
        # GPU support
        #-----------------
        self._gpu_op_id = None  # GPU operator ID (set on first GPU call)
        self._gpu_initialized = False
        self._gpu_rate_array_cache = None  # Cached rate array for GPU (avoids recreating every call)
        self._gpu_rate_min_cache = None    # Cached min value for statistics
        self._gpu_rate_max_cache = None    # Cached max value for statistics
        self._gpu_rate_changed = True      # Flag to indicate rate data needs to be transferred to GPU

    def _init_gpu(self):
        """Initialize accelerated operator for this rate operator."""
        if self._gpu_initialized:
            return

        # Check if we have a domain interface
        if not hasattr(self.domain, '_domain_interface') or self.domain._domain_interface is None:
            return

        domain_interface = self.domain._domain_interface
        if not hasattr(domain_interface, 'gpu_dom') or domain_interface.gpu_dom is None:
            return

        # Only support non-spatial, non-xarray rates for GPU
        # Spatial rates (x,y or x,y,t functions) and xarray rates need more complex handling
        if self.rate_spatial or self.rate_xarray:
            return

        # Supported rate types: scalar, t, quantity, centroid_array
        if self.rate_type not in ('scalar', 't', 'quantity', 'centroid_array'):
            return

        # Get indices - if None, apply to all elements
        if self.indices is None:
            indices = num.arange(self.domain.number_of_elements, dtype=num.intc)
            areas = self.domain.areas.copy()
        elif self.indices is []:
            return  # No indices, nothing to do
        else:
            indices = num.asarray(self.indices, dtype=num.intc)
            areas = self.domain.areas[indices].copy()

        # Get full indices for mass tracking
        if self.full_indices is not None and len(self.full_indices) > 0:
            full_indices = num.asarray(self.full_indices, dtype=num.intc)
        else:
            full_indices = None

        try:
            from anuga.shallow_water.sw_domain_ext import init_rate_operator
            self._gpu_op_id = init_rate_operator(
                domain_interface.gpu_dom,
                indices,
                areas.astype(num.float64),
                full_indices
            )
            if self._gpu_op_id >= 0:
                self._gpu_initialized = True
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to initialize GPU rate operator: {e}")
            self._gpu_op_id = None

    def __call__(self):
        """
        Apply rate to those triangles defined in indices

        indices == [], then don't apply anywhere
        indices is None, then apply everywhere
        otherwise apply for the specific indices
        """

        if self.indices is []:
            return

        # Check for accelerated execution path
        if (hasattr(self.domain, '_domain_interface') and
            self.domain._domain_interface is not None and
            not self.rate_spatial and
            not self.rate_xarray):

            # Lazy initialization of GPU operator
            if not self._gpu_initialized:
                self._init_gpu()

            if self._gpu_initialized and self._gpu_op_id is not None and self._gpu_op_id >= 0:
                t = self.domain.get_time()
                timestep = self.domain.get_timestep()
                factor = self.get_factor(t)

                # DEBUG: Confirm GPU path taken
                #print(f"DEBUG Rate_operator: GPU path, op_id={self._gpu_op_id}, rate_type={self.rate_type}, t={t}, timestep={timestep}, factor={factor}")
                # DEBUG: Check rate quantity info
                #if self.rate_type == 'quantity':
                #    print(f"DEBUG Rate_operator: rate quantity name={self.rate.name}, rate object id={id(self.rate)}")

                if self.rate_type == 'quantity':
                    # Quantity type - use array-based GPU kernel
                    from anuga.shallow_water.sw_domain_ext import apply_rate_operator_array_gpu
                    # Use cached rate array if available (avoids expensive array copy every RK2 step)
                    if self._gpu_rate_array_cache is None:
                        self._gpu_rate_array_cache = num.ascontiguousarray(self.rate.centroid_values, dtype=num.float64)
                        self._gpu_rate_changed = True  # New cache, needs transfer
                    rate_array = self._gpu_rate_array_cache

                    # Compute statistics BEFORE kernel modifies stage values
                    # CPU path: local_rates = max(factor*timestep*rate, -heights), then min/max / timestep
                    if self.indices is None:
                        heights = self.stage_c - self.elev_c
                        local_rates = num.maximum(factor * timestep * rate_array, -heights)
                        local_rates_for_stats = local_rates[self.full_indices]
                    else:
                        heights = self.stage_c[self.indices] - self.elev_c[self.indices]
                        local_rates = num.maximum(factor * timestep * rate_array[self.indices], -heights)
                        local_rates_for_stats = local_rates[self.full_indices]
                    self.local_max = local_rates_for_stats.max() / timestep
                    self.local_min = local_rates_for_stats.min() / timestep

                    # rate_array is full domain size, use_indices_into_rate=1
                    # Pass rate_changed flag to avoid unnecessary H2D transfer
                    rate_changed = 1 if self._gpu_rate_changed else 0
                    self.local_influx = apply_rate_operator_array_gpu(
                        self.domain._domain_interface.gpu_dom,
                        self._gpu_op_id,
                        rate_array,
                        1,  # use_indices_into_rate
                        rate_changed,
                        float(factor),
                        float(timestep)
                    )
                    self._gpu_rate_changed = False  # Data transferred, don't repeat

                elif self.rate_type == 'centroid_array':
                    # Centroid array type - use array-based GPU kernel
                    from anuga.shallow_water.sw_domain_ext import apply_rate_operator_array_gpu
                    # Use cached rate array if available
                    if self._gpu_rate_array_cache is None:
                        self._gpu_rate_array_cache = num.ascontiguousarray(self.rate, dtype=num.float64)
                        self._gpu_rate_changed = True  # New cache, needs transfer
                    rate_array = self._gpu_rate_array_cache

                    # Compute statistics BEFORE kernel modifies stage values
                    # CPU path: local_rates = max(factor*timestep*rate, -heights), then min/max / timestep
                    if self.indices is None:
                        heights = self.stage_c - self.elev_c
                        local_rates = num.maximum(factor * timestep * rate_array, -heights)
                        local_rates_for_stats = local_rates[self.full_indices]
                    else:
                        heights = self.stage_c[self.indices] - self.elev_c[self.indices]
                        local_rates = num.maximum(factor * timestep * rate_array[self.indices], -heights)
                        local_rates_for_stats = local_rates[self.full_indices]
                    self.local_max = local_rates_for_stats.max() / timestep
                    self.local_min = local_rates_for_stats.min() / timestep

                    # rate_array is full domain size, use_indices_into_rate=1
                    # Pass rate_changed flag to avoid unnecessary H2D transfer
                    rate_changed = 1 if self._gpu_rate_changed else 0
                    self.local_influx = apply_rate_operator_array_gpu(
                        self.domain._domain_interface.gpu_dom,
                        self._gpu_op_id,
                        rate_array,
                        1,  # use_indices_into_rate
                        rate_changed,
                        float(factor),
                        float(timestep)
                    )
                    self._gpu_rate_changed = False  # Data transferred, don't repeat

                else:
                    # Scalar or time-dependent rate - use scalar GPU kernel
                    from anuga.shallow_water.sw_domain_ext import apply_rate_operator_gpu
                    rate = self.get_non_spatial_rate(t)
                    # Handle case where rate is an array (from file_function)
                    try:
                        rate_scalar = float(rate)
                    except (TypeError, ValueError):
                        rate_scalar = float(rate[0])

                    # Compute statistics BEFORE kernel modifies stage values
                    # CPU path: local_rates = max(factor*timestep*rate, -heights), then min/max / timestep
                    if self.indices is None:
                        heights = self.stage_c - self.elev_c
                        local_rates = num.maximum(factor * timestep * rate_scalar, -heights)
                        local_rates_for_stats = local_rates[self.full_indices]
                    else:
                        heights = self.stage_c[self.indices] - self.elev_c[self.indices]
                        local_rates = num.maximum(factor * timestep * rate_scalar, -heights)
                        local_rates_for_stats = local_rates[self.full_indices]
                    self.local_max = local_rates_for_stats.max() / timestep
                    self.local_min = local_rates_for_stats.min() / timestep

                    self.local_influx = apply_rate_operator_gpu(
                        self.domain._domain_interface.gpu_dom,
                        self._gpu_op_id,
                        rate_scalar,
                        float(factor),
                        float(timestep)
                    )

                # Update tracking
                self.cumulative_influx += self.local_influx
                self.domain.fractional_step_volume_integral += self.local_influx

                return

        # Fall back to CPU path
        if self.rate_xarray:
            # setup centroid_array from xarray corresponding to current time
            self._update_Q_xarray()

        t = self.domain.get_time()
        timestep = self.domain.get_timestep()
        factor = self.get_factor()
        indices = self.indices


        if self.rate_spatial:
            if indices is None:
                x = self.coord_c[:,0]
                y = self.coord_c[:,1]
            else:
                x = self.coord_c[indices,0]
                y = self.coord_c[indices,1]

            rate = self.get_spatial_rate(x,y,t)
        elif self.rate_type == 'quantity':
            if indices is None:
                rate  = self.rate.centroid_values
            else:
                rate = self.rate.centroid_values[indices]
        elif self.rate_type == 'centroid_array':
            if indices is None:
                rate  = self.rate
            else:
                rate = self.rate[indices]
        else:
            rate = self.get_non_spatial_rate(t)


        factor = self.get_factor(t)

        # We need to adjust the momentums if rate < 0 since otherwise 
        # the xmom and ymom stay the same but height -> 0 which leads to xvel, yvel -> infty
        

        fid = self.full_indices
        if num.all(rate >= 0.0):
            # Record the local flux for mass conservation tracking
            if indices is None:
                local_rates = factor*timestep*rate
                self.local_influx = (local_rates*self.areas)[fid].sum()

                self.stage_c[:] = self.stage_c[:] + local_rates
            else:
                local_rates = factor*timestep*rate
                self.local_influx=(local_rates*self.areas)[fid].sum()

                self.stage_c[indices] = self.stage_c[indices] + local_rates
        else: # Be more careful if rate < 0
            if indices is None:
                #self.local_influx=(num.minimum(factor*timestep*rate, self.stage_c[:]-self.elev_c[:])*self.areas)[fid].sum()
                #self.stage_c[:] = num.maximum(self.stage_c  \
                #       + factor*rate*timestep, self.elev_c )
                self.height_c[:] = self.stage_c[:] - self.elev_c[:]
                local_rates = num.maximum(factor*timestep*rate, -self.height_c)
                local_factors = num.where(local_rates < 0.0, (local_rates+self.height_c)/(self.height_c+1.0e-10), 1.0)

                #print(local_factors, local_rates)
                self.local_influx = (local_rates*self.areas)[fid].sum()

                self.stage_c[:] = self.stage_c + local_rates
                self.xmom_c[:] = self.xmom_c[:]*local_factors
                self.ymom_c[:] = self.ymom_c[:]*local_factors
            else:
                #self.local_influx=(num.minimum(factor*timestep*rate, self.stage_c[indices]-self.elev_c[indices])*self.areas)[fid].sum()
                #self.stage_c[indices] = num.maximum(self.stage_c[indices] \
                #       + factor*rate*timestep, self.elev_c[indices])

                #local_rates = num.maximum(factor*timestep*rate, self.elev_c[indices]-self.stage_c[indices])

                heights = self.stage_c[indices] - self.elev_c[indices]
                local_rates = num.maximum(factor*timestep*rate, -heights)
                local_factors = num.where(local_rates < 0.0, (local_rates+heights)/(heights+1.0e-10), 1.0)

                #print(local_factors, local_rates, fid)

                self.local_influx = (local_rates*self.areas)[fid].sum()

                self.stage_c[indices] = self.stage_c[indices] + local_rates
                self.xmom_c[indices] = self.xmom_c[indices]*local_factors
                self.ymom_c[indices] = self.ymom_c[indices]*local_factors


        try:
            self.local_max = (local_rates[fid].max()/timestep)
            self.local_min = (local_rates[fid].min()/timestep)
        except:
            self.local_max = local_rates/timestep
            self.local_min = local_rates/timestep

        if isinstance(self.local_max, num.ndarray) and self.local_max.size == 0:
            self.local_max = 0.0
            self.local_min = 0.0

        # print(self.local_min, self.local_max)
        
        self.cumulative_influx += self.local_influx

        # Update mass inflows from fractional steps
        self.domain.fractional_step_volume_integral+=self.local_influx
        
        if self.monitor:
            log.critical('Local Flux at time %.2f = %f'
                         % (self.domain.get_time(), self.local_influx))

            

        return

    def get_non_spatial_rate(self, t=None):
        """Provide a rate to calculate added volume
        """

        if t is None:
            t = self.get_time()

        assert not self.rate_spatial

        rate = evaluate_temporal_function(self.rate, t,
                                          default_right_value=self.default_rate,
                                          default_left_value=self.default_rate)

        if rate is None:
            msg = ('Attribute rate must be specified in '+self.__name__+
                   ' before attempting to call it')
            raise Exception(msg)

        return rate

    def get_spatial_rate(self, x=None, y=None, t=None):
        """Provide a rate to calculate added volume
        only call if self.rate_spatial = True
        """

        assert self.rate_spatial

        if t is None:
            t = self.get_time()

        if x is None:
            assert y is None
            if self.indices is None:
                x = self.coord_c[:,0]
                y = self.coord_c[:,1]
            else:
                x = self.coord_c[self.indices,0]
                y = self.coord_c[self.indices,1]

        assert x is not None
        assert y is not None

        assert isinstance(t, (int, float))
        assert len(x) == len(y)

        #print xy
        #print t

        #print self.rate_type, self.rate_type == 'x,y,t'
        if self.rate_type == 'x,y,t':
            rate = self.rate(x,y,t)
        else:
            rate = self.rate(x,y)

        return rate


    def set_rate(self, rate):
        """Set rate. Can change rate while running


        Can be a scalar, numpy array, or a function of t or x,y or x,y,t or a quantity
        """

        # Test if rate is a quantity
        if isinstance(rate, Quantity):
            self.rate_type = 'quantity'
        elif isinstance(rate, num.ndarray):
            rate_shape = rate.shape
            msg =  f"The shape {rate_shape} of the input rate "
            msg += f"should match (number of triangles,) i.e. ({self.domain.number_of_triangles},)"
            assert rate_shape == (self.domain.number_of_triangles,) \
                or rate_shape == (self.domain.number_of_triangles, 1), msg
            self.rate_type = 'centroid_array'
            rate = rate.reshape((-1,)) 
        else:
            # Possible types are 'scalar', 't', 'x,y' and 'x,y,t'
            from anuga.utilities.function_utils import determine_function_type
            self.rate_type = determine_function_type(rate)


        self.rate = rate

        # Invalidate GPU caches (will be recreated on next GPU call)
        # Use hasattr since set_rate() can be called from __init__ before cache is initialized
        if hasattr(self, '_gpu_rate_array_cache'):
            self._gpu_rate_array_cache = None
            self._gpu_rate_min_cache = None
            self._gpu_rate_max_cache = None
            self._gpu_rate_changed = True  # Signal that rate data needs to be transferred to GPU

        if self.rate_type == 'scalar':
            self.rate_callable = False
            self.rate_spatial = False
        elif self.rate_type == 'quantity':
            self.rate_callable = False
            self.rate_spatial = False
        elif self.rate_type == 'centroid_array':
            self.rate_callable = False
            self.rate_spatial = False
        elif self.rate_type == 't':
            self.rate_callable = True
            self.rate_spatial = False
        else:
            self.rate_callable = True
            self.rate_spatial = True


    def set_factor(self, factor):
        """Set factor. Can change factor while running


        Can be a scalar, a function of t, or an n by 2 numpy array defining a time sequence
        """

        if isinstance(factor, num.ndarray):
            factor_shape = factor.shape
            msg =  f"The shape {factor_shape} of the input factor "
            msg += f"should be (2,n) so that a time function can be constructed"
            assert factor_shape[0] == 2, msg
            self.factor_type = 'time_sequence'
            from scipy.interpolate import interp1d
            factor = interp1d(factor[0,:], factor[1,:], kind='zero', bounds_error=False,  fill_value = (0.0, 0.0))


        from anuga.utilities.function_utils import determine_function_type
        self.factor_type = determine_function_type(factor)


        self.factor = factor

        if self.factor_type == 'scalar':
            self.factor_callable = False
        elif self.factor_type == 't':
            self.factor_callable = True
        else:
            msg = f'factor must be a scalar or a function of t. It was determined to be a function of {self.factor_type}'
            raise Exception(msg)

    def get_factor(self, t=None):
        """Provide a factor to calculate added volume
        """

        if t is None:
            t = self.get_time()

        assert isinstance(t, (int, float))


        if self.factor_type == 'scalar':
            factor = self.factor
        else:
            factor = self.factor(t)

        return factor

    def set_areas(self):

        if self.indices is None:
            self.areas = self.domain.areas
            return

        if self.indices is []:
            self.areas = []
            return

        self.areas = self.domain.areas[self.indices]

    def set_full_indices(self):

        if self.indices is None:
            self.full_indices = num.where(self.domain.tri_full_flag ==1)[0]
            return

        if self.indices is []:
            self.full_indices = []
            return

        self.full_indices = num.where(self.domain.tri_full_flag[self.indices] == 1)[0]

    def get_Q(self, full_only=True):
        """ Calculate current overall discharge
        """

        # FIXME SR: this does not take into account the zeroing of large negative rates

        factor = self.get_factor()
        
        if full_only:
            if self.rate_spatial:
                rate = self.get_spatial_rate() # rate is an array
                fid = self.full_indices
                return num.sum(self.areas[fid]*rate[fid])*factor
            elif self.rate_type == 'quantity':
                rate = self.rate.centroid_values # rate is a quantity
                fid = self.full_indices
                return num.sum(self.areas[fid]*rate[fid])*factor
            elif self.rate_type == 'centroid_array':
                rate = self.rate # rate is already a centroid sized array
                fid = self.full_indices
                return num.sum(self.areas[fid]*rate[fid])*factor
            else:
                rate = self.get_non_spatial_rate() # rate is a scalar
                fid = self.full_indices
                return num.sum(self.areas[fid]*rate)*factor
        else:
            if self.rate_spatial:
                rate = self.get_spatial_rate() # rate is an array
                return num.sum(self.areas*rate)*factor
            elif self.rate_type == 'quantity':
                rate = self.rate.centroid_values # rate is a quantity
                return num.sum(self.areas*rate)*factor
            elif self.rate_type == 'centroid_array':
                rate = self.rate # rate is already a centroid sized array
                return num.sum(self.areas*rate)*factor
            else:
                rate = self.get_non_spatial_rate() # rate is a scalar
                return num.sum(self.areas*rate)*factor

    def set_default_rate(self, default_rate):
        """
        Check and store default_rate
        """
        msg = ('Default_rate must be either None '
               'a scalar, or a function of time.\nI got %s.' % str(default_rate))
        assert (default_rate is None or
                isinstance(default_rate, (int, float)) or
                callable(default_rate)), msg


        #------------------------------------------
        # Allow longer than data
        #------------------------------------------
        if default_rate is not None:
            # If it is a constant, make it a function
            if not callable(default_rate):
                tmp = default_rate
                default_rate = lambda t: tmp

            # Check that default_rate is a function of one argument
            try:
                default_rate(0.0)
            except:
                raise Exception(msg)

        self.default_rate = default_rate

    def _prepare_xarray_rate(self, xa):

        import numpy as np
        import pandas

        # to speed up parallel code it helps to load the xarray
        self.xa = xa.load()

        self.xa['time'] = pandas.to_datetime(xa['time'], utc=True)

        # these are absolute coord (since we haven't implemented offsets)
        # Convert to relative coords to domain xllcorner and yllcorner

        xllcorner = self.domain.geo_reference.xllcorner
        yllcorner = self.domain.geo_reference.yllcorner
        self.xy = np.array([self.xa['eastings']-xllcorner, self.xa['northings']-yllcorner]).T


        # Determine data timestep from xarray. We assume the timestep is constant, so just test first 2 timeslices.
        try:
            data_dt = (self.xa['time'][1].values.astype('int64')-self.xa['time'][0].values.astype('int64'))/1.0e9
            self.domain.set_evolve_max_timestep(min(data_dt, self.domain.get_evolve_max_timestep()))
        except:  # if we can't determine the timestep probably means there is just one timeslice so just
            pass

        from scipy.spatial import KDTree
        tree = KDTree(self.xy)
        if self.verbose:
            print(tree.size, self.xy.shape)

        #dd, ii = tree.query(self.domain.centroid_coordinates)
        dd, ii = tree.query(self.domain.get_centroid_coordinates(absolute=True))

        self.ii = ii

        self.previous_Q_ref_time = None
        self.previous_Q_numpy = None


    def _update_Q_xarray(self):

        import pandas
        current_utc_datetime64 = pandas.to_datetime(self.domain.get_datetime()).tz_convert('UTC')#.replace(tzinfo=None)

        if self.verbose:
            print(f"{self.domain.get_time()} {self.domain.get_datetime()}")
            print(f"UTC time {current_utc_datetime64} type {type(current_utc_datetime64)} ")
            print(self.xa.sel(time=current_utc_datetime64, method='nearest'))
      
        try:
            Q_ref = self.xa.sel(time=current_utc_datetime64, method="ffill", tolerance='5m')

            Q_ref_time = Q_ref['time'].values

            if self.verbose:
                print(f"UTC time {current_utc_datetime64} Q_ref time {Q_ref_time} {Q_ref_time == self.previous_Q_ref_time} ")

            optimize = True
            if optimize:
                if Q_ref_time == self.previous_Q_ref_time :
                    Q_numpy = self.previous_Q_numpy
                else:
                    Q_numpy = Q_ref[self.ii].to_numpy()
                    self.previous_Q_numpy = Q_numpy
                    self.previous_Q_ref_time = Q_ref_time
            else:
                Q_numpy = Q_ref[self.ii].to_numpy()
                  
        except:
            Q_numpy = self.default_rate
            if self.verbose:
                print(f"UTC time {current_utc_datetime64} Using default rate Q = {Q_numpy(self.get_time())}")
            
        self.set_rate(rate=Q_numpy)             

    def parallel_safe(self):
        """Operator is applied independently on each cell and
        so is parallel safe.
        """
        return True

    def statistics(self):

        message = 'You need to implement operator statistics for your operator'
        return message


    def timestepping_statistics(self):

        # retrieve data from last __call__ call
        message  = indent + self.label + f': At time {self.domain.get_time()} Min rate = {self.local_min:.2e} m/s, Max rate = {self.local_max:.2e} m/s, Total Q = {self.cumulative_influx:.2e} m^3'


        # if self.rate_spatial:
        #     rate = self.get_spatial_rate()
        #     try:
        #         min_rate = num.min(rate)
        #     except ValueError:
        #         min_rate = 0.0
        #     try:
        #         max_rate = num.max(rate)
        #     except ValueError:
        #         max_rate = 0.0

        #     Q = self.get_Q()
        #     message  = indent + self.label + ': Min rate = %g m/s, Max rate = %g m/s, Total Q = %g m^3/s'% (min_rate,max_rate, Q)

        # elif self.rate_type == 'quantity':
        #     rate = self.get_non_spatial_rate() # return quantity
        #     min_rate = rate.get_minimum_value()
        #     max_rate = rate.get_maximum_value()
        #     Q = self.get_Q()
        #     message  = indent + self.label + ': Min rate = %g m/s, Max rate = %g m/s, Total Q = %g m^3/s'% (min_rate,max_rate, Q)

        # elif self.rate_type == 'centroid_array':
        #     rate = self.get_non_spatial_rate() # return centroid_array
        #     min_rate = rate.min()
        #     max_rate = rate.max()
        #     Q = self.get_Q()
        #     message  = indent + self.label + ': Min rate = %g m/s, Max rate = %g m/s, Total Q = %g m^3/s'% (min_rate,max_rate, Q)

        # else:
        #     rate = self.get_non_spatial_rate()
        #     Q = self.get_Q()
        #     message  = indent + self.label + ': Rate = %g m/s, Total Q = %g m^3/s' % (rate, Q)


        #print(message)

        return message

# ===============================================================================
# Specific Rate Operators for circular region.
# ===============================================================================
class Circular_rate_operator(Rate_operator):
    """
    Add water at certain rate (ms^{-1} = vol/Area/sec) over a
    circular region

    rate can be a function of time.

    Other units can be used by using the factor argument.

    """

    def __init__(self, domain,
                 rate=0.0,
                 factor=1.0,
                 center=None,
                 radius=None,
                 default_rate=None,
                 verbose=False):


        Rate_operator.__init__(self,
                               domain,
                               rate=rate,
                               factor=factor,
                               center=center,
                               radius=radius,
                               default_rate=default_rate,
                               verbose=verbose)



#===============================================================================
# Specific Rate Operators for polygonal region.
#===============================================================================
class Polygonal_rate_operator(Rate_operator):
    """
    Add water at certain rate (ms^{-1} = vol/Area/sec) over a
    polygonal region

    rate can be a function of time.

    Other units can be used by using the factor argument.

    """

    def __init__(self, domain,
                 rate=0.0,
                 factor=1.0,
                 polygon=None,
                 default_rate=None,
                 verbose=False):


        Rate_operator.__init__(self,
                               domain,
                               rate=rate,
                               factor=factor,
                               polygon=polygon,
                               default_rate=default_rate,
                               verbose=verbose)
