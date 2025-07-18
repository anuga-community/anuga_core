"""
Finite-volume computations of the shallow water wave equation.

Title: ANGUA shallow_water_domain - 2D triangular domains for finite-volume
       computations of the shallow water wave equation.


Author: Ole Nielsen, Ole.Nielsen@ga.gov.au
        Stephen Roberts, Stephen.Roberts@anu.edu.au
        Duncan Gray, Duncan.Gray@ga.gov.au
        Gareth Davies, gareth.davies.ga.code@gmail.com

CreationDate: 2004

Description:
    This module contains a specialisation of class Generic_Domain from
    module generic_domain.py consisting of methods specific to the
    Shallow Water Wave Equation

    U_t + E_x + G_y = S

    where

    U = [w, uh, vh]
    E = [uh, u^2h + gh^2/2, uvh]
    G = [vh, uvh, v^2h + gh^2/2]
    S represents source terms forcing the system
    (e.g. gravity, friction, wind stress, ...)

    and _t, _x, _y denote the derivative with respect to t, x and y
    respectively.


    The quantities are

    symbol    variable name    explanation

    x         x                horizontal distance from origin [m]
    y         y                vertical distance from origin [m]
    z         elevation        elevation of bed on which flow is modelled [m]
    h         height           water height above z [m]
    w         stage            absolute water level, w = z+h [m]
    u                          speed in the x direction [m/s]
    v                          speed in the y direction [m/s]
    uh        xmomentum        momentum in the x direction [m^2/s]
    vh        ymomentum        momentum in the y direction [m^2/s]


    eta                        mannings friction coefficient [to appear]
    nu                         wind stress coefficient [to appear]

    The conserved quantities are w, uh, vh

Reference:
    Catastrophic Collapse of Water Supply Reservoirs in Urban Areas,
    Christopher Zoppou and Stephen Roberts,
    Journal of Hydraulic Engineering, vol. 127, No. 7 July 1999

    Hydrodynamic modelling of coastal inundation.
    Nielsen, O., S. Roberts, D. Gray, A. McPherson and A. Hitchman
    In Zerger, A. and Argent, R.M. (eds) MODSIM 2005 International Congress on
    Modelling and Simulation. Modelling and Simulation Society of Australia and
    New Zealand, December 2005, pp. 518-523. ISBN: 0-9758400-2-9.
    http://www.mssanz.org.au/modsim05/papers/nielsen.pdf

    See also: https://anuga.anu.edu.au and https://en.wikipedia.org/wiki/ANUGA_Hydro


Constraints: See license in the user guide
"""




# Decorator added for profiling
#------------------------------


def profileit(name):
    def inner(func):
        def wrapper(*args, **kwargs):
            import cProfile
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)

            # Note use of name from outer scope
            print(str(args[1])+'_'+name)
            prof.dump_stats(str(args[1])+'_'+name)
            return retval
        return wrapper
    return inner
#-----------------------------

import numpy as num
import sys
import os
import time

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

try:
    import dill as pickle
except:
    import pickle

from anuga.abstract_2d_finite_volumes.generic_domain \
                    import Generic_Domain

from anuga.shallow_water.forcing import Cross_section
from anuga.utilities.numerical_tools import mean
from anuga.file.sww import SWW_file

import anuga.utilities.log as log

from anuga.utilities.parallel_abstraction import size, rank, get_processor_name
from anuga.utilities.parallel_abstraction import finalize, send, receive
from anuga.utilities.parallel_abstraction import pypar_available, barrier


#-----------------------------------------------------
# Code for profiling cuda version
#-----------------------------------------------------
def nvtxRangePush(*arg):
    pass
def nvtxRangePop(*arg):
    pass

try:
    from cupy.cuda.nvtx import RangePush as nvtxRangePush
    from cupy.cuda.nvtx import RangePop  as nvtxRangePop
except:
    pass

try:
    from nvtx import range_push as nvtxRangePush
    from nvtx import range_pop  as nvtxRangePop
except:
    pass


class Domain(Generic_Domain):
    """Object which encapulates the shallow water model


    This class is a specialization of class Generic_Domain from
    module generic_domain.py consisting of methods specific to the
    Shallow Water Wave Equation

    Shallow Water Wave Equation

    .. math::

        U_t + E_x + G_y = S

    where

    .. math::

        U = [w, uh, vh]^T

    .. math::

        E = [uh, u^2h + gh^2/2, uvh]

    .. math::

        G = [vh, uvh, v^2h + gh^2/2]



    S represents source terms forcing the system
    (e.g. gravity, friction, wind stress, ...)

    and _t, _x, _y denote the derivative with respect to t, x and y
    respectively.


    The quantities are

    .. list-table::
        :widths: 25 25 50
        :header-rows: 1

        * - symbol
          - variable name
          - explanation
        * - x
          - x
          - horizontal distance from origin [m]
        * - y
          - y
          - vertical distance from origin [m]
        * - z
          - elevation
          - elevation of bed on which flow is modelled [m]
        * - h
          - height
          - water height above z [m]
        * - w
          - stage
          - absolute water level, w = z+h [m]
        * - u
          -
          - speed in the x direction [m/s]
        * - v
          -
          - speed in the y direction [m/s]
        * - uh
          - xmomentum
          - momentum in the x direction [m^2/s]
        * - vh
          - ymomentum
          - momentum in the y direction [m^2/s]
        * -
          -
          -
        * - eta
          -
          - mannings friction coefficient [to appear]
        * - nu
          -
          - wind stress coefficient [to appear]

    The conserved quantities are w, uh, vh

    """

    def __init__(self,
                 coordinates=None,
                 vertices=None,
                 boundary=None,
                 tagged_elements=None,
                 geo_reference=None,
                 use_inscribed_circle=False,
                 mesh_filename=None,
                 use_cache=False,
                 verbose=False,
                 conserved_quantities = None,
                 evolved_quantities = None,
                 other_quantities = None,
                 full_send_dict=None,
                 ghost_recv_dict=None,
                 starttime=0,
                 processor=0,
                 numproc=1,
                 number_of_full_nodes=None,
                 number_of_full_triangles=None,
                 ghost_layer_width=2,
                 **kwargs):

        """Instantiate a shallow water domain.

        :param coordinates: vertex locations for the mesh
        :param vertices: vertex indices defining the triangles of the mesh
        :param boundary: boundaries of the mesh
        """

        # Define quantities for the shallow_water domain
        if conserved_quantities is None:
            conserved_quantities = ['stage', 'xmomentum', 'ymomentum']

        if evolved_quantities is None:
            evolved_quantities =  ['stage', 'xmomentum', 'ymomentum']

        if other_quantities is None:
            other_quantities = ['elevation', 'friction', 'height',
                                'xvelocity', 'yvelocity', 'x', 'y']





        Generic_Domain.__init__(self,
                            coordinates,
                            vertices,
                            boundary,
                            conserved_quantities,
                            evolved_quantities,
                            other_quantities,
                            tagged_elements,
                            geo_reference,
                            use_inscribed_circle,
                            mesh_filename,
                            use_cache,
                            verbose,
                            full_send_dict,
                            ghost_recv_dict,
                            starttime,
                            processor,
                            numproc,
                            number_of_full_nodes=number_of_full_nodes,
                            number_of_full_triangles=number_of_full_triangles,
                            ghost_layer_width=ghost_layer_width)

        #-------------------------------
        # Operator Data Structures
        #-------------------------------
        self.fractional_step_operators = []
        self.kv_operator = None
        self.dplotter = None

        #-------------------------------
        # Set flow defaults
        #-------------------------------
        self.set_flow_algorithm()

        #-------------------------------
        # Set default multiprocessor mode
        # 1. Openmp
        # 2. Cuda
        #-------------------------------
        self.gpu_interface = None
        self.set_multiprocessor_mode(1)

        #-------------------------------
        # If environment variable OMP_NUM_THREADS is not set, 
        # then set to default (1 thread). If a value is given to
        # the method, then it will override the default.
        #------------------------------
        self.set_omp_num_threads()

        #-------------------------------
        # datetime and timezone
        #-------------------------------
        self.set_timezone()

        #-------------------------------
        # Forcing Terms
        #
        # Gravity is now incorporated in
        # compute_fluxes routine
        #-------------------------------
        from .friction import manning_friction_semi_implicit
        self.forcing_terms.append(manning_friction_semi_implicit)


        #-------------------------------
        # Stored output
        #-------------------------------
        self.set_store(True)
        self.set_store_centroids(True)
        self.set_store_vertices_uniquely(False)
        self.quantities_to_be_stored = {'elevation': 1,
                                        'friction':1,
                                        'stage': 2,
                                        'xmomentum': 2,
                                        'ymomentum': 2}

        #-------------------------------
        # Set up check pointing every n
        # yieldsteps
        #-------------------------------
        self.checkpoint = False
        self.yieldstep_counter = 0
        self.checkpoint_step = 10

        #-------------------------------
        # Useful auxiliary quantity
        #-------------------------------
        n = self.number_of_elements
        self.quantities['x'].set_values(self.vertex_coordinates[:,0].reshape(n,3))
        self.quantities['x'].set_boundary_values_from_edges()

        self.quantities['y'].set_values(self.vertex_coordinates[:,1].reshape(n,3))
        self.quantities['y'].set_boundary_values_from_edges()

        # For riverwalls, we need to know the 'edge_flux_type' for each edge
        # Edge-flux-type of 0 == Normal edge, with shallow water flux
        #                   1 == riverwall
        #                   2 == ?
        #                   etc
        self.edge_flux_type=num.zeros(len(self.edge_coordinates[:,0])).astype(int)
        self.edge_river_wall_counter=num.zeros(len(self.edge_coordinates[:,0])).astype(int)
        self.number_of_riverwall_edges = 0

        # Riverwalls -- initialise with dummy values
        # Presently only works with DE algorithms, will fail otherwise
        import anuga.structures.riverwall
        self.riverwallData=anuga.structures.riverwall.RiverWall(self)
        self.create_riverwalls = self.riverwallData.create_riverwalls

        ## Keep track of the fluxes through the boundaries
        ## Only works for DE algorithms at present
        max_time_substeps=3 # Maximum number of substeps supported by any timestepping method
        # boundary_flux_sum holds boundary fluxes on each sub-step [unused substeps = 0.]
        self.boundary_flux_sum=num.array([0.]*max_time_substeps)
        from anuga.operators.boundary_flux_integral_operator import boundary_flux_integral_operator
        self.boundary_flux_integral=boundary_flux_integral_operator(self)
        # Make an integer counting how many times we call compute_fluxes_central -- so we know which substep we are on
        #self.call=1

        # List to store the volumes we computed before
        self.volume_history=[]

        # Work arrays [avoid allocate statements in compute_fluxes or extrapolate_second_order]
        self.edge_flux_work=num.zeros(len(self.edge_coordinates[:,0])*3) # Advective fluxes
        self.neigh_work=num.zeros(len(self.edge_coordinates[:,0])*3) # Advective fluxes
        self.pressuregrad_work=num.zeros(len(self.edge_coordinates[:,0])) # Gravity related terms
        self.x_centroid_work=num.zeros(len(self.edge_coordinates[:,0])//3)
        self.y_centroid_work=num.zeros(len(self.edge_coordinates[:,0])//3)

        ############################################################################
        ## Local-timestepping information
        #
        # Fluxes can be updated every 1, 2, 4, 8, .. max_flux_update_frequency timesteps
        # The global timestep is not allowed to increase except when
        # number_of_timesteps%max_flux_update_frequency==0
        self.max_flux_update_frequency=2**0 # Must be a power of 2.

        # flux_update_frequency. The edge flux terms are re-computed only when
        #    number_of_timesteps%flux_update_frequency[myEdge]==0
        self.flux_update_frequency=num.zeros(len(self.edge_coordinates[:,0])).astype(int)+1

        # Flag: should we update the flux on the next compute fluxes call?
        self.update_next_flux=num.zeros(len(self.edge_coordinates[:,0])).astype(int)+1

        # Flag: should we update the extrapolation on the next extrapolation call?
        # (Only do this if one or more of the fluxes on that triangle will be computed on
        # the next timestep, assuming only the flux computation uses edge/vertex values)
        self.update_extrapolation=num.zeros(len(self.edge_coordinates[:,0])//3).astype(int)+1

        # edge_timestep [wavespeed/radius] -- not updated every timestep
        self.edge_timestep=num.zeros(len(self.edge_coordinates[:,0]))+1.0e+100

        # Do we allow the timestep to increase (not every time if local
        # extrapolation/flux updating is used)
        self.allow_timestep_increase=num.zeros(1).astype(int)+1


        #-----------------------------------
        # parameters for structures
        #-----------------------------------
        self.use_new_velocity_head = False



    def set_plotter(self, *args, **kwargs):
        """Set the plotter for this domain
        """
        
        if self.dplotter is None:
            import anuga
            self.dplotter = anuga.Domain_plotter(self, *args, **kwargs) 

        self.triang = self.dplotter.triang
        self.stage = self.dplotter.stage
        self.xmom = self.dplotter.xmom
        self.ymom = self.dplotter.ymom
        self.elev = self.dplotter.elev
        self.friction = self.dplotter.friction
        self.xvel = self.dplotter.xvel
        self.yvel = self.dplotter.yvel
        self.x = self.dplotter.x
        self.y = self.dplotter.y
        self.xc = self.dplotter.xc
        self.yc = self.dplotter.yc

        self.save_depth_frame = self.dplotter.save_depth_frame
        self.plot_depth_frame = self.dplotter.plot_depth_frame
        self.make_depth_animation = self.dplotter.make_depth_animation

        self.save_stage_frame = self.dplotter.save_stage_frame
        self.plot_stage_frame = self.dplotter.plot_stage_frame
        self.make_stage_animation = self.dplotter.make_stage_animation        

        self.save_speed_frame = self.dplotter.save_speed_frame
        self.plot_speed_frame = self.dplotter.plot_speed_frame
        self.make_speed_animation = self.dplotter.make_speed_animation        

        
    def triplot(self, *args, **kwargs):

        self.set_plotter()

        import matplotlib.pyplot as plt
        plt.triplot(self.triang, *args, **kwargs)


    def tripcolor(self, *args, **kwargs):

        self.set_plotter()

        import matplotlib.pyplot as plt
        plt.tripcolor(self.triang,  *args, **kwargs)


    def _set_config_defaults(self):
        """Set the default values in this routine. That way we can inherit class
        and just redefine the defaults for the new class
        """

        from anuga.config import minimum_storable_height
        from anuga.config import minimum_allowed_height, maximum_allowed_speed
        from anuga.config import g
        from anuga.config import tight_slope_limiters
        from anuga.config import extrapolate_velocity_second_order
        from anuga.config import alpha_balance
        from anuga.config import optimise_dry_cells
        from anuga.config import optimised_gradient_limiter
        from anuga.config import use_edge_limiter
        from anuga.config import use_centroid_velocities
        from anuga.config import compute_fluxes_method
        from anuga.config import distribute_to_vertices_and_edges_method
        from anuga.config import sloped_mannings_function
        from anuga.config import low_froude


        # Early algorithms need elevation to remain continuous
        self.set_using_discontinuous_elevation(False)

        self.set_minimum_allowed_height(minimum_allowed_height)
        self.maximum_allowed_speed = maximum_allowed_speed

        self.minimum_storable_height = minimum_storable_height
        self.g = g

        self.alpha_balance = alpha_balance
        self.tight_slope_limiters = tight_slope_limiters

        self.set_low_froude(low_froude)

        self.set_use_optimise_dry_cells(optimise_dry_cells)
        self.set_extrapolate_velocity(extrapolate_velocity_second_order)

        self.set_use_edge_limiter(use_edge_limiter)
        self.optimised_gradient_limiter = optimised_gradient_limiter
        self.use_centroid_velocities = use_centroid_velocities

        self.set_sloped_mannings_function(sloped_mannings_function)
        self.set_compute_fluxes_method(compute_fluxes_method)

        self.set_distribute_to_vertices_and_edges_method(distribute_to_vertices_and_edges_method)

        self.set_store_centroids(False)

    def get_algorithm_parameters(self):
        """
        Get the standard parameter that are currently set (as a dictionary)
        """

        parameters = {}

        parameters['minimum_allowed_height']  = self.minimum_allowed_height
        parameters['maximum_allowed_speed']   = self.maximum_allowed_speed
        parameters['minimum_storable_height'] = self.minimum_storable_height
        parameters['g']                       = self.g
        parameters['alpha_balance']           = self.alpha_balance
        parameters['tight_slope_limiters']    = self.tight_slope_limiters
        parameters['optimise_dry_cells']      = self.optimise_dry_cells
        parameters['use_edge_limiter']        = self.use_edge_limiter
        parameters['low_froude']              = self.low_froude
        parameters['use_centroid_velocities'] = self.use_centroid_velocities
        parameters['use_sloped_mannings']     = self.use_sloped_mannings
        parameters['compute_fluxes_method']   = self.get_compute_fluxes_method()
        parameters['distribute_to_vertices_and_edges_method'] = \
                         self.get_distribute_to_vertices_and_edges_method()
        parameters['flow_algorithm']          = self.get_flow_algorithm()
        parameters['CFL']                     = self.get_CFL()
        parameters['timestepping_method']     = self.get_timestepping_method()

        parameters['optimised_gradient_limiter']        = self.optimised_gradient_limiter
        parameters['extrapolate_velocity_second_order'] = self.extrapolate_velocity_second_order



        return parameters

    def print_algorithm_parameters(self):
        """
        Print the standard parameters that are curently set (as a dictionary)
        """

        print('#============================')
        print('# Domain Algorithm Parameters ')
        print('#============================')
        from pprint import pprint
        pprint(self.get_algorithm_parameters(),indent=4)

        print('#----------------------------')




    def _set_DE0_defaults(self):
        """Set up the defaults for running the flow_algorithm "DE0"
           A 'discontinuous elevation' method
        """

        self._set_config_defaults()

        self.set_CFL(0.9)
        self.set_use_kinematic_viscosity(False)
        #self.timestepping_method='rk2'#'rk3'#'euler'#'rk2'
        self.set_timestepping_method('euler')

        self.set_using_discontinuous_elevation(True)
        self.set_using_centroid_averaging(True)
        self.set_compute_fluxes_method('DE')
        self.set_distribute_to_vertices_and_edges_method('DE')

        # Don't place any restriction on the minimum storable height
        #self.minimum_storable_height=-99999999999.0
        self.minimum_allowed_height=1.0e-12

        self.use_edge_limiter=True
        self.set_default_order(2)
        self.set_extrapolate_velocity()

        self.beta_w=0.5
        self.beta_w_dry=0.0
        self.beta_uh=0.5
        self.beta_uh_dry=0.0
        self.beta_vh=0.5
        self.beta_vh_dry=0.0


        #self.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2,
        #         'ymomentum': 2, 'elevation': 2, 'height':2})
        #self.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2,
        #         'ymomentum': 2, 'elevation': 1})
        self.set_store_centroids(True)

        self.optimise_dry_cells=False

        # We need the edge_coordinates for the extrapolation
        self.edge_coordinates=self.get_edge_midpoint_coordinates()

        # By default vertex values are NOT stored uniquely
        # for storage efficiency. We may override this (but not so important since
        # centroids are stored anyway
        # self.set_store_vertices_smoothly(False)

        self.maximum_allowed_speed=0.0

        if self.processor == 0 and self.verbose:
            print('##########################################################################')
            print('#')
            print('# Using discontinuous elevation solver DE0')
            print('#')
            print('# First order timestepping')
            print('#')
            print('# Make sure you use centroid values when reporting on important output quantities')
            print('#')
            print('##########################################################################')


    def _set_DE1_defaults(self):
        """Set up the defaults for running the flow_algorithm "DE1"
           A 'discontinuous elevation' method
        """

        self._set_config_defaults()

        self.set_CFL(1.0)
        self.set_use_kinematic_viscosity(False)
        #self.timestepping_method='rk2'#'rk3'#'euler'#'rk2'
        self.set_timestepping_method(2)

        self.set_using_discontinuous_elevation(True)
        self.set_using_centroid_averaging(True)
        self.set_compute_fluxes_method('DE')
        self.set_distribute_to_vertices_and_edges_method('DE')

        # Don't place any restriction on the minimum storable height
        #self.minimum_storable_height=-99999999999.0
        self.minimum_allowed_height=1.0e-5

        self.use_edge_limiter=True
        self.set_default_order(2)
        self.set_extrapolate_velocity()

        self.beta_w=1.0
        self.beta_w_dry=0.0
        self.beta_uh=1.0
        self.beta_uh_dry=0.0
        self.beta_vh=1.0
        self.beta_vh_dry=0.0


        #self.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2,
        #         'ymomentum': 2, 'elevation': 2, 'height':2})
        #self.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2,
        #         'ymomentum': 2, 'elevation': 1})
        self.set_store_centroids(True)

        self.optimise_dry_cells=False

        # We need the edge_coordinates for the extrapolation
        self.edge_coordinates=self.get_edge_midpoint_coordinates()

        # By default vertex values are NOT stored uniquely
        # for storage efficiency. We may override this (but not so important since
        # centroids are stored anyway
        # self.set_store_vertices_smoothly(False)

        self.maximum_allowed_speed=0.0


        if self.processor == 0 and self.verbose:
            print('##########################################################################')
            print('#')
            print('# Using discontinuous elevation solver DE1 ')
            print('#')
            print('# Uses rk2 timestepping')
            print('#')
            print('# Make sure you use centroid values when reporting on important output quantities')
            print('#')
            print('##########################################################################')

    def _set_DE2_defaults(self):
        """Set up the defaults for running the flow_algorithm "DE2"
           A 'discontinuous elevation' method
        """

        self._set_config_defaults()

        self.set_CFL(1.0)
        self.set_use_kinematic_viscosity(False)
        #self.timestepping_method='rk2'#'rk3'#'euler'#'rk2'
        self.set_timestepping_method(3)

        self.set_using_discontinuous_elevation(True)
        self.set_using_centroid_averaging(True)
        self.set_compute_fluxes_method('DE')
        self.set_distribute_to_vertices_and_edges_method('DE')

        # Don't place any restriction on the minimum storable height
        #self.minimum_storable_height=-99999999999.0
        self.minimum_allowed_height=1.0e-5

        self.use_edge_limiter=True
        self.set_default_order(2)
        self.set_extrapolate_velocity()

        self.beta_w=1.0
        self.beta_w_dry=0.0
        self.beta_uh=1.0
        self.beta_uh_dry=0.0
        self.beta_vh=1.0
        self.beta_vh_dry=0.0


        #self.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2,
        #         'ymomentum': 2, 'elevation': 2, 'height':2})
        #self.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2,
        #         'ymomentum': 2, 'elevation': 1})
        self.set_store_centroids(True)

        self.optimise_dry_cells=False

        # We need the edge_coordinates for the extrapolation
        self.edge_coordinates=self.get_edge_midpoint_coordinates()

        # By default vertex values are NOT stored uniquely
        # for storage efficiency. We may override this (but not so important since
        # centroids are stored anyway
        # self.set_store_vertices_smoothly(False)

        self.maximum_allowed_speed=0.0


        if self.processor == 0 and self.verbose:
            print('##########################################################################')
            print('#')
            print('# Using discontinuous elevation solver DE2')
            print('#')
            print('# Using rk3 timestepping')
            print('#')
            print('# Make sure you use centroid values when reporting on important output quantities')
            print('#')
            print('##########################################################################')

    def _set_DE1_7_defaults(self):
        """Set up the defaults for running the flow_algorithm "DE0_7"
           A 'discontinuous elevation' method
        """

        self._set_config_defaults()

        self.set_CFL(1.0)
        self.set_use_kinematic_viscosity(False)
        #self.timestepping_method='rk2'#'rk3'#'euler'#'rk2'
        self.set_timestepping_method(2)

        self.set_using_discontinuous_elevation(True)
        self.set_using_centroid_averaging(True)
        self.set_compute_fluxes_method('DE')
        self.set_distribute_to_vertices_and_edges_method('DE')

        # Don't place any restriction on the minimum storable height
        #self.minimum_storable_height=-99999999999.0
        self.minimum_allowed_height=1.0e-12

        self.use_edge_limiter=True
        self.set_default_order(2)
        self.set_extrapolate_velocity()

        self.beta_w=0.75
        self.beta_w_dry=0.1
        self.beta_uh=0.75
        self.beta_uh_dry=0.1
        self.beta_vh=0.75
        self.beta_vh_dry=0.1


        #self.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2,
        #         'ymomentum': 2, 'elevation': 2, 'height':2})
        #self.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2,
        #         'ymomentum': 2, 'elevation': 1})
        self.set_store_centroids(True)

        self.optimise_dry_cells=False

        # We need the edge_coordinates for the extrapolation
        self.edge_coordinates=self.get_edge_midpoint_coordinates()

        # By default vertex values are NOT stored uniquely
        # for storage efficiency. We may override this (but not so important since
        # centroids are stored anyway
        # self.set_store_vertices_smoothly(False)

        self.maximum_allowed_speed=0.0

        if self.processor == 0 and self.verbose:
            print('##########################################################################')
            print('#')
            print('# Using discontinuous elevation solver DE1_7 ')
            print('#')
            print('# A slightly more diffusive version of DE1, does use rk2 timestepping')
            print('#')
            print('# Make sure you use centroid values when reporting on important output quantities')
            print('#')
            print('##########################################################################')


    def _set_DE0_7_defaults(self):
        """Set up the defaults for running the flow_algorithm "DE3"
           A 'discontinuous elevation' method
        """

        self._set_config_defaults()

        self.set_CFL(0.9)
        self.set_use_kinematic_viscosity(False)
        #self.timestepping_method='rk2'#'rk3'#'euler'#'rk2'
        self.set_timestepping_method(1)

        self.set_using_discontinuous_elevation(True)
        self.set_using_centroid_averaging(True)
        self.set_compute_fluxes_method('DE')
        self.set_distribute_to_vertices_and_edges_method('DE')

        # Don't place any restriction on the minimum storable height
        #self.minimum_storable_height=-99999999999.0
        self.minimum_allowed_height=1.0e-12

        self.use_edge_limiter=True
        self.set_default_order(2)
        self.set_extrapolate_velocity()

        self.beta_w=0.7
        self.beta_w_dry=0.1
        self.beta_uh=0.7
        self.beta_uh_dry=0.1
        self.beta_vh=0.7
        self.beta_vh_dry=0.1


        #self.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2,
        #         'ymomentum': 2, 'elevation': 2, 'height':2})
        #self.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2,
        #         'ymomentum': 2, 'elevation': 1})
        self.set_store_centroids(True)

        self.optimise_dry_cells=False

        # We need the edge_coordinates for the extrapolation
        self.edge_coordinates=self.get_edge_midpoint_coordinates()

        # By default vertex values are NOT stored uniquely
        # for storage efficiency. We may override this (but not so important since
        # centroids are stored anyway
        # self.set_store_vertices_smoothly(False)

        self.maximum_allowed_speed=0.0

        if self.processor == 0 and self.verbose:
            print('##########################################################################')
            print('#')
            print('# Using discontinuous elevation solver DE0_7')
            print('#')
            print('# A slightly less diffusive version than DE0, uses euler timestepping')
            print('#')
            print('# Make sure you use centroid values when reporting on important output quantities')
            print('#')
            print('##########################################################################')



    def update_special_conditions(self):

        my_update_special_conditions(self)

    # Note Padarn 06/12/12: The following line decorates
    # the set_quantity function to be profiled individually.
    # Need to uncomment the decorator at top of file.
    #@profileit("set_quantity.profile")
    def set_quantity(self, name, *args, **kwargs):
        """Set values for named quantity

        We have to do something special for 'elevation'
        otherwise pass through to generic set_quantity
        """

#        if name == 'elevation':
#            stage_c = self.get_quantity('stage').centroid_values
#            elev_c =  self.get_quantity('elevation').centroid_values
#            height_c = stage_c - elev_c
#            Generic_Domain.set_quantity(self, name, *args, **kwargs)
#            stage_c[:] = elev_c + height_c
#        else:
#            Generic_Domain.set_quantity(self, name, *args, **kwargs)

        Generic_Domain.set_quantity(self, name, *args, **kwargs)


    def set_timezone(self, tz = None):
        """Set timezone for domain

        :param tz: either a timezone object or string

        We recommend using the ZoneInfo provided by zoneinfo.
        Default is ZoneInfo('UTC')

        Example: Set default timezone UTC

        >>> domain.set_timezone()

        Example: Set timezone using tsdata string

        >>> domain.set_timezone('Australia/Syndey')

        Example: Set timezone using ZoneInfo timezone

        >>> from zoneinfo import ZoneInfo
        >>> new_tz = ZoneInfo('Australia/Sydney')
        >>> domain.set_timezone(new_tz)
        """

        try:
            from zoneinfo import ZoneInfo
        except:
            from backports.zoneinfo import ZoneInfo

        if tz is None:
            new_tz = ZoneInfo('UTC')
        elif isinstance(tz,str):
            new_tz = ZoneInfo(tz)
        elif isinstance(tz, ZoneInfo):
            new_tz = tz
        else:
            msg = "Unknown timezone %s" % tz
            raise Exception(msg)


        self.timezone = new_tz

    def get_timezone(self):
        """Retrieve current domain timezone"""

        return self.timezone

    def get_datetime(self, timestamp=None):
        """Retrieve datetime corresponding to current timestamp wrt to domain timezone

        param: timestamp: return datetime corresponding to given timestamp"""

        from datetime import datetime
        
        try:
            from datetime import UTC
        except:
            from datetime import timezone
            UTC = timezone.utc 


        if timestamp is None:
            timestamp = self.get_time()

        #utc_datetime = datetime.utcfromtimestamp(timestamp).replace(tzinfo=ZoneInfo('UTC'))
        utc_datetime = datetime.fromtimestamp(timestamp,UTC)

        current_dt = utc_datetime.astimezone(self.timezone)
        return current_dt

    def set_starttime(self, timestamp=0.0):
        """Set the starttime for the evolution

        :param timestamp: Either a float or a datetime object

        Essentially we use unix time as our absolute time. So
        time = 0 corresponds to Jan 1st 1970 UTC

        Use naive datetime which will be localized to the domain timezone or
        or use zoneinfo.ZoneInfo to set the timezone of datetime.
        Don't use the tzinfo argument of datetime to set timezone as this does not work!

        Example:

            Without setting timezone for the `domain` and the `starttime` then time
            calculations are all based on UTC. Note the timestamp, which is time in seconds
            from 1st Jan 1970 UTC.

        >>> import anuga
        >>> from zoneinfo import ZoneInfo
        >>> from datetime import datetime
        >>>
        >>> domain = anuga.rectangular_cross_domain(10,10)
        >>> dt = datetime(2021,3,21,18,30)
        >>> domain.set_starttime(dt)
        >>> print(domain.get_datetime(), 'TZ', domain.get_timezone(), 'Timestamp: ', domain.get_time())
        2021-03-21 18:30:00+00:00 TZ UTC Timestamp:  1616351400.0

        Example:

            Setting timezone for the `domain`, then naive `datetime` will be localizes to
            the `domain` timezone. Note the timestamp, which is time in seconds
            from 1st Jan 1970 UTC.

        >>> import anuga
        >>> from zoneinfo import ZoneInfo
        >>> from datetime import datetime
        >>>
        >>> domain = anuga.rectangular_cross_domain(10,10)
        >>> AEST = ZoneInfo('Australia/Sydney')
        >>> domain.set_timezone(AEST)
        >>>
        >>> dt = datetime(2021,3,21,18,30)
        >>> domain.set_starttime(dt)
        >>> print(domain.get_datetime(), 'TZ', domain.get_timezone(), 'Timestamp: ', domain.get_time())
        2021-03-21 18:30:00+11:00 TZ Australia/Sydney Timestamp:  1616311800.0

        Example:

            Setting timezone for the `domain`, and setting the timezone for the `datetime`.
            Note the timestamp, which is time in seconds from 1st Jan 1970 UTC is the same
            as the previous example.

        >>> import anuga
        >>> from zoneinfo import ZoneInfo
        >>> from datetime import datetime
        >>>
        >>> domain = anuga.rectangular_cross_domain(10,10)
        >>>
        >>> ACST = ZoneInfo('Australia/Adelaide')
        >>> domain.set_timezone(ACST)
        >>>
        >>> AEST = ZoneInfo('Australia/Sydney')
        >>> dt = datetime(2021,3,21,18,30, tzinfo=AEST)
        >>>
        >>> domain.set_starttime(dt)
        >>> print(domain.get_datetime(), 'TZ', domain.get_timezone(), 'Timestamp: ', domain.get_time())
        2021-03-21 18:00:00+10:30 TZ Australia/Adelaide Timestamp:  1616311800.0
        """


        from datetime import datetime

        if self.evolved_called:
            msg = ('Can\'t change simulation start time once evolve has '
                   'been called')
            raise Exception(msg)

        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None:
                dt = timestamp.replace(tzinfo=self.timezone)
                time = dt.timestamp()
            else:
                time = timestamp.timestamp()
        else:
            time = float(timestamp)


        self.starttime = time
        # starttime is now the origin for relative_time
        self.set_relative_time(0.0)

    def get_starttime(self, datetime=False):
        """return starttime, either as timestamp, or as a datetime"""

        starttime = self.starttime

        if not datetime:
            return starttime
        else:
            return self.get_datetime(starttime)


    def set_store(self, flag=True):
        """Set whether data saved to sww file.
        """

        self.store = flag

    def get_store(self):
        """Get whether data saved to sww file.
        """

        return self.store


    def set_store_centroids(self, flag=True):
        """Set whether centroid data is saved to sww file.
        """

        self.store_centroids = flag

    def get_store_centroids(self):
        """Get whether data saved to sww file.
        """

        return self.store_centroids

    def set_checkpointing(self, checkpoint= True, checkpoint_dir = 'CHECKPOINTS', checkpoint_step=10, checkpoint_time = None):
        """Set up checkpointing.

        @param checkpoint: Default = True. Set to False will turn off checkpointing
        @param checkpoint_dir: Where to store checkpointing files
        @param checkpoint_step: Save checkpoint files after this many yieldsteps
        @param checkpoint_time: If set, over-rides checkpoint_step. save checkpoint files
        after this amount of walltime
        """



        if checkpoint:

            from anuga import myid
            # On processor 0 create checkpoint directory if necessary
            if myid == 0:
                if True:
                    if not os.path.exists(checkpoint_dir):
                        os.mkdir(checkpoint_dir)

                    assert os.path.exists(checkpoint_dir)

            self.checkpoint_dir = checkpoint_dir
            if checkpoint_time is not None:
                #import time
                self.walltime_prev = time.time()
                self.checkpoint_time = checkpoint_time
                self.checkpoint_step = 0
            else:
                self.checkpoint_step = checkpoint_step
            self.checkpoint = True
            #print(self.checkpoint_dir, self.checkpoint_step)
        else:
            self.checkpoint = False

    def set_sloped_mannings_function(self, flag=True):
        """Set mannings friction function to use the sloped
        wetted area.

        The flag is tested in the python wrapper
        mannings_friction_implicit
        """
        if flag:
            self.use_sloped_mannings = True
        else:
            self.use_sloped_mannings = False


    def set_compute_fluxes_method(self, flag='original'):
        """Set method for computing fluxes.

        Currently
           original
           wb_1
           wb_2
           wb_3
           tsunami
           DE
        """
        compute_fluxes_methods = ['original', 'wb_1', 'wb_2', 'wb_3', 'tsunami', 'DE']

        if flag in compute_fluxes_methods:
            self.compute_fluxes_method = flag
        else:
            msg = 'Unknown compute_fluxes_method. \nPossible choices are:\n'+ \
            ', '.join(compute_fluxes_methods)+'.'
            raise Exception(msg)


    def set_local_extrapolation_and_flux_updating(self,nlevels=8):
        """
            Use local flux and extrapolation updating

            nlevels == number of flux_update_frequency levels > 1

                   For example, to allow flux updating every 1,2,4,8
                   timesteps, do:

                    domain.set_local_extrapolation_and_flux_updating(nlevels=3)

                   (since 2**3==8)
        """

        self.max_flux_update_frequency=2**nlevels

        if(self.max_flux_update_frequency != 1):
            if self.timestepping_method != 'euler':
                raise Exception('Local extrapolation and flux updating only supported with euler timestepping')
            if self.compute_fluxes_method != 'DE':
                raise Exception('Local extrapolation and flux updating only supported for discontinuous flow algorithms')


    def get_compute_fluxes_method(self):
        """Get method for computing fluxes.

        See set_compute_fluxes_method for possible choices.
        """

        return self.compute_fluxes_method



    def set_distribute_to_vertices_and_edges_method(self, flag='original'):
        """Set method for computing fluxes.

        Currently
           original
           tsunami
        """
        distribute_to_vertices_and_edges_methods = ['original',  'tsunami', 'DE']

        if flag in distribute_to_vertices_and_edges_methods:
            self.distribute_to_vertices_and_edges_method = flag
        else:
            msg = 'Unknown distribute_to_vertices_and_edges_method. \nPossible choices are:\n'+ \
            ', '.join(distribute_to_vertices_and_edges_methods)+'.'
            raise Exception(msg)





    def get_distribute_to_vertices_and_edges_method(self):
        """Get method for distribute_to_vertices_and_edges.

        See set_distribute_to_vertices_and_edges_method for possible choices.
        """

        return self.distribute_to_vertices_and_edges_method





    def set_flow_algorithm(self, algorithm='DE0'):
        """Set combination of slope limiting and time stepping

        Currently
           DE0
           DE1
           DE2
           DE0_7
           DE1_7
        """

        algorithm = str(algorithm)

        # Replace any dots with dashes
        algorithm = algorithm.replace('.', '_')


        flow_algorithms = ['DE0', 'DE1', 'DE2', \
                           'DE0_7', 'DE1_7']

        if algorithm in flow_algorithms:
            self.flow_algorithm = algorithm
        else:
            msg = 'Unknown flow_algorithm. \nPossible choices are:\n'+ \
            ', '.join(flow_algorithms)+'.'
            raise Exception(msg)

        if self.flow_algorithm == 'DE0':
            self._set_DE0_defaults()

        if self.flow_algorithm == 'DE1':
            self._set_DE1_defaults()

        if self.flow_algorithm == 'DE2':
            self._set_DE2_defaults()

        if self.flow_algorithm == 'DE0_7':
            self._set_DE0_7_defaults()

        if self.flow_algorithm == 'DE1_7':
            self._set_DE1_7_defaults()


    def get_flow_algorithm(self):
        """
        Get method used for timestepping and spatial discretisation

        """

        return self.flow_algorithm


    # def set_gravity_method(self):
    #     """Gravity method is determined by the compute_fluxes_method
    #     This is now not used, as gravity is combine in the compute_fluxes method
    #     """

    #     if  self.get_compute_fluxes_method() == 'original':
    #         self.forcing_terms[0] = gravity

    #     elif self.get_compute_fluxes_method() == 'wb_1':
    #         self.forcing_terms[0] = gravity_wb

    #     elif self.get_compute_fluxes_method() == 'wb_2':
    #         self.forcing_terms[0] = gravity

    #     else:
    #         raise Exception('undefined compute_fluxes method')

    def set_extrapolate_velocity(self, flag=True):
        """ Extrapolation routine uses momentum by default,
        can change to velocity extrapolation which
        seems to work better.
        """

        if flag is True:
            self.extrapolate_velocity_second_order = True
        elif flag is False:
            self.extrapolate_velocity_second_order = False

    def set_use_edge_limiter(self, flag=True):
        """ Extrapolation routine uses vertex values by default,
        for limiting, can change to edge limiting which
        seems to work better in some cases.
        """

        if flag is True:
            self.use_edge_limiter = True
        elif flag is False:
            self.use_edge_limiter = False

    def set_low_froude(self, low_froude=0):
        """ For low Froude problems the standard flux calculations
        can lead to excessive damping. Set low_froude to 1 or 2 for
        flux calculations which minimize the damping in this case.
        """

        assert low_froude in [0,1,2]

        self.low_froude = low_froude

    def set_use_optimise_dry_cells(self, flag=True):
        """ Try to optimize calculations where region is dry
        """

        if flag is True:
            self.optimise_dry_cells = int(True)
        elif flag is False:
            self.optimise_dry_cells = int(False)




    def set_use_kinematic_viscosity(self, flag=True):

        from anuga.operators.kinematic_viscosity_operator import Kinematic_viscosity_operator

        if flag :
            # Create Operator if necessary
            if self.kv_operator is None:
                self.kv_operator = Kinematic_viscosity_operator(self)
        else:
            if self.kv_operator is None:
                return
            else:
                # Remove operator from fractional_step_operators
                self.fractional_step_operators.remove(self.kv_operator)
                self.kv_operator = None





    def set_beta(self, beta):
        """Shorthand to assign one constant value [0,2] to all limiters.
        0 Corresponds to first order, where as larger values make use of
        the second order scheme.
        """

        self.beta_w = beta
        self.beta_w_dry = beta
        self.quantities['stage'].beta = beta

        self.beta_uh = beta
        self.beta_uh_dry = beta
        self.quantities['xmomentum'].beta = beta

        self.beta_vh = beta
        self.beta_vh_dry = beta
        self.quantities['ymomentum'].beta = beta


    def set_betas(self, beta_w, beta_w_dry, beta_uh, beta_uh_dry, beta_vh, beta_vh_dry):
        """Assign beta values in the range  [0,2] to all limiters.
        0 Corresponds to first order, where as larger values make use of
        the second order scheme.
        """

        self.beta_w = beta_w
        self.beta_w_dry = beta_w_dry
        self.quantities['stage'].beta = beta_w

        self.beta_uh = beta_uh
        self.beta_uh_dry = beta_uh_dry
        self.quantities['xmomentum'].beta = beta_uh

        self.beta_vh = beta_vh
        self.beta_vh_dry = beta_vh_dry
        self.quantities['ymomentum'].beta = beta_vh


    def set_store_vertices_uniquely(self, flag=True, reduction=None):
        """Decide whether vertex values should be stored uniquely as
        computed in the model (True) or whether they should be reduced to one
        value per vertex using self.reduction (False).
        """

        # FIXME (Ole): how about using the word "continuous vertex values" or
        # "continuous stage surface"
        self.smooth = not flag

        # Reduction operation for get_vertex_values
        if reduction is None:
            self.reduction = mean
            #self.reduction = min  #Looks better near steep slopes

    def set_store_vertices_smoothly(self, flag=True, reduction=None):
        """Decide whether vertex values should be stored smoothly (one value per vertex)
        or uniquely as
        computed in the model (False).
        """

        # FIXME (Ole): how about using the word "continuous vertex values" or
        # "continuous stage surface"
        self.smooth = flag

        # Reduction operation for get_vertex_values
        if reduction is None:
            self.reduction = mean
            #self.reduction = min  #Looks better near steep slopes

    def set_minimum_storable_height(self, minimum_storable_height):
        """Set the minimum depth that will be written to an SWW file.

        minimum_storable_height  minimum allowed SWW depth is in meters

        This is useful for removing thin water layers that seems to be caused
        by friction creep.
        """

        self.minimum_storable_height = minimum_storable_height


    def get_minimum_storable_height(self):

        return self.minimum_storable_height


    def set_minimum_allowed_height(self, minimum_allowed_height):
        """Set minimum depth that will be recognised in the numerical scheme.

        minimum_allowed_height  minimum allowed depth in meters

        The parameter H0 (Minimal height for flux computation) is also set by
        this function.
        """

        #FIXME (Ole): rename H0 to minimum_allowed_height_in_flux_computation

        #FIXME (Ole): Maybe use histogram to identify isolated extreme speeds
        #and deal with them adaptively similarly to how we used to use 1 order
        #steps to recover.

        self.minimum_allowed_height = minimum_allowed_height
        self.H0 = minimum_allowed_height



    def get_minimum_allowed_height(self):

        return self.minimum_allowed_height

    def set_maximum_allowed_speed(self, maximum_allowed_speed):
        """Set the maximum particle speed that is allowed in water shallower
        than minimum_allowed_height.

        maximum_allowed_speed

        This is useful for controlling speeds in very thin layers of water and
        at the same time allow some movement avoiding pooling of water.
        """

        self.maximum_allowed_speed = maximum_allowed_speed

    def set_points_file_block_line_size(self, points_file_block_line_size):
        """
        """

        self.points_file_block_line_size = points_file_block_line_size


    # FIXME: Probably obsolete in its curren form
    def set_quantities_to_be_stored(self, q):
        """Specify which quantities will be stored in the SWW file.

        q must be either:
          - a dictionary with quantity names
          - a list of quantity names (for backwards compatibility)
          - None

        The format of the dictionary is as follows

        quantity_name: flag where flag must be either 1 or 2.
        If flag is 1, the quantity is considered static and will be
        stored once at the beginning of the simulation in a 1D array.

        If flag is 2, the quantity is considered time dependent and
        it will be stored at each yieldstep by appending it to the
        appropriate 2D array in the sww file.

        If q is None, storage will be switched off altogether.

        Once the simulation has started and thw sww file opened,
        this function will have no effect.

        The format, where q is a list of names is for backwards compatibility
        only.
        It will take the specified quantities to be time dependent and assume
        'elevation' to be static regardless.
        """

        if q is None:
            self.quantities_to_be_stored = {}
            self.store = False
            return

        # Check correctness
        for quantity_name in q:
            msg = ('Quantity %s is not a valid conserved quantity'
                   % quantity_name)
            assert quantity_name in self.quantities, msg

        assert isinstance(q, dict)
        self.quantities_to_be_stored = q

    def get_wet_elements(self, indices=None, minimum_height=None):
        """Return indices for elements where h > minimum_allowed_height

        Optional argument:
            indices is the set of element ids that the operation applies to.

        Usage:
            indices = get_wet_elements()

        Note, centroid values are used for this operation
        """

        # Water depth below which it is considered to be 0 in the model
        # FIXME (Ole): Allow this to be specified as a keyword argument as well
        from anuga.config import minimum_allowed_height

        if minimum_height is None:
            minimum_height = minimum_allowed_height

        elevation = self.get_quantity('elevation').\
                        get_values(location='centroids', indices=indices)
        stage = self.get_quantity('stage').\
                    get_values(location='centroids', indices=indices)
        depth = stage - elevation

        # Select indices for which depth > 0
        wet_indices = num.compress(depth > minimum_height,
                                   num.arange(len(depth)))
        return wet_indices

    def get_maximum_inundation_elevation(self, indices=None, minimum_height=None):
        """Return highest elevation where h > 0

        Optional argument:
            indices is the set of element ids that the operation applies to.
            minimum_height for testing h > minimum_height
        Usage:
            q = get_maximum_inundation_elevation()

        Note, centroid values are used for this operation
        """

        wet_elements = self.get_wet_elements(indices, minimum_height)
        return self.get_quantity('elevation').\
                   get_maximum_value(indices=wet_elements)

    def get_maximum_inundation_location(self, indices=None):
        """Return location of highest elevation where h > 0

        Optional argument:
            indices is the set of element ids that the operation applies to.

        Usage:
            q = get_maximum_inundation_location()

        Note, centroid values are used for this operation
        """

        wet_elements = self.get_wet_elements(indices)
        return self.get_quantity('elevation').\
                   get_maximum_location(indices=wet_elements)


    def get_water_volume(self):

        from anuga import numprocs

        #print self.evolved_called

        if not self.evolved_called:
            Stage = self.quantities['stage']
            Elev =  self.quantities['elevation']
            h_c = Stage.centroid_values - Elev.centroid_values
            #print h_c
            from anuga import Quantity
            Height = Quantity(self)
            Height.set_values(h_c, location='centroids')
            #print Height.centroid_values
            volume = Height.get_integral()
        elif self.get_using_discontinuous_elevation():
            Height = self.quantities['height']
            volume = Height.get_integral()
        else:
            Stage = self.quantities['stage']
            Elev =  self.quantities['elevation']
            Height = Stage-Elev
            volume = Height.get_integral()

        if numprocs == 1:
            self.volume_history.append(volume)
            return volume

        # isolated parallel code
        from anuga import myid, send, receive, barrier

        if myid == 0:
            water_volume = volume
            for i in range(1,numprocs):
                remote_volume = receive(i)
                water_volume = water_volume + remote_volume
        else:
            send(volume,0)

        #barrier()

        if myid == 0:
            for i in range(1,numprocs):
                send(water_volume,i)
        else:
            water_volume = receive(0)

        self.volume_history.append(water_volume)
        return water_volume

    def get_boundary_flux_integral(self):
        """Compute the boundary flux integral.

        Should work in parallel
        """

        from anuga import numprocs

        if not self.compute_fluxes_method=='DE':
            msg='Boundary flux integral only supported for DE fluxes '+\
                '(because computation of boundary_flux_sum is only implemented there)'
            raise Exception(msg)

        flux_integral = self.boundary_flux_integral.boundary_flux_integral[0]

        if numprocs == 1:
            return flux_integral

        # isolate parallel code
        from anuga import myid, send, receive, barrier

        if myid == 0:
            for i in range(1,numprocs):
                remote_flux_integral = receive(i)
                flux_integral = flux_integral + remote_flux_integral
        else:
            send(flux_integral,0)

        #barrier()

        if myid == 0:
            for i in range(1,numprocs):
                send(flux_integral,i)
        else:
            flux_integral = receive(0)

        return flux_integral

    def get_fractional_step_volume_integral(self):
        """Compute the integrated flows from fractional steps.

        This requires that the fractional step operators update the fractional_step_volume_integral.

        Should work in parallel
        """

        from anuga import numprocs

        flux_integral = self.fractional_step_volume_integral

        if numprocs == 1:
            return flux_integral

        # isolate parallel code
        from anuga import myid, send, receive, barrier

        if myid == 0:
            for i in range(1,numprocs):
                remote_flux_integral = receive(i)
                flux_integral = flux_integral + remote_flux_integral
        else:
            send(flux_integral,0)

        #barrier()

        if myid == 0:
            for i in range(1,numprocs):
                send(flux_integral,i)
        else:
            flux_integral = receive(0)

        return flux_integral

    def get_flow_through_cross_section(self, polyline, verbose=False):
        """Get the total flow through an arbitrary poly line.

        This is a run-time equivalent of the function with same name
        in sww_interrogate.py

        Input:
            polyline: Representation of desired cross section - it may contain
                      multiple sections allowing for complex shapes. Assume
                      absolute UTM coordinates.
                      Format [[x0, y0], [x1, y1], ...]

        Output:
            Q: Total flow [m^3/s] across given segments.
        """


        cross_section = Cross_section(self, polyline, verbose)

        return cross_section.get_flow_through_cross_section()


    def get_energy_through_cross_section(self, polyline,
                                         kind='total',
                                         verbose=False):
        """Obtain average energy head [m] across specified cross section.

        Inputs:
            polyline: Representation of desired cross section - it may contain
                      multiple sections allowing for complex shapes. Assume
                      absolute UTM coordinates.
                      Format [[x0, y0], [x1, y1], ...]
            kind:     Select which energy to compute.
                      Options are 'specific' and 'total' (default)

        Output:
            E: Average energy [m] across given segments for all stored times.

        The average velocity is computed for each triangle intersected by
        the polyline and averaged weighted by segment lengths.

        The typical usage of this function would be to get average energy of
        flow in a channel, and the polyline would then be a cross section
        perpendicular to the flow.

        #FIXME (Ole) - need name for this energy reflecting that its dimension
        is [m].
        """



        cross_section = Cross_section(self, polyline, verbose)

        return cross_section.get_energy_through_cross_section(kind)


    def check_integrity(self):
        """ Run integrity checks on shallow water domain. """
        Generic_Domain.check_integrity(self)

        #Check that we are solving the shallow water wave equation
        msg = 'First conserved quantity must be "stage"'
        assert self.conserved_quantities[0] == 'stage', msg
        msg = 'Second conserved quantity must be "xmomentum"'
        assert self.conserved_quantities[1] == 'xmomentum', msg
        msg = 'Third conserved quantity must be "ymomentum"'
        assert self.conserved_quantities[2] == 'ymomentum', msg


    #@profile
    def extrapolate_second_order_sw(self):
        """Fast version of extrapolation from centroids to edges"""

        # FIXME (Ole): This might be an obsolute function (it was migrated from from the old c extension)
        from .sw_domain_orig_ext import extrapolate_second_order_sw
        extrapolate_second_order_sw(self)

    #@profile
    def compute_fluxes(self):
        """Compute fluxes and timestep suitable for all volumes in domain.

        Compute total flux for each conserved quantity using "flux_function"

        Fluxes across each edge are scaled by edgelengths and summed up
        Resulting flux is then scaled by area and stored in
        explicit_update for each of the three conserved quantities
        stage, xmomentum and ymomentum

        The maximal allowable speed computed by the flux_function for each volume
        is converted to a timestep that must not be exceeded. The minimum of
        those is computed as the next overall timestep.

        Post conditions:
          domain.explicit_update is reset to computed flux values
          domain.flux_timestep is set to the largest step satisfying all volumes.

        This wrapper calls the underlying C version of compute fluxes
        """

        # Using Gareth Davies discontinuous elevation scheme
        # Flux calculation and gravity incorporated in same
        # procedure

        # nvtxRangePush("Compute Fluxes (Domain)")
        # Choose the correct extension module
        if self.multiprocessor_mode == 1:
            from .sw_domain_openmp_ext import compute_fluxes_ext_central
        elif self.multiprocessor_mode == 2:
            # change over to cuda routines as developed
            # from .sw_domain_simd_ext import compute_fluxes_ext_central
            # FIXME SR: 2023_10_16 currently compute_fluxes and distribute together
            # is producing incorrect results, but work separately!
            compute_fluxes_ext_central = self.gpu_interface.compute_fluxes_ext_central_kernel
        else:
            raise Exception('Not implemented')

        timestep = self.evolve_max_timestep
        self.flux_timestep = compute_fluxes_ext_central(self, timestep)

        # nvtxRangePop()

        
    def distribute_to_vertices_and_edges(self):
        """ extrapolate centroid values to vertices and edges"""

        # Do protection step
        nvtxRangePush('protect against negative heights')
        self.protect_against_infinitesimal_and_negative_heights()
        nvtxRangePop()

        # Do extrapolation step
        # nvtxRangePush('extrapolate')
        # Choose the correct extension module
        if self.multiprocessor_mode == 1:
            from .sw_domain_openmp_ext import extrapolate_second_order_edge_sw
        elif self.multiprocessor_mode == 2:
            # change over to cuda routines as developed
            #from .sw_domain_simd_ext import extrapolate_second_order_edge_sw
            extrapolate_second_order_edge_sw = self.gpu_interface.extrapolate_second_order_edge_sw_kernel
        else:
            raise Exception('Not implemented')

        nvtxRangePush('extrapolate_second_order_edge_sw')
        extrapolate_second_order_edge_sw(self)
        nvtxRangePop()

    def distribute_to_edges(self):
        """ extrapolate centroid values edges"""

        # Do protection step
        nvtxRangePush('protect_against_infinities')
        self.protect_against_infinitesimal_and_negative_heights()
        nvtxRangePop()

        # Do extrapolation step
        # nvtxRangePush('extrapolate')
        # Choose the correct extension module
        if self.multiprocessor_mode == 1:
            from .sw_domain_openmp_ext import extrapolate_second_order_edge_sw
            extrapolate_second_order_edge_sw(self)
        elif self.multiprocessor_mode == 2:
            # change over to cuda routines as developed
            #from .sw_domain_simd_ext import extrapolate_second_order_edge_sw
            extrapolate_second_order_edge_sw = self.gpu_interface.extrapolate_second_order_edge_sw_kernel
            extrapolate_second_order_edge_sw(self)
        else:
            raise Exception('Not implemented')

        # nvtxRangePop()        

    def distribute_edges_to_vertices(self):
        """Distribute edge values to vertices.
        
        This is a wrapper for the C implementation of the distribution
        from edges to vertices.
        """

        if self.multiprocessor_mode == 1:
            # Using OpenMP extension
            from .sw_domain_openmp_ext import distribute_edges_to_vertices as distribute_edges_to_vertices_ext
        elif self.multiprocessor_mode == 2:
            # Using CUDA extension
            # FIXME SR: Not implemented yet so use OpenMP version
            from .sw_domain_openmp_ext import distribute_edges_to_vertices as distribute_edges_to_vertices_ext
            # distribute_edges_to_vertices_ext = self.gpu_interface.distribute_edges_to_vertices_kernel
        else:
            raise Exception('Not implemented')
        
        # nvtxRangePush('distribute_edges_to_vertices')
        distribute_edges_to_vertices_ext(self)
        # nvtxRangePop()

    def distribute_using_edge_limiter(self):
        """Distribution from centroids to edges specific to the SWW eqn.

        It will ensure that h (w-z) is always non-negative even in the
        presence of steep bed-slopes by taking a weighted average between shallow
        and deep cases.

        In addition, all conserved quantities get distributed as per either a
        constant (order==1) or a piecewise linear function (order==2).


        Precondition:
          All quantities defined at centroids and bed elevation defined at
          vertices.

        Postcondition
          Conserved quantities defined at vertices
        """

        # Remove very thin layers of water
        self.protect_against_infinitesimal_and_negative_heights()

        for name in self.conserved_quantities:
            Q = self.quantities[name]
            if self._order_ == 1:
                Q.extrapolate_first_order()
            elif self._order_ == 2:
                Q.extrapolate_second_order_and_limit_by_edge()
            else:
                raise Exception('Unknown order')

        self.balance_deep_and_shallow()

        # Compute edge values by interpolation
        for name in self.conserved_quantities:
            Q = self.quantities[name]
            Q.interpolate_from_vertices_to_edges()

    def distribute_using_vertex_limiter(self):
        """Distribution from centroids to vertices specific to the SWW equation.

        It will ensure that h (w-z) is always non-negative even in the
        presence of steep bed-slopes by taking a weighted average between shallow
        and deep cases.

        In addition, all conserved quantities get distributed as per either a
        constant (order==1) or a piecewise linear function (order==2).

        FIXME: more explanation about removal of artificial variability etc

        Precondition:
          All quantities defined at centroids and bed elevation defined at
          vertices.

        Postcondition
          Conserved quantities defined at vertices
        """

        # Remove very thin layers of water
        self.protect_against_infinitesimal_and_negative_heights()

        # Extrapolate all conserved quantities
        if self.optimised_gradient_limiter:
            # MH090605 if second order,
            # perform the extrapolation and limiting on
            # all of the conserved quantities

            if (self._order_ == 1):
                for name in self.conserved_quantities:
                    Q = self.quantities[name]
                    Q.extrapolate_first_order()
            elif self._order_ == 2:
                self.extrapolate_second_order_sw()
            else:
                raise Exception('Unknown order')
        else:
            # Old code:
            for name in self.conserved_quantities:
                Q = self.quantities[name]

                if self._order_ == 1:
                    Q.extrapolate_first_order()
                elif self._order_ == 2:
                    Q.extrapolate_second_order_and_limit_by_vertex()
                else:
                    raise Exception('Unknown order')

        # Take bed elevation into account when water heights are small
        self.balance_deep_and_shallow()

        # Compute edge values by interpolation
        for name in self.conserved_quantities:
            Q = self.quantities[name]
            Q.interpolate_from_vertices_to_edges()


    def protect_against_infinitesimal_and_negative_heights(self):
        """ Clean up the stage and momentum values to ensure non-negative heights
        """

        # nvtxRangePush('protect_new')
        # Choose the correct extension module
        if self.multiprocessor_mode == 1:
            from .sw_domain_openmp_ext import protect_new
        elif self.multiprocessor_mode == 2:
            # change over to cuda routines as developed
            # # from .sw_domain_simd_ext import  protect_new
            #from .sw_domain_openmp_ext import protect_new
            protect_new = self.gpu_interface.protect_against_infinitesimal_and_negative_heights_kernel
        else:
            raise Exception('Not implemented')


        mass_error = protect_new(self)
        # nvtxRangePop()

        if mass_error > 0.0 and self.verbose :
            #print('Cumulative mass protection: ' + str(mass_error) + ' m^3 ')
            # From https://stackoverflow.com/questions/22397261/cant-convert-float-object-to-str-implicitly
            print('Cumulative mass protection: {0} m^3'.format(mass_error))


    def balance_deep_and_shallow(self):
        """Compute linear combination between stage as computed by
        gradient-limiters limiting using w, and stage computed by
        gradient-limiters limiting using h (h-limiter).

        The former takes precedence when heights are large compared to the
        bed slope while the latter takes precedence when heights are
        relatively small.  Anything in between is computed as a balanced
        linear combination in order to avoid numerical disturbances which
        would otherwise appear as a result of hard switching between
        modes.

        Wrapper for C implementation
        """

        from .sw_domain_orig_ext import balance_deep_and_shallow \
            as balance_deep_and_shallow_ext

        # Shortcuts
        wc = self.quantities['stage'].centroid_values
        zc = self.quantities['elevation'].centroid_values
        wv = self.quantities['stage'].vertex_values
        zv = self.quantities['elevation'].vertex_values

        # Momentums at centroids
        xmomc = self.quantities['xmomentum'].centroid_values
        ymomc = self.quantities['ymomentum'].centroid_values

        # Momentums at vertices
        xmomv = self.quantities['xmomentum'].vertex_values
        ymomv = self.quantities['ymomentum'].vertex_values

        balance_deep_and_shallow_ext(self,
                                   wc, zc, wv, zv, wc,
                                   xmomc, ymomc, xmomv, ymomv)

        nvtxRangePop()


    def apply_protection_against_isolated_degenerate_timesteps(self):

        if self.protect_against_isolated_degenerate_timesteps is False:
            return

        # FIXME (Ole): Make this configurable
        if num.max(self.max_speed) < 10.0:
            return

        # Setup 10 bins for speed histogram
        from anuga.utilities.numerical_tools import histogram, create_bins

        bins = create_bins(self.max_speed, 10)
        hist = histogram(self.max_speed, bins)

        # Look for characteristic signature
        if len(hist) > 1 and hist[-1] > 0 and \
           hist[4] == hist[5] == hist[6] == hist[7] == hist[8] == 0:
            # Danger of isolated degenerate triangles

            # Find triangles in last bin
            # FIXME - speed up using numeric package
            d = 0
            for i in range(self.number_of_triangles):
                if self.max_speed[i] > bins[-1]:
                    msg = 'Time=%f: Ignoring isolated high ' % self.get_time()
                    msg += 'speed triangle '
                    msg += '#%d of %d with max speed = %f' \
                        % (i, self.number_of_triangles, self.max_speed[i])

                    self.get_quantity('xmomentum').set_values(0.0, indices=[i])
                    self.get_quantity('ymomentum').set_values(0.0, indices=[i])
                    self.max_speed[i] = 0.0
                    d += 1


    def update_conserved_quantities(self):
        """Update vectors of conserved quantities using previously
        computed fluxes and specified forcing functions.
        """

        # nvtxRangePush('update_conserved_quantities')

        timestep = self.timestep


        # Update conserved_quantities
        Elev = self.quantities['elevation']
        Stage = self.quantities['stage']
        Xmom = self.quantities['xmomentum']
        Ymom = self.quantities['ymomentum']

        # FIXME SR: Should pull this together with fix_negative_cells and implemented in 
        # in sw_domain_orig_..._.c
        #Stage.update(timestep)
        #Xmom.update(timestep)
        #Ymom.update(timestep)
        
        assert self.get_using_discontinuous_elevation()

        # Update height based on discontinuous elevation
        if self.multiprocessor_mode == 1:
            
            from .sw_domain_openmp_ext import update_conserved_quantities
            num_negative_ids = update_conserved_quantities(self, timestep)

        elif self.multiprocessor_mode == 2:

            update_conserved_quantities = self.gpu_interface.update_conserved_quantities_kernel
            num_negative_ids = update_conserved_quantities(self, timestep)
        
        else:
            raise Exception('Not implemented')

        if num_negative_ids > 0:
            # FIXME: This only warns the first time -- maybe we should warn whenever loss occurs?
            import warnings
            msg = f'{num_negative_ids} negative cells being set to zero depth, possible loss of conservation. \n' +\
            'Consider using domain.report_water_volume_statistics() to check the extent of the problem'
            warnings.warn(msg)

        # nvtxRangePop()

    def update_other_quantities(self):
        """ There may be a need to calculates some of the other quantities
        based on the new values of conserved quantities
        """

        return


    def update_centroids_of_velocities_and_height(self):
        """Calculate the centroid values of velocities and height based
        on the values of the quantities stage and x and y momentum

        Assumes that stage and momentum are up to date

        Useful for kinematic viscosity calculations
        """

        # For shallow water we need to update height xvelocity and yvelocity

        #Shortcuts
        W  = self.quantities['stage']
        UH = self.quantities['xmomentum']
        VH = self.quantities['ymomentum']
        H  = self.quantities['height']
        Z  = self.quantities['elevation']
        U  = self.quantities['xvelocity']
        V  = self.quantities['yvelocity']

        #print num.min(W.centroid_values)

        # Make sure boundary values of conserved quantites
        # are consistent with value of functions at centroids
        Z.set_boundary_values_from_edges()

        # FIXME(Ole): Why are these not necessary?
        #W.set_boundary_values_from_edges()
        #UH.set_boundary_values_from_edges()
        #VH.set_boundary_values_from_edges()


        #Aliases
        w_C   = W.centroid_values
        z_C   = Z.centroid_values
        uh_C  = UH.centroid_values
        vh_C  = VH.centroid_values
        u_C   = U.centroid_values
        v_C   = V.centroid_values
        h_C   = H.centroid_values

        w_B   = W.boundary_values
        z_B   = Z.boundary_values
        uh_B  = UH.boundary_values
        vh_B  = VH.boundary_values
        u_B   = U.boundary_values
        v_B   = V.boundary_values
        h_B   = H.boundary_values

        h_C[:] = w_C-z_C
        h_C[:] = num.where(h_C >= 0, h_C , 0.0)

        h_B[:] = w_B-z_B
        h_B[:] = num.where(h_B >=0, h_B, 0.0)

        # Update height values
        #H.set_values( num.where(W.centroid_values-Z.centroid_values>=0,
        #                        W.centroid_values-Z.centroid_values, 0.0), location='centroids')
        #H.set_boundary_values( num.where(W.boundary_values-Z.boundary_values>=0,
        #                                 W.boundary_values-Z.boundary_values, 0.0))



        #assert num.min(h_C) >= 0
        #assert num.min(h_B) >= 0


        H0 = 1.0e-8

        #U.set_values(uh_C/(h_C + H0/h_C), location='centroids')
        #V.set_values(vh_C/(h_C + H0/h_C), location='centroids')

        factor = h_C/(h_C*h_C + H0)
        u_C[:]  = uh_C*factor
        v_C[:]  = vh_C*factor

        #U.set_boundary_values(uh_B/(h_B + H0/h_B))
        #V.set_boundary_values(vh_B/(h_B + H0/h_B))

        factor = h_B/(h_B*h_B + H0)
        u_B[:]  = uh_B*factor
        v_B[:]  = vh_B*factor



    def update_centroids_of_momentum_from_velocity(self):
        """Calculate the centroid value of x and y momentum from height and velocities

        Assumes centroids of height and velocities are up to date

        Useful for kinematic viscosity calculations
        """

        # For shallow water we need to update height xvelocity and yvelocity
        #Shortcuts
        UH = self.quantities['xmomentum']
        VH = self.quantities['ymomentum']
        H  = self.quantities['height']
        Z  = self.quantities['elevation']
        U  = self.quantities['xvelocity']
        V  = self.quantities['yvelocity']


        #Arrays
        u_C  = U.centroid_values
        v_C  = V.centroid_values
        uh_C = UH.centroid_values
        vh_C = VH.centroid_values
        h_C  = H.centroid_values

        u_B  = U.boundary_values
        v_B  = V.boundary_values
        uh_B = UH.boundary_values
        vh_B = VH.boundary_values
        h_B  = H.boundary_values

        uh_C[:] = u_C*h_C
        vh_C[:] = v_C*h_C
        #UH.set_values(u_C*h_C , location='centroids')
        #VH.set_values(v_C*h_C , location='centroids')

        self.distribute_to_vertices_and_edges()


    

    def evolve(self,
               yieldstep=None,
               outputstep=None,
               finaltime=None,
               duration=None,
               skip_initial_step=False):
        """Evolve method from Domain class.


        :param float yieldstep: yield every yieldstep time period
        :param float outputstep: Output to sww file every outputstep time period. outputstep should be an integer multiple of yieldstep.
        :param float finaltime: evolve until finaltime (can be a float (secs) or a datetime object)
        :param float duration: evolve for a time of length duration (secs)
        :param  boolean skip_inital_step: Can be used to restart a simulation (not often used).


        If outputstep is None, the output to sww file happens every yieldstep.
        If yieldstep is None then simply evolve to finaltime or for a duration.
        """

        # Call check integrity here rather than from user scripts
        # self.check_integrity()

        from datetime import datetime
        if finaltime is not None:
            if isinstance(finaltime, datetime):
                if finaltime.tzinfo is None:
                    dt = finaltime.replace(tzinfo=self.timezone)
                else:
                    dt = finaltime
                finaltime = dt.timestamp()
            else:
                finaltime = float(finaltime)

        if outputstep is None:
            outputstep = yieldstep

        if yieldstep is None:
            self.output_frequency = 1
        else:
            msg = f'outputstep ({outputstep}) should be an integer multiple of yieldstep ({yieldstep})'
            output_frequency = outputstep/yieldstep
            assert float(output_frequency).is_integer(), msg
            self.output_frequency = int(output_frequency)

        msg = 'Attribute self.beta_w must be in the interval [0, 2]'
        assert 0 <= self.beta_w <= 2.0, msg

        # Initial update of vertex and edge values before any STORAGE
        # and or visualisation.
        # This is done again in the initialisation of the Generic_Domain
        # evolve loop but we do it here to ensure the values are ok for storage.

        self.distribute_to_vertices_and_edges()

        if self.store is True and (self.get_relative_time() == 0.0 or self.evolved_called is False):
            self.initialise_storage()

        
    
        #nvtx marker
        nvtxRangePush('_evolve_base')

        # Call basic machinery from parent class
        for t in self._evolve_base(yieldstep=yieldstep,
                                   finaltime=finaltime, duration=duration,
                                   skip_initial_step=skip_initial_step):


            walltime = time.time()

            #print t , self.get_time()
            # Store model data, e.g. for subsequent visualisation
            if self.store:
                if self.yieldstep_counter%self.output_frequency == 0:
                    self.store_timestep()

            if self.checkpoint:
                save_checkpoint=False
                if self.checkpoint_step == 0:
                    if rank() == 0:
                        if walltime - self.walltime_prev > self.checkpoint_time:

                            save_checkpoint = True
                        for cpu in range(size()):
                            if cpu != rank():
                                send(save_checkpoint, cpu)
                    else:
                        save_checkpoint = receive(0)

                elif self.yieldstep_counter%self.checkpoint_step == 0:
                        save_checkpoint = True

                if save_checkpoint:
                    pickle_name = os.path.join(self.checkpoint_dir,self.get_name())+'_'+str(self.get_time())+'.pickle'
                    pickle.dump(self, open(pickle_name, 'wb'))

                    barrier()
                    self.walltime_prev = time.time()

                    #print 'Stored Checkpoint File '+pickle_name

            # Pass control on to outer loop for more specific actions
            yield(t)

            self.yieldstep_counter += 1

        #nvtx marker
        nvtxRangePop()


    def initialise_storage(self):
        """Create and initialise self.writer object for storing data.
        Also, save x,y and bed elevation
        """

        nvtxRangePush('SWW_file')

        # Initialise writer
        self.writer = SWW_file(self)

        # Store vertices and connectivity
        self.writer.store_connectivity()

        nvtxRangePop()


    def store_timestep(self):
        """Store time dependent quantities and time.

        Precondition:
           self.writer has been initialised
        """

        nvtxRangePush('store_timestep')
        self.writer.store_timestep()
        nvtxRangePop()


    def sww_merge(self,  *args, **kwargs):
        """Dummy function for sequential algorithms where the sww produced is the final products.

        For parallel runs, a similarly named routine in parallel_shallow_water will merge all the
        sub domain sww files into a global sww file

        :param bool verbose: Flag to produce more output
        :param bool delete_old: Flag to delete sub domain sww files after
            creating global sww file

        """

        pass


    def evolve_one_euler_step(self, yieldstep, finaltime):
        """One Euler Time Step
        Q^{n+1} = E(h) Q^n

        Does not assume that centroid values have been extrapolated to
        vertices and edges
        """

        #nvtx marker
        nvtxRangePush('distribute_to_vertices_and_edges')

        # From centroid values calculate edge and vertex values
        self.distribute_to_vertices_and_edges()

        #nvtx marker
        nvtxRangePop()

        #nvtx marker
        nvtxRangePush('update_boundary')
        # Apply boundary conditions
        self.update_boundary()
        #nvtx marker
        nvtxRangePop()

        #nvtx marker
        nvtxRangePush('compute_fluxes')
        # Compute fluxes across each element edge
        self.compute_fluxes()
        #nvtx marker
        nvtxRangePop()

        #nvtx marker
        nvtxRangePush('compute_forcing_terms')
        # Compute forcing terms
        self.compute_forcing_terms()
        #nvtx marker
        nvtxRangePop()

        #nvtx marker
        nvtxRangePush('update_timestep')
        # Update timestep to fit yieldstep and finaltime
        self.update_timestep(yieldstep, finaltime)
        #nvtx marker
        nvtxRangePop()

        #nvtx marker
        nvtxRangePush('compute_flux_update_frequency')
        if self.max_flux_update_frequency != 1:
            # Update flux_update_frequency using the new timestep
            self.compute_flux_update_frequency()
        #nvtx marker
        nvtxRangePop()

        #nvtx marker
        nvtxRangePush('update_conserved_quantities')
        # Update conserved quantities
        self.update_conserved_quantities()
        #nvtx marker
        nvtxRangePop()

    def evolve_one_rk2_step(self, yieldstep, finaltime):
        """One 2nd order RK timestep
        Q^{n+1} = 0.5 Q^n + 0.5 E(h)^2 Q^n

        Does not assume that centroid values have been extrapolated to
        vertices and edges
        """

        # Save initial initial conserved quantities values
        self.backup_conserved_quantities()

        #==========================================
        # First euler step
        #==========================================

        # From centroid values calculate edge and vertex values
        self.distribute_to_vertices_and_edges()

        # Apply boundary conditions
        self.update_boundary()

        # Compute fluxes across each element edge
        self.compute_fluxes()

        # Compute forcing terms
        self.compute_forcing_terms()

        # Update timestep to fit yieldstep and finaltime
        self.update_timestep(yieldstep, finaltime)

        # Update centroid values of conserved quantities
        self.update_conserved_quantities()

        # Update special conditions
        # self.update_special_conditions()

        # Update time
        self.set_relative_time(self.get_relative_time() + self.timestep)

        # Update ghosts
        if self.ghost_layer_width < 4:
            self.update_ghosts()

        # Update vertex and edge values
        self.distribute_to_vertices_and_edges()

        # Update boundary values
        self.update_boundary()

        #=========================================
        # Second Euler step using the same timestep
        # calculated in the first step. Might lead to
        # stability problems but we have not seen any
        # example.
        #=========================================

        # Compute fluxes across each element edge
        self.compute_fluxes()

        # Compute forcing terms
        self.compute_forcing_terms()

        # Update conserved quantities
        self.update_conserved_quantities()

        #========================================
        # Combine initial and final values
        # of conserved quantities and cleanup
        #========================================

        # Combine steps
        self.saxpy_conserved_quantities(0.5, 0.5)


    def evolve_one_rk3_step(self, yieldstep, finaltime):
        """One 3rd order RK timestep
        Q^(1) = 3/4 Q^n + 1/4 E(h)^2 Q^n  (at time t^n + h/2)
        Q^{n+1} = 1/3 Q^n + 2/3 E(h) Q^(1) (at time t^{n+1})

        Does not assume that centroid values have been extrapolated to
        vertices and edges
        """

        # Save initial initial conserved quantities values
        self.backup_conserved_quantities()

        initial_time = self.get_relative_time()

        ######
        # First euler step
        ######

        # From centroid values calculate edge and vertex values
        self.distribute_to_vertices_and_edges()

        # Apply boundary conditions
        self.update_boundary()

        # Compute fluxes across each element edge
        self.compute_fluxes()

        # Compute forcing terms
        self.compute_forcing_terms()

        # Update timestep to fit yieldstep and finaltime
        self.update_timestep(yieldstep, finaltime)

        # Update conserved quantities
        self.update_conserved_quantities()

        # Update special conditions
        # self.update_special_conditions()

        # Update time
        self.set_relative_time(self.relative_time+ self.timestep)

        # Update ghosts
        self.update_ghosts()

        # Update vertex and edge values
        self.distribute_to_vertices_and_edges()

        # Update boundary values
        self.update_boundary()

        ######
        # Second Euler step using the same timestep
        # calculated in the first step. Might lead to
        # stability problems but we have not seen any
        # example.
        ######

        # Compute fluxes across each element edge
        self.compute_fluxes()

        # Compute forcing terms
        self.compute_forcing_terms()

        # Update conserved quantities
        self.update_conserved_quantities()

        ######
        # Combine steps to obtain intermediate
        # solution at time t^n + 0.5 h
        ######

        # Combine steps
        self.saxpy_conserved_quantities(0.25, 0.75)

        # Update special conditions
        # self.update_special_conditions()

        # Set substep time
        self.set_relative_time(initial_time + self.timestep * 0.5)

        # Update ghosts
        self.update_ghosts()

        # Update vertex and edge values
        self.distribute_to_vertices_and_edges()

        # Update boundary values
        self.update_boundary()

        ######
        # Third Euler step
        ######

        # Compute fluxes across each element edge
        self.compute_fluxes()

        # Compute forcing terms
        self.compute_forcing_terms()

        # Update conserved quantities
        self.update_conserved_quantities()

        #=======================================
        # Combine final and initial values
        # and cleanup
        #=======================================
        
        # self.saxpy_conserved_quantities(2.0/3.0, 1.0/3.0)
        # This caused a roundoff error that created negative water heights

        # So do this instead!
        self.saxpy_conserved_quantities(2.0, 1.0, 3.0)


        # Set new time
        self.set_relative_time(initial_time + self.timestep)


    def backup_conserved_quantities(self):

        # Backup conserved_quantities centroid values
        if self.multiprocessor_mode == 1:
            from anuga.shallow_water.sw_domain_openmp_ext import backup_conserved_quantities
            backup_conserved_quantities(self)
        else:
            for name in self.conserved_quantities:
                Q = self.quantities[name]
                Q.backup_centroid_values()

    def saxpy_conserved_quantities(self, a, b, c=None):

        # saxpy conserved_quantities centroid values with backup values
        if self.multiprocessor_mode == 1:
            if c is None:
                c = 1.0
            from anuga.shallow_water.sw_domain_openmp_ext import saxpy_conserved_quantities
            saxpy_conserved_quantities(self, a, b, c)
        else:
            for name in self.conserved_quantities:
                Q = self.quantities[name]
                Q.saxpy_centroid_values(a, b)
                if c is not None:
                    Q.centroid_values[:] = Q.centroid_values / c

    def timestepping_statistics(self,
                                track_speeds=False,
                                triangle_id=None,
                                relative_time=False,
                                time_unit='sec',
                                datetime=False):
        """Return string with time stepping statistics for printing or logging

        :param time_units: 'sec', 'min', 'hr', 'day'
        :param bool datetime: flag to use timestamp or datetime
        :param track_speed: Optional boolean keyword track_speeds decides whether
                            to report location of smallest timestep as well as a
                            histogram and percentile report.
        :param bool relative_time: Flag to report relative time instead of absolute time
        :param int triangle_id: Can be used to specify a particular
                            triangle rather than the one with the largest speed.
        """

        from anuga.config import epsilon, g

        # Call basic machinery from parent class
        msg = Generic_Domain.timestepping_statistics(self,
                                                     track_speeds=track_speeds,
                                                     triangle_id=triangle_id,
                                                     relative_time=relative_time,
                                                     time_unit=time_unit,
                                                     datetime=datetime)

        if track_speeds is True:
            # qwidth determines the text field used for quantities
            qwidth = self.qwidth

            # Selected triangle
            k = self.k

            # Report some derived quantities at vertices, edges and centroid
            # specific to the shallow water wave equation
            z = self.quantities['elevation']
            w = self.quantities['stage']

            Vw = w.get_values(location='vertices', indices=[k])[0]
            Ew = w.get_values(location='edges', indices=[k])[0]
            Cw = w.get_values(location='centroids', indices=[k])

            Vz = z.get_values(location='vertices', indices=[k])[0]
            Ez = z.get_values(location='edges', indices=[k])[0]
            Cz = z.get_values(location='centroids', indices=[k])

            name = 'depth'
            Vh = Vw-Vz
            Eh = Ew-Ez
            Ch = Cw-Cz

            message  = '    %s: vertex_values =  %.4f,\t %.4f,\t %.4f\n'\
                 % (name.ljust(qwidth), Vh[0], Vh[1], Vh[2])

            message += '    %s: edge_values =    %.4f,\t %.4f,\t %.4f\n'\
                 % (name.ljust(qwidth), Eh[0], Eh[1], Eh[2])

            message += '    %s: centroid_value = %.4f\n'\
                 % (name.ljust(qwidth), Ch[0])

            msg += message

            uh = self.quantities['xmomentum']
            vh = self.quantities['ymomentum']

            Vuh = uh.get_values(location='vertices', indices=[k])[0]
            Euh = uh.get_values(location='edges', indices=[k])[0]
            Cuh = uh.get_values(location='centroids', indices=[k])

            Vvh = vh.get_values(location='vertices', indices=[k])[0]
            Evh = vh.get_values(location='edges', indices=[k])[0]
            Cvh = vh.get_values(location='centroids', indices=[k])

            # Speeds in each direction
            Vu = Vuh/(Vh + epsilon)
            Eu = Euh/(Eh + epsilon)
            Cu = Cuh/(Ch + epsilon)
            name = 'U'
            message  = '    %s: vertex_values =  %.4f,\t %.4f,\t %.4f\n' \
                 % (name.ljust(qwidth), Vu[0], Vu[1], Vu[2])

            message += '    %s: edge_values =    %.4f,\t %.4f,\t %.4f\n' \
                 % (name.ljust(qwidth), Eu[0], Eu[1], Eu[2])

            message += '    %s: centroid_value = %.4f\n' \
                 % (name.ljust(qwidth), Cu[0])

            msg += message

            Vv = Vvh/(Vh + epsilon)
            Ev = Evh/(Eh + epsilon)
            Cv = Cvh/(Ch + epsilon)
            name = 'V'
            message  = '    %s: vertex_values =  %.4f,\t %.4f,\t %.4f\n' \
                 % (name.ljust(qwidth), Vv[0], Vv[1], Vv[2])

            message += '    %s: edge_values =    %.4f,\t %.4f,\t %.4f\n' \
                 % (name.ljust(qwidth), Ev[0], Ev[1], Ev[2])

            message += '    %s: centroid_value = %.4f\n'\
                 %(name.ljust(qwidth), Cv[0])

            msg += message

            # Froude number in each direction
            name = 'Froude (x)'
            Vfx = Vu/(num.sqrt(g*Vh + epsilon))
            Efx = Eu/(num.sqrt(g*Eh + epsilon))
            Cfx = Cu/(num.sqrt(g*Ch + epsilon))

            message  = '    %s: vertex_values =  %.4f,\t %.4f,\t %.4f\n'\
                 % (name.ljust(qwidth), Vfx[0], Vfx[1], Vfx[2])

            message += '    %s: edge_values =    %.4f,\t %.4f,\t %.4f\n'\
                 % (name.ljust(qwidth), Efx[0], Efx[1], Efx[2])

            message += '    %s: centroid_value = %.4f\n'\
                 % (name.ljust(qwidth), Cfx[0])

            msg += message

            name = 'Froude (y)'
            Vfy = Vv/(num.sqrt(g*Vh + epsilon))
            Efy = Ev/(num.sqrt(g*Eh + epsilon))
            Cfy = Cv/(num.sqrt(g*Ch + epsilon))

            message  = '    %s: vertex_values =  %.4f,\t %.4f,\t %.4f\n'\
                 % (name.ljust(qwidth), Vfy[0], Vfy[1], Vfy[2])

            message += '    %s: edge_values =    %.4f,\t %.4f,\t %.4f\n'\
                 % (name.ljust(qwidth), Efy[0], Efy[1], Efy[2])

            message += '    %s: centroid_value = %.4f\n'\
                 % (name.ljust(qwidth), Cfy[0])

            msg += message

        return msg

    def print_timestepping_statistics(self, *args, **kwargs):
        """Print time stepping statistics

        :param time_units: 'sec', 'min', 'hr', 'day'
        :param bool datetime: flag to use timestamp or datetime
        :param track_speed: Optional boolean keyword track_speeds decides whether
                            to report location of smallest timestep as well as a
                            histogram and percentile report.
        :param bool relative_time: Flag to report relative time instead of absolute time
        :param int triangle_id: Can be used to specify a particular
                            triangle rather than the one with the largest speed.
        """

        msg = self.timestepping_statistics(*args, **kwargs)

        print(msg)


    def compute_boundary_flows(self):
        """Compute boundary flows at current timestep.

        Quantities computed are:
           Total inflow across boundary
           Total outflow across boundary
           Flow across each tagged boundary segment

        These calculations are only approximate since they don't use the
        flux calculation used in evolve

        See get_boundary_flux_integral for an exact computation
        """

        # Run through boundary array and compute for each segment
        # the normal momentum ((uh, vh) dot normal) times segment length.
        # Based on sign accumulate this into boundary_inflow and
        # boundary_outflow.

        # Compute flows along boundary

        uh = self.get_quantity('xmomentum').get_values(location='edges')
        vh = self.get_quantity('ymomentum').get_values(location='edges')

        # Loop through edges that lie on the boundary and calculate
        # flows
        boundary_flows = {}
        total_boundary_inflow = 0.0
        total_boundary_outflow = 0.0
        for vol_id, edge_id in self.boundary:
            # Compute normal flow across edge. Since normal vector points
            # away from triangle, a positive sign means that water
            # flows *out* from this triangle.

            momentum = [uh[vol_id, edge_id], vh[vol_id, edge_id]]
            normal = self.mesh.get_normal(vol_id, edge_id)
            length = self.mesh.get_edgelength(vol_id, edge_id)
            normal_flow = num.dot(momentum, normal)*length

            # Reverse sign so that + is taken to mean inflow
            # and - means outflow. This is more intuitive.
            edge_flow = -normal_flow

            # Tally up inflows and outflows separately
            if edge_flow > 0:
                # Flow is inflow
                total_boundary_inflow += edge_flow
            else:
                # Flow is outflow
                total_boundary_outflow += edge_flow

            # Tally up flows by boundary tag
            tag = self.boundary[(vol_id, edge_id)]

            if tag not in boundary_flows:
                boundary_flows[tag] = 0.0
            boundary_flows[tag] += edge_flow


        return boundary_flows, total_boundary_inflow, total_boundary_outflow


    def compute_total_volume(self):
        """
        Compute total volume (m^3) of water in entire domain

        """

        return self.get_water_volume()


    def volumetric_balance_statistics(self):
        """Create volumetric balance report suitable for printing or logging.
        """

        (boundary_flows, total_boundary_inflow,
         total_boundary_outflow) = self.compute_boundary_flows()

        message = '---------------------------\n'
        message += 'Volumetric balance report:\n'
        message += 'Note: Boundary fluxes are not exact\n'
        message += 'See get_boundary_flux_integral for exact computation\n'
        message += '--------------------------\n'
        message += 'Total boundary inflow [m^3/s]: %.2f\n' % total_boundary_inflow
        message += 'Total boundary outflow [m^3/s]: %.2f\n' % total_boundary_outflow
        message += 'Net boundary flow by tags [m^3/s]\n'
        for tag in boundary_flows:
            message += '    %s [m^3/s]: %.2f\n' % (tag, boundary_flows[tag])

        message += 'Total net boundary flow [m^3/s]: %.2f\n' % \
                    (total_boundary_inflow + total_boundary_outflow)
        message += 'Total volume in domain [m^3]: %.2f\n' % \
                    self.compute_total_volume()

        # The go through explicit forcing update and record the rate of change
        # for stage and
        # record into forcing_inflow and forcing_outflow. Finally compute
        # integral of depth to obtain total volume of domain.

        # FIXME(Ole): This part is not yet done.

        return message

    def print_volumetric_balance_statistics(self):

        print (self.volumetric_balance_statistics())

    def compute_flux_update_frequency(self):
        """
            Update the 'flux_update_frequency' and 'update_extrapolate' variables
            Used to control updating of fluxes / extrapolation for 'local-time-stepping'
        """

        nvtxRangePush('compute_flux_update_frequency')
        # Choose the correct extension module
        if self.multiprocessor_mode == 1:
            from .sw_domain_openmp_ext import compute_flux_update_frequency
        elif self.multiprocessor_mode == 2:
            # change over to cuda routines as developed
            from .sw_domain_openmp_ext import compute_flux_update_frequency
        else:
            raise Exception('Not implemented')


        compute_flux_update_frequency(self, self.timestep)

        nvtxRangePop()

    def report_water_volume_statistics(self, verbose=True, returnStats=False):
        """
        Compute the volume, boundary flux integral, fractional step volume integral, and their difference

        If verbose, print a summary
        If returnStats, return a list with the volume statistics
        """
        from anuga import myid

        if(self.compute_fluxes_method != 'DE'):
            if(myid == 0):
                print('Water_volume_statistics only supported for DE algorithm ')
            return

        # Compute the volume
        Vol = self.get_water_volume()

        # Compute the boundary flux integral
        fluxIntegral=self.get_boundary_flux_integral()
        fracIntegral=self.get_fractional_step_volume_integral()

        if(verbose and myid==0):
            print(' ')
            print('    Volume V is:', Vol)
            print('    Boundary Flux integral BF: ', fluxIntegral)
            print('    (rate + inlet) Fractional Step volume integral FS: ', fracIntegral)
            print('    V - BF - FS - InitialVolume :',  Vol- fluxIntegral -fracIntegral - self.volume_history[0])
            print(' ')

        if returnStats:
            return [Vol, fluxIntegral, fracIntegral]
        else:
            return

    def report_cells_with_small_local_timestep(self, threshold_depth=None):
        """
        Convenience function to print the locations of cells
        with a small local timestep.

        Computations are at cell centroids

        Useful in models
        with complex meshes, to find ways to speed up the model
        """

        from anuga.parallel import myid, numprocs
        from anuga.config import g, epsilon

        if threshold_depth is None:
            threshold_depth=self.minimum_allowed_height

        uh = self.quantities['xmomentum'].centroid_values
        vh = self.quantities['ymomentum'].centroid_values
        d =  self.quantities['stage'].centroid_values - self.quantities['elevation'].centroid_values
        d = num.maximum(d, threshold_depth)
        v = ((uh)**2 + (vh)**2)**0.5/d
        v = v * (d > threshold_depth)

        for i in range(numprocs):
            if myid == i:
                print('    Processor ', myid)
                gravSpeed = (g * d)**0.5
                waveSpeed = abs(v) + gravSpeed
                localTS = self.radii / num.maximum(waveSpeed, epsilon)
                controlling_pt_ind = localTS.argmin()
                print('    * Smallest LocalTS is: ', localTS[controlling_pt_ind])
                print('     -- Location: ', round(self.centroid_coordinates[controlling_pt_ind,0]+self.geo_reference.xllcorner,2),\
                                        round(self.centroid_coordinates[controlling_pt_ind,1]+self.geo_reference.yllcorner,2))
                print('     -+ Speed: ', v[controlling_pt_ind])
                print('     -* Gravity_wave_speed', gravSpeed[controlling_pt_ind])
                print(' ')

            barrier()
        return

# =======================================================================
# PETE: NEW METHODS FOR FOR PARALLEL STRUCTURES. Note that we assume the
# first "number_of_full_[nodes|triangles]" are full [nodes|triangles]
# For full triangles it is possible to enquire self.tri_full_flag == True
# =======================================================================

    def get_number_of_full_triangles(self, *args, **kwargs):
        return self.number_of_full_triangles

    def get_full_centroid_coordinates(self, *args, **kwargs):
        C = self.mesh.get_centroid_coordinates(*args, **kwargs)
        return C[:self.number_of_full_triangles, :]

    def get_full_vertex_coordinates(self, *args, **kwargs):
        V = self.mesh.get_vertex_coordinates(*args, **kwargs)
        return V[:3*self.number_of_full_triangles,:]

    def get_full_triangles(self, *args, **kwargs):
        T = self.mesh.get_triangles(*args, **kwargs)
        return T[:self.number_of_full_triangles,:]

    def get_full_nodes(self, *args, **kwargs):
        N = self.mesh.get_nodes(*args, **kwargs)
        return N[:self.number_of_full_nodes,:]

    def get_tri_map(self):
        return self.tri_map

    def get_inv_tri_map(self):
        return self.inv_tri_map

# ==============================================================================
# Multiprocessor Mode (1=openmp, 2=cuda (in development))
# ==============================================================================

    def set_multiprocessor_mode(self, multiprocessor_mode= 0):
        """
        Set multiprocessor mode 
         1. openmp (in development)
         2. cuda (in development)
        """

        if multiprocessor_mode not in [1,2]:
            raise ValueError('Invalid multiprocessor mode. Must be one of [1,2]')

        self.multiprocessor_mode = multiprocessor_mode

        if self.multiprocessor_mode == 2:
            self.set_gpu_interface()

    def get_multiprocessor_mode(self):
        """
        Get multiprocessor mode 
        
        1. openmp (in development)
        2. cuda (in development)
        """
        return self.multiprocessor_mode 

    def set_omp_num_threads(self, omp_num_threads=None):
        """
        Set the number of OpenMP threads to use for parallel processing.
        If OMP_NUM_THREADS is not set, this will set it to the specified 
        omp_num_threads value.
        By default omp_num_threads is set to 1, other , it will use the default setting.
        """

        import os
        if omp_num_threads is None:
            # Use the default setting
            omp_num_threads = os.environ.get('OMP_NUM_THREADS', None)
            #print(f'Using OMP_NUM_THREADS from environment: {omp_num_threads}')


        if omp_num_threads is None:
            omp_num_threads = 1  # Default to 1 if not set

        try:
            omp_num_threads = int(omp_num_threads)
        except ValueError:
            raise ValueError('OMP_NUM_THREADS must be an integer')            

        # Set the number of OpenMP threads
        self.omp_num_threads = omp_num_threads
        from .sw_domain_openmp_ext import set_omp_num_threads
        set_omp_num_threads(omp_num_threads)
        
        print(f'Setting omp_num_threads to {omp_num_threads}')


    def set_gpu_interface(self):

        if self.multiprocessor_mode == 2 and self.gpu_interface is None:

            # first check that cupy is available
            try:
                import cupy as cp
                test_cupy_array = cp.array([1,2,3])

            except:
                print('+==============================================================================+')
                print('|                                                                              |')
                print('| WARNING: cupy or gpu not available, so falling back to multiprocessor_mode 1 |')
                print('|                                                                              |')
                print('+==============================================================================+')
                self.set_multiprocessor_mode(1)
                return

            from .sw_domain_cuda import GPU_interface
            self.gpu_interface = GPU_interface(self)
            self.gpu_interface.allocate_gpu_arrays()
            self.gpu_interface.compile_gpu_kernels()
           
        

################################################################################
# End of class Shallow Water Domain
################################################################################



################################################################################
# Module functions for gradient limiting
################################################################################


def distribute_using_vertex_limiter(domain):
    """Distribution from centroids to vertices specific to the SWW equation.

    It will ensure that h (w-z) is always non-negative even in the
    presence of steep bed-slopes by taking a weighted average between shallow
    and deep cases.

    In addition, all conserved quantities get distributed as per either a
    constant (order==1) or a piecewise linear function (order==2).

    FIXME: more explanation about removal of artificial variability etc

    Precondition:
      All quantities defined at centroids and bed elevation defined at
      vertices.

    Postcondition
      Conserved quantities defined at vertices
    """

    # Remove very thin layers of water
    domain.protect_against_infinitesimal_and_negative_heights()

    # Extrapolate all conserved quantities
    if domain.optimised_gradient_limiter:
        # MH090605 if second order,
        # perform the extrapolation and limiting on
        # all of the conserved quantities

        if domain._order_ == 1:
            for name in domain.conserved_quantities:
                Q = domain.quantities[name]
                Q.extrapolate_first_order()
        elif domain._order_ == 2:
            domain.extrapolate_second_order_sw()
        else:
            raise Exception('Unknown order')
    else:
        # Old code:
        for name in domain.conserved_quantities:
            Q = domain.quantities[name]

            if domain._order_ == 1:
                Q.extrapolate_first_order()
            elif domain._order_ == 2:
                Q.extrapolate_second_order_and_limit_by_vertex()
            else:
                raise Exception('Unknown order')

    # Take bed elevation into account when water heights are small
    domain.balance_deep_and_shallow()

    # Compute edge values by interpolation
    for name in domain.conserved_quantities:
        Q = domain.quantities[name]
        Q.interpolate_from_vertices_to_edges()



# FIXME (Ole): Does this do anything?
def my_update_special_conditions(domain):

    pass



if __name__ == "__main__":
    pass
