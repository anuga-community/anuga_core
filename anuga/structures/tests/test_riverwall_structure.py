#!/usr/bin/env python


import unittest
import os

#from anuga.structures.riverwall import Boyd_box_operator
#from anuga.structures.riverwall import boyd_box_function

#from anuga.abstract_2d_finite_volumes.mesh_factory import rectangular_cross
import anuga
import numpy
from anuga.shallow_water.shallow_water_domain import Domain
from anuga.utilities import plot_utils as util
from anuga.config import g

## Generic data for scenario

verbose = False
boundaryPolygon=[ [0., 0.], [0., 100.], [100.0, 100.0], [100.0, 0.0]]
wallLoc=50.
# The boundary polygon + riverwall breaks the mesh into multiple regions
# Must define the resolution in these areas with an xy point + maximum area
# Otherwise triangle.c gets confused
regionPtAreas=[ [99., 99., 20.0*20.0*0.5],
                [1., 1., 20.0*20.0*0.5]]

class Test_riverwall_structure(unittest.TestCase):
    """
	Test the riverwall structure
    """

    def setUp(self):
        pass

    def tearDown(self):
        try:
            os.remove('test_riverwall.sww')
        except OSError:
            pass

        try:
            os.remove('testRiverwall.msh')
        except OSError:
            pass

    def create_domain_DE0(self, wallHeight, InitialOceanStage, InitialLandStage, riverWall=None, riverWall_Par=None):
        # Riverwall = list of lists, each with a set of x,y,z (and optional QFactor) values

        if(riverWall is None):
            riverWall={ 'centralWall':
                            [ [wallLoc, 0.0, wallHeight],
                              [wallLoc, 100.0, wallHeight]]
                      }
        if(riverWall_Par is None):
            riverWall_Par={'centralWall':{'Qfactor':1.0}}
        # Make the domain
        anuga.create_pmesh_from_regions(boundaryPolygon,
                                 boundary_tags={'left': [0],
                                                'top': [1],
                                                'right': [2],
                                                'bottom': [3]},
                                   maximum_triangle_area = 200.,
                                   minimum_triangle_angle = 28.0,
                                   filename = 'testRiverwall.msh',
                                   interior_regions =[ ], #[ [higherResPolygon, 1.*1.*0.5],
                                                          #  [midResPolygon, 3.0*3.0*0.5]],
                                   breaklines=list(riverWall.values()),
                                   use_cache=False,
                                   verbose=verbose,
                                   regionPtArea=regionPtAreas)

        domain=anuga.create_domain_from_file('testRiverwall.msh')

        # 05/05/2014 -- riverwalls only work with DE0 and DE1
        domain.set_flow_algorithm('DE0')
        domain.set_name('test_riverwall')

        domain.set_store_vertices_uniquely()

        def topography(x,y):
            return -x/150.

        def stagefun(x,y):
            stg=InitialOceanStage*(x>=50.) + InitialLandStage*(x<50.)
            return stg

        # NOTE: Setting quantities at centroids is important for exactness of tests
        domain.set_quantity('elevation',topography,location='centroids')
        domain.set_quantity('friction',0.03)
        domain.set_quantity('stage', stagefun,location='centroids')

        domain.riverwallData.create_riverwalls(riverWall,riverWall_Par,verbose=verbose)

        # Boundary conditions
        Br=anuga.Reflective_boundary(domain)
        domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom':Br})

        return domain

    def create_domain_DE1(self, wallHeight, InitialOceanStage, InitialLandStage):
        # Riverwall = list of lists, each with a set of x,y,z (and optional QFactor) values
        riverWall={ 'centralWall':
                        [ [wallLoc, 0.0, wallHeight],
                          [wallLoc, 100.0, wallHeight]]
                  }
        riverWall_Par={'centralWall':{'Qfactor':1.0}}
        # Make the domain
        anuga.create_pmesh_from_regions(boundaryPolygon,
                                 boundary_tags={'left': [0],
                                                'top': [1],
                                                'right': [2],
                                                'bottom': [3]},
                                   maximum_triangle_area = 200.,
                                   minimum_triangle_angle = 28.0,
                                   filename = 'testRiverwall.msh',
                                   interior_regions =[ ], #[ [higherResPolygon, 1.*1.*0.5],
                                                          #  [midResPolygon, 3.0*3.0*0.5]],
                                   breaklines=list(riverWall.values()),
                                   use_cache=False,
                                   verbose=verbose,
                                   regionPtArea=regionPtAreas)

        domain=anuga.create_domain_from_file('testRiverwall.msh')

        # 05/05/2014 -- riverwalls only work with DE0 and DE1
        domain.set_flow_algorithm('DE1')
        domain.set_name('test_riverwall')

        domain.set_store_vertices_uniquely()

        def topography(x,y):
            return -x/150.

        def stagefun(x,y):
            stg=InitialOceanStage*(x>=50.) + InitialLandStage*(x<50.)
            return stg

        # NOTE: Setting quantities at centroids is important for exactness of tests
        domain.set_quantity('elevation',topography,location='centroids')
        domain.set_quantity('friction',0.03)
        domain.set_quantity('stage', stagefun,location='centroids')

        domain.riverwallData.create_riverwalls(riverWall,riverWall_Par,verbose=verbose)

        # Boundary conditions
        Br=anuga.Reflective_boundary(domain)
        domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom':Br})

        return domain

    def test_noflux_riverwall_DE1(self):
        """test_noflux_riverwall

            Tests that the riverwall blocks water when the stage is < wall height

        """
        wallHeight=-0.2
        InitialOceanStage=-0.3
        InitialLandStage=-999999.

        domain=self.create_domain_DE1(wallHeight,InitialOceanStage, InitialLandStage)

        # Run the model for a few seconds, and check that no water has flowed past the riverwall
        for t in domain.evolve(yieldstep=10.0,finaltime=10.0):
            if(verbose):
                print(domain.timestepping_statistics())

        # Indices landward of the riverwall
        landInds=(domain.centroid_coordinates[:,0]<50.).nonzero()[0]
        # Compute volume of water landward of riverwall
        landVol=domain.quantities['height'].centroid_values[landInds]*domain.areas[landInds]
        landVol=landVol.sum()

        if(verbose):
            print('Land Vol: ', landVol, 'theoretical vol: ', 0.)

        assert numpy.allclose(landVol,0., atol=1.0e-12)

    def test_simpleflux_riverwall_DE1(self):
        """test_simpleflux_riverwall

            Tests that the riverwall flux (when dry on one edge) is
            2/3*H*sqrt(2/3*g*H)

        """
        wallHeight=2.
        InitialOceanStage=2.50
        InitialLandStage=-999999.

        domain=self.create_domain_DE1(wallHeight,InitialOceanStage, InitialLandStage)

        # Run the model for a few fractions of a second
        # Any longer, and the evolution of stage starts causing
        # significant changes to the flow
        yst=1.0e-04
        ft=1.0e-03
        for t in domain.evolve(yieldstep=yst,finaltime=ft):
            if(verbose):
                print(domain.timestepping_statistics())

        # Compare with theoretical result
        L= 100. # Length of riverwall
        H=InitialOceanStage-wallHeight # Upstream head
        dt=ft
        theoretical_flux_vol=dt*L*2./3.*H*(2./3.*g*H)**0.5

        # Indices landward of the riverwall
        landInds=(domain.centroid_coordinates[:,0]<50.).nonzero()[0]
        # Compute volume of water landward of riverwall
        landVol=domain.quantities['height'].centroid_values[landInds]*domain.areas[landInds]
        landVol=landVol.sum()

        if(verbose):
            print('Land Vol: ', landVol, 'theoretical vol: ', theoretical_flux_vol)

        assert numpy.allclose(landVol,theoretical_flux_vol, rtol=1.0e-03)

    def test_simpleflux_riverwall_with_Qfactor_DE1(self):
        """test_simpleflux_riverwall with Qfactor != 1.0

            Tests that the riverwall flux (when dry on one edge) is
            2/3*H*sqrt(2/3*g*H)*Qfactor

        """
        wallHeight=2.
        InitialOceanStage=2.50
        InitialLandStage=-999999.

        domain=self.create_domain_DE1(wallHeight,InitialOceanStage, InitialLandStage)
        # Redefine the riverwall, with 2x the discharge Qfactor
        riverWall={ 'centralWall':
                        [ [wallLoc, 0.0, wallHeight],
                          [wallLoc, 100.0, wallHeight]]
                  }
        Qmult=2.0
        riverWall_Par={'centralWall':{'Qfactor':Qmult}}
        domain.riverwallData.create_riverwalls(riverWall,riverWall_Par,verbose=verbose)

        #import pdb
        #pdb.set_trace()

        # Run the model for a few fractions of a second
        # Any longer, and the evolution of stage starts causing
        # significant changes to the flow
        yst=1.0e-04
        ft=1.0e-03
        for t in domain.evolve(yieldstep=yst,finaltime=ft):
            if(verbose):
                print(domain.timestepping_statistics())

        # Compare with theoretical result
        L= 100. # Length of riverwall
        H=InitialOceanStage-wallHeight # Upstream head
        dt=ft
        theoretical_flux_vol=Qmult*dt*L*2./3.*H*(2./3.*g*H)**0.5

        # Indices landward of the riverwall
        landInds=(domain.centroid_coordinates[:,0]<50.).nonzero()[0]
        # Compute volume of water landward of riverwall
        landVol=domain.quantities['height'].centroid_values[landInds]*domain.areas[landInds]
        landVol=landVol.sum()

        #import pdb
        #pdb.set_trace()

        if(verbose):
            print('Land Vol: ', landVol, 'theoretical vol: ', theoretical_flux_vol)

        assert numpy.allclose(landVol,theoretical_flux_vol, rtol=1.0e-03)


    def test_submergeflux_riverwall_DE1(self):
        """test_submergedflux_riverwall

            Tests that the riverwall flux is
            Q1*(1-Q2/Q1)**(0.385)

            Where Q1, Q2 =2/3*H*sqrt(2/3*g*H)
             with H computed from the smallest (Q2) / largest (Q1) head side
        """
        wallHeight=20.
        InitialOceanStage=20.50
        InitialLandStage=20.44

        domain=self.create_domain_DE1(wallHeight,InitialOceanStage, InitialLandStage)

        # Indices landward of the riverwall
        landInds=(domain.centroid_coordinates[:,0]<50.).nonzero()[0]
        InitialHeight=(domain.quantities['stage'].centroid_values[landInds]-domain.quantities['elevation'].centroid_values[landInds])
        InitialLandVol=InitialHeight*(InitialHeight>0.)*domain.areas[landInds]
        InitialLandVol=InitialLandVol.sum()

        # Run the model for a few fractions of a second
        # Any longer, and the evolution of stage starts causing
        # significant changes to the flow
        yst=1.0e-04
        ft=1.0e-03
        for t in domain.evolve(yieldstep=yst,finaltime=ft):
            if(verbose):
                print(domain.timestepping_statistics())

        # Compare with theoretical result
        L= 100. # Length of riverwall
        dt=ft
        H1=max(InitialOceanStage-wallHeight,0.) # Upstream head
        H2=max(InitialLandStage-wallHeight,0.) # Downstream head

        Q1=2./3.*H1*(2./3.*g*H1)**0.5
        Q2=2./3.*H2*(2./3.*g*H2)**0.5

        theoretical_flux_vol=dt*L*Q1*(1.-Q2/Q1)**0.385

        # Compute volume of water landward of riverwall
        FinalLandVol=domain.quantities['height'].centroid_values[landInds]*domain.areas[landInds]
        FinalLandVol=FinalLandVol.sum()

        landVol=FinalLandVol-InitialLandVol

        #import pdb
        #pdb.set_trace()

        if(verbose):
            print('Land Vol: ', landVol, 'theoretical vol: ', theoretical_flux_vol)

        assert numpy.allclose(landVol,theoretical_flux_vol, rtol=1.0e-03)

    def test_noflux_riverwall_DE0(self):
        """test_noflux_riverwall

            Tests that the riverwall blocks water when the stage is < wall height

        """
        wallHeight=-0.2
        InitialOceanStage=-0.3
        InitialLandStage=-999999.

        domain=self.create_domain_DE0(wallHeight,InitialOceanStage, InitialLandStage)

        # Run the model for a few seconds, and check that no water has flowed past the riverwall
        for t in domain.evolve(yieldstep=10.0,finaltime=10.0):
            if(verbose):
                print(domain.timestepping_statistics())

        # Indices landward of the riverwall
        landInds=(domain.centroid_coordinates[:,0]<50.).nonzero()[0]
        # Compute volume of water landward of riverwall
        landVol=domain.quantities['height'].centroid_values[landInds]*domain.areas[landInds]
        landVol=landVol.sum()

        if(verbose):
            print('Land Vol: ', landVol, 'theoretical vol: ', 0.)

        assert numpy.allclose(landVol,0., atol=1.0e-12)

    def test_simpleflux_riverwall_DE0(self):
        """test_simpleflux_riverwall

            Tests that the riverwall flux (when dry on one edge) is
            2/3*H*sqrt(2/3*g*H)

        """
        wallHeight=2.
        InitialOceanStage=2.50
        InitialLandStage=-999999.

        domain=self.create_domain_DE0(wallHeight,InitialOceanStage, InitialLandStage)

        # Run the model for a few fractions of a second
        # Any longer, and the evolution of stage starts causing
        # significant changes to the flow
        yst=1.0e-04
        ft=1.0e-03
        for t in domain.evolve(yieldstep=yst,finaltime=ft):
            if(verbose):
                print(domain.timestepping_statistics())

        # Compare with theoretical result
        L= 100. # Length of riverwall
        H=InitialOceanStage-wallHeight # Upstream head
        dt=ft
        theoretical_flux_vol=dt*L*2./3.*H*(2./3.*g*H)**0.5

        # Indices landward of the riverwall
        landInds=(domain.centroid_coordinates[:,0]<50.).nonzero()[0]
        # Compute volume of water landward of riverwall
        landVol=domain.quantities['height'].centroid_values[landInds]*domain.areas[landInds]
        landVol=landVol.sum()

        if(verbose):
            print('Land Vol: ', landVol, 'theoretical vol: ', theoretical_flux_vol)

        assert numpy.allclose(landVol,theoretical_flux_vol, rtol=1.0e-03)

    def test_simpleflux_riverwall_with_Qfactor_DE0(self):
        """test_simpleflux_riverwall with Qfactor != 1.0

            Tests that the riverwall flux (when dry on one edge) is
            2/3*H*sqrt(2/3*g*H)*Qfactor

        """
        wallHeight=2.
        InitialOceanStage=2.50
        InitialLandStage=-999999.

        domain=self.create_domain_DE0(wallHeight,InitialOceanStage, InitialLandStage)
        # Redefine the riverwall, with 2x the discharge Qfactor
        riverWall={ 'centralWall':
                        [ [wallLoc, 0.0, wallHeight],
                          [wallLoc, 100.0, wallHeight]]
                  }
        Qmult=2.0
        riverWall_Par={'centralWall':{'Qfactor':Qmult}}
        domain.riverwallData.create_riverwalls(riverWall,riverWall_Par,verbose=verbose)

        #import pdb
        #pdb.set_trace()

        # Run the model for a few fractions of a second
        # Any longer, and the evolution of stage starts causing
        # significant changes to the flow
        yst=1.0e-04
        ft=1.0e-03
        for t in domain.evolve(yieldstep=yst,finaltime=ft):
            if(verbose):
                print(domain.timestepping_statistics())

        # Compare with theoretical result
        L= 100. # Length of riverwall
        H=InitialOceanStage-wallHeight # Upstream head
        dt=ft
        theoretical_flux_vol=Qmult*dt*L*2./3.*H*(2./3.*g*H)**0.5

        # Indices landward of the riverwall
        landInds=(domain.centroid_coordinates[:,0]<50.).nonzero()[0]
        # Compute volume of water landward of riverwall
        landVol=domain.quantities['height'].centroid_values[landInds]*domain.areas[landInds]
        landVol=landVol.sum()

        #import pdb
        #pdb.set_trace()

        if(verbose):
            print('Land Vol: ', landVol, 'theoretical vol: ', theoretical_flux_vol)

        assert numpy.allclose(landVol,theoretical_flux_vol, rtol=1.0e-03)


    def test_submergeflux_riverwall_DE0(self):
        """test_submergedflux_riverwall

            Tests that the riverwall flux is
            Q1*(1-Q2/Q1)**(0.385)

            Where Q1, Q2 =2/3*H*sqrt(2/3*g*H)
             with H computed from the smallest (Q2) / largest (Q1) head side
        """
        wallHeight=20.
        InitialOceanStage=20.50
        InitialLandStage=20.44

        domain=self.create_domain_DE0(wallHeight,InitialOceanStage, InitialLandStage)

        # Indices landward of the riverwall
        landInds=(domain.centroid_coordinates[:,0]<50.).nonzero()[0]
        InitialHeight=(domain.quantities['stage'].centroid_values[landInds]-domain.quantities['elevation'].centroid_values[landInds])
        InitialLandVol=InitialHeight*(InitialHeight>0.)*domain.areas[landInds]
        InitialLandVol=InitialLandVol.sum()

        # Run the model for a few fractions of a second
        # Any longer, and the evolution of stage starts causing
        # significant changes to the flow
        yst=1.0e-04
        ft=1.0e-03
        for t in domain.evolve(yieldstep=yst,finaltime=ft):
            if(verbose):
                print(domain.timestepping_statistics())

        # Compare with theoretical result
        L= 100. # Length of riverwall
        dt=ft
        H1=max(InitialOceanStage-wallHeight,0.) # Upstream head
        H2=max(InitialLandStage-wallHeight,0.) # Downstream head

        Q1=2./3.*H1*(2./3.*g*H1)**0.5
        Q2=2./3.*H2*(2./3.*g*H2)**0.5

        theoretical_flux_vol=dt*L*Q1*(1.-Q2/Q1)**0.385

        # Compute volume of water landward of riverwall
        FinalLandVol=domain.quantities['height'].centroid_values[landInds]*domain.areas[landInds]
        FinalLandVol=FinalLandVol.sum()

        landVol=FinalLandVol-InitialLandVol

        #import pdb
        #pdb.set_trace()

        if(verbose):
            print('Land Vol: ', landVol, 'theoretical vol: ', theoretical_flux_vol)

        assert numpy.allclose(landVol,theoretical_flux_vol, rtol=1.0e-03)

    def test_riverwall_includes_specified_points_in_domain(self):
        """
            Check that all domain points that should be on the riverwall
            actually are, and that there are no 'non-riverwall' points on the riverwall
        """
        wallHeight=-0.2
        InitialOceanStage=-0.3
        InitialLandStage=-999999.

        domain=self.create_domain_DE1(wallHeight,InitialOceanStage, InitialLandStage)

        edgeInds_on_wall=domain.riverwallData.riverwall_edges.tolist()

        riverWall_x_coord=domain.edge_coordinates[edgeInds_on_wall,0]-wallLoc

        assert(numpy.allclose(riverWall_x_coord,0.))

        # Now check that all the other domain edge coordinates are not on the wall
        # Note the threshold requires a sufficiently coarse mesh
        notriverWall_x_coord=numpy.delete(domain.edge_coordinates[:,0], edgeInds_on_wall)
        assert(min(abs(notriverWall_x_coord-wallLoc))>1.0e-01)

    def test_is_vertex_on_boundary(self):
        """
            Check that is_vertex_on_boundary is working as expected
        """
        wallHeight=-0.2
        InitialOceanStage=-0.3
        InitialLandStage=-999999.

        domain=self.create_domain_DE1(wallHeight,InitialOceanStage, InitialLandStage)

        allVertices=numpy.array(list(range(len(domain.vertex_coordinates))))
        boundaryFlag=domain.riverwallData.is_vertex_on_boundary(allVertices)
        boundaryVerts=boundaryFlag.nonzero()[0].tolist()

        # Check that all boundary vertices are on the boundary
        check2=(domain.vertex_coordinates[boundaryVerts,0]==0.)+\
               (domain.vertex_coordinates[boundaryVerts,0]==100.)+\
               (domain.vertex_coordinates[boundaryVerts,1]==100.)+\
               (domain.vertex_coordinates[boundaryVerts,1]==0.)

        assert(all(check2>0.))

        # Check that all non-boundary vertices are not
        nonboundaryVerts=(boundaryFlag==0).nonzero()[0].tolist()
        check2=(domain.vertex_coordinates[nonboundaryVerts,0]==0.)+\
               (domain.vertex_coordinates[nonboundaryVerts,0]==100.)+\
               (domain.vertex_coordinates[nonboundaryVerts,1]==100.)+\
               (domain.vertex_coordinates[nonboundaryVerts,1]==0.)

        assert(all(check2==0))

    def test_multiple_riverwalls(self):
        """
            Testcase with multiple riverwalls -- check all is working as required

         Idea -- add other riverwalls with different Qfactor / height.
                 Set them up to have no hydraulic effect, but
                 so that we are likely to catch bugs if the code is not right
        """
        wallHeight=2.
        InitialOceanStage=2.50
        InitialLandStage=-999999.


        riverWall={ 'awall1':
                        [ [wallLoc+20., 0.0, -9999.],
                          [wallLoc+20., 100.0, -9999.]],
                    'centralWall':
                        [ [wallLoc, 0.0, wallHeight],
                          [wallLoc, 100.0, wallHeight]] ,
                    'awall2':
                        [ [wallLoc-20., 0.0, 30.],
                          [wallLoc-20., 100.0, 30.]],
                  }

        newQfac=2.0
        riverWall_Par={'centralWall':{'Qfactor':newQfac}, 'awall1':{'Qfactor':100.}, 'awall2':{'Qfactor':0.}}

        domain=self.create_domain_DE0(wallHeight,InitialOceanStage, InitialLandStage, riverWall=riverWall,riverWall_Par=riverWall_Par)


        domain.riverwallData.create_riverwalls(riverWall,riverWall_Par,verbose=verbose)

        # Run the model for a few fractions of a second
        # Any longer, and the evolution of stage starts causing
        # significant changes to the flow
        yst=1.0e-04
        ft=1.0e-03
        for t in domain.evolve(yieldstep=yst,finaltime=ft):
            if(verbose):
                print(domain.timestepping_statistics())

        # Compare with theoretical result
        L= 100. # Length of riverwall
        H=InitialOceanStage-wallHeight # Upstream head
        dt=ft
        theoretical_flux_vol=newQfac*dt*L*2./3.*H*(2./3.*g*H)**0.5

        # Indices landward of the riverwall
        landInds=(domain.centroid_coordinates[:,0]<50.).nonzero()[0]
        # Compute volume of water landward of riverwall
        landVol=domain.quantities['height'].centroid_values[landInds]*domain.areas[landInds]
        landVol=landVol.sum()

        if(verbose):
            print('Land Vol: ', landVol, 'theoretical vol: ', theoretical_flux_vol)

        assert numpy.allclose(landVol,theoretical_flux_vol, rtol=1.0e-03)

    # -----------------------------------------------------------------------
    # Throughflow (Cd_through) tests
    # -----------------------------------------------------------------------

    def test_throughflow_parameter_stored(self):
        """Cd_through is stored in hydraulic_properties table at column index 5."""
        wallHeight = 2.0
        Cd = 0.5
        riverWall = {'centralWall': [[wallLoc, 0.0, wallHeight],
                                     [wallLoc, 100.0, wallHeight]]}
        riverWall_Par = {'centralWall': {'Cd_through': Cd}}
        domain = self.create_domain_DE0(wallHeight, 1.0, 0.5,
                                        riverWall=riverWall,
                                        riverWall_Par=riverWall_Par)
        hp = domain.riverwallData.hydraulic_properties
        names = domain.riverwallData.hydraulic_variable_names
        assert 'Cd_through' in names, "Cd_through missing from hydraulic_variable_names"
        col = list(names).index('Cd_through')
        assert col == 5, f"Cd_through should be column 5, got {col}"
        assert numpy.allclose(hp[0, col], Cd), \
            f"Expected Cd_through={Cd}, got {hp[0, col]}"

    def test_throughflow_default_zero(self):
        """Cd_through defaults to 0 — behaviour matches impermeable wall."""
        wallHeight = 2.0
        InitialOceanStage = 1.0
        InitialLandStage = -999999.

        # With default Cd_through=0 the wall is impermeable below the crest.
        # Stage below the wall on both sides: no overtopping, no throughflow → land stays dry.
        domain = self.create_domain_DE0(wallHeight, InitialOceanStage, InitialLandStage)

        for t in domain.evolve(yieldstep=10.0, finaltime=10.0):
            pass

        landInds = (domain.centroid_coordinates[:, 0] < 50.).nonzero()[0]
        landVol = (domain.quantities['height'].centroid_values[landInds]
                   * domain.areas[landInds]).sum()
        assert landVol < 1.0e-6, \
            f"With Cd_through=0 land should stay dry, got volume={landVol}"

    def test_throughflow_dry_downstream(self):
        """With Cd_through>0 and dry downstream side, water flows through the wall."""
        wallHeight = 2.0
        InitialOceanStage = 1.0    # below crest, so no overtopping
        InitialLandStage = -999999.

        riverWall = {'centralWall': [[wallLoc, 0.0, wallHeight],
                                     [wallLoc, 100.0, wallHeight]]}
        riverWall_Par = {'centralWall': {'Cd_through': 0.5}}

        domain = self.create_domain_DE0(wallHeight, InitialOceanStage, InitialLandStage,
                                        riverWall=riverWall, riverWall_Par=riverWall_Par)

        for t in domain.evolve(yieldstep=10.0, finaltime=10.0):
            pass

        landInds = (domain.centroid_coordinates[:, 0] < 50.).nonzero()[0]
        landVol = (domain.quantities['height'].centroid_values[landInds]
                   * domain.areas[landInds]).sum()
        assert landVol > 1.0e-4, \
            f"With Cd_through=0.5 and dry downstream, expected water flow, got volume={landVol}"

    def test_throughflow_more_cd_more_flow(self):
        """Higher Cd_through produces more throughflow volume."""
        wallHeight = 2.0
        InitialOceanStage = 1.0
        InitialLandStage = -999999.

        def run(Cd):
            riverWall = {'centralWall': [[wallLoc, 0.0, wallHeight],
                                         [wallLoc, 100.0, wallHeight]]}
            riverWall_Par = {'centralWall': {'Cd_through': Cd}}
            domain = self.create_domain_DE0(wallHeight, InitialOceanStage, InitialLandStage,
                                            riverWall=riverWall, riverWall_Par=riverWall_Par)
            for t in domain.evolve(yieldstep=0.001, finaltime=0.001):
                pass
            landInds = (domain.centroid_coordinates[:, 0] < 50.).nonzero()[0]
            return (domain.quantities['height'].centroid_values[landInds]
                    * domain.areas[landInds]).sum()

        vol_lo = run(0.1)
        vol_hi = run(0.5)
        assert vol_hi > vol_lo, \
            f"Higher Cd_through should produce more throughflow: Cd=0.1 → {vol_lo}, Cd=0.5 → {vol_hi}"

    def test_throughflow_additive_to_overtopping(self):
        """Throughflow adds to overtopping: total flux > overtopping-only flux."""
        wallHeight = 0.5
        InitialOceanStage = 1.0   # above crest: overtopping occurs
        InitialLandStage = -999999.

        yst = 0.001
        ft  = 0.001

        # No throughflow
        domain_no = self.create_domain_DE0(wallHeight, InitialOceanStage, InitialLandStage)
        for t in domain_no.evolve(yieldstep=yst, finaltime=ft):
            pass
        landInds = (domain_no.centroid_coordinates[:, 0] < 50.).nonzero()[0]
        vol_no = (domain_no.quantities['height'].centroid_values[landInds]
                  * domain_no.areas[landInds]).sum()

        # With throughflow
        riverWall = {'centralWall': [[wallLoc, 0.0, wallHeight],
                                     [wallLoc, 100.0, wallHeight]]}
        riverWall_Par = {'centralWall': {'Cd_through': 0.5}}
        domain_cd = self.create_domain_DE0(wallHeight, InitialOceanStage, InitialLandStage,
                                           riverWall=riverWall, riverWall_Par=riverWall_Par)
        for t in domain_cd.evolve(yieldstep=yst, finaltime=ft):
            pass
        vol_cd = (domain_cd.quantities['height'].centroid_values[landInds]
                  * domain_cd.areas[landInds]).sum()

        assert vol_cd > vol_no, \
            f"Throughflow+overtopping should exceed overtopping alone: {vol_cd} vs {vol_no}"

    def test_throughflow_backward_compatible(self):
        """Existing riverwall tests unaffected: Cd_through=0 leaves hydraulic_properties unchanged."""
        wallHeight = 2.0
        InitialOceanStage = 2.5
        InitialLandStage = -999999.

        domain = self.create_domain_DE0(wallHeight, InitialOceanStage, InitialLandStage)
        hp = domain.riverwallData.hydraulic_properties
        names = domain.riverwallData.hydraulic_variable_names
        col = list(names).index('Cd_through')
        # Default must be exactly 0.0
        assert hp[0, col] == 0.0, f"Default Cd_through must be 0.0, got {hp[0, col]}"
        # Should now have 6 columns
        assert hp.shape[1] == 6, f"Expected 6 hydraulic property columns, got {hp.shape[1]}"


# =========================================================================
if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Test_riverwall_structure)
    runner = unittest.TextTestRunner()
    runner.run(suite)
