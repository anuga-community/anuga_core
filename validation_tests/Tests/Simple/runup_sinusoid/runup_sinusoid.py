"""Runup example from the manual, slightly modified
"""
#---------
#Import Modules
#--------
import anuga

import numpy

from math import sin, pi, exp
from anuga.shallow_water.shallow_water_domain import Domain as Domain
#from anuga.shallow_water_balanced2.swb2_domain import Domain as Domain
#path.append('/home/gareth/storage/anuga_clean/anuga_jan12/trunk/anuga_work/development/gareth/balanced_basic')
#from swb2_domain import *
#from balanced_basic import *
#from balanced_dev import *
#---------
#Setup computational domain
#---------
points, vertices, boundary = anuga.rectangular_cross(40,40, len1=1., len2=1.)

domain=Domain(points,vertices,boundary)    # Create Domain
domain.set_name('runup_sinusoid_v2')                         # Output to file runup.sww
domain.set_datadir('.')                          # Use current folder
domain.set_quantities_to_be_stored({'stage': 2, 'xmomentum': 2, 'ymomentum': 2, 'elevation': 1})
#domain.set_store_vertices_uniquely(True)
#domain.minimum_allowed_height=0.001

#------------------------------------------------------------------------------
# Setup Algorithm, either using command line arguments
# or override manually yourself
#------------------------------------------------------------------------------
from anuga.utilities.argparsing import parse_standard_args
alg, cfl = parse_standard_args()
domain.set_flow_algorithm(alg)
domain.set_CFL(cfl)


#------------------
# Define topography
#------------------
scale_me=1.0

#domain.minimum_allowed_height=domain.minimum_allowed_height*scale_me # Seems needed to make the algorithms behave

def topography(x,y):
	return (-x/2.0 +0.05*numpy.sin((x+y)*50.0))*scale_me

def stagefun(x,y):
    stge=-0.2*scale_me #-0.1*(x>0.5) -0.2*(x<=0.5)
    #topo=topography(x,y) 
    return stge#*(stge>topo) + (topo)*(stge<=topo)

domain.set_quantity('elevation',topography)     # Use function for elevation
domain.get_quantity('elevation').smooth_vertex_values() 

domain.set_quantity('friction',0.00)             # Constant friction

#def frict_change(x,y):
#	return 0.2*(x>0.5)+0.1*(x<=0.5)
#
#domain.set_quantity('friction',frict_change)

domain.set_quantity('stage', stagefun)              # Constant negative initial stage
domain.get_quantity('stage').smooth_vertex_values()

# Experiment with rain.
# rainin = anuga.shallow_water.forcing.Rainfall(domain, rate=0.001) #, center=(0.,0.), radius=1000. )
# domain.forcing_terms.append(rainin)

#--------------------------
# Setup boundary conditions
#--------------------------
Br=anuga.Reflective_boundary(domain)            # Solid reflective wall
Bt=anuga.Transmissive_boundary(domain)          # Continue all values of boundary -- not used in this example
Bd=anuga.Dirichlet_boundary([-0.1*scale_me,0.,0.])       # Constant boundary values -- not used in this example
#Bw=anuga.Time_boundary(domain=domain,
#	f=lambda t: [(0.0*sin(t*2*pi)-0.1)*exp(-t)-0.1,0.0,0.0]) # Time varying boundary -- get rid of the 0.0 to do a runup.

#----------------------------------------------
# Associate boundary tags with boundary objects
#----------------------------------------------
domain.set_boundary({'left': Br, 'right': Bd, 'top': Br, 'bottom':Br})

#------------------------------
#Evolve the system through time
#------------------------------

for t in domain.evolve(yieldstep=0.1,finaltime=20.0):
    print domain.timestepping_statistics()
    xx = domain.quantities['xmomentum'].centroid_values
    yy = domain.quantities['ymomentum'].centroid_values
    dd = domain.quantities['stage'].centroid_values - domain.quantities['elevation'].centroid_values 
    dd = (dd)*(dd>1.0e-03)+1.0e-06
    vv = ( (xx/dd)**2 + (yy/dd)**2 )**0.5
    vv = vv*(dd>1.0e-03)
    print 'Peak velocity is: ', vv.max(), vv.argmax()

print 'Finished'