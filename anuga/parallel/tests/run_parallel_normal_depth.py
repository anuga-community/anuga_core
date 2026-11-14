"""Run a channel with a normal depth outlet (sequentially or in parallel)
to support test_parallel_normal_depth.py
"""

import sys

import anuga
from anuga import distribute, finalize, myid, numprocs

# Set up MPI to abort on error
from anuga.utilities.parallel_abstraction import global_except_hook
sys.excepthook = global_except_hook

alg = sys.argv[1] if len(sys.argv) > 1 else 'DE1'

slope = 0.002
manning = 0.03
depth = 1.0
q = depth**(5.0/3.0)*slope**0.5/manning    # uniform discharge per unit width

if myid == 0:
    base_domain = anuga.rectangular_cross_domain(40, 8, len1=400., len2=80.)
    base_domain.set_flow_algorithm(alg)
    base_domain.set_datadir('.')
    base_domain.set_quantity('elevation', lambda x, y: -slope*x)
    base_domain.set_quantity('friction', manning)
    # Start deeper than uniform flow, so that the outlet has work to do
    base_domain.set_quantity('stage', lambda x, y: -slope*x + 1.5*depth)
    base_domain.set_quantity('xmomentum', q)
else:
    base_domain = None

if numprocs == 1:
    domain = base_domain
    domain.set_name('s_normal_depth')
else:
    domain = distribute(base_domain)
    domain.set_name('p_normal_depth')

Br = anuga.Reflective_boundary(domain)
Bd = anuga.Dirichlet_boundary([depth, q, 0.0])
Bn = anuga.Normal_depth_boundary(domain, slope=slope)
domain.set_boundary({'left': Bd, 'right': Bn, 'top': Br, 'bottom': Br})

for t in domain.evolve(yieldstep=50.0, finaltime=400.0):
    pass

domain.sww_merge(delete_old=True)
finalize()
