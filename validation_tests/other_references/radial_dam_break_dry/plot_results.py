"""
    Quick plot of the dam break outputs
"""
import anuga.utilities.plot_utils as util
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as pyplot
import csv, pprint
from numpy import zeros
import pandas as pd

#--------------------------------------------
# Recall the reference solution
# The reference solution is computed using 10^4 cells
# with FVM on varying bottom and width
# See: Roberts and Wilson, ANZIAM Journal (CTAC2010).
#--------------------------------------------
#################Opening the reference solution################
df = pd.read_csv('Ver_numerical_2.000000.csv', header=None, 
                  names=['Vertices', 'VerW', 'VerP', 'VerZ', 'VerH', 'VerU'])

Vertices = df['Vertices'].to_numpy()
VerW = df['VerW'].to_numpy()
VerP = df['VerP'].to_numpy()
VerZ = df['VerZ'].to_numpy()
VerH = df['VerH'].to_numpy()
VerU = df['VerU'].to_numpy()
Vertices_left = -1.0*df['Vertices'].iloc[::-1]
VerW_left = df['VerW'].iloc[::-1]
VerP_left = -1.0*df['VerP'].iloc[::-1]
VerZ_left = df['VerZ'].iloc[::-1]
VerH_left = df['VerH'].iloc[::-1]
VerU_left = -1.0*df['VerU'].iloc[::-1]

p_st = util.get_output('radial_dam_break.sww')
p2_st=util.get_centroids(p_st)

v = p2_st.y[79598]
v2=(p2_st.y==v)

#Plot stages
pyplot.clf()
pyplot.plot(p2_st.x[v2], p2_st.stage[-1,v2],'b.-', label='numerical stage')
pyplot.plot(Vertices, VerW,'r-', label='reference stage')
pyplot.plot(Vertices_left, VerW_left,'r-')
pyplot.title('Stage at an instant in time')
pyplot.legend(loc='best')
pyplot.xlabel('Radial position')
pyplot.ylabel('Stage')
pyplot.savefig('stage_plot.png')
#pyplot.show()


#Plot rmomentum
pyplot.clf()
pyplot.plot(p2_st.x[v2], p2_st.xmom[-1,v2], 'b.-', label='numerical')
pyplot.plot(Vertices, VerP,'r-', label='reference')
pyplot.plot(Vertices_left, VerP_left,'r-')
pyplot.title('Radial momentum at an instant in time')
pyplot.legend(loc='best')
pyplot.xlabel('Radial position')
pyplot.ylabel('Radial momentum')
pyplot.savefig('rmom_plot.png')


#Plot rvelocities
pyplot.clf()
pyplot.plot(p2_st.x[v2], p2_st.xvel[-1,v2], 'b.-', label='numerical')
pyplot.plot(Vertices, VerU,'r-', label='reference')
pyplot.plot(Vertices_left, VerU_left,'r-')
pyplot.title('Radial velocity at an instant in time')
pyplot.legend(loc='best')
pyplot.xlabel('Radial position')
pyplot.ylabel('Radial velocity')
pyplot.savefig('rvel_plot.png')
