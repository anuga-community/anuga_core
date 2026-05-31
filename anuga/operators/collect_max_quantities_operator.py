"""
Collect max stage quantity info
"""

__author__ = "gareth"
__date__ = "$01/08/2014 4:46:39 PM$"


import numpy as num
from anuga.operators.base_operator import Operator
from anuga import Quantity
from anuga.parallel import myid
from anuga.config import max_float


class Collect_max_quantities_operator(Operator):
    """
    Operator to collect the max stage, depth, speed, and momentum magnitude
    (speed * depth, i.e. ||(uh, vh)||) during a run.

    Maxima are updated every update_frequency timesteps [any integer >= 1],
    after t exceeds collection_start_time.

    If store_to_sww=True the running maxima are written to the SWW file every
    yield step as centroid quantities max_stage_c, max_depth_c, max_speed_c,
    max_uh_c.  This avoids the need to compute maxima by scanning all timeslices
    of the SWW stage output after the run.

    Velocity is zeroed below velocity_zero_height (defaults to
    minimum_allowed_height) to suppress spurious spikes in nearly-dry cells.
    """

    def __init__(self,
                 domain,
                 update_frequency=1,
                 collection_start_time=0.,
                 velocity_zero_height=None,
                 store_to_sww=False,
                 description=None,
                 label=None,
                 logging=False,
                 verbose=False):

        Operator.__init__(self, domain, description, label, logging, verbose)

        self.domain = domain

        n = len(domain.centroid_coordinates[:, 0])
        self.max_stage = num.full(n, -max_float)
        self.max_depth = num.zeros(n)
        self.max_speed = num.zeros(n)
        self.max_uh    = num.zeros(n)

        self.xy    = domain.centroid_coordinates
        self.stage = domain.quantities['stage']
        self.elev  = domain.quantities['elevation']
        self.xmom  = domain.quantities['xmomentum']
        self.ymom  = domain.quantities['ymomentum']

        self.counter = 0
        assert update_frequency > 0, 'Update frequency must be >= 1'
        self.update_frequency = update_frequency
        self.collection_start_time = collection_start_time

        if velocity_zero_height is not None:
            self.velocity_zero_height = velocity_zero_height
        else:
            self.velocity_zero_height = domain.minimum_allowed_height

        self.store_to_sww = store_to_sww
        if store_to_sww:
            for qname in ('max_stage', 'max_depth', 'max_speed', 'max_uh'):
                Quantity(domain, name=qname, register=True, qty_type='centroid_only')
            # flag=3: centroid-only dynamic — written each yield step, no vertex storage
            domain.quantities_to_be_stored.update({
                'max_stage': 3,
                'max_depth': 3,
                'max_speed': 3,
                'max_uh':    3,
            })

    def __call__(self):
        """
        Update max quantities every update_frequency timesteps once
        t > collection_start_time.
        """

        t = self.domain.get_time()

        if t > self.collection_start_time:
            self.counter += 1

            if self.counter == self.update_frequency:
                stage_c = self.stage.centroid_values
                elev_c  = self.elev.centroid_values
                xmom_c  = self.xmom.centroid_values
                ymom_c  = self.ymom.centroid_values

                self.max_stage = num.maximum(self.max_stage, stage_c)

                mom_norm = num.hypot(xmom_c, ymom_c)
                self.max_uh = num.maximum(self.max_uh, mom_norm)

                depth = num.maximum(stage_c - elev_c, 0.0)
                self.max_depth = num.maximum(self.max_depth, depth)

                vel = (mom_norm / num.maximum(depth, self.velocity_zero_height)) \
                      * (depth > self.velocity_zero_height)
                self.max_speed = num.maximum(self.max_speed, vel)

                if self.store_to_sww:
                    d = self.domain
                    d.quantities['max_stage'].centroid_values[:] = self.max_stage
                    d.quantities['max_depth'].centroid_values[:] = self.max_depth
                    d.quantities['max_speed'].centroid_values[:] = self.max_speed
                    d.quantities['max_uh'].centroid_values[:]    = self.max_uh

                self.counter = 0

    def parallel_safe(self):
        """Operator is applied independently on each cell and so is parallel safe."""
        return True

    def statistics(self):
        return self.label + ': Collect_max_quantity operator'

    def timestepping_statistics(self):
        from anuga import indent
        return indent + self.label + ': Collecting_max_quantity'

    def export_max_quantities_to_csv(self, filename_start='Max_Quantities_'):
        """Export max quantities to a CSV file."""
        full_ids = self.domain.tri_full_flag.nonzero()[0]
        geo = self.domain.geo_reference
        out = num.vstack([
            self.xy[full_ids, 0] + geo.xllcorner,
            self.xy[full_ids, 1] + geo.yllcorner,
            self.max_stage[full_ids],
            self.max_depth[full_ids],
            self.max_speed[full_ids],
            self.max_uh[full_ids],
        ]).T
        outname = filename_start + 'P' + str(myid) + '_X_Y_Stage_Depth_Speed_UH_MAX.csv'
        num.savetxt(outname, out, delimiter=',')
