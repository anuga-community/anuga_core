"""Backward-compatibility shim — use culvert_class instead.

All classes previously defined here have been merged into
:mod:`anuga.culvert_flows.culvert_class`.  Both modules are deprecated;
use :class:`anuga.Boyd_box_operator`, :class:`anuga.Boyd_pipe_operator`,
or :class:`anuga.Weir_orifice_trapezoid_operator` instead.
"""

from anuga.culvert_flows.culvert_class import (  # noqa: F401
    Below_interval,
    Above_interval,
    Culvert_flow_general,
    Culvert_flow_rating,
    Culvert_flow_energy,
    Culvert_flow,
)
