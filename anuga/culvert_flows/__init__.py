"""Legacy culvert flow module — deprecated.

Use :class:`anuga.Boyd_box_operator` (box culverts),
:class:`anuga.Boyd_pipe_operator` (pipe/circular culverts), or
:class:`anuga.Weir_orifice_trapezoid_operator` (weirs) instead.

TODO: Remove this entire package in v5.0 once the one remaining example
(examples/structures/run_open_slot_wide_bridge.py) has been updated to
use Boyd_box_operator, and after confirming no user scripts depend on
culvert_class.Culvert_flow or culvert_routines.boyd_generalised_culvert_model.
"""

import warnings
warnings.warn(
    "anuga.culvert_flows is deprecated and will be removed in a future release. "
    "Use anuga.Boyd_box_operator, anuga.Boyd_pipe_operator, or "
    "anuga.Weir_orifice_trapezoid_operator instead.",
    DeprecationWarning,
    stacklevel=2,
)
