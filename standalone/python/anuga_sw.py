"""ctypes driver for the standalone libanuga_sw shallow-water solver.

This is the numpy-free bridge's Python side: it loads libanuga_sw.so (built by
CMake, no Cython/meson) and drives it by passing raw pointers to an ANUGA
domain's own numpy arrays. It is the runtime-only counterpart to the build-time
Cython extension sw_domain_gpu_ext, and mirrors get_domain_pointers() /
init_*_boundary() from sw_domain_gpu_ext.pyx.

Correctness contract: every value array is *aliased*, never copied -- the C
solver reads and writes the domain's actual buffers, so results appear in
domain.quantities[...] after sync_from_device(). Arrays that don't already have
the required dtype/contiguity raise instead of being silently copied.

Usage
-----
    from anuga_sw import AnugaSW
    sw = AnugaSW(domain)        # builds descriptor, inits boundaries
    sw.map_to_device()
    while t < finaltime:
        dt = sw.evolve_one_rk2_step(max_dt, apply_forcing=1)
        ...
    sw.sync_from_device()       # pull results back into domain.quantities
"""

import ctypes
import os
import numpy as np

_c_i64 = ctypes.c_int64
_c_dbl = ctypes.c_double
_c_ptr = ctypes.c_void_p


class AnugaDomainDesc(ctypes.Structure):
    """Mirror of AnugaDomainDesc in standalone/include/anuga_sw.h.

    Field order and types MUST match the C struct exactly. All members are 8
    bytes (int64 / double / pointer) so the layout has no padding.
    """
    _fields_ = [
        # scalar sizes
        ("number_of_elements", _c_i64),
        ("boundary_length", _c_i64),
        ("number_of_riverwall_edges", _c_i64),
        # scalar parameters
        ("optimise_dry_cells", _c_i64),
        ("extrapolate_velocity_second_order", _c_i64),
        ("low_froude", _c_i64),
        ("timestep_fluxcalls", _c_i64),
        ("ncol_riverwall_hydraulic_properties", _c_i64),
        ("nrow_riverwall_hydraulic_properties", _c_i64),
        ("epsilon", _c_dbl),
        ("H0", _c_dbl),
        ("g", _c_dbl),
        ("evolve_max_timestep", _c_dbl),
        ("evolve_min_timestep", _c_dbl),
        ("minimum_allowed_height", _c_dbl),
        ("maximum_allowed_speed", _c_dbl),
        ("beta_w", _c_dbl),
        ("beta_w_dry", _c_dbl),
        ("beta_uh", _c_dbl),
        ("beta_uh_dry", _c_dbl),
        ("beta_vh", _c_dbl),
        ("beta_vh_dry", _c_dbl),
        ("CFL", _c_dbl),
        ("fixed_flux_timestep", _c_dbl),
        # mesh connectivity / geometry
        ("neighbours", _c_ptr),
        ("neighbour_edges", _c_ptr),
        ("surrogate_neighbours", _c_ptr),
        ("number_of_boundaries", _c_ptr),
        ("tri_full_flag", _c_ptr),
        ("normals", _c_ptr),
        ("edgelengths", _c_ptr),
        ("radii", _c_ptr),
        ("areas", _c_ptr),
        ("max_speed", _c_ptr),
        ("centroid_coordinates", _c_ptr),
        ("edge_coordinates", _c_ptr),
        ("x_centroid_work", _c_ptr),
        ("y_centroid_work", _c_ptr),
        # stage
        ("stage_centroid_values", _c_ptr),
        ("stage_edge_values", _c_ptr),
        ("stage_boundary_values", _c_ptr),
        ("stage_explicit_update", _c_ptr),
        ("stage_semi_implicit_update", _c_ptr),
        ("stage_backup_values", _c_ptr),
        # xmomentum
        ("xmom_centroid_values", _c_ptr),
        ("xmom_edge_values", _c_ptr),
        ("xmom_boundary_values", _c_ptr),
        ("xmom_explicit_update", _c_ptr),
        ("xmom_semi_implicit_update", _c_ptr),
        ("xmom_backup_values", _c_ptr),
        # ymomentum
        ("ymom_centroid_values", _c_ptr),
        ("ymom_edge_values", _c_ptr),
        ("ymom_boundary_values", _c_ptr),
        ("ymom_explicit_update", _c_ptr),
        ("ymom_semi_implicit_update", _c_ptr),
        ("ymom_backup_values", _c_ptr),
        # elevation (bed)
        ("bed_centroid_values", _c_ptr),
        ("bed_edge_values", _c_ptr),
        ("bed_boundary_values", _c_ptr),
        # height
        ("height_centroid_values", _c_ptr),
        ("height_edge_values", _c_ptr),
        ("height_boundary_values", _c_ptr),
        # friction
        ("friction_centroid_values", _c_ptr),
        # riverwall (nullable)
        ("edge_flux_type", _c_ptr),
        ("edge_river_wall_counter", _c_ptr),
        ("riverwall_elevation", _c_ptr),
        ("riverwall_rowIndex", _c_ptr),
        ("riverwall_hydraulic_properties", _c_ptr),
    ]


def _find_library(explicit=None):
    """Locate libanuga_sw.so. Order: explicit arg, ANUGA_SW_LIB env, build dirs."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("ANUGA_SW_LIB"):
        candidates.append(os.environ["ANUGA_SW_LIB"])
    here = os.path.dirname(os.path.abspath(__file__))
    standalone = os.path.dirname(here)
    for name in ("libanuga_sw.so", "libanuga_sw.dylib", "anuga_sw.dll"):
        candidates.append(os.path.join(standalone, "build", name))
        candidates.append(os.path.join(standalone, "build", "lib", name))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise OSError(
        "libanuga_sw not found. Build it with:\n"
        "  cmake -S standalone -B standalone/build -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build standalone/build -j\n"
        "or set ANUGA_SW_LIB to its path. Tried: " + ", ".join(map(str, candidates))
    )


def _bind(lib):
    """Declare argument/return types for the C ABI."""
    P = ctypes.POINTER(AnugaDomainDesc)
    H = ctypes.c_void_p  # opaque AnugaDomain*
    sig = {
        "anuga_built_with_offload": ([], ctypes.c_int),
        "anuga_built_with_mpi": ([], ctypes.c_int),
        "anuga_gpu_available": ([], ctypes.c_int),
        "anuga_domain_create": ([P], H),
        "anuga_domain_set_mpi_comm_fint": ([H, ctypes.c_int], None),
        "anuga_domain_map_to_device": ([H], ctypes.c_int),
        "anuga_domain_destroy": ([H], None),
        "anuga_sync_to_device": ([H], None),
        "anuga_sync_from_device": ([H], None),
        "anuga_sync_all_from_device": ([H], None),
        "anuga_init_reflective": ([H, ctypes.c_int, _c_ptr, _c_ptr, _c_ptr], None),
        "anuga_init_dirichlet": ([H, ctypes.c_int, _c_ptr, _c_ptr, _c_ptr,
                                  _c_dbl, _c_dbl, _c_dbl], None),
        "anuga_init_transmissive": ([H, ctypes.c_int, _c_ptr, _c_ptr, _c_ptr,
                                     ctypes.c_int], None),
        "anuga_evaluate_boundaries": ([H], None),
        "anuga_evolve_one_euler_step": ([H, _c_dbl, ctypes.c_int], _c_dbl),
        "anuga_evolve_one_rk2_step": ([H, _c_dbl, ctypes.c_int], _c_dbl),
        "anuga_evolve_one_rk3_step": ([H, _c_dbl, ctypes.c_int], _c_dbl),
        "anuga_evolve_one_ader2_step": ([H, _c_dbl, ctypes.c_int, _c_dbl], _c_dbl),
        "anuga_extrapolate_second_order": ([H], None),
        "anuga_compute_fluxes": ([H], _c_dbl),
        "anuga_update_conserved_quantities": ([H, _c_dbl], None),
        "anuga_protect": ([H], _c_dbl),
        "anuga_manning_friction": ([H], None),
        "anuga_backup_conserved_quantities": ([H], None),
        "anuga_saxpy_conserved_quantities": ([H, _c_dbl, _c_dbl], None),
        "anuga_compute_water_volume": ([H], _c_dbl),
    }
    for name, (argtypes, restype) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


class AnugaSW:
    """Drive an ANUGA domain through libanuga_sw via ctypes."""

    def __init__(self, domain, libpath=None):
        self.domain = domain
        self.lib = _bind(ctypes.CDLL(_find_library(libpath)))
        self._keep = []          # references to aliased numpy arrays (lifetime)
        self._handle = None
        self.desc = None
        self._build_descriptor()
        self._handle = self.lib.anuga_domain_create(ctypes.byref(self.desc))
        if not self._handle:
            raise RuntimeError("anuga_domain_create returned NULL")
        self._init_boundaries()

    # ----- build info -------------------------------------------------------
    @property
    def built_with_offload(self):
        return bool(self.lib.anuga_built_with_offload())

    @property
    def built_with_mpi(self):
        return bool(self.lib.anuga_built_with_mpi())

    # ----- descriptor construction -----------------------------------------
    def _alias(self, arr, dtype, name):
        """Return arr.ctypes.data, asserting dtype+C-contiguity (no copy).

        Aliasing (not copying) is required so the C solver writes results back
        into the domain's own buffers.
        """
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.dtype != np.dtype(dtype):
            raise TypeError(
                f"{name}: expected dtype {np.dtype(dtype)}, got {a.dtype}. "
                "Aliasing requires the exact dtype to avoid a silent copy."
            )
        if not a.flags["C_CONTIGUOUS"]:
            raise ValueError(f"{name}: array must be C-contiguous for aliasing.")
        self._keep.append(a)
        return a.ctypes.data

    def _build_descriptor(self):
        d = self.domain
        if hasattr(d, "_ensure_work_arrays"):
            d._ensure_work_arrays()

        q = d.quantities
        stage, xmom, ymom = q["stage"], q["xmomentum"], q["ymomentum"]
        elev, height, friction = q["elevation"], q["height"], q["friction"]

        desc = AnugaDomainDesc()

        # scalar sizes / params
        desc.number_of_elements = int(d.number_of_elements)
        desc.boundary_length = int(d.boundary_length)
        desc.number_of_riverwall_edges = int(getattr(d, "number_of_riverwall_edges", 0))
        desc.optimise_dry_cells = int(getattr(d, "optimise_dry_cells", 0))
        desc.extrapolate_velocity_second_order = int(d.extrapolate_velocity_second_order)
        desc.low_froude = int(d.low_froude)
        desc.timestep_fluxcalls = int(getattr(d, "timestep_fluxcalls", 1))
        rwd = getattr(d, "riverwallData", None)
        desc.ncol_riverwall_hydraulic_properties = int(getattr(rwd, "ncol_hydraulic_properties", 0)) if rwd else 0
        desc.nrow_riverwall_hydraulic_properties = len(rwd.names) if (rwd and hasattr(rwd, "names")) else 0

        desc.epsilon = float(d.epsilon)
        desc.H0 = float(d.H0)
        desc.g = float(d.g)
        desc.evolve_max_timestep = float(d.evolve_max_timestep)
        desc.evolve_min_timestep = float(getattr(d, "evolve_min_timestep", 0.0))
        desc.minimum_allowed_height = float(d.minimum_allowed_height)
        desc.maximum_allowed_speed = float(d.maximum_allowed_speed)
        desc.beta_w = float(d.beta_w)
        desc.beta_w_dry = float(d.beta_w_dry)
        desc.beta_uh = float(d.beta_uh)
        desc.beta_uh_dry = float(d.beta_uh_dry)
        desc.beta_vh = float(d.beta_vh)
        desc.beta_vh_dry = float(d.beta_vh_dry)
        desc.CFL = float(d.CFL)
        fft = getattr(d, "fixed_flux_timestep", None)
        desc.fixed_flux_timestep = float(fft) if fft else -1.0

        i64, f64 = np.int64, np.float64

        # mesh connectivity / geometry
        desc.neighbours = self._alias(d.neighbours, i64, "neighbours")
        desc.neighbour_edges = self._alias(d.neighbour_edges, i64, "neighbour_edges")
        desc.surrogate_neighbours = self._alias(d.surrogate_neighbours, i64, "surrogate_neighbours")
        desc.number_of_boundaries = self._alias(d.number_of_boundaries, i64, "number_of_boundaries")
        desc.tri_full_flag = self._alias(getattr(d, "tri_full_flag", None), i64, "tri_full_flag")
        desc.normals = self._alias(d.normals, f64, "normals")
        desc.edgelengths = self._alias(d.edgelengths, f64, "edgelengths")
        desc.radii = self._alias(d.radii, f64, "radii")
        desc.areas = self._alias(d.areas, f64, "areas")
        desc.max_speed = self._alias(d.max_speed, f64, "max_speed")
        desc.centroid_coordinates = self._alias(d.centroid_coordinates, f64, "centroid_coordinates")
        desc.edge_coordinates = self._alias(d.edge_coordinates, f64, "edge_coordinates")
        desc.x_centroid_work = self._alias(d.x_centroid_work, f64, "x_centroid_work")
        desc.y_centroid_work = self._alias(d.y_centroid_work, f64, "y_centroid_work")

        # quantities
        for prefix, qty in (("stage", stage), ("xmom", xmom), ("ymom", ymom)):
            desc.__setattr__(f"{prefix}_centroid_values",
                             self._alias(qty.centroid_values, f64, f"{prefix}_centroid_values"))
            desc.__setattr__(f"{prefix}_edge_values",
                             self._alias(qty.edge_values, f64, f"{prefix}_edge_values"))
            desc.__setattr__(f"{prefix}_boundary_values",
                             self._alias(qty.boundary_values, f64, f"{prefix}_boundary_values"))
            desc.__setattr__(f"{prefix}_explicit_update",
                             self._alias(qty.explicit_update, f64, f"{prefix}_explicit_update"))
            desc.__setattr__(f"{prefix}_semi_implicit_update",
                             self._alias(qty.semi_implicit_update, f64, f"{prefix}_semi_implicit_update"))
            desc.__setattr__(f"{prefix}_backup_values",
                             self._alias(qty.centroid_backup_values, f64, f"{prefix}_backup_values"))

        desc.bed_centroid_values = self._alias(elev.centroid_values, f64, "bed_centroid_values")
        desc.bed_edge_values = self._alias(elev.edge_values, f64, "bed_edge_values")
        desc.bed_boundary_values = self._alias(elev.boundary_values, f64, "bed_boundary_values")

        desc.height_centroid_values = self._alias(height.centroid_values, f64, "height_centroid_values")
        desc.height_edge_values = self._alias(height.edge_values, f64, "height_edge_values")
        desc.height_boundary_values = self._alias(height.boundary_values, f64, "height_boundary_values")

        desc.friction_centroid_values = self._alias(friction.centroid_values, f64, "friction_centroid_values")

        # riverwall (left NULL when absent)
        desc.edge_flux_type = self._alias(getattr(d, "edge_flux_type", None), i64, "edge_flux_type")
        desc.edge_river_wall_counter = self._alias(getattr(d, "edge_river_wall_counter", None), i64, "edge_river_wall_counter")
        if rwd is not None:
            desc.riverwall_elevation = self._alias(getattr(rwd, "riverwall_elevation", None), f64, "riverwall_elevation")
            desc.riverwall_rowIndex = self._alias(getattr(rwd, "hydraulic_properties_rowIndex", None), i64, "riverwall_rowIndex")
            desc.riverwall_hydraulic_properties = self._alias(getattr(rwd, "hydraulic_properties", None), f64, "riverwall_hydraulic_properties")

        self.desc = desc

    # ----- boundary setup (mirrors init_*_boundary in the .pyx) -------------
    def _segment_ids(self, classname):
        d = self.domain
        if d.boundary_map is None:
            return None
        ids = []
        for tag, b in d.boundary_map.items():
            if b is not None and b.__class__.__name__ == classname:
                seg = d.tag_boundary_cells.get(tag, None)
                if seg is not None and len(seg) > 0:
                    ids.extend(seg)
        return ids

    def _index_arrays(self, ids):
        d = self.domain
        ids = np.array(ids, dtype=np.intc)
        bidx = np.ascontiguousarray(ids, dtype=np.intc)
        vol = np.ascontiguousarray(d.boundary_cells[ids], dtype=np.intc)
        edg = np.ascontiguousarray(d.boundary_edges[ids], dtype=np.intc)
        # C side copies these immediately, but keep refs for the call duration.
        return bidx, vol, edg

    def _init_boundaries(self):
        d = self.domain
        if d.boundary_map is None:
            return

        ids = self._segment_ids("Reflective_boundary")
        if ids:
            bidx, vol, edg = self._index_arrays(ids)
            self.lib.anuga_init_reflective(self._handle, len(bidx),
                                           bidx.ctypes.data, vol.ctypes.data, edg.ctypes.data)

        # Dirichlet: GPU path uses a single (stage,xmom,ymom) for all such edges.
        d_ids, dvals = [], (0.0, 0.0, 0.0)
        if d.boundary_map is not None:
            for tag, b in d.boundary_map.items():
                if b is not None and b.__class__.__name__ == "Dirichlet_boundary":
                    seg = d.tag_boundary_cells.get(tag, None)
                    if seg is not None and len(seg) > 0:
                        d_ids.extend(seg)
                        dv = getattr(b, "dirichlet_values", None)
                        if dv is not None and len(dv) >= 3:
                            dvals = (float(dv[0]), float(dv[1]), float(dv[2]))
        if d_ids:
            bidx, vol, edg = self._index_arrays(d_ids)
            self.lib.anuga_init_dirichlet(self._handle, len(bidx),
                                          bidx.ctypes.data, vol.ctypes.data, edg.ctypes.data,
                                          dvals[0], dvals[1], dvals[2])

        ids = self._segment_ids("Transmissive_boundary")
        if ids:
            use_centroid = 1 if getattr(d, "centroid_transmissive_bc", False) else 0
            bidx, vol, edg = self._index_arrays(ids)
            self.lib.anuga_init_transmissive(self._handle, len(bidx),
                                             bidx.ctypes.data, vol.ctypes.data, edg.ctypes.data,
                                             use_centroid)

    # ----- lifecycle / sync -------------------------------------------------
    def map_to_device(self):
        if not self.lib.anuga_domain_map_to_device(self._handle):
            raise RuntimeError("anuga_domain_map_to_device failed (device memory?)")

    def sync_to_device(self):
        self.lib.anuga_sync_to_device(self._handle)

    def sync_from_device(self):
        self.lib.anuga_sync_from_device(self._handle)

    def sync_all_from_device(self):
        self.lib.anuga_sync_all_from_device(self._handle)

    # ----- evolve -----------------------------------------------------------
    def evolve_one_euler_step(self, max_timestep, apply_forcing=1):
        return self.lib.anuga_evolve_one_euler_step(self._handle, max_timestep, apply_forcing)

    def evolve_one_rk2_step(self, max_timestep, apply_forcing=1):
        return self.lib.anuga_evolve_one_rk2_step(self._handle, max_timestep, apply_forcing)

    def evolve_one_rk3_step(self, max_timestep, apply_forcing=1):
        return self.lib.anuga_evolve_one_rk3_step(self._handle, max_timestep, apply_forcing)

    def evolve_one_ader2_step(self, max_timestep, apply_forcing=1, prev_dt=0.0):
        return self.lib.anuga_evolve_one_ader2_step(self._handle, max_timestep, apply_forcing, prev_dt)

    # ----- kernels (for parity testing) ------------------------------------
    def extrapolate_second_order(self):
        self.lib.anuga_extrapolate_second_order(self._handle)

    def compute_fluxes(self):
        return self.lib.anuga_compute_fluxes(self._handle)

    def update_conserved_quantities(self, timestep):
        self.lib.anuga_update_conserved_quantities(self._handle, timestep)

    def protect(self):
        return self.lib.anuga_protect(self._handle)

    def manning_friction(self):
        self.lib.anuga_manning_friction(self._handle)

    def backup_conserved_quantities(self):
        self.lib.anuga_backup_conserved_quantities(self._handle)

    def saxpy_conserved_quantities(self, a, b):
        self.lib.anuga_saxpy_conserved_quantities(self._handle, a, b)

    def evaluate_boundaries(self):
        self.lib.anuga_evaluate_boundaries(self._handle)

    def compute_water_volume(self):
        return self.lib.anuga_compute_water_volume(self._handle)

    # ----- teardown ---------------------------------------------------------
    def close(self):
        if self._handle:
            self.lib.anuga_domain_destroy(self._handle)
            self._handle = None
        self._keep = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
