#!/usr/bin/python
"""
Read a TOML configuration file for an ANUGA scenario.

Drop-in replacement for the Excel-based AnugaXls / ProjectData interface in
parse_input_data.py.  Exposes exactly the same public attributes so that
prepare_data.py and all setup_*.py scripts work unchanged.

Requires Python 3.11+ (tomllib in stdlib) or the 'tomli' package for older
Python versions (pip install tomli).
"""

import os
import glob as glob_module

from anuga.utilities import spatialInputUtil as su

_VALID_FLOW_ALGORITHMS = ('DE0', 'DE1', 'DE2', 'DE0_7', 'DE1_7')

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

class _Validator:
    """Accumulates TOML configuration errors for batch reporting.

    Each check method silently records an error entry rather than raising
    immediately, so that all problems in a config file are reported at once.
    Call :meth:`raise_if_errors` at the end of parsing to raise a single
    ``ValueError`` listing every problem found.
    """

    def __init__(self):
        self.errors = []

    def require(self, mapping, key, section):
        """Return ``mapping[key]``, recording an error if the key is absent."""
        if key not in mapping:
            self.errors.append(
                f'[{section}]: required field {key!r} is missing')
            return None
        return mapping[key]

    def to_float(self, mapping, key, section):
        """Return ``float(mapping[key])``, recording errors for absent or
        non-numeric values."""
        val = self.require(mapping, key, section)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            self.errors.append(
                f'[{section}] {key!r}: expected a number, got {val!r}')
            return None

    def positive(self, val, name, section):
        """Record an error if *val* is not ``> 0``."""
        if val is not None and val <= 0:
            self.errors.append(
                f'[{section}] {name!r}: must be > 0, got {val!r}')

    def non_negative(self, val, name, section):
        """Record an error if *val* is negative."""
        if val is not None and val < 0:
            self.errors.append(
                f'[{section}] {name!r}: must be >= 0, got {val!r}')

    def in_range(self, val, lo, hi, name, section):
        """Record an error if *val* is outside ``[lo, hi]``."""
        if val is not None and not (lo <= val <= hi):
            self.errors.append(
                f'[{section}] {name!r}: must be in [{lo}, {hi}], got {val!r}')

    def one_of(self, val, choices, name, section):
        """Record an error if *val* is not in *choices*."""
        if val is not None and val not in choices:
            self.errors.append(
                f'[{section}] {name!r}: must be one of {list(choices)}, '
                f'got {val!r}')

    def integer_multiple(self, val, base, name, section):
        """Record an error if *val* is not an integer multiple of *base*."""
        if val is not None and base is not None and base > 0:
            ratio = val / base
            if abs(ratio - round(ratio)) > 1e-6:
                self.errors.append(
                    f'[{section}] {name!r}: must be an integer multiple of '
                    f'yieldstep ({base}), got {val!r}')

    def raise_if_errors(self, filename):
        """Raise ``ValueError`` listing all accumulated errors, or do nothing."""
        if self.errors:
            bullets = '\n'.join(f'  • {e}' for e in self.errors)
            raise ValueError(
                f'Configuration errors in {filename!r}:\n{bullets}')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_toml(filename):
    """Load a TOML file using tomllib (3.11+) or the tomli back-port."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            raise ImportError(
                'TOML support requires Python 3.11+ or the tomli package '
                '(pip install tomli)')
    with open(filename, 'rb') as f:
        return tomllib.load(f)


def _normpath(path):
    """Normalise *path* to the OS-native directory separator.

    TOML files conventionally use forward slashes (``/``) for portability.
    On Windows this function converts them to backslashes so that downstream
    ANUGA code receives native paths.  On Linux/macOS the path is unchanged.

    The special quantity-polygon keywords ``'All'``, ``'None'``, and
    ``'Extent'`` are returned unchanged as they are not filesystem paths.
    """
    if path in ('All', 'None', 'Extent'):
        return path
    return os.path.normpath(path)


def _glob_files(patterns):
    """Expand a list of glob patterns into a flat, sorted list of file paths.

    Patterns may use either forward or backward slashes; both are normalised
    to the OS-native separator before globbing and in the returned paths.

    Raises FileNotFoundError if any pattern matches nothing.
    """
    files = []
    for pattern in patterns:
        native = os.path.normpath(pattern)
        matches = sorted(glob_module.glob(native))
        if not matches:
            raise FileNotFoundError(
                f'No files matched pattern: {pattern!r}')
        files.extend(os.path.normpath(m) for m in matches)
    return files


def _parse_quantity_entries(entries, print_info):
    """Convert a list of TOML table entries for one quantity into the
    (data, clip_range) tuple that composite_quantity_setting_function expects.

    Each entry must have 'polygon' and 'value'.  clip_min / clip_max default
    to ±inf (no clipping).

    Handles the two-file wildcard polygon case: if 'polygon' is a glob pattern
    that matches exactly 2 line files, they are joined into a closed polygon
    (mirrors the Excel interface behaviour).
    """
    data = []
    clip_range = []

    for entry in entries:
        polygon  = _normpath(str(entry['polygon']))
        value    = entry['value']
        if isinstance(value, str):
            value = _normpath(value)
        clip_min = entry.get('clip_min', float('-inf'))
        clip_max = entry.get('clip_max', float('inf'))

        # Wildcard polygon: join 2 matching line files into a closed polygon
        if (isinstance(polygon, str) and
                polygon not in ('All', 'None', 'Extent') and
                ('*' in polygon or '?' in polygon)):
            matched = sorted(glob_module.glob(polygon))
            if len(matched) == 2:
                print_info.append(
                    f'Combining 2 files into polygon: {matched}')
                l0 = su.read_polygon(matched[0])
                l1 = su.read_polygon(matched[1])
                fake_bl = {matched[0]: l0, matched[1]: l1}
                prefix = polygon.split('*')[0]
                polygon = su.polygon_from_matching_breaklines(prefix, fake_bl)
            elif len(matched) > 2:
                raise ValueError(
                    f'Polygon pattern {polygon!r} matched {len(matched)} files; '
                    f'at most 2 are supported (they are joined into a polygon)')

        data.append([polygon, value])
        clip_range.append([clip_min, clip_max])

    return data, clip_range


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ProjectDataTOML:
    """
    Parses a TOML scenario configuration file and exposes the same public
    attributes as the Excel-based ProjectData class so that prepare_data.py
    and the setup_*.py scripts work without modification.
    """

    def __init__(self, filename):
        self.config_filename = filename
        self.print_info = [
            '---------------------',
            'PARSING CONFIG FILE (TOML)',
            '--------------------',
            '',
        ]

        cfg = _load_toml(filename)
        _v = _Validator()

        self._parse_project(cfg.get('project', {}), _v)
        self._parse_mesh(cfg.get('mesh', {}), _v)
        # Initial conditions must be parsed before bridges / pumping stations
        # because those prepend entries to elevation_data / breakline_files.
        self._parse_initial_conditions(
            cfg.get('initial_conditions', {}),
            cfg.get('initial_condition_additions', {}))
        self._parse_boundary_conditions(cfg.get('boundary_conditions', {}))
        self._parse_inlets(cfg.get('inlets', []))
        self._parse_rainfall(cfg.get('rainfall', []))
        self._parse_bridges(cfg.get('bridges', []))
        self._parse_pumping_stations(cfg.get('pumping_stations', []))
        self._parse_culverts(cfg.get('culverts', []), _v)
        self._parse_weirs(cfg.get('weirs', []), _v)

        _v.raise_if_errors(filename)

    # -----------------------------------------------------------------------
    # Project settings
    # -----------------------------------------------------------------------

    def _parse_project(self, p, _v):
        raw_scenario = _v.require(p, 'scenario', 'project')
        self.scenario = str(raw_scenario) if raw_scenario is not None else ''

        raw_outdir = _v.require(p, 'output_base_directory', 'project')
        self.output_basedir = str(raw_outdir) if raw_outdir is not None else ''

        yieldstep = _v.to_float(p, 'yieldstep', 'project')
        _v.positive(yieldstep, 'yieldstep', 'project')
        self.yieldstep = yieldstep if yieldstep is not None else 1.0

        finaltime = _v.to_float(p, 'finaltime', 'project')
        _v.positive(finaltime, 'finaltime', 'project')
        self.finaltime = finaltime if finaltime is not None else float('inf')

        raw_proj = _v.require(p, 'projection_information', 'project')
        if raw_proj is None:
            self.projection_information = None
        elif isinstance(raw_proj, float):
            self.projection_information = int(raw_proj)
        elif isinstance(raw_proj, int):
            self.projection_information = raw_proj
        else:
            self.projection_information = str(raw_proj)

        raw_alg = _v.require(p, 'flow_algorithm', 'project')
        _v.one_of(raw_alg, _VALID_FLOW_ALGORITHMS, 'flow_algorithm', 'project')
        self.flow_algorithm = str(raw_alg) if raw_alg is not None else 'DE0'

        self.output_tif_cellsize = float(p.get('output_tif_cellsize', 50.0))
        _v.positive(self.output_tif_cellsize, 'output_tif_cellsize', 'project')

        otbp = p.get('output_tif_bounding_polygon', '')
        self.output_tif_bounding_polygon = str(otbp) if otbp else None

        self.max_quantity_update_frequency = int(
            p.get('max_quantity_update_frequency', 1))
        _v.positive(self.max_quantity_update_frequency,
                    'max_quantity_update_frequency', 'project')

        self.max_quantity_collection_start_time = float(
            p.get('max_quantity_collection_starttime', 0.0))
        if finaltime is not None and \
                self.max_quantity_collection_start_time >= self.finaltime:
            _v.errors.append(
                '[project] max_quantity_collection_starttime must be < finaltime'
                f' (got {self.max_quantity_collection_start_time}, finaltime={self.finaltime})')

        self.store_vertices_uniquely        = bool(p.get('store_vertices_uniquely', False))
        self.store_elevation_every_timestep = bool(p.get('store_elevation_every_timestep', False))
        self.spatial_text_output_dir        = str(p.get('spatial_text_output_dir', 'SPATIAL_TEXT'))

        self.report_mass_conservation_statistics      = bool(p.get('report_mass_conservation_statistics', False))
        self.report_peak_velocity_statistics          = bool(p.get('report_peak_velocity_statistics', False))
        self.report_smallest_edge_timestep_statistics = bool(p.get('report_smallest_edge_timestep_statistics', False))
        self.report_operator_statistics               = bool(p.get('report_operator_statistics', False))

        # Number of OpenMP threads (1 = single-threaded; None = use OMP_NUM_THREADS env var)
        raw_omp = p.get('omp_num_threads', None)
        if raw_omp is not None:
            self.omp_num_threads = int(raw_omp)
            _v.positive(self.omp_num_threads, 'omp_num_threads', 'project')
        else:
            self.omp_num_threads = None

        # Multiprocessor mode: 1 = OpenMP (default), 2 = CuPy/GPU
        self.multiprocessor_mode = int(p.get('multiprocessor_mode', 1))
        _v.one_of(self.multiprocessor_mode, (1, 2), 'multiprocessor_mode', 'project')

        # SWW output interval [seconds]; None means write every yieldstep.
        # Must be an integer multiple of yieldstep.
        raw_os = p.get('outputstep', None)
        if raw_os is not None:
            self.outputstep = float(raw_os)
            _v.positive(self.outputstep, 'outputstep', 'project')
            _v.integer_multiple(self.outputstep, yieldstep, 'outputstep', 'project')
        else:
            self.outputstep = None

    # -----------------------------------------------------------------------
    # Mesh
    # -----------------------------------------------------------------------

    def _parse_mesh(self, m, _v):
        self.use_existing_mesh_pickle = bool(m.get('use_existing_mesh_pickle', False))

        raw_bp = _v.require(m, 'bounding_polygon', 'mesh')
        self.bounding_polygon_and_tags_file = (
            _normpath(str(raw_bp)) if raw_bp is not None else '')

        default_res = _v.to_float(m, 'default_res', 'mesh')
        _v.positive(default_res, 'default_res', 'mesh')
        self.default_res = default_res if default_res is not None else 1.0

        self.interior_regions_data = []
        for ir in m.get('interior_regions', []):
            poly = _normpath(str(ir['polygon']))
            res  = float(ir['resolution'])
            _v.positive(res, f'interior_regions resolution ({poly!r})', 'mesh')
            self.interior_regions_data.append([poly, res])

        # Explicit boundary tags — required when bounding_polygon is a CSV file,
        # ignored when it is a shapefile (tags come from the shapefile attributes)
        raw_tags = m.get('boundary_tags', [])
        if raw_tags:
            self.bounding_polygon_explicit_tags = [
                {'tag': str(t['tag']), 'edges': list(t['edges'])}
                for t in raw_tags
            ]
        else:
            self.bounding_polygon_explicit_tags = None

        self.breakline_files     = _glob_files(m.get('breakline_files', []))
        self.riverwall_csv_files = _glob_files(m.get('riverwall_csv_files', []))

        threshold = m.get('breakline_intersection_threshold', 'ignore')
        if isinstance(threshold, str):
            self.break_line_intersect_point_movement_threshold = threshold
        else:
            self.break_line_intersect_point_movement_threshold = float(threshold)

        region_file = m.get('region_areas_file', '')
        self.pt_areas = _normpath(str(region_file)) if region_file else None

        if self.pt_areas is not None:
            areas_type = str(m.get('region_areas_type', 'area'))
            if areas_type == 'length':
                self.region_resolutions_from_length = True
            elif areas_type == 'area':
                self.region_resolutions_from_length = False
            else:
                raise ValueError(
                    f'region_areas_type must be "area" or "length", '
                    f'got {areas_type!r}')

        use_ir = bool(self.interior_regions_data)
        use_bl = (bool(self.breakline_files) or
                  bool(self.riverwall_csv_files) or
                  self.pt_areas is not None)
        if use_ir and use_bl:
            raise ValueError(
                'Cannot specify both interior_regions and '
                'breaklines / riverwall_csv_files / region_areas_file')

    # -----------------------------------------------------------------------
    # Initial conditions
    # -----------------------------------------------------------------------

    def _parse_initial_conditions(self, ic, ic_add):
        quantities = ['elevation', 'friction', 'stage', 'xmomentum', 'ymomentum']

        for q in quantities:
            entries  = ic.get(q, [])
            data, clip_range = _parse_quantity_entries(entries, self.print_info)

            # Optional per-quantity spatial averaging grid spacing [m]
            mean = ic.get(f'{q}_spatial_average', None)
            if mean is not None:
                mean = float(mean)

            setattr(self, f'{q}_data',       data)
            setattr(self, f'{q}_clip_range', clip_range)
            setattr(self, f'{q}_mean',       mean)

            # Additions — clip_range is not used downstream, so discard it
            add_entries = ic_add.get(q, [])
            additions, _ = _parse_quantity_entries(add_entries, self.print_info)
            setattr(self, f'{q}_additions', additions)

    # -----------------------------------------------------------------------
    # Boundary conditions
    # -----------------------------------------------------------------------

    def _parse_boundary_conditions(self, bc):
        self.boundary_tags_attribute_name = str(
            bc.get('boundary_tags_attribute_name', 'bnd_tag'))

        self.boundary_data = []
        for b in bc.get('boundaries', []):
            tag   = str(b['tag'])
            btype = str(b['type'])
            if btype in ('Stage', 'Flather_Stage'):
                fpath = _normpath(str(b['file']))
                if not os.path.exists(fpath):
                    raise FileNotFoundError(
                        f"Boundary '{tag}': timeseries file not found: {fpath!r}")
                row = [tag, btype, fpath, float(b.get('start_time', 0.0))]
            elif btype == 'Reflective':
                row = [tag, btype]
            else:
                raise ValueError(
                    f'Unknown boundary type {btype!r} for tag {tag!r}. '
                    f'Valid types: "Reflective", "Stage", "Flather_Stage"')
            self.boundary_data.append(row)

    # -----------------------------------------------------------------------
    # Inlets
    # -----------------------------------------------------------------------

    def _parse_inlets(self, inlets):
        self.inlet_data = []
        for i in inlets:
            name  = str(i['name'])
            lfile = _normpath(str(i['line_file']))
            tfile = _normpath(str(i['timeseries_file']))
            if not os.path.exists(lfile):
                raise FileNotFoundError(
                    f"Inlet '{name}': line_file not found: {lfile!r}")
            if not os.path.exists(tfile):
                raise FileNotFoundError(
                    f"Inlet '{name}': timeseries_file not found: {tfile!r}")
            self.inlet_data.append(
                [name, lfile, tfile, float(i.get('start_time', 0.0))])

    # -----------------------------------------------------------------------
    # Rainfall
    # -----------------------------------------------------------------------

    def _parse_rainfall(self, rainfall):
        self.rain_data = []
        for r in rainfall:
            tfile = _normpath(str(r['timeseries_file']))
            if not os.path.exists(tfile):
                raise FileNotFoundError(
                    f"Rainfall: timeseries_file not found: {tfile!r}")
            polygon = _normpath(str(r.get('polygon', 'All')))
            if polygon not in ('All',) and not os.path.exists(polygon):
                raise FileNotFoundError(
                    f"Rainfall: polygon file not found: {polygon!r}")
            row = [
                tfile,
                float(r.get('start_time', 0.0)),
                str(r.get('interpolation_type', 'linear')),
                polygon,
                float(r.get('multiplier', 1.0)),
            ]
            self.rain_data.append(row)

    # -----------------------------------------------------------------------
    # Bridges
    # -----------------------------------------------------------------------

    def _parse_bridges(self, bridges):
        self.bridge_data = []
        for b in bridges:
            if not b.get('enabled', True):
                continue

            row = [
                str(b['label']),                             # [0]
                _normpath(str(b['deck_file'])),              # [1] — also used as breakline
                float(b['deck_elevation']),                  # [2] — also used as elevation value
                _normpath(str(b['exchange_line_0'])),        # [3]
                _normpath(str(b['exchange_line_1'])),        # [4]
                float(b['enquiry_gap']),                     # [5]
                _normpath(str(b['internal_boundary_curve_file'])),  # [6]
                float(b.get('vertical_datum_offset', 0.0)), # [7]
                float(b.get('smoothing_timescale', 0.0)),   # [8]
            ]

            for idx, desc in [(1, 'deck_file'),
                              (3, 'exchange_line_0'),
                              (4, 'exchange_line_1'),
                              (6, 'internal_boundary_curve_file')]:
                if not os.path.exists(row[idx]):
                    raise FileNotFoundError(
                        f"Bridge '{row[0]}': {desc} not found: {row[idx]!r}")

            # Prepend deck as a breakline and elevation source (highest priority)
            self.breakline_files.insert(0, row[1])
            self.elevation_data      = [[row[1], row[2]]] + self.elevation_data
            self.elevation_clip_range = [[float('-inf'), float('inf')]] + self.elevation_clip_range

            self.bridge_data.append(row)

    # -----------------------------------------------------------------------
    # Pumping stations
    # -----------------------------------------------------------------------

    def _parse_pumping_stations(self, stations):
        self.pumping_station_data = []
        for ps in stations:
            if not ps.get('enabled', True):
                continue

            row = [
                str(ps['label']),                           # [0]
                float(ps['pump_capacity']),                 # [1]
                float(ps['pump_rate_of_increase']),         # [2]
                float(ps['pump_rate_of_decrease']),         # [3]
                float(ps['hw_to_start_pumping']),           # [4]
                float(ps['hw_to_stop_pumping']),            # [5]
                _normpath(str(ps['basin_polygon_file'])),   # [6] — also breakline + elevation
                float(ps['basin_elevation']),               # [7] — elevation value
                _normpath(str(ps['exchange_line_0'])),      # [8]
                _normpath(str(ps['exchange_line_1'])),      # [9]
                float(ps.get('smoothing_timescale', 0.0)), # [10]
            ]

            for idx, desc in [(6, 'basin_polygon_file'),
                              (8, 'exchange_line_0'),
                              (9, 'exchange_line_1')]:
                if not os.path.exists(row[idx]):
                    raise FileNotFoundError(
                        f"Pumping station '{row[0]}': {desc} not found: {row[idx]!r}")

            # Prepend basin as a breakline and elevation source (highest priority)
            self.breakline_files.insert(0, row[6])
            self.elevation_data      = [[row[6], row[7]]] + self.elevation_data
            self.elevation_clip_range = [[float('-inf'), float('inf')]] + self.elevation_clip_range

            self.pumping_station_data.append(row)

    # -----------------------------------------------------------------------
    # Culverts  (Boyd box and Boyd pipe)
    # -----------------------------------------------------------------------

    def _parse_culverts(self, culverts, _v):
        """Parse ``[[culverts]]`` entries into ``self.culvert_data``.

        Each entry is stored as a dict so that ``setup_culverts.py`` can
        access parameters by name rather than by positional index.

        Supported ``type`` values: ``"boyd_box"`` (default), ``"boyd_pipe"``.

        Geometry is specified with either:
        * ``exchange_line_0`` / ``exchange_line_1`` — paths to CSV polyline
          files that define the upstream / downstream exchange zones, or
        * ``end_point_0`` / ``end_point_1`` — ``[x, y]`` arrays for the two
          culvert barrel ends (simpler; no polyline files required).
        """
        self.culvert_data = []
        for c in culverts:
            if not c.get('enabled', True):
                continue

            ctype = str(c.get('type', 'boyd_box'))
            if ctype not in ('boyd_box', 'boyd_pipe'):
                raise ValueError(
                    f"Culvert '{c.get('label', '?')}': unknown type {ctype!r}. "
                    f"Expected 'boyd_box' or 'boyd_pipe'.")

            label   = str(c['label'])
            sec     = f'culverts[{label!r}]'
            losses  = float(c.get('losses', 0.0))
            barrels = float(c.get('barrels', 1.0))
            blockage= float(c.get('blockage', 0.0))
            manning = float(c.get('manning', 0.013))
            eq_gap  = float(c.get('enquiry_gap', 0.2))
            apron   = float(c.get('apron', 0.1))
            smoothing = float(c.get('smoothing_timescale', 0.0))

            _v.positive(barrels,  'barrels',  sec)
            _v.positive(manning,  'manning',  sec)
            _v.non_negative(losses,   'losses',   sec)
            _v.non_negative(eq_gap,   'enquiry_gap', sec)
            _v.non_negative(apron,    'apron',    sec)
            _v.non_negative(smoothing,'smoothing_timescale', sec)
            _v.in_range(blockage, 0.0, 1.0, 'blockage', sec)

            row = {
                'label':               label,
                'type':                ctype,
                'losses':              losses,
                'barrels':             barrels,
                'blockage':            blockage,
                'z1':                  float(c.get('z1', 0.0)),
                'z2':                  float(c.get('z2', 0.0)),
                'apron':               apron,
                'manning':             manning,
                'enquiry_gap':         eq_gap,
                'smoothing_timescale': smoothing,
                'use_momentum_jet':    bool(c.get('use_momentum_jet', True)),
                'use_velocity_head':   bool(c.get('use_velocity_head', True)),
            }

            # Type-specific geometry
            if ctype == 'boyd_box':
                row['width']    = float(c['width'])
                row['height']   = float(c['height']) if 'height' in c else None
                row['diameter'] = None
                _v.positive(row['width'], 'width', sec)
                if row['height'] is not None:
                    _v.positive(row['height'], 'height', sec)
            else:
                row['diameter'] = float(c['diameter'])
                row['width']    = None
                row['height']   = None
                _v.positive(row['diameter'], 'diameter', sec)

            # Exchange lines (file paths) or end points ([x, y] pairs)
            if 'exchange_line_0' in c:
                el0 = _normpath(str(c['exchange_line_0']))
                el1 = _normpath(str(c['exchange_line_1']))
                for fpath, key in [(el0, 'exchange_line_0'),
                                   (el1, 'exchange_line_1')]:
                    if not os.path.exists(fpath):
                        raise FileNotFoundError(
                            f"Culvert '{row['label']}': {key} not found: {fpath!r}")
                row['exchange_line_0'] = el0
                row['exchange_line_1'] = el1
                row['end_point_0']     = None
                row['end_point_1']     = None
            else:
                row['exchange_line_0'] = None
                row['exchange_line_1'] = None
                row['end_point_0']     = list(c['end_point_0'])
                row['end_point_1']     = list(c['end_point_1'])

            # Optional explicit invert elevations [upstream_m, downstream_m]
            if 'invert_elevations' in c:
                row['invert_elevations'] = [float(v) for v in c['invert_elevations']]
            else:
                row['invert_elevations'] = None

            self.culvert_data.append(row)

    # -----------------------------------------------------------------------
    # Weirs  (weir / orifice with trapezoidal cross-section)
    # -----------------------------------------------------------------------

    def _parse_weirs(self, weirs, _v):
        """Parse ``[[weirs]]`` entries into ``self.weir_data``.

        Uses the ``Weir_orifice_trapezoid_operator``.  Parameters and geometry
        specification follow the same conventions as ``[[culverts]]``.
        """
        self.weir_data = []
        for w in weirs:
            if not w.get('enabled', True):
                continue

            label    = str(w['label'])
            sec      = f'weirs[{label!r}]'
            width    = float(w['width'])
            height   = float(w['height']) if 'height' in w else None
            losses   = float(w.get('losses', 0.0))
            barrels  = float(w.get('barrels', 1.0))
            blockage = float(w.get('blockage', 0.0))
            manning  = float(w.get('manning', 0.013))
            eq_gap   = float(w.get('enquiry_gap', 0.0))
            apron    = float(w.get('apron', 0.1))
            smoothing= float(w.get('smoothing_timescale', 0.0))

            _v.positive(width,   'width',   sec)
            _v.positive(barrels, 'barrels', sec)
            _v.positive(manning, 'manning', sec)
            _v.non_negative(losses,    'losses',   sec)
            _v.non_negative(eq_gap,    'enquiry_gap', sec)
            _v.non_negative(apron,     'apron',    sec)
            _v.non_negative(smoothing, 'smoothing_timescale', sec)
            _v.in_range(blockage, 0.0, 1.0, 'blockage', sec)
            if height is not None:
                _v.positive(height, 'height', sec)

            row = {
                'label':               label,
                'width':               width,
                'height':              height,
                'losses':              losses,
                'barrels':             barrels,
                'blockage':            blockage,
                'z1':                  float(w.get('z1', 0.0)),
                'z2':                  float(w.get('z2', 0.0)),
                'apron':               apron,
                'manning':             manning,
                'enquiry_gap':         eq_gap,
                'smoothing_timescale': smoothing,
                'use_momentum_jet':    bool(w.get('use_momentum_jet', True)),
                'use_velocity_head':   bool(w.get('use_velocity_head', True)),
            }

            # Exchange lines or end points
            if 'exchange_line_0' in w:
                el0 = _normpath(str(w['exchange_line_0']))
                el1 = _normpath(str(w['exchange_line_1']))
                for fpath, key in [(el0, 'exchange_line_0'),
                                   (el1, 'exchange_line_1')]:
                    if not os.path.exists(fpath):
                        raise FileNotFoundError(
                            f"Weir '{row['label']}': {key} not found: {fpath!r}")
                row['exchange_line_0'] = el0
                row['exchange_line_1'] = el1
                row['end_point_0']     = None
                row['end_point_1']     = None
            else:
                row['exchange_line_0'] = None
                row['exchange_line_1'] = None
                row['end_point_0']     = list(w['end_point_0'])
                row['end_point_1']     = list(w['end_point_1'])

            # Optional explicit invert elevations [upstream_m, downstream_m]
            if 'invert_elevations' in w:
                row['invert_elevations'] = [float(v) for v in w['invert_elevations']]
            else:
                row['invert_elevations'] = None

            self.weir_data.append(row)
