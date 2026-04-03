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

        self._parse_project(cfg.get('project', {}))
        self._parse_mesh(cfg.get('mesh', {}))
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

    # -----------------------------------------------------------------------
    # Project settings
    # -----------------------------------------------------------------------

    def _parse_project(self, p):
        self.scenario       = str(p['scenario'])
        self.output_basedir = str(p['output_base_directory'])
        self.yieldstep      = float(p['yieldstep'])
        self.finaltime      = float(p['finaltime'])

        proj = p['projection_information']
        if isinstance(proj, float):
            self.projection_information = int(proj)
        elif isinstance(proj, int):
            self.projection_information = proj
        else:
            self.projection_information = str(proj)

        self.flow_algorithm = str(p['flow_algorithm'])

        self.use_local_extrapolation_and_flux_updating = bool(
            p.get('use_local_extrapolation_and_flux_updating', False))

        self.output_tif_cellsize = float(p.get('output_tif_cellsize', 50.0))

        otbp = p.get('output_tif_bounding_polygon', '')
        self.output_tif_bounding_polygon = str(otbp) if otbp else None

        self.max_quantity_update_frequency = int(
            p.get('max_quantity_update_frequency', 1))
        self.max_quantity_collection_start_time = float(
            p.get('max_quantity_collection_starttime', 0.0))

        if self.max_quantity_collection_start_time >= self.finaltime:
            raise ValueError(
                'max_quantity_collection_starttime must be < finaltime')

        self.store_vertices_uniquely        = bool(p.get('store_vertices_uniquely', False))
        self.store_elevation_every_timestep = bool(p.get('store_elevation_every_timestep', False))
        self.spatial_text_output_dir        = str(p.get('spatial_text_output_dir', 'SPATIAL_TEXT'))

        self.report_mass_conservation_statistics      = bool(p.get('report_mass_conservation_statistics', False))
        self.report_peak_velocity_statistics          = bool(p.get('report_peak_velocity_statistics', False))
        self.report_smallest_edge_timestep_statistics = bool(p.get('report_smallest_edge_timestep_statistics', False))
        self.report_operator_statistics               = bool(p.get('report_operator_statistics', False))

        # Number of OpenMP threads (1 = single-threaded; None = use OMP_NUM_THREADS env var)
        raw_omp = p.get('omp_num_threads', None)
        self.omp_num_threads = int(raw_omp) if raw_omp is not None else None

        # Multiprocessor mode: 1 = OpenMP (default), 2 = CuPy/GPU
        self.multiprocessor_mode = int(p.get('multiprocessor_mode', 1))

        # SWW output interval [seconds]; None means write every yieldstep.
        # Must be an integer multiple of yieldstep.
        raw_os = p.get('outputstep', None)
        self.outputstep = float(raw_os) if raw_os is not None else None

    # -----------------------------------------------------------------------
    # Mesh
    # -----------------------------------------------------------------------

    def _parse_mesh(self, m):
        self.use_existing_mesh_pickle       = bool(m.get('use_existing_mesh_pickle', False))
        self.bounding_polygon_and_tags_file = _normpath(str(m['bounding_polygon']))
        self.default_res                    = float(m['default_res'])

        self.interior_regions_data = [
            [_normpath(str(ir['polygon'])), float(ir['resolution'])]
            for ir in m.get('interior_regions', [])
        ]

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
