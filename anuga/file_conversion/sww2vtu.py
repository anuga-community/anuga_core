"""
Convert an ANUGA SWW file to VTU + PVD files for ParaView.

No VTK or pyvista installation required — VTU XML is written directly.

Output
------
<outdir>/<stem>_NNNN.vtu   one file per timestep
<outdir>/<stem>.pvd        ParaView collection file — open this in ParaView

Usage (command line)
--------------------
python -m anuga.file_conversion.sww2vtu myfile.sww
python -m anuga.file_conversion.sww2vtu myfile.sww --outdir ./vtu --verbose
python -m anuga.file_conversion.sww2vtu myfile.sww --z-scale 10 --absolute-coords

Usage (Python)
--------------
from anuga.file_conversion.sww2vtu import sww2vtu
sww2vtu('myfile.sww', output_dir='./vtu', verbose=True)
"""

import argparse
import base64
import os
import struct
import sys
import numpy as np

try:
    from netCDF4 import Dataset
except ImportError:
    sys.exit("netCDF4 is required: pip install netCDF4")

# VTK cell type for a linear triangle
_VTK_TRIANGLE = 5

# Quantities to extract from the SWW file (vertex-based)
_VERTEX_QUANTITIES = [
    'elevation', 'stage', 'xmomentum', 'ymomentum',
    'friction', 'height', 'xvelocity', 'yvelocity',
]


def _read_sww(path):
    """Read an SWW file and return a dict of arrays and metadata.

    Parameters
    ----------
    path : str
        Path to the SWW file.

    Returns
    -------
    dict with keys:
        x, y          : float64 arrays of node coordinates (relative to ll corner)
        tris          : int32 array of shape (n_tri, 3)
        abs_time      : float64 array of absolute simulation times (s)
        n_steps       : int
        quantities    : dict  name -> ('static', arr) | ('dynamic', arr)
                        static shape: (n_pts,)
                        dynamic shape: (n_steps, n_pts)
        xllcorner     : float
        yllcorner     : float
    """
    ds = Dataset(path, 'r')

    x = np.array(ds.variables['x'][:], dtype=np.float64)
    y = np.array(ds.variables['y'][:], dtype=np.float64)
    tris = np.array(ds.variables['volumes'][:], dtype=np.int32)

    time = np.array(ds.variables['time'][:], dtype=np.float64)
    starttime = float(getattr(ds, 'starttime', 0.0))
    abs_time = time + starttime
    n_steps = len(time)

    quantities = {}
    for name in _VERTEX_QUANTITIES:
        if name not in ds.variables:
            continue
        arr = np.array(ds.variables[name][:], dtype=np.float32)
        if arr.ndim == 1 or arr.shape[0] != n_steps:
            quantities[name] = ('static', arr)
        else:
            quantities[name] = ('dynamic', arr)

    xllcorner = float(getattr(ds, 'xllcorner', 0.0))
    yllcorner = float(getattr(ds, 'yllcorner', 0.0))

    ds.close()

    return {
        'x': x, 'y': y, 'tris': tris,
        'abs_time': abs_time, 'n_steps': n_steps,
        'quantities': quantities,
        'xllcorner': xllcorner, 'yllcorner': yllcorner,
    }


def _b64_encode(arr):
    """Encode a numpy array as base64 with a 4-byte little-endian length prefix.

    This is the format expected by VTK's binary base64 DataArray reader.
    """
    raw = arr.tobytes()
    header = struct.pack('<I', len(raw))
    return base64.b64encode(header + raw).decode('ascii')


def _write_vtu(path, x, y, z, tris, point_data):
    """Write a single VTU (VTK Unstructured Grid XML) file.

    Uses binary base64 encoding for compact output.

    Parameters
    ----------
    path : str
        Output file path.
    x, y, z : float64 arrays of shape (n_pts,)
        Node coordinates.
    tris : int32 array of shape (n_tri, 3)
        Triangle vertex indices.
    point_data : dict
        name -> float32 array of shape (n_pts,)
    """
    n_pts = len(x)
    n_cells = len(tris)

    pts = np.column_stack([x, y, z]).astype(np.float64)
    connectivity = tris.ravel().astype(np.int32)
    offsets = np.arange(3, 3 * n_cells + 1, 3, dtype=np.int32)
    cell_types = np.full(n_cells, _VTK_TRIANGLE, dtype=np.uint8)

    def da(type_str, name, components, arr):
        """Build a binary base64 DataArray element."""
        if name:
            header = (f'<DataArray type="{type_str}" Name="{name}" '
                      f'NumberOfComponents="{components}" format="binary">')
        else:
            header = (f'<DataArray type="{type_str}" '
                      f'NumberOfComponents="{components}" format="binary">')
        return [f'          {header}',
                f'            {_b64_encode(arr)}',
                '          </DataArray>']

    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" '
        'byte_order="LittleEndian" header_type="UInt32">',
        '  <UnstructuredGrid>',
        f'    <Piece NumberOfPoints="{n_pts}" NumberOfCells="{n_cells}">',
        '      <Points>',
        *da('Float64', '', 3, pts),
        '      </Points>',
        '      <Cells>',
        *da('Int32', 'connectivity', 1, connectivity),
        *da('Int32', 'offsets', 1, offsets),
        *da('UInt8', 'types', 1, cell_types),
        '      </Cells>',
        '      <PointData>',
    ]

    for name, arr in point_data.items():
        lines.extend(da('Float32', name, 1, arr.astype(np.float32)))

    lines += [
        '      </PointData>',
        '    </Piece>',
        '  </UnstructuredGrid>',
        '</VTKFile>',
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _write_pvd(path, vtu_names, abs_times):
    """Write a ParaView Data collection (PVD) file.

    Parameters
    ----------
    path : str
        Output PVD file path.
    vtu_names : list of str
        Bare filenames (no directory) of the VTU files, in time order.
    abs_times : array-like
        Absolute simulation time for each VTU file.
    """
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for t, name in zip(abs_times, vtu_names):
        lines.append(
            f'    <DataSet timestep="{t:.6f}" group="" part="0" file="{name}"/>')
    lines += [
        '  </Collection>',
        '</VTKFile>',
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def sww2vtu(sww_path, output_dir=None, z_scale=1.0,
            absolute_coords=False, verbose=False):
    """Convert an ANUGA SWW file to VTU + PVD files for ParaView.

    Parameters
    ----------
    sww_path : str
        Path to the input SWW file.
    output_dir : str or None
        Directory for output files.  Defaults to the same directory as
        the SWW file.
    z_scale : float
        Vertical exaggeration applied to the bed-elevation z coordinate.
        Use values > 1 to emphasise topography in flat terrain.
    absolute_coords : bool
        If True, add xllcorner / yllcorner to x / y so coordinates are
        in the CRS of the SWW file.  Useful when overlaying with other
        georeferenced data in ParaView.  Defaults to False (relative
        coordinates, better numerical precision for large offsets).
    verbose : bool
        Print progress messages.

    Returns
    -------
    str
        Path to the PVD collection file.
    """
    if verbose:
        print(f'Reading {sww_path}')

    data = _read_sww(sww_path)

    x = data['x']
    y = data['y']
    if absolute_coords:
        x = x + data['xllcorner']
        y = y + data['yllcorner']

    tris = data['tris']
    abs_time = data['abs_time']
    n_steps = data['n_steps']
    quantities = data['quantities']

    stem = os.path.splitext(os.path.basename(sww_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(sww_path))
    os.makedirs(output_dir, exist_ok=True)

    # Static elevation — used for z coordinate and depth calculation
    elev_static = None
    if 'elevation' in quantities and quantities['elevation'][0] == 'static':
        elev_static = quantities['elevation'][1]

    vtu_names = []

    for i in range(n_steps):
        vtu_name = f'{stem}_{i:04d}.vtu'
        vtu_path = os.path.join(output_dir, vtu_name)
        vtu_names.append(vtu_name)

        if verbose:
            print(f'  timestep {i:4d}/{n_steps - 1}  '
                  f't={abs_time[i]:.2f}s  -> {vtu_name}')

        # Collect point data for this timestep
        point_data = {}
        for name, (kind, arr) in quantities.items():
            point_data[name] = arr if kind == 'static' else arr[i]

        # Derived: depth
        stage = point_data.get('stage')
        elev = point_data.get('elevation', elev_static)
        if stage is not None and elev is not None:
            depth = np.maximum(stage - elev, 0.0).astype(np.float32)
            point_data['depth'] = depth

            # Derived: speed from momentum / depth (avoid division by zero)
            xmom = point_data.get('xmomentum')
            ymom = point_data.get('ymomentum')
            if xmom is not None and ymom is not None:
                d_safe = np.where(depth > 1e-6, depth, 1e-6)
                speed = np.sqrt((xmom / d_safe)**2 +
                                (ymom / d_safe)**2).astype(np.float32)
                speed[depth < 1e-6] = 0.0
                point_data['speed'] = speed

        # Z coordinate: bed elevation (scaled), or flat if unavailable
        if elev is not None:
            z = elev.astype(np.float64) * z_scale
        else:
            z = np.zeros_like(x)

        _write_vtu(vtu_path, x, y, z, tris, point_data)

    pvd_path = os.path.join(output_dir, f'{stem}.pvd')
    _write_pvd(pvd_path, vtu_names, abs_time)

    if verbose:
        print(f'Done.  Open in ParaView: {pvd_path}')

    return pvd_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert an ANUGA SWW file to VTU + PVD for ParaView.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('sww', help='Input SWW file')
    parser.add_argument('--outdir', default=None,
                        help='Output directory (default: same directory as SWW file)')
    parser.add_argument('--z-scale', type=float, default=1.0,
                        help='Vertical exaggeration for bed elevation')
    parser.add_argument('--absolute-coords', action='store_true',
                        help='Add xllcorner/yllcorner offset to x/y coordinates')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print progress messages')
    args = parser.parse_args()

    if not os.path.isfile(args.sww):
        sys.exit(f'Error: file not found: {args.sww}')

    sww2vtu(args.sww,
            output_dir=args.outdir,
            z_scale=args.z_scale,
            absolute_coords=args.absolute_coords,
            verbose=args.verbose)


if __name__ == '__main__':
    main()
