"""Implementation of tools to do with system administration made as platform independent as possible.
"""

import sys
import os
import platform

# Record Python version
major_version = int(platform.python_version_tuple()[0])
version = platform.python_version()

import hashlib


def log_to_file(filename, s, verbose=False, mode='a'):
    """Log string to file name
    """

    with open(filename, mode) as fid:
        if verbose: print(s)
        fid.write(s + '\n')


def get_user_name():
    """Get user name provide by operating system
    """

    import getpass
    user = getpass.getuser()

    return user

def get_host_name():
    """Get host name provide by operating system
    """

    if sys.platform == 'win32':
        host = os.getenv('COMPUTERNAME')
    else:
        host = os.uname()[1]


    return host


def get_version():
    """Get anuga version number as stored in anuga.__version__
    """

    import anuga
    return anuga.__version__


def get_revision_number():
    """Get the (git) sha of this repository copy.
    """
    from anuga import __git_sha__ as revision
    return revision


def get_revision_date():
    """Get the (git) revision date of this repository copy.
    """

    from anuga import __git_committed_datetime__ as revision_date
    return revision_date



def safe_crc(string):
    """64 bit safe crc computation.

    See http://docs.python.org/library/zlib.html#zlib.crc32:

        To generate the same numeric value across all Python versions
        and platforms use crc32(data) & 0xffffffff.
    """

    from zlib import crc32

    return crc32(string) & 0xffffffff


def compute_checksum(filename, max_length=2**20):
    """Compute the CRC32 checksum for specified file

    Optional parameter max_length sets the maximum number
    of bytes used to limit time used with large files.
    Default = 2**20 (1MB)
    """

    fid = open(filename, 'rb') # Use binary for portability
    crcval = safe_crc(fid.read(max_length))
    fid.close()

    return crcval


def get_anuga_pathname():
    """Get pathname of anuga install location

    Typically, this is required in unit tests depending
    on external files.

    """

    import anuga

    return os.path.dirname(anuga.__file__)


def get_pathname_from_package(package):
    """Get pathname of given package (provided as string)

    This is useful for reading files residing in the same directory as
    a particular module. Typically, this is required in unit tests depending
    on external files.

    The given module must start from a directory on the pythonpath
    and be importable using the import statement.

    Example
    path = get_pathname_from_package('anuga.utilities')

    """

    # Execute import command
    # See https://stackoverflow.com/questions/1463306/how-does-exec-work-with-locals
    exec('import %s as x' % package, globals())

    # # Get and return path
    # return x.__path__[0]
    return os.path.dirname(x.__file__)


def clean_line(str, delimiter):
    """Split a string into 'clean' fields.

    str        the string to process
    delimiter  the delimiter string to split 'line' with

    Returns a list of 'cleaned' field strings.

    Any fields that were initially zero length will be removed.
    If a field contains '\n' it isn't zero length.
    """

    return [x.strip() for x in str.strip().split(delimiter) if x != '']


################################################################################
# The following two functions are used to get around a problem with numpy and
# NetCDF files.  Previously, using Numeric, we could take a list of strings and
# convert to a Numeric array resulting in this:
#     Numeric.array(['abc', 'xy']) -> [['a', 'b', 'c'],
#                                      ['x', 'y', ' ']]
#
# However, under numpy we get:
#     numpy.array(['abc', 'xy']) -> ['abc',
#                                    'xy']
#
# And writing *strings* to a NetCDF file is problematic.
#
# The solution is to use these two routines to convert a 1-D list of strings
# to the 2-D list of chars form and back.  The 2-D form can be written to a
# NetCDF file as before.
#
# The other option, of inverting a list of tag strings into a dictionary with
# keys being the unique tag strings and the key value a list of indices of where
# the tag string was in the original list was rejected because:
#    1. It's a lot of work
#    2. We'd have to rewite the I/O code a bit (extra variables instead of one)
#    3. The code below is fast enough in an I/O scenario
################################################################################

def string_to_char(l):
    """Convert 1-D list of strings to 2-D list of chars."""

    if not l:
        return []

    if l == ['']:
        l = [' ']


    maxlen = max(len(x) for x in l)
    ll = [x.ljust(maxlen) for x in l]
    result = []
    for s in ll:
        result.append([x for x in s])
    return result



def char_to_string(ll):
    """Convert 2-D list of chars to 1-D list of strings."""

    # https://stackoverflow.com/questions/23618218/numpy-bytes-to-plain-string
    # bytes_string.decode('UTF-8')

    # We might be able to do this a bit more shorthand as we did in Python2.x
    # i.e return [''.join(x).strip() for x in ll]

    # But this works for now.

    result = []
    for i in range(len(ll)):
        x = ll[i]
        string = ''
        for j in range(len(x)):
            c = x[j]
            if type(c) == str:
                string += c
            else:
                string += c.decode()

        result.append(string.strip())

    return result


################################################################################

def get_vars_in_expression(source):
    """Get list of variable names in a python expression."""

    # https://stackoverflow.com/questions/37993137/how-do-i-detect-variables-in-a-python-eval-expression

    import ast

    variables = {}
    syntax_tree = ast.parse(source)
    for node in ast.walk(syntax_tree):
        if type(node) is ast.Name:
            variables[node.id] = 0  # Keep first one, but not duplicates

    # Only return keys
    result = list(variables.keys()) # Only return keys i.e. the variable names
    result.sort() # Sort for uniqueness
    return result



def file_length(in_file):
    """Function to return the length of a file."""

    fid = open(in_file)
    data = fid.readlines()
    fid.close()
    return len(data)


#### Memory functions
_proc_status = '/proc/%d/status' % os.getpid()
_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

def _VmB(VmKey):
    """private method"""
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except OSError:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to MB
    return (float(v[1]) * _scale[v[2]]) // (1024.0 * 1024.0)



def _get_rss_mb():
    """Return current resident set size in MB.

    Uses psutil if available, otherwise falls back to /proc/self/status on
    Linux.  Returns 0.0 if neither source is available.
    """
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except ImportError:
        rss = _VmB('VmRSS:')
        return rss  # already in MB from _VmB


def memory_stats():
    """Return a formatted string showing the current process memory usage.

    Uses the resident set size (RSS) — physical RAM currently occupied by
    the process.  Values below 1 GB are reported in MB; values at or above
    1 GB are reported in GB.

    Returns
    -------
    str
        Memory usage string, e.g. ``'mem=342MB'`` or ``'mem=1.50GB'``.

    Examples
    --------
    >>> from anuga import memory_stats
    >>> print(memory_stats())
    mem=342MB
    """
    rss_mb = _get_rss_mb()
    if rss_mb >= 1024:
        return f'mem={rss_mb / 1024:.2f}GB'
    return f'mem={rss_mb:.0f}MB'


def print_memory_stats():
    """Print the current process memory usage to stdout.

    Convenience wrapper around :func:`memory_stats`.

    Examples
    --------
    >>> from anuga import print_memory_stats
    >>> print_memory_stats()
    mem=342MB
    """
    print(memory_stats())


def quantity_memory_stats(domain):
    """Return a detailed breakdown of memory used by all quantities on *domain*.

    For each registered quantity the function reports which backing arrays are
    allocated and their individual sizes, then prints a per-quantity subtotal
    and a grand total.  Arrays that have been kept lazy (``None``) are shown
    as ``--`` with zero cost.

    Parameters
    ----------
    domain : anuga.Domain
        The simulation domain whose quantities are to be inspected.

    Returns
    -------
    str
        A multi-line table suitable for printing or logging.

    Examples
    --------
    >>> import anuga
    >>> domain = anuga.rectangular_cross_domain(10, 10)
    >>> print(anuga.quantity_memory_stats(domain))
    Quantity memory breakdown  (N=400 triangles, L=80 boundary edges)
    ...
    """
    # Abbreviated column headers (attr name on the Quantity object).
    _ARRAYS = [
        ('cntrd',  'centroid_values'),
        ('edge',   'edge_values'),
        ('vert',   '_vertex_values'),
        ('bndry',  'boundary_values'),
        ('ex_up',  'explicit_update'),
        ('si_up',  'semi_implicit_update'),
        ('ct_bk',  'centroid_backup_values'),
        ('x_grd',  '_x_gradient'),
        ('y_grd',  '_y_gradient'),
        ('phi',    '_phi'),
    ]

    # Abbreviated quantity type names.
    _TYPE_ABBREV = {
        'evolved':          'evol',
        'edge_diagnostic':  'e_diag',
        'centroid_only':    'c_only',
        'coordinate':       'coord',
    }

    N = domain.number_of_elements
    try:
        L = domain.boundary_length
    except AttributeError:
        L = 0

    col_w  = max(len(label) for label, _ in _ARRAYS) + 2   # 7
    name_w = max(len(n) for n in domain.quantities) + 1
    type_w = max(len(_TYPE_ABBREV.get(getattr(q, '_qty_type', '?'), '?'))
                 for q in domain.quantities.values()) + 1

    headers = [label for label, _ in _ARRAYS]
    header_row = (f"{'Quantity':<{name_w}} {'type':<{type_w}} " +
                  ''.join(f"{h:>{col_w}}" for h in headers) +
                  f"{'TOTAL':>{col_w}}")
    sep = '-' * len(header_row)

    lines = [
        f"Quantity memory (N={N:,} tri, L={L:,} bndry) — kB, '--'=lazy",
        f"  type: evol=evolved  e_diag=edge_diagnostic  c_only=centroid_only  coord=coordinate",
        f"  cols: cntrd=centroid  edge=edge  vert=vertex  bndry=boundary  ex_up=explicit_update",
        f"        si_up=semi_implicit_update  ct_bk=centroid_backup  x/y_grd=gradients",
        sep,
        header_row,
        sep,
    ]

    grand_total_bytes = 0

    for qty_name, qty in sorted(domain.quantities.items()):
        raw_type = getattr(qty, '_qty_type', '?')
        qty_type = _TYPE_ABBREV.get(raw_type, raw_type)
        row_bytes = 0
        cells = []

        for _label, attr in _ARRAYS:
            arr = getattr(qty, attr, None)
            if arr is None:
                cells.append('--')
            else:
                b = arr.nbytes
                row_bytes += b
                cells.append(f'{b / 1024:.1f}')

        grand_total_bytes += row_bytes
        total_str = f'{row_bytes / 1024:.1f}'
        line = (f"{qty_name:<{name_w}} {qty_type:<{type_w}} " +
                ''.join(f"{c:>{col_w}}" for c in cells) +
                f"{total_str:>{col_w}}")
        lines.append(line)

    lines.append(sep)
    grand_kb = grand_total_bytes / 1024
    grand_mb = grand_total_bytes / 1024**2
    lines.append(f"{'GRAND TOTAL':<{name_w + type_w + 1}}"
                 f"{grand_kb:>{col_w * (len(_ARRAYS) + 1)},.1f} kB = {grand_mb:.2f} MB")
    lines.append(sep)

    return '\n'.join(lines)


def print_quantity_memory_stats(domain):
    """Print a detailed breakdown of memory used by all quantities on *domain*.

    Convenience wrapper around :func:`quantity_memory_stats`.

    Parameters
    ----------
    domain : anuga.Domain
        The simulation domain whose quantities are to be inspected.

    Examples
    --------
    >>> import anuga
    >>> domain = anuga.rectangular_cross_domain(10, 10)
    >>> anuga.print_quantity_memory_stats(domain)
    Quantity memory breakdown  (N=400 triangles, L=80 boundary edges)
    ...
    """
    print(quantity_memory_stats(domain))


