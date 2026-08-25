"""
Preloaded rainfall for the Mahanadi delta runs.

The original run_model_3*.py / run_prod.py scripts re-read the rainfall CSV
inside the evolve loop: at every yieldstep (3 h) they call linecache.getline()
three times per triangle, then domain.set_quantity('Rain', ..., location=
'vertices'), then rain_operator.set_rate(Q).  On a 2.4M-element mesh across
many MPI ranks that is millions of tiny file reads per yieldstep, all hitting
the same file -- the I/O is the bottleneck, not the solver.

This module does the whole thing once, before evolve:

  1. Replay the original file-selection state machine over every yieldstep so
     the same file (or "zero", or "keep previous") is chosen for each t.
  2. Rank 0 reads each *unique* file exactly once with numpy, applies the
     unit factor, and MPI-broadcasts the global per-node vector.
  3. Each rank gathers its local nodes via domain.node_l2g and averages the
     three vertices of each triangle -> a per-centroid rate array (m/s).
  4. All schedules are stacked into one (n_slices, N_local) float64 array.

In the loop the only work is one array copy into the Rain Quantity when the
slice index changes.  No I/O, no set_quantity fit.  Works identically on the
CPU/MPI path and on the GPU path (the array-kernel rate_type is preserved and
the GPU cache is invalidated via the operator flags).
"""

import os
import numpy as np

try:
    from mpi4py import MPI
    _COMM = MPI.COMM_WORLD
except Exception:  # sequential run without mpi4py
    _COMM = None


def _read_column(path, skiprows):
    """Read one float per line, skipping `skiprows` header lines.

    Mirrors linecache.getline(path, vertex + 1 + skiprows) -> np.double(line).
    """
    try:
        return np.loadtxt(path, skiprows=skiprows, dtype=np.float64, ndmin=1)
    except ValueError:
        # fall back to a tolerant line-by-line parse (blank/odd lines)
        with open(path) as f:
            lines = f.readlines()[skiprows:]
        return np.array([float(l.split(',')[0]) for l in lines if l.strip()],
                        dtype=np.float64)


def build_schedule(yieldstep, finaltime, daily_time, sources):
    """Replay run_prod.py's rainfall selection logic for every yieldstep.

    sources: ordered list of (name, path_template, skiprows, factor, is_daily)
             path_template is a callable t -> path.
             is_daily=True  -> once chosen the file is kept until next daily_time
             is_daily=False -> re-evaluated every yieldstep (3-hourly products)

    Returns list of (t, key) where key is (path, skiprows, factor), None for
    zero rain, or 'keep' to leave the previous rate in place.
    """
    schedule = []
    rain_set_zero = True
    times = np.arange(0.0, finaltime + 0.5 * yieldstep, yieldstep)
    for t in times:
        if not rain_set_zero and len(np.where(daily_time == t)[0]) == 1:
            rain_set_zero = True
        if not rain_set_zero:
            schedule.append((t, 'keep'))
            continue
        chosen = None
        for name, tmpl, skiprows, factor, is_daily in sources:
            path = tmpl(t)
            if os.path.exists(path):
                chosen = (path, skiprows, factor)
                if is_daily:
                    rain_set_zero = False
                break
        schedule.append((t, chosen))
    return schedule


class CachedRain:
    """Preloaded per-centroid rain rates for every yieldstep.

    Usage (after distribute()):

        rain = CachedRain(domain, yieldstep, finaltime, daily_time, sources)
        Q = Quantity(domain, name='Rain', register=True)
        rain_operator = Rate_operator(domain, rate=Q, default_rate=0.0)
        for t in domain.evolve(yieldstep=yieldstep, finaltime=finaltime):
            rain.apply(t, Q, rain_operator)
    """

    def __init__(self, domain, yieldstep, finaltime, daily_time, sources,
                 verbose=True):
        self.domain = domain
        self.myid = _COMM.Get_rank() if _COMM else 0
        self.times = None
        self.slice_of_t = {}      # t -> slice index (or None for 'keep')
        self.rates = None         # (n_slices, N_local)
        self.labels = []          # per-slice description for logging
        self._last_idx = -1

        N = domain.number_of_elements
        tri = domain.triangles                       # (N, 3) local node ids
        l2g = getattr(domain, 'node_l2g', None)
        if l2g is None:                               # sequential: local == global
            l2g = np.arange(domain.number_of_nodes)
        gnodes = np.asarray(l2g)[tri]                # (N, 3) global node ids

        # --- 1. schedule (identical on every rank: same filesystem view) ---
        schedule = build_schedule(yieldstep, finaltime, daily_time, sources)

        # --- 2. unique files, read once on rank 0, broadcast ---
        unique_keys = []
        for _, key in schedule:
            if key not in (None, 'keep') and key not in unique_keys:
                unique_keys.append(key)

        slices = [np.zeros(N, dtype=np.float64)]     # slice 0 == zero rain
        self.labels = ['zero']
        key_to_idx = {None: 0}
        for key in unique_keys:
            path, skiprows, factor = key
            vec = None
            if self.myid == 0:
                vec = _read_column(path, skiprows) * factor
                if verbose:
                    print(f'[rain_cache] read {path}: {vec.size} nodes, '
                          f'max {vec.max():.3e} m/s', flush=True)
            if _COMM:
                vec = _COMM.bcast(vec, root=0)
            # --- 3. gather local vertices, average to centroid ---
            local = vec[gnodes]                       # (N, 3)
            slices.append(local.mean(axis=1))
            self.labels.append(os.path.basename(path))
            key_to_idx[key] = len(slices) - 1

        # --- 4. stack ---
        self.rates = np.ascontiguousarray(np.vstack(slices))
        for t, key in schedule:
            self.slice_of_t[float(t)] = None if key == 'keep' else key_to_idx[key]

        if self.myid == 0 and verbose:
            nz = sum(1 for k in schedule if k[1] is None)
            print(f'[rain_cache] {len(schedule)} yieldsteps, '
                  f'{len(unique_keys)} files preloaded, {nz} zero-rain steps, '
                  f'cache {self.rates.nbytes / 1e6:.1f} MB/rank', flush=True)

    def apply(self, t, Q, rain_operator):
        """Call once per yieldstep. Returns True if the rate was changed."""
        idx = self.slice_of_t.get(float(t))
        if idx is None:                               # 'keep previous'
            if self.myid == 0:
                print('Using previously set Daily Rainfall!!', flush=True)
            return False
        if idx == self._last_idx:
            return False
        self._last_idx = idx
        if self.myid == 0:
            print(f'Setting rainfall from cache: {self.labels[idx]}', flush=True)
        Q.centroid_values[:] = self.rates[idx]
        rain_operator.set_rate(rate=Q)                # keeps rate_type='quantity'
        # invalidate GPU-side copy (no-ops on the CPU path)
        rain_operator._gpu_rate_array_cache = None
        rain_operator._gpu_rate_changed = True
        return True
