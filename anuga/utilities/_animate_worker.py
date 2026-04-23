"""
Worker-process helpers for parallel SWW frame generation.

IMPORTANT: do not add any matplotlib imports at module level.
worker_init() must call matplotlib.use('Agg') before pyplot is imported.
"""

_wp = None   # SWW_plotter instance — one per worker process, loaded once

_COUNTER_ATTR = {
    'depth':           '_depth_frame_count',
    'stage':           '_stage_frame_count',
    'speed':           '_speed_frame_count',
    'speed_depth':     '_speed_depth_frame_count',
    'max_depth':       '_max_depth_frame_count',
    'max_speed':       '_max_speed_frame_count',
    'max_speed_depth': '_max_speed_depth_frame_count',
    'elev':            '_elev_frame_count',
}


def worker_init(sww_path, plot_dir, min_depth, epsg):
    """Load the SWW file once per worker process with the Agg backend.

    Uses plt.switch_backend() so this works whether the worker was
    created via 'fork' (pyplot already imported) or 'spawn' (fresh process).
    """
    global _wp
    try:
        import matplotlib.pyplot as _plt
        _plt.switch_backend('agg')
    except Exception:
        import matplotlib as _mpl
        _mpl.use('Agg')
    from anuga.utilities.animate import SWW_plotter
    _wp = SWW_plotter(swwfile=sww_path, plot_dir=plot_dir, min_depth=min_depth)
    if epsg is not None:
        _wp.set_epsg(epsg)


def worker_frame(sww_frame, out_pos, save_method_name, qty,
                 dpi, vmin, vmax, cmap, basemap, alpha, basemap_provider,
                 xlim=None, ylim=None, smooth=False):
    """Render one frame to disk; returns out_pos on success."""
    global _wp
    setattr(_wp, _COUNTER_ATTR[qty], out_pos)
    save_fn = getattr(_wp, save_method_name)
    save_fn(frame=sww_frame, dpi=dpi, vmin=vmin, vmax=vmax,
            cmap=cmap, basemap=basemap, alpha=alpha,
            basemap_provider=basemap_provider,
            xlim=xlim, ylim=ylim, smooth=smooth)
    return out_pos
