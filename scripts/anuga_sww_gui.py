#!/usr/bin/env python3
"""GUI viewer for ANUGA SWW file animations.

Opens an SWW file, generates PNG frames for a chosen quantity using
SWW_plotter, then plays them as an animation.

Usage::

    anuga_sww_gui
    anuga_sww_gui --sww domain.sww
    anuga_sww_gui --sww domain.sww --qty depth
"""

import os
import glob
import sys

try:
    import tkinter as tk
    from tkinter import ttk, filedialog
except ImportError:
    sys.exit(
        "Error: tkinter is not available.\n"
        "On Debian/Ubuntu: sudo apt install python3-tk\n"
        "On conda: conda install tk"
    )

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib import image as mpimage
except ImportError as _e:
    sys.exit(
        f"Error: matplotlib is required but could not be loaded: {_e}\n"
        "Install it with: conda install matplotlib  or  pip install matplotlib"
    )
except Exception as _e:
    sys.exit(
        f"Error: could not initialise the TkAgg matplotlib backend: {_e}\n"
        "Ensure a display is available and tkinter is installed."
    )


# ------------------------------------------------------------------ #
# Quantity configuration                                              #
# ------------------------------------------------------------------ #

_QUANTITIES = ('depth', 'stage', 'speed', 'speed_depth',
               'max_depth', 'max_speed', 'max_speed_depth',
               'elev')

_QTY_DEFAULTS = {
    'depth':           dict(vmin=0.0,   vmax=20.0),
    'stage':           dict(vmin=-20.0, vmax=20.0),
    'speed':           dict(vmin=0.0,   vmax=10.0),
    'speed_depth':     dict(vmin=0.0,   vmax=20.0),
    'max_depth':       dict(vmin=0.0,   vmax=20.0),
    'max_speed':       dict(vmin=0.0,   vmax=10.0),
    'max_speed_depth': dict(vmin=0.0,   vmax=20.0),
    'elev':            dict(vmin=-20.0, vmax=100.0),
}

# Attribute on SWW_plotter used to compute auto vmin/vmax
_QTY_DATA_ATTR = {
    'depth':           'depth',
    'stage':           'stage',
    'speed':           'speed',
    'speed_depth':     'speed_depth',
    'max_depth':       'depth',
    'max_speed':       'speed',
    'max_speed_depth': 'speed_depth',
    'elev':            'elev',
}

# Method name on SWW_plotter to save a single frame
_QTY_SAVE_METHOD = {
    'depth':           'save_depth_frame',
    'stage':           'save_stage_frame',
    'speed':           'save_speed_frame',
    'speed_depth':     'save_speed_depth_frame',
    'max_depth':       'save_max_depth_frame',
    'max_speed':       'save_max_speed_frame',
    'max_speed_depth': 'save_max_speed_depth_frame',
    'elev':            'save_elev_frame',
}


# ------------------------------------------------------------------ #
# Frame helpers (shared with anuga_animate_gui convention)           #
# ------------------------------------------------------------------ #

def _find_frames(plot_dir, quantity, prefix):
    """Return sorted PNG paths matching prefix_quantity_*.png in plot_dir."""
    pattern = os.path.join(plot_dir, f'{prefix}_{quantity}_*.png')
    return sorted(glob.glob(pattern))


# ------------------------------------------------------------------ #
# GUI                                                                 #
# ------------------------------------------------------------------ #

class SWWAnimationGUI:
    """Tkinter + matplotlib GUI: open SWW -> generate frames -> animate."""

    def __init__(self, root, initial_sww=None, initial_qty=None):
        self.root = root
        self.root.title('ANUGA SWW Animator')
        self.root.minsize(860, 640)
        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        # animation state
        self._frames = []
        self._current = 0
        self._playing = False
        self._after_id = None
        self._im = None

        # generation state
        self._splotter = None
        self._cancel_flag = False
        self._gen_after_id = None
        self._executor = None
        self._futures = []
        self._gen_plot_dir = ''
        self._gen_qty = 'depth'
        self._n_to_gen = 0

        # zoom state
        self._zoom_xlim = None
        self._zoom_ylim = None
        self._zoom_rect_patch = None
        self._zoom_selector = None
        self._zoom_mode = False

        # timeseries state
        self._ts_triangle = None
        self._ts_vline = None
        self._pick_mode = False
        self._pick_cid = None
        self._pick_key_cid = None
        self._pick_overlay = None
        self._pick_text = None
        self._plot_transform = None
        self._gen_used_basemap = False
        self._last_gen_dpi = 100
        self._last_gen_vmin = 0.0
        self._last_gen_vmax = 20.0
        self._last_gen_cmap = 'viridis'
        self._last_gen_qty = 'depth'

        self._build_ui()

        if initial_qty and initial_qty in _QUANTITIES:
            self._qty_var.set(initial_qty)
        if initial_sww:
            self._sww_var.set(os.path.abspath(initial_sww))
            self._on_sww_change()

    # -------------------------------------------------------------- #
    # UI construction                                                 #
    # -------------------------------------------------------------- #

    def _build_ui(self):
        ctrl = ttk.Frame(self.root, padding=6)
        ctrl.pack(side=tk.TOP, fill=tk.X)

        # ---- Row 1: SWW file ----
        row1 = ttk.Frame(ctrl)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text='SWW file:').pack(side=tk.LEFT)
        self._sww_var = tk.StringVar()
        sww_entry = ttk.Entry(row1, textvariable=self._sww_var, width=50)
        sww_entry.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        sww_entry.bind('<Return>', lambda _e: self._on_sww_change())
        ttk.Button(row1, text='Browse...', command=self._browse_sww).pack(side=tk.LEFT)

        # ---- Row 2: output dir + quantity ----
        row2 = ttk.Frame(ctrl)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text='Output dir:').pack(side=tk.LEFT)
        self._dir_var = tk.StringVar(value='_plot')
        ttk.Entry(row2, textvariable=self._dir_var, width=30).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(row2, text='Browse...', command=self._browse_dir).pack(side=tk.LEFT)

        ttk.Label(row2, text='   Quantity:').pack(side=tk.LEFT)
        self._qty_var = tk.StringVar(value='depth')
        qty_combo = ttk.Combobox(row2, textvariable=self._qty_var,
                                  values=list(_QUANTITIES), width=12,
                                  state='readonly')
        qty_combo.pack(side=tk.LEFT, padx=4)
        qty_combo.bind('<<ComboboxSelected>>', lambda _e: self._on_qty_change())

        # ---- Row 3: vmin / vmax / auto ----
        row3 = ttk.Frame(ctrl)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text='vmin:').pack(side=tk.LEFT)
        self._vmin_var = tk.StringVar(value='0.0')
        ttk.Entry(row3, textvariable=self._vmin_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(row3, text='vmax:').pack(side=tk.LEFT, padx=(8, 0))
        self._vmax_var = tk.StringVar(value='20.0')
        ttk.Entry(row3, textvariable=self._vmax_var, width=8).pack(side=tk.LEFT, padx=2)
        self._auto_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3, text='Auto from data',
                        variable=self._auto_var,
                        command=self._on_auto_toggle).pack(side=tk.LEFT, padx=8)

        ttk.Label(row3, text='   DPI:').pack(side=tk.LEFT)
        self._dpi_var = tk.StringVar(value='100')
        ttk.Entry(row3, textvariable=self._dpi_var, width=5).pack(side=tk.LEFT, padx=2)

        ttk.Label(row3, text='   min depth:').pack(side=tk.LEFT, padx=(8, 0))
        self._mindepth_var = tk.StringVar(value='0.001')
        ttk.Entry(row3, textvariable=self._mindepth_var, width=7).pack(side=tk.LEFT, padx=2)

        ttk.Label(row3, text='   Every N:').pack(side=tk.LEFT, padx=(8, 0))
        self._stride_var = tk.StringVar(value='1')
        ttk.Spinbox(row3, from_=1, to=1000, increment=1,
                    textvariable=self._stride_var, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Label(row3, text='frames').pack(side=tk.LEFT)

        ttk.Label(row3, text='   Colormap:').pack(side=tk.LEFT, padx=(8, 0))
        self._cmap_var = tk.StringVar(value='viridis')
        _CMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis',
                  'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrRd',
                  'RdBu_r', 'coolwarm', 'seismic', 'jet', 'turbo',
                  'terrain', 'gist_earth', 'gray']
        ttk.Combobox(row3, textvariable=self._cmap_var,
                     values=_CMAPS, width=10).pack(side=tk.LEFT, padx=2)
        self._cmap_reverse_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row3, text='Reverse',
                        variable=self._cmap_reverse_var).pack(side=tk.LEFT, padx=(2, 0))

        self._basemap_var = tk.BooleanVar(value=False)
        self._basemap_chk = ttk.Checkbutton(row3, text='Basemap:',
                                             variable=self._basemap_var,
                                             command=self._on_basemap_toggle,
                                             state=tk.DISABLED)
        self._basemap_chk.pack(side=tk.LEFT, padx=(8, 0))

        from anuga.utilities.animate import BASEMAP_PROVIDERS
        self._basemap_provider_var = tk.StringVar(value='OpenStreetMap')
        self._basemap_provider_combo = ttk.Combobox(
            row3, textvariable=self._basemap_provider_var,
            values=list(BASEMAP_PROVIDERS.keys()),
            width=20, state=tk.DISABLED)
        self._basemap_provider_combo.pack(side=tk.LEFT, padx=2)

        ttk.Label(row3, text='  Alpha:').pack(side=tk.LEFT, padx=(4, 0))
        self._alpha_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(row3, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self._alpha_var,
                    format='%.2f', width=5).pack(side=tk.LEFT, padx=2)

        # ---- Row 4: SWW info + EPSG override ----
        row4 = ttk.Frame(ctrl)
        row4.pack(fill=tk.X, pady=(4, 0))
        self._sww_info_label = ttk.Label(row4, text='No SWW file loaded.',
                                          foreground='grey')
        self._sww_info_label.pack(side=tk.LEFT, padx=4)
        ttk.Label(row4, text='EPSG:').pack(side=tk.LEFT, padx=(12, 0))
        self._epsg_var = tk.StringVar(value='')
        self._epsg_entry = ttk.Entry(row4, textvariable=self._epsg_var, width=8)
        self._epsg_entry.pack(side=tk.LEFT, padx=2)
        self._epsg_entry.bind('<Return>', lambda _e: self._on_set_epsg())
        ttk.Button(row4, text='Set', width=4,
                   command=self._on_set_epsg).pack(side=tk.LEFT, padx=2)

        # ---- Row 5: generate/cancel + progress bar ----
        row5 = ttk.Frame(ctrl)
        row5.pack(fill=tk.X, pady=2)
        self._gen_btn = ttk.Button(row5, text='Generate Frames',
                                    command=self._start_generation,
                                    state=tk.DISABLED)
        self._gen_btn.pack(side=tk.LEFT, padx=(0, 2))
        self._cancel_btn = ttk.Button(row5, text='Cancel',
                                       command=self._cancel_generation,
                                       state=tk.DISABLED)
        self._cancel_btn.pack(side=tk.LEFT, padx=2)
        self._save_anim_btn = ttk.Button(row5, text='Save Animation…',
                                          command=self._save_animation,
                                          state=tk.DISABLED)
        self._save_anim_btn.pack(side=tk.RIGHT, padx=4)
        self._progress_var = tk.IntVar(value=0)
        self._progress_bar = ttk.Progressbar(row5, variable=self._progress_var,
                                              maximum=100)
        self._progress_bar.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        self._progress_label = ttk.Label(row5, text='', width=14)
        self._progress_label.pack(side=tk.LEFT)

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # ---- Row 6: animation playback controls ----
        row6 = ttk.Frame(ctrl)
        row6.pack(fill=tk.X, pady=2)

        self._play_btn = ttk.Button(row6, text='Play', width=6,
                                     command=self._toggle_play)
        self._play_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(row6, text='|<', width=3, command=self._go_first).pack(side=tk.LEFT, padx=1)
        ttk.Button(row6, text='<',  width=3, command=self._step_back).pack(side=tk.LEFT, padx=1)
        ttk.Button(row6, text='>',  width=3, command=self._step_fwd).pack(side=tk.LEFT, padx=1)
        ttk.Button(row6, text='>|', width=3, command=self._go_last).pack(side=tk.LEFT, padx=1)

        ttk.Label(row6, text='   FPS:').pack(side=tk.LEFT)
        self._fps_var = tk.DoubleVar(value=5.0)
        ttk.Spinbox(row6, from_=0.5, to=30.0, increment=0.5,
                    textvariable=self._fps_var, width=6).pack(side=tk.LEFT, padx=4)

        ttk.Separator(row6, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._pick_btn = ttk.Button(row6, text='Pick timeseries',
                                     command=self._toggle_pick,
                                     state=tk.DISABLED)
        self._pick_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(row6, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        self._zoom_btn = ttk.Button(row6, text='Set Zoom',
                                     command=self._toggle_zoom_mode,
                                     state=tk.DISABLED)
        self._zoom_btn.pack(side=tk.LEFT, padx=2)
        self._reset_zoom_btn = ttk.Button(row6, text='Reset Zoom',
                                           command=self._reset_zoom,
                                           state=tk.DISABLED)
        self._reset_zoom_btn.pack(side=tk.LEFT, padx=2)

        ttk.Button(row6, text='Help', command=self._show_help).pack(
            side=tk.RIGHT, padx=(4, 0))
        self._mesh_btn = ttk.Button(row6, text='View Mesh',
                                     command=self._show_mesh,
                                     state=tk.DISABLED)
        self._mesh_btn.pack(side=tk.RIGHT, padx=4)
        self._save_frame_btn = ttk.Button(row6, text='Save Frame',
                                           command=self._save_frame,
                                           state=tk.DISABLED)
        self._save_frame_btn.pack(side=tk.RIGHT, padx=4)
        self._frame_label = ttk.Label(row6, text='-')
        self._frame_label.pack(side=tk.RIGHT, padx=10)

        # ---- Row 7: frame slider ----
        row7 = ttk.Frame(ctrl)
        row7.pack(fill=tk.X, pady=2)
        ttk.Label(row7, text='Frame:').pack(side=tk.LEFT)
        self._slider_var = tk.IntVar(value=0)
        self._slider = ttk.Scale(row7, from_=0, to=0,
                                  variable=self._slider_var,
                                  orient=tk.HORIZONTAL,
                                  command=self._on_slider)
        self._slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # ---- status bar (packed first so it anchors to bottom) ----
        self._status_var = tk.StringVar(value='Open an SWW file to begin.')
        ttk.Label(self.root, textvariable=self._status_var,
                  relief=tk.SUNKEN, anchor=tk.W,
                  padding=(4, 1)).pack(side=tk.BOTTOM, fill=tk.X)

        # ---- timeseries panel (hidden until a point is picked) ----
        self._ts_outer = ttk.Frame(self.root)
        # not packed yet

        ts_ctrl = ttk.Frame(self._ts_outer)
        ts_ctrl.pack(fill=tk.X)
        ttk.Label(ts_ctrl, text='Timeseries:').pack(side=tk.LEFT, padx=4)
        self._ts_qty_var = tk.StringVar(value='depth')
        ts_qty_combo = ttk.Combobox(ts_ctrl, textvariable=self._ts_qty_var,
                                    values=['depth', 'stage', 'speed',
                                            'speed_depth', 'elev'],
                                    width=12, state='readonly')
        ts_qty_combo.pack(side=tk.LEFT, padx=2)
        ts_qty_combo.bind('<<ComboboxSelected>>',
                          lambda _e: self._update_timeseries())
        self._ts_info_label = ttk.Label(ts_ctrl, text='', foreground='grey')
        self._ts_info_label.pack(side=tk.LEFT, padx=8)
        ttk.Button(ts_ctrl, text='Close',
                   command=self._close_timeseries).pack(side=tk.RIGHT, padx=4)
        ttk.Button(ts_ctrl, text='Export CSV',
                   command=self._export_timeseries).pack(side=tk.RIGHT, padx=4)

        self._ts_fig, self._ts_ax = plt.subplots(figsize=(10, 1.8))
        self._ts_fig.tight_layout(pad=1.5)
        self._ts_canvas = FigureCanvasTkAgg(self._ts_fig, master=self._ts_outer)
        self._ts_canvas.draw()
        self._ts_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ---- matplotlib map canvas ----
        self._fig, self._ax = plt.subplots(figsize=(10, 6))
        self._ax.axis('off')
        self._fig.tight_layout(pad=0)

        self._canvas_frame = ttk.Frame(self.root)
        self._canvas_frame.pack(fill=tk.BOTH, expand=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self._canvas_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -------------------------------------------------------------- #
    # SWW file selection                                              #
    # -------------------------------------------------------------- #

    def _browse_sww(self):
        path = filedialog.askopenfilename(
            title='Select SWW file',
            filetypes=[('SWW files', '*.sww'), ('All files', '*.*')],
            initialdir=os.path.dirname(self._sww_var.get()) or '.')
        if path:
            self._sww_var.set(path)
            self._on_sww_change()

    def _browse_dir(self):
        d = filedialog.askdirectory(
            title='Select output directory',
            initialdir=self._dir_var.get() or '.')
        if d:
            self._dir_var.set(d)

    def _on_sww_change(self):
        sww = self._sww_var.get().strip()
        if not sww or not os.path.isfile(sww):
            self._sww_info_label.config(text='File not found.', foreground='red')
            self._gen_btn.config(state=tk.DISABLED)
            self._pick_btn.config(state=tk.DISABLED)
            self._mesh_btn.config(state=tk.DISABLED)
            self._splotter = None
            return

        self._set_status(f'Loading {os.path.basename(sww)}...')
        self.root.update_idletasks()

        try:
            self._load_splotter(sww)
        except Exception as e:
            self._sww_info_label.config(text=f'Error: {e}', foreground='red')
            self._gen_btn.config(state=tk.DISABLED)
            self._set_status(f'Failed to load {sww}')
            return

        n = len(self._splotter.time)
        t0 = self._splotter.time[0]
        t1 = self._splotter.time[-1]
        self._sww_info_label.config(
            text=f'{n} timesteps  |  t={t0:.1f} .. {t1:.1f} s',
            foreground='grey')
        self._gen_btn.config(state=tk.NORMAL)
        self._pick_btn.config(state=tk.NORMAL)
        self._mesh_btn.config(state=tk.NORMAL)
        self._set_status(f'Loaded {os.path.basename(sww)} -{n} timesteps')
        self._update_auto_limits()
        self._exit_pick_mode()
        self._close_timeseries()
        self._reset_zoom()

    def _load_splotter(self, sww):
        from anuga.utilities.animate import SWW_plotter

        self._splotter = SWW_plotter(
            swwfile=sww,
            plot_dir=None,        # suppress automatic make_plot_dir
            min_depth=float(self._mindepth_var.get()))

        self._sww_prefix = self._splotter.name  # bare basename, no extension

        # Populate the EPSG entry from the file (may be empty for old SWW files)
        epsg_val = self._splotter.epsg
        self._epsg_var.set(str(epsg_val) if epsg_val is not None else '')

        self._refresh_basemap_state()

    def _refresh_basemap_state(self):
        """Enable/disable the basemap checkbox based on current EPSG + contextily."""
        if self._splotter is None:
            return
        has_epsg = self._splotter.epsg is not None
        has_contextily = False
        if has_epsg:
            try:
                import contextily  # noqa: F401
                has_contextily = True
            except ImportError:
                pass
        if has_epsg and has_contextily:
            self._basemap_chk.config(state=tk.NORMAL)
        elif has_epsg:
            self._basemap_chk.config(state=tk.DISABLED)
            self._basemap_var.set(False)
            self._basemap_provider_combo.config(state=tk.DISABLED)
        else:
            self._basemap_chk.config(state=tk.DISABLED)
            self._basemap_var.set(False)
            self._basemap_provider_combo.config(state=tk.DISABLED)

    def _on_set_epsg(self):
        """Apply the EPSG code typed in the entry field to the loaded SWW_plotter."""
        if self._splotter is None:
            self._set_status('Load an SWW file first.')
            return
        raw = self._epsg_var.get().strip()
        if raw == '':
            self._splotter.set_epsg(None)
            self._refresh_basemap_state()
            self._set_status('EPSG cleared — basemap disabled.')
            return
        try:
            code = int(raw)
        except ValueError:
            self._set_status(f'Invalid EPSG code "{raw}" — enter an integer, e.g. 32756.')
            return
        self._splotter.set_epsg(code)
        self._refresh_basemap_state()
        has_contextily = False
        try:
            import contextily  # noqa: F401
            has_contextily = True
        except ImportError:
            pass
        if has_contextily:
            self._set_status(f'EPSG set to {code} — basemap enabled.')
        else:
            self._set_status(
                f'EPSG set to {code} — install contextily for basemap support.')

    def _on_basemap_toggle(self):
        if self._basemap_var.get():
            self._alpha_var.set(0.6)
            self._basemap_provider_combo.config(state='readonly')
        else:
            self._alpha_var.set(1.0)
            self._basemap_provider_combo.config(state=tk.DISABLED)

    def _on_qty_change(self):
        qty = self._qty_var.get()
        defaults = _QTY_DEFAULTS[qty]
        self._vmin_var.set(str(defaults['vmin']))
        self._vmax_var.set(str(defaults['vmax']))
        if self._auto_var.get():
            self._update_auto_limits()

    def _on_auto_toggle(self):
        if self._auto_var.get():
            self._update_auto_limits()

    def _update_auto_limits(self):
        if self._splotter is None or not self._auto_var.get():
            return
        qty = self._qty_var.get()
        attr = _QTY_DATA_ATTR[qty]
        data = getattr(self._splotter, attr)
        self._vmin_var.set(f'{float(data.min()):.4g}')
        self._vmax_var.set(f'{float(data.max()):.4g}')

    def _is_static_qty(self, qty):
        """Return True if qty produces a single frame (not time-animated).

        max_* quantities always produce one frame.  elev produces one frame
        when elevation is static (1-D array); it is animated when the SWW
        contains time-varying elevation (2-D array, e.g. from an erosion run).
        """
        if qty.startswith('max_'):
            return True
        if qty == 'elev' and self._splotter is not None:
            return self._splotter.elev.ndim == 1
        return False

    # -------------------------------------------------------------- #
    # Frame generation                                                #
    # -------------------------------------------------------------- #

    def _start_generation(self):
        if self._splotter is None:
            return

        # Stop playback and discard frame list before deleting files so that
        # _advance cannot try to read a file that is about to be removed.
        self._stop_playback()
        self._frames = []

        # resolve output directory
        plot_dir = self._dir_var.get().strip() or '_plot'
        plot_dir = os.path.abspath(plot_dir)
        os.makedirs(plot_dir, exist_ok=True)

        qty   = self._qty_var.get()

        # Remove stale frames from a previous generation
        for old_file in _find_frames(plot_dir, qty, self._sww_prefix):
            try:
                os.remove(old_file)
            except OSError:
                pass

        try:
            vmin   = float(self._vmin_var.get())
            vmax   = float(self._vmax_var.get())
            dpi    = int(self._dpi_var.get())
            stride = max(1, int(self._stride_var.get()))
        except ValueError as e:
            self._set_status(f'Invalid parameter: {e}')
            return

        cmap = self._cmap_var.get().strip() or 'viridis'
        if self._cmap_reverse_var.get() and not cmap.endswith('_r'):
            cmap = cmap + '_r'
        basemap = self._basemap_var.get()
        alpha   = max(0.0, min(1.0, self._alpha_var.get()))

        from anuga.utilities.animate import BASEMAP_PROVIDERS, BASEMAP_DEFAULT
        provider_label = self._basemap_provider_var.get()
        basemap_provider = BASEMAP_PROVIDERS.get(provider_label, BASEMAP_DEFAULT)

        n_total = len(self._splotter.time)
        if self._is_static_qty(qty):
            # Static quantities (max_*, static elev) collapse to one image
            sww_frames = [0]
            stride_msg = ''
        else:
            sww_frames = list(range(0, n_total, stride))
            stride_msg = f' (every {stride})' if stride > 1 else ''
        n_to_gen = len(sww_frames)
        self._progress_bar.config(maximum=n_to_gen)
        self._progress_var.set(0)
        self._progress_label.config(text=f'0 / {n_to_gen}')
        self._gen_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._cancel_flag = False
        self._set_status(
            f'Generating {n_to_gen} {qty} frames{stride_msg}...')
        self.root.update_idletasks()

        # Reset plotter frame counters and re-point it at the output dir
        self._splotter.plot_dir = plot_dir
        for attr in ('_depth_frame_count', '_stage_frame_count',
                     '_speed_frame_count', '_speed_depth_frame_count',
                     '_max_depth_frame_count', '_max_speed_frame_count',
                     '_max_speed_depth_frame_count', '_elev_frame_count'):
            setattr(self._splotter, attr, 0)
        # Close any figures cached from a previous generation run
        self._splotter._clear_figure_cache()

        self._gen_used_basemap = basemap
        self._last_gen_dpi = dpi
        self._last_gen_vmin = vmin
        self._last_gen_vmax = vmax
        self._last_gen_cmap = cmap
        self._last_gen_qty = qty

        # Single frame (max_* quantities) always runs sequentially.
        # For multi-frame quantities, attempt parallel generation and fall
        # back to sequential if multiprocessing is unavailable.
        if n_to_gen <= 1:
            save_method = getattr(self._splotter, _QTY_SAVE_METHOD[qty])
            self._generate_next_frame(0, sww_frames, save_method, dpi, vmin, vmax,
                                      plot_dir, qty, cmap, basemap, alpha,
                                      basemap_provider)
        else:
            try:
                self._start_parallel_generation(
                    sww_frames, plot_dir, qty, dpi, vmin, vmax,
                    cmap, basemap, alpha, basemap_provider)
            except Exception as e:
                import traceback
                print(f'[anuga_animate] parallel init failed: {e}',
                      file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                self._set_status(
                    f'Parallel init failed ({e}); falling back to sequential.')
                save_method = getattr(self._splotter, _QTY_SAVE_METHOD[qty])
                self._generate_next_frame(0, sww_frames, save_method, dpi, vmin,
                                          vmax, plot_dir, qty, cmap, basemap,
                                          alpha, basemap_provider)

    def _generate_next_frame(self, pos, sww_frames, save_method,
                              dpi, vmin, vmax, plot_dir, qty,
                              cmap='viridis', basemap=False, alpha=1.0,
                              basemap_provider='OpenStreetMap.Mapnik'):
        n_to_gen  = len(sww_frames)
        sww_frame = sww_frames[pos]

        if self._cancel_flag:
            self._set_status(f'Generation cancelled after {pos} frames.')
            self._gen_btn.config(state=tk.NORMAL)
            self._cancel_btn.config(state=tk.DISABLED)
            return
        try:
            save_method(frame=sww_frame, dpi=dpi, vmin=vmin, vmax=vmax,
                        cmap=cmap, basemap=basemap, alpha=alpha,
                        basemap_provider=basemap_provider,
                        xlim=self._zoom_xlim, ylim=self._zoom_ylim)
        except Exception as e:
            self._set_status(f'Error generating frame {sww_frame}: {e}')
            self._gen_btn.config(state=tk.NORMAL)
            self._cancel_btn.config(state=tk.DISABLED)
            return

        self._progress_var.set(pos + 1)
        self._progress_label.config(text=f'{pos + 1} / {n_to_gen}')
        self.root.update_idletasks()

        if pos + 1 < n_to_gen:
            self._gen_after_id = self.root.after(
                1, lambda: self._generate_next_frame(
                    pos + 1, sww_frames, save_method,
                    dpi, vmin, vmax, plot_dir, qty, cmap, basemap, alpha,
                    basemap_provider))
        else:
            self._on_generation_done(plot_dir, qty, n_to_gen)

    def _start_parallel_generation(self, sww_frames, plot_dir, qty,
                                     dpi, vmin, vmax, cmap, basemap,
                                     alpha, basemap_provider):
        """Spawn a ProcessPoolExecutor and submit all frames."""
        import multiprocessing as mp
        import platform
        from concurrent.futures import ProcessPoolExecutor
        from anuga.utilities._animate_worker import worker_init, worker_frame

        sww_path  = self._sww_var.get()
        min_depth = float(self._mindepth_var.get())
        epsg      = self._splotter.epsg
        n_workers = max(1, min(os.cpu_count() or 4, len(sww_frames), 4))

        # 'fork' on Linux: workers inherit parent memory, no reimport overhead.
        # 'spawn' on Windows/macOS: safer for GUI apps.
        ctx_name = 'fork' if platform.system() == 'Linux' else 'spawn'
        ctx = mp.get_context(ctx_name)
        self._executor = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=worker_init,
            initargs=(sww_path, plot_dir, min_depth, epsg))

        save_name = _QTY_SAVE_METHOD[qty]
        self._futures = [
            self._executor.submit(
                worker_frame,
                frame, pos, save_name, qty,
                dpi, vmin, vmax, cmap, basemap, alpha, basemap_provider,
                self._zoom_xlim, self._zoom_ylim)
            for pos, frame in enumerate(sww_frames)
        ]
        self._gen_plot_dir = plot_dir
        self._gen_qty      = qty
        self._n_to_gen     = len(sww_frames)
        self._set_status(
            f'Generating {self._n_to_gen} {qty} frames '
            f'({n_workers} workers)...')
        self._poll_generation()

    def _poll_generation(self):
        """Check parallel futures every 200 ms; update the progress bar."""
        if self._cancel_flag:
            # _cancel_generation already shut down the executor.
            return

        n_done = sum(1 for f in self._futures if f.done())
        self._progress_var.set(n_done)
        self._progress_label.config(text=f'{n_done} / {self._n_to_gen}')
        self.root.update_idletasks()

        # Surface first worker exception immediately.
        for fut in self._futures:
            if fut.done() and fut.exception() is not None:
                exc = fut.exception()
                for f in self._futures:
                    f.cancel()
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
                self._futures = []
                self._gen_btn.config(state=tk.NORMAL)
                self._cancel_btn.config(state=tk.DISABLED)
                self._set_status(f'Frame generation error: {exc}')
                return

        if n_done == self._n_to_gen:
            self._executor.shutdown(wait=False)
            self._executor = None
            self._futures = []
            self._on_generation_done(
                self._gen_plot_dir, self._gen_qty, self._n_to_gen)
        else:
            self._gen_after_id = self.root.after(200, self._poll_generation)

    def _cancel_generation(self):
        self._cancel_flag = True
        if self._gen_after_id is not None:
            self.root.after_cancel(self._gen_after_id)
            self._gen_after_id = None
        if self._executor is not None:
            for fut in self._futures:
                fut.cancel()
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
            self._futures = []
        self._gen_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)
        self._set_status('Generation cancelled.')

    def _on_generation_done(self, plot_dir, qty, n_frames):
        self._progress_var.set(n_frames)
        self._progress_label.config(text=f'{n_frames} / {n_frames}')
        self._gen_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)
        prefix = self._sww_prefix
        self._set_status(
            f'Done -{n_frames} frames saved to {plot_dir}')
        self._compute_plot_transform(self._last_gen_dpi)
        self._load_frames(plot_dir, qty, prefix)

    def _compute_plot_transform(self, dpi):
        """Render frame 0 off-screen (Agg) with the same params used to generate
        frames so that the resulting axes position/limits exactly match the PNGs."""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        sp = self._splotter
        if sp is None:
            self._plot_transform = None
            return
        try:
            import numpy as np
            # For max quantities use the actual max; others use frame 0
            if self._last_gen_qty.startswith('max_'):
                depth = np.max(sp.depth, axis=0)
            else:
                depth = sp.depth[0, :]
            try:
                elev = sp.elev[0, :]
            except (IndexError, TypeError):
                elev = sp.elev

            vmin = self._last_gen_vmin
            vmax = self._last_gen_vmax
            cmap = self._last_gen_cmap
            use_basemap = self._gen_used_basemap and (sp.epsg is not None)
            triang = sp.triang_abs if use_basemap else sp.triang

            fig = Figure(figsize=(10, 6), dpi=dpi)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            if not use_basemap:
                triang.set_mask(depth > sp.min_depth)
                ax.tripcolor(triang, facecolors=elev, cmap='Greys_r')
            triang.set_mask(depth < sp.min_depth)
            im = ax.tripcolor(triang, facecolors=depth, cmap=cmap,
                              vmin=vmin, vmax=vmax)
            triang.set_mask(None)
            ax.set_aspect('equal')
            ax.set_xlabel('Easting (m)')
            ax.set_ylabel('Northing (m)')
            if self._zoom_xlim is not None:
                ax.set_xlim(self._zoom_xlim)
            if self._zoom_ylim is not None:
                ax.set_ylim(self._zoom_ylim)
            fig.colorbar(im, ax=ax)

            if use_basemap:
                from anuga.utilities.animate import _add_basemap, BASEMAP_PROVIDERS
                provider_label = self._basemap_provider_var.get()
                provider_str = BASEMAP_PROVIDERS.get(
                    provider_label, 'OpenStreetMap.Mapnik')
                _add_basemap(ax, sp.epsg, provider_str, cache=sp._basemap_cache)

            fig.canvas.draw()

            self._plot_transform = dict(
                pos=ax.get_position(),
                xlim=ax.get_xlim(),
                ylim=ax.get_ylim(),
                W=int(fig.get_figwidth() * fig.get_dpi()),
                H=int(fig.get_figheight() * fig.get_dpi()),
                use_basemap=use_basemap,
            )
        except Exception:
            self._plot_transform = None

    # -------------------------------------------------------------- #
    # Frame loading and display                                       #
    # -------------------------------------------------------------- #

    def _load_frames(self, plot_dir, qty, prefix):
        self._stop_playback()
        self._frames = _find_frames(plot_dir, qty, prefix)
        n = len(self._frames)
        if n == 0:
            self._set_status(f'No frames found for {prefix}_{qty}_*.png in {plot_dir}')
            return
        self._slider.configure(to=n - 1)
        self._slider_var.set(0)
        self._current = 0
        # Remove any zoom rectangle — new frames already reflect the zoom
        self._remove_zoom_patch()
        self._show_frame(0)
        self._save_frame_btn.config(state=tk.NORMAL)
        self._save_anim_btn.config(state=tk.NORMAL)
        self._zoom_btn.config(state=tk.NORMAL)
        self._set_status(f'Loaded {n} frames  |  {plot_dir}')

    def _show_frame(self, idx):
        if not self._frames:
            return
        idx = max(0, min(idx, len(self._frames) - 1))
        self._current = idx
        self._slider_var.set(idx)
        self._frame_label.config(text=f'Frame {idx + 1} / {len(self._frames)}')

        try:
            img = mpimage.imread(self._frames[idx])
        except FileNotFoundError:
            self._set_status('Frame file missing — click Generate Frames to rebuild.')
            self._stop_playback()
            return
        if self._im is None:
            self._im = self._ax.imshow(img, aspect='equal')
            self._im.set_extent([0, img.shape[1], img.shape[0], 0])
        else:
            self._im.set_data(img)
            self._im.set_extent([0, img.shape[1], img.shape[0], 0])
        self._update_pick_overlay()
        self._canvas.draw_idle()
        self._update_ts_cursor()

    def _on_slider(self, val):
        try:
            idx = int(float(val))
        except ValueError:
            return
        if idx != self._current:
            self._stop_playback()
            self._show_frame(idx)

    # -------------------------------------------------------------- #
    # Playback controls                                               #
    # -------------------------------------------------------------- #

    def _toggle_play(self):
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if not self._frames:
            return
        self._playing = True
        self._play_btn.config(text='Pause')
        self._schedule_next()

    def _stop_playback(self):
        self._playing = False
        self._play_btn.config(text='Play')
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _schedule_next(self):
        if not self._playing:
            return
        try:
            fps = max(0.1, self._fps_var.get())
        except tk.TclError:
            fps = 5.0
        self._after_id = self.root.after(int(1000 / fps), self._advance)

    def _advance(self):
        if not self._playing:
            return
        next_idx = (self._current + 1) % len(self._frames)
        self._show_frame(next_idx)
        self._schedule_next()

    def _step_fwd(self):
        self._stop_playback()
        self._show_frame(self._current + 1)

    def _step_back(self):
        self._stop_playback()
        self._show_frame(self._current - 1)

    def _go_first(self):
        self._stop_playback()
        self._show_frame(0)

    def _go_last(self):
        self._stop_playback()
        self._show_frame(len(self._frames) - 1)

    # -------------------------------------------------------------- #
    # Timeseries                                                      #
    # -------------------------------------------------------------- #

    def _toggle_pick(self):
        if self._pick_mode:
            self._exit_pick_mode()
        else:
            self._enter_pick_mode()

    def _enter_pick_mode(self):
        """Enter pick mode: keep the current animation frame as-is, overlay an
        instruction banner, and connect click events.  The frame image does not
        change size or appearance."""
        if self._splotter is None or not self._frames:
            return
        if self._plot_transform is None:
            self._set_status('Computing pick transform…')
            self.root.update_idletasks()
            self._compute_plot_transform(self._last_gen_dpi)
        if self._plot_transform is None:
            self._set_status('Pick mode unavailable: transform could not be computed')
            return

        # Overlay a semi-transparent instruction banner using axes-fraction coords
        # so it is independent of image size.
        self._pick_text = self._ax.text(
            0.5, 0.02,
            'Click to pick a timeseries point  |  Esc to cancel',
            ha='center', va='bottom', fontsize=9,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.65),
            zorder=20,
            transform=self._ax.transAxes)

        self._canvas.draw_idle()
        self._canvas.get_tk_widget().config(cursor='crosshair')
        self._canvas.get_tk_widget().focus_set()  # needed for key_press_event (Esc)

        self._pick_cid = self._canvas.mpl_connect(
            'button_press_event', self._on_pick_click)
        self._pick_key_cid = self._canvas.mpl_connect(
            'key_press_event', self._on_pick_key)

        self._pick_mode = True
        self._pick_btn.config(text='Cancel pick')

    def _on_pick_click(self, event):
        if event.inaxes != self._ax or event.xdata is None:
            return
        import numpy as np
        sp = self._splotter

        xmesh, ymesh = self._imshow_to_mesh(event.xdata, event.ydata)

        # Centroids are stored in relative coords; for basemap the xlim/ylim
        # from the transform already include the xllcorner/yllcorner offset.
        use_basemap = self._gen_used_basemap and (sp.epsg is not None)
        if use_basemap:
            xc = sp.xc + sp.xllcorner
            yc = sp.yc + sp.yllcorner
        else:
            xc, yc = sp.xc, sp.yc

        dist = np.sqrt((xc - xmesh)**2 + (yc - ymesh)**2)
        self._ts_triangle = int(dist.argmin())
        # Stay in pick mode so the user can keep clicking new points.
        # _update_timeseries may pack the ts panel (resize), so force a
        # synchronous draw afterwards to ensure the star is visible.
        self._update_timeseries()
        self._update_pick_overlay()
        self._canvas.draw()

    def _on_pick_key(self, event):
        if event.key == 'escape':
            self._exit_pick_mode()

    def _exit_pick_mode(self):
        """Disconnect pick events and remove the instruction banner."""
        if not self._pick_mode:
            return
        for cid in (self._pick_cid, self._pick_key_cid):
            if cid is not None:
                self._canvas.mpl_disconnect(cid)
        self._pick_cid = None
        self._pick_key_cid = None
        self._pick_mode = False
        self._pick_btn.config(text='Pick timeseries')
        self._canvas.get_tk_widget().config(cursor='')
        if self._pick_text is not None:
            try:
                self._pick_text.remove()
            except Exception:
                pass
            self._pick_text = None
        # Re-show the current frame: reapplies the correct imshow extent/limits
        # before adding the pick overlay (plain draw_idle is not enough because
        # ax.plot() in _update_pick_overlay can corrupt the axes limits).
        if self._frames:
            self._show_frame(self._current)
        else:
            self._canvas.draw_idle()

    def _update_timeseries(self):
        """Plot quantity vs time for the picked centroid."""
        if self._ts_triangle is None or self._splotter is None:
            return
        import numpy as np

        sp  = self._splotter
        qty = self._ts_qty_var.get()
        tri = self._ts_triangle

        import numpy as _np

        data_map = {
            'depth':       sp.depth,
            'stage':       sp.stage,
            'speed':       sp.speed,
            'speed_depth': sp.speed_depth,
        }
        ylabel_map = {
            'depth':       'Depth (m)',
            'stage':       'Stage (m)',
            'speed':       'Speed (m/s)',
            'speed_depth': 'Speed×Depth (m²/s)',
            'elev':        'Elevation (m)',
        }

        if qty == 'elev':
            if sp.elev.ndim == 2:
                y = sp.elev[:, tri]
            else:
                y = _np.full(len(sp.time), sp.elev[tri])
        else:
            y = data_map[qty][:, tri]
        t = sp.time

        self._ts_ax.cla()
        self._ts_ax.plot(t, y, color='steelblue', linewidth=1.2)
        self._ts_ax.set_xlabel('Time (s)')
        self._ts_ax.set_ylabel(ylabel_map[qty])
        self._ts_ax.set_xlim(t[0], t[-1])
        self._ts_ax.grid(True, linestyle=':', alpha=0.5)

        # Vertical cursor at current animation time
        current_time = sp.time[0]
        if self._frames and self._current < len(self._frames):
            # map current PNG frame index back to a SWW time index
            # frames were generated at stride intervals; approximate by
            # scaling through the stored time array
            frac = self._current / max(len(self._frames) - 1, 1)
            ts_idx = int(round(frac * (len(t) - 1)))
            current_time = t[ts_idx]

        self._ts_vline = self._ts_ax.axvline(current_time,
                                              color='red', linewidth=1.0,
                                              linestyle='--')
        self._ts_fig.tight_layout(pad=1.5)
        self._ts_canvas.draw_idle()

        xc = sp.xc[tri] + sp.xllcorner
        yc = sp.yc[tri] + sp.yllcorner
        self._ts_info_label.config(
            text=f'Triangle {tri}  |  x={xc:.1f}  y={yc:.1f}')

        # Show the panel if not already visible.
        # Pack it before the canvas frame so it takes space from the bottom.
        if not self._ts_outer.winfo_ismapped():
            self._ts_outer.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False,
                                before=self._canvas_frame)

    def _export_timeseries(self):
        """Save the current timeseries to a CSV file chosen by the user."""
        if self._ts_triangle is None or self._splotter is None:
            return
        import csv
        from tkinter import filedialog

        sp  = self._splotter
        qty = self._ts_qty_var.get()
        tri = self._ts_triangle
        t   = sp.time

        import numpy as _np2

        data_map = {
            'depth':       sp.depth,
            'stage':       sp.stage,
            'speed':       sp.speed,
            'speed_depth': sp.speed_depth,
        }
        if qty == 'elev':
            if sp.elev.ndim == 2:
                y = sp.elev[:, tri]
            else:
                y = _np2.full(len(t), sp.elev[tri])
        else:
            y = data_map[qty][:, tri]

        xc = sp.xc[tri] + sp.xllcorner
        yc = sp.yc[tri] + sp.yllcorner

        default_name = f'{self._sww_prefix}_{qty}_tri{tri}.csv'
        path = filedialog.asksaveasfilename(
            title='Export timeseries',
            initialfile=default_name,
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if not path:
            return

        qty_label = {
            'depth':       'depth_m',
            'stage':       'stage_m',
            'speed':       'speed_m_s',
            'speed_depth': 'speed_depth_m2_s',
            'elev':        'elevation_m',
        }[qty]

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'# SWW file: {self._sww_prefix}',
                             f'triangle: {tri}',
                             f'x: {xc:.3f}', f'y: {yc:.3f}'])
            writer.writerow(['time_s', qty_label])
            for ti, yi in zip(t, y):
                writer.writerow([f'{ti:.6g}', f'{yi:.6g}'])

        self._set_status(f'Exported {len(t)} rows → {path}')

    def _update_ts_cursor(self):
        """Move the vertical cursor line to the current animation time."""
        if self._ts_vline is None or self._splotter is None:
            return
        t = self._splotter.time
        if len(self._frames) > 1:
            frac = self._current / (len(self._frames) - 1)
        else:
            frac = 0.0
        ts_idx = int(round(frac * (len(t) - 1)))
        self._ts_vline.set_xdata([t[ts_idx], t[ts_idx]])
        self._ts_canvas.draw_idle()

    def _close_timeseries(self):
        """Hide the timeseries panel and reset state."""
        self._ts_triangle = None
        self._ts_vline = None
        self._ts_ax.cla()
        self._ts_canvas.draw_idle()
        if self._ts_outer.winfo_ismapped():
            self._ts_outer.pack_forget()
        self._remove_pick_overlay()
        self._canvas.draw_idle()

    def _remove_pick_overlay(self):
        """Remove the centroid marker from the animation canvas."""
        if self._pick_overlay is not None:
            try:
                self._pick_overlay.remove()
            except Exception:
                pass
            self._pick_overlay = None

    def _update_pick_overlay(self):
        """Add/update the centroid marker on the animation PNG (no draw_idle call)."""
        self._remove_pick_overlay()
        if (self._ts_triangle is None or self._plot_transform is None
                or not self._frames):
            return

        sp = self._splotter
        tri = self._ts_triangle
        pt = self._plot_transform

        # Centroid in the coordinate system used to generate frames.
        # _compute_plot_transform renders with triang_abs when basemap is used,
        # so pt['xlim'/'ylim'] are already in absolute coords for basemap and
        # relative coords for non-basemap — no further offset needed.
        use_basemap = self._gen_used_basemap and (sp.epsg is not None)
        if use_basemap:
            xd = sp.xc[tri] + sp.xllcorner
            yd = sp.yc[tri] + sp.yllcorner
        else:
            xd, yd = sp.xc[tri], sp.yc[tri]
        xlim, ylim = pt['xlim'], pt['ylim']

        pos = pt['pos']
        W, H = pt['W'], pt['H']

        xfrac = (xd - xlim[0]) / (xlim[1] - xlim[0])
        yfrac = (yd - ylim[0]) / (ylim[1] - ylim[0])

        # Convert figure-fraction position to image pixel coords,
        # then map to axes data coords (axes x = col, axes y = row for our extent).
        ax_x = (pos.x0 + pos.width * xfrac) * W
        ax_y = (1.0 - (pos.y0 + pos.height * yfrac)) * H

        self._pick_overlay, = self._ax.plot(
            ax_x, ax_y, 'r*', markersize=8, zorder=10,
            markeredgecolor='white', markeredgewidth=0.5,
            scalex=False, scaley=False)

    # -------------------------------------------------------------- #
    # Zoom                                                            #
    # -------------------------------------------------------------- #

    def _toggle_zoom_mode(self):
        if self._zoom_mode:
            self._exit_zoom_mode()
        else:
            self._enter_zoom_mode()

    def _enter_zoom_mode(self):
        """Activate rubber-band selection mode for choosing a zoom region."""
        if self._plot_transform is None:
            self._set_status('Generate frames first to enable zoom.')
            return
        from matplotlib.widgets import RectangleSelector
        self._zoom_selector = RectangleSelector(
            self._ax, self._on_zoom_select,
            useblit=False,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=False)
        self._zoom_mode = True
        self._zoom_btn.config(text='Cancel Zoom')
        self._canvas.get_tk_widget().config(cursor='crosshair')
        self._set_status('Drag a rectangle on the image to set the zoom region.')

    def _exit_zoom_mode(self):
        if self._zoom_selector is not None:
            self._zoom_selector.set_active(False)
            self._zoom_selector = None
        self._zoom_mode = False
        self._zoom_btn.config(text='Set Zoom')
        self._canvas.get_tk_widget().config(cursor='')

    def _on_zoom_select(self, eclick, erelease):
        """Convert the rubber-band selection to mesh coordinates and store."""
        pt = self._plot_transform
        if pt is None:
            return
        x1, y1 = self._imshow_to_mesh(eclick.xdata,   eclick.ydata)
        x2, y2 = self._imshow_to_mesh(erelease.xdata, erelease.ydata)
        self._zoom_xlim = (min(x1, x2), max(x1, x2))
        self._zoom_ylim = (min(y1, y2), max(y1, y2))
        self._exit_zoom_mode()
        self._draw_zoom_patch()
        self._reset_zoom_btn.config(state=tk.NORMAL)
        self._set_status(
            f'Zoom set — x: {self._zoom_xlim[0]:.1f}–{self._zoom_xlim[1]:.1f}  '
            f'y: {self._zoom_ylim[0]:.1f}–{self._zoom_ylim[1]:.1f}  '
            '— click Generate Frames to apply.')

    def _draw_zoom_patch(self):
        """Draw a yellow rectangle on the animation canvas showing the zoom region."""
        self._remove_zoom_patch()
        if self._zoom_xlim is None or self._plot_transform is None:
            return
        from matplotlib.patches import Rectangle
        px0, py0 = self._mesh_to_imshow(self._zoom_xlim[0], self._zoom_ylim[0])
        px1, py1 = self._mesh_to_imshow(self._zoom_xlim[1], self._zoom_ylim[1])
        x = min(px0, px1)
        y = min(py0, py1)
        w = abs(px1 - px0)
        h = abs(py1 - py0)
        self._zoom_rect_patch = Rectangle(
            (x, y), w, h,
            linewidth=2, edgecolor='yellow', facecolor='yellow',
            alpha=0.15, zorder=12)
        self._ax.add_patch(self._zoom_rect_patch)
        self._canvas.draw_idle()

    def _remove_zoom_patch(self):
        if self._zoom_rect_patch is not None:
            try:
                self._zoom_rect_patch.remove()
            except Exception:
                pass
            self._zoom_rect_patch = None
        self._canvas.draw_idle()

    def _reset_zoom(self):
        """Clear the zoom region and remove the overlay patch."""
        self._exit_zoom_mode()
        self._zoom_xlim = None
        self._zoom_ylim = None
        self._remove_zoom_patch()
        self._reset_zoom_btn.config(state=tk.DISABLED)
        self._set_status('Zoom reset — full extent will be used for generation.')

    def _imshow_to_mesh(self, px, py):
        """Convert imshow pixel coordinates to mesh/absolute coordinates."""
        pt = self._plot_transform
        pos = pt['pos']
        xlim, ylim = pt['xlim'], pt['ylim']
        W, H = pt['W'], pt['H']
        xfrac = (px / W - pos.x0) / pos.width
        yfrac = (1.0 - py / H - pos.y0) / pos.height
        mx = xlim[0] + xfrac * (xlim[1] - xlim[0])
        my = ylim[0] + yfrac * (ylim[1] - ylim[0])
        return mx, my

    def _mesh_to_imshow(self, mx, my):
        """Convert mesh/absolute coordinates to imshow pixel coordinates."""
        pt = self._plot_transform
        pos = pt['pos']
        xlim, ylim = pt['xlim'], pt['ylim']
        W, H = pt['W'], pt['H']
        xfrac = (mx - xlim[0]) / (xlim[1] - xlim[0])
        yfrac = (my - ylim[0]) / (ylim[1] - ylim[0])
        px = (pos.x0 + pos.width  * xfrac) * W
        py = (1.0 - (pos.y0 + pos.height * yfrac)) * H
        return px, py

    # -------------------------------------------------------------- #
    # Save frame / animation                                         #
    # -------------------------------------------------------------- #

    def _save_frame(self):
        """Save the currently displayed frame (with any pick overlay) to a file."""
        if not self._frames:
            return
        from tkinter import filedialog
        default = (f'{self._sww_prefix}_{self._last_gen_qty}'
                   f'_frame{self._current + 1:04d}')
        path = filedialog.asksaveasfilename(
            title='Save current frame',
            initialfile=default,
            defaultextension='.png',
            filetypes=[('PNG image', '*.png'),
                       ('PDF document', '*.pdf'),
                       ('SVG image', '*.svg'),
                       ('All files', '*.*')])
        if not path:
            return
        self._fig.savefig(path, dpi=self._last_gen_dpi, bbox_inches='tight')
        self._set_status(f'Frame saved → {path}')

    def _save_animation(self):
        """Save all loaded frames as a GIF or MP4 animation."""
        if not self._frames:
            return
        import shutil
        from tkinter import filedialog

        has_ffmpeg = shutil.which('ffmpeg') is not None
        filetypes = []
        if has_ffmpeg:
            filetypes.append(('MP4 video', '*.mp4'))
        filetypes.append(('GIF animation', '*.gif'))
        filetypes.append(('All files', '*.*'))

        default_ext = '.mp4' if has_ffmpeg else '.gif'
        default = f'{self._sww_prefix}_{self._last_gen_qty}'
        path = filedialog.asksaveasfilename(
            title='Save animation',
            initialfile=default,
            defaultextension=default_ext,
            filetypes=filetypes)
        if not path:
            return

        fps = max(0.5, self._fps_var.get())
        if path.lower().endswith('.mp4'):
            self._save_animation_mp4(path, fps)
        else:
            self._save_animation_gif(path, fps)

    def _save_animation_gif(self, path, fps):
        try:
            from PIL import Image
        except ImportError:
            from tkinter import messagebox
            messagebox.showerror(
                'Missing dependency',
                'Pillow is required to save GIF animations.\n'
                'Install with:  pip install Pillow')
            return
        self._set_status(f'Saving GIF ({len(self._frames)} frames)…')
        self.root.update_idletasks()
        duration_ms = max(20, int(1000 / fps))
        imgs = [Image.open(f).convert('RGBA') for f in self._frames]
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     loop=0, duration=duration_ms, optimize=False)
        self._set_status(f'GIF saved ({len(imgs)} frames) → {path}')

    def _save_animation_mp4(self, path, fps):
        import shutil
        if not shutil.which('ffmpeg'):
            from tkinter import messagebox
            messagebox.showerror(
                'ffmpeg not found',
                'ffmpeg must be installed and on PATH to save MP4 videos.\n'
                'Download from https://ffmpeg.org or install via conda/apt/brew.')
            return
        import subprocess
        import tempfile
        import os
        self._set_status(f'Saving MP4 ({len(self._frames)} frames)…')
        self.root.update_idletasks()
        # Write a concat list so arbitrary frame sets work (stride, etc.)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False, encoding='utf-8') as f:
            dur = 1.0 / fps
            for frame_path in self._frames:
                f.write(f"file '{frame_path}'\n")
                f.write(f"duration {dur:.6f}\n")
            # ffmpeg concat requires the last entry twice (no duration on final)
            f.write(f"file '{self._frames[-1]}'\n")
            concat_path = f.name
        try:
            result = subprocess.run(
                ['ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                 '-i', concat_path,
                 '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                 '-movflags', '+faststart', path],
                capture_output=True, text=True)
            if result.returncode != 0:
                from tkinter import messagebox
                messagebox.showerror('ffmpeg error',
                                     result.stderr[-800:] or result.stdout[-800:])
                return
        finally:
            os.unlink(concat_path)
        self._set_status(f'MP4 saved ({len(self._frames)} frames) → {path}')

    def _show_mesh(self):
        """Open a Toplevel window showing the triangulation."""
        if self._splotter is None:
            return
        sp = self._splotter
        use_basemap = self._gen_used_basemap and sp.epsg is not None
        triang = sp.triang_abs if use_basemap else sp.triang

        win = tk.Toplevel(self.root)
        win.title(f'Mesh — {len(sp.triangles)} triangles')
        win.resizable(True, True)

        from matplotlib.figure import Figure
        mesh_fig = Figure(figsize=(8, 6))
        ax = mesh_fig.add_subplot(111)
        ax.triplot(triang, color='steelblue', linewidth=0.4, alpha=0.7)
        ax.set_aspect('equal')

        if use_basemap:
            ax.set_xlabel('Easting (m)')
            ax.set_ylabel('Northing (m)')
            ax.set_title(f'Mesh  ({len(sp.triangles)} triangles)')
            from anuga.utilities.animate import BASEMAP_PROVIDERS, _add_basemap
            provider_label = self._basemap_provider_var.get()
            provider_str = BASEMAP_PROVIDERS.get(provider_label, 'OpenStreetMap.Mapnik')
            try:
                _add_basemap(ax, sp.epsg, provider_str, cache=sp._basemap_cache)
            except Exception as e:
                ax.set_title(
                    f'Mesh  ({len(sp.triangles)} triangles)  — basemap failed: {e}')
        else:
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title(f'Mesh  ({len(sp.triangles)} triangles)')

        mesh_fig.tight_layout()

        mesh_canvas = FigureCanvasTkAgg(mesh_fig, master=win)
        mesh_canvas.draw()
        mesh_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def _close_mesh():
            plt.close(mesh_fig)
            win.destroy()

        win.protocol('WM_DELETE_WINDOW', _close_mesh)
        ttk.Button(win, text='Close', command=_close_mesh).pack(pady=4)

    def _show_help(self):
        """Open a scrollable help window."""
        win = tk.Toplevel(self.root)
        win.title('ANUGA SWW Animation GUI — Help')
        win.resizable(True, True)

        text = tk.Text(win, wrap=tk.WORD, width=72, height=42,
                       padx=10, pady=8, relief=tk.FLAT)
        sb = ttk.Scrollbar(win, command=text.yview)
        text.configure(yscrollcommand=sb.set)
        ttk.Button(win, text='Close', command=win.destroy).pack(side=tk.BOTTOM, pady=6)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        text.tag_configure('h1', font=('TkDefaultFont', 13, 'bold'), spacing3=4)
        text.tag_configure('h2', font=('TkDefaultFont', 10, 'bold'), spacing1=6, spacing3=2)
        text.tag_configure('kw', font=('TkFixedFont', 9))

        def h1(s): text.insert(tk.END, s + '\n', 'h1')
        def h2(s): text.insert(tk.END, s + '\n', 'h2')
        def p(s):  text.insert(tk.END, s + '\n')
        def kw(s): text.insert(tk.END, s, 'kw')
        def nl():  text.insert(tk.END, '\n')

        h1('ANUGA SWW Animation GUI')
        p('Visualise and explore SWW simulation output as animated frames.')
        nl()

        h2('Quick-start workflow')
        p('1. Open an SWW file with Browse… or pass --sww on the command line.')
        p('2. Choose a Quantity, set vmin/vmax (or tick Auto from data).')
        p('3. Click Generate Frames — PNGs are written to the Output dir.')
        p('4. Use Play / step buttons to animate, or drag the Frame slider.')
        nl()

        h2('Generation settings')
        h2('  Quantity')
        p('  depth, stage, speed, speed_depth — animated per timestep.')
        p('  max_depth, max_speed, max_speed_depth — single frame showing')
        p('  the maximum value at each triangle over all timesteps.')
        p('  elev — elevation (bed level).  Produces a single static frame')
        p('  when elevation is constant, or one frame per timestep when the')
        p('  SWW contains time-varying elevation (e.g. erosion simulations).')
        nl()
        h2('  vmin / vmax')
        p('  Colormap range.  Tick "Auto from data" to set automatically.')
        nl()
        h2('  DPI')
        p('  Resolution of generated PNG frames (default 100).')
        nl()
        h2('  min depth')
        p('  Triangles with depth below this value are treated as dry and')
        p('  shown in grey (elevation shading).')
        nl()
        h2('  Every N frames')
        p('  Stride: generate one frame every N SWW timesteps.')
        p('  Use a larger value for a quick preview of a long simulation.')
        nl()
        h2('  Colormap / Reverse')
        p('  Any matplotlib colormap name.  Tick Reverse to invert it.')
        nl()
        h2('  EPSG')
        p('  Override or supply the coordinate-system code.  Older SWW')
        p('  files do not store an EPSG code — type the integer (e.g. 32756')
        p('  for UTM zone 56 S) and press Set or Enter to enable basemap.')
        p('  If the file already carries an EPSG it is pre-populated.')
        nl()
        h2('  Basemap / provider / Alpha')
        p('  Overlay an online tile basemap (OpenStreetMap, Esri Satellite,')
        p('  etc.).  Requires an EPSG code (from file or entered manually)')
        p('  and an internet connection.  Alpha controls mesh transparency.')
        p('  Requires contextily:  pip install contextily')
        nl()

        h2('Playback controls')
        p('  Play/Pause — start or stop the animation.')
        p('  |< < > >|  — jump to first, step back, step forward, last frame.')
        p('  FPS        — playback speed in frames per second.')
        p('  Frame slider — drag to scrub through frames.')
        nl()

        h2('Pick timeseries')
        p('Click "Pick timeseries" to enter pick mode:')
        p('  • The cursor changes to a crosshair.')
        p('  • Click any point on the image to select the nearest triangle')
        p('    centroid and plot its time series in the panel below.')
        p('  • Click again to pick a different point — the timeseries and')
        p('    the red star marker update immediately.')
        p('  • Change the quantity in the timeseries dropdown to replot')
        p('    the same triangle for depth / stage / speed / speed_depth.')
        p('  • Press Esc or click "Cancel pick" to exit pick mode.')
        nl()

        h2('Timeseries panel')
        p('  Quantity dropdown — switch the plotted variable.')
        p('  Red dashed line  — current animation frame time.')
        p('  Export CSV       — save the displayed time series to a CSV file.')
        p('  Close            — hide the timeseries panel.')
        nl()

        h2('Zoom region')
        p('Click "Set Zoom" to enter rubber-band selection mode:')
        p('  • Drag a rectangle on the animation frame to select a region.')
        p('  • A yellow highlight shows the selected area.')
        p('  • The status bar shows the mesh coordinate bounds of the selection.')
        p('  • Click "Generate Frames" to regenerate at full resolution for')
        p('    the selected region only.')
        p('  • Click "Reset Zoom" to clear the selection and return to')
        p('    full-extent generation.')
        p('  Note: Set Zoom is only available after frames have been generated.')
        nl()

        h2('View Mesh')
        p('  Opens a separate window showing the full triangulation.  If a')
        p('  basemap was used for the last generation the mesh is drawn in')
        p('  absolute coordinates with the same basemap provider overlaid.')
        nl()

        h2('Saving frames and animations')
        p('  Save Frame       — saves the current frame (with any pick-marker')
        p('  overlay) as PNG, PDF, or SVG.')
        nl()
        p('  Save Animation…  — saves all loaded frames as:')
        p('    GIF  — requires Pillow:  pip install Pillow')
        p('    MP4  — requires ffmpeg on PATH; produces smaller, higher-quality')
        p('           files.  MP4 is offered first when ffmpeg is detected.')
        p('           Install:  conda install ffmpeg  or  apt install ffmpeg')
        p('  Playback FPS is used as the animation frame rate.')
        nl()

        h2('Performance')
        p('  Frame generation is automatically parallelised across up to 4')
        p('  CPU cores.  The progress bar advances in chunks as batches of')
        p('  frames complete.  Basemap tiles are cached in memory so the')
        p('  network is only hit once per generation run.')
        nl()

        h2('Optional dependencies')
        p('  Install all GUI extras with:  pip install anuga[gui]')
        p('  or individually:')
        p('    contextily — basemap tile overlay')
        p('    Pillow     — GIF animation export')
        p('    ffmpeg     — MP4 animation export (system package)')
        nl()

        h2('Command-line usage')
        kw('  anuga_sww_gui [--sww FILE] [--qty QUANTITY]\n')
        nl()

        text.configure(state=tk.DISABLED)

    def _set_status(self, msg):
        self._status_var.set(msg)

    def _on_close(self):
        self._cancel_generation()
        self._stop_playback()
        plt.close('all')
        self.root.quit()
        self.root.destroy()


# ------------------------------------------------------------------ #
# Entry point                                                         #
# ------------------------------------------------------------------ #

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='ANUGA SWW animation viewer',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sww', default=None,
                        help='SWW file to open on startup')
    parser.add_argument('--qty', default=None,
                        choices=list(_QUANTITIES),
                        help='Quantity to display on startup')
    args = parser.parse_args()

    root = tk.Tk()
    SWWAnimationGUI(root, initial_sww=args.sww, initial_qty=args.qty)
    root.mainloop()


if __name__ == '__main__':
    main()
