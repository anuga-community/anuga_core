#!/usr/bin/env python3
"""GUI viewer for ANUGA SWW file animations.

Opens an SWW file, generates PNG frames for a chosen quantity using
SWW_plotter, then plays them as an animation.

Usage::

    anuga_animate_sww_gui
    anuga_animate_sww_gui --sww domain.sww
    anuga_animate_sww_gui --sww domain.sww --qty depth
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

_QUANTITIES = ('depth', 'stage', 'speed', 'speed_depth')

_QTY_DEFAULTS = {
    'depth':       dict(vmin=0.0,   vmax=20.0),
    'stage':       dict(vmin=-20.0, vmax=20.0),
    'speed':       dict(vmin=0.0,   vmax=10.0),
    'speed_depth': dict(vmin=0.0,   vmax=20.0),
}

# Attribute on SWW_plotter that holds the data array for each quantity
_QTY_DATA_ATTR = {
    'depth':       'depth',
    'stage':       'stage',
    'speed':       'speed',
    'speed_depth': 'speed_depth',
}

# Method name on SWW_plotter to save a single frame
_QTY_SAVE_METHOD = {
    'depth':       'save_depth_frame',
    'stage':       'save_stage_frame',
    'speed':       'save_speed_frame',
    'speed_depth': 'save_speed_depth_frame',
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

        # timeseries state
        self._ts_triangle = None
        self._ts_vline = None
        self._pick_mode = False
        self._pick_cid = None
        self._pick_key_cid = None

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
                  'RdBu_r', 'coolwarm', 'seismic', 'jet', 'turbo']
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

        # ---- Row 4: SWW info ----
        row4 = ttk.Frame(ctrl)
        row4.pack(fill=tk.X, pady=(4, 0))
        self._sww_info_label = ttk.Label(row4, text='No SWW file loaded.',
                                          foreground='grey')
        self._sww_info_label.pack(side=tk.LEFT, padx=4)

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
                                    values=['depth', 'stage', 'speed', 'speed_depth'],
                                    width=12, state='readonly')
        ts_qty_combo.pack(side=tk.LEFT, padx=2)
        ts_qty_combo.bind('<<ComboboxSelected>>',
                          lambda _e: self._update_timeseries())
        self._ts_info_label = ttk.Label(ts_ctrl, text='', foreground='grey')
        self._ts_info_label.pack(side=tk.LEFT, padx=8)
        ttk.Button(ts_ctrl, text='Close',
                   command=self._close_timeseries).pack(side=tk.RIGHT, padx=4)

        self._ts_fig, self._ts_ax = plt.subplots(figsize=(10, 2.5))
        self._ts_fig.tight_layout(pad=1.5)
        self._ts_canvas = FigureCanvasTkAgg(self._ts_fig, master=self._ts_outer)
        self._ts_canvas.draw()
        self._ts_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ---- matplotlib map canvas ----
        self._fig, self._ax = plt.subplots(figsize=(10, 6))
        self._ax.axis('off')
        self._fig.tight_layout(pad=0)

        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self._canvas = FigureCanvasTkAgg(self._fig, master=canvas_frame)
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
        epsg_msg = getattr(self, '_epsg_msg', '')
        self._sww_info_label.config(
            text=f'{n} timesteps  |  t={t0:.1f} .. {t1:.1f} s{epsg_msg}',
            foreground='grey')
        self._gen_btn.config(state=tk.NORMAL)
        self._pick_btn.config(state=tk.NORMAL)
        self._set_status(f'Loaded {os.path.basename(sww)} -{n} timesteps')
        self._update_auto_limits()
        self._exit_pick_mode()
        self._close_timeseries()

    def _load_splotter(self, sww):
        from anuga.utilities.animate import SWW_plotter

        self._splotter = SWW_plotter(
            swwfile=sww,
            plot_dir=None,        # suppress automatic make_plot_dir
            min_depth=float(self._mindepth_var.get()))

        self._sww_prefix = self._splotter.name  # bare basename, no extension

        # Enable OSM basemap checkbox only when the SWW has an EPSG code
        # and contextily is installed
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
            epsg_msg = f'  EPSG:{self._splotter.epsg}'
        elif has_epsg:
            self._basemap_chk.config(state=tk.DISABLED)
            self._basemap_var.set(False)
            self._basemap_provider_combo.config(state=tk.DISABLED)
            epsg_msg = f'  EPSG:{self._splotter.epsg} (install contextily for basemap)'
        else:
            self._basemap_chk.config(state=tk.DISABLED)
            self._basemap_var.set(False)
            self._basemap_provider_combo.config(state=tk.DISABLED)
            epsg_msg = ''
        self._epsg_msg = epsg_msg

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

    # -------------------------------------------------------------- #
    # Frame generation                                                #
    # -------------------------------------------------------------- #

    def _start_generation(self):
        if self._splotter is None:
            return

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

        n_total   = len(self._splotter.time)
        sww_frames = list(range(0, n_total, stride))
        n_to_gen   = len(sww_frames)

        stride_msg = f' (every {stride})' if stride > 1 else ''
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
                     '_speed_frame_count', '_speed_depth_frame_count'):
            setattr(self._splotter, attr, 0)

        save_method = getattr(self._splotter, _QTY_SAVE_METHOD[qty])
        self._generate_next_frame(0, sww_frames, save_method, dpi, vmin, vmax,
                                   plot_dir, qty, cmap, basemap, alpha,
                                   basemap_provider)

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
                        basemap_provider=basemap_provider)
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

    def _cancel_generation(self):
        self._cancel_flag = True
        if self._gen_after_id is not None:
            self.root.after_cancel(self._gen_after_id)
            self._gen_after_id = None

    def _on_generation_done(self, plot_dir, qty, n_frames):
        self._progress_var.set(n_frames)
        self._progress_label.config(text=f'{n_frames} / {n_frames}')
        self._gen_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)
        prefix = self._sww_prefix
        self._set_status(
            f'Done -{n_frames} frames saved to {plot_dir}')
        self._load_frames(plot_dir, qty, prefix)

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
        self._show_frame(0)
        self._set_status(f'Loaded {n} frames  |  {plot_dir}')

    def _show_frame(self, idx):
        if not self._frames:
            return
        idx = max(0, min(idx, len(self._frames) - 1))
        self._current = idx
        self._slider_var.set(idx)
        self._frame_label.config(text=f'Frame {idx + 1} / {len(self._frames)}')

        img = mpimage.imread(self._frames[idx])
        if self._im is None:
            self._im = self._ax.imshow(img, aspect='equal')
        else:
            self._im.set_data(img)
            self._im.set_extent([0, img.shape[1], img.shape[0], 0])
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
        fps = max(0.1, self._fps_var.get())
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
        """Render the live mesh in the main canvas for interactive point picking."""
        if self._splotter is None:
            return
        import numpy as np

        sp = self._splotter
        n_ts = len(sp.time)
        if self._frames and len(self._frames) > 1:
            frac = self._current / (len(self._frames) - 1)
            fidx = int(round(frac * (n_ts - 1)))
        else:
            fidx = 0

        depth = sp.depth[fidx, :]
        md = sp.min_depth
        try:
            elev = sp.elev[fidx, :]
        except IndexError:
            elev = sp.elev

        self._ax.cla()
        self._ax.set_aspect('equal')
        self._ax.set_title(
            'Click to pick a timeseries point  |  Esc to cancel', fontsize=9)

        sp.triang.set_mask(depth > md)
        self._ax.tripcolor(sp.triang, facecolors=elev, cmap='Greys_r')
        sp.triang.set_mask(depth <= md)
        self._ax.tripcolor(sp.triang, facecolors=depth, cmap='viridis', alpha=0.8)
        sp.triang.set_mask(None)

        # Mark existing pick if present
        if self._ts_triangle is not None:
            self._ax.plot(sp.xc[self._ts_triangle], sp.yc[self._ts_triangle],
                          'r*', markersize=14, zorder=5, label='current pick')

        self._ax.set_xlabel('Easting (m)')
        self._ax.set_ylabel('Northing (m)')
        self._fig.tight_layout(pad=1.0)
        self._canvas.draw()
        self._canvas.get_tk_widget().config(cursor='crosshair')

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
        dist = np.sqrt((sp.xc - event.xdata)**2 + (sp.yc - event.ydata)**2)
        self._ts_triangle = int(dist.argmin())
        self._exit_pick_mode()
        self._update_timeseries()

    def _on_pick_key(self, event):
        if event.key == 'escape':
            self._exit_pick_mode()

    def _exit_pick_mode(self):
        """Disconnect pick events and restore the animation frame."""
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
        # restore PNG display
        self._ax.cla()
        self._ax.axis('off')
        self._fig.tight_layout(pad=0)
        self._im = None
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
        }

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

        # Show the panel if not already visible
        if not self._ts_outer.winfo_ismapped():
            self._ts_outer.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False)

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

    # -------------------------------------------------------------- #
    # Helpers                                                         #
    # -------------------------------------------------------------- #

    def _set_status(self, msg):
        self._status_var.set(msg)

    def _on_close(self):
        self._cancel_generation()
        self._stop_playback()
        plt.close(self._fig)
        plt.close(self._ts_fig)
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
