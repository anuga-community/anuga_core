"""
A module to allow interactive plotting in a Jupyter notebook of quantities and mesh
associated with an ANUGA domain and SWW file.
"""
import numpy as np
import os
import matplotlib.pyplot as plt


class Domain_plotter:
    """
    A class to wrap ANUGA domain centroid values for stage, height, elevation
    xmomentunm and ymomentum, and triangulation information.
    """


    def __init__(self, domain, plot_dir='_plot', min_depth=0.01, absolute=False):

        self.plot_dir = plot_dir
        self.make_plot_dir()

        self.zone = domain.geo_reference.zone

        self.min_depth = min_depth

        self.nodes = domain.nodes
        self.triangles = domain.triangles

        self.xllcorner = domain.geo_reference.xllcorner
        self.yllcorner = domain.geo_reference.yllcorner

        if absolute is False:
            self.x = domain.nodes[:, 0]
            self.y = domain.nodes[:, 1]

            self.xc = domain.centroid_coordinates[:, 0]
            self.yc = domain.centroid_coordinates[:, 1]
        else:
            self.x = domain.nodes[:, 0] + self.xllcorner
            self.y = domain.nodes[:, 1] + self.yllcorner

            self.xc = domain.centroid_coordinates[:, 0] + self.xllcorner
            self.yc = domain.centroid_coordinates[:, 1] + self.yllcorner


        import matplotlib.tri as tri
        self.triang = tri.Triangulation(self.x, self.y, self.triangles)

        self.elev = domain.quantities['elevation'].centroid_values
        self.stage = domain.quantities['stage'].centroid_values

        self.xmom = domain.quantities['xmomentum'].centroid_values
        self.ymom = domain.quantities['ymomentum'].centroid_values

        self.friction = domain.quantities['friction'].centroid_values

        self.depth = self.stage - self.elev

        with np.errstate(invalid='ignore'):
            self.xvel = np.where(self.depth > self.min_depth,
                             self.xmom / self.depth, 0.0)
            self.yvel = np.where(self.depth > self.min_depth,
                             self.ymom / self.depth, 0.0)

        self.speed = np.sqrt(self.xvel**2 + self.yvel**2)

        self.speed_depth = self.speed*self.depth

        self.domain = domain


    #------------------------------------------
    # General plots
    #------------------------------------------
    def plot_mesh(self, figsize=(10, 6), dpi=80):

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax.triplot(self.triang, linewidth=0.4)

        ax.set_title('Mesh')
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')

        #plt.show()

        return fig, ax

    def _depth_frame(self, figsize, dpi, vmin, vmax):


        name = self.domain.get_name()
        time = self.domain.get_time()

        self.depth[:] = self.stage - self.elev

        md = self.min_depth

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax.set_title('Depth: Time {0:0>4}'.format(time))

        self.triang.set_mask(self.depth > md)
        ax.tripcolor(self.triang,
                      facecolors=self.elev,
                      cmap='Greys_r')


        self.triang.set_mask(self.depth <= md)
        im = ax.tripcolor(self.triang,
                      facecolors=self.depth,
                      cmap='viridis',
                      vmin=vmin, vmax=vmax)

        self.triang.set_mask(None)

        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        fig.colorbar(im, ax=ax)

        return fig, ax

    def save_depth_frame(self, figsize=(10, 6), dpi=80,
                         vmin=0.0, vmax=20):



        plot_dir = self.plot_dir
        name = self.domain.get_name()
        time = self.domain.get_time()

        fig, ax = self._depth_frame(figsize, dpi, vmin, vmax)

        if plot_dir is None:
            fig.savefig(name+'_depth_{0:0>10}.png'.format(int(time)))
        else:
            fig.savefig(os.path.join(plot_dir, name
                                     + '_depth_{0:0>10}.png'.format(int(time))))

        fig.clf()
        plt.close()

    def plot_depth_frame(self, figsize=(10, 6), dpi=80,
                         vmin=0.0, vmax=20.0):

        import matplotlib.pyplot as plt

        fig, ax = self._depth_frame(figsize, dpi, vmin, vmax)

        #plt.show()

        return fig, ax

    def make_depth_animation(self):

        import numpy as np
        import glob
        from matplotlib import image, animation
        from matplotlib import pyplot as plt

        plot_dir = self.plot_dir
        name = self.domain.get_name()
        time = self.domain.get_time()

        if plot_dir is None:
            expression = name+'_depth_*.png'
        else:
            expression = os.path.join(plot_dir, name+'_depth_*.png')
        img_files = sorted(glob.glob(expression))

        figsize = (10, 6)

        fig = plt.figure(figsize=figsize, dpi=80)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')  # so there's not a second set of axes
        im = plt.imshow(image.imread(img_files[0]))

        def init():

            im.set_data(image.imread(img_files[0]))
            return im,

        def animate(i):

            image_i = image.imread(img_files[i])
            im.set_data(image_i)
            return im,

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(img_files), interval=200, blit=True)

        plt.close()

        return anim

    def _stage_frame(self, figsize, dpi, vmin, vmax):

        name = self.domain.get_name()
        time = self.domain.get_time()

        self.depth[:] = self.stage - self.elev

        md = self.min_depth

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax.set_title('Stage: Time {0:0>4}'.format(time))

        self.triang.set_mask(self.depth > md)
        ax.tripcolor(self.triang,
                      facecolors=self.elev,
                      cmap='Greys_r')

        self.triang.set_mask(self.depth <= md)
        im = ax.tripcolor(self.triang,
                      facecolors=self.stage,
                      cmap='viridis',
                      vmin=vmin, vmax=vmax)


        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        fig.colorbar(im, ax=ax)

        self.triang.set_mask(None)

        return fig, ax

    def save_stage_frame(self, figsize=(10, 6), dpi=80,
                         vmin=-20.0, vmax=20.0):

        import matplotlib.pyplot as plt

        plot_dir = self.plot_dir
        name = self.domain.get_name()
        time = self.domain.get_time()

        fig, ax = self._stage_frame(figsize, dpi, vmin, vmax)

        if plot_dir is None:
            fig.savefig(name+'_stage_{0:0>10}.png'.format(int(time)))
        else:
            fig.savefig(os.path.join(plot_dir, name
                                     + '_stage_{0:0>10}.png'.format(int(time))))

        fig.clf()
        plt.close()

    def plot_stage_frame(self, figsize=(10, 6), dpi=80,
                         vmin=-20.0, vmax=20.0):

        import matplotlib.pyplot as plt

        fig, ax = self._stage_frame(figsize, dpi, vmin, vmax)

        #plt.show()

        return fig, ax

    def make_stage_animation(self):

        import numpy as np
        import glob
        from matplotlib import image, animation
        from matplotlib import pyplot as plt

        plot_dir = self.plot_dir
        name = self.domain.get_name()
        time = self.domain.get_time()

        if plot_dir is None:
            expression = name+'_stage_*.png'
        else:
            expression = os.path.join(plot_dir, name+'_stage_*.png')

        img_files = sorted(glob.glob(expression))

        figsize = (10, 6)

        fig = plt.figure(figsize=figsize, dpi=80)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')  # so there's not a second set of axes
        im = plt.imshow(image.imread(img_files[0]))

        def init():

            im.set_data(image.imread(img_files[0]))
            return im,

        def animate(i):

            image_i = image.imread(img_files[i])
            im.set_data(image_i)
            return im,

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(img_files), interval=200, blit=True)

        plt.close()

        return anim

    def _speed_frame(self, figsize, dpi, vmin, vmax):


        name = self.domain.get_name()
        time = self.domain.get_time()

        md = self.min_depth

        self.depth[:] = self.stage - self.elev

        with np.errstate(invalid='ignore'):
            self.xvel = np.where(self.depth > self.min_depth,
                             self.xmom / self.depth, 0.0)
            self.yvel = np.where(self.depth > self.min_depth,
                             self.ymom / self.depth, 0.0)

        self.speed = np.sqrt(self.xvel**2 + self.yvel**2)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        ax.set_title('Speed: Time {0:0>4}'.format(time))

        self.triang.set_mask(self.depth > md)
        ax.tripcolor(self.triang,
                      facecolors=self.elev,
                      cmap='Greys_r')

        self.triang.set_mask(self.depth <= md)
        im = ax.tripcolor(self.triang,
                      facecolors=self.speed,
                      cmap='viridis',
                      vmin=vmin, vmax=vmax)

        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        fig.colorbar(im, ax=ax)

        self.triang.set_mask(None)

        return fig, ax

    def save_speed_frame(self, figsize=(10, 6), dpi=80,
                         vmin=-20.0, vmax=20.0):

        import matplotlib.pyplot as plt

        plot_dir = self.plot_dir
        name = self.domain.get_name()
        time = self.domain.get_time()

        fig, ax = self._speed_frame(figsize, dpi, vmin, vmax)

        if plot_dir is None:
            fig.savefig(name+'_speed_{0:0>10}.png'.format(int(time)))
        else:
            fig.savefig(os.path.join(plot_dir, name
                                     + '_speed_{0:0>10}.png'.format(int(time))))
        fig.clf()
        plt.close()

    def plot_speed_frame(self, figsize=(5, 3), dpi=80,
                         vmin=-20.0, vmax=20.0):

        import matplotlib.pyplot as plt

        fig, ax = self._speed_frame(figsize, dpi, vmin, vmax)

        #plt.show()

        return fig, ax

    def make_speed_animation(self):

        import numpy as np
        import glob
        from matplotlib import image, animation
        from matplotlib import pyplot as plt

        plot_dir = self.plot_dir
        name = self.domain.get_name()
        time = self.domain.get_time()

        if plot_dir is None:
            expression = name+'_speed_*.png'
        else:
            expression = os.path.join(plot_dir, name+'_speed_*.png')

        img_files = sorted(glob.glob(expression))

        figsize = (10, 6)

        fig = plt.figure(figsize=figsize, dpi=80)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')  # so there's not a second set of axes
        im = plt.imshow(image.imread(img_files[0]))

        def init():

            im.set_data(image.imread(img_files[0]))
            return im,

        def animate(i):

            image_i = image.imread(img_files[i])
            im.set_data(image_i)
            return im,

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(img_files), interval=200, blit=True)

        plt.close()

        return anim

    def make_plot_dir(self, clobber=True):
        """
        Utility function to create a directory for storing a sequence of plot
        files, or if the directory already exists, clear out any old plots.
        If clobber==False then it will abort instead of deleting existing files.
        """

        plot_dir = self.plot_dir
        if plot_dir is None:
            return
        else:
            if os.path.isdir(plot_dir):
                if clobber:
                    try:
                        os.remove("%s/*" % plot_dir)
                    except OSError:
                        pass
                else:
                    raise OSError(
                        '*** Cannot clobber existing directory %s' % plot_dir)
            else:
                os.mkdir("%s" % plot_dir)
            print("Figure files for each frame will be stored in " + plot_dir)


class SWW_plotter:
    """
    A class to wrap ANUGA swwfile centroid values for stage, height, elevation
    xmomentum and ymomentum, and triangulation information.

    Plotting functions are provided to plot depth, speed, speed_depth, stage and mesh,
    and to create animations of depth, speed, speed_depth (momentum) and stage.

    Plotting based on matplotlib's tripcolor and triplot functions.

    Example:

        Open an SWW file and plot various quantities.

    >>> import anuga
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Enable interactive mode
    >>> plot.ion()
    >>>
    >>> # Create SWW plotter object by opening an SWW file
    >>> splotter = anuga.SWW_plotter('domain.sww')
    >>>
    >>> # Find min and max depth for plotting
    >>> vmin = splotter.depth.min()
    >>> vmax = splotter.depth.max()
    >>>
    >>> # Plot depth at last frame.
    >>> fig, ax = splotter.plot_depth_frame(-1, vmin=vmin, vmax=vmax)
    >>> ax.set_title('Depth at final time')
    >>>
    >>> # Plot speed at second frame
    >>> fig, ax = splotter.plot_speed_frame(1)
    >>> ax.set_title('Speed at second frame')
    >>>
    >>> # Plot Mesh
    >>> fig, ax, im = splotter.plot_mesh()
    >>> ax.set_title('Mesh')
    >>>
    >>> # Animate depth
    >>> for frame in range(len(splotter.time)):
    ...     splotter.save_depth_frame(frame, vmin=vmin, vmax=vmax)
    >>> anim = splotter.make_depth_animation()
    """

    def __init__(self, swwfile='domain.sww', plot_dir='_plot',
                 min_depth = 0.001,
                 absolute=False):

        self.filename = swwfile

        self.plot_dir = plot_dir
        self.make_plot_dir()

        self.min_depth = min_depth

        import matplotlib.tri as tri
        import numpy as np

        self.name = os.path.splitext(swwfile)[0]

        from anuga.file.netcdf import NetCDFFile
        p = NetCDFFile(swwfile)

        self.x = np.array(p.variables['x'])
        self.y = np.array(p.variables['y'])
        self.triangles = np.array(p.variables['volumes'])

        vols0 = self.triangles[:, 0]
        vols1 = self.triangles[:, 1]
        vols2 = self.triangles[:, 2]

        self.triang = tri.Triangulation(self.x, self.y, self.triangles)

        self.xc = (self.x[vols0]+self.x[vols1]+self.x[vols2])/3.0
        self.yc = (self.y[vols0]+self.y[vols1]+self.y[vols2])/3.0

        self.xllcorner = p.xllcorner
        self.yllcorner = p.yllcorner
        self.zone = p.zone
        self.timezone = p.timezone
        self.starttime = p.starttime

        if absolute is True:
            self.x[:] = self.x + self.xllcorner
            self.y[:] = self.y + self.yllcorner

            self.xc[:] = self.xc + self.xllcorner
            self.yc[:] = self.yc + self.yllcorner


        self.elev = np.array(p.variables['elevation_c'])
        self.stage = np.array(p.variables['stage_c'])
        self.xmom = np.array(p.variables['xmomentum_c'])
        self.ymom = np.array(p.variables['ymomentum_c'])

        self.depth = np.zeros_like(self.stage)
        if(len(self.elev.shape) == 2):
            self.depth = self.stage - self.elev
        else:
            for i in range(self.depth.shape[0]):
                self.depth[i, :] = self.stage[i, :]-self.elev


        with np.errstate(invalid='ignore'):
            self.xvel = np.where(self.depth > self.min_depth,
                             self.xmom / self.depth, 0.0)
            self.yvel = np.where(self.depth > self.min_depth,
                             self.ymom / self.depth, 0.0)

        self.speed = np.sqrt(self.xvel**2 + self.yvel**2)

        self.speed_depth = self.speed*self.depth

        self.time = np.array(p.variables['time'])

    #------------------------------------------
    # General plots
    #------------------------------------------
    def plot_mesh(self, figsize=(10, 8), dpi=80, **kwargs):

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        im = ax.triplot(self.triang, linewidth=0.4, **kwargs)

        ax.set_title('Mesh')
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')

        #plt.show()

        return fig, ax, im

    #------------------------------------------
    # Depth procedures
    #------------------------------------------
    def _depth_frame(self, frame, figsize, dpi, vmin, vmax):

        name = self.name
        time = self.time[frame]
        depth = self.depth[frame, :]

        md = self.min_depth

        try:
            elev = self.elev[frame, :]
        except IndexError:
            elev = self.elev

        ims = []

        #fig = plt.figure(figsize=figsize, dpi=dpi)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        plt.title('Depth: Time {0:0>4}'.format(time))

        self.triang.set_mask(depth > md)
        ax.tripcolor(self.triang,
                      facecolors=elev,
                      cmap='Greys_r')

        self.triang.set_mask(depth < md)
        im = ax.tripcolor(self.triang,
                      facecolors=depth,
                      cmap='viridis',
                      vmin=vmin, vmax=vmax)


        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')

        self.triang.set_mask(None)

        fig.colorbar(im, ax=ax)

        return fig, ax

    def save_depth_frame(self, frame=-1, figsize=(10, 6), dpi=160,
                         vmin=0.0, vmax=20.0):

        import matplotlib.pyplot as plt

        name = self.name
        time = self.time[frame]
        plot_dir = self.plot_dir

        fig, ax = self._depth_frame(frame, figsize, dpi, vmin, vmax)

        if plot_dir is None:
            fig.savefig(name+'_depth_{0:0>10}.png'.format(int(time)))
        else:
            fig.savefig(os.path.join(plot_dir, name
                                     + '_depth_{0:0>10}.png'.format(int(time))))
        plt.close()
        fig.clf()

    def plot_depth_frame(self, frame=-1, figsize=(10, 6), dpi = 80,
                         vmin=0.0, vmax=20.0):

        import matplotlib.pyplot as plt

        fig, ax = self._depth_frame(frame, figsize, dpi, vmin, vmax)

        #plt.show()

        return fig, ax


    #------------------------------------------
    # Stage procedures
    #------------------------------------------
    def _stage_frame(self, frame, figsize, dpi, vmin, vmax):

        import matplotlib.pyplot as plt

        name = self.name
        time = self.time[frame]
        stage = self.stage[frame, :]
        depth = self.depth[frame, :]

        md = self.min_depth

        try:
            elev = self.elev[frame, :]
        except IndexError:
            elev = self.elev

        ims = []

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        plt.title('Stage: Time {0:0>4}'.format(time))

        self.triang.set_mask(depth > md)
        ax.tripcolor(self.triang,
                      facecolors=elev,
                      cmap='Greys_r')

        self.triang.set_mask(depth < md)
        im = ax.tripcolor(self.triang,
                      facecolors=stage,
                      cmap='viridis',
                      vmin=vmin, vmax=vmax)

        self.triang.set_mask(None)


        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')

        fig.colorbar(im, ax=ax)

        return fig, ax

    def save_stage_frame(self, frame=-1, figsize=(10, 6), dpi=160,
                         vmin=-20.0, vmax=20.0):

        import matplotlib.pyplot as plt

        name = self.name
        time = self.time[frame]
        plot_dir = self.plot_dir

        fig, ax = self._stage_frame(frame, figsize, dpi, vmin, vmax)

        if plot_dir is None:
            fig.savefig(name+'_stage_{0:0>10}.png'.format(int(time)))
        else:
            fig.savefig(os.path.join(plot_dir, name
                                     + '_stage_{0:0>10}.png'.format(int(time))))
        plt.close()
        fig.clf()

    def plot_stage_frame(self, frame=-1, figsize=(5, 3), dpi=80,
                         vmin=-20, vmax=20.0):

        import matplotlib.pyplot as plt

        fig, ax = self._stage_frame(frame, figsize, dpi, vmin, vmax)

        #plt.show()

        return fig, ax

    #------------------------------------------
    # Depth Speed procedures
    #------------------------------------------
    def _speed_depth_frame(self, frame, figsize, dpi, vmin, vmax):

        import matplotlib.pyplot as plt

        name = self.name
        time = self.time[frame]
        stage = self.stage[frame, :]
        speed_depth = self.speed_depth[frame, :]
        depth = self.depth[frame, :]

        md = self.min_depth

        try:
            elev = self.elev[frame, :]
        except IndexError:
            elev = self.elev

        ims = []

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        plt.title('Speed_Depth: Time {0:0>4}'.format(time))

        self.triang.set_mask(depth > md)
        ax.tripcolor(self.triang,
                      facecolors=speed_depth,
                      cmap='Greys_r')

        self.triang.set_mask(depth < md)
        im = ax.tripcolor(self.triang,
                      facecolors=elev,
                      cmap='viridis',
                      vmin=vmin, vmax=vmax)

        self.triang.set_mask(None)


        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        fig.colorbar(im, ax=ax)

        return fig, ax

    def save_speed_depth_frame(self, frame=-1, figsize=(10, 6), dpi=160,
                         vmin=-20.0, vmax=20.0):

        import matplotlib.pyplot as plt

        name = self.name
        time = self.time[frame]
        plot_dir = self.plot_dir

        fig, ax = self._speed_depth_frame(frame, figsize, dpi, vmin, vmax)

        if plot_dir is None:
            fig.savefig(name+'_speed_depth_{0:0>10}.png'.format(int(time)))
        else:
            fig.savefig(os.path.join(plot_dir, name
                                     + '_speed_depth_{0:0>10}.png'.format(int(time))))
        plt.close()
        fig.clf()

    def plot_speed_depth_frame(self, frame=-1, figsize=(5, 3), dpi=80,
                         vmin=-20, vmax=20.0):

        import matplotlib.pyplot as plt

        self._speed_depth_frame(frame, figsize, dpi, vmin, vmax)

        #plt.show()

        return fig, ax

    #------------------------------------------
    # Speed procedures
    #------------------------------------------
    def _speed_frame(self, frame, figsize, dpi, vmin, vmax):

        import matplotlib.pyplot as plt

        name = self.name
        time = self.time[frame]
        depth = self.depth[frame, :]

        md = self.min_depth

        try:
            elev = self.elev[frame, :]
        except IndexError:
            elev = self.elev
        speed = self.speed[frame, :]

        ims = []

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        plt.title('Speed: Time {0:0>4}'.format(time))

        self.triang.set_mask(depth > md)
        ax.tripcolor(self.triang,
                      facecolors=elev,
                      cmap='Greys_r')

        self.triang.set_mask(depth < md)
        im = ax.tripcolor(self.triang,
                      facecolors=speed,
                      cmap='viridis',
                      vmin=vmin, vmax=vmax)

        self.triang.set_mask(None)

        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        fig.colorbar(im, ax=ax)

        return fig, ax

    def save_speed_frame(self, frame=-1, figsize=(10, 6), dpi=160,
                         vmin=0.0, vmax=10.0):

        name = self.name
        time = self.time[frame]
        plot_dir = self.plot_dir

        fig, ax = self._speed_frame(frame, figsize, dpi, vmin, vmax)

        if plot_dir is None:
            fig.savefig(name+'_speed_{0:0>10}.png'.format(int(time)))
        else:
            fig.savefig(os.path.join(plot_dir, name
                                     + '_speed_{0:0>10}.png'.format(int(time))))
        plt.close()
        fig.clf()

    def plot_speed_frame(self, frame=-1, figsize=(10, 6), dpi=80,
                         vmin=0.0, vmax=10.0):

        import matplotlib.pyplot as plt

        fig, ax = self._speed_frame(frame, figsize, dpi, vmin, vmax)

        #plt.show()

        return fig, ax

    #------------------------------------------
    # Animation procedures
    #------------------------------------------
    def make_depth_animation(self):

        return self._make_quantity_animation(quantity='depth')

    def make_speed_animation(self):

        return self._make_quantity_animation(quantity='speed')

    def make_stage_animation(self):

        return self._make_quantity_animation(quantity='stage')

    def make_speed_depth_animation(self):

        return self._make_quantity_animation(quantity='speed_depth')


    def _make_quantity_animation(self, quantity='depth'):

        import numpy as np
        import glob
        from matplotlib import image, animation
        from matplotlib import pyplot as plt

        plot_dir = self.plot_dir
        name = self.name

        if plot_dir is None:
            expression = name+'_'+quantity+'_*.png'
        else:
            expression = os.path.join(plot_dir, name+'_'+quantity+'_*.png')
        img_files = sorted(glob.glob(expression))

        figsize = (10, 6)

        fig = plt.figure(figsize=figsize, dpi=80)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')  # so there's not a second set of axes
        im = plt.imshow(image.imread(img_files[0]))

        def init():
            im.set_data(image.imread(img_files[0]))
            return im,

        def animate(i):
            image_i = image.imread(img_files[i])
            im.set_data(image_i)
            return im,

        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(img_files),
                                       interval=200, blit=True)

        plt.close()

        return anim

    def make_plot_dir(self, clobber=True):
        """
        Utility function to create a directory for storing a sequence of plot
        files, or if the directory already exists, clear out any old plots.
        If clobber==False then it will abort instead of deleting existing files.
        """

        plot_dir = self.plot_dir
        if plot_dir is None:
            return
        else:
            if os.path.isdir(plot_dir):
                if clobber:
                    try:
                        os.remove("%s/*" % plot_dir)
                    except OSError:
                        pass
                else:
                    raise OSError(
                      '*** Cannot clobber existing directory %s' % plot_dir)
            else:
                os.mkdir("%s" % plot_dir)
            print("Figure files for each frame will be stored in " + plot_dir)

    def triplot(self, figsize = (10, 6), dpi=80, **kwargs):
        """
        Create a triplot of the mesh.

        Args:
            figsize: Figure size
            dpi: Dots per inch
            **kwargs: Additional arguments to pass to triplot
        Returns:
            fig: Matplotlib figure object
            ax: Matplotlib axes object
            lines: The lines created by triplot
        """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        lines = ax.triplot(self.triang, *args, **kwargs)
        return fig, ax, lines


    def tripcolor(self, figsize = (10, 6), dpi=80, **kwargs):
        """
        Create a tripcolor plot of the mesh.

        Args:
            figsize: Figure size
            dpi: Dots per inch
            **kwargs: Additional arguments to pass to tripcolor
        Returns:
            fig: Matplotlib figure object
            ax: Matplotlib axes object
            im: The image created by tripcolor
        """

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        im = ax.tripcolor(self.triang,  *args, **kwargs)
        return fig, ax, im

    def get_flow_through_cross_section(self, polyline: list, verbose: bool = False) -> tuple[np.ndarray, list]:
        """
        Calculate flow through a cross-section defined by a polyline.

        Args:
            polyline: List of [x, y] coordinates defining the cross-section
            verbose: Whether to print debug information

        Returns:
            tuple containing:
                - time: numpy array of time values
                - Q: list of flow values at each timestep
        """

        if not hasattr(self, 'mesh'):
            from anuga.file.sww import get_mesh_and_quantities_from_file
            self.mesh, _, __ = get_mesh_and_quantities_from_file(self.filename, verbose=verbose)

        segments = self.mesh.get_intersecting_segments(polyline, verbose=verbose)

        if verbose:
            print(f'Cross-section intersects {len(segments)} triangles')
            print('Computing hydrograph')

        # Pre-extract per-segment geometry as arrays for vectorised computation
        tri_ids = np.array([seg.triangle_id for seg in segments])  # (S,)
        normals = np.array([seg.normal for seg in segments])        # (S, 2)
        lengths = np.array([seg.length for seg in segments])        # (S,)

        # Slice momentum arrays for the intersected triangles: (T, S)
        uh = self.xmom[:, tri_ids]
        vh = self.ymom[:, tri_ids]

        # Normal momentum at each segment and timestep, summed to hydrograph
        normal_mom = uh * normals[:, 0] + vh * normals[:, 1]  # (T, S)
        Q = (normal_mom * lengths).sum(axis=1)                 # (T,)

        if verbose:
            print(f'Done: {len(self.time)} timesteps, Q range [{Q.min():.3g}, {Q.max():.3g}] m^3/s')

        return self.time, Q

    def get_triangles_inside_polygon(self, polygon: list | np.ndarray, verbose: bool = False) -> list | np.ndarray:
        """
        Get list of triangle IDs whose centroids lie within a given polygon.

        Args:
            polygon: List of [x, y] coordinates defining the polygon
            verbose: Whether to print debug information

        Returns:
            List of triangle IDs inside the polygon
        """

        try:
            _ = self.mesh
        except AttributeError:
            from anuga.file.sww import get_mesh_and_quantities_from_file
            self.mesh, _, __ = get_mesh_and_quantities_from_file(self.filename, verbose=verbose)

        triangle_ids = self.mesh.get_triangles_inside_polygon(polygon, verbose=verbose)

        return triangle_ids

    def water_volume(self, per_unit_area=False, triangle_ids=None, polygon=None, verbose=False) -> np.ndarray:
        """
        Compute the water volume associated within a subset of triangles or within a polygon.

        Args:
            per_unit_area: If True, return volume per unit area
            triangle_ids: List of triangle IDs to include
            polygon: Polygon defining area of interest
            verbose: Whether to print debug information

        Returns:
            Numpy array of water volume at each timestep
        """

        if not hasattr(self, "mesh"):
            from anuga.file.sww import get_mesh_and_quantities_from_file
            self.mesh, _, __ = get_mesh_and_quantities_from_file(self.filename, verbose=verbose)

        self.areas = self.mesh.areas



        if(triangle_ids is None and polygon is None):
            triangle_ids=list(range(len(self.xc)))
        elif(triangle_ids is not None):
            pass
        else:
            triangle_ids = self.get_triangles_inside_polygon(polygon, verbose=verbose)


        l=len(self.time)
        areas=self.areas[triangle_ids]

        total_area=areas.sum()
        volume=self.time*0.

        if self.elev.ndim ==1:
            for i in range(l):
                volume[i]=((self.stage[i,triangle_ids]-self.elev[triangle_ids])*(self.stage[i,triangle_ids]>self.elev[triangle_ids])*areas).sum()
        else:
            for i in range(l):
                volume[i]=((self.stage[i,triangle_ids]-self.elev[i,triangle_ids])*(self.stage[i,triangle_ids]>self.elev[i,triangle_ids])*areas).sum()

        if(per_unit_area):
            volume = volume / total_area

        return volume



