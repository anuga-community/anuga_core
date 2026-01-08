.. _use_sww_plotter:

.. currentmodule:: anuga

SWW Plotter
===========

The `SWW_plotter` module provides a Python interface for visualizing ANUGA simulation 
output files (.sww format).

Features
--------

- Plot and animate results from ANUGA SWW files such as stage, elevation, depth, xmom, ymom, xvel, 
  and yvel.
- Useful for inspecting model output and presenting results in graphical format and to interogate 
  model results.

Example Usage
-------------

.. code-block:: python

    # Run this code in a Jupyter notebook environment
    import anuga
    import matplotlib.pyplot as plt

    # Allow inline jshtml animations in Jupyter notebooks
    %matplotlib inline
    from matplotlib import rc
    rc('animation', html='jshtml')

    # Enable interactive mode if not in a notebook
    # plt.ion()

    # Create SWW plotter object by opening an SWW file
    splotter = anuga.SWW_plotter('domain.sww')

    # Find min and max depth for plotting
    vmin = splotter.depth.min()
    vmax = splotter.depth.max()

    # Plot depth at last frame.
    splotter.plot_depth_frame(-1, vmin=vmin, vmax=vmax)

    # Plot speed at second frame, collect fig, ax info and change title by hand.
    fig, ax = splotter.plot_speed_frame(1)
    ax.set_title('Speed at second frame')

    # Plot Mesh
    splotter.plot_mesh()

    # Save depth frames for all time steps to be used in animation
    for frame in range(len(splotter.time)):
        print(f'Processing frame {frame+1} of {len(splotter.time)}')
        splotter.save_depth_frame(frame, vmin=vmin, vmax=vmax)

    # Creating depth animation using saved frames
    anim = splotter.make_depth_animation()

    # Saving animation to depth_animation.mp4
    anim.save('depth_animation.mp4', writer='ffmpeg', fps=5)

    # Display the animation in a jupyter notebook
    anim

    # Calculate water volume over entire domain (or a subset of triangles or polygon)
    volume = splotter.water_volume()
    print('Water volume at each time step:', volume)

    # Calculate flow through a polyline
    polyline = [[759195.0, 5912922.0],
                [759250.0, 5912892.0]]
    flow = splotter.get_flow_through_cross_section(polyline)
    print('Flow through cross section at each time step:', flow)


    

See Also
--------

.. autosummary::
   :toctree:

   SWW_plotter

Notes
-----

SWW_plotter is designed to work directly with SWW files output from 
ANUGA domains and provides both static plots and time-based animations 
for analysis and presentation.
