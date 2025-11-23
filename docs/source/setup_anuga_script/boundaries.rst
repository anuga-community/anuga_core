.. currentmodule:: anuga


Setting up the  Boundaries
==========================

Boundaries are an important part of the ANUGA model. They are used to set the
boundary conditions for the model.

To set up the boundaries you first need to create boundary objects. These boundary objects are then
assigned to the edges of the domain using the `set_boundary` method of the :class:`Domain` class.

For example, to set up reflective boundaries on all sides of a rectangular domain, 
you would do the following:

.. code-block:: python

    from anuga import Domain, Reflective_boundary

    # Create a rectangular domain
    domain = Domain(...)

    # Create a reflective boundary object
    Br = Reflective_boundary(domain)

    # Set the boundaries of the domain
    domain.set_boundary({'left': Br, 'right': Br, 'top': Br, 'bottom': Br})

Standard Boundary Types

 * :class:`Reflective_boundary` 
 * :class:`Dirichlet_boundary` 
 * :class:`Transmissive_n_momentum_zero_t_momentum_set_stage_boundary`
 * :class:`Flather_external_stage_zero_velocity_boundary`
 * :class:`File_boundary`
 * :class:`Field_boundary` 

Reference
---------   

.. autoclass:: Reflective_boundary
.. autoclass:: Dirichlet_boundary        
.. autoclass:: Transmissive_n_momentum_zero_t_momentum_set_stage_boundary
.. autoclass:: Flather_external_stage_zero_velocity_boundary
.. autoclass:: File_boundary
.. autoclass:: Field_boundary


   

