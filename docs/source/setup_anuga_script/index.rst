.. currentmodule:: anuga



Setting up an ANUGA Script
==========================

The comon way to run an ANUGA model is to use a script. This script will setup the
model. Running the script will build the model, evolve the model and concurrently 
save the results.

Setting up an ANUGA model involves five basic steps:

1. Define the computational domain

2. Set the initial conditions

3. Define the boundary conditions

4. Specify the operators

5. Evolve the model

A simple example of an ANUGA script is shown below:

>>> # Setup the domain
>>> domain = anuga.rectangular_cross_domain(10,5)
>>> #
>>> # Set the initial conditions
>>> domain.set_quantity('elevation', function = lambda x,y : x/10)
>>> domain.set_quantity('stage', expression = "elevation + 0.2" )
>>> #
>>> # Define the boundary conditions
>>> Br = anuga.Reflective_boundary(domain)
>>> domain.set_boundary({'left' : Br, 'right' : Br, 'top' : Br, 'bottom' : Br})
>>> #
>>> # Specify the operators
>>> rain = anuga.Rate_operator(domain, rate=lambda t: math.exp( -t**2 ), factor=0.001)
>>> #
>>> # Evolve the model
>>> for t in domain.evolve(yieldstep=1.0, finaltime=10.0):
>>>    pass

For more detailed information on each of these steps, please refer to the 
individual sections in the table of contents.

.. only:: html

.. toctree::
   :maxdepth: 1

   domain
   initial_conditions
   boundaries
   operators
   evolve
   
.. only:: html