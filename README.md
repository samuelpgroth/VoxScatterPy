# VoxScatterPy

The beginnings of a python version of my VoxScatter Matlab repository.

Currently, a Jupyter notebook shows how a circulant preconditioner can be 
used in conjuncation with the discrete dipole approximation for simulations
of light scattering by hexagonal ice plates.

I would like to next replicated the code of Thanos and others of the 
MARIE volume integral equation (VIE) Matlab package. This is a Galerkin implementation
of VIE so uses some sophiticated quadrature routines. Specifically, it
uses the C++ DIRECTFN and DEMCEM packages. I just need to figure out
how to use these in python.
