# PyFVTool
This is a Python implementation of my Matlab/Octave FVM solver [FVTool](http://github.com/simulkade/FVTool) heavily inspired by [FiPy](http://www.ctcms.nist.gov/fipy/) albeit with a small fraction of FiPy features. The boundary conditions, however, is much easier to implement in PyFVTool.   

This tool can solve a transient convection-diffusion equation with variable velocity field/diffusion coefficients. The discretization schemes include:  
  * central difference diffusion term
  * central difference convection term
  * upwind convection term
  * TVD convection term with various flux limiters
  * transient term
  * Dirichlet, Neumann, Robin, and periodic boundary conditions

The code is still being tested although the preliminary simulations match analytical solutions. More validation will be done soon.  

There are simple tricks to use the code for systems of nonlinear partial differential equations that take the form of advection-diffusion-reaction equations. There are not so much documentations for the code but the example folder is the best place to start. If you know the topic, there is a good chance you will never need any documentations.  

## Installation
For now, install it directly from the repo using pip. You will need `Python 3.5` or higher and `numpy`, `scipy`, and `matplotlib`:  

```
pip install git+https://github.com/simulkade/PyFVTool.git
```

Soon, it will be possible to install from conda-forge and Python Package Index too.

## Example
Here is a simple example of 1D transient diffusion equation:

```python
from pyfvtool import *

# Solving a 1D diffusion equation with a fixed concentration 
# at the left boundary and a closed boundary on the right side
Nx = 20 # number of finite volume cells
Lx = 1.0 # [m] length of the domain 
c_left = 1.0 # left boundary concentration
c_init = 0.0 # initial concentration
D_val = 1e-5 # diffusion coefficient (gas phase)
t_simulation = 3600.0 # [s] simulation time
dt = 60.0 # [s] time step

m1 = createMesh1D(Nx, Lx) # mesh object
bc = createBC(m1) # Neumann boundary condition by default

# switch the left boundary to Dirichlet: fixed concentration
bc.left.a[:] = 0.0
bc.left.b[:] = 1.0
bc.left.c[:] = c_left

# create a cell variable with initial concentration
c_old = createCellVariable(m1, c_init, bc)

# assign diffusivity to cells
D_cell = createCellVariable(m1, D_val)
D_face = geometricMean(D_cell) # average value of diffusivity at the interfaces between cells

# Discretization
Mdiff = diffusionTerm(D_face)
Mbc, RHSbc = boundaryConditionTerm(bc)

# time loop
t = 0
while t<t_simulation:
    t+=dt
    Mt, RHSt = transientTerm(c_old, dt, 1.0)
    c_new = solvePDE(m1, Mt-Mdiff+Mbc, RHSbc+RHSt)
    c_old.update_value(c_new)

visualizeCells(c_old)
```