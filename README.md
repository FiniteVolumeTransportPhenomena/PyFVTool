# PyFVTool: Python toolbox for the finite-volume method

This is a Python implementation of [A. A. Eftekhari](https://github.com/simulkade)'s Matlab/Octave FVM solver [FVTool](http://github.com/simulkade/FVTool) heavily inspired by [FiPy](http://www.ctcms.nist.gov/fipy/) albeit with a fraction of FiPy features. The boundary conditions, however, are much easier (and perhaps more consistent) to implement in PyFVTool.   

This package can discretize and solve the conservative form of transient [convection-diffusion](https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation)-reaction equation(s) with variable velocity field/diffusion coefficients:  

```math
\underbrace{\alpha\frac{\partial\phi}{\partial t}}_{\textrm{Transient term}}+\underbrace{\nabla.\left(\mathbf{u}\phi\right)}_{\text{Advection term}}+\underbrace{\nabla.(-\mathcal{D}\nabla\phi)}_{\text{Diffusion term}}+\underbrace{\beta\phi}_{\text{Linear source term}}+\underbrace{\gamma}_{\text{Constant source term}}=0
```
with the following general form of boundary conditions (by specifying constants `a`, `b`, and `c`):

```math
a\nabla\phi.\mathbf{e}+b\phi=c
```

The [finite-volume](https://en.wikipedia.org/wiki/Finite_volume_method) discretization schemes include:  
  * 1D, 2D and 3D Cartesian and Cylindrical grids (only 1D spherical)
  * Second order (central difference) [diffusion](https://en.wikipedia.org/wiki/Diffusion_equation) terms
  * Second order (central difference), first order ([upwind](https://en.wikipedia.org/wiki/Upwind_scheme)), and [total variation diminishing](https://en.wikipedia.org/wiki/Total_variation_diminishing) (TVD) for advection terms
  * Constant and linear source terms
  * Backward and forward [Euler](https://en.wikipedia.org/wiki/Euler_method) for transient terms
  * [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_boundary_condition), [Neumann](https://en.wikipedia.org/wiki/Neumann_boundary_condition), [Robin](https://en.wikipedia.org/wiki/Robin_boundary_condition), and [periodic](https://en.wikipedia.org/wiki/Periodic_boundary_conditions) boundary conditions
  * (Relatively) easy linearization of nonlinear PDEs
  * Averaging methods (linear, arithmetic, geometric, harmonic, upwind, TVD)
  * Divergence and gradient terms

The code is under active development. Preliminary simulations match analytical solutions. More validation is under way, and the use of this PyFVTool toolbox in ongoing research projects will further consolidate the code and verify its validity.  

There are simple tricks to use the code for systems of nonlinear partial differential equations that take the form of advection-diffusion-reaction equations. There is not much documentation for the code yet (help wanted!) but the example folder is the best place to start. If you know the topic, there is a good chance you will never need any documentations.  

## Installation
An important feature of PyFVTool is that it is 'pure scientific Python' (*i.e.* it needs only Python and the standard scientific computing libraries  `numpy`, `scipy` and `matplotlib` to run). Further optional dependencies may appear in the future, *e.g.*, for increasing the computational speed via optimised numerical libraries, but these will remain optional.

For now, install it directly from the repo using pip. You will need `Python 3.9` or higher and `numpy`, `scipy`, and `matplotlib`:  

```
pip install git+https://github.com/FiniteVolumeTransportPhenomena/PyFVTool.git
```

## Example
Here is a simple example of a 1D transient diffusion equation:

```python
import pyfvtool as pf

# Solving a 1D diffusion equation with a fixed concentration 
# at the left boundary and a closed boundary on the right side

Nx = 20 # number of finite volume cells
Lx = 1.0 # [m] length of the domain 
c_left = 1.0 # left boundary concentration
c_init = 0.0 # initial concentration
D_val = 1e-5 # diffusion coefficient (gas phase)
t_simulation = 7200.0 # [s] simulation time
dt = 60.0 # [s] time step
Nskip = 10 # plot every Nskip-th profile

m1 = pf.createMesh1D(Nx, Lx) # mesh object
bc = pf.createBC(m1) # Neumann boundary condition by default

# switch the left boundary to Dirichlet: fixed concentration
bc.left.a[:] = 0.0
bc.left.b[:] = 1.0
bc.left.c[:] = c_left

# create a cell variable with initial concentration
c_old = pf.createCellVariable(m1, c_init, bc)

# assign diffusivity to cells
D_cell = pf.createCellVariable(m1, D_val)
D_face = pf.geometricMean(D_cell) # average value of diffusivity at the interfaces between cells

# Discretization
Mdiff = pf.diffusionTerm(D_face)
Mbc, RHSbc = pf.boundaryConditionTerm(bc)

# time loop
t = 0
nplot = 0
while t<t_simulation:
    t+=dt
    Mt, RHSt = pf.transientTerm(c_old, dt, 1.0)
    c_new = pf.solvePDE(m1, Mt-Mdiff+Mbc, RHSbc+RHSt)
    c_old.update_value(c_new)
    if (nplot % Nskip == 0):
        pf.visualizeCells(c_old)
    nplot+=1
```
