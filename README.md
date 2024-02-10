# PyFVTool: Python toolbox for the finite volume method

This is a Python implementation of [A. A. Eftekhari](https://github.com/simulkade)'s Matlab/Octave FVM solver [FVTool](http://github.com/simulkade/FVTool). Inspired by [FiPy](http://www.ctcms.nist.gov/fipy/), it has only a fraction of FiPy's features. Boundary conditions, however, are much easier (and arguably more consistent) to implement in PyFVTool. 

PyFVTool can discretize and solve the conservative form of transient [convection-diffusion](https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation)-reaction equation(s) with variable velocity field/diffusion coefficients and source terms:  

```math
\underbrace{\alpha\frac{\partial\phi}{\partial t}}_{\textrm{Transient term}}+\underbrace{\nabla \cdot \left(\mathbf{u}\phi\right)}_{\text{Advection term}}+\underbrace{\nabla \cdot (-\mathcal{D}\nabla\phi)}_{\text{Diffusion term}}+\underbrace{\beta\phi}_{\text{Linear source term}}+\underbrace{\gamma}_{\text{Constant source term}}=0
```
with the following general form of boundary conditions (specified by constants `a`, `b`, and `c`):

```math
a\nabla\phi \cdot \mathbf{e}+b\phi=c
```
PyFVTool is limited to calculations on structured meshes (regular grids). It is oriented to calculation of heat and mass transport phenomena (diffusion-advection-reaction) for the frequent cases where the flow velocity field is already known (or where flow is absent). It is not particularly suited for fluid dynamics (solving Navier-Stokes), which requires implementation of further numerical schemes on top of the current PyFVTool ([simulkade](https://github.com/simulkade) knows how).  For fluid dynamics, other specialized finite-volume codes exist.

The [finite-volume](https://en.wikipedia.org/wiki/Finite_volume_method) discretization schemes include:  
  * 1D, 2D and 3D Cartesian and Cylindrical grids
  * Second order (central difference) [diffusion](https://en.wikipedia.org/wiki/Diffusion_equation) terms
  * Second order (central difference), first order ([upwind](https://en.wikipedia.org/wiki/Upwind_scheme)), and [total variation diminishing](https://en.wikipedia.org/wiki/Total_variation_diminishing) (TVD) for advection terms
  * Constant and linear source terms
  * Backward and forward [Euler](https://en.wikipedia.org/wiki/Euler_method) for transient terms
  * [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_boundary_condition), [Neumann](https://en.wikipedia.org/wiki/Neumann_boundary_condition), [Robin](https://en.wikipedia.org/wiki/Robin_boundary_condition), and [periodic](https://en.wikipedia.org/wiki/Periodic_boundary_conditions) boundary conditions
  * (Relatively) easy linearization of nonlinear PDEs
  * Averaging methods (linear, arithmetic, geometric, harmonic, upwind, TVD)
  * Divergence and gradient terms

An important feature of PyFVTool is that it is 'pure scientific Python' (*i.e.* it needs only Python and the standard scientific computing libraries  `numpy`, `scipy` and `matplotlib` to run). Further optional dependencies may appear in the future, *e.g.*, for increasing the computational speed via optimised numerical libraries, but these will remain optional.

The code is under active development. Preliminary simulations match analytical solutions. More validation is under way, and the use of this PyFVTool toolbox in ongoing research projects will further consolidate the code and verify its validity.   There is not much documentation for the code yet (help wanted!) but the example folder is the best place to start. If you know the topic, there is a good chance you will never need any documentations.  

## Installation
For now, install PyFVTool directly from the GitHub repository using `pip`. You will need `Python 3.9` or higher and `numpy`, `scipy`, and `matplotlib`:  

```
pip install git+https://github.com/FiniteVolumeTransportPhenomena/PyFVTool.git
```

### Working in a conda environment

It is convenient to use the Anaconda/miniconda Python distributions and set up a specific environment for PyFVTool (we'll call the environment `pyfvtool_user`).

This requires three commands to be launched from the command-line prompt.
```
conda create --name pyfvtool_user numpy scipy matplotlib spyder jupyterlab

conda activate pyfvtool_user

pip install git+https://github.com/FiniteVolumeTransportPhenomena/PyFVTool.git
```

Of course, do not forget to  `conda activate pyfvtool_user`  the environment every time you run Python code that uses PyFVTool.


### Development installation
If you would like to work on the source code, it is possible to install a development version using `pip`. See `CONTRIBUTING.md`




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
