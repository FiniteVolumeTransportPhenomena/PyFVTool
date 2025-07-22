# PyFVTool: Python toolbox for the finite volume method

PyFVTool discretizes and numerically solves the conservative form of transient [convection-diffusion-reaction](https://en.wikipedia.org/wiki/Convection%E2%80%93diffusion_equation) equations with variable velocity field/diffusion coefficients and source terms. PyFVTool uses the [finite volume method](https://en.wikipedia.org/wiki/Finite_volume_method) (FVM) to do this. 

The partial differential equations that can be solved numerically with PyFVTool have the general form

```math
\underbrace{\alpha\frac{\partial\phi}{\partial t}}_{\textrm{Transient term}}+\underbrace{\nabla \cdot \left(\mathbf{u}\phi\right)}_{\text{Advection term}}+\underbrace{\nabla \cdot (-\mathcal{D}\nabla\phi)}_{\text{Diffusion term}}+\underbrace{\beta\phi}_{\text{Linear source term}}+\underbrace{\gamma}_{\text{Constant source term}}=0
```
with the following general form of boundary conditions (specified by constants `a`, `b`, and `c`):

```math
a\nabla\phi \cdot \mathbf{e}+b\phi=c
```
An important feature of PyFVTool is that it is 'pure scientific Python' (*i.e.* it needs only Python and the standard scientific computing libraries  `numpy`, `scipy` and `matplotlib` to run). Further optional dependencies may appear in the future, *e.g.*, for increasing the computational speed via optimised numerical libraries, but these will remain optional.

PyFVTool is a Python implementation of [A. A. Eftekhari](https://github.com/simulkade)'s Matlab/Octave FVM solver [FVTool](http://github.com/simulkade/FVTool). It was strongly inspired by [FiPy](http://www.ctcms.nist.gov/fipy/), but it has only a fraction of FiPy's features. Boundary conditions, however, are much easier (and arguably more consistently) implemented in PyFVTool. 

PyFVTool is limited to calculations on structured meshes (regular grids). It is oriented to calculation of heat and mass transport phenomena (diffusion-advection-reaction) for the frequent cases where the flow velocity field is already known (or where flow is absent). It is not particularly suited for fluid dynamics (solving Navier-Stokes), which requires implementation of further numerical schemes on top of the current PyFVTool ([simulkade](https://github.com/simulkade) knows how).  For fluid dynamics, other specialized finite-volume codes exist.

The [finite-volume](https://en.wikipedia.org/wiki/Finite_volume_method) discretization schemes in PyFVTool include:  
  * 1D, 2D and 3D Cartesian, cylindrical and spherical grids
  * Second order (central difference) [diffusion](https://en.wikipedia.org/wiki/Diffusion_equation) terms
  * Second order (central difference), first order ([upwind](https://en.wikipedia.org/wiki/Upwind_scheme)), and [total variation diminishing](https://en.wikipedia.org/wiki/Total_variation_diminishing) (TVD) for advection terms
  * Constant and linear source terms
  * Backward and forward [Euler](https://en.wikipedia.org/wiki/Euler_method) for transient terms
  * [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_boundary_condition), [Neumann](https://en.wikipedia.org/wiki/Neumann_boundary_condition), [Robin](https://en.wikipedia.org/wiki/Robin_boundary_condition), and [periodic](https://en.wikipedia.org/wiki/Periodic_boundary_conditions) boundary conditions
  * (Relatively) easy linearization of nonlinear PDEs
  * Averaging methods (linear, arithmetic, geometric, harmonic, upwind, TVD)
  * Divergence and gradient terms

PyFVTool is under active development. Several test cases have been validated to match analytical solutions. More validation is under way, in particular through the use of this toolbox in ongoing research projects.  There is not much documentation for the code yet (help wanted!) but the `example` and `example-notebooks` folders are the best places to start. The latter folder contains annotated Jupyter Notebooks. From the examples, it is easy to understand how to set up finite-volume solvers for heat and mass transfer.

## Installation

### Python environment

We recommend to use PyFVTool with Python 3.12 (no more, no less) and the most recent NumPy and SciPy versions. It is also highly recommended to use the [MiniForge](https://conda-forge.org/download/) / Anaconda / miniconda Python distributions and to set up a specific environment for PyFVTool (we'll call the environment `pyfvtool_user`).

This requires two commands to be launched from the command-line prompt.
```
conda create --name pyfvtool_user python=3.12 numpy scipy matplotlib spyder jupyterlab tqdm

conda activate pyfvtool_user
```

Of course, do not forget to  `conda activate pyfvtool_user`  the environment every time you run Python code that uses PyFVTool.

### Installation of PyFVTool

Install PyFVTool into your specific PyFVTool Conda environment using `pip`. You will need `Python 3.12` (or later) and `numpy` (version 2.0.0 or later), `scipy`, and `matplotlib`, which are provided for by the Conda `pyfvtool_user` environment. The current `pip` install sources PyFVTool directly from GitHub.

```
pip install git+https://github.com/FiniteVolumeTransportPhenomena/PyFVTool.git
```

If you'd like to use PyFVTool in [Google Colab](https://colab.research.google.com/), you can enter the following in the first cell of a Colab Notebook:

```
!pip install git+https://github.com/FiniteVolumeTransportPhenomena/PyFVTool.git
```

This will install PyFVTool in the current Colab instance, and make it available for import in the Notebook.




### Development installation
If you would like to work on the source code, it is possible to install a development version using `pip`. See [`CONTRIBUTING.md`](https://github.com/FiniteVolumeTransportPhenomena/PyFVTool/blob/main/CONTRIBUTING.md)




## Example
Here is a simple example of a 1D transient diffusion equation:

```python
import pyfvtool as pf

# Solving a 1D diffusion equation with a fixed concentration 
# at the left boundary and a closed boundary on the right side


# Calculation parameters
Nx = 20 # number of finite volume cells
Lx = 1.0 # [m] length of the domain 
c_left = 1.0 # left boundary concentration
c_init = 0.0 # initial concentration
D_val = 1e-5 # diffusion coefficient (gas phase)
t_simulation = 7200.0 # [s] simulation time
dt = 60.0 # [s] time step
Nskip = 10 # plot every Nskip-th profile

# Define mesh
mesh = pf.Grid1D(Nx, Lx)

# Create a cell variable with initial concentration
# By default, 'no flux' boundary conditions are applied
c = pf.CellVariable(mesh, c_init)

# Switch the left boundary to Dirichlet: fixed concentration
c.BCs.left.a[:] = 0.0
c.BCs.left.b[:] = 1.0
c.BCs.left.c[:] = c_left
c.apply_BCs()

# Assign diffusivity to cells
D_cell = pf.CellVariable(mesh, D_val)
D_face = pf.geometricMean(D_cell) # average value of diffusivity at the interfaces between cells

# Time loop
t = 0
nplot = 0
while t<t_simulation:
    # Compose discretized terms for matrix equation
    eqnterms = [ pf.transientTerm(c, dt, 1.0),
                -pf.diffusionTerm(D_face)]

    # Solve PDE
    pf.solvePDE(c, eqnterms)
    t+=dt

    if (nplot % Nskip == 0):
        pf.visualizeCells(c)
    nplot+=1
```
