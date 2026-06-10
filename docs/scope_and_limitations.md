# Scope and limitations

Before digging deeper into PyFVTool, it is useful to consider its scope and its limitations. These are not likely to evolve much in the near futre, since the developers' time is limited and also because there are other, well-established codes that become of interest when your problems move beyond the scope of PyFVTool.


## Scope

PyFVTool is primarily intended as a "pure Python-numpy-scipy" tool for the numerical solution of partial differential equations (PDEs) describing relatively simple heat and mass transport phenomena for which the velocity field of the transporting fluid is already known. Computational fluid dynamics (CFD, i.e. solving the Navier-Stokes equation) is out of the scope of PyFVTool (for CFD, consider [OpenFOAM](https://gitlab.com/openfoam/core/openfoam)).

The goal is to create simple models of relevant transport phenomena yielding quantitative predictions of how heat or mass in the system evolves in space and time. Indeed, with some effort and good thinking, PyFVTool is capable of some quite amazing calculations, but its primary goal is simple models.

In a typical PyFVTool modeling approach, one would start with a model for which an exact or approximate analytic solution exists. This base model could then be evaluated numerically with PyFVTool and the results verified against the analytic solution. Subsequently, the numerical base model could be adapted to obtain a more realistic representation of the system, e.g. by changing the initial condition, the boundary conditions, or the transport coefficients. 

For systems with approximate analytic solutions that are only valid in a particular range of operational conditions, the numerical PyFVTool model can be used to explore what happens to the system outside of the range where the analytic approximation is valid.


## Limitations

There are a few limitations to be aware of.


### Structured grids only

PyFVTool only supports structured grids (Cartesian 1D, 2D, 3D; Cylindrical 1D, 2D, 3D; Spherical 1D, 3D; Polar 2D). There is no support for those nice unstructured grids (tesselations) which are so often associated with finite-volume and finite-element models.

If you want to do diffusion-advection-reaction modeling with FVM on unstructured grids, still using Python, you could move up to [FiPy](https://github.com/usnistgov/fipy). 


### No direct support for nonlinear PDEs

Referring to the equation solved by PyFVTool,

$$
	\alpha\frac{\partial\phi}{\partial t}
	+ \nabla \cdot \left(\mathbf{u}\phi\right)
	+ \nabla \cdot (-\mathcal{D}\nabla\phi)
	+ \beta\phi
	+ \gamma = 0
$$

If any of the coefficients $\mathbf{u}$, $\mathcal{D}$, $\alpha$, $\beta$, $\gamma$ is not constant, but a function of the unknown $\phi$, then you have a nonlinear PDE. The numerical solution of such a situation requires additional computation cycles for each time step. At present, PyFVTool does not provide any helper functions to deal with this.

The additional computations for nonlinear PDEs should be handled by the user. We hope to include, in a near future, some examples on how to do this. As already indicated, solving the Navier-Stokes equation for computational fluid dynamics is not the objective of PyFVTool. For CFD, use other software, such as [OpenFOAM](https://www.openfoam.com/).



### The built-in sparse matrix solver becomes slow for larger problems

PyFVTool is a library built for "pure and simple" Python-numpy-scipy. It uses SciPy's built-in SuperLU solver to solve the sparse matrix equation that is at the core of the finite-volume computations. This built-in SciPy solver is very robust, but "only" single-threaded and can appear slow with grids containing large numbers of finite-volume cells.

This was discussed in [this GitHub issue](https://github.com/FiniteVolumeTransportPhenomena/PyFVTool/issues/45). If you work on a an Intel x86-64 system, a faster, multi-threaded solver is available in Intel's OneMKL math library which can be made to work with PyFVTool. This is discribed in an [example notebook](./source/notebook-examples/how-to-use-oneMKL-PARDISO-solver.ipynb)). At present, this is the only operational acceleration option for PyFVTool.

Looking beyond PyFVTool, the developers of [FiPy](https://github.com/usnistgov/fipy) have made significant efforts to interface FiPy with high-performance parallel solvers for use on HPC clusters. If your finite-volume problem is big, you may find salvation there. The possibility to work with unstructured meshes in FiPy can further reduce the computational burden in certain cases.

A word of warning: if you run heavy calculations on a laptop computer, its processor will heat up significantly and actual damage to your system may occur if the calculation lasts too long, in spite of the cooling fan spinning. For heavy calculations, it is really worthwhile to set up a dedicated fixed workstation or to find access to a computational cluster. This way, you can leave your calculations running over night, without burdening your laptop.
