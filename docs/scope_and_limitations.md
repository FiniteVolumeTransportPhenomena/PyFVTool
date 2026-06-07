# Scope and limitations

Before digging deeper into PyFVTool, it is useful to consider the scope and the limitations of this software. These are not likely to evolve much, since the developers' time is limited and also because there are other codes that become of interest when your problems move beyond the scope of PyFVTool.


## Scope

PyFVTool is primarily intended as a "pure Python-numpy-scipy" tool for the numerical solution of partial differential equations (PDEs) describing relatively simple heat and mass transport phenomena for which the velocity field of the transporting fluid is already known. Computational fluid dynamics (CFD, i.e. solving the Navier-Stokes equation) is out of the scope of PyFVTool (for CFD, consider [OpenFOAM](https://gitlab.com/openfoam/core/openfoam)).

The goal is to create simple models of relevant transport phenomena yielding quantitative predictions of how heat or mass in the system evolves in space and time.

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

If any of the coefficients $\mathbf{u}$, $\mathcal{D}$, $\alpha$, $\beta$, $\gamma$ is not constant, but a function of the unknown $\phi$, then you have a nonlinear PDE. The numerical solution of such a situation requires additional computation cycles for each time-steps. At present, PyFVTool does not provide any helper functions to deal with this.

The additional computations for nonlinear PDEs should be handled by the user. We hope to include, in a near future, some examples on how to do this.



### The built-in sparse matrix solver becomes slow for larger problems


This is being discussed in [this GitHub issue](https://github.com/FiniteVolumeTransportPhenomena/PyFVTool/issues/45). If you work on a modern Intel x86-64 system, a faster solver is available in the OneMKL math library which can be made to work with PyFVTool (see Examples).

The developers of [FiPy](https://github.com/usnistgov/fipy) have made significant efforts to interface FiPy with high-performance parallel solvers on HPC clusters. If your problem is big, you may find salvation there.

A word of warning: if you run heavy calculations on a laptop computer, its processor will heat up significantly and actual damage to your system may occur if the calculation lasts too long, in spite of the cooling fan spinning. For heavy calculations, it is really worthwhile to set up a dedicated fixed workstation or to find access to a computational cluster. This way, you can leave your calculations running over night, without burdening your laptop.
