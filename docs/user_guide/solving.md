# Solving PDEs

Once the equation terms are assembled into a list, pass them to `solvePDE` together with the
`CellVariable` that holds the unknown:

```python
pf.solvePDE(c, equation_terms)
```

`solvePDE` builds the global sparse matrix equation, including the terms related to the boundary conditions, solves the linear system, and writes the updated values back into `c` in-place.



## Time-stepping

PyFVTool uses **implicit (backward) Euler** time integration by default via `transientTerm`. This is unconditionally stable, allowing larger time steps than explicit schemes at the cost of solving a linear system each step.



## Nonlinear PDEs

For nonlinear problems, **Picard iteration** (fixed-point iteration) may be used: linearize the nonlinear coefficients around the current solution, solve, update, and repeat until convergence. [FiPy calls this "sweeping"](https://pages.nist.gov/fipy/en/latest/FAQ.html#iterations-timesteps-and-sweeps-oh-my). In other FVM literature, this may be referred to as "nonlinear iterations".

With PyFVTool, Picard iteration has not yet been used, and there are at present no examples of solving nonlinear PDEs (hello, Navier-Stokes!) in PyFVTool.



## Problem size, computation time, and choice of sparse solver

### Current limitations of PyFVTool

PyFVTool can solve steady-state and time-dependent equations. Steady-state solutions are obtained in a single solver step, whereas time-dependent equations require successively solving the transient equation for each time step. This latter process can involve a large number of time steps. Especially in the latter case, the computation time for each solver step becomes an issue. This is directly related to how fast the sparse matrix solver routine can crunch the matrix equation resulting from the FVM formulation of the transport PDE.

PyFVTool, aiming for simple modeling of transport phenomena, simply calls SciPy's built-in SuperLU sparse solver by default, without any fine-tuning or optimization. This works fine for small 1D or moderate 2D problems (up to about 50,000 finite-volume cells, as a rule of thumb on 2026 hardware), but becomes very slow for larger 2D and 3D problems. 

For those users wishing to use a different, more powerful solver, PyFVTool provides the possiblity to supply an external solver via the `externalsolver` keyword argument for `solvePDE()`. This external solver may then be tuned to work optimally for a specific simulation. Be warned that in this case, you are entering mostly uncharted territory, as this is beyond the basic usage of PyFVTool.


### Opportunities for increasing PyFVTool's computational power 

*See also:* [Sparse solver benchmarks](./solver_benchmarks.md)

Many very powerful sparse matrix solver libraries exist. These are the workhorses of modern scientific computing involving finite-difference, finite-volume and finite-element methods. These libraries are actually specialized computer programs in their own right and need an expert hand to tune them to optimal performance. Further information can be found in the [HPC lecture notes by T. Betcke](https://tbetcke.github.io/hpc_lecture_notes/sparse_solvers_introduction.html) 

As indicated, for 1D and 2D meshes up to about 50,000 FV cells, the built-in single-threaded SuperLU solver is expected to work very well and allows for more-or-less interactive simulations (seconds to minutes). 

As an alternative, the [`scikit-sparse`](https://github.com/scikit-sparse/scikit-sparse) package provides expert access to solvers contained in the well-known SuiteSparse library, such as UMFPACK and CHOLMOD, which are in the same single-threaded category as the built-in SuperLU. [`scikit-sparse`](https://github.com/scikit-sparse/scikit-sparse), under active development as of 2026, should not to be confused with `scikit-umfpack`. We do not recommend `scikit-umfpack` as there are hurdles to its use and it is not well supported across platforms and Python versions.

For larger problems, the computational effort inevitably increases strongly, already for transient problems on 2D meshes of 50,000 ... 1,000,000 cells, but even more so for 3D meshes. Here, the computation time with SuperLU becomes so long that any interactivity is lost. Provisions should be made to separate the computation and the post-processing, i.e. storing the FVM results in a data file (e.g. HDF5) as the calculation proceeds and then processing these results with a separate Python script. Also, the computation should be done on a fixed computer workstation or a node of an HPC cluster instead of on your laptop, so that the latter does not suffer thermal stress and the computation can be left to run overnight.

For these larger systems (50,000 ... 1,000,000 cells, rough estimate), a "multi-processor, shared-memory" direct sparse solver can be used. Until now, The only solver that has been [successfully used with PyFVTool is Intel's oneMKL PARDISO solver](../source/notebook-examples/how-to-use-oneMKL-PARDISO-solver.ipynb), thanks to the `pypardiso` project. The solver works both on Windows and Linux systems, but requires an Intel multi-core processor. 

In the same "multi-processor, shared-memory" category, 'SuperLU_MT' is the multithreaded variant of the aforementioned SuperLU solver. It is not currently interfaced with PyFVTool but is a promising open-source candidate.

Beyond this (say, > 1,000,000 cells, 3D meshes), it may be possible to interface PyFVTool to the MUMPS direct sparse solver, but for now this is considered far beyond the intended use of PyFVTool. If your simulation becomes this big, you should discuss with a friendly computational expert and consider moving your simulation to a whole different software altogether. A change of computational approach could bring salvation: for very large 3D structured grids, the 'alternating direction, implicit' (ADI) method may - under certain conditions - be more efficient than bluntly trying to solve the giant matrix equation with a massively parallel sparse direct solver.


### Further background on solving sparse matrix equations

It is important to bear in mind that the computational effort for solving the sparse system does not only depend on the number of cells but also on the specific computational grid (1D, 2D, or 3D, Cartesian, cylindrical, spherical) and the type of equation being solved (e.g. pure diffusion or diffusion-advection).  The solvers in SciPy require manual selection and tuning for optimal performance. 

This contasts with MATLAB's built-in `x = A\b` sparse solver, which has been engineered to automatically choose its algorithm (UMFPACK, CHOLMOD or even banded solvers) and 'auto-tune' it for optimum performance by cleverly analyzing the input matrix. In SciPy, SuperLU is called with default parameter settings. It solves the sparse system, but not necessarily in the minimum amount of time.

With the built-in SuperLU, as with external solvers, several strategies may still be tried in PyFVTool to make computation more efficient. One promising and unexplored strategy is to re-use, where possible, the symbolic factorization performed by the sparse solver or even the numerical LU factorization. The symbolic factorization can be re-used when the 'sparsity pattern' of the sparse matrix remains the same between time steps. This calls for separating the different steps taken by the sparse solve code: symbolic factorization, numerical factorization and solution of the linear system. Interestingly, the `pypardiso` interface uses this strategy in a basic fashion, and this may partially explain the substantial decrease in computation time when using the MKL PARDISO solver.

As a final remark, we only consider direct sparse solvers for PyFVTool, since the more efficient alternative, iterative solvers, come with their own set of (largely unexplored) head-aches.


