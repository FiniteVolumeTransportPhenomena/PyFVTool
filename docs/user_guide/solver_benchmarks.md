# Sparse solver benchmarks

For selection and tuning of specific sparse solvers, it is good to have relevant benchmarks. As indicated in ['Solving PDEs'](./solving.md), the performance of a solver depends on the type of equation being solved, the mesh geometry, boundary conditions. Different solver algorithms have different strengths and weaknesses, and should be tested for different transport phenomena.

Here, we collect relevant benchmarks, i.e. actually solving PDEs appearing in the study of transport phenomena.


## PyFVTool sparse solver benchmark 260826

Simple wall time measurement.

- SuperLU (single-threaded): `scipy.sparse` default built-in `spsolve` (SciPy 1.18.0)
- PARDISO (multi-threaded): Intel MKL PARDISO via `pypardiso` (MKL 2026.0.1)

Hardware: MOLTECH-Anjou HPC compute node (32+ Gb RAM, Intel Xeon, 20+ threads, Linux)

Calculation: time-dependent advection-diffusion on 2D axisymmetric cylindrical grid $(r, z)$



| Grid size  | FVM cells | SuperLU | PARDISO | Speed-up |
| --- | ---  | --- | --- | --- |
| *Nr x Nz*  | *incl. ghost* | *iter/s* | *iter/s* | |
| 50 x 2070	 | 107744 | 2.365 | 	16.90 | 	7.14 |
| 50 x 4500  | 234104 | 1.118 | 8.524 | 7.62 |
| 100 x 2070 | 211344 | 0.599 | 8.899 | 14.86 |
| 100 x 4500 | 459204 | 0.290 | 4.060 | 14.00 |


SciPy's built-in `scipy.sparse.linalg.spsolve` with SuperLU provides a solid, default baseline. Note that it is single-threaded and does not perform any optimization (e.g. re-using symbolic/numerical factorization). Make sure that SciPy actually uses the built-in SuperLU by calling `scipy.sparse.linalg.use_solver(useUmfpack=False)`. Without this call, SciPy's sparse solving behaviour becomes unpredictable.

Intel's MKL PARDISO solver was used without explicit control of the number of threads used (Using `top`, we estimate between 10 and 20 threads, depending on problem size). The parallelization by PARDISO is quite efficient in our 2D cylindrical case.

