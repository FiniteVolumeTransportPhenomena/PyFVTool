# PyFVTool functions and classes

Here, we may document the functions and classes needed for setting up set up finite-volume calculations with PyFVTool, organized according to category.

## Initial list (not necessarily complete yet)

See also `__init__.py` in the `pyfvtool` module directory.

Taken from test script:

```python
from pyfvtool import Grid1D, Grid2D, Grid3D
from pyfvtool import CylindricalGrid1D, CylindricalGrid2D
from pyfvtool import createMeshCylindrical3D, PolarGrid2D
from pyfvtool import CellVariable, FaceVariable
from pyfvtool import BoundaryConditions
from pyfvtool import boundaryConditionTerm, diffusionTerm
from pyfvtool import convectionTerm, convectionUpwindTerm, convectionTvdRHSTerm
from pyfvtool import gradientTerm, divergenceTerm
from pyfvtool import linearSourceTerm, constantSourceTerm
from pyfvtool import transientTerm
from pyfvtool import solvePDE, solveExplicitPDE
from pyfvtool import harmonicMean, linearMean, arithmeticMean, geometricMean
from pyfvtool import upwindMean
from pyfvtool import fluxLimiter
from pyfvtool import visualizeCells
```

## Mesh definition

### Cartesian grids: Grid1D, Grid2D, PolarGrid2D

### Cylindrical grids: CylindricalGrid1D, CylindricalGrid2D, createMeshCylindrical3D

### Polar grid: PolarGrid2D


## Variable definition and boundary conditions

### CellVariable

### FaceVariable

### BoundaryConditions

### calculation of mean values

- harmonicMean, linearMean, arithmeticMean, geometricMean
- upwindMean


## Discretization: construction of (sparse) matrix equation

- boundaryConditionTerm, diffusionTerm
- convectionTerm, convectionUpwindTerm, convectionTvdRHSTerm
	- fluxLimiter
- gradientTerm, divergenceTerm
- linearSourceTerm, constantSourceTerm
- transientTerm



## Solving the discretized PDE (sparse matrix equation)

### solvePDE

### solveExplicitPDE

