# PyFVTool functions and classes

Here, we are collecting an overview the functions and classes needed for setting up set up finite-volume calculations with PyFVTool, organized according to category.

You can also have a look the [API documentation](../api/index.md).


## Initial list (not necessarily complete!)

See also `__init__.py` in the `pyfvtool` module directory.

Taken from test script:

```python
from .mesh import Grid1D, CylindricalGrid1D, SphericalGrid1D
from .mesh import Grid2D, CylindricalGrid2D, PolarGrid2D 
from .mesh import Grid3D, CylindricalGrid3D, SphericalGrid3D
from .advection import convectionTerm, convectionTVDupwindRHSTerm,\
                       convectionUpwindTerm
from .diffusion import diffusionTerm
from .source import linearSourceTerm, constantSourceTerm, transientTerm
from .boundary import BoundaryConditions, boundaryConditionsTerm
from .utilities import fluxLimiter
from .calculus import gradient, gradientFixedBC
from .calculus import divergence, divergenceTerm
from .averaging import linearMean, arithmeticMean, upwindMean,\
                       harmonicMean, geometricMean, tvdMean
from .pdesolver import solveMatrixPDE, solvePDE, solveExplicitPDE
from .cell import CellVariable
from .cell import funceval, celleval
from .cell import cellLocations
from .face import FaceVariable, faceeval
from .face import faceLocations
from .visualization import visualizeCells
```

## Mesh definition

### Cartesian grids: Grid1D, Grid2D, PolarGrid2D

### Cylindrical grids: CylindricalGrid1D, CylindricalGrid2D, CylindricalGrid3D

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
- convectionTerm, convectionUpwindTerm, convectionTVDupwindRHSTerm
	- fluxLimiter
- gradient, divergence, divergenceTerm
- linearSourceTerm, constantSourceTerm
- transientTerm



## Solving the discretized PDE (sparse matrix equation)

### solvePDE

### solveMatrixPDE

### solveExplicitPDE

