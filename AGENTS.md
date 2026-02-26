# AGENTS.md - PyFVTool

Guidelines for AI coding agents working in this repository.

## Project Overview

PyFVTool is a Python finite volume method toolbox for solving transient
convection-diffusion-reaction PDEs on structured meshes (1D/2D/3D,
Cartesian/cylindrical/spherical) and unstructured meshes (triangular 2D,
tetrahedral 3D). It is a Python port of the original MATLAB/Octave FVTool
with a modern unified connectivity-based architecture.

- **Python**: >=3.12
- **Dependencies**: numpy>=2, scipy, matplotlib
- **Package manager**: uv (uv.lock present)
- **Package layout**: src layout (`src/pyfvtool/`)
- **License**: MIT

## Build & Install

```bash
# Install editable (development) version
pip install --editable .

# Or with uv
uv pip install --editable .
```

## Testing

Test framework: **pytest** (some tests also use `unittest.TestCase`).
Tests live in `tests/`. The full suite takes ~4 minutes (numerical PDE solves).

```bash
# Run all tests (-s shows console output, recommended)
pytest -s

# Run a single test file
pytest -s tests/test_benchmark_1d.py

# Run a single test function
pytest -s tests/test_benchmark_1d.py::TestBenchmark1D::test_dirichlet

# Run tests matching a keyword
pytest -s -k "benchmark"

# Run with verbose output
pytest -sv
```

There is no conftest.py. Test configuration is in `pytest.ini` (minimal:
`minversion = 6.0`, scans all directories for `test_*.py` / `*_test.py`).

### Test patterns

Tests use two patterns:

1. **unittest.TestCase** - Class-based with `self.assertX()` methods:
   ```python
   import unittest
   class TestExample(unittest.TestCase):
       def test_something(self):
           self.assertAlmostEqual(result, expected, places=3)
   ```

2. **Module-level asserts with pytest marker** - Script-style code with a
   `test_success()` function at the end:
   ```python
   complete_success = False
   # ... module-level code with assert statements ...
   complete_success = True
   def test_success():
       assert complete_success
   ```

Most tests solve actual PDEs and compare against analytical solutions.

## Linting & Formatting

**No linters or formatters are configured.** There is no black, ruff, flake8,
mypy, isort, pylint, or pre-commit setup. No CI pipelines exist. Match the
existing code style when making changes.

## Code Style

### Naming Conventions

- **Classes**: PascalCase (`CellVariable`, `FaceVariable`, `MeshStructure`,
  `Grid1D`, `CylindricalGrid2D`, `BoundaryConditions`)
- **Functions**: camelCase (`diffusionTerm`, `convectionTerm`, `solvePDE`,
  `linearMean`, `harmonicMean`, `boundaryConditionsTerm`). This mirrors the
  original MATLAB FVTool and is intentional.
- **Variables**: snake_case mixed with short math notation (`phi_val`,
  `cell_size`, `dx_1`, `Nx`, `Ny`, `rp`, `thetap`)
- **Private members**: underscore prefix (`_value`, `_xvalue`, `_modified`,
  `_getCellVolumes`)

**Important**: Do NOT "fix" camelCase function names to snake_case. The
camelCase convention is deliberate for MATLAB compatibility.

### Imports

- Standard library first, third-party second, local (relative) last
- Local imports use relative dot notation: `from .mesh import MeshStructure`
- The public API is re-exported from `__init__.py`
- Users import as: `import pyfvtool as pf`

```python
import copy
import warnings
from typing import overload

import numpy as np
from scipy.sparse import csr_array

from .mesh import MeshStructure, Grid1D
from .cell import CellVariable
```

### Type Hints

Partial usage. Key public functions and class constructors have type hints;
many internal functions do not. Use `typing.overload` for multiple constructor
signatures. Follow existing patterns — add type hints to new public APIs.

```python
def solvePDE(phi: CellVariable, eqnterms: list, ...) -> CellVariable:
```

### Docstrings

NumPy/numpydoc style where present, with `Parameters`, `Returns`, `Raises`,
`Examples`, `Notes` sections. Coverage is incomplete — add docstrings to new
public functions following numpydoc style.

### Error Handling

- `ValueError` for shape/value mismatches
- `TypeError` for type mismatches
- `NotImplementedError` for abstract methods
- `AttributeError` for coordinate label mismatches
- `warnings.warn()` for non-fatal issues
- No custom exception classes exist

### General Formatting

- No enforced line length (some lines are long)
- Mixed single/double quotes (no consistent preference)
- `#%%` cell markers used in some files (Spyder/VS Code cell mode)
- Heavy use of NumPy broadcasting and scipy.sparse matrix construction

## Architecture

PyFVTool uses a **unified connectivity-based representation** for both structured and unstructured meshes. All mesh types (`Grid1D`, `Grid2D`, `Grid3D`, cylindrical/spherical variants, `UnstructuredMesh2D`, `UnstructuredMesh3D`) inherit from `MeshStructure` and store a common set of connectivity arrays:

- `owner`, `neighbor`: cell indices for each face (ghost cells for boundary)
- `face_areas`, `face_normals`, `face_centers`
- `cell_volumes`, `cell_centers`
- `d_CF`, `d_Cf`, `d_fF`: distances between cells and faces
- `boundary_tags`: named groups of boundary faces
- `non_ortho_correction`: optional non‑orthogonality correction vectors

Discretization functions (`diffusionTerm`, `convectionTerm`, etc.) operate solely on these connectivity arrays, providing a single code path for all mesh types. Ghost cells handle boundary conditions via Robin formulas.

### Module Layout

```
src/pyfvtool/
  __init__.py        # Public API re-exports
  mesh.py            # MeshStructure base class; Grid* (structured) and UnstructuredMesh* classes
  cell.py            # CellVariable with flat array storage (interior + ghost cells)
  face.py            # FaceVariable with flat array storage
  boundary.py        # BoundaryConditions, BoundaryFace, tag‑based BC specification
  advection.py       # Convection/advection term discretization
  diffusion.py       # Diffusion term discretization
  source.py          # Source terms (linear, constant, transient)
  calculus.py        # Gradient and divergence operators
  averaging.py       # Cell‑to‑face averaging (linear, harmonic, geometric, etc.)
  pdesolver.py       # PDE solving (solvePDE, solveMatrixPDE, solveExplicitPDE)
  utilities.py       # Helpers (int_range, fluxLimiter, TrackedArray)
  visualization.py   # Matplotlib‑based plotting (structured & unstructured)
  solvers/           # Optional solver backends (Intel MKL PARDISO)
```

### Typical Workflow

1. **Create a mesh** – `Grid1D(Nx, L)` for structured, `UnstructuredMesh2D.from_delaunay(...)` for triangular, etc.
2. **Define boundary conditions** – `BoundaryConditions(mesh)` with tag‑based assignment (`BC["inlet"].a[:] = ...`).
3. **Create cell variables** – `CellVariable(mesh, value, BC)`.
4. **Build discretization terms** – `diffusionTerm(D)`, `convectionUpwindTerm(u)`, `sourceTerm`, `transientTerm`.
5. **Solve** – `solvePDE(phi, eqnterms)` or `solveMatrixPDE(M, RHS)`.

For unstructured meshes, optional mesh‑generator dependencies (gmsh, meshpy, dmsh) can be used; scipy.spatial.Delaunay is always available.

## Documentation

Sphinx docs in `docs/`. Example Jupyter notebooks in
`docs/source/notebook-examples/` (19 notebooks). Build docs with:

```bash
cd docs && make html
```

Requires: sphinx, myst-parser, nbsphinx, sphinx_rtd_theme, pandoc.
