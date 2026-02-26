## Plan: Adding Unstructured Mesh Support to PyFVTool

### Design Philosophy

The core idea is to **unify structured and unstructured meshes under a single internal representation** based on face-cell connectivity. All structured meshes (Grid1D, Grid2D, Grid3D and their cylindrical/spherical/polar variants) will construct this connectivity internally, so there is **one code path** for all discretization. The user-friendly API (easy BC specification, easy variable access, simple term assembly, additive matrix+RHS system) is preserved.

### Architecture Overview

```
                    ┌─────────────────┐
                    │   MeshStructure  │  (base class)
                    │  - connectivity  │  (face→cell owner/neighbor)
                    │  - cell_volumes  │
                    │  - face_areas    │
                    │  - face_normals  │
                    │  - cell_centers  │
                    │  - face_centers  │
                    │  - boundary_tags │
                    └────────┬────────┘
               ┌─────────────┼─────────────┐
               │             │             │
     StructuredMesh   UnstructuredMesh2D  UnstructuredMesh3D
     (Grid1D, Grid2D,    (triangular)     (tetrahedral)
      Grid3D, Cyl, Sph,
      Polar variants)
```

### Phase 1: Core Data Model Refactoring

**1.1 New Mesh Connectivity Model (`mesh.py`)**

Introduce a universal connectivity representation on `MeshStructure`:

| Attribute | Type | Shape | Description |
|---|---|---|---|
| `num_cells` | `int` | - | Total interior cells |
| `num_faces` | `int` | - | Total internal + boundary faces |
| `num_ghost_cells` | `int` | - | Number of ghost cells (one per boundary face) |
| `cell_centers` | `ndarray` | `(num_cells, dim)` | Cell centroid coordinates |
| `cell_volumes` | `ndarray` | `(num_cells,)` | Cell volumes |
| `face_centers` | `ndarray` | `(num_faces, dim)` | Face centroid coordinates |
| `face_areas` | `ndarray` | `(num_faces,)` | Face areas |
| `face_normals` | `ndarray` | `(num_faces, dim)` | Unit outward normal (from owner to neighbor) |
| `owner` | `ndarray[int]` | `(num_faces,)` | Owner cell index for each face |
| `neighbor` | `ndarray[int]` | `(num_faces,)` | Neighbor cell index (-1 or ghost index for boundary faces) |
| `boundary_faces` | `ndarray[int]` | `(num_boundary_faces,)` | Indices of faces on the boundary |
| `boundary_tags` | `dict[str, ndarray[int]]` | - | Named groups of boundary face indices (e.g., `"left"`, `"right"`, `"inlet"`) |
| `d_CF` | `ndarray` | `(num_faces,)` | Distance from owner cell center to neighbor cell center |
| `d_Cf` | `ndarray` | `(num_faces,)` | Distance from owner cell center to face center |
| `d_fF` | `ndarray` | `(num_faces,)` | Distance from face center to neighbor cell center |
| `face_interpolation_weight` | `ndarray` | `(num_faces,)` | Weight for linear interpolation from owner to face (= d_fF / d_CF) |
| `non_ortho_correction` | `ndarray` | `(num_faces, dim)` | Non-orthogonality correction vector (face_normal - CF_unit_vector projected onto face) |

Ghost cells: For each boundary face, a ghost cell is created. Ghost cells are numbered after interior cells: indices `num_cells` through `num_cells + num_ghost_cells - 1`. The `neighbor` array stores the ghost cell index for boundary faces. The total system size is `num_cells + num_ghost_cells`.

**1.2 Structured Mesh Constructors**

All existing constructors (`Grid1D(Nx, L)`, `CylindricalGrid2D(Nr, Nz, Lr, Lz)`, etc.) remain but now internally build the connectivity arrays described above. The coordinate label system (`cellcenters.r`, `facecenters.z`, etc.) is preserved as a **convenience layer** that indexes into the `cell_centers[:, dim_index]` and `face_centers[:, dim_index]` arrays.

Structured meshes auto-tag their boundaries: `"left"`, `"right"`, `"bottom"`, `"top"`, `"back"`, `"front"` -- so existing BC patterns work.

**1.3 CellVariable Refactoring (`cell.py`)**

- `_value`: flat `ndarray` of shape `(num_cells + num_ghost_cells,)` (interior cells + ghost cells concatenated).
- `.value` property: returns `_value[:num_cells]` (interior cells only).
- Ghost cell values: `_value[num_cells:]`, indexed by boundary face.
- All arithmetic operators work on flat arrays (simpler than the current N-D approach).
- Boundary conditions are stored on the CellVariable (as today) and applied via the same ghost cell algebra.

**1.4 FaceVariable Refactoring (`face.py`)**

- Single flat `_value` array of shape `(num_faces,)` instead of separate `_xvalue`, `_yvalue`, `_zvalue`.
- For structured meshes, convenience properties (`.xvalue`, `.rvalue`, etc.) index into the flat array using face index ranges stored on the mesh (e.g., `mesh.x_face_indices`, `mesh.y_face_indices`).
- For unstructured meshes, the flat array is the only interface.

**1.5 BoundaryConditions Refactoring (`boundary.py`)**

- `BoundaryConditions` stores a dict mapping boundary tag strings to `BoundaryFace` objects.
- `BoundaryFace` stores `a`, `b`, `c` arrays of length equal to the number of faces in that tag group.
- For structured meshes, the factory auto-creates tags `"left"`, `"right"`, etc. and exposes them as properties: `BC.left`, `BC.right`, etc. (backward-compatible feel).
- For unstructured meshes, users assign BCs by tag: `BC["inlet"].a[:] = 0; BC["inlet"].b[:] = 1; BC["inlet"].c[:] = T_inlet`.
- Ghost cell computation uses face connectivity: for each boundary face `f`, compute `ghost_value` from `phi[owner[f]]` using the Robin formula, just as today but using connectivity arrays instead of axis slicing.

### Phase 2: Discretization Term Rewrite

All discretization functions are rewritten to use the connectivity-based data model. There is **one function per term type** (no more per-geometry/per-dimension variants).

**2.1 Diffusion Term (`diffusion.py`)**

```python
def diffusionTerm(D: FaceVariable) -> csr_array:
    mesh = D.domain
    N = mesh.num_cells + mesh.num_ghost_cells
    # For each internal face: -D_f * A_f / d_CF contribution to owner and neighbor rows
    # Builds COO triplets using mesh.owner, mesh.neighbor arrays
    # Returns csr_array of shape (N, N)
```

Non-orthogonality: The minimum correction approach (deferred correction). The orthogonal part goes into the implicit matrix; the non-orthogonal correction is computed explicitly from the current gradient field and added to the RHS. A `nonortho_correction=True` flag controls this.

**2.2 Convection Terms (`advection.py`)**

- `convectionUpwindTerm(u: FaceVariable)`: One function using `owner`/`neighbor` and face flux sign for upwind direction.
- `convectionTerm(u: FaceVariable)`: Central differencing using `face_interpolation_weight`.
- `convectionTVDupwindRHSTerm(u, phi, FL)`: TVD correction as explicit RHS using face connectivity and flux limiter.

**2.3 Source Terms (`source.py`)**

These are already simple (diagonal matrix + RHS vector) and barely need changes. They operate on cell values directly, sized `(N, N)` and `(N,)`.

**2.4 Boundary Condition Term (`boundary.py`)**

`boundaryConditionsTerm(BC)` iterates over all boundary tags, and for each boundary face writes the Robin discretization equation into the ghost cell row. Corner/edge handling is eliminated (no concept of corners in the connectivity model).

**2.5 Calculus (`calculus.py`)**

- `gradientTerm(phi)`: For each face, `(phi[neighbor] - phi[owner]) / d_CF`. Returns a `FaceVariable`.
- `divergenceTerm(F)`: For each cell, sums `F[face] * face_area * sign` over all faces. Returns an RHS vector. Uses a pre-built sparse cell-face incidence matrix stored on the mesh.

**2.6 Averaging (`averaging.py`)**

All averaging functions use `face_interpolation_weight` and `owner`/`neighbor` indexing:
- `linearMean(phi)`: `w * phi[owner] + (1-w) * phi[neighbor]`
- `harmonicMean(phi)`: `1 / (w/phi[owner] + (1-w)/phi[neighbor])`
- etc.

### Phase 3: Unstructured Mesh Classes

**3.1 `UnstructuredMesh2D` (triangular)**

Constructor options:
- `UnstructuredMesh2D(nodes, cells)` -- direct from arrays
- `UnstructuredMesh2D.from_delaunay(Nx, Ny, Lx, Ly)` -- scipy.spatial.Delaunay on a rectangular domain
- `UnstructuredMesh2D.from_gmsh(filename_or_geo_string)` -- via gmsh Python API
- `UnstructuredMesh2D.from_meshpy(...)` -- via meshpy/Triangle
- `UnstructuredMesh2D.from_dmsh(geo_function)` -- via dmsh

Builds connectivity arrays from the cell-node definition: identifies faces (edges in 2D), computes `owner`/`neighbor`, face areas, face normals, cell volumes (from triangle areas), cell/face centers.

Boundary faces are auto-detected (faces with only one cell) and tagged by geometric location for rectangular domains (e.g., faces on `x=0` get tag `"left"`).

**3.2 `UnstructuredMesh3D` (tetrahedral)**

Same pattern but with tetrahedral cells and triangular faces. Constructors wrap gmsh, meshpy (TetGen), and scipy.spatial.Delaunay.

**3.3 Optional dependencies**

Mesh generators are optional. The import is deferred (inside the factory method). If the library is not installed, a clear error message is given: `"Please install gmsh: uv pip install gmsh"`.

### Phase 4: Solver Updates

**4.1 `pdesolver.py`**

`solvePDE` and `solveMatrixPDE` need minimal changes -- they already work with sparse matrices and flat RHS vectors. The main change is reshaping the solution: instead of `reshape(dims+2)`, the solution is already a flat vector of length `num_cells + num_ghost_cells` that is directly stored in `CellVariable._value`.

`solveExplicitPDE` similarly operates on flat vectors.

**4.2 Non-orthogonal deferred correction loop**

For meshes with non-orthogonality, `solvePDE` can optionally iterate:
```python
for i in range(max_nonortho_corrections):
    solve system
    recompute non-orthogonal correction from new gradient
    update RHS
```
Controlled by a parameter `nonortho_iterations=0` (default: no correction = orthogonal assumption).

### Phase 5: Visualization

**5.1 `visualization.py`**

- 1D: unchanged (line plot).
- Structured 2D: can still use `pcolormesh` (the mesh knows it's structured and can reconstruct the 2D arrays).
- Unstructured 2D: use `matplotlib.tri.Triangulation` with `tripcolor` or `tricontourf`.
- 3D: keep existing surface plots for structured; for unstructured 3D, use boundary face rendering or slice planes.

### Phase 6: Testing

- **Preserve all existing tests** (adapted to the new API where needed). These serve as regression tests.
- **Add new tests:**
  - Unstructured 2D: diffusion on a triangular mesh of a square domain, compared to analytical solution.
  - Unstructured 2D: convection-diffusion benchmark.
  - Unstructured 3D: diffusion on a tetrahedral mesh of a box domain.
  - Structured-to-unstructured equivalence: verify that solving the same problem on a structured grid (via `Grid2D`) and an equivalent unstructured triangulation gives the same answer to within mesh resolution.
  - Non-orthogonal correction convergence test.
  - Boundary tag assignment tests.
  - Mesh import tests (gmsh, meshpy, etc.).

### Phase 7: Documentation and Public API

- Update `__init__.py` to export new mesh classes and any new functions.
- Update `AGENTS.md` with new architecture description.
- Add example notebooks for unstructured meshes.

---

### Implementation Order

| Step | Description | Depends on | Status |
|------|------------|------------|--------|
| 1 | Refactor `MeshStructure` base class with connectivity model | - | ✅ |
| 2 | Refactor `CellVariable` to use flat arrays | Step 1 | ✅ |
| 3 | Refactor `FaceVariable` to use flat arrays | Step 1 | ✅ |
| 4 | Refactor `BoundaryConditions` to use tag-based system | Steps 1-3 | ✅ |
| 5 | Rewrite `diffusionTerm` (single function) | Steps 1-4 | ✅ |
| 6 | Rewrite `source.py` terms | Steps 1-4 | ✅ |
| 7 | Rewrite `boundaryConditionsTerm` | Steps 1-4 | ✅ |
| 8 | Rewrite `advection.py` (convection terms) | Steps 1-4 | ✅ |
| 9 | Rewrite `averaging.py` | Steps 1-3 | ✅ |
| 10 | Rewrite `calculus.py` (gradient, divergence) | Steps 1-3 | ✅ |
| 11 | Update `pdesolver.py` | Steps 5-10 | ✅ |
| 12 | Rebuild structured mesh constructors (Grid1D, Grid2D, etc.) to produce connectivity | Step 1 | ✅ |
| 13 | Implement `UnstructuredMesh2D` with internal Delaunay | Step 1 | ✅ |
| 14 | Add gmsh/meshpy/dmsh interfaces | Step 13 | ⚠️ (basic) |
| 15 | Implement `UnstructuredMesh3D` | Step 1 | ✅ |
 | 16 | Add non-orthogonal deferred correction | Steps 5, 11 | ❌ |
 | 17 | Update `visualization.py` | Steps 2-3, 13 | ✅ |
 | 18 | Update `__init__.py` and public API | All | ✅ |
 | 19 | Adapt existing tests to new API | All | ✅ |
  | 20 | Write new unstructured mesh tests | Steps 13-15 | ✅ (2D & 3D passing) |
 | 21 | Update AGENTS.md and docs | All | ✅ |

---

### Key Design Decisions Summary

1. **Unified representation**: Structured grids are converted to connectivity arrays internally. One code path for all discretization.
2. **Ghost cells preserved**: Every boundary face gets a ghost cell. The Robin BC formula stays the same. System size = `num_cells + num_ghost_cells`.
3. **Clean API break is OK**: But the *feel* stays the same -- users create meshes, set BCs, build terms, solve.
4. **Boundary tags + named faces**: Structured meshes auto-tag `"left"`, `"right"`, etc. Users can also use string tags for unstructured meshes. `BC.left` is syntactic sugar for `BC["left"]`.
5. **Non-orthogonal correction**: Optional deferred correction, disabled by default.
6. **Multiple mesh generators**: scipy.spatial (always available), gmsh, meshpy, dmsh as optional deps.
7. **Coordinate labels preserved**: `cellcenters.r`, `facecenters.z` still work on structured meshes as convenience accessors into the flat arrays.

### Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Performance regression for structured grids (connectivity lookup vs. direct array slicing) | Profile after implementation. If needed, provide a "structured fast path" flag that uses vectorized NumPy operations on the known-regular connectivity pattern. |
| Breaking all 19 existing tests | Adapt tests incrementally as each module is rewritten. Run full test suite after each step. |
| Complex mesh generation dependency management | All generators are optional. scipy.spatial.Delaunay is always available. Clear error messages guide users to install extras. |
| Non-orthogonal correction convergence | Default to off. Document mesh quality requirements. Provide mesh quality metrics (non-orthogonality angle). |
