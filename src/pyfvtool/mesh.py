# Mesh generation — connectivity-based unified representation
#
# All meshes (structured and unstructured) share the same internal
# representation based on face-cell connectivity arrays.  Structured
# meshes build these arrays from tensor-product grids.

import numpy as np
from warnings import warn
from typing import overload
from .utilities import int_range


# ---------------------------------------------------------------------------
#  Coordinate accessor — provides .x, .r, .z, .theta, .phi attribute access
#  into a (num_points, dim) array, controlled by a label mapping.
# ---------------------------------------------------------------------------


class CoordinateAccessor:
    """Provides named coordinate access into a flat (N, dim) array.

    For structured meshes the label map is e.g. ``{'x': 0, 'y': 1}`` or
    ``{'r': 0, 'z': 1}``.  Accessing ``obj.x`` returns the column
    ``_data[:, 0]``.
    """

    def __init__(self, data: np.ndarray, coordlabels: dict):
        # data shape: (N, dim) or (N,) for 1-D
        self._data = data
        self.coordlabels = coordlabels  # e.g. {'x': 0} or {'r': 0, 'z': 1}

    # ---- generic column getter ------------------------------------------
    def _col(self, label):
        if label not in self.coordlabels:
            raise AttributeError(f"This mesh has no coordinate labeled '{label}'.")
        idx = self.coordlabels[label]
        if self._data.ndim == 1:
            return self._data
        return self._data[:, idx]

    # ---- named properties -----------------------------------------------
    @property
    def x(self):
        return self._col("x")

    @property
    def y(self):
        return self._col("y")

    @property
    def z(self):
        return self._col("z")

    @property
    def r(self):
        return self._col("r")

    @property
    def theta(self):
        return self._col("theta")

    @property
    def phi(self):
        return self._col("phi")

    # ---- convenience: raw array -----------------------------------------
    @property
    def _x(self):
        """Return column 0 (first coordinate)."""
        if self._data.ndim == 1:
            return self._data
        return self._data[:, 0]

    @property
    def _y(self):
        """Return column 1 (second coordinate)."""
        if self._data.ndim < 2 or self._data.shape[1] < 2:
            return np.array([0.0])
        return self._data[:, 1]

    @property
    def _z(self):
        """Return column 2 (third coordinate)."""
        if self._data.ndim < 2 or self._data.shape[1] < 3:
            return np.array([0.0])
        return self._data[:, 2]

    def __str__(self):
        return (
            f"CoordinateAccessor(shape={self._data.shape}, labels={self.coordlabels})"
        )

    def __repr__(self):
        return str(self)


class StructuredCoordinateAccessor(CoordinateAccessor):
    """CoordinateAccessor that returns 1-D tensor-product arrays for structured meshes.

    For backward compatibility, ``.x`` returns the unique 1-D array of
    x-coordinates (length Nx for cell centers, Nx+1 for face locations),
    not the flattened column from the (N, dim) expanded array.

    The ``_x``, ``_y``, ``_z`` convenience properties also return the
    1-D arrays.
    """

    def __init__(self, arrays_1d: list, coordlabels: dict, data_full: np.ndarray):
        # arrays_1d: list of 1-D arrays, one per dimension
        # data_full: the (N, dim) expanded array (kept for internal use)
        super().__init__(data_full, coordlabels)
        self._arrays_1d = arrays_1d  # e.g. [cc_x] or [cc_x, cc_y] etc.

    def _col(self, label):
        if label not in self.coordlabels:
            raise AttributeError(f"This mesh has no coordinate labeled '{label}'.")
        idx = self.coordlabels[label]
        return self._arrays_1d[idx]

    @property
    def _x(self):
        """Return 1-D array for the first coordinate."""
        return self._arrays_1d[0]

    @property
    def _y(self):
        """Return 1-D array for the second coordinate."""
        if len(self._arrays_1d) < 2:
            return np.array([0.0])
        return self._arrays_1d[1]

    @property
    def _z(self):
        """Return 1-D array for the third coordinate."""
        if len(self._arrays_1d) < 3:
            return np.array([0.0])
        return self._arrays_1d[2]

    def __str__(self):
        shapes = [a.shape for a in self._arrays_1d]
        return (
            f"StructuredCoordinateAccessor(shapes={shapes}, labels={self.coordlabels})"
        )


# ---------------------------------------------------------------------------
#  MeshStructure — base class for all meshes
# ---------------------------------------------------------------------------


class MeshStructure:
    """Base class for all finite-volume meshes.

    Stores the connectivity-based representation used by *all*
    discretisation routines (diffusion, convection, source, …).

    Parameters
    ----------
    num_cells : int
        Number of interior cells.
    num_faces : int
        Number of faces (internal + boundary).
    num_ghost_cells : int
        One ghost cell per boundary face.
    dimension : int
        Spatial dimension (1, 2, or 3).
    cell_centers : ndarray, shape (num_cells, dim)
        Centroid coordinates of interior cells.
    cell_volumes : ndarray, shape (num_cells,)
        Volume (area in 2-D, length in 1-D) of interior cells.
    face_centers : ndarray, shape (num_faces, dim)
        Centroid coordinates of faces.
    face_areas : ndarray, shape (num_faces,)
        Area (length in 2-D, 1.0 in 1-D) of each face.
    face_normals : ndarray, shape (num_faces, dim)
        Unit outward normal from *owner* to *neighbor*.
    owner : ndarray[int], shape (num_faces,)
        Index of the owner cell for each face.
    neighbor : ndarray[int], shape (num_faces,)
        Index of the neighbor cell.  For boundary faces this is a
        ghost-cell index >= num_cells.
    boundary_faces : ndarray[int]
        Indices of boundary faces.
    boundary_tags : dict[str, ndarray[int]]
        Named groups of boundary-face indices.
    d_CF : ndarray, shape (num_faces,)
        Distance owner-center to neighbor-center.
    d_Cf : ndarray, shape (num_faces,)
        Distance owner-center to face-center.
    d_fF : ndarray, shape (num_faces,)
        Distance face-center to neighbor-center.
    face_interpolation_weight : ndarray, shape (num_faces,)
        ``d_fF / d_CF`` — weight for linear interp from owner to face.
    non_ortho_correction : ndarray, shape (num_faces, dim)
        Correction vector for non-orthogonality (zero for structured).
    coordlabels : dict
        Mapping of coordinate names to column indices.
    """

    def __init__(
        self,
        *,
        num_cells: int,
        num_faces: int,
        num_ghost_cells: int,
        dimension: int,
        cell_centers: np.ndarray,
        cell_volumes: np.ndarray,
        face_centers: np.ndarray,
        face_areas: np.ndarray,
        face_normals: np.ndarray,
        owner: np.ndarray,
        neighbor: np.ndarray,
        boundary_faces: np.ndarray,
        boundary_tags: dict,
        d_CF: np.ndarray,
        d_Cf: np.ndarray,
        d_fF: np.ndarray,
        face_interpolation_weight: np.ndarray,
        non_ortho_correction: np.ndarray,
        face_nodes: np.ndarray | None = None,
        coordlabels: dict,
        dims: np.ndarray | None = None,
        boundary_normal_sign: dict | None = None,
    ):
        self.num_cells = num_cells
        self.num_faces = num_faces
        self.num_ghost_cells = num_ghost_cells
        self.dimension = dimension
        self._cell_centers = cell_centers  # (num_cells, dim)
        self.cell_volumes = cell_volumes  # (num_cells,)
        self._face_centers = face_centers  # (num_faces, dim)
        self.face_areas = face_areas  # (num_faces,)
        self.face_normals = face_normals  # (num_faces, dim)
        self.owner = owner  # (num_faces,)
        self.neighbor = neighbor  # (num_faces,)
        self.boundary_faces = boundary_faces  # (n_bnd,)
        self.boundary_tags = boundary_tags  # {'left': [...], ...}
        self.d_CF = d_CF
        self.d_Cf = d_Cf
        self.d_fF = d_fF
        self.face_interpolation_weight = face_interpolation_weight
        self.non_ortho_correction = non_ortho_correction
        self.coordlabels = coordlabels
        self._face_nodes = face_nodes
        self.dims = dims  # legacy: np.array([Nx]) etc. for structured
        # Boundary normal sign: maps tag -> scalar sign (+1 or -1).
        # +1 means the owner→neighbor direction matches the outward normal
        # (right/top/front); -1 means it's opposite (left/bottom/back).
        # This is used by boundaryConditionsTerm and apply_BCs to match
        # the original MATLAB FVTool sign convention for BCs.
        if boundary_normal_sign is not None:
            self.boundary_normal_sign = boundary_normal_sign
        else:
            # Default: +1 for all tags (outward normal = owner→neighbor)
            self.boundary_normal_sign = {tag: 1.0 for tag in boundary_tags}

        # Build CoordinateAccessor objects (default: flat columns)
        # Structured grid subclasses override these with
        # StructuredCoordinateAccessor after calling super().__init__.
        self.cellcenters = CoordinateAccessor(cell_centers, coordlabels)
        self.facecenters = CoordinateAccessor(face_centers, coordlabels)

    # read-only property
    @property
    def cellvolume(self):
        return self.cell_volumes

    @property
    def cellsize(self):
        """Cell sizes as a CoordinateAccessor (backward-compat).

        For structured meshes, returns 1-D arrays of cell sizes per
        axis (excluding ghost cells).  For unstructured meshes, raises
        AttributeError.
        """
        if hasattr(self, "_cellsize"):
            return self._cellsize
        raise AttributeError("cellsize is only available for structured meshes.")

    def __repr__(self):
        return str(self)


def _install_structured_accessors(mesh):
    """Replace CoordinateAccessor with StructuredCoordinateAccessor on a structured mesh.

    Must be called after the mesh has been fully initialised (i.e. after
    ``super().__init__()`` and after ``_cell_center_x`` etc. are stored).
    """
    cl = mesh.coordlabels
    # Collect 1-D cell-center arrays
    cc_arrays = [mesh._cell_center_x]
    if hasattr(mesh, "_cell_center_y"):
        cc_arrays.append(mesh._cell_center_y)
    if hasattr(mesh, "_cell_center_z"):
        cc_arrays.append(mesh._cell_center_z)

    # Collect 1-D face-location arrays
    fl_arrays = [mesh._face_loc_x]
    if hasattr(mesh, "_face_loc_y"):
        fl_arrays.append(mesh._face_loc_y)
    if hasattr(mesh, "_face_loc_z"):
        fl_arrays.append(mesh._face_loc_z)

    # Collect 1-D cell-size arrays (interior only, excluding ghost cells)
    cs_arrays = [mesh._cell_size_x[1:-1]]
    if hasattr(mesh, "_cell_size_y"):
        cs_arrays.append(mesh._cell_size_y[1:-1])
    if hasattr(mesh, "_cell_size_z"):
        cs_arrays.append(mesh._cell_size_z[1:-1])

    mesh.cellcenters = StructuredCoordinateAccessor(cc_arrays, cl, mesh._cell_centers)
    mesh.facecenters = StructuredCoordinateAccessor(fl_arrays, cl, mesh._face_centers)
    # For cellsize the per-axis arrays may have different lengths (Nx != Ny)
    # so we can't column_stack them.  Pass the first array as a placeholder
    # for the base-class _data; StructuredCoordinateAccessor._col never uses it.
    mesh._cellsize = StructuredCoordinateAccessor(
        cs_arrays,
        cl,
        cs_arrays[0],
    )


# ---------------------------------------------------------------------------
#  Helper: build connectivity for a structured (tensor-product) mesh
# ---------------------------------------------------------------------------


def _build_structured_connectivity_1d(
    face_loc_x,
    cell_center_x,
    cell_size_x,
    coordlabels,
    volume_scale_fn,
    face_area_scale_fn=None,
):
    """Build connectivity arrays for a 1-D structured mesh.

    Parameters
    ----------
    face_loc_x : 1-D array, length Nx+1
        Face locations along the single axis.
    cell_center_x : 1-D array, length Nx
        Cell centre locations.
    cell_size_x : 1-D array, length Nx+2
        Cell sizes including ghost cells at [0] and [-1].
    coordlabels : dict
        e.g. {'x': 0} or {'r': 0}.
    volume_scale_fn : callable(cell_centers, cell_sizes) -> volumes
        Returns (num_cells,) array of cell volumes.
    face_area_scale_fn : callable(face_locations) -> areas, optional
        Returns (num_faces,) array of face areas.  If ``None``, all
        face areas are set to 1.0 (Cartesian 1-D).
    """
    Nx = cell_center_x.size
    num_cells = Nx
    # Faces: Nx+1 internal/boundary faces  (Nx-1 internal + 2 boundary)
    num_faces = Nx + 1
    num_ghost_cells = 2  # left + right

    # Cell centres  (num_cells, 1)
    cc = cell_center_x.reshape(-1, 1)

    # Face centres  (num_faces, 1)
    fc = face_loc_x.reshape(-1, 1)

    # Face areas
    if face_area_scale_fn is not None:
        face_areas = face_area_scale_fn(face_loc_x)
    else:
        face_areas = np.ones(num_faces)

    # Face normals: point from owner to neighbor.
    # Internal faces and right boundary: +1 (leftward cell -> rightward cell)
    # Left boundary face (face 0): owner=cell 0, neighbor=ghost_left
    #   => normal points from cell 0 toward ghost (leftward) = -1
    face_normals = np.ones((num_faces, 1))
    face_normals[0, 0] = -1.0  # left boundary: owner->ghost is -x direction

    # Owner / neighbor
    # Face i sits between cell i-1 (owner) and cell i (neighbor)
    # except face 0: owner = cell 0, neighbor = ghost_left
    #        face Nx: owner = cell Nx-1, neighbor = ghost_right
    owner = np.empty(num_faces, dtype=int)
    neighbor = np.empty(num_faces, dtype=int)

    ghost_left = num_cells  # index Nx
    ghost_right = num_cells + 1  # index Nx+1

    # Internal faces: face j (j=1..Nx-1) between cell j-1 and cell j
    for j in range(1, Nx):
        owner[j] = j - 1
        neighbor[j] = j

    # Boundary faces
    # face 0: leftmost face — owner is cell 0, neighbor is ghost_left
    owner[0] = 0
    neighbor[0] = ghost_left
    # face Nx: rightmost face — owner is cell Nx-1, neighbor is ghost_right
    owner[Nx] = Nx - 1
    neighbor[Nx] = ghost_right

    # Boundary face indices
    boundary_faces_arr = np.array([0, Nx], dtype=int)
    boundary_tags = {
        "left": np.array([0], dtype=int),
        "right": np.array([Nx], dtype=int),
    }

    # Distances
    # d_Cf[j] = |face_center[j] - cell_center[owner[j]]|
    # d_fF[j] = |cell_center[neighbor[j]] - face_center[j]|
    # d_CF[j] = d_Cf[j] + d_fF[j]
    #
    # For boundary faces, the ghost cell centre is reflected so that
    # the face sits at the midpoint of owner and ghost.
    # ghost_center = 2*face_center - owner_center
    # So d_fF_boundary = d_Cf_boundary, d_CF = 2*d_Cf.

    d_Cf = np.empty(num_faces)
    d_fF = np.empty(num_faces)

    # Internal faces
    for j in range(1, Nx):
        d_Cf[j] = fc[j, 0] - cc[j - 1, 0]
        d_fF[j] = cc[j, 0] - fc[j, 0]

    # Boundary faces
    d_Cf[0] = cc[0, 0] - fc[0, 0]  # owner is cell 0, face is to the left
    d_fF[0] = d_Cf[0]  # ghost mirrors
    d_Cf[Nx] = fc[Nx, 0] - cc[Nx - 1, 0]
    d_fF[Nx] = d_Cf[Nx]

    d_CF = d_Cf + d_fF
    # Avoid division by zero
    d_CF_safe = np.where(d_CF > 0, d_CF, 1.0)
    face_interpolation_weight = d_fF / d_CF_safe

    # Non-orthogonality: zero for structured grids
    non_ortho = np.zeros((num_faces, 1))

    # Cell volumes
    cell_vols = volume_scale_fn(cell_center_x, cell_size_x[1:-1])

    return dict(
        num_cells=num_cells,
        num_faces=num_faces,
        num_ghost_cells=num_ghost_cells,
        dimension=1,
        cell_centers=cc,
        cell_volumes=cell_vols,
        face_centers=fc,
        face_areas=face_areas,
        face_normals=face_normals,
        owner=owner,
        neighbor=neighbor,
        boundary_faces=boundary_faces_arr,
        boundary_tags=boundary_tags,
        boundary_normal_sign={"left": -1.0, "right": 1.0},
        d_CF=d_CF,
        d_Cf=d_Cf,
        d_fF=d_fF,
        face_interpolation_weight=face_interpolation_weight,
        non_ortho_correction=non_ortho,
        coordlabels=coordlabels,
    )


def _build_structured_connectivity_2d(
    face_loc_x,
    face_loc_y,
    cell_center_x,
    cell_center_y,
    cell_size_x,
    cell_size_y,
    coordlabels,
    volume_scale_fn,
    face_area_scale_fn,
):
    """Build connectivity arrays for a 2-D structured mesh.

    The face ordering convention:
      - X-faces first (vertical faces between columns): Ny rows of (Nx+1) faces
        Total: (Nx+1)*Ny faces, indices 0..(Nx+1)*Ny-1
      - Y-faces next (horizontal faces between rows): Nx columns of (Ny+1) faces
        Total: Nx*(Ny+1) faces, indices (Nx+1)*Ny .. end
    """
    Nx = cell_center_x.size
    Ny = cell_center_y.size

    num_cells = Nx * Ny
    n_xfaces = (Nx + 1) * Ny  # faces normal to x-axis
    n_yfaces = Nx * (Ny + 1)  # faces normal to y-axis
    num_faces = n_xfaces + n_yfaces
    # Ghost cells: 2*Ny (left/right columns) + 2*Nx (bottom/top rows)
    num_ghost_cells = 2 * Ny + 2 * Nx

    # Cell index helper: cell(i, j) -> i*Ny + j  (i=0..Nx-1, j=0..Ny-1)
    def cell_idx(i, j):
        return i * Ny + j

    # Ghost cell indices start at num_cells
    # Layout: left(Ny) | right(Ny) | bottom(Nx) | top(Nx)
    ghost_base = num_cells

    def ghost_left(j):
        return ghost_base + j

    def ghost_right(j):
        return ghost_base + Ny + j

    def ghost_bottom(i):
        return ghost_base + 2 * Ny + i

    def ghost_top(i):
        return ghost_base + 2 * Ny + Nx + i

    # ----- Cell centers (num_cells, 2) -----
    # Outer product: i varies first (x), j second (y)
    cx, cy = np.meshgrid(cell_center_x, cell_center_y, indexing="ij")
    cc = np.column_stack([cx.ravel(), cy.ravel()])

    # ----- Cell volumes -----
    cell_vols = volume_scale_fn(
        cell_center_x, cell_center_y, cell_size_x[1:-1], cell_size_y[1:-1]
    )
    cell_vols = cell_vols.ravel()

    # ----- Build face arrays -----
    face_centers = np.empty((num_faces, 2))
    face_areas_arr = np.empty(num_faces)
    face_normals_arr = np.empty((num_faces, 2))
    owner = np.empty(num_faces, dtype=int)
    neighbor = np.empty(num_faces, dtype=int)
    d_Cf = np.empty(num_faces)
    d_fF = np.empty(num_faces)

    bnd_left = []
    bnd_right = []
    bnd_bottom = []
    bnd_top = []

    # Compute face areas using the provided scale function
    x_face_areas, y_face_areas = face_area_scale_fn(
        face_loc_x, face_loc_y, cell_center_x, cell_center_y, cell_size_x, cell_size_y
    )

    # --- X-faces (normal to x-axis) ---
    # x-face(i, j): face between column i-1 and column i, at row j
    # i = 0..Nx, j = 0..Ny-1
    # face index = i*Ny + j
    for i in range(Nx + 1):
        for j in range(Ny):
            f = i * Ny + j
            face_centers[f] = [face_loc_x[i], cell_center_y[j]]
            face_normals_arr[f] = [1.0, 0.0]
            face_areas_arr[f] = x_face_areas[i, j]

            if i == 0:
                # Left boundary — normal points from owner toward ghost (-x)
                owner[f] = cell_idx(0, j)
                neighbor[f] = ghost_left(j)
                face_normals_arr[f] = [-1.0, 0.0]
                d_Cf[f] = cell_center_x[0] - face_loc_x[0]
                d_fF[f] = d_Cf[f]
                bnd_left.append(f)
            elif i == Nx:
                # Right boundary
                owner[f] = cell_idx(Nx - 1, j)
                neighbor[f] = ghost_right(j)
                d_Cf[f] = face_loc_x[Nx] - cell_center_x[Nx - 1]
                d_fF[f] = d_Cf[f]
                bnd_right.append(f)
            else:
                # Internal x-face
                owner[f] = cell_idx(i - 1, j)
                neighbor[f] = cell_idx(i, j)
                d_Cf[f] = face_loc_x[i] - cell_center_x[i - 1]
                d_fF[f] = cell_center_x[i] - face_loc_x[i]

    # --- Y-faces (normal to y-axis) ---
    # y-face(i, j): face between row j-1 and row j, at column i
    # i = 0..Nx-1, j = 0..Ny
    # face index = n_xfaces + i*(Ny+1) + j
    for i in range(Nx):
        for j in range(Ny + 1):
            f = n_xfaces + i * (Ny + 1) + j
            face_centers[f] = [cell_center_x[i], face_loc_y[j]]
            face_normals_arr[f] = [0.0, 1.0]
            face_areas_arr[f] = y_face_areas[i, j]

            if j == 0:
                # Bottom boundary — normal points from owner toward ghost (-y)
                owner[f] = cell_idx(i, 0)
                neighbor[f] = ghost_bottom(i)
                face_normals_arr[f] = [0.0, -1.0]
                d_Cf[f] = cell_center_y[0] - face_loc_y[0]
                d_fF[f] = d_Cf[f]
                bnd_bottom.append(f)
            elif j == Ny:
                # Top boundary
                owner[f] = cell_idx(i, Ny - 1)
                neighbor[f] = ghost_top(i)
                d_Cf[f] = face_loc_y[Ny] - cell_center_y[Ny - 1]
                d_fF[f] = d_Cf[f]
                bnd_top.append(f)
            else:
                # Internal y-face
                owner[f] = cell_idx(i, j - 1)
                neighbor[f] = cell_idx(i, j)
                d_Cf[f] = face_loc_y[j] - cell_center_y[j - 1]
                d_fF[f] = cell_center_y[j] - face_loc_y[j]

    d_CF = d_Cf + d_fF
    d_CF_safe = np.where(d_CF > 0, d_CF, 1.0)
    face_interpolation_weight = d_fF / d_CF_safe

    boundary_faces_arr = np.array(
        bnd_left + bnd_right + bnd_bottom + bnd_top, dtype=int
    )
    boundary_tags = {
        "left": np.array(bnd_left, dtype=int),
        "right": np.array(bnd_right, dtype=int),
        "bottom": np.array(bnd_bottom, dtype=int),
        "top": np.array(bnd_top, dtype=int),
    }

    non_ortho = np.zeros((num_faces, 2))

    return dict(
        num_cells=num_cells,
        num_faces=num_faces,
        num_ghost_cells=num_ghost_cells,
        dimension=2,
        cell_centers=cc,
        cell_volumes=cell_vols,
        face_centers=face_centers,
        face_areas=face_areas_arr,
        face_normals=face_normals_arr,
        owner=owner,
        neighbor=neighbor,
        boundary_faces=boundary_faces_arr,
        boundary_tags=boundary_tags,
        boundary_normal_sign={
            "left": -1.0,
            "right": 1.0,
            "bottom": -1.0,
            "top": 1.0,
        },
        d_CF=d_CF,
        d_Cf=d_Cf,
        d_fF=d_fF,
        face_interpolation_weight=face_interpolation_weight,
        non_ortho_correction=non_ortho,
        coordlabels=coordlabels,
        n_xfaces=n_xfaces,
        n_yfaces=n_yfaces,
    )


def _build_structured_connectivity_3d(
    face_loc_x,
    face_loc_y,
    face_loc_z,
    cell_center_x,
    cell_center_y,
    cell_center_z,
    cell_size_x,
    cell_size_y,
    cell_size_z,
    coordlabels,
    volume_scale_fn,
    face_area_scale_fn,
):
    """Build connectivity arrays for a 3-D structured mesh."""
    Nx = cell_center_x.size
    Ny = cell_center_y.size
    Nz = cell_center_z.size

    num_cells = Nx * Ny * Nz
    n_xfaces = (Nx + 1) * Ny * Nz
    n_yfaces = Nx * (Ny + 1) * Nz
    n_zfaces = Nx * Ny * (Nz + 1)
    num_faces = n_xfaces + n_yfaces + n_zfaces
    # Ghost cells: 2*Ny*Nz (left/right) + 2*Nx*Nz (bottom/top) + 2*Nx*Ny (back/front)
    num_ghost_cells = 2 * Ny * Nz + 2 * Nx * Nz + 2 * Nx * Ny

    def cell_idx(i, j, k):
        return i * (Ny * Nz) + j * Nz + k

    # Ghost cell layout: left | right | bottom | top | back | front
    g = num_cells

    def ghost_left(j, k):
        return g + j * Nz + k

    def ghost_right(j, k):
        return g + Ny * Nz + j * Nz + k

    g2 = g + 2 * Ny * Nz

    def ghost_bottom(i, k):
        return g2 + i * Nz + k

    def ghost_top(i, k):
        return g2 + Nx * Nz + i * Nz + k

    g3 = g2 + 2 * Nx * Nz

    def ghost_back(i, j):
        return g3 + i * Ny + j

    def ghost_front(i, j):
        return g3 + Nx * Ny + i * Ny + j

    # Cell centers (num_cells, 3)
    cx, cy, cz = np.meshgrid(cell_center_x, cell_center_y, cell_center_z, indexing="ij")
    cc = np.column_stack([cx.ravel(), cy.ravel(), cz.ravel()])

    # Cell volumes
    cell_vols = volume_scale_fn(
        cell_center_x,
        cell_center_y,
        cell_center_z,
        cell_size_x[1:-1],
        cell_size_y[1:-1],
        cell_size_z[1:-1],
    )
    cell_vols = cell_vols.ravel()

    # Pre-compute face areas
    x_fa, y_fa, z_fa = face_area_scale_fn(
        face_loc_x,
        face_loc_y,
        face_loc_z,
        cell_center_x,
        cell_center_y,
        cell_center_z,
        cell_size_x,
        cell_size_y,
        cell_size_z,
    )

    # Allocate arrays
    face_centers = np.empty((num_faces, 3))
    face_areas_arr = np.empty(num_faces)
    face_normals_arr = np.empty((num_faces, 3))
    owner_arr = np.empty(num_faces, dtype=int)
    neighbor_arr = np.empty(num_faces, dtype=int)
    d_Cf_arr = np.empty(num_faces)
    d_fF_arr = np.empty(num_faces)

    bnd = {n: [] for n in ("left", "right", "bottom", "top", "back", "front")}

    # --- X-faces: (Nx+1) * Ny * Nz faces ---
    # x-face(i,j,k): face index = i*(Ny*Nz) + j*Nz + k
    for i in range(Nx + 1):
        for j in range(Ny):
            for k in range(Nz):
                f = i * (Ny * Nz) + j * Nz + k
                face_centers[f] = [face_loc_x[i], cell_center_y[j], cell_center_z[k]]
                face_normals_arr[f] = [1.0, 0.0, 0.0]
                face_areas_arr[f] = x_fa[i, j, k]

                if i == 0:
                    # Left boundary — normal points from owner toward ghost (-x)
                    owner_arr[f] = cell_idx(0, j, k)
                    neighbor_arr[f] = ghost_left(j, k)
                    face_normals_arr[f] = [-1.0, 0.0, 0.0]
                    d_Cf_arr[f] = cell_center_x[0] - face_loc_x[0]
                    d_fF_arr[f] = d_Cf_arr[f]
                    bnd["left"].append(f)
                elif i == Nx:
                    owner_arr[f] = cell_idx(Nx - 1, j, k)
                    neighbor_arr[f] = ghost_right(j, k)
                    d_Cf_arr[f] = face_loc_x[Nx] - cell_center_x[Nx - 1]
                    d_fF_arr[f] = d_Cf_arr[f]
                    bnd["right"].append(f)
                else:
                    owner_arr[f] = cell_idx(i - 1, j, k)
                    neighbor_arr[f] = cell_idx(i, j, k)
                    d_Cf_arr[f] = face_loc_x[i] - cell_center_x[i - 1]
                    d_fF_arr[f] = cell_center_x[i] - face_loc_x[i]

    # --- Y-faces: Nx * (Ny+1) * Nz faces ---
    # y-face(i,j,k): face index = n_xfaces + i*((Ny+1)*Nz) + j*Nz + k
    off_y = n_xfaces
    for i in range(Nx):
        for j in range(Ny + 1):
            for k in range(Nz):
                f = off_y + i * ((Ny + 1) * Nz) + j * Nz + k
                face_centers[f] = [cell_center_x[i], face_loc_y[j], cell_center_z[k]]
                face_normals_arr[f] = [0.0, 1.0, 0.0]
                face_areas_arr[f] = y_fa[i, j, k]

                if j == 0:
                    # Bottom boundary — normal points from owner toward ghost (-y)
                    owner_arr[f] = cell_idx(i, 0, k)
                    neighbor_arr[f] = ghost_bottom(i, k)
                    face_normals_arr[f] = [0.0, -1.0, 0.0]
                    d_Cf_arr[f] = cell_center_y[0] - face_loc_y[0]
                    d_fF_arr[f] = d_Cf_arr[f]
                    bnd["bottom"].append(f)
                elif j == Ny:
                    owner_arr[f] = cell_idx(i, Ny - 1, k)
                    neighbor_arr[f] = ghost_top(i, k)
                    d_Cf_arr[f] = face_loc_y[Ny] - cell_center_y[Ny - 1]
                    d_fF_arr[f] = d_Cf_arr[f]
                    bnd["top"].append(f)
                else:
                    owner_arr[f] = cell_idx(i, j - 1, k)
                    neighbor_arr[f] = cell_idx(i, j, k)
                    d_Cf_arr[f] = face_loc_y[j] - cell_center_y[j - 1]
                    d_fF_arr[f] = cell_center_y[j] - face_loc_y[j]

    # --- Z-faces: Nx * Ny * (Nz+1) faces ---
    # z-face(i,j,k): face index = n_xfaces + n_yfaces + i*(Ny*(Nz+1)) + j*(Nz+1) + k
    off_z = n_xfaces + n_yfaces
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz + 1):
                f = off_z + i * (Ny * (Nz + 1)) + j * (Nz + 1) + k
                face_centers[f] = [cell_center_x[i], cell_center_y[j], face_loc_z[k]]
                face_normals_arr[f] = [0.0, 0.0, 1.0]
                face_areas_arr[f] = z_fa[i, j, k]

                if k == 0:
                    # Back boundary — normal points from owner toward ghost (-z)
                    owner_arr[f] = cell_idx(i, j, 0)
                    neighbor_arr[f] = ghost_back(i, j)
                    face_normals_arr[f] = [0.0, 0.0, -1.0]
                    d_Cf_arr[f] = cell_center_z[0] - face_loc_z[0]
                    d_fF_arr[f] = d_Cf_arr[f]
                    bnd["back"].append(f)
                elif k == Nz:
                    owner_arr[f] = cell_idx(i, j, Nz - 1)
                    neighbor_arr[f] = ghost_front(i, j)
                    d_Cf_arr[f] = face_loc_z[Nz] - cell_center_z[Nz - 1]
                    d_fF_arr[f] = d_Cf_arr[f]
                    bnd["front"].append(f)
                else:
                    owner_arr[f] = cell_idx(i, j, k - 1)
                    neighbor_arr[f] = cell_idx(i, j, k)
                    d_Cf_arr[f] = face_loc_z[k] - cell_center_z[k - 1]
                    d_fF_arr[f] = cell_center_z[k] - face_loc_z[k]

    d_CF = d_Cf_arr + d_fF_arr
    d_CF_safe = np.where(d_CF > 0, d_CF, 1.0)
    face_interpolation_weight = d_fF_arr / d_CF_safe

    all_bnd = []
    for v in bnd.values():
        all_bnd.extend(v)
    boundary_faces_arr = np.array(all_bnd, dtype=int)
    boundary_tags = {k: np.array(v, dtype=int) for k, v in bnd.items()}

    non_ortho = np.zeros((num_faces, 3))

    return dict(
        num_cells=num_cells,
        num_faces=num_faces,
        num_ghost_cells=num_ghost_cells,
        dimension=3,
        cell_centers=cc,
        cell_volumes=cell_vols,
        face_centers=face_centers,
        face_areas=face_areas_arr,
        face_normals=face_normals_arr,
        owner=owner_arr,
        neighbor=neighbor_arr,
        boundary_faces=boundary_faces_arr,
        boundary_tags=boundary_tags,
        boundary_normal_sign={
            "left": -1.0,
            "right": 1.0,
            "bottom": -1.0,
            "top": 1.0,
            "back": -1.0,
            "front": 1.0,
        },
        d_CF=d_CF,
        d_Cf=d_Cf_arr,
        d_fF=d_fF_arr,
        face_interpolation_weight=face_interpolation_weight,
        non_ortho_correction=non_ortho,
        coordlabels=coordlabels,
        n_xfaces=n_xfaces,
        n_yfaces=n_yfaces,
        n_zfaces=n_zfaces,
    )


# ---------------------------------------------------------------------------
#  Parse user arguments into face locations / cell centers / cell sizes
# ---------------------------------------------------------------------------


def _scale_angular_dCF(data, face_start, face_end, axis, axis2=None):
    """Scale d_CF/d_Cf/d_fF for angular faces to physical distance.

    For angular coordinates (theta, phi), the coordinate-space distances
    computed by the structured builder are just ``delta_theta`` or
    ``delta_phi``.  The physical distance is:

    - For theta: ``r_owner * delta_theta``
    - For phi (spherical): ``r_owner * sin(theta_owner) * delta_phi``

    This function multiplies d_Cf, d_fF, d_CF by the appropriate
    metric factor, using the owner cell's coordinates from the cell
    centers array stored in ``data``.

    Parameters
    ----------
    data : dict
        Builder output dict (modified in place).
    face_start, face_end : int
        Face index range ``[face_start, face_end)`` for the angular faces.
    axis : int
        Column index in ``cell_centers`` for the first metric coordinate
        (e.g. 0 for r).
    axis2 : int or None
        If not None, second metric coordinate column (e.g. 1 for theta
        in spherical phi-faces), and we multiply by ``sin(coord2)``.
    """
    cc = data["cell_centers"]
    owner = data["owner"]
    d_Cf = data["d_Cf"]
    d_fF = data["d_fF"]
    d_CF = data["d_CF"]

    sl = slice(face_start, face_end)
    ow = owner[sl]
    metric = cc[ow, axis]  # r for theta-faces
    if axis2 is not None:
        metric = metric * np.sin(cc[ow, axis2])  # r*sin(theta) for phi-faces

    d_Cf[sl] *= metric
    d_fF[sl] *= metric
    d_CF[sl] = d_Cf[sl] + d_fF[sl]

    # Recompute face interpolation weight for these faces
    fiw = data["face_interpolation_weight"]
    d_CF_safe = np.where(d_CF[sl] > 0, d_CF[sl], 1.0)
    fiw[sl] = d_fF[sl] / d_CF_safe


def _build_unstructured_connectivity_2d(
    nodes: np.ndarray,
    cells: np.ndarray,
    boundary_tags: dict | None = None,
):
    """Build connectivity arrays for a 2-D unstructured triangular mesh.

    Parameters
    ----------
    nodes : ndarray, shape (N_nodes, 2)
        Vertex coordinates.
    cells : ndarray, shape (N_cells, 3)
        Triangle vertex indices (0-based).
    boundary_tags : dict, optional
        Mapping from tag string to list of boundary face indices.
        If not provided, all boundary faces are tagged as "boundary".

    Returns
    -------
    data : dict
        Connectivity data suitable for MeshStructure constructor.
    """
    import numpy as np

    N_nodes = nodes.shape[0]
    N_cells = cells.shape[0]
    if cells.shape[1] != 3:
        raise ValueError("Cells must be triangles (shape (N,3)).")

    # 1. Cell centers and volumes
    cell_centers = np.mean(nodes[cells], axis=1)  # (N_cells, 2)
    # Area of triangle: 0.5 * |(v1-v0) x (v2-v0)|
    v0 = nodes[cells[:, 0]]
    v1 = nodes[cells[:, 1]]
    v2 = nodes[cells[:, 2]]
    cell_volumes = 0.5 * np.abs(
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    )

    # 2. Build edge -> (cell, edge_index) mapping
    edge_map = {}  # (n1, n2) -> (cell, edge_idx, neighbor_cell?)
    face_info = []  # will store (edge, cell1, cell2, is_boundary)
    # Pre-allocate arrays later after counting faces
    # First pass: count faces and collect boundary/internal info
    # We'll iterate over each cell's three edges
    for cell_idx, tri in enumerate(cells):
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for edge_idx, (a, b) in enumerate(edges):
            if a > b:
                a, b = b, a  # canonical ordering
            key = (a, b)
            if key in edge_map:
                # Second occurrence -> internal face
                prev_cell, prev_edge, _ = edge_map[key]
                # Mark as internal, record both cells
                face_info.append((key, prev_cell, cell_idx, False))
                # Update mapping to indicate face processed
                edge_map[key] = (prev_cell, prev_edge, cell_idx)
            else:
                # First occurrence, may be boundary or later internal
                edge_map[key] = (cell_idx, edge_idx, None)

    # After first pass, collect remaining edges that have only one cell -> boundary
    for key, (cell_idx, edge_idx, other_cell) in edge_map.items():
        if other_cell is None:
            face_info.append((key, cell_idx, -1, True))
        else:
            # internal face already added in first pass (when second cell encountered)
            pass

    N_faces = len(face_info)
    # Ghost cells: one per boundary face
    N_boundary = sum(1 for _, _, _, is_bnd in face_info if is_bnd)
    N_ghost = N_boundary

    # Allocate arrays
    face_centers = np.empty((N_faces, 2))
    face_areas = np.empty(N_faces)
    face_normals = np.empty((N_faces, 2))
    owner = np.empty(N_faces, dtype=int)
    neighbor = np.empty(N_faces, dtype=int)
    d_Cf = np.empty(N_faces)
    d_fF = np.empty(N_faces)
    face_nodes = np.empty((N_faces, 2), dtype=int)

    boundary_faces = []
    boundary_tags_dict = {}

    # Mapping from edge key to face index for boundary tag assignment
    edge_to_face = {}

    # Second pass: compute face geometry
    for f_idx, (key, cell1, cell2, is_boundary) in enumerate(face_info):
        a, b = key
        face_nodes[f_idx] = [a, b]
        # Edge vector
        vec = nodes[b] - nodes[a]
        edge_length = np.linalg.norm(vec)
        face_areas[f_idx] = edge_length
        # Face center (midpoint)
        face_centers[f_idx] = 0.5 * (nodes[a] + nodes[b])

        # Determine owner and neighbor
        if is_boundary:
            owner[f_idx] = cell1
            # Ghost cell index
            ghost_idx = N_cells + len(boundary_faces)
            neighbor[f_idx] = ghost_idx
            boundary_faces.append(f_idx)
        else:
            # internal face: assign owner = cell1, neighbor = cell2
            owner[f_idx] = cell1
            neighbor[f_idx] = cell2

        # Compute outward normal from owner cell
        # Need to know which side of edge the owner cell lies.
        # Compute edge normal (perpendicular) and check direction.
        # Normal = rotated edge vector by 90° counterclockwise: (-dy, dx)
        # (or clockwise depending on orientation).
        # We'll compute normal pointing outward from owner cell.
        # Method: compute centroid of owner triangle, compute vector from edge midpoint to centroid.
        # If dot product with normal is negative, flip normal.
        edge_normal = np.array([-vec[1], vec[0]])  # perpendicular (counterclockwise)
        edge_normal = edge_normal / edge_length  # unit normal
        # Owner cell center
        owner_center = cell_centers[owner[f_idx]]
        # Vector from face center to owner center
        to_owner = owner_center - face_centers[f_idx]
        # If dot(to_owner, edge_normal) > 0, normal points toward owner (inward), flip
        if np.dot(to_owner, edge_normal) > 0:
            edge_normal = -edge_normal
        face_normals[f_idx] = edge_normal

        # Distances
        d_Cf[f_idx] = np.linalg.norm(face_centers[f_idx] - owner_center)
        if is_boundary:
            d_fF[f_idx] = d_Cf[f_idx]  # symmetric ghost placement
        else:
            neighbor_center = cell_centers[neighbor[f_idx]]
            d_fF[f_idx] = np.linalg.norm(neighbor_center - face_centers[f_idx])

        edge_to_face[key] = f_idx

    # Boundary tags
    if boundary_tags is not None:
        # User-provided mapping from tag to list of boundary face indices
        for tag, face_indices in boundary_tags.items():
            boundary_tags_dict[tag] = np.array(face_indices, dtype=int)
    else:
        # Default: all boundary faces tagged "boundary"
        boundary_tags_dict["boundary"] = np.array(boundary_faces, dtype=int)

    # Compute derived quantities
    d_CF = d_Cf + d_fF
    d_CF_safe = np.where(d_CF > 0, d_CF, 1.0)
    face_interpolation_weight = d_fF / d_CF_safe
    non_ortho = np.zeros((N_faces, 2))

    # boundary_normal_sign: +1 for all tags (outward normal matches owner->neighbor direction)
    boundary_normal_sign = {tag: 1.0 for tag in boundary_tags_dict}

    return dict(
        num_cells=N_cells,
        num_faces=N_faces,
        num_ghost_cells=N_ghost,
        dimension=2,
        cell_centers=cell_centers,
        cell_volumes=cell_volumes,
        face_centers=face_centers,
        face_areas=face_areas,
        face_normals=face_normals,
        owner=owner,
        neighbor=neighbor,
        boundary_faces=np.array(boundary_faces, dtype=int),
        boundary_tags=boundary_tags_dict,
        boundary_normal_sign=boundary_normal_sign,
        d_CF=d_CF,
        d_Cf=d_Cf,
        d_fF=d_fF,
        face_interpolation_weight=face_interpolation_weight,
        non_ortho_correction=non_ortho,
        face_nodes=face_nodes,
        coordlabels={"x": 0, "y": 1},
    )


def _build_unstructured_connectivity_3d(
    nodes: np.ndarray,
    cells: np.ndarray,
    boundary_tags: dict | None = None,
):
    """Build connectivity arrays for a 3-D unstructured tetrahedral mesh.

    Parameters
    ----------
    nodes : ndarray, shape (N_nodes, 3)
        Vertex coordinates.
    cells : ndarray, shape (N_cells, 4)
        Tetrahedron vertex indices (0-based).
    boundary_tags : dict, optional
        Mapping from tag string to list of boundary face indices.
        If not provided, all boundary faces are tagged as "boundary".

    Returns
    -------
    data : dict
        Connectivity data suitable for MeshStructure constructor.
    """
    import numpy as np

    N_nodes = nodes.shape[0]
    N_cells = cells.shape[0]
    if cells.shape[1] != 4:
        raise ValueError("Cells must be tetrahedra (shape (N,4)).")

    # 1. Cell centers and volumes
    # Tetrahedron volume: V = |(a-d)·((b-d)×(c-d))| / 6
    # Using formula with determinant
    def tetra_volume(v0, v1, v2, v3):
        return np.abs(np.dot(v0 - v3, np.cross(v1 - v3, v2 - v3))) / 6.0

    cell_centers = np.mean(nodes[cells], axis=1)  # (N_cells, 3)
    cell_volumes = np.empty(N_cells)
    for i, tet in enumerate(cells):
        v0, v1, v2, v3 = nodes[tet]
        cell_volumes[i] = tetra_volume(v0, v1, v2, v3)

    # 2. Build face -> (cell, face_index) mapping
    # Faces are triangles defined by three vertices; canonical ordering (sorted)
    face_map = {}  # (n1, n2, n3) -> (cell, face_idx, neighbor_cell?)
    face_info = []  # will store (face_key, cell1, cell2, is_boundary)
    # Each tetrahedron has 4 triangular faces
    face_indices_of_tet = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for cell_idx, tet in enumerate(cells):
        for face_idx, (a, b, c) in enumerate(face_indices_of_tet):
            va, vb, vc = tet[a], tet[b], tet[c]
            # sort vertices for canonical key
            key = tuple(sorted((va, vb, vc)))
            if key in face_map:
                # Second occurrence -> internal face
                prev_cell, prev_face, _ = face_map[key]
                # Mark as internal, record both cells
                face_info.append((key, prev_cell, cell_idx, False))
                # Update mapping to indicate face processed
                face_map[key] = (prev_cell, prev_face, cell_idx)
            else:
                # First occurrence, may be boundary or later internal
                face_map[key] = (cell_idx, face_idx, None)

    # After first pass, collect remaining faces that have only one cell -> boundary
    for key, (cell_idx, face_idx, other_cell) in face_map.items():
        if other_cell is None:
            face_info.append((key, cell_idx, -1, True))
        else:
            # internal face already added in first pass (when second cell encountered)
            pass

    N_faces = len(face_info)
    # Ghost cells: one per boundary face
    N_boundary = sum(1 for _, _, _, is_bnd in face_info if is_bnd)
    N_ghost = N_boundary

    # Allocate arrays
    face_centers = np.empty((N_faces, 3))
    face_areas = np.empty(N_faces)
    face_normals = np.empty((N_faces, 3))
    owner = np.empty(N_faces, dtype=int)
    neighbor = np.empty(N_faces, dtype=int)
    d_Cf = np.empty(N_faces)
    d_fF = np.empty(N_faces)
    face_nodes = np.empty((N_faces, 3), dtype=int)

    boundary_faces = []
    boundary_tags_dict = {}

    # Mapping from face key to face index for boundary tag assignment
    facekey_to_index = {}

    # Second pass: compute face geometry
    for f_idx, (key, cell1, cell2, is_boundary) in enumerate(face_info):
        a, b, c = key
        face_nodes[f_idx] = [a, b, c]
        # Triangle vertices
        v0, v1, v2 = nodes[a], nodes[b], nodes[c]
        # Face center (centroid)
        face_centers[f_idx] = (v0 + v1 + v2) / 3.0
        # Face area = 0.5 * || (v1-v0) × (v2-v0) ||
        cross = np.cross(v1 - v0, v2 - v0)
        face_areas[f_idx] = 0.5 * np.linalg.norm(cross)
        # Unit normal (direction depends on ordering; we'll adjust later)
        if face_areas[f_idx] > 0:
            face_normals[f_idx] = cross / (2.0 * face_areas[f_idx])
        else:
            face_normals[f_idx] = np.array([0.0, 0.0, 1.0])

        # Determine owner and neighbor
        if is_boundary:
            owner[f_idx] = cell1
            # Ghost cell index
            ghost_idx = N_cells + len(boundary_faces)
            neighbor[f_idx] = ghost_idx
            boundary_faces.append(f_idx)
        else:
            # internal face: assign owner = cell1, neighbor = cell2
            owner[f_idx] = cell1
            neighbor[f_idx] = cell2

        # Ensure normal points outward from owner cell
        # Compute vector from face center to owner cell center
        owner_center = cell_centers[owner[f_idx]]
        to_owner = owner_center - face_centers[f_idx]
        # If dot(to_owner, normal) > 0, normal points toward owner (inward), flip
        if np.dot(to_owner, face_normals[f_idx]) > 0:
            face_normals[f_idx] = -face_normals[f_idx]

        # Distances
        d_Cf[f_idx] = np.linalg.norm(face_centers[f_idx] - owner_center)
        if is_boundary:
            d_fF[f_idx] = d_Cf[f_idx]  # symmetric ghost placement
        else:
            neighbor_center = cell_centers[neighbor[f_idx]]
            d_fF[f_idx] = np.linalg.norm(neighbor_center - face_centers[f_idx])

        facekey_to_index[key] = f_idx

    # Boundary tags
    if boundary_tags is not None:
        # User-provided mapping from tag to list of boundary face indices
        for tag, face_indices in boundary_tags.items():
            boundary_tags_dict[tag] = np.array(face_indices, dtype=int)
    else:
        # Default: all boundary faces tagged "boundary"
        boundary_tags_dict["boundary"] = np.array(boundary_faces, dtype=int)

    # Compute derived quantities
    d_CF = d_Cf + d_fF
    d_CF_safe = np.where(d_CF > 0, d_CF, 1.0)
    face_interpolation_weight = d_fF / d_CF_safe
    non_ortho = np.zeros((N_faces, 3))

    # boundary_normal_sign: +1 for all tags (outward normal matches owner->neighbor direction)
    boundary_normal_sign = {tag: 1.0 for tag in boundary_tags_dict}

    return dict(
        num_cells=N_cells,
        num_faces=N_faces,
        num_ghost_cells=N_ghost,
        dimension=3,
        cell_centers=cell_centers,
        cell_volumes=cell_volumes,
        face_centers=face_centers,
        face_areas=face_areas,
        face_normals=face_normals,
        owner=owner,
        neighbor=neighbor,
        boundary_faces=np.array(boundary_faces, dtype=int),
        boundary_tags=boundary_tags_dict,
        boundary_normal_sign=boundary_normal_sign,
        d_CF=d_CF,
        d_Cf=d_Cf,
        d_fF=d_fF,
        face_interpolation_weight=face_interpolation_weight,
        non_ortho_correction=non_ortho,
        face_nodes=face_nodes,
        coordlabels={"x": 0, "y": 1, "z": 2},
    )


def _parse_1d_args(args):
    """Return (face_loc, cell_center, cell_size) from 1-D constructor args."""
    if len(args) == 1:
        fl = np.asarray(args[0], dtype=float)
        Nx = fl.size - 1
    elif len(args) == 2:
        Nx = int(args[0])
        Lx = float(args[1])
        dx = Lx / Nx
        fl = int_range(0, Nx) * dx
    else:
        raise ValueError("Grid1D expects (Nx, Lx) or (face_locations,)")
    cc = 0.5 * (fl[1:] + fl[:-1])
    cs = np.hstack([fl[1] - fl[0], fl[1:] - fl[:-1], fl[-1] - fl[-2]])
    return fl, cc, cs, Nx


def _parse_2d_args(args):
    """Return (fl_x, fl_y, cc_x, cc_y, cs_x, cs_y, Nx, Ny)."""
    if len(args) == 2:
        fl_x = np.asarray(args[0], dtype=float)
        fl_y = np.asarray(args[1], dtype=float)
        Nx = fl_x.size - 1
        Ny = fl_y.size - 1
    elif len(args) == 4:
        Nx, Ny = int(args[0]), int(args[1])
        Lx, Ly = float(args[2]), float(args[3])
        fl_x = int_range(0, Nx) * (Lx / Nx)
        fl_y = int_range(0, Ny) * (Ly / Ny)
    else:
        raise ValueError(
            "Grid2D expects (Nx, Ny, Lx, Ly) or (face_locationsX, face_locationsY)"
        )
    cc_x = 0.5 * (fl_x[1:] + fl_x[:-1])
    cc_y = 0.5 * (fl_y[1:] + fl_y[:-1])
    _fltocs = lambda fl: np.hstack([fl[1] - fl[0], fl[1:] - fl[:-1], fl[-1] - fl[-2]])
    cs_x = _fltocs(fl_x)
    cs_y = _fltocs(fl_y)
    return fl_x, fl_y, cc_x, cc_y, cs_x, cs_y, Nx, Ny


def _parse_3d_args(args):
    """Return (fl_x, fl_y, fl_z, cc_x, cc_y, cc_z, cs_x, cs_y, cs_z, Nx, Ny, Nz)."""
    if len(args) == 3:
        fl_x = np.asarray(args[0], dtype=float)
        fl_y = np.asarray(args[1], dtype=float)
        fl_z = np.asarray(args[2], dtype=float)
        Nx = fl_x.size - 1
        Ny = fl_y.size - 1
        Nz = fl_z.size - 1
    elif len(args) == 6:
        Nx, Ny, Nz = int(args[0]), int(args[1]), int(args[2])
        Lx, Ly, Lz = float(args[3]), float(args[4]), float(args[5])
        fl_x = int_range(0, Nx) * (Lx / Nx)
        fl_y = int_range(0, Ny) * (Ly / Ny)
        fl_z = int_range(0, Nz) * (Lz / Nz)
    else:
        raise ValueError(
            "Grid3D expects (Nx, Ny, Nz, Lx, Ly, Lz) or "
            "(face_locationsX, face_locationsY, face_locationsZ)"
        )
    cc = lambda fl: 0.5 * (fl[1:] + fl[:-1])
    _fltocs = lambda fl: np.hstack([fl[1] - fl[0], fl[1:] - fl[:-1], fl[-1] - fl[-2]])
    cc_x, cc_y, cc_z = cc(fl_x), cc(fl_y), cc(fl_z)
    cs_x, cs_y, cs_z = _fltocs(fl_x), _fltocs(fl_y), _fltocs(fl_z)
    return fl_x, fl_y, fl_z, cc_x, cc_y, cc_z, cs_x, cs_y, cs_z, Nx, Ny, Nz


# ---------------------------------------------------------------------------
#  Volume / face-area scale functions for each coordinate system
# ---------------------------------------------------------------------------

# ---- 1-D ----


def _vol_cart_1d(cc, cs):
    return cs.copy()


def _fa_cart_1d(fl):
    """Face areas for Cartesian 1-D: all 1.0."""
    return np.ones(fl.size)


def _vol_cyl_1d(cc, cs):
    return 2.0 * np.pi * cs * cc


def _fa_cyl_1d(fl):
    """Face areas for cylindrical 1-D: A = 2*pi*r_f."""
    return 2.0 * np.pi * fl


def _vol_sph_1d(cc, cs):
    return 4.0 * np.pi * cs * cc**2


def _fa_sph_1d(fl):
    """Face areas for spherical 1-D: A = 4*pi*r_f^2."""
    return 4.0 * np.pi * fl**2


# ---- 2-D ----


def _vol_cart_2d(cc_x, cc_y, cs_x, cs_y):
    return cs_x[:, np.newaxis] * cs_y[np.newaxis, :]


def _fa_cart_2d(fl_x, fl_y, cc_x, cc_y, cs_x, cs_y):
    """Face areas for 2-D Cartesian: x-faces have area dy, y-faces have area dx."""
    Nx, Ny = cc_x.size, cc_y.size
    x_fa = np.empty((Nx + 1, Ny))
    for i in range(Nx + 1):
        x_fa[i, :] = cs_y[1:-1]
    y_fa = np.empty((Nx, Ny + 1))
    for j in range(Ny + 1):
        y_fa[:, j] = cs_x[1:-1]
    return x_fa, y_fa


def _vol_cyl_2d(cc_r, cc_z, cs_r, cs_z):
    """Cylindrical 2D (r, z): V = 2*pi*r * dr * dz"""
    return 2.0 * np.pi * cc_r[:, np.newaxis] * cs_r[:, np.newaxis] * cs_z[np.newaxis, :]


def _fa_cyl_2d(fl_r, fl_z, cc_r, cc_z, cs_r, cs_z):
    """Cylindrical 2D face areas.
    r-faces (normal to r): A = 2*pi*r_f * dz
    z-faces (normal to z): A = 2*pi*r_c * dr
    """
    Nr, Nz = cc_r.size, cc_z.size
    r_fa = np.empty((Nr + 1, Nz))
    for i in range(Nr + 1):
        r_fa[i, :] = 2.0 * np.pi * fl_r[i] * cs_z[1:-1]
    z_fa = np.empty((Nr, Nz + 1))
    for j in range(Nz + 1):
        z_fa[:, j] = 2.0 * np.pi * cc_r * cs_r[1:-1]
    return r_fa, z_fa


def _vol_polar_2d(cc_r, cc_theta, cs_r, cs_theta):
    """Polar 2D (r, theta): V = r * dr * dtheta"""
    return cc_r[:, np.newaxis] * cs_r[:, np.newaxis] * cs_theta[np.newaxis, :]


def _fa_polar_2d(fl_r, fl_theta, cc_r, cc_theta, cs_r, cs_theta):
    """Polar 2D face areas.
    r-faces (normal to r): A = r_f * dtheta
    theta-faces (normal to theta): A = dr
    """
    Nr, Ntheta = cc_r.size, cc_theta.size
    r_fa = np.empty((Nr + 1, Ntheta))
    for i in range(Nr + 1):
        r_fa[i, :] = fl_r[i] * cs_theta[1:-1]
    theta_fa = np.empty((Nr, Ntheta + 1))
    for j in range(Ntheta + 1):
        theta_fa[:, j] = cs_r[1:-1]
    return r_fa, theta_fa


# ---- 3-D ----


def _vol_cart_3d(cc_x, cc_y, cc_z, cs_x, cs_y, cs_z):
    return (
        cs_x[:, np.newaxis, np.newaxis]
        * cs_y[np.newaxis, :, np.newaxis]
        * cs_z[np.newaxis, np.newaxis, :]
    )


def _fa_cart_3d(fl_x, fl_y, fl_z, cc_x, cc_y, cc_z, cs_x, cs_y, cs_z):
    Nx, Ny, Nz = cc_x.size, cc_y.size, cc_z.size
    # x-faces: area = dy*dz
    x_fa = np.empty((Nx + 1, Ny, Nz))
    for i in range(Nx + 1):
        x_fa[i, :, :] = cs_y[1:-1][:, np.newaxis] * cs_z[1:-1][np.newaxis, :]
    # y-faces: area = dx*dz
    y_fa = np.empty((Nx, Ny + 1, Nz))
    for j in range(Ny + 1):
        y_fa[:, j, :] = cs_x[1:-1][:, np.newaxis] * cs_z[1:-1][np.newaxis, :]
    # z-faces: area = dx*dy
    z_fa = np.empty((Nx, Ny, Nz + 1))
    for k in range(Nz + 1):
        z_fa[:, :, k] = cs_x[1:-1][:, np.newaxis] * cs_y[1:-1][np.newaxis, :]
    return x_fa, y_fa, z_fa


def _vol_cyl_3d(cc_r, cc_theta, cc_z, cs_r, cs_theta, cs_z):
    """Cylindrical 3D (r, theta, z): V = r*dr*dtheta*dz"""
    return (
        cc_r[:, np.newaxis, np.newaxis]
        * cs_r[:, np.newaxis, np.newaxis]
        * cs_theta[np.newaxis, :, np.newaxis]
        * cs_z[np.newaxis, np.newaxis, :]
    )


def _fa_cyl_3d(fl_r, fl_theta, fl_z, cc_r, cc_theta, cc_z, cs_r, cs_theta, cs_z):
    Nr, Ntheta, Nz = cc_r.size, cc_theta.size, cc_z.size
    # r-faces: A = r_f * dtheta * dz
    x_fa = np.empty((Nr + 1, Ntheta, Nz))
    for i in range(Nr + 1):
        x_fa[i, :, :] = (
            fl_r[i] * cs_theta[1:-1][:, np.newaxis] * cs_z[1:-1][np.newaxis, :]
        )
    # theta-faces: A = dr * dz
    y_fa = np.empty((Nr, Ntheta + 1, Nz))
    for j in range(Ntheta + 1):
        y_fa[:, j, :] = cs_r[1:-1][:, np.newaxis] * cs_z[1:-1][np.newaxis, :]
    # z-faces: A = r * dr * dtheta
    z_fa = np.empty((Nr, Ntheta, Nz + 1))
    for k in range(Nz + 1):
        z_fa[:, :, k] = (
            cc_r[:, np.newaxis]
            * cs_r[1:-1][:, np.newaxis]
            * cs_theta[1:-1][np.newaxis, :]
        )
    return x_fa, y_fa, z_fa


def _vol_sph_3d(cc_r, cc_theta, cc_phi, cs_r, cs_theta, cs_phi):
    """Spherical 3D (r, theta, phi): V = r^2*sin(theta)*dr*dtheta*dphi"""
    return (
        cc_r[:, np.newaxis, np.newaxis] ** 2
        * np.sin(cc_theta[np.newaxis, :, np.newaxis])
        * cs_r[:, np.newaxis, np.newaxis]
        * cs_theta[np.newaxis, :, np.newaxis]
        * cs_phi[np.newaxis, np.newaxis, :]
    )


def _fa_sph_3d(fl_r, fl_theta, fl_phi, cc_r, cc_theta, cc_phi, cs_r, cs_theta, cs_phi):
    Nr, Ntheta, Nphi = cc_r.size, cc_theta.size, cc_phi.size
    # r-faces: A = r_f^2 * sin(theta) * dtheta * dphi
    x_fa = np.empty((Nr + 1, Ntheta, Nphi))
    for i in range(Nr + 1):
        x_fa[i, :, :] = (
            fl_r[i] ** 2
            * np.sin(cc_theta[:, np.newaxis])
            * cs_theta[1:-1][:, np.newaxis]
            * cs_phi[1:-1][np.newaxis, :]
        )
    # theta-faces: A = r * dr * dphi  (no sin(theta) — theta is the direction)
    y_fa = np.empty((Nr, Ntheta + 1, Nphi))
    for j in range(Ntheta + 1):
        y_fa[:, j, :] = (
            cc_r[:, np.newaxis]
            * cs_r[1:-1][:, np.newaxis]
            * np.sin(fl_theta[j])
            * cs_phi[1:-1][np.newaxis, :]
        )
    # phi-faces: A = r * dr * dtheta  (no sin for the direction that varies)
    z_fa = np.empty((Nr, Ntheta, Nphi + 1))
    for k in range(Nphi + 1):
        z_fa[:, :, k] = (
            cc_r[:, np.newaxis]
            * cs_r[1:-1][:, np.newaxis]
            * cs_theta[1:-1][np.newaxis, :]
        )
    return x_fa, y_fa, z_fa


# ---------------------------------------------------------------------------
#  1-D Grid classes
# ---------------------------------------------------------------------------


class Grid1D(MeshStructure):
    """Mesh based on a 1D Cartesian grid (x)

    Instantiation Options
    ---------------------
    - ``Grid1D(Nx, Lx)``
    - ``Grid1D(face_locationsX)``

    Parameters
    ----------
    Nx : int
        Number of cells in the x direction.
    Lx : float
        Length of the domain in the x direction.
    face_locationsX : ndarray
        Locations of the cell faces in the x direction.
    """

    @overload
    def __init__(self, Nx: int, Lx: float): ...
    @overload
    def __init__(self, face_locations: np.ndarray): ...

    def __init__(self, *args):
        fl, cc, cs, Nx = _parse_1d_args(args)
        data = _build_structured_connectivity_1d(
            fl,
            cc,
            cs,
            coordlabels=self._coordlabels(),
            volume_scale_fn=self._volume_scale(),
            face_area_scale_fn=self._face_area_scale(),
        )
        data["dims"] = np.array([Nx], dtype=int)
        # Store structured-mesh metadata for other modules
        self._face_loc_x = fl
        self._cell_center_x = cc
        self._cell_size_x = cs
        super().__init__(**data)
        _install_structured_accessors(self)

    @staticmethod
    def _coordlabels():
        return {"x": 0}

    @staticmethod
    def _volume_scale():
        return _vol_cart_1d

    @staticmethod
    def _face_area_scale():
        return _fa_cart_1d

    def __repr__(self):
        return f"1D Cartesian mesh with {self.dims[0]} cells"

    # Legacy helper: structured cell numbering (for tests that use it)
    def cell_numbers(self):
        Nx = self.dims[0]
        return int_range(0, Nx + 1)


class CylindricalGrid1D(Grid1D):
    """Mesh based on a 1D cylindrical grid (r)

    Instantiation Options
    ---------------------
    - ``CylindricalGrid1D(Nr, Lr)``
    - ``CylindricalGrid1D(face_locationsR)``
    """

    @overload
    def __init__(self, Nr: int, Lr: float): ...
    @overload
    def __init__(self, face_locationsR: np.ndarray): ...

    def __init__(self, *args):
        fl, cc, cs, Nx = _parse_1d_args(args)
        data = _build_structured_connectivity_1d(
            fl,
            cc,
            cs,
            coordlabels={"r": 0},
            volume_scale_fn=_vol_cyl_1d,
            face_area_scale_fn=_fa_cyl_1d,
        )
        data["dims"] = np.array([Nx], dtype=int)
        self._face_loc_x = fl
        self._cell_center_x = cc
        self._cell_size_x = cs
        # Skip Grid1D.__init__, go straight to MeshStructure
        MeshStructure.__init__(self, **data)
        _install_structured_accessors(self)

    def __repr__(self):
        return f"1D Cylindrical (radial) mesh with Nr={self.dims[0]} cells"


class SphericalGrid1D(Grid1D):
    """Mesh based on a 1D spherical grid (r)

    Instantiation Options
    ---------------------
    - ``SphericalGrid1D(Nr, Lr)``
    - ``SphericalGrid1D(face_locationsR)``
    """

    @overload
    def __init__(self, Nr: int, Lr: float): ...
    @overload
    def __init__(self, face_locationsR: np.ndarray): ...

    def __init__(self, *args):
        fl, cc, cs, Nx = _parse_1d_args(args)
        data = _build_structured_connectivity_1d(
            fl,
            cc,
            cs,
            coordlabels={"r": 0},
            volume_scale_fn=_vol_sph_1d,
            face_area_scale_fn=_fa_sph_1d,
        )
        data["dims"] = np.array([Nx], dtype=int)
        self._face_loc_x = fl
        self._cell_center_x = cc
        self._cell_size_x = cs
        MeshStructure.__init__(self, **data)
        _install_structured_accessors(self)

    def __repr__(self):
        return f"1D Spherical mesh with Nr={self.dims[0]} cells"


# ---------------------------------------------------------------------------
#  2-D Grid classes
# ---------------------------------------------------------------------------


class Grid2D(MeshStructure):
    """Mesh based on a 2D Cartesian grid (x, y)

    Instantiation Options
    ---------------------
    - ``Grid2D(Nx, Ny, Lx, Ly)``
    - ``Grid2D(face_locationsX, face_locationsY)``
    """

    @overload
    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float): ...
    @overload
    def __init__(self, face_locationsX: np.ndarray, face_locationsY: np.ndarray): ...

    def __init__(self, *args):
        fl_x, fl_y, cc_x, cc_y, cs_x, cs_y, Nx, Ny = _parse_2d_args(args)
        data = _build_structured_connectivity_2d(
            fl_x,
            fl_y,
            cc_x,
            cc_y,
            cs_x,
            cs_y,
            coordlabels=self._coordlabels(),
            volume_scale_fn=self._volume_scale(),
            face_area_scale_fn=self._face_area_scale(),
        )
        # Store structured-mesh metadata
        data["dims"] = np.array([Nx, Ny], dtype=int)
        self._face_loc_x = fl_x
        self._face_loc_y = fl_y
        self._cell_center_x = cc_x
        self._cell_center_y = cc_y
        self._cell_size_x = cs_x
        self._cell_size_y = cs_y
        self._n_xfaces = data.pop("n_xfaces")
        self._n_yfaces = data.pop("n_yfaces")
        super().__init__(**data)
        _install_structured_accessors(self)

    @staticmethod
    def _coordlabels():
        return {"x": 0, "y": 1}

    @staticmethod
    def _volume_scale():
        return _vol_cart_2d

    @staticmethod
    def _face_area_scale():
        return _fa_cart_2d

    def __repr__(self):
        return f"2D Cartesian mesh with {self.dims[0]}x{self.dims[1]} cells"

    def cell_numbers(self):
        Nx, Ny = self.dims
        G = int_range(0, (Nx + 2) * (Ny + 2) - 1)
        return G.reshape(Nx + 2, Ny + 2)


class CylindricalGrid2D(Grid2D):
    """Mesh based on a 2D cylindrical grid (r, z)

    Instantiation Options
    ---------------------
    - ``CylindricalGrid2D(Nr, Nz, Lr, Lz)``
    - ``CylindricalGrid2D(face_locationsR, face_locationsZ)``
    """

    @overload
    def __init__(self, Nr: int, Nz: int, Lr: float, Lz: float): ...
    @overload
    def __init__(self, face_locationsR: np.ndarray, face_locationsZ: np.ndarray): ...

    def __init__(self, *args):
        fl_x, fl_y, cc_x, cc_y, cs_x, cs_y, Nx, Ny = _parse_2d_args(args)
        data = _build_structured_connectivity_2d(
            fl_x,
            fl_y,
            cc_x,
            cc_y,
            cs_x,
            cs_y,
            coordlabels={"r": 0, "z": 1},
            volume_scale_fn=_vol_cyl_2d,
            face_area_scale_fn=_fa_cyl_2d,
        )
        data["dims"] = np.array([Nx, Ny], dtype=int)
        self._face_loc_x = fl_x
        self._face_loc_y = fl_y
        self._cell_center_x = cc_x
        self._cell_center_y = cc_y
        self._cell_size_x = cs_x
        self._cell_size_y = cs_y
        self._n_xfaces = data.pop("n_xfaces")
        self._n_yfaces = data.pop("n_yfaces")
        MeshStructure.__init__(self, **data)
        _install_structured_accessors(self)

    def __repr__(self):
        return f"2D Cylindrical mesh with Nr={self.dims[0]}xNz={self.dims[1]} cells"


class PolarGrid2D(Grid2D):
    """Mesh based on a 2D polar grid (r, theta)

    Instantiation Options
    ---------------------
    - ``PolarGrid2D(Nr, Ntheta, Lr, Ltheta)``
    - ``PolarGrid2D(face_locationsR, face_locationsTheta)``
    """

    @overload
    def __init__(self, Nr: int, Ntheta: int, Lr: float, Ltheta: float): ...
    @overload
    def __init__(
        self, face_locationsR: np.ndarray, face_locationsTheta: np.ndarray
    ): ...

    def __init__(self, *args):
        if len(args) == 2:
            theta_max = args[1][-1]
        else:
            theta_max = args[3]
        if theta_max > 2 * np.pi:
            warn(
                "Recreate the mesh with an upper bound of 2*pi for "
                "\\theta or there will be unknown consequences!"
            )
        fl_x, fl_y, cc_x, cc_y, cs_x, cs_y, Nx, Ny = _parse_2d_args(args)
        data = _build_structured_connectivity_2d(
            fl_x,
            fl_y,
            cc_x,
            cc_y,
            cs_x,
            cs_y,
            coordlabels={"r": 0, "theta": 1},
            volume_scale_fn=_vol_polar_2d,
            face_area_scale_fn=_fa_polar_2d,
        )
        data["dims"] = np.array([Nx, Ny], dtype=int)
        self._face_loc_x = fl_x
        self._face_loc_y = fl_y
        self._cell_center_x = cc_x
        self._cell_center_y = cc_y
        self._cell_size_x = cs_x
        self._cell_size_y = cs_y
        n_xfaces = data.pop("n_xfaces")
        n_yfaces = data.pop("n_yfaces")
        self._n_xfaces = n_xfaces
        self._n_yfaces = n_yfaces

        # Scale d_CF for theta-direction faces to physical distance.
        # Theta-faces span index range [n_xfaces, n_xfaces+n_yfaces).
        # Physical distance between cell centers sharing a theta-face is
        # r_owner * delta_theta (in coordinate space).  We scale d_Cf, d_fF,
        # and d_CF by the owner cell's r-coordinate.
        _scale_angular_dCF(data, n_xfaces, n_xfaces + n_yfaces, axis=0)

        MeshStructure.__init__(self, **data)
        _install_structured_accessors(self)

    def __repr__(self):
        return f"2D Polar mesh with N_r={self.dims[0]}xN_theta={self.dims[1]} cells"


# ---------------------------------------------------------------------------
#  3-D Grid classes
# ---------------------------------------------------------------------------


class Grid3D(MeshStructure):
    """Mesh based on a 3D Cartesian grid (x, y, z)

    Instantiation Options
    ---------------------
    - ``Grid3D(Nx, Ny, Nz, Lx, Ly, Lz)``
    - ``Grid3D(face_locationsX, face_locationsY, face_locationsZ)``
    """

    @overload
    def __init__(self, Nx: int, Ny: int, Nz: int, Lx: float, Ly: float, Lz: float): ...
    @overload
    def __init__(
        self,
        face_locationsX: np.ndarray,
        face_locationsY: np.ndarray,
        face_locationsZ: np.ndarray,
    ): ...

    def __init__(self, *args):
        parsed = _parse_3d_args(args)
        fl_x, fl_y, fl_z = parsed[0], parsed[1], parsed[2]
        cc_x, cc_y, cc_z = parsed[3], parsed[4], parsed[5]
        cs_x, cs_y, cs_z = parsed[6], parsed[7], parsed[8]
        Nx, Ny, Nz = parsed[9], parsed[10], parsed[11]
        data = _build_structured_connectivity_3d(
            fl_x,
            fl_y,
            fl_z,
            cc_x,
            cc_y,
            cc_z,
            cs_x,
            cs_y,
            cs_z,
            coordlabels=self._coordlabels(),
            volume_scale_fn=self._volume_scale(),
            face_area_scale_fn=self._face_area_scale(),
        )
        data["dims"] = np.array([Nx, Ny, Nz], dtype=int)
        self._face_loc_x = fl_x
        self._face_loc_y = fl_y
        self._face_loc_z = fl_z
        self._cell_center_x = cc_x
        self._cell_center_y = cc_y
        self._cell_center_z = cc_z
        self._cell_size_x = cs_x
        self._cell_size_y = cs_y
        self._cell_size_z = cs_z
        self._n_xfaces = data.pop("n_xfaces")
        self._n_yfaces = data.pop("n_yfaces")
        self._n_zfaces = data.pop("n_zfaces")
        super().__init__(**data)
        _install_structured_accessors(self)

    @staticmethod
    def _coordlabels():
        return {"x": 0, "y": 1, "z": 2}

    @staticmethod
    def _volume_scale():
        return _vol_cart_3d

    @staticmethod
    def _face_area_scale():
        return _fa_cart_3d

    def __repr__(self):
        return (
            f"3D Cartesian mesh with "
            f"Nx={self.dims[0]}xNy={self.dims[1]}xNz={self.dims[2]} cells"
        )

    def cell_numbers(self):
        Nx, Ny, Nz = self.dims
        G = int_range(0, (Nx + 2) * (Ny + 2) * (Nz + 2) - 1)
        return G.reshape(Nx + 2, Ny + 2, Nz + 2)


class CylindricalGrid3D(Grid3D):
    """Mesh based on a 3D cylindrical grid (r, theta, z)

    Instantiation Options
    ---------------------
    - ``CylindricalGrid3D(Nr, Ntheta, Nz, Lr, Ltheta, Lz)``
    - ``CylindricalGrid3D(face_locationsR, face_locationsTheta, face_locationsZ)``
    """

    @overload
    def __init__(
        self, Nr: int, Ntheta: int, Nz: int, Lr: float, Ltheta: float, Lz: float
    ): ...
    @overload
    def __init__(
        self,
        face_locationsR: np.ndarray,
        face_locationsTheta: np.ndarray,
        face_locationsZ: np.ndarray,
    ): ...

    def __init__(self, *args):
        parsed = _parse_3d_args(args)
        fl_x, fl_y, fl_z = parsed[0], parsed[1], parsed[2]
        cc_x, cc_y, cc_z = parsed[3], parsed[4], parsed[5]
        cs_x, cs_y, cs_z = parsed[6], parsed[7], parsed[8]
        Nx, Ny, Nz = parsed[9], parsed[10], parsed[11]

        if len(args) == 3:
            theta_max = args[1][-1]
        else:
            theta_max = args[4]
        if theta_max > 2 * np.pi:
            warn(
                "Recreate the mesh with an upper bound of 2*pi for theta "
                "or there will be unknown consequences!"
            )

        data = _build_structured_connectivity_3d(
            fl_x,
            fl_y,
            fl_z,
            cc_x,
            cc_y,
            cc_z,
            cs_x,
            cs_y,
            cs_z,
            coordlabels={"r": 0, "theta": 1, "z": 2},
            volume_scale_fn=_vol_cyl_3d,
            face_area_scale_fn=_fa_cyl_3d,
        )
        data["dims"] = np.array([Nx, Ny, Nz], dtype=int)
        self._face_loc_x = fl_x
        self._face_loc_y = fl_y
        self._face_loc_z = fl_z
        self._cell_center_x = cc_x
        self._cell_center_y = cc_y
        self._cell_center_z = cc_z
        self._cell_size_x = cs_x
        self._cell_size_y = cs_y
        self._cell_size_z = cs_z
        n_xfaces = data.pop("n_xfaces")
        n_yfaces = data.pop("n_yfaces")
        n_zfaces = data.pop("n_zfaces")
        self._n_xfaces = n_xfaces
        self._n_yfaces = n_yfaces
        self._n_zfaces = n_zfaces

        # Scale d_CF for theta-direction faces to physical distance.
        # Theta-faces are the y-faces: [n_xfaces, n_xfaces+n_yfaces).
        # Physical distance = r_owner * delta_theta.
        _scale_angular_dCF(data, n_xfaces, n_xfaces + n_yfaces, axis=0)

        MeshStructure.__init__(self, **data)
        _install_structured_accessors(self)

    def __repr__(self):
        return (
            f"3D Cylindrical mesh with Nr={self.dims[0]}x"
            f"N_theta={self.dims[1]}xNz={self.dims[2]} cells"
        )


class SphericalGrid3D(Grid3D):
    """Mesh based on a 3D spherical grid (r, theta, phi)

    Instantiation Options
    ---------------------
    - ``SphericalGrid3D(Nr, Ntheta, Nphi, Lr, Ltheta, Lphi)``
    - ``SphericalGrid3D(face_locationsR, face_locationsTheta, face_locationsPhi)``
    """

    @overload
    def __init__(
        self, Nr: int, Ntheta: int, Nphi: int, Lr: float, Ltheta: float, Lphi: float
    ): ...
    @overload
    def __init__(
        self,
        face_locationsR: np.ndarray,
        face_locationsTheta: np.ndarray,
        face_locationsPhi: np.ndarray,
    ): ...

    def __init__(self, *args):
        parsed = _parse_3d_args(args)
        fl_x, fl_y, fl_z = parsed[0], parsed[1], parsed[2]
        cc_x, cc_y, cc_z = parsed[3], parsed[4], parsed[5]
        cs_x, cs_y, cs_z = parsed[6], parsed[7], parsed[8]
        Nx, Ny, Nz = parsed[9], parsed[10], parsed[11]

        if len(args) == 3:
            theta_max = args[1][-1]
            phi_max = args[2][-1]
        elif len(args) == 6:
            theta_max = args[4]
            phi_max = args[5]
        if theta_max > np.pi:
            warn(
                "Recreate the mesh with an upper bound of pi for \\theta"
                " or there will be unknown consequences!"
            )
        if phi_max > 2 * np.pi:
            warn(
                "Recreate the mesh with an upper bound of 2*pi for \\phi"
                " or there will be unknown consequences!"
            )

        data = _build_structured_connectivity_3d(
            fl_x,
            fl_y,
            fl_z,
            cc_x,
            cc_y,
            cc_z,
            cs_x,
            cs_y,
            cs_z,
            coordlabels={"r": 0, "theta": 1, "phi": 2},
            volume_scale_fn=_vol_sph_3d,
            face_area_scale_fn=_fa_sph_3d,
        )
        data["dims"] = np.array([Nx, Ny, Nz], dtype=int)
        self._face_loc_x = fl_x
        self._face_loc_y = fl_y
        self._face_loc_z = fl_z
        self._cell_center_x = cc_x
        self._cell_center_y = cc_y
        self._cell_center_z = cc_z
        self._cell_size_x = cs_x
        self._cell_size_y = cs_y
        self._cell_size_z = cs_z
        n_xfaces = data.pop("n_xfaces")
        n_yfaces = data.pop("n_yfaces")
        n_zfaces = data.pop("n_zfaces")
        self._n_xfaces = n_xfaces
        self._n_yfaces = n_yfaces
        self._n_zfaces = n_zfaces

        # Scale d_CF for theta-direction faces (y-faces) to physical distance.
        # Physical distance = r_owner * delta_theta.
        _scale_angular_dCF(data, n_xfaces, n_xfaces + n_yfaces, axis=0)

        # Scale d_CF for phi-direction faces (z-faces) to physical distance.
        # Physical distance = r_owner * sin(theta_owner) * delta_phi.
        _scale_angular_dCF(
            data,
            n_xfaces + n_yfaces,
            n_xfaces + n_yfaces + n_zfaces,
            axis=0,
            axis2=1,
        )

        MeshStructure.__init__(self, **data)
        _install_structured_accessors(self)

    def __repr__(self):
        return (
            f"3D Spherical mesh with Nr={self.dims[0]}x"
            f"N_theta={self.dims[1]}xN_phi={self.dims[2]} cells"
        )


class UnstructuredMesh2D(MeshStructure):
    """Unstructured triangular mesh in 2-D.

    Instantiation Options
    ---------------------
    - ``UnstructuredMesh2D(nodes, cells, boundary_tags=None)``
    - ``UnstructuredMesh2D.from_delaunay(Nx, Ny, Lx, Ly)``
    - ``UnstructuredMesh2D.from_gmsh(geo_file=None, geo_string=None, physical_group_map=None)``
    - ``UnstructuredMesh2D.from_meshpy(points, facets, holes=None, max_area=None, min_angle=None)``
    - ``UnstructuredMesh2D.from_dmsh(geo, max_edge_length=None)``
    - ``UnstructuredMesh2D.generate_rectangle_with_boundary_refinement(Lx=1.0, Ly=1.0, background_size=0.1, boundary_refinement_distance=0.1, boundary_refinement_size=0.02, physical_group_map=None, refinement_zones=None)``
    """

    def __init__(self, nodes: np.ndarray, cells: np.ndarray, boundary_tags=None):
        """
        Parameters
        ----------
        nodes : ndarray, shape (N_nodes, 2)
            Vertex coordinates.
        cells : ndarray, shape (N_cells, 3)
            Triangle vertex indices (0-based).
        boundary_tags : dict, optional
            Mapping from tag string to list of boundary face indices.
            If not provided, all boundary faces are tagged as "boundary".
        """
        data = _build_unstructured_connectivity_2d(nodes, cells, boundary_tags)
        # No dims attribute for unstructured meshes
        MeshStructure.__init__(self, **data)
        # Store nodes and cells for visualization
        self._nodes = nodes.copy()
        self._cells = cells.copy()
        # No structured coordinate accessors; keep default CoordinateAccessor
        # which provides .x, .y properties via coordlabels {"x":0, "y":1}

    @staticmethod
    def geometric_boundary_tags(mesh, x_range=(0.0, 1.0), y_range=(0.0, 1.0), tol=1e-6):
        """Assign boundary tags based on face center coordinates.

        Parameters
        ----------
        mesh : MeshStructure
        x_range : tuple (xmin, xmax)
        y_range : tuple (ymin, ymax)
        tol : float
            Tolerance for coordinate matching.

        Returns
        -------
        dict
            Mapping tag -> boundary face indices.
        """
        import numpy as np

        xmin, xmax = x_range
        ymin, ymax = y_range
        fc = mesh.facecenters
        tags = {}
        left = []
        right = []
        bottom = []
        top = []
        for f in mesh.boundary_faces:
            x, y = fc.x[f], fc.y[f]
            if abs(x - xmin) < tol:
                left.append(f)
            elif abs(x - xmax) < tol:
                right.append(f)
            elif abs(y - ymin) < tol:
                bottom.append(f)
            elif abs(y - ymax) < tol:
                top.append(f)
            else:
                # Other boundary faces (e.g., interior holes) remain untagged
                pass
        if left:
            tags["left"] = np.array(left, dtype=int)
        if right:
            tags["right"] = np.array(right, dtype=int)
        if bottom:
            tags["bottom"] = np.array(bottom, dtype=int)
        if top:
            tags["top"] = np.array(top, dtype=int)
        # Any remaining boundary faces not assigned? we can assign "other"
        # but for simplicity ignore.
        return tags

    def assign_boundary_tags_from_gmsh_edges(
        self, edge_to_phys, tag_to_idx, physical_group_map=None
    ):
        """Assign boundary tags based on Gmsh physical groups on edges.

        Parameters
        ----------
        edge_to_phys : dict
            Mapping from edge key (node_tag_a, node_tag_b) to physical group tag.
        tag_to_idx : dict
            Mapping from Gmsh node tag to index in self._nodes.
        physical_group_map : dict, optional
            Mapping from physical group tag to user-defined tag name.
            If not provided, physical group tag numbers are used as strings.
        """
        import numpy as np

        # Map physical group tag to tag name
        tag_map = {}
        for phys_tag in set(edge_to_phys.values()):
            if physical_group_map and phys_tag in physical_group_map:
                tag_name = physical_group_map[phys_tag]
            else:
                tag_name = str(phys_tag)
            tag_map[phys_tag] = tag_name

        edge_to_phys_idx = {}
        for (tag_a, tag_b), phys_tag in edge_to_phys.items():
            idx_a = tag_to_idx[tag_a]
            idx_b = tag_to_idx[tag_b]
            key = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
            edge_to_phys_idx[key] = phys_tag

        # Now iterate over boundary faces
        new_tags = {}
        for f_idx in self.boundary_faces:
            a, b = self._face_nodes[f_idx]
            key = (a, b) if a < b else (b, a)
            if key in edge_to_phys_idx:
                phys_tag = edge_to_phys_idx[key]
                tag_name = tag_map[phys_tag]
                new_tags.setdefault(tag_name, []).append(f_idx)
            else:
                # Edge not found? Should not happen for boundary faces.
                # Assign default tag "boundary"
                new_tags.setdefault("boundary", []).append(f_idx)

        # Convert lists to numpy arrays
        for tag_name, faces in new_tags.items():
            new_tags[tag_name] = np.array(faces, dtype=int)

        # Update mesh attributes
        self.boundary_tags = new_tags
        self.boundary_normal_sign = {tag: 1.0 for tag in new_tags}

    @classmethod
    def from_delaunay(cls, Nx: int, Ny: int, Lx: float, Ly: float):
        """Create a triangular mesh of a rectangular domain using Delaunay.

        Generates a regular point set of Nx*Ny points inside [0, Lx] x [0, Ly],
        then triangulates with scipy.spatial.Delaunay.

        Parameters
        ----------
        Nx, Ny : int
            Number of points in x and y directions (total points = Nx*Ny).
        Lx, Ly : float
            Domain dimensions.

        Returns
        -------
        UnstructuredMesh2D
            Triangular mesh covering the rectangle.
        """
        import numpy as np
        from scipy.spatial import Delaunay

        # Regular grid of points
        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel()])

        # Delaunay triangulation
        tri = Delaunay(points)
        nodes = points
        cells = tri.simplices  # (N_cells, 3)

        # Boundary tags: auto-detect left, right, bottom, top by face center coordinates
        # We'll compute boundary faces and assign tags based on geometric location.
        # For simplicity, we'll call the helper with boundary_tags=None (all "boundary").
        # Advanced: compute face centers and tag by proximity to domain edges.
        # We'll implement later.
        boundary_tags = None  # placeholder

        return cls(nodes, cells, boundary_tags)

    @classmethod
    def from_gmsh(cls, geo_file=None, geo_string=None, physical_group_map=None):
        """Create a triangular mesh from a Gmsh .geo file or geometry string.

        Parameters
        ----------
        geo_file : str, optional
            Path to a .geo file.
        geo_string : str, optional
            Geometry definition string (Gmsh .geo format).
        physical_group_map : dict, optional
            Mapping from Gmsh physical group numbers to tag names.
            If not provided, physical group numbers are used as tags.

        Returns
        -------
        UnstructuredMesh2D
            Triangular mesh with boundary tags derived from Gmsh physical groups.

        Raises
        ------
        ImportError
            If gmsh library is not installed.
        ValueError
            If neither geo_file nor geo_string is provided.
        """
        try:
            import gmsh
        except ImportError:
            raise ImportError(
                "Gmsh Python API not installed. Please install with: uv pip install gmsh"
            )
        import numpy as np

        if geo_file is None and geo_string is None:
            raise ValueError("Either geo_file or geo_string must be provided.")

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # suppress console output

        if geo_file:
            gmsh.open(geo_file)
        else:
            gmsh.model.add("geometry")
            gmsh.model.geo.add(geo_string)
            gmsh.model.geo.synchronize()

        # Generate 2D mesh
        gmsh.model.mesh.generate(2)

        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        # reshape coords: (x1, y1, z1, x2, y2, z2, ...) -> (N, 3)
        coords = np.array(node_coords).reshape(-1, 3)
        # keep only x,y (discard z for 2D)
        nodes = coords[:, :2]
        # map gmsh node tag to index
        tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

        # Get triangular elements (type 2)
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()
        triangles = []
        for etype, tags, node_tags_per_elem in zip(
            elem_types, elem_tags, elem_node_tags
        ):
            if etype == 2:  # 3-node triangle
                # node_tags_per_elem is flat list of node tags for all triangles
                tri_nodes = np.array(node_tags_per_elem).reshape(-1, 3)
                # map gmsh node tags to indices
                tri_idx = np.vectorize(lambda t: tag_to_idx[t])(tri_nodes)
                triangles.append(tri_idx)
        if not triangles:
            raise ValueError("No triangular elements found in mesh.")
        cells = np.vstack(triangles)

        # Build mapping from edge key (sorted node indices) to physical group tag
        # First, collect all line elements (type 1) that belong to physical groups of dimension 1.
        edge_to_phys = {}  # (node_a, node_b) -> physical_tag
        # Get physical groups of dimension 1 (curves)
        phys_groups = gmsh.model.getPhysicalGroups(dim=1)
        for dim, tag in phys_groups:
            # Get entities (curves) associated with this physical group
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            for entity_tag in entities:
                # Get mesh elements (lines) on this entity
                elem_types2, elem_tags2, elem_node_tags2 = gmsh.model.mesh.getElements(
                    dim=1, tag=entity_tag
                )
                for etype2, tags2, node_tags_per_elem2 in zip(
                    elem_types2, elem_tags2, elem_node_tags2
                ):
                    if etype2 == 1:  # line
                        edges = np.array(node_tags_per_elem2).reshape(-1, 2)
                        for edge in edges:
                            a, b = edge
                            if a > b:
                                a, b = b, a
                            edge_to_phys[(a, b)] = tag

        # If no physical groups found, default all boundary edges to tag 0
        if not edge_to_phys:
            # Get all line elements (including internal edges?)
            for etype, tags, node_tags_per_elem in zip(
                elem_types, elem_tags, elem_node_tags
            ):
                if etype == 1:
                    edges = np.array(node_tags_per_elem).reshape(-1, 2)
                    for edge in edges:
                        a, b = edge
                        if a > b:
                            a, b = b, a
                        edge_to_phys[(a, b)] = 0  # default tag

        gmsh.finalize()

        # Create mesh with default tags (all boundary faces tagged "boundary")
        mesh = cls(nodes, cells, boundary_tags=None)
        # Assign boundary tags based on physical groups
        mesh.assign_boundary_tags_from_gmsh_edges(
            edge_to_phys, tag_to_idx, physical_group_map
        )
        return mesh

    @classmethod
    def generate_rectangle_with_boundary_refinement(
        cls,
        Lx=1.0,
        Ly=1.0,
        background_size=0.1,
        boundary_refinement_distance=0.1,
        boundary_refinement_size=0.02,
        physical_group_map=None,
        refinement_zones=None,
    ):
        """Generate a rectangular triangular mesh with boundary refinement using Gmsh.

        Parameters
        ----------
        Lx, Ly : float
            Domain dimensions.
        background_size : float
            Mesh size far from boundaries.
        boundary_refinement_distance : float
            Distance from boundaries where refinement is applied.
        boundary_refinement_size : float
            Mesh size near boundaries.
        physical_group_map : dict, optional
            Mapping from Gmsh physical group numbers to tag names.
            Default: {1: "left", 2: "right", 3: "bottom", 4: "top"}.
        refinement_zones : list of dict, optional
            List of refinement zone specifications. Each dict must contain:
            'type' ('box' or 'circle'), 'parameters' (shape parameters),
            'refinement_size' (target mesh size within zone), and optionally
            'distance_max' (transition distance). Default: None.

        Returns
        -------
        UnstructuredMesh2D
            Triangular mesh with boundary tags.

        Raises
        ------
        ImportError
            If gmsh library is not installed.
        """
        try:
            import gmsh
        except ImportError:
            raise ImportError(
                "Gmsh Python API not installed. Please install with: uv pip install gmsh"
            )
        import numpy as np

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        # Create rectangle geometry
        gmsh.model.add("rectangle")
        # Points
        p1 = gmsh.model.geo.addPoint(0, 0, 0)
        p2 = gmsh.model.geo.addPoint(Lx, 0, 0)
        p3 = gmsh.model.geo.addPoint(Lx, Ly, 0)
        p4 = gmsh.model.geo.addPoint(0, Ly, 0)
        # Lines
        l1 = gmsh.model.geo.addLine(p1, p2)  # bottom
        l2 = gmsh.model.geo.addLine(p2, p3)  # right
        l3 = gmsh.model.geo.addLine(p3, p4)  # top
        l4 = gmsh.model.geo.addLine(p4, p1)  # left
        # Curve loop and surface
        loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surf = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()

        # Physical groups for boundaries
        # Assign tags: 1 left, 2 right, 3 bottom, 4 top
        gmsh.model.addPhysicalGroup(1, [l4], tag=1)
        gmsh.model.addPhysicalGroup(1, [l2], tag=2)
        gmsh.model.addPhysicalGroup(1, [l1], tag=3)
        gmsh.model.addPhysicalGroup(1, [l3], tag=4)
        # Physical group for surface (required for 2D mesh)
        gmsh.model.addPhysicalGroup(2, [surf], tag=5)

        # Define size field for boundary refinement
        # Distance field from boundary curves
        dist_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_field, "CurvesList", [l1, l2, l3, l4])
        # Threshold field that varies size based on distance
        thresh_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh_field, "IField", dist_field)
        gmsh.model.mesh.field.setNumber(thresh_field, "LcMin", boundary_refinement_size)
        gmsh.model.mesh.field.setNumber(thresh_field, "LcMax", background_size)
        gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(
            thresh_field, "DistMax", boundary_refinement_distance
        )
        field_ids = [thresh_field]

        # Add refinement zones if specified
        if refinement_zones:
            for zone in refinement_zones:
                ztype = zone.get("type")
                params = zone.get("parameters", {})
                size = zone.get("refinement_size")
                dist_max = zone.get("distance_max", 0.0)
                if ztype == "box":
                    # Box field: constant size inside box, transition outside
                    box_field = gmsh.model.mesh.field.add("Box")
                    gmsh.model.mesh.field.setNumber(
                        box_field, "XMin", params.get("xmin", 0.0)
                    )
                    gmsh.model.mesh.field.setNumber(
                        box_field, "XMax", params.get("xmax", Lx)
                    )
                    gmsh.model.mesh.field.setNumber(
                        box_field, "YMin", params.get("ymin", 0.0)
                    )
                    gmsh.model.mesh.field.setNumber(
                        box_field, "YMax", params.get("ymax", Ly)
                    )
                    gmsh.model.mesh.field.setNumber(box_field, "ZMin", -1.0)
                    gmsh.model.mesh.field.setNumber(box_field, "ZMax", 1.0)
                    gmsh.model.mesh.field.setNumber(box_field, "VIn", size)
                    gmsh.model.mesh.field.setNumber(box_field, "VOut", background_size)
                    gmsh.model.mesh.field.setNumber(box_field, "Thickness", dist_max)
                    field_ids.append(box_field)
                elif ztype == "circle":
                    # Distance field from center point
                    cx, cy = params.get("center", (Lx / 2, Ly / 2))
                    radius = params.get("radius", 0.1)
                    # Create a point in Gmsh geometry (tag will be negative)
                    pt = gmsh.model.geo.addPoint(cx, cy, 0)
                    gmsh.model.geo.synchronize()
                    dist_field_pt = gmsh.model.mesh.field.add("Distance")
                    gmsh.model.mesh.field.setNumbers(dist_field_pt, "PointsList", [pt])
                    thresh_field_pt = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(
                        thresh_field_pt, "IField", dist_field_pt
                    )
                    gmsh.model.mesh.field.setNumber(thresh_field_pt, "LcMin", size)
                    gmsh.model.mesh.field.setNumber(
                        thresh_field_pt, "LcMax", background_size
                    )
                    gmsh.model.mesh.field.setNumber(thresh_field_pt, "DistMin", 0.0)
                    gmsh.model.mesh.field.setNumber(
                        thresh_field_pt, "DistMax", max(dist_max, radius)
                    )
                    field_ids.append(thresh_field_pt)
                else:
                    raise ValueError(f"Unknown refinement zone type: {ztype}")

        # Combine all fields with Min
        if len(field_ids) == 1:
            gmsh.model.mesh.field.setAsBackgroundMesh(field_ids[0])
        else:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_ids)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        # Generate mesh
        gmsh.model.mesh.generate(2)

        # Extract mesh data (same as from_gmsh)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        nodes = coords[:, :2]
        tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()
        triangles = []
        for etype, tags, node_tags_per_elem in zip(
            elem_types, elem_tags, elem_node_tags
        ):
            if etype == 2:  # 3-node triangle
                tri_nodes = np.array(node_tags_per_elem).reshape(-1, 3)
                tri_idx = np.vectorize(lambda t: tag_to_idx[t])(tri_nodes)
                triangles.append(tri_idx)
        if not triangles:
            raise ValueError("No triangular elements found in mesh.")
        cells = np.vstack(triangles)

        # Build mapping from edge key to physical group tag
        edge_to_phys = {}
        phys_groups = gmsh.model.getPhysicalGroups(dim=1)
        for dim, tag in phys_groups:
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            for entity_tag in entities:
                elem_types2, elem_tags2, elem_node_tags2 = gmsh.model.mesh.getElements(
                    dim=1, tag=entity_tag
                )
                for etype2, tags2, node_tags_per_elem2 in zip(
                    elem_types2, elem_tags2, elem_node_tags2
                ):
                    if etype2 == 1:  # line
                        edges = np.array(node_tags_per_elem2).reshape(-1, 2)
                        for edge in edges:
                            a, b = edge
                            if a > b:
                                a, b = b, a
                            edge_to_phys[(a, b)] = tag

        gmsh.finalize()

        # Create mesh with default tags
        mesh = cls(nodes, cells, boundary_tags=None)
        # Assign boundary tags based on physical groups
        if physical_group_map is None:
            physical_group_map = {1: "left", 2: "right", 3: "bottom", 4: "top"}
        mesh.assign_boundary_tags_from_gmsh_edges(
            edge_to_phys, tag_to_idx, physical_group_map
        )
        return mesh

    @classmethod
    def from_meshpy(cls, points, facets, holes=None, max_area=None, min_angle=None):
        """Create a triangular mesh using meshpy (Triangle wrapper).

        Parameters
        ----------
        points : ndarray, shape (N_points, 2)
            Point coordinates.
        facets : ndarray, shape (N_facets, 2)
            Segment indices (0‑based) forming the domain boundary.
        holes : ndarray, shape (N_holes, 2), optional
            Interior hole coordinates.
        max_area : float, optional
            Maximum triangle area constraint.
        min_angle : float, optional
            Minimum angle constraint (degrees).

        Returns
        -------
        UnstructuredMesh2D
            Triangular mesh. Boundary tags are derived from facet markers
            if present; otherwise all boundary edges are tagged "boundary".

        Raises
        ------
        ImportError
            If meshpy library is not installed.
        """
        try:
            import meshpy.triangle as triangle
        except ImportError:
            raise ImportError(
                "meshpy not installed. Please install with: uv pip install meshpy"
            )
        import numpy as np

        # Build Triangle info
        info = triangle.MeshInfo()
        info.set_points(points)
        info.set_facets(facets)
        if holes is not None:
            info.set_holes(holes)

        # Mesh generation
        mesh = triangle.build(
            info,
            max_volume=max_area,
            min_angle=min_angle,
            attributes=False,
            volume_constraints=True if max_area else False,
        )

        nodes = np.array(mesh.points)
        cells = np.array(mesh.elements)
        # TODO: map facet markers to boundary tags
        boundary_tags = None
        return cls(nodes, cells, boundary_tags)

    @classmethod
    def from_dmsh(cls, geo, max_edge_length=None):
        """Create a triangular mesh using dmsh.

        Parameters
        ----------
        geo : dmsh geometry object
            Geometry from dmsh (e.g., dmsh.Rectangle, dmsh.Circle, etc.)
        max_edge_length : float, optional
            Maximum edge length constraint.

        Returns
        -------
        UnstructuredMesh2D
            Triangular mesh. All boundary edges are tagged "boundary".

        Raises
        ------
        ImportError
            If dmsh library is not installed.
        """
        try:
            import dmsh
        except ImportError:
            raise ImportError(
                "dmsh not installed. Please install with: uv pip install dmsh"
            )
        import numpy as np

        nodes, cells = dmsh.generate(geo, max_edge_length)
        boundary_tags = None
        return cls(nodes, cells, boundary_tags)

    def __repr__(self):
        return f"Unstructured triangular mesh with {self.num_cells} cells, {self.num_faces} faces"


class UnstructuredMesh3D(MeshStructure):
    """Unstructured tetrahedral mesh in 3-D.

    Instantiation Options
    ---------------------
    - ``UnstructuredMesh3D(nodes, cells, boundary_tags=None)``
    - ``UnstructuredMesh3D.from_delaunay(Nx, Ny, Nz, Lx, Ly, Lz)``
    - ``UnstructuredMesh3D.from_gmsh(geo_file=None, geo_string=None, physical_group_map=None)``
    - ``UnstructuredMesh3D.generate_box_with_boundary_refinement(Lx=1.0, Ly=1.0, Lz=1.0, background_size=0.2, boundary_refinement_distance=0.1, boundary_refinement_size=0.05, physical_group_map=None, refinement_zones=None)``
    """

    def __init__(self, nodes: np.ndarray, cells: np.ndarray, boundary_tags=None):
        """
        Parameters
        ----------
        nodes : ndarray, shape (N_nodes, 3)
            Vertex coordinates.
        cells : ndarray, shape (N_cells, 4)
            Tetrahedron vertex indices (0-based).
        boundary_tags : dict, optional
            Mapping from tag string to list of boundary face indices.
            If not provided, all boundary faces are tagged as "boundary".
        """
        data = _build_unstructured_connectivity_3d(nodes, cells, boundary_tags)
        MeshStructure.__init__(self, **data)
        # Store nodes and cells for visualization
        self._nodes = nodes.copy()
        self._cells = cells.copy()

    def assign_boundary_tags_from_gmsh_faces(
        self, face_to_phys, tag_to_idx, physical_group_map=None
    ):
        """Assign boundary tags based on Gmsh physical groups on surfaces.

        Parameters
        ----------
        face_to_phys : dict
            Mapping from face key (node_tag_a, node_tag_b, node_tag_c) to physical group tag.
        tag_to_idx : dict
            Mapping from Gmsh node tag to index in self._nodes.
        physical_group_map : dict, optional
            Mapping from physical group tag to user-defined tag name.
            If not provided, physical group tag numbers are used as strings.
        """
        import numpy as np

        # Map physical group tag to tag name
        tag_map = {}
        for phys_tag in set(face_to_phys.values()):
            if physical_group_map and phys_tag in physical_group_map:
                tag_name = physical_group_map[phys_tag]
            else:
                tag_name = str(phys_tag)
            tag_map[phys_tag] = tag_name

        # Convert face keys from Gmsh node tags to node indices
        face_to_phys_idx = {}
        for (tag_a, tag_b, tag_c), phys_tag in face_to_phys.items():
            idx_a = tag_to_idx[tag_a]
            idx_b = tag_to_idx[tag_b]
            idx_c = tag_to_idx[tag_c]
            key = tuple(sorted((idx_a, idx_b, idx_c)))
            face_to_phys_idx[key] = phys_tag

        # Now iterate over boundary faces
        new_tags = {}
        for f_idx in self.boundary_faces:
            a, b, c = self._face_nodes[f_idx]
            key = tuple(sorted((a, b, c)))
            if key in face_to_phys_idx:
                phys_tag = face_to_phys_idx[key]
                tag_name = tag_map[phys_tag]
                new_tags.setdefault(tag_name, []).append(f_idx)
            else:
                # Face not found? Should not happen for boundary faces.
                # Assign default tag "boundary"
                new_tags.setdefault("boundary", []).append(f_idx)

        # Convert lists to numpy arrays
        for tag_name, faces in new_tags.items():
            new_tags[tag_name] = np.array(faces, dtype=int)

        # Update mesh attributes
        self.boundary_tags = new_tags
        self.boundary_normal_sign = {tag: 1.0 for tag in new_tags}

    @classmethod
    def from_delaunay(cls, Nx: int, Ny: int, Nz: int, Lx: float, Ly: float, Lz: float):
        """Create a tetrahedral mesh of a rectangular box using Delaunay.

        Generates a regular point set of Nx*Ny*Nz points inside
        [0, Lx] x [0, Ly] x [0, Lz], then tetrahedralizes with
        scipy.spatial.Delaunay (which works for any dimension).

        Parameters
        ----------
        Nx, Ny, Nz : int
            Number of points in x, y, z directions (total points = Nx*Ny*Nz).
        Lx, Ly, Lz : float
            Domain dimensions.

        Returns
        -------
        UnstructuredMesh3D
            Tetrahedral mesh covering the box.
        """
        import numpy as np
        from scipy.spatial import Delaunay

        # Regular grid of points
        x = np.linspace(0, Lx, Nx)
        y = np.linspace(0, Ly, Ny)
        z = np.linspace(0, Lz, Nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        # Delaunay tetrahedralization (3D)
        tri = Delaunay(points)
        nodes = points
        cells = tri.simplices  # (N_cells, 4)

        boundary_tags = None  # placeholder
        return cls(nodes, cells, boundary_tags)

    @classmethod
    def from_gmsh(cls, geo_file=None, geo_string=None, physical_group_map=None):
        """Create a tetrahedral mesh from a Gmsh .geo file or geometry string.

        Parameters
        ----------
        geo_file : str, optional
            Path to a .geo file.
        geo_string : str, optional
            Geometry definition string (Gmsh .geo format).
        physical_group_map : dict, optional
            Mapping from Gmsh physical group numbers to tag names.
            If not provided, physical group numbers are used as tags.

        Returns
        -------
        UnstructuredMesh3D
            Tetrahedral mesh with boundary tags derived from Gmsh physical groups.

        Raises
        ------
        ImportError
            If gmsh library is not installed.
        ValueError
            If neither geo_file nor geo_string is provided.
        """
        try:
            import gmsh
        except ImportError:
            raise ImportError(
                "Gmsh Python API not installed. Please install with: uv pip install gmsh"
            )
        import numpy as np

        if geo_file is None and geo_string is None:
            raise ValueError("Either geo_file or geo_string must be provided.")

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # suppress console output

        if geo_file:
            gmsh.open(geo_file)
        else:
            gmsh.model.add("geometry")
            gmsh.model.geo.add(geo_string)
            gmsh.model.geo.synchronize()

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)

        # Get nodes
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        # reshape coords: (x1, y1, z1, x2, y2, z2, ...) -> (N, 3)
        coords = np.array(node_coords).reshape(-1, 3)
        nodes = coords  # keep x,y,z
        # map gmsh node tag to index
        tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

        # Get tetrahedral elements (type 4)
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()
        tets = []
        for etype, tags, node_tags_per_elem in zip(
            elem_types, elem_tags, elem_node_tags
        ):
            if etype == 4:  # 4-node tetrahedron
                # node_tags_per_elem is flat list of node tags for all tetrahedra
                tet_nodes = np.array(node_tags_per_elem).reshape(-1, 4)
                # map gmsh node tags to indices
                tet_idx = np.vectorize(lambda t: tag_to_idx[t])(tet_nodes)
                tets.append(tet_idx)
        if not tets:
            raise ValueError("No tetrahedral elements found in mesh.")
        cells = np.vstack(tets)

        # Build mapping from face key (sorted node indices) to physical group tag
        # Collect all triangular elements (type 2) that belong to physical groups of dimension 2.
        face_to_phys = {}  # (node_a, node_b, node_c) -> physical_tag
        # Get physical groups of dimension 2 (surfaces)
        phys_groups = gmsh.model.getPhysicalGroups(dim=2)
        for dim, tag in phys_groups:
            # Get entities (surfaces) associated with this physical group
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            for entity_tag in entities:
                # Get mesh elements (triangles) on this entity
                elem_types2, elem_tags2, elem_node_tags2 = gmsh.model.mesh.getElements(
                    dim=2, tag=entity_tag
                )
                for etype2, tags2, node_tags_per_elem2 in zip(
                    elem_types2, elem_tags2, elem_node_tags2
                ):
                    if etype2 == 2:  # triangle
                        faces = np.array(node_tags_per_elem2).reshape(-1, 3)
                        for face in faces:
                            a, b, c = face
                            key = tuple(sorted((a, b, c)))
                            face_to_phys[key] = tag

        # If no physical groups found, default all boundary faces to tag 0
        if not face_to_phys:
            # Get all triangular elements (including internal faces?)
            for etype, tags, node_tags_per_elem in zip(
                elem_types, elem_tags, elem_node_tags
            ):
                if etype == 2:
                    faces = np.array(node_tags_per_elem).reshape(-1, 3)
                    for face in faces:
                        a, b, c = face
                        key = tuple(sorted((a, b, c)))
                        face_to_phys[key] = 0  # default tag

        gmsh.finalize()

        # Create mesh with default tags (all boundary faces tagged "boundary")
        mesh = cls(nodes, cells, boundary_tags=None)
        # Assign boundary tags based on physical groups
        mesh.assign_boundary_tags_from_gmsh_faces(
            face_to_phys, tag_to_idx, physical_group_map
        )
        return mesh

    @classmethod
    def generate_box_with_boundary_refinement(
        cls,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        background_size=0.2,
        boundary_refinement_distance=0.1,
        boundary_refinement_size=0.05,
        physical_group_map=None,
        refinement_zones=None,
    ):
        """Generate a tetrahedral mesh of a box with boundary refinement using Gmsh.

        Parameters
        ----------
        Lx, Ly, Lz : float
            Domain dimensions.
        background_size : float
            Mesh size far from boundaries.
        boundary_refinement_distance : float
            Distance from boundaries where refinement is applied.
        boundary_refinement_size : float
            Mesh size near boundaries.
        physical_group_map : dict, optional
            Mapping from Gmsh physical group numbers to tag names.
            Default: {1: "left", 2: "right", 3: "bottom", 4: "top", 5: "front", 6: "back"}.
        refinement_zones : list of dict, optional
            List of refinement zone specifications. Each dict must contain:
            'type' ('box', 'sphere', or 'cylinder'), 'parameters' (shape parameters),
            'refinement_size' (target mesh size within zone), and optionally
            'distance_max' (transition distance). Default: None.

        Returns
        -------
        UnstructuredMesh3D
            Tetrahedral mesh with boundary tags.

        Raises
        ------
        ImportError
            If gmsh library is not installed.
        """
        try:
            import gmsh
        except ImportError:
            raise ImportError(
                "Gmsh Python API not installed. Please install with: uv pip install gmsh"
            )
        import numpy as np

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        # Create box geometry using elementary operations
        gmsh.model.add("box")
        # Points
        p1 = gmsh.model.geo.addPoint(0, 0, 0)
        p2 = gmsh.model.geo.addPoint(Lx, 0, 0)
        p3 = gmsh.model.geo.addPoint(Lx, Ly, 0)
        p4 = gmsh.model.geo.addPoint(0, Ly, 0)
        p5 = gmsh.model.geo.addPoint(0, 0, Lz)
        p6 = gmsh.model.geo.addPoint(Lx, 0, Lz)
        p7 = gmsh.model.geo.addPoint(Lx, Ly, Lz)
        p8 = gmsh.model.geo.addPoint(0, Ly, Lz)
        # Edges
        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)
        l5 = gmsh.model.geo.addLine(p5, p6)
        l6 = gmsh.model.geo.addLine(p6, p7)
        l7 = gmsh.model.geo.addLine(p7, p8)
        l8 = gmsh.model.geo.addLine(p8, p5)
        l9 = gmsh.model.geo.addLine(p1, p5)
        l10 = gmsh.model.geo.addLine(p2, p6)
        l11 = gmsh.model.geo.addLine(p3, p7)
        l12 = gmsh.model.geo.addLine(p4, p8)
        # Surfaces
        # bottom (z=0)
        loop1 = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        bottom = gmsh.model.geo.addPlaneSurface([loop1])
        # top (z=Lz)
        loop2 = gmsh.model.geo.addCurveLoop([l5, l6, l7, l8])
        top = gmsh.model.geo.addPlaneSurface([loop2])
        # left (x=0)
        loop3 = gmsh.model.geo.addCurveLoop([l4, l12, -l8, -l9])
        left = gmsh.model.geo.addPlaneSurface([loop3])
        # right (x=Lx)
        loop4 = gmsh.model.geo.addCurveLoop([l2, l11, -l6, -l10])
        right = gmsh.model.geo.addPlaneSurface([loop4])
        # front (y=0)
        loop5 = gmsh.model.geo.addCurveLoop([l1, l10, -l5, -l9])
        front = gmsh.model.geo.addPlaneSurface([loop5])
        # back (y=Ly)
        loop6 = gmsh.model.geo.addCurveLoop([l3, l12, -l7, -l11])
        back = gmsh.model.geo.addPlaneSurface([loop6])
        # Volume
        surf_loop = gmsh.model.geo.addSurfaceLoop(
            [bottom, top, left, right, front, back]
        )
        volume = gmsh.model.geo.addVolume([surf_loop])
        gmsh.model.geo.synchronize()

        # Physical groups for boundaries
        # Assign tags: 1 left, 2 right, 3 bottom, 4 top, 5 front, 6 back
        gmsh.model.addPhysicalGroup(2, [left], tag=1)
        gmsh.model.addPhysicalGroup(2, [right], tag=2)
        gmsh.model.addPhysicalGroup(2, [bottom], tag=3)
        gmsh.model.addPhysicalGroup(2, [top], tag=4)
        gmsh.model.addPhysicalGroup(2, [front], tag=5)
        gmsh.model.addPhysicalGroup(2, [back], tag=6)
        # Physical group for volume
        gmsh.model.addPhysicalGroup(3, [volume], tag=7)

        # Define size field for boundary refinement
        # Distance field from boundary surfaces
        dist_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            dist_field, "SurfacesList", [left, right, bottom, top, front, back]
        )
        # Threshold field that varies size based on distance
        thresh_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh_field, "IField", dist_field)
        gmsh.model.mesh.field.setNumber(thresh_field, "LcMin", boundary_refinement_size)
        gmsh.model.mesh.field.setNumber(thresh_field, "LcMax", background_size)
        gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(
            thresh_field, "DistMax", boundary_refinement_distance
        )
        field_ids = [thresh_field]

        # Add refinement zones if specified
        if refinement_zones:
            for zone in refinement_zones:
                ztype = zone.get("type")
                params = zone.get("parameters", {})
                size = zone.get("refinement_size")
                dist_max = zone.get("distance_max", 0.0)
                if ztype == "box":
                    # Box field: constant size inside box, transition outside
                    box_field = gmsh.model.mesh.field.add("Box")
                    gmsh.model.mesh.field.setNumber(
                        box_field, "XMin", params.get("xmin", 0.0)
                    )
                    gmsh.model.mesh.field.setNumber(
                        box_field, "XMax", params.get("xmax", Lx)
                    )
                    gmsh.model.mesh.field.setNumber(
                        box_field, "YMin", params.get("ymin", 0.0)
                    )
                    gmsh.model.mesh.field.setNumber(
                        box_field, "YMax", params.get("ymax", Ly)
                    )
                    gmsh.model.mesh.field.setNumber(
                        box_field, "ZMin", params.get("zmin", 0.0)
                    )
                    gmsh.model.mesh.field.setNumber(
                        box_field, "ZMax", params.get("zmax", Lz)
                    )
                    gmsh.model.mesh.field.setNumber(box_field, "VIn", size)
                    gmsh.model.mesh.field.setNumber(box_field, "VOut", background_size)
                    gmsh.model.mesh.field.setNumber(box_field, "Thickness", dist_max)
                    field_ids.append(box_field)
                elif ztype == "sphere":
                    # Sphere field: constant size inside sphere
                    cx, cy, cz = params.get("center", (Lx / 2, Ly / 2, Lz / 2))
                    radius = params.get("radius", 0.1)
                    sphere_field = gmsh.model.mesh.field.add("Sphere")
                    gmsh.model.mesh.field.setNumber(sphere_field, "XCenter", cx)
                    gmsh.model.mesh.field.setNumber(sphere_field, "YCenter", cy)
                    gmsh.model.mesh.field.setNumber(sphere_field, "ZCenter", cz)
                    gmsh.model.mesh.field.setNumber(sphere_field, "Radius", radius)
                    gmsh.model.mesh.field.setNumber(sphere_field, "VIn", size)
                    gmsh.model.mesh.field.setNumber(
                        sphere_field, "VOut", background_size
                    )
                    gmsh.model.mesh.field.setNumber(sphere_field, "Thickness", dist_max)
                    field_ids.append(sphere_field)
                elif ztype == "cylinder":
                    # Cylinder field: constant size inside cylinder
                    # Axis defined by two points (p1, p2)
                    p1 = params.get("p1", (0.0, 0.0, 0.0))
                    p2 = params.get("p2", (0.0, 0.0, Lz))
                    radius = params.get("radius", 0.1)
                    cyl_field = gmsh.model.mesh.field.add("Cylinder")
                    gmsh.model.mesh.field.setNumber(cyl_field, "XCenter", p1[0])
                    gmsh.model.mesh.field.setNumber(cyl_field, "YCenter", p1[1])
                    gmsh.model.mesh.field.setNumber(cyl_field, "ZCenter", p1[2])
                    gmsh.model.mesh.field.setNumber(cyl_field, "XAxis", p2[0] - p1[0])
                    gmsh.model.mesh.field.setNumber(cyl_field, "YAxis", p2[1] - p1[1])
                    gmsh.model.mesh.field.setNumber(cyl_field, "ZAxis", p2[2] - p1[2])
                    gmsh.model.mesh.field.setNumber(cyl_field, "Radius", radius)
                    gmsh.model.mesh.field.setNumber(cyl_field, "VIn", size)
                    gmsh.model.mesh.field.setNumber(cyl_field, "VOut", background_size)
                    gmsh.model.mesh.field.setNumber(cyl_field, "Thickness", dist_max)
                    field_ids.append(cyl_field)
                else:
                    raise ValueError(f"Unknown refinement zone type: {ztype}")

        # Combine all fields with Min
        if len(field_ids) == 1:
            gmsh.model.mesh.field.setAsBackgroundMesh(field_ids[0])
        else:
            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", field_ids)
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        # Generate 3D mesh
        gmsh.model.mesh.generate(3)

        # Extract mesh data (same as from_gmsh)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        nodes = coords  # keep x,y,z
        tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements()
        tets = []
        for etype, tags, node_tags_per_elem in zip(
            elem_types, elem_tags, elem_node_tags
        ):
            if etype == 4:  # 4-node tetrahedron
                tet_nodes = np.array(node_tags_per_elem).reshape(-1, 4)
                tet_idx = np.vectorize(lambda t: tag_to_idx[t])(tet_nodes)
                tets.append(tet_idx)
        if not tets:
            raise ValueError("No tetrahedral elements found in mesh.")
        cells = np.vstack(tets)

        # Build mapping from face key to physical group tag
        face_to_phys = {}
        phys_groups = gmsh.model.getPhysicalGroups(dim=2)
        for dim, tag in phys_groups:
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            for entity_tag in entities:
                elem_types2, elem_tags2, elem_node_tags2 = gmsh.model.mesh.getElements(
                    dim=2, tag=entity_tag
                )
                for etype2, tags2, node_tags_per_elem2 in zip(
                    elem_types2, elem_tags2, elem_node_tags2
                ):
                    if etype2 == 2:  # triangle
                        faces = np.array(node_tags_per_elem2).reshape(-1, 3)
                        for face in faces:
                            a, b, c = face
                            key = tuple(sorted((a, b, c)))
                            face_to_phys[key] = tag

        gmsh.finalize()

        # Create mesh with default tags
        mesh = cls(nodes, cells, boundary_tags=None)
        # Assign boundary tags based on physical groups
        if physical_group_map is None:
            physical_group_map = {
                1: "left",
                2: "right",
                3: "bottom",
                4: "top",
                5: "front",
                6: "back",
            }
        mesh.assign_boundary_tags_from_gmsh_faces(
            face_to_phys, tag_to_idx, physical_group_map
        )
        return mesh

    def __repr__(self):
        return f"Unstructured tetrahedral mesh with {self.num_cells} cells, {self.num_faces} faces"
