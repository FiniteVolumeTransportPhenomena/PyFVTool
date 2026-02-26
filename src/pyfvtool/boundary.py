# Boundary condition classes — connectivity-based unified representation
#
# BoundaryConditions stores a dict mapping boundary tag strings to
# BoundaryFace objects.  For structured meshes, convenience properties
# (.left, .right, .bottom, .top, .back, .front) are syntactic sugar
# that index into this dict.
#
# The Robin BC convention is:
#   a * dphi/dn + b * phi = c
# where n is the outward unit normal on the boundary face.

import numpy as np
from scipy.sparse import csr_array

from .mesh import MeshStructure
from .utilities import TrackedArray


# -----------------------------------------------------------------------
#  BoundaryFace — stores a, b, c arrays for one group of boundary faces
# -----------------------------------------------------------------------


class BoundaryFace:
    """Boundary condition coefficients for a named group of faces.

    The Robin boundary condition is:

        ``a * dphi/dn + b * phi = c``

    Parameters
    ----------
    a, b, c : ndarray
        Coefficient arrays, each of length equal to the number of faces
        in this boundary group.
    periodic : bool
        True if this boundary group is periodic.
    """

    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, periodic=False):
        if (
            not isinstance(a, np.ndarray)
            or not isinstance(b, np.ndarray)
            or not isinstance(c, np.ndarray)
        ):
            raise TypeError("a, b, c must be np.ndarray")
        self._a = TrackedArray(a.ravel().astype(float))
        self._b = TrackedArray(b.ravel().astype(float))
        self._c = TrackedArray(c.ravel().astype(float))
        self._periodic = periodic

    def __str__(self):
        return f"BoundaryFace(n={len(self._a)}, periodic={self._periodic})"

    def __repr__(self):
        return str(self)

    @property
    def modified(self):
        return self._a.modified or self._b.modified or self._c.modified

    @modified.setter
    def modified(self, val):
        modval = bool(val)
        self._a.modified = modval
        self._b.modified = modval
        self._c.modified = modval

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, val):
        self._a[:] = val

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, val):
        self._b[:] = val

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, val):
        self._c[:] = val

    @property
    def periodic(self):
        return self._periodic

    @periodic.setter
    def periodic(self, val):
        self.modified = True
        self._periodic = bool(val)


# -----------------------------------------------------------------------
#  BoundaryConditions — tag-based dict of BoundaryFace objects
# -----------------------------------------------------------------------

# Structured-mesh face names that get convenience property access
_STRUCTURED_NAMES = ("left", "right", "bottom", "top", "back", "front")


class BoundaryConditions:
    """Boundary conditions for a mesh, stored as a dict of tag -> BoundaryFace.

    For structured meshes, the factory function auto-creates the standard
    tags ("left", "right", "bottom", "top", "back", "front") with Neumann
    (a=1, b=0, c=0) defaults.  These are accessible as properties::

        BC = BoundaryConditions(mesh)
        BC.left.a[:] = 0   # Dirichlet
        BC.left.b[:] = 1
        BC.left.c[:] = T_left

    For unstructured meshes, users assign BCs by tag::

        BC["inlet"].a[:] = 0
        BC["inlet"].b[:] = 1
        BC["inlet"].c[:] = T_inlet

    Parameters
    ----------
    mesh : MeshStructure
        The mesh on which these BCs are defined.
    """

    def __init__(self, mesh: MeshStructure):
        self.domain = mesh
        self._faces = {}  # tag -> BoundaryFace

        # Auto-create BoundaryFace for each tag in the mesh
        for tag, face_indices in mesh.boundary_tags.items():
            n = len(face_indices)
            # Default: Neumann (a=1, b=0, c=0) => dphi/dn = 0
            self._faces[tag] = BoundaryFace(np.ones(n), np.zeros(n), np.zeros(n))

    # ---- dict-like access -----------------------------------------------

    def __getitem__(self, tag: str) -> BoundaryFace:
        return self._faces[tag]

    def __setitem__(self, tag: str, face: BoundaryFace):
        if not isinstance(face, BoundaryFace):
            raise TypeError("Value must be a BoundaryFace instance")
        self._faces[tag] = face

    def __contains__(self, tag: str) -> bool:
        return tag in self._faces

    def tags(self):
        """Return list of boundary tag names."""
        return list(self._faces.keys())

    def items(self):
        """Return (tag, BoundaryFace) pairs."""
        return self._faces.items()

    # ---- convenience properties for structured meshes --------------------

    @property
    def left(self) -> BoundaryFace:
        return self._faces["left"]

    @left.setter
    def left(self, val: BoundaryFace):
        self._faces["left"] = val

    @property
    def right(self) -> BoundaryFace:
        return self._faces["right"]

    @right.setter
    def right(self, val: BoundaryFace):
        self._faces["right"] = val

    @property
    def bottom(self) -> BoundaryFace:
        return self._faces["bottom"]

    @bottom.setter
    def bottom(self, val: BoundaryFace):
        self._faces["bottom"] = val

    @property
    def top(self) -> BoundaryFace:
        return self._faces["top"]

    @top.setter
    def top(self, val: BoundaryFace):
        self._faces["top"] = val

    @property
    def back(self) -> BoundaryFace:
        return self._faces["back"]

    @back.setter
    def back(self, val: BoundaryFace):
        self._faces["back"] = val

    @property
    def front(self) -> BoundaryFace:
        return self._faces["front"]

    @front.setter
    def front(self, val: BoundaryFace):
        self._faces["front"] = val

    @property
    def modified(self):
        """True if any BoundaryFace has been modified since last reset."""
        return any(bf.modified for bf in self._faces.values())

    @modified.setter
    def modified(self, val):
        for bf in self._faces.values():
            bf.modified = val

    def __str__(self):
        tags_str = ", ".join(f"{tag}({len(bf.a)})" for tag, bf in self._faces.items())
        return f"BoundaryConditions({tags_str})"

    def __repr__(self):
        return str(self)


# -----------------------------------------------------------------------
#  cellValuesWithBoundaries — unified, connectivity-based
# -----------------------------------------------------------------------


def cellValuesWithBoundaries(phi_internal, BC):
    """Compute ghost-cell values from BCs and return the full value array.

    Given interior cell values and boundary conditions, compute the
    ghost-cell values using the Robin BC formula and return the combined
    array ``[interior_values, ghost_values]``.

    The Robin BC at a boundary face is:

        ``a * (phi_G - phi_P) / d_CF + b * (phi_G + phi_P) / 2 = c``

    Solving for phi_G:

        ``phi_G = (c - phi_P * (-a/d_CF + b/2)) / (a/d_CF + b/2)``

    Parameters
    ----------
    phi_internal : ndarray, shape (num_cells,)
        Interior cell values (or a flat array that is at least num_cells long).
    BC : BoundaryConditions
        Boundary conditions.

    Returns
    -------
    phi_all : ndarray, shape (num_cells + num_ghost_cells,)
        Combined interior + ghost cell values.
    """
    mesh = BC.domain
    N = mesh.num_cells
    Ng = mesh.num_ghost_cells

    # Ensure phi_internal is at least length N
    if hasattr(phi_internal, "_value"):
        # It's a CellVariable — extract the interior portion
        phi_int = phi_internal._value[:N]
    elif isinstance(phi_internal, np.ndarray):
        phi_int = phi_internal.ravel()[:N]
    else:
        phi_int = np.asarray(phi_internal).ravel()[:N]

    phi_all = np.zeros(N + Ng)
    phi_all[:N] = phi_int

    # For each boundary tag, compute ghost cell values
    for tag, bf in BC.items():
        face_indices = mesh.boundary_tags[tag]
        if len(face_indices) == 0:
            continue

        owner_cells = mesh.owner[face_indices]
        ghost_cells = mesh.neighbor[face_indices]  # >= N
        d = mesh.d_CF[face_indices]  # distance owner-center to ghost-center

        phi_P = phi_all[owner_cells]  # owner (interior) cell values

        if bf.periodic:
            # For periodic BCs, we need to find the paired boundary.
            # The periodic partner information is stored in
            # mesh._periodic_pairs if available; otherwise we handle
            # it tag-by-tag below.
            _apply_periodic_ghost(phi_all, mesh, tag, face_indices)
        else:
            # Robin: a*(phi_G - phi_P)/d + b*(phi_G + phi_P)/2 = c
            # phi_G*(a/d + b/2) = c - phi_P*(-a/d + b/2)
            # phi_G = (c - phi_P*(-a/d + b/2)) / (a/d + b/2)
            # Apply boundary normal sign to match MATLAB FVTool convention
            sign = mesh.boundary_normal_sign.get(tag, 1.0)
            a = bf.a * sign
            b = bf.b
            c = bf.c
            denom = a / d + b / 2.0
            numer = c - phi_P * (-a / d + b / 2.0)
            phi_all[ghost_cells] = numer / denom

    return phi_all


def _apply_periodic_ghost(phi_all, mesh, tag, face_indices):
    """Fill ghost cells for periodic boundary using the paired tag.

    For structured meshes, periodic pairs are:
      left <-> right, bottom <-> top, back <-> front
    The ghost cell on one side gets the value of the interior cell on
    the other side (mirrored).
    """
    pairs = {
        "left": "right",
        "right": "left",
        "bottom": "top",
        "top": "bottom",
        "back": "front",
        "front": "back",
    }
    partner_tag = pairs.get(tag)
    if partner_tag is None or partner_tag not in mesh.boundary_tags:
        raise ValueError(f"Cannot find periodic partner for boundary tag '{tag}'.")

    partner_faces = mesh.boundary_tags[partner_tag]
    # Ghost cell on this side = interior cell on partner side
    ghost_cells = mesh.neighbor[face_indices]
    partner_owners = mesh.owner[partner_faces]
    phi_all[ghost_cells] = phi_all[partner_owners]


# -----------------------------------------------------------------------
#  boundaryConditionsTerm — unified, connectivity-based
# -----------------------------------------------------------------------


def boundaryConditionsTerm(BC):
    """Build the sparse matrix and RHS vector for boundary conditions.

    For each boundary face, the ghost-cell row in the system matrix
    encodes the Robin BC:

        ``a * (phi_G - phi_P) / d_CF + b * (phi_G + phi_P) / 2 = c``

    which gives:

        ``phi_G * (a/d + b/2) + phi_P * (-a/d + b/2) = c``

    For periodic BCs the ghost cell equation is:

        ``phi_G - phi_partner_interior = 0``

    combined with a gradient-matching equation on the other side.

    Parameters
    ----------
    BC : BoundaryConditions
        Boundary conditions.

    Returns
    -------
    BCMatrix : csr_array, shape (N_total, N_total)
        Sparse matrix with BC equations in ghost-cell rows.
    BCRHS : ndarray, shape (N_total,)
        RHS vector with BC source terms in ghost-cell rows.
    """
    mesh = BC.domain
    N = mesh.num_cells
    Ng = mesh.num_ghost_cells
    N_total = N + Ng

    # Pre-count: each non-periodic boundary face generates 2 entries
    # (ghost-ghost diagonal and ghost-owner off-diagonal).
    # Periodic faces generate more entries (up to 4 per face).
    n_entries = 0
    for tag, bf in BC.items():
        face_indices = mesh.boundary_tags[tag]
        nf = len(face_indices)
        if nf == 0:
            continue
        # Both periodic and Robin need 2 entries per face
        n_entries += 2 * nf

    ii = np.empty(n_entries, dtype=int)
    jj = np.empty(n_entries, dtype=int)
    ss = np.empty(n_entries, dtype=float)
    BCRHS = np.zeros(N_total)

    q = 0  # write cursor

    for tag, bf in BC.items():
        face_indices = mesh.boundary_tags[tag]
        nf = len(face_indices)
        if nf == 0:
            continue

        owner_cells = mesh.owner[face_indices]
        ghost_cells = mesh.neighbor[face_indices]
        d = mesh.d_CF[face_indices]

        if bf.periodic:
            q = _periodic_bc_term(
                ii,
                jj,
                ss,
                BCRHS,
                q,
                mesh,
                tag,
                face_indices,
                owner_cells,
                ghost_cells,
                d,
            )
        else:
            # Robin BC in ghost-cell row:
            #   phi_G * (a/d + b/2) + phi_P * (-a/d + b/2) = c
            # We write this with a sign convention that keeps the
            # diagonal coefficient positive.
            # Apply boundary normal sign to match MATLAB FVTool convention
            sign = mesh.boundary_normal_sign.get(tag, 1.0)
            a = bf.a * sign
            b = bf.b
            c = bf.c
            diag_coeff = a / d + b / 2.0  # coefficient of phi_G
            off_coeff = -a / d + b / 2.0  # coefficient of phi_P

            # ghost row, ghost column (diagonal)
            ii[q : q + nf] = ghost_cells
            jj[q : q + nf] = ghost_cells
            ss[q : q + nf] = diag_coeff
            q += nf

            # ghost row, owner column (off-diagonal)
            ii[q : q + nf] = ghost_cells
            jj[q : q + nf] = owner_cells
            ss[q : q + nf] = off_coeff
            q += nf

            # RHS
            BCRHS[ghost_cells] = c

    BCMatrix = csr_array((ss[:q], (ii[:q], jj[:q])), shape=(N_total, N_total))
    return BCMatrix, BCRHS


def _periodic_bc_term(
    ii, jj, ss, BCRHS, q, mesh, tag, face_indices, owner_cells, ghost_cells, d
):
    """Write periodic BC entries into the COO arrays.

    For a periodic pair (e.g. left/right), the ghost cell on one side
    gets the value of the interior cell on the opposite side:

        ``phi_G = phi_partner_owner``   =>   ``phi_G - phi_partner_owner = 0``

    Both sides are processed independently, so the system is symmetric:
    the left ghost equals the right interior and vice versa.

    For structured meshes with matching face counts on both sides, faces
    are paired index-by-index.
    """
    pairs = {
        "left": "right",
        "right": "left",
        "bottom": "top",
        "top": "bottom",
        "back": "front",
        "front": "back",
    }
    partner_tag = pairs.get(tag)
    if partner_tag is None or partner_tag not in mesh.boundary_tags:
        raise ValueError(f"Cannot find periodic partner for boundary tag '{tag}'.")

    partner_face_indices = mesh.boundary_tags[partner_tag]
    partner_owners = mesh.owner[partner_face_indices]
    nf = len(face_indices)

    # Ghost row: phi_G * 1 - phi_partner_owner * 1 = 0
    ii[q : q + nf] = ghost_cells
    jj[q : q + nf] = ghost_cells
    ss[q : q + nf] = 1.0
    q += nf

    ii[q : q + nf] = ghost_cells
    jj[q : q + nf] = partner_owners
    ss[q : q + nf] = -1.0
    q += nf

    BCRHS[ghost_cells] = 0.0

    return q


# -----------------------------------------------------------------------
#  Convenience: apply_BCs — update ghost cells on a CellVariable
# -----------------------------------------------------------------------


def apply_BCs(phi, BC):
    """Update the ghost-cell values of a CellVariable from BCs.

    Parameters
    ----------
    phi : CellVariable
        The variable whose ghost cells will be updated in-place.
    BC : BoundaryConditions
        Boundary conditions.
    """
    mesh = BC.domain
    N = mesh.num_cells
    phi_int = phi._value[:N]

    for tag, bf in BC.items():
        face_indices = mesh.boundary_tags[tag]
        if len(face_indices) == 0:
            continue

        owner_cells = mesh.owner[face_indices]
        ghost_cells = mesh.neighbor[face_indices]
        d = mesh.d_CF[face_indices]

        if bf.periodic:
            _apply_periodic_ghost(phi._value, mesh, tag, face_indices)
        else:
            # Apply boundary normal sign to match MATLAB FVTool convention
            sign = mesh.boundary_normal_sign.get(tag, 1.0)
            a = bf.a * sign
            b = bf.b
            c = bf.c
            phi_P = phi._value[owner_cells]
            denom = a / d + b / 2.0
            numer = c - phi_P * (-a / d + b / 2.0)
            phi._value[ghost_cells] = numer / denom
