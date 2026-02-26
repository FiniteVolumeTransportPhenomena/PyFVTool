# FaceVariable class definition and operator overloading
#
# FaceVariable stores values on face centres as a single flat array of
# shape (num_faces,).  For structured meshes, convenience properties
# (.xvalue, .yvalue, .zvalue, .rvalue, .thetavalue, .phivalue) index
# into the flat array using face-group ranges stored on the mesh
# (_n_xfaces, _n_yfaces, _n_zfaces).

import numpy as np
from typing import overload

from .mesh import MeshStructure


# ------------------------------------------------------------------
#  Module-level helpers for face-group slicing (used by __init__ and
#  backward-compat properties).
# ------------------------------------------------------------------


def _x_slice_for(mesh):
    """Return a slice for x-faces (first coordinate direction)."""
    n = getattr(mesh, "_n_xfaces", mesh.num_faces)
    return slice(0, n)


def _y_slice_for(mesh):
    """Return a slice for y-faces (second coordinate direction)."""
    n_x = getattr(mesh, "_n_xfaces", None)
    n_y = getattr(mesh, "_n_yfaces", None)
    if n_x is None or n_y is None:
        raise AttributeError(
            "y-faces are only available for 2D and 3D structured meshes."
        )
    return slice(n_x, n_x + n_y)


def _z_slice_for(mesh):
    """Return a slice for z-faces (third coordinate direction)."""
    n_x = getattr(mesh, "_n_xfaces", None)
    n_y = getattr(mesh, "_n_yfaces", None)
    n_z = getattr(mesh, "_n_zfaces", None)
    if n_x is None or n_y is None or n_z is None:
        raise AttributeError("z-faces are only available for 3D structured meshes.")
    return slice(n_x + n_y, n_x + n_y + n_z)


class FaceVariable:
    """A scalar field defined on the face centres of a mesh.

    Parameters
    ----------
    mesh : MeshStructure
        The mesh on which this variable lives.
    face_value : float or ndarray
        Initial value(s).  A scalar broadcasts to all faces.  An
        ndarray of shape ``(num_faces,)`` sets all faces directly.

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> f = pf.FaceVariable(m, 1.0)
    """

    @overload
    def __init__(self, mesh: MeshStructure, face_value: float): ...
    @overload
    def __init__(self, mesh: MeshStructure, face_value: np.ndarray): ...

    def __init__(self, mesh: MeshStructure, face_value=0.0):
        self.domain = mesh
        nf = mesh.num_faces

        # Handle tuple/list input (legacy compat):
        # FaceVariable(mesh, (val,)) or FaceVariable(mesh, [val])
        # FaceVariable(mesh, [xval, yval]) for 2D — per-component broadcast
        # FaceVariable(mesh, [xval, yval, zval]) for 3D — per-component broadcast
        # FaceVariable(mesh, [xval, yval, zval]) for 1D/2D — legacy compat:
        #   extra trailing components are silently ignored (zeros in old API)
        if isinstance(face_value, (tuple, list)):
            if len(face_value) == 1:
                face_value = face_value[0]
            elif len(face_value) >= mesh.dimension and len(face_value) <= 3:
                # Per-component broadcast: each element fills the
                # corresponding face group (x-faces, y-faces, z-faces).
                # Extra elements beyond mesh.dimension are ignored (this
                # handles the old convention FaceVariable(m, [f_val, 0, 0])
                # on 1D meshes where only the first component matters).
                result = np.zeros(nf)
                slices = [_x_slice_for(mesh)]
                if mesh.dimension >= 2:
                    slices.append(_y_slice_for(mesh))
                if mesh.dimension >= 3:
                    slices.append(_z_slice_for(mesh))
                for s, v in zip(slices, face_value[: mesh.dimension]):
                    result[s] = (
                        float(v)
                        if np.isscalar(v)
                        else np.asarray(v, dtype=float).ravel()
                    )
                face_value = result
            else:
                face_value = np.concatenate(
                    [np.atleast_1d(np.asarray(v, dtype=float)) for v in face_value]
                )

        if np.isscalar(face_value):
            self._value = np.full(nf, float(face_value))
        elif isinstance(face_value, np.ndarray):
            if face_value.size == 1:
                self._value = np.full(nf, float(face_value.flat[0]))
            elif face_value.shape == (nf,):
                self._value = face_value.copy()
            else:
                # Try flattening
                flat = face_value.ravel()
                if flat.shape == (nf,):
                    self._value = flat.copy()
                else:
                    raise ValueError(
                        f"face_value shape {face_value.shape} is not valid for "
                        f"a mesh with {nf} faces.  Expected scalar or ({nf},)."
                    )
        else:
            arr = np.asarray(face_value, dtype=float)
            flat = arr.ravel()
            if flat.shape == (nf,):
                self._value = flat.copy()
            elif flat.size == 1:
                self._value = np.full(nf, float(flat[0]))
            else:
                raise ValueError(
                    f"face_value shape {arr.shape} is not valid for "
                    f"a mesh with {nf} faces.  Expected scalar or ({nf},)."
                )

    # ------------------------------------------------------------------
    #  Face-group index helpers (structured meshes only)
    # ------------------------------------------------------------------

    def _x_slice(self):
        """Return a slice for x-faces (first coordinate direction)."""
        n = getattr(self.domain, "_n_xfaces", self.domain.num_faces)
        return slice(0, n)

    def _y_slice(self):
        """Return a slice for y-faces (second coordinate direction)."""
        n_x = getattr(self.domain, "_n_xfaces", None)
        n_y = getattr(self.domain, "_n_yfaces", None)
        if n_x is None or n_y is None:
            raise AttributeError(
                "y-faces are only available for 2D and 3D structured meshes."
            )
        return slice(n_x, n_x + n_y)

    def _z_slice(self):
        """Return a slice for z-faces (third coordinate direction)."""
        n_x = getattr(self.domain, "_n_xfaces", None)
        n_y = getattr(self.domain, "_n_yfaces", None)
        n_z = getattr(self.domain, "_n_zfaces", None)
        if n_x is None or n_y is None or n_z is None:
            raise AttributeError("z-faces are only available for 3D structured meshes.")
        return slice(n_x + n_y, n_x + n_y + n_z)

    # ------------------------------------------------------------------
    #  Coordinate-name convenience properties (structured meshes)
    #
    #  These map coordinate labels to face groups:
    #    x/r      -> x-faces (first direction)
    #    y/z/theta -> y-faces (second direction)
    #    z/phi    -> z-faces (third direction)
    #
    #  The mapping is done via the mesh's coordlabels dict.
    # ------------------------------------------------------------------

    def _faces_for_label(self, label):
        """Return the slice for a named coordinate direction.

        Raises AttributeError if the label doesn't match any axis of the mesh.
        """
        labels = self.domain.coordlabels
        if label not in labels:
            raise AttributeError(
                f"'{label}' is not a coordinate of this mesh. "
                f"Available: {list(labels.keys())}"
            )
        dim_idx = labels[label]
        if dim_idx == 0:
            return self._x_slice()
        elif dim_idx == 1:
            return self._y_slice()
        elif dim_idx == 2:
            return self._z_slice()
        raise AttributeError(f"Unexpected dimension index {dim_idx}")

    def _shaped_face_group(self, dim_idx):
        """Return a reshaped view of a face group for structured meshes.

        For a 2D mesh with dims=[Nx, Ny]:
          dim_idx 0 (x-faces): shape (Nx+1, Ny)
          dim_idx 1 (y-faces): shape (Nx, Ny+1)

        For a 3D mesh with dims=[Nx, Ny, Nz]:
          dim_idx 0 (x-faces): shape (Nx+1, Ny, Nz)
          dim_idx 1 (y-faces): shape (Nx, Ny+1, Nz)
          dim_idx 2 (z-faces): shape (Nx, Ny, Nz+1)

        For 1D meshes or unstructured meshes, returns the flat slice.
        """
        if dim_idx == 0:
            sl = self._x_slice()
        elif dim_idx == 1:
            sl = self._y_slice()
        elif dim_idx == 2:
            sl = self._z_slice()
        else:
            raise ValueError(f"Unexpected dim_idx {dim_idx}")

        flat = self._value[sl]
        dims = getattr(self.domain, "dims", None)
        if dims is None or len(dims) <= 1:
            return flat

        shape = list(int(d) for d in dims)
        shape[dim_idx] += 1
        try:
            return flat.reshape(tuple(shape))
        except ValueError:
            return flat

    def _shaped_face_group_for_label(self, label):
        """Return a reshaped view of the face group for a coordinate label."""
        labels = self.domain.coordlabels
        if label not in labels:
            raise AttributeError(
                f"'{label}' is not a coordinate of this mesh. "
                f"Available: {list(labels.keys())}"
            )
        return self._shaped_face_group(labels[label])

    # ---- Cartesian labels ----

    @property
    def xvalue(self):
        return self._shaped_face_group_for_label("x")

    @xvalue.setter
    def xvalue(self, val):
        self._shaped_face_group_for_label("x")[:] = val

    @property
    def yvalue(self):
        return self._shaped_face_group_for_label("y")

    @yvalue.setter
    def yvalue(self, val):
        self._shaped_face_group_for_label("y")[:] = val

    @property
    def zvalue(self):
        return self._shaped_face_group_for_label("z")

    @zvalue.setter
    def zvalue(self, val):
        self._shaped_face_group_for_label("z")[:] = val

    # ---- Backward-compat underscore aliases ----
    # These map to face group index (0, 1, 2) regardless of
    # coordinate labels.  So _xvalue is always face group 0 (first
    # direction), _yvalue is face group 1 (second direction), and
    # _zvalue is face group 2 (third direction).  This is important
    # for meshes like CylindricalGrid2D where the labels are (r, z)
    # but _yvalue should return the z-direction faces (group 1).

    @property
    def _xvalue(self):
        return self._shaped_face_group(0)

    @_xvalue.setter
    def _xvalue(self, val):
        self._shaped_face_group(0)[:] = val

    @property
    def _yvalue(self):
        return self._shaped_face_group(1)

    @_yvalue.setter
    def _yvalue(self, val):
        self._shaped_face_group(1)[:] = val

    @property
    def _zvalue(self):
        return self._shaped_face_group(2)

    @_zvalue.setter
    def _zvalue(self, val):
        self._shaped_face_group(2)[:] = val

    # ---- Cylindrical / Spherical / Polar labels ----

    @property
    def rvalue(self):
        return self._shaped_face_group_for_label("r")

    @rvalue.setter
    def rvalue(self, val):
        self._shaped_face_group_for_label("r")[:] = val

    @property
    def thetavalue(self):
        return self._shaped_face_group_for_label("theta")

    @thetavalue.setter
    def thetavalue(self, val):
        self._shaped_face_group_for_label("theta")[:] = val

    @property
    def phivalue(self):
        return self._shaped_face_group_for_label("phi")

    @phivalue.setter
    def phivalue(self, val):
        self._shaped_face_group_for_label("phi")[:] = val

    # ------------------------------------------------------------------
    #  Arithmetic operators
    # ------------------------------------------------------------------

    def _binop(self, other, op):
        if isinstance(other, FaceVariable):
            result = FaceVariable(self.domain, op(self._value, other._value))
        else:
            result = FaceVariable(self.domain, op(self._value, other))
        return result

    def _rbinop(self, other, op):
        if isinstance(other, FaceVariable):
            result = FaceVariable(self.domain, op(other._value, self._value))
        else:
            result = FaceVariable(self.domain, op(other, self._value))
        return result

    def __add__(self, other):
        return self._binop(other, np.add)

    def __radd__(self, other):
        return self._rbinop(other, np.add)

    def __sub__(self, other):
        return self._binop(other, np.subtract)

    def __rsub__(self, other):
        return self._rbinop(other, np.subtract)

    def __mul__(self, other):
        return self._binop(other, np.multiply)

    def __rmul__(self, other):
        return self._rbinop(other, np.multiply)

    def __truediv__(self, other):
        return self._binop(other, np.true_divide)

    def __rtruediv__(self, other):
        return self._rbinop(other, np.true_divide)

    def __pow__(self, other):
        return self._binop(other, np.power)

    def __rpow__(self, other):
        return self._rbinop(other, np.power)

    def __neg__(self):
        return FaceVariable(self.domain, -self._value)

    def __pos__(self):
        return FaceVariable(self.domain, self._value.copy())

    def __abs__(self):
        return FaceVariable(self.domain, np.abs(self._value))

    # ------------------------------------------------------------------
    #  Comparison / logical operators
    # ------------------------------------------------------------------

    def __gt__(self, other):
        return self._binop(other, np.greater)

    def __ge__(self, other):
        return self._binop(other, np.greater_equal)

    def __lt__(self, other):
        return self._binop(other, np.less)

    def __le__(self, other):
        return self._binop(other, np.less_equal)

    def __and__(self, other):
        return self._binop(other, np.logical_and)

    def __or__(self, other):
        return self._binop(other, np.logical_or)

    # ------------------------------------------------------------------
    #  Utility methods
    # ------------------------------------------------------------------

    def copy(self):
        """Return a copy of this FaceVariable."""
        return FaceVariable(self.domain, self._value.copy())

    def __repr__(self):
        return (
            f"FaceVariable(mesh={self.domain.__class__.__name__}, "
            f"num_faces={self.domain.num_faces})"
        )

    def __str__(self):
        return repr(self)


# ------------------------------------------------------------------
#  Module-level helper functions
# ------------------------------------------------------------------


def faceLocations(m: MeshStructure):
    """Return FaceVariable(s) containing face centre coordinates.

    Parameters
    ----------
    m : MeshStructure
        The mesh.

    Returns
    -------
    For 1-D meshes: ``X``
    For 2-D meshes: ``(X, Y)``
    For 3-D meshes: ``(X, Y, Z)``

    Each is a FaceVariable holding the corresponding coordinate
    values at face centres.  For structured meshes, the per-direction
    face groups are filled so that each face knows the *full*
    coordinate of its centre in that direction.

    Notes
    -----
    Unlike the old implementation, each returned FaceVariable holds a
    single coordinate component across *all* faces.  For structured 2D
    meshes ``X`` stores the first coordinate for every face (both
    x-faces and y-faces), and similarly ``Y`` stores the second coordinate.
    """
    dim = m.dimension
    fc = m._face_centers  # (num_faces, dim)

    if dim == 1:
        return FaceVariable(m, fc[:, 0])
    elif dim == 2:
        X = FaceVariable(m, fc[:, 0])
        Y = FaceVariable(m, fc[:, 1])
        return X, Y
    elif dim == 3:
        X = FaceVariable(m, fc[:, 0])
        Y = FaceVariable(m, fc[:, 1])
        Z = FaceVariable(m, fc[:, 2])
        return X, Y, Z
    raise TypeError(f"Unsupported mesh dimension: {dim}")


def faceeval(f, *args):
    """Apply a function to the values of one or more FaceVariables.

    Parameters
    ----------
    f : callable
        Function to apply element-wise.
    *args : FaceVariable
        Input FaceVariables.

    Returns
    -------
    FaceVariable
        Result of ``f(arg0._value, arg1._value, ...)``.

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> fv = pf.FaceVariable(m, 1.0)
    >>> g = pf.faceeval(lambda x: x**2, fv)
    """
    values = [a._value for a in args]
    result_val = f(*values)
    return FaceVariable(args[0].domain, result_val)
