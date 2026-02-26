"""Gradient and divergence operators.

All functions use the connectivity-based representation (``owner``,
``neighbor``, ``d_CF``, ``face_areas``, ``cell_volumes``) so a single
code path handles all mesh types and dimensions.

Functions
---------
gradientTerm : face-normal gradient (cell -> face)
divergenceTerm : finite-volume divergence (face -> cell RHS vector)
"""

import numpy as np

from .mesh import MeshStructure
from .cell import CellVariable
from .face import FaceVariable


def gradientTerm(phi: CellVariable) -> FaceVariable:
    """Compute the face-normal gradient of a cell variable.

    For each face, the gradient is computed as:

    .. math::

        \\left.\\frac{\\partial \\phi}{\\partial n}\\right|_f
        = \\frac{\\phi_{\\text{neighbor}} - \\phi_{\\text{owner}}}{d_{CF}}

    Parameters
    ----------
    phi : CellVariable
        Cell-centred field.  Ghost-cell values must be current.

    Returns
    -------
    FaceVariable
        Face-normal gradient values at face centres.

    Notes
    -----
    This returns the gradient in *coordinate space*.  For curvilinear
    coordinate systems the metric factors are already absorbed into
    ``face_areas``, so the combination ``gradientTerm(phi) * face_areas``
    gives the correct physical flux.

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> phi = pf.CellVariable(m, 1.0)
    >>> gradPhi = pf.gradientTerm(phi)
    """
    m = phi.domain
    phi_O = phi._value[m.owner]
    phi_N = phi._value[m.neighbor]
    # Avoid division by zero for degenerate faces
    d_CF_safe = np.where(m.d_CF > 0, m.d_CF, 1.0)
    grad_f = (phi_N - phi_O) / d_CF_safe
    return FaceVariable(m, grad_f)


def divergenceTerm(F: FaceVariable) -> np.ndarray:
    """Compute the finite-volume divergence of a face variable.

    For each interior cell, the divergence is:

    .. math::

        (\\nabla \\cdot F)_P = \\frac{1}{V_P}
        \\sum_{f \\in \\text{faces}(P)} F_f \\, A_f \\, s_f

    where :math:`s_f = +1` when cell P is the owner (face normal
    points away from P) and :math:`s_f = -1` when P is the neighbor.

    Parameters
    ----------
    F : FaceVariable
        Face-centred flux density (flux per unit area).

    Returns
    -------
    ndarray, shape (num_cells + num_ghost_cells,)
        RHS divergence vector.  Ghost-cell entries are zero.

    Notes
    -----
    The result is suitable for direct addition into the RHS vector of
    a finite-volume linear system.

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> phi = pf.CellVariable(m, 1.0)
    >>> gradPhi = pf.gradientTerm(phi)
    >>> RHSdiv = pf.divergenceTerm(gradPhi)
    """
    m = F.domain
    N = m.num_cells
    Ng = m.num_ghost_cells
    total = N + Ng

    # Face flux: F_f * A_f
    flux = F._value * m.face_areas  # (num_faces,)

    # Accumulate into cells:
    #   owner sees +flux (normal points away from owner)
    #   neighbor sees -flux (normal points towards neighbor)
    RHS = np.zeros(total)
    np.add.at(RHS, m.owner, flux)
    np.add.at(RHS, m.neighbor, -flux)

    # Divide by cell volume for interior cells
    RHS[:N] /= m.cell_volumes

    # Zero out ghost cell entries (they were accumulated but are meaningless)
    RHS[N:] = 0.0

    return RHS


def gradientTermFixedBC(phi: CellVariable) -> FaceVariable:
    """Compute gradient with boundary correction for fixed ghost cells.

    This function computes the gradient and then doubles the boundary
    face gradients.  It is intended for use when ``phi`` is a derived
    quantity (e.g. ``f(phi)``) whose ghost-cell values were set by
    ``cellValuesWithBoundaries`` — in that case the ghost cell stores
    the face average, so the true gradient to the boundary is twice the
    computed gradient.

    .. warning::

        Unless you know you need this function, use :func:`gradientTerm`
        instead.

    Parameters
    ----------
    phi : CellVariable
        Cell-centred field with ghost-cell values set by
        ``cellValuesWithBoundaries``.

    Returns
    -------
    FaceVariable
        Corrected gradient at face centres.
    """
    faceGrad = gradientTerm(phi)
    m = phi.domain
    # Double the gradient on boundary faces
    faceGrad._value[m.boundary_faces] *= 2.0
    return faceGrad
