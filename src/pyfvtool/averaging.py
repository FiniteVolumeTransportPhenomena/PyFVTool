"""Cell-to-face averaging functions.

All functions take a :class:`CellVariable` (and optionally a
:class:`FaceVariable` for upwind/TVD) and return a :class:`FaceVariable`
with values interpolated to face centres.

Functions use the connectivity-based representation (``owner``,
``neighbor``, ``face_interpolation_weight``) so a single code path
handles all mesh types and dimensions.

Functions
---------
linearMean : linear (distance-weighted) interpolation
arithmeticMean : alias for linearMean
geometricMean : geometric (log-linear) interpolation
harmonicMean : harmonic interpolation
upwindMean : upwind (donor-cell) interpolation
tvdMean : TVD interpolation with flux limiter

Notes
-----
Ghost-cell values in ``phi._value`` **must** be up-to-date before
calling any averaging function.  Use :func:`boundary.apply_BCs` or
:func:`boundary.cellValuesWithBoundaries` first.
"""

import numpy as np

from .cell import CellVariable
from .face import FaceVariable


def linearMean(phi: CellVariable) -> FaceVariable:
    """Linearly interpolate cell values to face centres.

    Uses distance-based weighting:

        phi_f = w * phi[owner] + (1 - w) * phi[neighbor]

    where ``w = d_fF / d_CF`` (the face interpolation weight stored on
    the mesh).

    Parameters
    ----------
    phi : CellVariable
        Cell-centred field.  Ghost-cell values must be current.

    Returns
    -------
    FaceVariable
        Interpolated values at face centres.

    See Also
    --------
    arithmeticMean, geometricMean, harmonicMean
    """
    m = phi.domain
    w = m.face_interpolation_weight  # (num_faces,)
    phi_O = phi._value[m.owner]  # owner values
    phi_N = phi._value[m.neighbor]  # neighbor values
    phi_f = w * phi_O + (1.0 - w) * phi_N
    return FaceVariable(m, phi_f)


def arithmeticMean(phi: CellVariable) -> FaceVariable:
    """Interpolate cell values to faces via arithmetic (distance-weighted) mean.

    This is identical to :func:`linearMean` — the arithmetic mean
    weighted by cell sizes reduces to distance-weighted linear
    interpolation on a finite-volume mesh.

    Parameters
    ----------
    phi : CellVariable
        Cell-centred field.  Ghost-cell values must be current.

    Returns
    -------
    FaceVariable
        Interpolated values at face centres.

    See Also
    --------
    linearMean, geometricMean, harmonicMean
    """
    return linearMean(phi)


def geometricMean(phi: CellVariable) -> FaceVariable:
    """Interpolate cell values to faces via geometric mean.

    .. math::

        \\phi_f = \\exp\\bigl(w \\ln \\phi_O + (1-w) \\ln \\phi_N\\bigr)

    where :math:`w` is the face interpolation weight.

    Values equal to zero are handled specially: if either owner or
    neighbor value is zero the face value is set to zero.

    Parameters
    ----------
    phi : CellVariable
        Cell-centred field.  Ghost-cell values must be current.
        All values should be non-negative.

    Returns
    -------
    FaceVariable
        Geometric-mean interpolated values at face centres.

    See Also
    --------
    linearMean, harmonicMean
    """
    m = phi.domain
    w = m.face_interpolation_weight
    phi_O = phi._value[m.owner]
    phi_N = phi._value[m.neighbor]

    # Mask out zeros to avoid log(0)
    nonzero = (phi_O != 0.0) & (phi_N != 0.0)
    phi_f = np.zeros(m.num_faces)
    # Safe log only where both values are nonzero
    phi_f[nonzero] = np.exp(
        w[nonzero] * np.log(phi_O[nonzero])
        + (1.0 - w[nonzero]) * np.log(phi_N[nonzero])
    )
    return FaceVariable(m, phi_f)


def harmonicMean(phi: CellVariable) -> FaceVariable:
    """Interpolate cell values to faces via harmonic mean.

    .. math::

        \\phi_f = \\frac{1}{w / \\phi_O + (1-w) / \\phi_N}

    where :math:`w` is the face interpolation weight.

    Values equal to zero are handled specially: if either owner or
    neighbor value is zero the face value is set to zero.

    Parameters
    ----------
    phi : CellVariable
        Cell-centred field.  Ghost-cell values must be current.

    Returns
    -------
    FaceVariable
        Harmonic-mean interpolated values at face centres.

    See Also
    --------
    linearMean, geometricMean

    Notes
    -----
    The harmonic mean is sensitive to small values and is the
    preferred averaging for diffusion coefficients with discontinuous
    material properties.
    """
    m = phi.domain
    w = m.face_interpolation_weight
    phi_O = phi._value[m.owner]
    phi_N = phi._value[m.neighbor]

    # Mask out zeros to avoid division by zero
    nonzero = (phi_O != 0.0) & (phi_N != 0.0)
    phi_f = np.zeros(m.num_faces)
    phi_f[nonzero] = 1.0 / (
        w[nonzero] / phi_O[nonzero] + (1.0 - w[nonzero]) / phi_N[nonzero]
    )
    return FaceVariable(m, phi_f)


def upwindMean(phi: CellVariable, u: FaceVariable) -> FaceVariable:
    """Upwind (donor-cell) interpolation based on face flux direction.

    For each face, the value is taken from the cell that the flow is
    coming *from*:

    * If ``u[face] > 0``: flow goes owner -> neighbor, so ``phi_f = phi[owner]``
    * If ``u[face] < 0``: flow goes neighbor -> owner, so ``phi_f = phi[neighbor]``
    * If ``u[face] == 0``: linear average ``0.5 * (phi[owner] + phi[neighbor])``

    Parameters
    ----------
    phi : CellVariable
        Cell-centred field.  Ghost-cell values must be current.
    u : FaceVariable
        Face velocity (or flux).  Positive direction is owner -> neighbor.

    Returns
    -------
    FaceVariable
        Upwind-interpolated values at face centres.

    See Also
    --------
    linearMean, tvdMean
    """
    m = phi.domain
    phi_O = phi._value[m.owner]
    phi_N = phi._value[m.neighbor]
    uf = u._value

    phi_f = np.where(
        uf > 0.0,
        phi_O,
        np.where(
            uf < 0.0,
            phi_N,
            0.5 * (phi_O + phi_N),
        ),
    )
    return FaceVariable(m, phi_f)


def tvdMean(phi: CellVariable, u: FaceVariable, FL) -> FaceVariable:
    """TVD (Total Variation Diminishing) interpolation with flux limiter.

    Applies second-order upwind-biased interpolation with a flux
    limiter to suppress oscillations near discontinuities.

    Parameters
    ----------
    phi : CellVariable
        Cell-centred field.  Ghost-cell values must be current.
    u : FaceVariable
        Face velocity (or flux).  Positive direction is owner -> neighbor.
    FL : callable
        Flux limiter function ``FL(r) -> limiter_value``.  Common
        choices: Van Leer, superbee, minmod, etc.  Available from
        :func:`utilities.fluxLimiter`.

    Returns
    -------
    FaceVariable
        TVD-interpolated values at face centres.

    See Also
    --------
    upwindMean
    """
    raise NotImplementedError(
        "tvdMean is not yet implemented for the connectivity-based "
        "representation.  Use upwindMean as a fallback."
    )
