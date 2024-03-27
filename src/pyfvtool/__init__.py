__version__ = "0.2.1"

__author__ = [
    "Ali A. Eftekhari",
    "Gavin M. Weir",
    "Martinus H. V. Werts"
]

__contact__ = "e.eftekhari@gmail.com"

__copyright__ = "Copyright 2023-2024, The Authors"

__license__ = "MIT License"



ENABLE_LEGACY = False # enable/disable backward compatibility

from .mesh import Grid1D, CylindricalGrid1D, SphericalGrid1D
from .mesh import Grid2D, CylindricalGrid2D, PolarGrid2D 
from .mesh import Grid3D, CylindricalGrid3D, SphericalGrid3D
from .advection import convectionTerm, convectionTvdRHSTerm,\
                       convectionUpwindTerm
from .diffusion import diffusionTerm
from .source import linearSourceTerm, constantSourceTerm, transientTerm
from .boundary import BoundaryConditions, boundaryConditionsTerm
from .utilities import fluxLimiter
from .calculus import gradientTerm, divergenceTerm, gradientTermFixedBC
from .averaging import linearMean, arithmeticMean, upwindMean,\
                       harmonicMean, geometricMean, tvdMean
from .pdesolver import solveMatrixPDE, solvePDE, solveExplicitPDE
from .cell import CellVariable, cellVolume, BC2GhostCells
from .cell import funceval, celleval
from .cell import domainInt, domainIntegrate
from .cell import cellLocations
from .face import FaceVariable, faceeval
from .face import faceLocations
from .visualization import visualizeCells

# legacy naming of certain functions and classes
if ENABLE_LEGACY:
    from .legacy import boundaryConditionTerm
    from .legacy import createBC
    from .legacy import createCellVariable
    from .legacy import createFaceVariable
    from .legacy import createMesh1D
    from .legacy import createMeshCylindrical1D
    from .legacy import createMeshSpherical1D
    from .legacy import createMesh2D
    from .legacy import createMeshCylindrical2D
    from .legacy import createMeshRadial2D
    from .legacy import createMesh3D
    from .legacy import createMeshCylindrical3D
    from .legacy import createMeshSpherical3D
    from .legacy import get_CellVariable_profile1D
    from .legacy import get_CellVariable_profile2D
    from .legacy import get_CellVariable_profile3D    
    