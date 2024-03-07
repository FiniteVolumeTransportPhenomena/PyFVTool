ENABLE_LEGACY = True # enable/disable backward compatibility

from .mesh import Grid1D, CylindricalGrid1D
from .mesh import createMesh2D, createMesh3D, createMeshSpherical1D, \
                  createMeshCylindrical2D,\
                  createMeshCylindrical3D, createMeshRadial2D,\
                  createMeshSpherical3D
from .advection import convectionTerm, convectionTvdRHSTerm,\
                       convectionUpwindTerm
from .diffusion import diffusionTerm
from .source import linearSourceTerm, constantSourceTerm, transientTerm
from .boundary import BoundaryConditions, boundaryConditionsTerm
from .utilities import fluxLimiter
from .calculus import gradientTerm, divergenceTerm, gradientTermFixedBC
from .averaging import linearMean, arithmeticMean, upwindMean,\
                       harmonicMean, geometricMean, tvdMean
from .pdesolver import solvePDE, solveExplicitPDE
from .cell import CellVariable, cellVolume, BC2GhostCells 
from .cell import copyCellVariable
from .cell import funceval, celleval
from .cell import domainInt, domainIntegrate
from .cell import get_CellVariable_profile1D
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
    

__author__ = (
    "Ali A. Eftekhari"
)
__copyright__ = (
    "Copyright 2023 Ali A. Eftekhari,"
    "MIT License"
)