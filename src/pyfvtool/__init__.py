__version__ = "0.3.2"

__author__ = [
    "Ali A. Eftekhari",
    "Gavin M. Weir",
    "Martinus H. V. Werts"
]

__contact__ = "e.eftekhari@gmail.com"

__copyright__ = "Copyright 2023-2024, The Authors"

__license__ = "MIT License"



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
from .cell import CellVariable
from .cell import funceval, celleval
from .cell import cellLocations
from .face import FaceVariable, faceeval
from .face import faceLocations
from .visualization import visualizeCells
    