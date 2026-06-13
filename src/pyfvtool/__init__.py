__version__ = "0.5.1"

__author__ = ["Ali A. Eftekhari", "Martinus H. V. Werts"]

__email__ = ["aliak@dtu.dk", "martinus.werts@univ-angers.fr"]

__copyright__ = "Copyright 2023-2026, The Authors"

__license__ = "MIT License"



from .mesh import Grid1D, CylindricalGrid1D, SphericalGrid1D
from .mesh import Grid2D, CylindricalGrid2D, PolarGrid2D 
from .mesh import Grid3D, CylindricalGrid3D, SphericalGrid3D
from .advection import convectionTerm, convectionTVDupwindRHSTerm,\
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
