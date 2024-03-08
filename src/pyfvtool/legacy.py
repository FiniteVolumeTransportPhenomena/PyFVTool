# Maintaining of backwards compatibility when renaming classes and functions
#
# At a certain moment, we may introduce deprecation warnings or just boldly
# suppress these redirections.

from .boundary import boundaryConditionsTerm
from .boundary import BoundaryConditions
from .cell import CellVariable
from .face import FaceVariable
from .mesh import Grid1D, CylindricalGrid1D, SphericalGrid1D
from .mesh import Grid2D

def boundaryConditionTerm(BC):
    """Redirects to boundaryConditionsTerm()"""
    
    return boundaryConditionsTerm(BC)


def createBC(mesh):
    """Redirects to BoundaryConditions() factory function"""
    
    return BoundaryConditions(mesh)


def createCellVariable(*args, **kwargs):
    """Returns a new CellVariable instance"""
    
    return CellVariable(*args, **kwargs)


def createFaceVariable(mesh, faceval):
    return FaceVariable(mesh, faceval)


def createMesh1D(*args) -> Grid1D:
    """Legacy factory function for Grid1D"""
    return Grid1D(*args)


def createMeshCylindrical1D(*args) -> CylindricalGrid1D:
    """Legacy factory function for CylindricalGrid1D"""
    return CylindricalGrid1D(*args)


def createMeshSpherical1D(*args) -> SphericalGrid1D:
    """Legacy factory function for SphericalGrid1D"""
    return SphericalGrid1D(*args)


def createMesh2D(*args) -> Grid2D:
    """Legacy factory function for Grid2D"""
    return Grid2D(*args)
