# Maintaining of backwards compatibility when renaming classes and functions
#
# At a certain moment, we may introduce deprecation warnings or just boldly
# suppress these redirections.

from .boundary import boundaryConditionsTerm
from .boundary import BoundaryConditions
from .cell import CellVariable
from .face import FaceVariable

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
