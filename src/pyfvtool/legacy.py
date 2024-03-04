# Maintaining of backwards compatibility when renaming classes and functions
#
# At a certain moment, we may introduce deprecation warnings or just boldly
# suppress these redirections.

from .boundary import boundaryConditionsTerm
from .boundary import BoundaryConditions

def boundaryConditionTerm(BC):
    """Redirects to boundaryConditionsTerm()"""
    
    return boundaryConditionsTerm(BC)


def createBC(mesh):
    """Redirects to BoundaryConditions() factory function"""
    
    return BoundaryConditions(mesh)
