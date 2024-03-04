# Maintaining of backwards compatibility when renaming classes and functions
#
# At a certain moment, we may introduce deprecation warnings or just boldly
# suppress these redirections.

from .boundary import boundaryConditionsTerm

def boundaryConditionTerm(BC):
    """Redirects to boundaryConditionsTerm()"""
    
    return boundaryConditionsTerm(BC)


