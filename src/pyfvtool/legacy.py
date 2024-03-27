# Maintaining of backwards compatibility when renaming classes and functions
#
# At a certain moment, we may introduce deprecation warnings or just boldly
# suppress these redirections.

from .boundary import boundaryConditionsTerm
from .boundary import BoundaryConditions
from .cell import CellVariable
from .face import FaceVariable
from .mesh import Grid1D, CylindricalGrid1D, SphericalGrid1D
from .mesh import Grid2D, CylindricalGrid2D, PolarGrid2D
from .mesh import Grid3D, CylindricalGrid3D, SphericalGrid3D



def boundaryConditionTerm(BC):
    """Redirects to boundaryConditionsTerm()"""
    return boundaryConditionsTerm(BC)


def createBC(mesh):
    """Redirects to BoundaryConditions() factory function"""
    return BoundaryConditions(mesh)


def createCellVariable(*args, **kwargs):
    """Legacy factory function returning a new CellVariable instance"""
    return CellVariable(*args, **kwargs)


def createFaceVariable(mesh, faceval):
    """Legacy factory function for FaceVariable"""
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


def createMeshCylindrical2D(*args) -> CylindricalGrid2D:
    """Legacy factory function for CylindricalGrid2D"""
    return CylindricalGrid2D(*args)


def createMeshRadial2D(*args) -> PolarGrid2D:
    """Legacy factory function for PolarGrid2D"""
    return PolarGrid2D(*args)


def createMesh3D(*args) -> Grid3D:
    """Legacy factory function for Grid3D"""
    return Grid3D(*args)


def createMeshCylindrical3D(*args) -> CylindricalGrid3D:
    """Legacy factory function for CylindricalGrid3D"""
    return CylindricalGrid3D(*args)


def createMeshSpherical3D(*args) -> SphericalGrid3D:
    """Legacy factory function for SphericalGrid3D"""
    return SphericalGrid3D(*args)


def get_CellVariable_profile1D(phi: CellVariable):
    """Legacy function"""
    return phi.plotprofile()

def get_CellVariable_profile2D(phi: CellVariable):
    return phi.plotprofile()

def get_CellVariable_profile3D(phi: CellVariable):
    return phi.plotprofile()

def domainInt(phi: CellVariable) -> float:
    return phi.domainIntegral()


