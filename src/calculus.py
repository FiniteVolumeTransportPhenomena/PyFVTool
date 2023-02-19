import numpy as np
from scipy.sparse import csr_array
from mesh import *
from utilities import *
from cell import *
from face import *


def gradientTerm(phi: CellVariable):
    # calculates the gradient of a variable
    # the output is a face variable
    if issubclass(type(phi.domain), Mesh1D):
        dx = 0.5*(phi.domain.cellsize.x[0:-1]+phi.domain.cellsize.x[1:])
        FaceVariable(phi.domain,
                     (phi.value[1:]-phi.value[0:-1])/dx,
                     [1.0],
                     [1.0])
    elif (type(phi.domain) is Mesh2D) or (type(phi.domain) is MeshCylindrical2D):
        dx = 0.5*(phi.domain.cellsize.x[0:-1]+phi.domain.cellsize.x[1:])
        dy = 0.5*(phi.domain.cellsize.y[0:-1]+phi.domain.cellsize.y[1:])
        FaceVariable(phi.domain,
                     (phi.value[1:, 1:-1]-phi.value[0:-1, 1:-1])/dx[:,np.newaxis],
                     (phi.value[1:-1, 1:]-phi.value[1:-1, 0:-1])/dy,
                     [1.0])
    elif (type(phi.domain) is MeshRadial2D):
        dx = 0.5*(phi.domain.cellsize.x[0:-1]+phi.domain.cellsize.x[1:])
        dtheta = 0.5*(phi.domain.cellsize.y[0:-1]+phi.domain.cellsize.y[1:])
        rp = phi.domain.cellcenters.x
        FaceVariable(phi.domain,
                     (phi.value[1:, 1:-1]-phi.value[0:-1, 1:-1])/dx[:,np.newaxis],
                     (phi.value[1:-1, 1:]-phi.value[1:-1, 0:-1])/(dtheta[np.newaxis,:]*rp[:,np.newaxis]),
                     [1.0])
    elif (type(phi.domain) is Mesh3D):
        dx = 0.5*(phi.domain.cellsize.x[0:-1]+phi.domain.cellsize.x[1:])
        dy = 0.5*(phi.domain.cellsize.y[0:-1]+phi.domain.cellsize.y[1:])
        dz = 0.5*(phi.domain.cellsize.z[0:-1]+phi.domain.cellsize.z[1:])
        FaceVariable(phi.domain,
                     (phi.value[1:, 1:-1, 1:-1] -
                     phi.value[0:-1, 1:-1, 1:-1])/dx[:,np.newaxis,np.newaxis],
                     (phi.value[1:-1, 1:, 1:-1] -
                      phi.value[1:-1, 0:-1, 1:-1])/dy[np.newaxis,:,np.newaxis],
                     (phi.value[1:-1, 1:-1, 1:] -
                     phi.value[1:-1, 1:-1, 0:-1])/dz[np.newaxis,np.newaxis,:])
    elif (type(phi.domain) is MeshCylindrical3D):
        dx = 0.5*(phi.domain.cellsize.x[0:-1]+phi.domain.cellsize.x[1:])
        dy = 0.5*(phi.domain.cellsize.y[0:-1]+phi.domain.cellsize.y[1:])
        dz = 0.5*(phi.domain.cellsize.z[0:-1]+phi.domain.cellsize.z[1:])
        rp = phi.domain.cellcenters.x
        FaceVariable(phi.domain,
                     (phi.value[1:, 1:-1, 1:-1] -
                      phi.value[0:-1, 1:-1, 1:-1])/dx[:,np.newaxis,np.newaxis],
                     (phi.value[1:-1, 1:, 1:-1] -
                      phi.value[1:-1, 0:-1, 1:-1])/(dy[np.newaxis,:,np.newaxis]*rp[:,np.newaxis,np.newaxis]),
                     (phi.value[1:-1, 1:-1, 1:] -
                     phi.value[1:-1, 1:-1, 0:-1])/dz[np.newaxis,np.newaxis,:])
