
import numpy as np
from scipy.sparse import csr_array
from mesh import *
from utilities import *
from cell import *
from face import *


def linearMean(phi: CellVariable) -> FaceVariable:
    # calculates the average values of a cell variable. The output is a
    # face variable
    if issubclass(type(phi.domain), Mesh1D):
        dx = phi.domain.cellsize.x
        return FaceVariable(phi.domain,
                     (dx[1:]*phi.value[0:-1]+dx[0:-1] *
                      phi.value[1:])/(dx[1:]+dx[0:-1]),
                     [1.0],
                     [1.0])
    elif issubclass(type(phi.domain), Mesh2D):
        dx = phi.domain.cellsize.x[:,np.newaxis]
        dy = phi.domain.cellsize.y
        return FaceVariable(phi.domain,
                     (dx[1:]*phi.value[0:-1, 1:-1]+dx[0:-1] *
                      phi.value[1:, 1:-1])/(dx[1:]+dx[0:-1]),
                     (dy[1:]*phi.value[1:-1, 0:-1]+dy[0:-1] *
                      phi.value[1:-1, 1:])/(dy[1:]+dy[0:-1]),
                     [1.0])
    elif issubclass(type(phi.domain), Mesh3D):
        dx = phi.domain.cellsize.x[:,np.newaxis,np.newaxis]
        dy = phi.domain.cellsize.y[np.newaxis,:,np.newaxis]
        dz = phi.domain.cellsize.z[np.newaxis,np.newaxis,:]
        return FaceVariable(phi.domain,
                     (dx[1:]*phi.value[0:-1, 1:-1, 1:-1]+dx[0:-1] *
                      phi.value[1:, 1:-1, 1:-1])/(dx[1:]+dx[0:-1]),
                     (dy[:,1:]*phi.value[1:-1, 0:-1, 1:-1]+dy[:,0:-1] *
                      phi.value[1:-1, 1:, 1:-1])/(dy[:,0:-1]+dy[:,1:]),
                     (dz[:,:,1:]*phi.value[1:-1, 1:-1, 0:-1]+dz[:,:,0:-1] *
                      phi.value[1:-1, 1:-1, 1:])/(dz[:,:,0:-1]+dz[:,:,1:]))
