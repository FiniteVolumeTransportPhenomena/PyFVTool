import numpy as np
from .mesh import *


class FaceVariable:
    def __init__(self,
                 mesh_struct: MeshStructure,
                 xvalue: np.ndarray,
                 yvalue: np.ndarray,
                 zvalue: np.ndarray):
        self.domain = mesh_struct
        self.xvalue = xvalue
        self.yvalue = yvalue
        self.zvalue = zvalue

    def __add__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain,
                                self.xvalue+other.xvalue,
                                self.yvalue+other.yvalue,
                                self.zvalue+other.zvalue)
        else:
            return FaceVariable(self.domain,
                                self.xvalue+other,
                                self.yvalue+other,
                                self.zvalue+other)

    def __radd__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain,
                                self.xvalue+other.xvalue,
                                self.yvalue+other.yvalue,
                                self.zvalue+other.zvalue)
        else:
            return FaceVariable(self.domain,
                                self.xvalue+other,
                                self.yvalue+other,
                                self.zvalue+other)

    def __rsub__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, other.xvalue-self.xvalue,
                                other.yvalue-self.yvalue,
                                other.zvalue-self.zvalue)
        else:
            return FaceVariable(self.domain,
                                other-self.xvalue,
                                other-self.yvalue,
                                other-self.zvalue)

    def __sub__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue-other.xvalue,
                                self.yvalue-other.yvalue,
                                self.zvalue-other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue-other,
                                self.yvalue-other,
                                self.zvalue-other)

    def __mul__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue*other.xvalue,
                                self.yvalue*other.yvalue,
                                self.zvalue*other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue*other,
                                self.yvalue*other,
                                self.zvalue*other)

    def __rmul__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue*other.xvalue,
                                self.yvalue*other.yvalue,
                                self.zvalue*other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue*other,
                                self.yvalue*other,
                                self.zvalue*other)

    def __truediv__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue/other.xvalue,
                                self.yvalue/other.yvalue,
                                self.zvalue/other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue/other,
                                self.yvalue/other,
                                self.zvalue/other)

    def __rtruediv__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, other.xvalue/self.xvalue,
                                other.yvalue/self.yvalue,
                                other.zvalue/self.zvalue)
        else:
            return FaceVariable(self.domain, other/self.xvalue,
                                other/self.yvalue,
                                other/self.zvalue)

    def __neg__(self):
        return FaceVariable(self.domain, -self.xvalue,
                            -self.yvalue,
                            -self.zvalue)

    def __pow__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue**other.xvalue,
                                self.yvalue**other.yvalue,
                                self.zvalue**other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue**other,
                                self.yvalue**other,
                                self.zvalue**other)

    def __rpow__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, other.xvalue**self.xvalue,
                                other.yvalue**self.yvalue,
                                other.zvalue**self.zvalue)
        else:
            return FaceVariable(self.domain, other**self.xvalue,
                                other**self.yvalue,
                                other**self.zvalue)

    def __gt__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue > other.xvalue,
                                self.yvalue > other.yvalue,
                                self.zvalue > other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue > other,
                                self.yvalue > other,
                                self.zvalue > other)

    def __ge__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue >= other.xvalue,
                                self.yvalue >= other.yvalue,
                                self.zvalue >= other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue >= other,
                                self.yvalue >= other,
                                self.zvalue >= other)

    def __lt__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue < other.xvalue,
                                self.yvalue < other.yvalue,
                                self.zvalue < other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue < other,
                                self.yvalue < other,
                                self.zvalue < other)

    def __le__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, self.xvalue <= other.xvalue,
                                self.yvalue <= other.yvalue,
                                self.zvalue <= other.zvalue)
        else:
            return FaceVariable(self.domain, self.xvalue <= other,
                                self.yvalue <= other,
                                self.zvalue <= other)

    def __and__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, np.logical_and(self.xvalue, other.xvalue),
                                np.logical_and(self.yvalue, other.yvalue),
                                np.logical_and(self.zvalue, other.zvalue))
        else:
            return FaceVariable(self.domain, np.logical_and(self.xvalue, other),
                                np.logical_and(self.yvalue, other),
                                np.logical_and(self.zvalue, other))

    def __or__(self, other):
        if type(other) is FaceVariable:
            return FaceVariable(self.domain, np.logical_or(self.xvalue, other.xvalue),
                                np.logical_or(self.yvalue, other.yvalue),
                                np.logical_or(self.zvalue, other.zvalue))
        else:
            return FaceVariable(self.domain, np.logical_or(self.xvalue, other),
                                np.logical_or(self.yvalue, other),
                                np.logical_or(self.zvalue, other))

    def __abs__(self):
        return FaceVariable(self.domain, np.abs(self.xvalue),
                            np.abs(self.yvalue),
                            np.abs(self.zvalue))


def createFaceVariable(mesh, faceval):
    if issubclass(type(mesh), Mesh1D):
        Nx = mesh.dims
        if np.isscalar(faceval):
            return FaceVariable(mesh, faceval*np.ones(Nx+1), np.array([]), np.array([]))
        else:
            return FaceVariable(mesh, faceval[0]*np.ones(Nx+1), np.array([]), np.array([]))
    elif issubclass(type(mesh), Mesh2D):
        Nx, Ny = mesh.dims
        if np.isscalar(faceval):
            return FaceVariable(mesh, faceval*np.ones((Nx+1, Ny)),
                            faceval*np.ones((Nx, Ny+1)), np.array([]))
        else:
            return FaceVariable(mesh, faceval[0]*np.ones((Nx+1, Ny)),
                            faceval[1]*np.ones((Nx, Ny+1)), np.array([]))
    elif issubclass(type(mesh), Mesh3D):
        Nx, Ny, Nz = mesh.dims
        if np.isscalar(faceval):
            return FaceVariable(mesh, faceval*np.ones((Nx+1, Ny, Nz)),
                            faceval*np.ones((Nx, Ny+1, Nz)),
                            faceval*np.ones((Nx, Ny, Nz+1)))
        else:
            return FaceVariable(mesh, faceval[0]*np.ones((Nx+1, Ny, Nz)),
                            faceval[1]*np.ones((Nx, Ny+1, Nz)),
                            faceval[2]*np.ones((Nx, Ny, Nz+1)))
