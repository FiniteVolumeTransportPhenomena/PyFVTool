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


def faceeval(f, *args):
    if len(args)==1:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue),
                            f(args[0].yvalue),
                            f(args[0].zvalue))
    elif len(args)==2:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue),
                            f(args[0].yvalue, args[1].yvalue),
                            f(args[0].zvalue, args[1].zvalue))
    elif len(args)==3:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue, args[2].xvalue),
                            f(args[0].yvalue, args[1].yvalue, args[2].yvalue),
                            f(args[0].zvalue, args[1].zvalue, args[2].zvalue))
    elif len(args)==4:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue, args[2].xvalue, args[3].xvalue),
                            f(args[0].yvalue, args[1].yvalue, args[2].yvalue, args[3].yvalue),
                            f(args[0].zvalue, args[1].zvalue, args[2].zvalue, args[3].zvalue))
    elif len(args)==5:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue, args[2].xvalue, args[3].xvalue, args[4].xvalue),
                            f(args[0].yvalue, args[1].yvalue, args[2].yvalue, args[3].yvalue, args[4].yvalue),
                            f(args[0].zvalue, args[1].zvalue, args[2].zvalue, args[3].zvalue, args[4].zvalue))
    elif len(args)==6:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue, args[2].xvalue, args[3].xvalue, args[4].xvalue, args[5].xvalue),
                            f(args[0].yvalue, args[1].yvalue, args[2].yvalue, args[3].yvalue, args[4].yvalue, args[5].yvalue),
                            f(args[0].zvalue, args[1].zvalue, args[2].zvalue, args[3].zvalue, args[4].zvalue, args[5].zvalue))
    elif len(args)==7:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue, args[2].xvalue, args[3].xvalue, args[4].xvalue, args[5].xvalue, args[6].xvalue),
                            f(args[0].yvalue, args[1].yvalue, args[2].yvalue, args[3].yvalue, args[4].yvalue, args[5].yvalue, args[6].yvalue),
                            f(args[0].zvalue, args[1].zvalue, args[2].zvalue, args[3].zvalue, args[4].zvalue, args[5].zvalue, args[6].zvalue))
    elif len(args)==8:
        return FaceVariable(args[0].domain,
                            f(args[0].xvalue, args[1].xvalue, args[2].xvalue, args[3].xvalue, args[4].xvalue, args[5].xvalue, args[6].xvalue, args[7].xvalue),
                            f(args[0].yvalue, args[1].yvalue, args[2].yvalue, args[3].yvalue, args[4].yvalue, args[5].yvalue, args[6].yvalue, args[7].yvalue),
                            f(args[0].zvalue, args[1].zvalue, args[2].zvalue, args[3].zvalue, args[4].zvalue, args[5].zvalue, args[6].zvalue, args[7].zvalue))
