import numpy as np
from mesh import *

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

def createFaceVariable(mesh, faceval):
    if issubclass(type(mesh), Mesh1D):
        Nx = mesh.dims
        return FaceVariable(mesh, faceval[0]*np.ones(Nx), np.array([]), np.array([]))
    elif issubclass(type(mesh), Mesh2D):
        Nx, Ny = mesh.dims
        return FaceVariable(mesh, faceval[0]*np.ones((Nx+1, Ny)), 
                            faceval[1]*np.ones((Nx, Ny+1)), np.array([]))
    elif issubclass(type(mesh), Mesh3D):
        Nx, Ny, Nz = mesh.dims
        return FaceVariable(mesh, faceval[0]*np.ones((Nx+1, Ny, Nz)), 
                            faceval[1]*np.ones((Nx, Ny+1, Nz)), 
                            faceval[2]*np.ones((Nx, Ny, Nz+1)))