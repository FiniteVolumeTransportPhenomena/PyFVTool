"""
Mesh generation
"""
import numpy as np


class CellSize:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = x
        self.y = y
        self.z = z
    def __str__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""
    def __repr__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""


class CellLocation:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = x
        self.y = y
        self.z = z
    def __str__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""
    def __repr__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""


class FaceLocation:
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray):
        self.x = x
        self.y = y
        self.z = z
    def __str__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""
    def __repr__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""

class MeshStructure:
    def __init__(self, dimension, dims, cellsize,
                 cellcenters, facecenters, corners, edges) -> None:
        self.dimension = dimension
        self.dims = dims
        self.cellsize = cellsize
        self.cellcenters = cellcenters
        self.facecenters = facecenters
        self.corners = corners
        self.edges = edges

    def visualize(self):
        pass

    def __str__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""

    def __repr__(self):
        temp = vars(self)
        for item in temp:
            print(item, ':', temp[item])
        return ""


class Mesh1D(MeshStructure):
    def __init__(self, *args):
        self.dimension = 1
        if len(args) == 1:
            # Use face locations
            facelocationX = args[0]
            Nx = facelocationX.size-1
            cell_size_x = np.hstack([facelocationX[1]-facelocationX[0],
                                     facelocationX[1:]-facelocationX[0:-1],
                                     facelocationX[-1]-facelocationX[-2]])
            cell_size = CellSize(cell_size_x, np.array([0.0]), np.array([0.0]))
            cell_location = CellLocation(
                0.5*(facelocationX[1:]+facelocationX[0:-1]), np.array([0.0]), np.array([0.0]))
            face_location = FaceLocation(
                facelocationX, np.array([0.0]), np.array([0.0]))
        elif len(args) == 2:
            # Use number of cells and domain length
            Nx = args[0]
            Width = args[1]
            dx = Width/Nx
            cell_size = CellSize(
                dx*np.ones(Nx+2), np.array([0.0]), np.array([0.0]))
            cell_location = CellLocation(np.linspace(
                1, Nx, Nx)*dx-dx/2, np.array([0.0]), np.array([0.0]))
            face_location = FaceLocation(np.linspace(
                0, Nx, Nx)*dx, np.array([0.0]), np.array([0.0]))
        
        self.dims = np.array([Nx], dtype=int)
        self.cellsize = cell_size
        self.cellcenters = cell_location
        self.facecenters = face_location
        self.corners = np.array([1], dtype=int)
        self.edges = np.array([1], dtype=int)
    def __repr__(self):
        print(f"1D Cartesian mesh with {self.dims[0]} cells")
        return ""

class Mesh2D(MeshStructure):
    pass

class Mesh3D(MeshStructure):
    pass

class MeshCylindrical1D(MeshStructure):
    pass




