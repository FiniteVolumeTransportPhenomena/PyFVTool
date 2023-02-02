"""
Mesh generation
"""
class MeshStructure:
    def __init__(self, dimension, dims, cellsize,
          cellcenters, facecenters, corners, edges):
        self.dimension = dimension
        self.dims = dims
        self.cellsize = dimension
        self.cellcenters = cellcenters
        self.facecenters = facecenters
        self.corners = corners
        self.edges = edges
