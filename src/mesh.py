"""
Mesh generation
"""
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

class MeshStructure1D(MeshStructure):
  def create_from_size(self, Nx:int, Width) -> MeshStructure:
    pass

  def create_from_faces(self, facelocations):
    pass
