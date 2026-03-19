#!/usr/bin/env python3
"""Tests for unstructured mesh support."""

import unittest
import numpy as np
import pyfvtool as pf


# Use the method from UnstructuredMesh2D
geometric_boundary_tags = (
    lambda mesh, x_range=(0.0, 1.0), y_range=(0.0, 1.0), tol=1e-6: (
        pf.UnstructuredMesh2D.geometric_boundary_tags(mesh, x_range, y_range, tol)
    )
)


class TestUnstructuredMesh(unittest.TestCase):
    def _create_valid_cube_mesh(self):
        """Return (nodes, cells, boundary_tags) for a valid tetrahedral cube mesh."""
        import numpy as np
        from scipy.spatial import Delaunay

        # Unit cube vertices
        nodes = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        )
        # Add interior point to avoid degenerate tetrahedra
        nodes = np.vstack([nodes, [0.5, 0.5, 0.5]])
        # Delaunay tetrahedralization
        tri = Delaunay(nodes)
        cells = tri.simplices
        # Create temporary mesh to compute face centers for tagging
        import pyfvtool as pf

        mesh_temp = pf.UnstructuredMesh3D(nodes, cells)
        # Tag boundaries by geometric location
        boundary_tags = {}
        tol = 1e-6
        fc = mesh_temp._face_centers
        boundary_tags["left"] = np.where(fc[:, 0] < tol)[0]
        boundary_tags["right"] = np.where(fc[:, 0] > 1.0 - tol)[0]
        boundary_tags["bottom"] = np.where(fc[:, 1] < tol)[0]
        boundary_tags["top"] = np.where(fc[:, 1] > 1.0 - tol)[0]
        boundary_tags["front"] = np.where(fc[:, 2] < tol)[0]
        boundary_tags["back"] = np.where(fc[:, 2] > 1.0 - tol)[0]
        return nodes, cells, boundary_tags

    def test_single_triangle(self):
        """Basic creation of a single triangle mesh."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        cells = np.array([[0, 1, 2]])
        mesh = pf.UnstructuredMesh2D(nodes, cells)
        self.assertEqual(mesh.num_cells, 1)
        self.assertEqual(mesh.num_faces, 3)
        self.assertEqual(mesh.num_ghost_cells, 3)
        self.assertIn("boundary", mesh.boundary_tags)
        self.assertEqual(len(mesh.boundary_tags["boundary"]), 3)

    def test_two_triangles_square(self):
        """Two triangles forming a unit square."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        cells = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = pf.UnstructuredMesh2D(nodes, cells)
        self.assertEqual(mesh.num_cells, 2)
        self.assertEqual(mesh.num_faces, 5)  # 4 boundary edges + 1 internal edge
        self.assertEqual(mesh.num_ghost_cells, 4)
        # All boundary faces tagged as "boundary"
        self.assertEqual(len(mesh.boundary_tags["boundary"]), 4)

    def test_from_delaunay(self):
        """Create mesh via Delaunay triangulation."""
        mesh = pf.UnstructuredMesh2D.from_delaunay(5, 4, 1.0, 1.0)
        self.assertGreater(mesh.num_cells, 0)
        self.assertGreater(mesh.num_faces, 0)
        # Check that all cells have positive volume
        self.assertTrue(np.all(mesh.cell_volumes > 0.0))

    def test_diffusion_dirichlet(self):
        """Laplace equation with zero Dirichlet BC on all boundaries."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        cells = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = pf.UnstructuredMesh2D(nodes, cells)
        BC = pf.BoundaryConditions(mesh)
        BC["boundary"].a[:] = 0.0
        BC["boundary"].b[:] = 1.0
        BC["boundary"].c[:] = 0.0
        phi = pf.CellVariable(mesh, 0.0, BC)
        D = pf.CellVariable(mesh, 1.0)
        Dave = pf.harmonicMean(D)
        Mdiff = pf.diffusionTerm(Dave)
        Mbc, RHSbc = pf.boundaryConditionsTerm(BC)
        M = Mdiff + Mbc
        RHS = RHSbc
        phi_sol = pf.solveMatrixPDE(mesh, M, RHS)
        # Solution should be zero everywhere
        self.assertTrue(np.allclose(phi_sol.value, 0.0, atol=1e-12))

    def test_geometric_boundary_tags(self):
        """Test automatic tagging of boundaries by geometric location."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        cells = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = pf.UnstructuredMesh2D(nodes, cells)
        tags = geometric_boundary_tags(mesh, (0.0, 1.0), (0.0, 1.0))
        self.assertEqual(set(tags.keys()), {"left", "right", "bottom", "top"})
        # Check each tag has exactly one face
        self.assertEqual(len(tags["left"]), 1)
        self.assertEqual(len(tags["right"]), 1)
        self.assertEqual(len(tags["bottom"]), 1)
        self.assertEqual(len(tags["top"]), 1)
        # Verify face centers
        fc = mesh.facecenters
        for f in tags["left"]:
            self.assertAlmostEqual(fc.x[f], 0.0, places=6)
        for f in tags["right"]:
            self.assertAlmostEqual(fc.x[f], 1.0, places=6)
        for f in tags["bottom"]:
            self.assertAlmostEqual(fc.y[f], 0.0, places=6)
        for f in tags["top"]:
            self.assertAlmostEqual(fc.y[f], 1.0, places=6)

    def test_diffusion_mixed_bc(self):
        """Diffusion with Dirichlet left, Neumann right, bottom/top zero flux."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        cells = np.array([[0, 1, 2], [0, 2, 3]])
        mesh = pf.UnstructuredMesh2D(nodes, cells)
        # Use geometric tagging
        boundary_tags = geometric_boundary_tags(mesh, (0.0, 1.0), (0.0, 1.0))
        # Re-create mesh with custom boundary tags (overwrites default "boundary")
        mesh2 = pf.UnstructuredMesh2D(nodes, cells, boundary_tags)
        BC = pf.BoundaryConditions(mesh2)
        # Left: Dirichlet phi = 1
        BC.left.a[:] = 0.0
        BC.left.b[:] = 1.0
        BC.left.c[:] = 1.0
        # Right: Neumann dphi/dn = 0 (default a=1,b=0,c=0)
        # Bottom: Neumann zero flux (default)
        # Top: Neumann zero flux (default)
        D = pf.CellVariable(mesh2, 1.0)
        Dave = pf.harmonicMean(D)
        Mdiff = pf.diffusionTerm(Dave)
        Mbc, RHSbc = pf.boundaryConditionsTerm(BC)
        M = Mdiff + Mbc
        RHS = RHSbc
        phi_sol = pf.solveMatrixPDE(mesh2, M, RHS)
        # Since right/bottom/top are insulated, left is constant phi=1,
        # steady-state solution should be phi = 1 everywhere.
        self.assertTrue(np.allclose(phi_sol.value, 1.0, atol=1e-12))

    def test_convection_diffusion_2d(self):
        """Steady convection-diffusion with Dirichlet BCs, compare analytical."""
        import math

        # Create a rectangular domain triangulation
        mesh = pf.UnstructuredMesh2D.from_delaunay(10, 5, 1.0, 0.5)
        # Use geometric boundary tagging
        boundary_tags = geometric_boundary_tags(mesh, (0.0, 1.0), (0.0, 0.5))
        # Re-create mesh with custom boundary tags (overwrites default "boundary")
        mesh2 = pf.UnstructuredMesh2D(mesh._nodes, mesh._cells, boundary_tags)
        BC = pf.BoundaryConditions(mesh2)
        # Left: Dirichlet phi = 0
        BC.left.a[:] = 0.0
        BC.left.b[:] = 1.0
        BC.left.c[:] = 0.0
        # Right: Dirichlet phi = 1
        BC.right.a[:] = 0.0
        BC.right.b[:] = 1.0
        BC.right.c[:] = 1.0
        # Top/bottom: zero flux (default Neumann a=1,b=0,c=0)
        # Parameters
        u = 1.0  # velocity in x-direction
        D = 10.0  # diffusion coefficient
        L = 1.0  # domain length
        Pe = u * L / D  # Peclet number = 0.1
        # Velocity field: constant x-direction
        u_face = pf.FaceVariable(mesh2, 0.0)
        u_face._value[:] = u * mesh2.face_normals[:, 0]
        # Diffusion coefficient
        D_cell = pf.CellVariable(mesh2, D)
        D_face = pf.harmonicMean(D_cell)
        # Build terms
        Mconv = pf.convectionUpwindTerm(u_face)
        Mdiff = pf.diffusionTerm(D_face)
        Mbc, RHSbc = pf.boundaryConditionsTerm(BC)
        M = Mconv + Mdiff + Mbc
        RHS = RHSbc
        # Solve
        phi_sol = pf.solveMatrixPDE(mesh2, M, RHS)
        # Analytical solution phi(x) = (exp(Pe * x/L) - 1) / (exp(Pe) - 1)
        x_cells = mesh2._cell_centers[:, 0]
        phi_analytical = (np.exp(Pe * x_cells / L) - 1.0) / (np.exp(Pe) - 1.0)
        # Compare (allow some error due to mesh discretization)
        self.assertTrue(np.allclose(phi_sol.value, phi_analytical, atol=0.1))

    def test_diffusion_3d(self):
        """Diffusion in a tetrahedral mesh of a unit cube with Dirichlet BC."""
        nodes, cells, boundary_tags = self._create_valid_cube_mesh()
        import pyfvtool as pf
        import numpy as np

        # Re-create mesh with these tags
        mesh2 = pf.UnstructuredMesh3D(nodes, cells, boundary_tags)
        BC = pf.BoundaryConditions(mesh2)
        # Left: Dirichlet phi = 0
        BC.left.a[:] = 0.0
        BC.left.b[:] = 1.0
        BC.left.c[:] = 0.0
        # Right: Dirichlet phi = 1
        BC.right.a[:] = 0.0
        BC.right.b[:] = 1.0
        BC.right.c[:] = 1.0
        # Other faces: zero flux (default)
        D = pf.CellVariable(mesh2, 1.0)
        D_face = pf.harmonicMean(D)
        Mdiff = pf.diffusionTerm(D_face)
        Mbc, RHSbc = pf.boundaryConditionsTerm(BC)
        M = Mdiff + Mbc
        RHS = RHSbc
        phi_sol = pf.solveMatrixPDE(mesh2, M, RHS)
        # Expect linear variation in x-direction: phi(x) ≈ x
        x_cells = mesh2._cell_centers[:, 0]
        self.assertTrue(np.allclose(phi_sol.value, x_cells, atol=0.05))

    def test_convection_diffusion_3d(self):
        """Steady convection-diffusion in 3D tetrahedral mesh."""
        nodes, cells, boundary_tags = self._create_valid_cube_mesh()
        import pyfvtool as pf
        import numpy as np

        mesh2 = pf.UnstructuredMesh3D(nodes, cells, boundary_tags)
        BC = pf.BoundaryConditions(mesh2)
        BC.left.a[:] = 0.0
        BC.left.b[:] = 1.0
        BC.left.c[:] = 0.0
        BC.right.a[:] = 0.0
        BC.right.b[:] = 1.0
        BC.right.c[:] = 1.0
        # Constant velocity in x-direction
        u = 1.0
        D = 10.0
        L = 1.0
        Pe = u * L / D
        u_face = pf.FaceVariable(mesh2, 0.0)
        # Face normal velocity component: u * n_x
        u_face._value[:] = u * mesh2.face_normals[:, 0]
        D_cell = pf.CellVariable(mesh2, D)
        D_face = pf.harmonicMean(D_cell)
        Mconv = pf.convectionUpwindTerm(u_face)
        Mdiff = pf.diffusionTerm(D_face)
        Mbc, RHSbc = pf.boundaryConditionsTerm(BC)
        M = Mconv + Mdiff + Mbc
        RHS = RHSbc
        phi_sol = pf.solveMatrixPDE(mesh2, M, RHS)
        # Analytical solution phi(x) = (exp(Pe * x/L) - 1) / (exp(Pe) - 1)
        x_cells = mesh2._cell_centers[:, 0]
        phi_analytical = (np.exp(Pe * x_cells / L) - 1.0) / (np.exp(Pe) - 1.0)
        self.assertTrue(np.allclose(phi_sol.value, phi_analytical, atol=0.05))


class TestGmshMeshGeneration(unittest.TestCase):
    """Test Gmsh-based mesh generation methods (skip if Gmsh not installed)."""

    @classmethod
    def setUpClass(cls):
        try:
            import gmsh

            cls.has_gmsh = True
        except ImportError:
            cls.has_gmsh = False

    def test_generate_rectangle_with_boundary_refinement(self):
        if not self.has_gmsh:
            self.skipTest("Gmsh Python API not installed")
        import pyfvtool as pf

        mesh = pf.UnstructuredMesh2D.generate_rectangle_with_boundary_refinement(
            Lx=2.0,
            Ly=1.0,
            background_size=0.2,
            boundary_refinement_distance=0.2,
            boundary_refinement_size=0.05,
        )
        self.assertIsInstance(mesh, pf.UnstructuredMesh2D)
        self.assertGreater(mesh.num_cells, 10)
        self.assertGreater(mesh.num_faces, 10)
        # Check boundary tags exist
        self.assertIn("left", mesh.boundary_tags)
        self.assertIn("right", mesh.boundary_tags)
        self.assertIn("bottom", mesh.boundary_tags)
        self.assertIn("top", mesh.boundary_tags)

    def test_generate_rectangle_with_box_zone(self):
        if not self.has_gmsh:
            self.skipTest("Gmsh Python API not installed")
        import pyfvtool as pf

        refinement_zones = [
            {
                "type": "box",
                "parameters": {"xmin": 0.5, "xmax": 1.5, "ymin": 0.3, "ymax": 0.7},
                "refinement_size": 0.01,
                "distance_max": 0.1,
            }
        ]
        mesh = pf.UnstructuredMesh2D.generate_rectangle_with_boundary_refinement(
            Lx=2.0,
            Ly=1.0,
            background_size=0.2,
            boundary_refinement_distance=0.2,
            boundary_refinement_size=0.05,
            refinement_zones=refinement_zones,
        )
        self.assertIsInstance(mesh, pf.UnstructuredMesh2D)
        self.assertGreater(mesh.num_cells, 10)

    def test_generate_box_with_boundary_refinement(self):
        if not self.has_gmsh:
            self.skipTest("Gmsh Python API not installed")
        import pyfvtool as pf

        mesh = pf.UnstructuredMesh3D.generate_box_with_boundary_refinement(
            Lx=1.0,
            Ly=1.0,
            Lz=0.5,
            background_size=0.2,
            boundary_refinement_distance=0.1,
            boundary_refinement_size=0.05,
        )
        self.assertIsInstance(mesh, pf.UnstructuredMesh3D)
        self.assertGreater(mesh.num_cells, 10)
        self.assertGreater(mesh.num_faces, 10)
        # Check boundary tags exist (default mapping)
        self.assertIn("left", mesh.boundary_tags)
        self.assertIn("right", mesh.boundary_tags)
        self.assertIn("bottom", mesh.boundary_tags)
        self.assertIn("top", mesh.boundary_tags)
        self.assertIn("front", mesh.boundary_tags)
        self.assertIn("back", mesh.boundary_tags)

    def test_generate_box_with_sphere_zone(self):
        if not self.has_gmsh:
            self.skipTest("Gmsh Python API not installed")
        import pyfvtool as pf

        refinement_zones = [
            {
                "type": "sphere",
                "parameters": {"center": (0.5, 0.5, 0.25), "radius": 0.15},
                "refinement_size": 0.01,
                "distance_max": 0.05,
            }
        ]
        try:
            mesh = pf.UnstructuredMesh3D.generate_box_with_boundary_refinement(
                Lx=1.0,
                Ly=1.0,
                Lz=0.5,
                background_size=0.2,
                boundary_refinement_distance=0.1,
                boundary_refinement_size=0.05,
                refinement_zones=refinement_zones,
            )
        except Exception as e:
            if "Unknown field type" in str(e):
                self.skipTest(f"Gmsh does not support Sphere field: {e}")
            raise
        self.assertIsInstance(mesh, pf.UnstructuredMesh3D)
        self.assertGreater(mesh.num_cells, 10)
