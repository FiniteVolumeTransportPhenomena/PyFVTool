"""
Example: Unstructured mesh generation with Gmsh and refinement zones.

This script demonstrates the new mesh generation methods in PyFVTool:
- UnstructuredMesh2D.generate_rectangle_with_boundary_refinement
- UnstructuredMesh3D.generate_box_with_boundary_refinement

with optional refinement zones (box, circle in 2D; box, sphere, cylinder in 3D).

Requires Gmsh Python API (optional). If not installed, the script will skip
the Gmsh-dependent sections.

The generated meshes are used to solve a simple diffusion equation.
"""

try:
    import gmsh

    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False
    print("Gmsh Python API not installed. Skipping mesh generation examples.")
    print("Install with: uv pip install gmsh")

import numpy as np
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import pyfvtool in development mode
sys.path.insert(0, "../..")

try:
    import pyfvtool as pf
except ImportError:
    print("PyFVTool not found. Ensure you have installed the package.")
    sys.exit(1)


def example_2d_boundary_refinement():
    """Generate a 2D triangular mesh with boundary refinement."""
    if not HAS_GMSH:
        print("Skipping 2D boundary refinement example (Gmsh missing).")
        return
    print("=== 2D rectangle with boundary refinement ===")
    mesh = pf.UnstructuredMesh2D.generate_rectangle_with_boundary_refinement(
        Lx=2.0,
        Ly=1.0,
        background_size=0.1,
        boundary_refinement_distance=0.2,
        boundary_refinement_size=0.02,
        physical_group_map={1: "left", 2: "right", 3: "bottom", 4: "top"},
    )
    print(f"Mesh: {mesh.num_cells} cells, {mesh.num_faces} faces")
    print("Boundary tags:", list(mesh.boundary_tags.keys()))

    # Plot mesh
    fig, ax = plt.subplots(figsize=(8, 4))
    pf.visualization.plot_mesh_2d(mesh, ax=ax, show_cell_centers=False)
    ax.set_title("2D triangular mesh with boundary refinement")
    plt.tight_layout()
    plt.show()

    return mesh


def example_2d_box_zone():
    """Add a box refinement zone."""
    if not HAS_GMSH:
        print("Skipping 2D box zone example (Gmsh missing).")
        return
    print("=== 2D rectangle with box refinement zone ===")
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
        background_size=0.1,
        boundary_refinement_distance=0.2,
        boundary_refinement_size=0.02,
        refinement_zones=refinement_zones,
    )
    print(f"Mesh: {mesh.num_cells} cells, {mesh.num_faces} faces")

    fig, ax = plt.subplots(figsize=(8, 4))
    pf.visualization.plot_mesh_2d(mesh, ax=ax, show_cell_centers=False)
    ax.set_title("2D mesh with box refinement zone")
    plt.tight_layout()
    plt.show()

    return mesh


def example_2d_circle_zone():
    """Add a circle refinement zone."""
    if not HAS_GMSH:
        print("Skipping 2D circle zone example (Gmsh missing).")
        return
    print("=== 2D rectangle with circle refinement zone ===")
    refinement_zones = [
        {
            "type": "circle",
            "parameters": {"center": (1.0, 0.5), "radius": 0.2},
            "refinement_size": 0.005,
            "distance_max": 0.1,
        }
    ]
    mesh = pf.UnstructuredMesh2D.generate_rectangle_with_boundary_refinement(
        Lx=2.0,
        Ly=1.0,
        background_size=0.1,
        boundary_refinement_distance=0.2,
        boundary_refinement_size=0.02,
        refinement_zones=refinement_zones,
    )
    print(f"Mesh: {mesh.num_cells} cells, {mesh.num_faces} faces")

    fig, ax = plt.subplots(figsize=(8, 4))
    pf.visualization.plot_mesh_2d(mesh, ax=ax, show_cell_centers=False)
    ax.set_title("2D mesh with circle refinement zone")
    plt.tight_layout()
    plt.show()

    return mesh


def solve_diffusion_2d(mesh):
    """Solve a simple diffusion equation on the given mesh."""
    print("=== Solving diffusion equation on 2D mesh ===")
    # Define diffusion coefficient
    D = pf.CellVariable(mesh, 1.0)
    # Define boundary conditions
    BC = pf.BoundaryConditions(mesh)
    BC["left"].a[:] = 0.0
    BC["left"].b[:] = 1.0
    BC["left"].c[:] = 1.0  # Dirichlet phi = 1
    BC["right"].a[:] = 0.0
    BC["right"].b[:] = 1.0
    BC["right"].c[:] = 0.0  # Dirichlet phi = 0
    # Top and bottom: zero flux (Neumann a=1, b=0, c=0)
    for tag in ["top", "bottom"]:
        if tag in BC:
            BC[tag].a[:] = 1.0
            BC[tag].b[:] = 0.0
            BC[tag].c[:] = 0.0

    # Create initial guess
    phi = pf.CellVariable(mesh, 0.5, BC)

    # Discretize
    M_diff = pf.diffusionTerm(D)
    # No source
    RHS = pf.sourceTerm(pf.CellVariable(mesh, 0.0))

    # Solve
    phi = pf.solveMatrixPDE(M_diff, RHS, phi)
    print(f"Solution range: {phi.value.min():.3f} .. {phi.value.max():.3f}")

    # Plot solution
    fig, ax = plt.subplots(figsize=(8, 4))
    pf.visualization.plot_mesh_2d(
        mesh, ax=ax, cell_value=phi.value, show_cell_centers=False
    )
    ax.set_title("Diffusion solution on refined mesh")
    plt.tight_layout()
    plt.show()

    return phi


def example_3d_boundary_refinement():
    """Generate a 3D tetrahedral mesh with boundary refinement."""
    if not HAS_GMSH:
        print("Skipping 3D boundary refinement example (Gmsh missing).")
        return
    print("=== 3D box with boundary refinement ===")
    mesh = pf.UnstructuredMesh3D.generate_box_with_boundary_refinement(
        Lx=1.0,
        Ly=1.0,
        Lz=0.5,
        background_size=0.15,
        boundary_refinement_distance=0.1,
        boundary_refinement_size=0.03,
        physical_group_map={
            1: "left",
            2: "right",
            3: "bottom",
            4: "top",
            5: "front",
            6: "back",
        },
    )
    print(f"Mesh: {mesh.num_cells} cells, {mesh.num_faces} faces")
    print("Boundary tags:", list(mesh.boundary_tags.keys()))

    # 3D plot (optional, may be slow)
    # pf.visualization.plot_mesh_3d(mesh)
    # plt.title("3D tetrahedral mesh with boundary refinement")
    # plt.show()

    return mesh


def example_3d_sphere_zone():
    """Add a sphere refinement zone."""
    if not HAS_GMSH:
        print("Skipping 3D sphere zone example (Gmsh missing).")
        return
    print("=== 3D box with sphere refinement zone ===")
    refinement_zones = [
        {
            "type": "sphere",
            "parameters": {"center": (0.5, 0.5, 0.25), "radius": 0.15},
            "refinement_size": 0.01,
            "distance_max": 0.05,
        }
    ]
    mesh = pf.UnstructuredMesh3D.generate_box_with_boundary_refinement(
        Lx=1.0,
        Ly=1.0,
        Lz=0.5,
        background_size=0.15,
        boundary_refinement_distance=0.1,
        boundary_refinement_size=0.03,
        refinement_zones=refinement_zones,
    )
    print(f"Mesh: {mesh.num_cells} cells, {mesh.num_faces} faces")

    return mesh


def main():
    """Run all examples."""
    print("PyFVTool unstructured mesh generation with Gmsh")
    print("=" * 50)

    # 2D examples
    mesh1 = example_2d_boundary_refinement()
    mesh2 = example_2d_box_zone()
    mesh3 = example_2d_circle_zone()

    # Solve PDE on one of the meshes
    if HAS_GMSH and mesh1 is not None:
        solve_diffusion_2d(mesh1)

    # 3D examples
    mesh4 = example_3d_boundary_refinement()
    mesh5 = example_3d_sphere_zone()

    print("\nAll examples completed.")


if __name__ == "__main__":
    main()
