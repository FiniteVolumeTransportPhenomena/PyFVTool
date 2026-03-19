import numpy as np

from .mesh import Grid1D, Grid2D, Grid3D
from .mesh import CylindricalGrid2D
from .mesh import PolarGrid2D, CylindricalGrid3D, SphericalGrid3D
from .mesh import UnstructuredMesh2D, UnstructuredMesh3D
from .cell import CellVariable
from warnings import warn

import matplotlib.pyplot as plt
from matplotlib import tri
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def visualizeCells(
    phi: CellVariable, vmin=None, vmax=None, cmap="viridis", shading="gouraud"
):
    """
    Visualize the cell variable.

    Parameters
    ----------
    phi: CellVariable
         Cell variable to be visualized
    vmin: float
         Minimum value of the colormap
    vmax: float
         Maximum value of the colormap
    cmap: str
         Colormap
    shading: str
         Shading method

    Examples
    --------
    >>> import pyfvtool as pf
    >>> m = pf.Grid1D(10, 1.0)
    >>> phi = pf.CellVariable(m, 1.0)
    >>> pf.visualizeCells(phi)
    """
    if isinstance(phi.domain, Grid1D):
        x, phi0 = phi.plotprofile()
        plt.plot(x, phi0)
        # plt.show()

    elif (type(phi.domain) is Grid2D) or (type(phi.domain) is CylindricalGrid2D):
        x, y, phi0 = phi.plotprofile()
        if vmin is None:
            vmin = phi0.min()
        if vmax is None:
            vmax = phi0.max()
        plt.pcolormesh(x, y, phi0.T, vmin=vmin, vmax=vmax, cmap=cmap, shading=shading)
        # plt.show()

    elif type(phi.domain) is PolarGrid2D:
        x, y, phi0 = phi.plotprofile()
        plt.subplot(111, polar="true")
        plt.pcolor(y, x, phi0)
        # plt.show()

    elif type(phi.domain) is Grid3D:
        x, y, z, phi0 = phi.plotprofile()
        vmin = np.min(phi0)
        vmax = np.max(phi0)
        mynormalize = lambda a: (a - vmin) / (vmax - vmin)
        Nx, Ny, Nz = phi.domain.dims
        a = np.ones((Nx + 2, Ny + 2, Nz + 2))
        X = x * a
        Y = y * a
        Z = z * a

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(
            X[0, :, :],
            Y[0, :, :],
            Z[0, :, :],
            facecolors=plt.cm.viridis(mynormalize(phi0[0, :, :])),
            alpha=0.8,
        )
        ax.plot_surface(
            X[-1, :, :],
            Y[-1, :, :],
            Z[-1, :, :],
            facecolors=plt.cm.viridis(mynormalize(phi0[-1, :, :])),
            alpha=0.8,
        )
        ax.plot_surface(
            X[:, 0, :],
            Y[:, 0, :],
            Z[:, 0, :],
            facecolors=plt.cm.viridis(mynormalize(phi0[:, 0, :])),
            alpha=0.8,
        )
        ax.plot_surface(
            X[:, -1, :],
            Y[:, -1, :],
            Z[:, -1, :],
            facecolors=plt.cm.viridis(mynormalize(phi0[:, -1, :])),
            alpha=0.8,
        )
        ax.plot_surface(
            X[:, :, 0],
            Y[:, :, 0],
            Z[:, :, 0],
            facecolors=plt.cm.viridis(mynormalize(phi0[:, :, 0])),
            alpha=0.8,
        )
        ax.plot_surface(
            X[:, :, -1],
            Y[:, :, -1],
            Z[:, :, -1],
            facecolors=plt.cm.viridis(mynormalize(phi0[:, :, -1])),
            alpha=0.8,
        )
        # plt.show()

    elif type(phi.domain) is CylindricalGrid3D:
        r, theta, z, phi0 = phi.plotprofile()
        Nx, Ny, Nz = phi.domain.dims
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        vmin = np.min(phi0)
        vmax = np.max(phi0)
        mynormalize = lambda a: (a - vmin) / (vmax - vmin)
        a = np.ones((Nx + 2, Ny + 2, Nz + 2))
        X = x * a
        Y = y * a
        Z = z * a
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        alfa = 1.0
        ax.plot_surface(
            X[:, 0, :],
            Y[:, 0, :],
            Z[:, 0, :],
            facecolors=plt.cm.viridis(mynormalize(phi0[:, 0, :])),
            alpha=alfa,
        )
        ax.plot_surface(
            X[:, int(Ny / 2) + 1, :],
            Y[:, int(Ny / 2) + 1, :],
            Z[:, int(Ny / 2) + 1, :],
            facecolors=plt.cm.viridis(mynormalize(phi0[:, int(Ny / 2) + 1, :])),
            alpha=alfa,
        )
        ax.plot_surface(
            X[:, :, 0],
            Y[:, :, 0],
            Z[:, :, 0],
            facecolors=plt.cm.viridis(mynormalize(phi0[:, :, 0])),
            alpha=alfa,
        )
        ax.plot_surface(
            X[:, :, 0],
            Y[:, :, 0],
            Z[:, :, 0],
            facecolors=plt.cm.viridis(mynormalize(phi0[:, :, 0])),
            alpha=alfa,
        )
        ax.plot_surface(
            X[:, :, int(Nz / 2)],
            Y[:, :, int(Nz / 2)],
            Z[:, :, int(Nz / 2)],
            facecolors=plt.cm.viridis(mynormalize(phi0[:, :, int(Nz / 2)])),
            alpha=alfa,
        )
        ax.plot_surface(
            X[:, :, -1],
            Y[:, :, -1],
            Z[:, :, -1],
            facecolors=plt.cm.viridis(mynormalize(phi0[:, :, -1])),
            alpha=alfa,
        )
        # plt.show()

    elif type(phi.domain) is SphericalGrid3D:
        warn("SphericalGrid3D visualization is not working properly yet.")
        r, theta, PHI, phi0 = phi.plotprofile()
        Nx, Ny, Nz = phi.domain.dims
        x = r * np.sin(theta) * np.cos(PHI)
        y = r * np.sin(theta) * np.sin(PHI)
        z = r * np.cos(theta)
        vmin = np.min(phi0)
        vmax = np.max(phi0)
        mynormalize = lambda a: (a - vmin) / (vmax - vmin)
        a = np.ones((Nx + 2, Ny + 2, Nz + 2))
        X = x * a
        Y = y * a
        Z = z * a
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        alfa = 1.0
        # ax.plot_surface(X[:, 0, :], Y[:, 0, :], Z[:, 0, :],
        #                facecolors=plt.cm.viridis(mynormalize(phi0[:, 0, :])),
        #                alpha=alfa)
        # ax.plot_surface(X[:, int(Ny/2)+1, :], Y[:, int(Ny/2)+1, :], Z[:, int(Ny/2)+1, :],
        #                facecolors=plt.cm.viridis(mynormalize(phi0[:, int(Ny/2)+1, :])),
        #                alpha=alfa)
        # ax.plot_surface(X[:, :, 0], Y[:, :, 0], Z[:, :, 0],
        #                facecolors=plt.cm.viridis(mynormalize(phi0[:, :, 0])),
        #                alpha=alfa)
        # ax.plot_surface(X[:, :, 0], Y[:, :, 0], Z[:, :, 0],
        #                facecolors=plt.cm.viridis(mynormalize(phi0[:, :, 0])),
        #                alpha=alfa)
        # ax.plot_surface(X[:, :, int(Nz/2)], Y[:, :, int(Nz/2)], Z[:, :, int(Nz/2)],
        #                facecolors=plt.cm.viridis(mynormalize(phi0[:, :, int(Nz/2)])),
        #                alpha=alfa)
        ax.plot_surface(
            X[-1, :, :],
            Y[-1, :, :],
            Z[-1, :, :],
            facecolors=plt.cm.viridis(mynormalize(phi0[-1, :, :])),
            alpha=alfa,
        )

    elif isinstance(phi.domain, UnstructuredMesh2D):
        x, y, phi0 = phi.plotprofile()
        # x, y are cell centers (flat arrays), phi0 interior values
        mesh = phi.domain
        triang = tri.Triangulation(mesh._nodes[:, 0], mesh._nodes[:, 1], mesh._cells)
        if vmin is None:
            vmin = phi0.min()
        if vmax is None:
            vmax = phi0.max()
        plt.tripcolor(triang, phi0, vmin=vmin, vmax=vmax, cmap=cmap, shading="flat")
        # plt.show()

    elif isinstance(phi.domain, UnstructuredMesh3D):
        # For tetrahedral mesh, plot boundary faces colored by owner cell value
        warn("UnstructuredMesh3D visualization is experimental.")
        mesh = phi.domain
        # Get boundary faces
        bnd_faces = mesh.boundary_faces
        # For each boundary face, get owner cell index
        owners = mesh.owner[bnd_faces]
        # Get cell values (interior only)
        phi0 = phi.value  # shape (num_cells,)
        face_values = phi0[owners]
        # Get face nodes (shape (N_faces, 3) for 3D triangles)
        face_nodes = mesh._face_nodes[bnd_faces]  # (N_bnd, 3)
        nodes = mesh._nodes  # (N_nodes, 3)
        # Create a triangulation for boundary faces
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Normalize face values for colormap
        if vmin is None:
            vmin = face_values.min()
        if vmax is None:
            vmax = face_values.max()
        norm = plt.Normalize(vmin, vmax)
        cmap_obj = plt.get_cmap(cmap)
        face_colors = cmap_obj(norm(face_values))
        # Create polygons for each boundary triangle
        polys = nodes[face_nodes]  # shape (N_bnd, 3, 3)
        # Create Poly3DCollection
        coll = Poly3DCollection(polys, facecolors=face_colors, linewidth=0)
        ax.add_collection3d(coll)
        # Set axis limits
        ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
        ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
        ax.set_zlim(nodes[:, 2].min(), nodes[:, 2].max())
        # plt.show()

    else:
        # just in case...
        raise ValueError("Unsupported mesh: " + str(type(phi.domain)))


def plot_mesh_2d(mesh, ax=None, cell_value=None, show_cell_centers=False, **kwargs):
    """
    Plot a 2D unstructured mesh (triangular).

    Parameters
    ----------
    mesh : UnstructuredMesh2D
        The mesh to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, current axes is used.
    cell_value : ndarray, shape (mesh.num_cells,), optional
        Cell values for coloring. If None, only mesh edges are drawn.
    show_cell_centers : bool, default False
        If True, plot cell centers as points.
    **kwargs : dict
        Additional keyword arguments passed to tripcolor or triplot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes used for plotting.
    """
    import matplotlib.pyplot as plt
    from matplotlib import tri

    if ax is None:
        ax = plt.gca()

    triang = tri.Triangulation(mesh._nodes[:, 0], mesh._nodes[:, 1], mesh._cells)

    if cell_value is not None:
        # Color by cell value
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        cmap = kwargs.pop("cmap", "viridis")
        shading = kwargs.pop("shading", "flat")
        im = ax.tripcolor(
            triang,
            cell_value,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            shading=shading,
            **kwargs,
        )
    else:
        # Draw mesh edges only
        ax.triplot(triang, **kwargs)

    if show_cell_centers:
        ax.plot(mesh.cellcenters.x, mesh.cellcenters.y, "k.", markersize=2)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_mesh_3d(mesh, ax=None, cell_value=None, **kwargs):
    """
    Plot boundary faces of a 3D unstructured mesh (tetrahedral).

    Parameters
    ----------
    mesh : UnstructuredMesh3D
        The mesh to plot.
    ax : matplotlib.axes.Axes, optional
        3D axes to plot on. If None, a new figure with 3D axes is created.
    cell_value : ndarray, shape (mesh.num_cells,), optional
        Cell values for coloring boundary faces by owner cell value.
        If None, faces are drawn with a single color.
    **kwargs : dict
        Additional keyword arguments passed to Poly3DCollection.

    Returns
    -------
    matplotlib.axes.Axes
        The 3D axes used for plotting.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    # Get boundary faces
    bnd_faces = mesh.boundary_faces
    face_nodes = mesh._face_nodes[bnd_faces]  # (N_bnd, 3)
    nodes = mesh._nodes  # (N_nodes, 3)

    # Create polygons for each boundary triangle
    polys = nodes[face_nodes]  # shape (N_bnd, 3, 3)

    if cell_value is not None:
        # Color by owner cell value
        owners = mesh.owner[bnd_faces]
        face_values = cell_value[owners]
        vmin = kwargs.pop("vmin", face_values.min())
        vmax = kwargs.pop("vmax", face_values.max())
        cmap = kwargs.pop("cmap", "viridis")
        norm = plt.Normalize(vmin, vmax)
        cmap_obj = plt.get_cmap(cmap)
        face_colors = cmap_obj(norm(face_values))
    else:
        face_colors = kwargs.pop("facecolors", "lightgray")

    linewidth = kwargs.pop("linewidth", 0)
    coll = Poly3DCollection(
        polys, facecolors=face_colors, linewidth=linewidth, **kwargs
    )
    ax.add_collection3d(coll)

    # Set axis limits
    ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
    ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
    ax.set_zlim(nodes[:, 2].min(), nodes[:, 2].max())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return ax
