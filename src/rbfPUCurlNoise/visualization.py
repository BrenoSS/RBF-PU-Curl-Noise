import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from scipy.interpolate import LinearNDInterpolator

from .triang_mesh import TriMesh
from .patches import PatchCoverage

def plot_combined_field(vertices, triangles, vector_field, scalar_field, res=150):

    x, y = vertices[:, 0], vertices[:, 1]

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    tpc = ax.tripcolor(x, y, triangles, scalar_field, shading='gouraud', cmap='viridis', alpha=0.8)
    fig = plt.gcf()

    ax.triplot(x, y, triangles, color='white', lw=0.3, alpha=0.2)

    xi = np.linspace(x.min(), x.max(), res)
    yi = np.linspace(y.min(), y.max(), res)
    X, Y = np.meshgrid(xi, yi)
    
    interp_u = LinearNDInterpolator(vertices, vector_field[:, 0])
    interp_v = LinearNDInterpolator(vertices, vector_field[:, 1])
    U = interp_u(X, Y)
    V = interp_v(X, Y)

    ax.streamplot(X, Y, U, V, color='white', linewidth=0.8, density=4, arrowsize=1)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plot_mesh(mesh: TriMesh, ax=None, *,
              show_boundary: bool = True,
              show_interior: bool = False,
              title: str = "Malha Triangular",
              figsize=(7, 6)) -> plt.Axes:
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    triang = mtri.Triangulation(mesh.vertices[:, 0], mesh.vertices[:, 1],
                                 mesh.triangles)
    ax.triplot(triang, color="steelblue", linewidth=0.5, alpha=0.7)

    if show_boundary:
        bv = mesh.vertices[mesh.boundary_nodes]
        ax.scatter(bv[:, 0], bv[:, 1], s=5, c="firebrick",
                   zorder=5)

    if show_interior:
        iv = mesh.vertices[mesh.interior_nodes]
        ax.scatter(iv[:, 0], iv[:, 1], s=10, c="navy",
                   zorder=4, alpha=0.6)

    ax.set_aspect("equal")
    ax.set_title(title)
    #if show_boundary or show_interior:
    #    ax.legend(fontsize=8)
    return ax

def plot_patches(mesh: TriMesh, coverage: PatchCoverage, ax=None, *,
                 alpha: float = 0.15,
                 show_centers: bool = True,
                 highlight = None,
                 title: str = "Cobertura por Patches",
                 figsize=(7, 6)) -> plt.Axes:
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    plot_mesh(mesh, ax=ax, show_boundary=True, show_interior=False, title="")

    highlight_set = set(highlight) if highlight else set()

    circles = []
    colors  = []
    for j, patch in enumerate(coverage):
        c   = Circle(patch.center, patch.radius)
        circles.append(c)
        colors.append("orange" if j in highlight_set else "royalblue")

    coll = PatchCollection(circles, alpha=alpha, facecolors=colors,
                           edgecolors="navy", linewidths=0.6)
    ax.add_collection(coll)

    if show_centers:
        centers = np.array([p.center for p in coverage])
        ax.scatter(centers[:, 0], centers[:, 1], s=15, c="navy",
                   zorder=6, marker="+")

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.autoscale_view()
    return ax


def plot_patches_v2(mesh, coverage, ax=None, *,
                    alpha: float = 0.25, 
                    show_centers: bool = True,
                    highlight=None,
                    title: str = "Patches Coverage",
                    figsize=(7, 6),
                    center_size: float = 8,
                    linewidth: float = 0.6,):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)

    highlight_set = set(highlight) if highlight else set()

    circles = []
    colors = []

    for j, patch in enumerate(coverage):
        c = Circle(patch.center, patch.radius)
        circles.append(c)
        colors.append("orange" if j in highlight_set else "darkgray")

    coll = PatchCollection(circles, alpha=alpha, 
                           facecolors=colors, edgecolors="navy",
                           linewidths=linewidth)
    ax.add_collection(coll)

    plot_mesh(mesh, ax=ax, show_boundary=True, show_interior=False, title="")

    if show_centers:
        centers = np.array([p.center for p in coverage])
        ax.scatter(centers[:, 0], centers[:, 1], s=center_size,
                    c="navy", zorder=6, marker="o", linewidths=0.4)

    ax.set_aspect("equal")
    #ax.set_title(title, fontsize=16)

    #ax.tick_params(axis='both', labelsize=12)
    ax.axis("off")
    ax.autoscale_view()
    plt.tight_layout()

    return ax