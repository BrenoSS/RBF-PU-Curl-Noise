import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import copy
from scipy.spatial import cKDTree

kdtree = None

def save_noise_mesh_fig(mesh, values):
    
    triang = tri.Triangulation(mesh["vertices"][:,0],
                               mesh["vertices"][:,1],
                               mesh["triangles"])

    plt.figure()
    plt.tripcolor(triang, values, shading='gouraud')
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.savefig("noise_mesh.png", dpi=300, bbox_inches='tight')


def noise_texture(mesh, noise, freq):
    values = np.zeros(len(mesh["vertices"]))
    for i, (x, y) in enumerate(mesh["vertices"]):
        #print(i, x, y)
        values[i] = noise([freq * x, freq * y])
    return values


def plot_mesh(mesh):
    V = mesh["vertices"]
    T = mesh["triangles"]
    
    fig, ax = plt.subplots(dpi=500)
    ax.triplot(V[:, 0], V[:, 1], T, linewidth=0.2)
    plt.gca().set_aspect("equal")
    #plt.title("Unstructured Triangle Mesh")
    plt.show()


def plot_neighbors_triangle(mesh, vid, neighbors):
    V = mesh["vertices"]
    T = mesh["triangles"]

    plt.figure(figsize=(6,6))

    # Plot triangles
    for tri in T:
        pts = V[tri]
        pts = np.vstack([pts, pts[0]])
        plt.plot(pts[:,0], pts[:,1], color="lightgray", linewidth=0.5)

    # Plot all vertices
    plt.scatter(V[:,0], V[:,1], s=5, color="black", alpha=0.4)

    # Plot neighbors
    plt.scatter(V[neighbors,0], V[neighbors,1], s=50, color="dodgerblue", label="neighbors")

    # Plot center
    plt.scatter(V[vid,0], V[vid,1], s=80, color="red", label="center")

    plt.axis("equal")
    plt.legend()
    plt.show()

def plot_sdf_mesh_map(mesh, phi):
    V = mesh["vertices"]
    F = mesh["triangles"]
    # V: (N,2), F: (M,3), phi: (N,)
    tringulation = tri.Triangulation(V[:,0], V[:,1], F)
    
    plt.figure(figsize=(6,6))
    plt.tripcolor(tringulation, phi, shading='gouraud')
    plt.colorbar(label="φ (signed distance)")
    plt.triplot(tringulation, lw=0.3, color='k', alpha=0.3)  # optional wireframe
    plt.gca().set_aspect("equal")
    plt.title("Signed Distance Field on Triangular Mesh")
    plt.show()

def plot_boundary_contour(mesh, delaunay, boundary):
    plt.figure(figsize=(7, 5))
    plt.scatter(mesh["vertices"][:,0], mesh["vertices"][:,1], s=2, color="black", alpha=0.3)
    plt.triplot(mesh["vertices"][:,0], mesh["vertices"][:,1], delaunay.simplices, lw=0.3, color="gray", alpha=0.4)
    
    for (u, v) in boundary:
        p, q = mesh["vertices"][u], mesh["vertices"][v]
        plt.plot([p[0], q[0]], [p[1], q[1]],
                 color="red", linewidth=2.0)

    plt.show()

def plot_noise_mesh(mesh, values):
    
    triang = tri.Triangulation(mesh["vertices"][:,0],
                               mesh["vertices"][:,1],
                               mesh["triangles"])

    plt.figure()
    plt.tripcolor(triang, values, shading='gouraud')
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.show()

def plot_layers_mesh(mesh, layer_id):
    # plot mesh wireframe
    plt.triplot(mesh["vertices"][:, 0],
                mesh["vertices"][:, 1],
                mesh["triangles"],
                color='lightgray', linewidth=0.5)

    # scatter vertices colored by layer
    sc = plt.scatter(
        mesh["vertices"][:, 0],
        mesh["vertices"][:, 1],
        c=layer_id,
        cmap='viridis',
        s=20
    )

    plt.colorbar(sc, label="Layer (topological distance)")
    plt.gca().set_aspect('equal')

    plt.title("Mesh Layers from Boundary")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()

def parse_poly_file(path):
    pts = []
    segs = []
    holes = []

    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    # First line: number of vertices
    n_verts = int(lines[0].split()[0])
    idx = 1
    for i in range(n_verts):
        parts = lines[idx].split()
        # skip number, take coordinates
        x = float(parts[1])
        y = float(parts[2])
        pts.append([x, y])
        idx += 1

    # Segment count
    n_segs = int(lines[idx].split()[0])
    idx += 1
    for i in range(n_segs):
        parts = lines[idx].split()
        s0 = int(parts[1]) - 1  # convert to 0-based
        s1 = int(parts[2]) - 1
        segs.append([s0, s1])
        idx += 1

    # Hole count
    if idx < len(lines):
        n_holes = int(lines[idx].split()[0])
        idx += 1
        for i in range(n_holes):
            parts = lines[idx].split()
            holes.append([float(parts[1]), float(parts[2])])
            idx += 1

    dict_mesh = {"vertices": np.array(pts),
                "segments": np.array(segs),}

    if holes:
        dict_mesh["holes"] = np.array(holes)

    return dict_mesh


### Mesh refinement utilities (TODO: Create a class to group these methods)
# It doesn't make sense to define it in TriangMesh class because when we
# refine, topology changes, and attributes of the class, such as boundary
# points, edges, graph, will all be outdated and if not updated properly
# can become very difficult to track and maintain.
### Refinement around the edge methods ###

def define_edges_to_refine(bdry_edges, mesh):
    refine_edges = set(bdry_edges)

    for (a, b) in bdry_edges:
        for tri in mesh["triangles"]:
            if a in tri or b in tri:
                for i in range(3):
                    e = tuple(sorted((tri[i], tri[(i+1)%3])))
                    refine_edges.add(e)
    return refine_edges
    
"""
def get_midpoint(i, j, new_vertices, mesh):
    key = tuple(sorted((i, j)))
    
    vi = mesh["vertices"][i]
    vj = mesh["vertices"][j]
    vm = 0.5 * (vi + vj)

    idx = len(new_vertices)
    new_vertices.append(vm.tolist())
    return idx
"""

def get_midpoint(i, j, new_vertices, mesh, edge_mid):
    global kdtree
    eps = 1e-8
    key = tuple(sorted((i, j)))

    # 1. reuse edge midpoint
    if key in edge_mid:
        return edge_mid[key]

    # 2. compute midpoint
    vi = mesh["vertices"][i]
    vj = mesh["vertices"][j]
    vm = 0.5 * (vi + vj)

    # 3. query nearest neighbor
    if len(new_vertices) > 0:
        dist, idx = kdtree.query(vm)

        if dist < eps:
            edge_mid[key] = idx
            return idx

    # 4. add new vertex
    idx = len(new_vertices)
    new_vertices.append(vm)

    # 5. update KD-tree (simple version: rebuild)
    vertices_array = np.array(new_vertices)
    kdtree = cKDTree(vertices_array)

    edge_mid[key] = idx

    return idx
    
def rotate(v, f):
    # v = [v0, v1, v2]
    # f = [f0, f1, f2]
    v0, v1, v2 = v
    f0, f1, f2 = f
    v_rot = [v1, v2, v0]
    f_rot = [f1, f2, f0]

    return v_rot, f_rot


def normalize(v, f, n):
    if n == 1:
        while f[0] != True:
            v, f = rotate(v, f)
    elif n == 2:
        while f[2] != False:
            v, f = rotate(v, f)

    return v, f
    
def get_new_simplexes(mesh, refine_edges):
    new_triangles = []
    new_vertices = copy.deepcopy(mesh["vertices"].tolist())
    edge_mid = {}
    for (i, j, k) in mesh["triangles"]:
        e_ij = tuple(sorted((i, j)))
        e_jk = tuple(sorted((j, k)))
        e_ki = tuple(sorted((k, i)))

        split_ij = 1 if e_ij in refine_edges else 0
        split_jk = 1 if e_jk in refine_edges else 0
        split_ki = 1 if e_ki in refine_edges else 0

        f = [split_ij, split_jk, split_ki]
        v = [i, j, k]

        # Number of edges in each triangle marked to split.
        # There are four possible split cases:
        # 0: No edges to split.
        # 1: one edge to split. Considering edge i, j, we create a
        # new vertex between i and j, and two triangles are created: (i, m, k) and (m, j, k)
        # 2: two edges to split, two new midpoints. Considering a triangle i, j, k and midpoints
        # m_ij and m_jk, the new triangles will be (i, m_ij, k), (m_ij, m_jk, k), (m_ij, j, m_jk).
        # 3: three edges to split. In this case, it will be a full triangle refinement. The new
        # midpoints will be m_ij, m_ik, m_jk. The new triangles, (i, m_ij, m_ki), (m_ij, j, m_jk)
        # (m_ki, m_jk, k), (m_ij, m_jk, m_ki).
        n = split_ij + split_jk + split_ki

        # This step is here to create a canonical representation of the triangle. This way, we don't
        # need to worry about creating several tests to determine which edge will be split, we always
        # split the first one, in the case of n ==1, the first two ordered edges, in the case of n ==2.
        # The case n == 3, is straightforward, we split all edges. When n ==0, we don't split any
        # of them.
        v, f = normalize(v, f, n)
        # rename after normalization
        v0, v1, v2 = v
        
        if n == 1:
            m01 = get_midpoint(v0, v1, new_vertices, mesh, edge_mid)
            new_triangles.append([v0, m01, v2])
            new_triangles.append([m01, v1, v2])
        elif n == 2:
            m01 = get_midpoint(v0, v1, new_vertices, mesh, edge_mid)
            m12 = get_midpoint(v1, v2, new_vertices, mesh, edge_mid)
            new_triangles.append([v0, m01, v2])
            new_triangles.append([m01, m12, v2])
            new_triangles.append([m01, v1, m12])
        elif n == 3:
            m01 = get_midpoint(v0, v1, new_vertices, mesh, edge_mid)
            m12 = get_midpoint(v1, v2, new_vertices, mesh, edge_mid)
            m20 = get_midpoint(v2, v0, new_vertices, mesh, edge_mid)
            new_triangles.append([v0, m01, m20])
            new_triangles.append([m01, v1, m12])
            new_triangles.append([m20, m12, v2])
            new_triangles.append([m01, m12, m20])
        
    #F_new = mesh["triangles"].tolist() + new_triangles
    #F_new = np.array(F_new)

    #new_vertices = np.array(new_vertices)
    return new_triangles, new_vertices
    
def refine_around_brder(mesh, bdry_edges):
    global kdtree
    kdtree = cKDTree(mesh["vertices"])
    edges_to_refine = define_edges_to_refine(bdry_edges, mesh)
    F_new, new_vertices = get_new_simplexes(mesh, edges_to_refine)
    new_triangles = mesh["triangles"].tolist() + F_new

    mesh_dict = {}
    mesh_dict["vertices"] = np.array(new_vertices)
    mesh_dict["triangles"] = np.array(new_triangles)

    return mesh_dict