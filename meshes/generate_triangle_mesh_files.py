import numpy as np
import matplotlib.pyplot as plt
import triangle as tr
import alphashape
from shapely.geometry import Polygon
import itertools
from collections import defaultdict
from matplotlib.path import Path
from scipy.spatial import Delaunay


def write_vtk(filename, vertices, triangles):
    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Triangle mesh\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {len(vertices)} float\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} 0.0\n")

        # Triangles
        f.write(f"POLYGONS {len(triangles)} {len(triangles)*4}\n")
        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

def save_npz(A, B, filename="mesh.npz"):
    np.savez(filename,
             vertices=B["vertices"],
             triangles=B["triangles"],
             segments=A["segments"],
             holes=A.get("holes"))


def write_node(filename, vertices):
    with open(filename, "w") as f:

        # header
        f.write(f"{len(vertices)} 2 0 0\n")

        # index x y
        for i, v in enumerate(vertices):
            f.write(f"{i} {v[0]} {v[1]}\n")


def write_poly(filename, vertices, segments, hole_loops=None):

    with open(filename, "w") as f:
        # Part 1: vertices
        # 0 means vertices are stored in .node file
        f.write("0 2 0 0\n")

        # Part 2: segments
        f.write(f"{len(segments)} 0\n")

        for i, (a, b) in enumerate(segments):
            f.write(f"{i} {a} {b}\n")

        # Part 3: holes
        if hole_loops is None or len(hole_loops) == 0:
            f.write("0\n")
        else:
            f.write(f"{len(hole_loops)}\n")
            for i, loop in enumerate(hole_loops):
                # Compute one point inside hole
                pts = vertices[loop]
                hole_point = np.mean(pts, axis=0)

                f.write(f"{i} "
                    	f"{hole_point[0]} "
                    	f"{hole_point[1]}\n")


def sig(u, v, R, S):
    p_u = S[u]
    p_v = S[v]
    dist_uv = np.linalg.norm(p_u - p_v)
    return dist_uv <= R[u] + R[v]


def nn_radii(tri, S, mu=1.0, eps=1e-12):
    """
    Nearest-neighbor radii computed from Delaunay adjacency only.
    """
    N = len(S)
    neighbors = [set() for _ in range(N)]

    # Determining the neighbors of each point in the dataset through the
    # given Delaunay triangulation connections. 
    # neighbors is a list of set() objects to ensure no repetition of index
    # points
    for simplex in tri.simplices:
        a, b, c = simplex
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])

    R = np.zeros(N)

    # Computing the distances between each point and its neighbors.
    # After that, we determine the minimum distance  as our smallest
    # neighbor disk. The mu parameter is used to scale the final
    # radius value.
    for i in range(N):
        if len(neighbors[i]) == 0:
            R[i] = eps
        else:
            d = [np.linalg.norm(S[ngbor] - S[i]) for ngbor in neighbors[i]]
            R[i] = mu * np.min(d)

    return R


def sid_edges(edges, R, S):
    # Deletes edges from the dictionary of edges if they
    # fail in the Radius test. Initialize a tag value
    # with 0 for the edges that pass the test.
    for (u, v) in list(edges.keys()):
        if sig(u, v, R, S):
            edges[tuple(sorted((u, v)))] = 0
        else:
            del edges[tuple(sorted((u, v)))]

def sid_faces(faces, edges, R, S):
    # Creates a list of the faces (triangles) kept in the Delaunay
    # filtering. Increments the tag values if all triangle edges 
    # pass the test.
    faces_kept = []
    for (u, v, w) in faces:
        if(sig(u, v, R, S) and sig(v, w, R, S) and sig(w, u, R, S)):
            if(tuple(sorted((u, v))) in edges):
                edges[tuple(sorted((u, v)))]+=1
            if(tuple(sorted((v, w))) in edges):
                edges[tuple(sorted((v, w)))]+=1
            if(tuple(sorted((w, u))) in edges):
                edges[tuple(sorted((w, u)))]+=1
            faces_kept.append((u, v, w))
    return faces_kept
            

def sid(S, mu):
    # Procedure sphere-of-influence diagram. We start by computing
    # the Delaunay triangulation. The faces and edges are determined.
    # The scaled nearest neighbor radius is computed for each point
    # in the dataset. Then we proceed with the edges test and tagging
    # and the filtering through the faces.
    delaunay = Delaunay(S)

    faces = [tuple(f) for f in delaunay.simplices]

    edges = {}

    for simplice in delaunay.simplices:
        for edge in itertools.combinations(simplice, 2):
            if tuple(sorted(edge)) not in edges:
                edges[tuple(sorted(edge))] = None
    R = nn_radii(delaunay, S, mu=mu, eps=1e-12)
    sid_edges(edges, R, S)
    faces_kept = sid_faces(faces, edges, R, S)

    return delaunay, edges, faces_kept


def boundary_points(S, mu):
    # After the tagging process is finished, the boundary edges
    # are determined by selecting the edges with value 1 tag. 
    delaunay, edges, faces = sid(S, mu)
    boundary = [e for e, tag in edges.items() if tag == 1]
    return boundary, delaunay


def build_loops(edges):
    # Build adjacency
    adj = defaultdict(list)

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # Track visited edges
    visited_edges = set()

    loops = []

    # Traverse all edges
    for a, b in edges:
        edge = tuple(sorted((a, b)))
        if edge in visited_edges:
            continue
        loop = [a]
        prev = a
        current = b
        visited_edges.add(edge)

        while True:
            loop.append(current)
            neighbors = adj[current]

            # Find next unused edge
            next_node = None

            for n in neighbors:
                if n == prev:
                    continue

                e = tuple(sorted((current, n)))

                if e not in visited_edges:
                    next_node = n
                    visited_edges.add(e)
                    break

            # Closed loop
            if next_node is None:
                if loop[0] == current:
                    break
                # Try to close explicitly
                if loop[0] in neighbors:
                    visited_edges.add(tuple(sorted((current, loop[0]))))
                    loop.append(loop[0])
                break

            prev, current = current, next_node
        loops.append(loop)

    return loops

def signed_area(points, loop):
    pts = points[loop]
    x, y = pts[:,0], pts[:,1]
    return 0.5*np.sum(x*np.roll(y,-1) - y*np.roll(x,-1))


def contains(loop_outer, loop_inner, points):
    poly = Path(points[loop_outer])
    pt = np.mean(points[loop_inner], axis=0)
    return poly.contains_point(pt)


def main():
	meshes = ["fishdp.txt", "alienbs.txt", "cartdp.txt", "dog.txt", "deer20dp.txt", \
		   "dovedp.txt", "duckdp.txt", "knightdp.txt", "dogbs.txt", "crowndp.txt"]

	for mesh in meshes:
		points = np.loadtxt("C:\\Users\\silve\\Desktop\\phd\\curl_noise_analysis\\meshes\\Sample_Data\\" + mesh)
		points[:,1] *= -1

		print(points.shape)

		boundary, delaunay = boundary_points(points, 1.3)

		loops = build_loops(boundary)

		segments = np.array(boundary)
		holes = []

		for loop in loops:
		    if signed_area(points, loop) < 0:
		        hole_point = np.mean(points[loop], axis=0)
		        holes.append(hole_point)

		holes = np.array(holes) if holes else None

		for i, loop in enumerate(loops):
		    print(i, loop[0], loop[-1])

		for i, loop in enumerate(loops):
		    area = signed_area(points, loop)
		    print(f"Loop {i}: area = {area}")

		centroids = [np.mean(points[loop], axis=0) for loop in loops]

		holes = []
		outers = []

		for i, loop_i in enumerate(loops):
		    inside_any = False

		    for j, loop_j in enumerate(loops):
		        if i == j:
		            continue

		        if contains(loop_j, loop_i, points):
		            inside_any = True
		            break

		    if inside_any:
		        holes.append(loop_i)
		    else:
		        outers.append(loop_i)

		segments = np.array(boundary)

		hole_points = []

		for loop in holes:
		    hole_points.append(np.mean(points[loop], axis=0))

		hole_points = np.array(hole_points) if hole_points else None

		A = {
		    "vertices": points,
		    "segments": segments
		}

		if hole_points is not None:
		    A["holes"] = hole_points

		B = tr.triangulate(A, 'pa100.0')
		v = B["vertices"]
		t = B["triangles"]

		#write_vtk("alienbs.vtk", v, t)
		#save_npz(A, B, filename="alienbs.npz")
		basename = mesh.split(".")[0]
		write_node(basename + ".node", v)
		write_poly(basename+ ".poly", v, segments, holes)



main()