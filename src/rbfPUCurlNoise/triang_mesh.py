import numpy as np
import networkx as nx
import itertools
import meshio
import copy
from scipy.spatial import KDTree

class TriMesh:

    mesh = dict
    vertices: np.ndarray
    triangles: np.ndarray
    boundary_nodes: np.ndarray
    interior_nodes: np.ndarray
    closest_pts: np.ndarray
    phi: np.ndarray
    _kdtree: KDTree

    def __init__(self, mesh):
        self.mesh = mesh
        self.vertices  = self.mesh["vertices"]
        self.triangles = self.mesh["triangles"]
        self.boundary_nodes: np.ndarray
        self.interior_nodes: np.ndarray
        self._classify_nodes()
        self._kdtree = KDTree(self.vertices)
        self.sdf = None
        self.bdry_edges = None
        self.closest_pts = None

    def _classify_nodes(self):
        edges = {}
        for tri in self.triangles:
            for i in range(3):
                e = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
                edges[e] = edges.get(e, 0) + 1
        boundary_set = set()
        for e, count in edges.items():
            if count == 1:
                boundary_set.update(e)
        all_idx = np.arange(len(self.vertices))
        self.boundary_nodes = np.array(sorted(boundary_set), dtype=int)
        self.interior_nodes = np.setdiff1d(all_idx, self.boundary_nodes)

    def nodes_in_ball(self, center: np.ndarray, radius: float) -> np.ndarray:
        return np.array(self.kdtree.query_ball_point(center, radius), dtype=int)

    def nodes_in_ball_batch(self, centers: np.ndarray, radius: float):
        return self.kdtree.query_ball_point(centers, radius)

    @property
    def kdtree(self):
        return self._kdtree

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    @property
    def bounding_box(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def get_mesh_limits(self):
        xmin = self.vertices[:, 0].min()
        xmax = self.vertices[:, 0].max()
        ymin = self.vertices[:, 1].min()
        ymax = self.vertices[:, 1].max()

        return xmin, xmax, ymin, ymax

    def get_sdf_field(self):
        if self.sdf is not None:
            return self.sdf
        else:
            print("SDF Field not initialized yet.")

    def get_closest_pts(self):
        if self.closest_pts is not None:
            return self.closest_pts
        else:
            print("Closest points not initialized yet.")

    def get_boundary_edges(self):
        triangles = self.triangles

        edges = {}
        
        for tri in triangles:
            for edge in itertools.combinations(tri, 2):
                if tuple(sorted(edge)) not in edges:
                    edges[tuple(sorted(edge))] = 1
                else:
                    edges[tuple(sorted(edge))] +=1

        boundary_edges = [e for e,c in edges.items() if c == 1]

        return boundary_edges
    
    def get_boundary_edges_v2(self, mesh):
        triangles = mesh["triangles"]

        edges = {}
        
        for tri in triangles:
            for edge in itertools.combinations(tri, 2):
                if tuple(sorted(edge)) not in edges:
                    edges[tuple(sorted(edge))] = 1
                else:
                    edges[tuple(sorted(edge))] +=1

        boundary_edges = [e for e,c in edges.items() if c == 1]

        return boundary_edges

    def _distance_point_to_segment(self, p, a, b):
        ap = p - a # vector from a to point (a is the end of the segment line)
        ab = b - a # vector from a to b (vector in the segment line)
        # projection of ap into ab, dot product divided by the norm
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        # locating the closest point in the segment (parametric equation of the segment)
        closest = a + t * ab
        # return the Euclidean distance
        return np.linalg.norm(p - closest), closest

    def sdf_distance_to_segments_min(self):
        vertices = self.vertices
        nverts = vertices.shape[0]

        if self.bdry_edges is None:
            self.bdry_edges = self.get_boundary_edges()
        segments = self.bdry_edges

        if self.sdf is None:
            self.sdf = np.full(nverts, np.inf)
        
        if self.closest_pts is None:
            self.closest_pts = np.zeros((nverts, 2))

        if self.interior_nodes is None:
            self._classify_nodes()

        for vertex in self.interior_nodes:
            p = vertices[vertex]
            for a, b in segments:
                a_p = vertices[a]
                b_p = vertices[b]
                d, closest = self._distance_point_to_segment(p, a_p, b_p)
                # This is the point where the hard min distance is computed
                # following Bridson's approach
                if d < self.sdf[vertex]:
                    self.sdf[vertex] = d
                    self.closest_pts[vertex] = closest

        for vertex in self.boundary_nodes:
            self.sdf[vertex] = 0.0
            self.closest_pts[vertex] = vertices[vertex]

    def generate_mesh_noise_scalar_field(self, noise, freq):
        f_values = []

        for vertex in self.mesh["vertices"]:
            noise_val = noise([vertex[0] * freq, vertex[1] * freq])
            f_values.append(noise_val)

        f_values = np.array(f_values)
        return f_values
    
    def generate_noise_scalar_field_from_vertices(self, vertices, noise, freq):
        f_values = []

        for vertex in vertices:
            noise_val = noise([vertex[0] * freq, vertex[1] * freq])
            f_values.append(noise_val)

        f_values = np.array(f_values)
        return f_values
    
    def save_mesh_paraview(self, mesh_to_save, vels=None, phi=None, psi=None,  sdf=None, out_name="lic.vtu", save_norm=False):
        x = mesh_to_save["vertices"][:,0]
        y = mesh_to_save["vertices"][:,1]
        points = np.column_stack([x, y, np.zeros_like(x)])

        cells = [("triangle", mesh_to_save["triangles"])]

        # Disctionary to save the data
        point_data = {}

        if vels is not None:
            assert vels.shape[0] == mesh_to_save["vertices"].shape[0]
            # velocity per vertex
            u = vels[:,0]
            v = vels[:,1]
            velocity = np.column_stack([u, v, np.zeros_like(u)])
            point_data["velocity"] = velocity

        if phi is not None:
            point_data["phi"] = phi
        
        if psi is not None:
            point_data["psi"] = psi

        if sdf is not None:
            point_data["sdf"] = sdf
        
        if save_norm:
            M = np.max(np.abs(phi))
            if M == 0:
                M = 1.0
            original_norm = phi / M
            masked_norm = psi / M

            point_data["phi_norm"] = original_norm
            point_data["psi_norm"] = masked_norm

            print("Normalization factor:", M)

            print("Original normalized range:")
            print(original_norm.min(), original_norm.max())

            print("Masked normalized range:")
            print(masked_norm.min(), masked_norm.max())


        out_mesh = meshio.Mesh(points=points,
                               cells=cells,
                               point_data=point_data)

        meshio.write(out_name, out_mesh)

    def create_adjacency_dictionary(self):
        num_vertices = len(self.mesh["vertices"])
        adj = {v: set() for v in range(num_vertices)}

        for tri in self.mesh["triangles"]:
            for i in range(3):
                v = tri[i]
                u = tri[(i+1) % 3]
                w = tri[(i+2) % 3]
                
                adj[v].add(u)
                adj[v].add(w)
        return adj


    def refine_bdry_pts(self):
        if self.bdry_edges is None:
            self.bdry_edges = self.get_boundary_edges()
        new_points = []
        for i, j in self.bdry_edges:
            midpoint = 0.5 * (self.mesh["vertices"][i] + \
                              self.mesh["vertices"][j])
            new_points.append(midpoint)
        
        new_points = np.array(new_points)
        return new_points

    def get_1_ring_neighbors(self, vertex_set, vertex_to_triangles):
        neighbors = set()
        
        for v in vertex_set:
            for t_id in vertex_to_triangles[v]:
                tri = self.mesh["triangles"][t_id]
                
                for u in tri:
                    neighbors.add(u)
        
        return neighbors

    def vertex_to_triangle_map(self):
        num_vertices = len(self.mesh["vertices"])
        vertex_to_triangles = {v: [] for v in range(num_vertices)}
        
        for t_id, tri in enumerate(self.mesh["triangles"]):
            for v in tri:
                vertex_to_triangles[v].append(t_id)
        return vertex_to_triangles

    def get_depth_neighbors(self, boundary_vertices, vertex_to_triangles):
        layers = []
        visited = set(boundary_vertices)
        
        current = set(boundary_vertices)
        layers.append(current)
        
        while True:
            next_layer = self.get_1_ring_neighbors(current, vertex_to_triangles)
            
            # remove already visited
            next_layer = next_layer - visited
            
            if not next_layer:
                break
            
            layers.append(next_layer)
            visited.update(next_layer)
            current = next_layer

        return layers
    
    def get_vertex_layers(self):

        if self.bdry_edges is None:
            self.bdry_edges = self.get_boundary_edges()

        num_vertices = len(self.mesh["vertices"])
        boundary_vertices = set(np.array(self.bdry_edges).flatten())
        vertex_to_triangles = self.vertex_to_triangle_map()
        layers = self.get_depth_neighbors(boundary_vertices, vertex_to_triangles)

        layer_id = np.ones(num_vertices, dtype=int)
        for i, layer in enumerate(layers):
            for v in layer:
                layer_id[v] = i
        
        return layer_id

def refine_mesh_1to4(vertices, triangles):

    vertices = list(vertices)
    new_triangles = []
    midpoint_cache = {}

    def get_midpoint_idx(v1_idx, v2_idx):
        edge = tuple(sorted((v1_idx, v2_idx)))
        if edge not in midpoint_cache:
            p1, p2 = np.array(vertices[edge[0]]), np.array(vertices[edge[1]])
            midpoint = (p1 + p2) / 2.0
            
            new_idx = len(vertices)
            vertices.append(midpoint.tolist())
            midpoint_cache[edge] = new_idx
            return new_idx
        return midpoint_cache[edge]

    for tri in triangles:
        v0, v1, v2 = tri
        
        m01 = get_midpoint_idx(v0, v1)
        m12 = get_midpoint_idx(v1, v2)
        m20 = get_midpoint_idx(v2, v0)

        new_triangles.append([v0, m01, m20])
        new_triangles.append([v1, m12, m01])
        new_triangles.append([v2, m20, m12])
        new_triangles.append([m01, m12, m20])
        
    return np.array(vertices), np.array(new_triangles)