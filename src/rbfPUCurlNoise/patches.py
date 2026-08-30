import numpy as np
from scipy.spatial import KDTree
from .triang_mesh import TriMesh

class Patch:

    def __init__(self, center, radius, node_indices):
        self.center = center
        self.radius= radius
        self.node_indices = node_indices

    def contains(self, points: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(points - self.center, axis=1)
        return dist <= self.radius

class PatchCoverage:

    def __init__(self, mesh: TriMesh, H: float, delta: float = 0.2,
                 min_nodes_per_patch: int = 3):
        self.mesh  = mesh
        self.H     = H
        self.delta = delta
        self.min_nodes_per_patch = min_nodes_per_patch
        self.patches = []
        self.radius_list = []
        self._build()


    def _build(self):
        mesh  = self.mesh
        H     = self.H
        delta = self.delta

        lo, hi = mesh.bounding_box
        xs = np.arange(lo[0] + H / 2, hi[0], H)
        ys = np.arange(lo[1] + H / 2, hi[1], H)
        XX, YY = np.meshgrid(xs, ys)
        candidates = np.column_stack([XX.ravel(), YY.ravel()])

        inside_mask = self._points_in_mesh(candidates)
        centers = candidates[inside_mask]

        rho0 = np.sqrt(2) * H / 2
        radius = (1 + delta) * rho0

        tree = mesh.kdtree
        patches = []
        
        idx_lens = []
        idx_list = []
        centers_list = []
        for c in centers:
            idx = np.array(tree.query_ball_point(c, radius), dtype=int)
            print(len(idx))
            idx_lens.append(len(idx))
            idx_list.append(idx)
            centers_list.append(c)

        mean_len = np.mean(idx_lens)

        for ind, idx in enumerate(idx_list):
            if len(idx) >= self.min_nodes_per_patch and len(idx) > (mean_len / 3.0):
                self.radius_list.append(radius)
                c = centers_list[ind]
                patches.append(Patch(center=c.copy(), radius=radius,
                                        node_indices=idx))

        patches = self._cover_boundary(patches, radius, tree)
        self.patches = patches

    def _points_in_mesh(self, points: np.ndarray):
        from scipy.spatial import Delaunay
        tri = Delaunay(self.mesh.vertices)
        return tri.find_simplex(points) >= 0

    def _cover_boundary(self, patches, base_radius: float,
                        tree: KDTree):
        if not patches:
            return patches

        centers = np.array([p.center for p in patches])
        ctree   = KDTree(centers)

        for bi in self.mesh.boundary_nodes:
            bpt = self.mesh.vertices[bi]
            covered = any(bi in p.node_indices for p in patches)
            if not covered:
                # Closest patch
                _, pid = ctree.query(bpt)
                p = patches[pid]
                new_r = np.linalg.norm(bpt - p.center) * 1.05
                p.radius = max(p.radius, new_r)
                # Patch nodes update
                p.node_indices = np.array(
                    tree.query_ball_point(p.center, p.radius), dtype=int)

        return patches

    def __len__(self):
        return len(self.patches)

    def __iter__(self):
        return iter(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

    def local_indices(self, patch_idx):
        return self.patches[patch_idx].node_indices

    def patches_of_node(self, node_idx: int):
        return [i for i, p in enumerate(self.patches)
                if node_idx in p.node_indices]

    def coverage_count(self) -> np.ndarray:
        counts = np.zeros(self.mesh.n_vertices, dtype=int)
        for p in self.patches:
            counts[p.node_indices] += 1
        return counts

    def verify_coverage(self) -> bool:
        counts = self.coverage_count()
        uncovered = np.where(counts == 0)[0]
        if len(uncovered) > 0:
            print(f"[PatchCoverage] Warning: {len(uncovered)} vertex without coverage: "
                  f"{uncovered[:10]}...")
            return False
        return True

    def summary(self):
        counts = self.coverage_count()
        sizes  = [len(p.node_indices) for p in self.patches]
        s =  {
            "n_patches"       : len(self.patches),
            "H"               : self.H,
            "delta"           : self.delta,
            "min_nodes"       : int(min(sizes)) if sizes else 0,
            "max_nodes"       : int(max(sizes)) if sizes else 0,
            "mean_nodes"      : float(np.mean(sizes)) if sizes else 0,
            "min_coverage"    : int(counts.min()),
            "max_coverage"    : int(counts.max()),
            "mean_coverage"   : float(counts.mean()),
            "full_coverage"   : bool(counts.min() >= 1),
        }

        print("\nCoverage Summary:")
        for k, v in s.items():
            print(f"  {k:20s}: {v}")