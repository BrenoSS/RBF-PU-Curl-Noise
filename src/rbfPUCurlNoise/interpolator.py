import numpy as np
from scipy.spatial import distance_matrix as scipy_dist
from .triang_mesh    import TriMesh
from .patches import PatchCoverage
from .weights import PUWeights
from .rbf     import RBF
import time
import tracemalloc

class PUMInterpolator:

    def __init__(self, mesh: TriMesh, coverage: PatchCoverage, rbf: RBF):

        self.mesh     = mesh
        self.coverage = coverage
        self.rbf      = rbf
        self.pu_weights = None
        self._lambdas = []
        self._fitted  = False
        self.cond_num_list = []
        self.time_exec_list = []
        self.memory_usage_peak_list = []

    def fit(self, u_nodes: np.ndarray):

        u_nodes = np.asarray(u_nodes, dtype=float)
        assert len(u_nodes) == self.mesh.n_vertices, \
            "u_nodes deve ter um valor por vértice da malha."

        self.pu_weights = PUWeights(self.mesh, self.coverage)
        verts = self.mesh.vertices
        self._lambdas = []

        for patch in self.coverage:
            idx     = patch.node_indices
            n       = len(idx)

            if n == 0:
                self._lambdas.append(np.array([]))
                continue

            X_local = verts[idx]
            u_local = u_nodes[idx]

            R = scipy_dist(X_local, X_local)
            A = self.rbf.phi(R)
            self.cond_num_list.append(np.linalg.cond(A))

            t0 = time.perf_counter()
            tracemalloc.start()

            try:
                lam = np.linalg.solve(A, u_local)
            except np.linalg.LinAlgError:
                lam = np.linalg.lstsq(A, u_local, rcond=None)[0]
            
            t1 = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self.time_exec_list.append(t1 - t0)
            self.memory_usage_peak_list.append(peak)

            self._lambdas.append(lam)

        self._fitted = True
        return self

    def evaluate(self, points: np.ndarray):

        if not self._fitted:
            raise RuntimeError("Chame fit() antes de evaluate().")

        points = np.asarray(points, dtype=float)
        M      = len(points)

        W = self.pu_weights.evaluate_at(points)

        u_eval = np.zeros(M)
        verts  = self.mesh.vertices

        for j, (patch, lam) in enumerate(zip(self.coverage, self._lambdas)):
            if len(lam) == 0:
                continue

            in_patch = patch.contains(points)
            if not np.any(in_patch):
                continue

            X_eval_j = points[in_patch]
            w_j      = W[in_patch, j]
            X_local  = verts[patch.node_indices]

            R_eval   = scipy_dist(X_eval_j, X_local)
            Phi_eval = self.rbf.phi(R_eval)
            u_local  = Phi_eval @ lam

            u_eval[in_patch] += w_j * u_local

        return u_eval

    def gradient(self, points: np.ndarray):

        if not self._fitted:
            raise RuntimeError("Chame fit() antes de gradient().")

        points = np.asarray(points, dtype=float)
        M      = len(points)
        verts  = self.mesh.vertices

        W, Wx, Wy = self.pu_weights.evaluate_with_grad(points)

        grad_x = np.zeros(M)
        grad_y = np.zeros(M)

        for j, (patch, lam) in enumerate(zip(self.coverage, self._lambdas)):
            if len(lam) == 0:
                continue

            in_patch = patch.contains(points)
            if not np.any(in_patch):
                continue

            X_eval_j = points[in_patch]
            X_local  = verts[patch.node_indices]

            R_eval   = scipy_dist(X_eval_j, X_local)
            Phi_eval = self.rbf.phi(R_eval)
            u_local  = Phi_eval @ lam

            dx_mn = X_eval_j[:, 0:1] - X_local[:, 0]
            dy_mn = X_eval_j[:, 1:2] - X_local[:, 1]

            dPhix = self.rbf.dxphi(dx_mn, R_eval)
            dPhiy = self.rbf.dyphi(dy_mn, R_eval)

            du_local_x = dPhix @ lam
            du_local_y = dPhiy @ lam

            wx_j = Wx[in_patch, j]
            wy_j = Wy[in_patch, j]

            w_j  = W[in_patch, j]

            grad_x[in_patch] += wx_j * u_local + w_j * du_local_x
            grad_y[in_patch] += wy_j * u_local + w_j * du_local_y

        return grad_x, grad_y