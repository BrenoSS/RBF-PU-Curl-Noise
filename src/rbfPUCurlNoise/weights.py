import numpy as np
from .triang_mesh import TriMesh
from .patches import PatchCoverage

def _wendland_c2(r: np.ndarray, rho: float):
    t = r / rho
    t = np.clip(t, 0, None)
    pos = np.maximum(1.0 - t, 0.0)
    return (4.0 * t + 1.0) * pos ** 4

def _wendland_c2_deriv(dx: np.ndarray, dy: np.ndarray, r: np.ndarray, rho: float):
    t = r / rho
    pos = np.maximum(1.0 - t, 0.0)
    coef = -20.0 / rho ** 2 * pos ** 3
    phix = coef * dx
    phiy = coef * dy
    return phix, phiy

class PatchWeights:
    def __init__(self, w, wx, wy):
        self.w = w
        self.wx = wx
        self.wy = wy

class PUWeights:
    def __init__(self, mesh: TriMesh, coverage: PatchCoverage):
        self.mesh     = mesh
        self.coverage = coverage
        self._weights = []
        self._compute()

    def _compute(self):
        mesh     = self.mesh
        coverage = self.coverage
        N        = mesh.n_vertices
        Np       = len(coverage)

        phi_list   = [None] * Np
        phix_list  = [None] * Np
        phiy_list  = [None] * Np

        S    = np.zeros(N)
        Sx   = np.zeros(N)
        Sy   = np.zeros(N)

        for j, patch in enumerate(coverage):
            idx  = patch.node_indices
            pts  = mesh.vertices[idx]
            dx   = pts[:, 0] - patch.center[0]
            dy   = pts[:, 1] - patch.center[1]
            r    = np.sqrt(dx**2 + dy**2)
            rho  = patch.radius

            phi    = _wendland_c2(r, rho)
            px, py = _wendland_c2_deriv(dx, dy, r, rho)

            phi_list[j]   = phi
            phix_list[j]  = px
            phiy_list[j]  = py

            S[idx]   += phi
            Sx[idx]  += px
            Sy[idx]  += py

        self._weights = []
        for j, patch in enumerate(coverage):
            idx = patch.node_indices

            phi  = phi_list[j]
            px   = phix_list[j]
            py   = phiy_list[j]

            s    = S[idx]
            sx  = Sx[idx]
            sy  = Sy[idx]
            s2 = s ** 2

            w   = phi / s
            wx  = px / s - phi * sx / s2
            wy  = py / s - phi * sy / s2

            self._weights.append(PatchWeights(w=w, wx=wx, wy=wy))

    def __getitem__(self, patch_idx: int) -> PatchWeights:
        return self._weights[patch_idx]

    def __len__(self) -> int:
        return len(self._weights)

    def evaluate_at(self, points: np.ndarray):

        Np = len(self.coverage)
        n  = len(points)
        phi_all = np.zeros((n, Np))

        for j, patch in enumerate(self.coverage):
            dx  = points[:, 0] - patch.center[0]
            dy  = points[:, 1] - patch.center[1]
            r   = np.sqrt(dx**2 + dy**2)
            phi_all[:, j] = _wendland_c2(r, patch.radius)

        S = phi_all.sum(axis=1, keepdims=True)
        S = np.where(S > 0, S, 1.0)
        return phi_all / S

    def evaluate_with_grad(self, points: np.ndarray):

        Np = len(self.coverage)
        M  = len(points)

        phi_all  = np.zeros((M, Np))
        phix_all = np.zeros((M, Np))
        phiy_all = np.zeros((M, Np))

        for j, patch in enumerate(self.coverage):
            dx = points[:, 0] - patch.center[0]
            dy = points[:, 1] - patch.center[1]
            r  = np.sqrt(dx**2 + dy**2)
            rho = patch.radius

            phi_all[:, j]  = _wendland_c2(r, rho)
            px, py         = _wendland_c2_deriv(dx, dy, r, rho)
            phix_all[:, j] = px
            phiy_all[:, j] = py

        S  = phi_all.sum(axis=1)
        Sx = phix_all.sum(axis=1)
        Sy = phiy_all.sum(axis=1)

        safe_S  = np.where(S  > 0, S,  1.0)
        safe_S2 = safe_S ** 2

        W  = phi_all  / safe_S[:, None]
        Wx = (phix_all * safe_S[:, None] - phi_all * Sx[:, None]) / safe_S2[:, None]
        Wy = (phiy_all * safe_S[:, None] - phi_all * Sy[:, None]) / safe_S2[:, None]

        return W, Wx, Wy