import numpy as np
from scipy.spatial import distance_matrix

class RBF:

    def __init__(self, epsilon: float = 1.0):
        self.ep  = epsilon
        self.phi = lambda r: np.maximum(1-(r /self.ep), 0)**4 * (4*(r / self.ep) + 1)
        self.dxphi = lambda dx, r:  self.phix(dx, r)
        self.dyphi = lambda dy, r:  self.phiy(dy, r)
    
    def phix(self, dx: np.ndarray, r: np.ndarray):
        t = r / self.ep
        pos = np.maximum(1.0 - t, 0.0)
        coef = -20.0 / self.ep ** 2 * pos ** 3
        phix = coef * dx
        return phix
    
    def phiy(self, dy: np.ndarray, r: np.ndarray):
        t = r / self.ep
        pos = np.maximum(1.0 - t, 0.0)
        coef = -20.0 / self.ep ** 2 * pos ** 3
        phiy = coef * dy
        return phiy

    def interpolation_matrix(self, X: np.ndarray) -> np.ndarray:
        IDM = distance_matrix(X, X)
        return self.phi(IDM)
    # testar condition number self.phi(IDM)

    def eval_matrix(self, X_eval: np.ndarray, X_nodes: np.ndarray) -> np.ndarray:
        EDM = distance_matrix(X_eval, X_nodes)
        return self.phi(EDM)

def local_eval_matrix(rbf: RBF, X_eval: np.ndarray, X_local: np.ndarray,
                      A: np.ndarray, w_eval: np.ndarray) -> np.ndarray:

    E_raw = rbf.eval_matrix(X_eval, X_local)
    coeff = np.linalg.solve(A.T, E_raw.T).T
    return np.diag(w_eval) @ coeff