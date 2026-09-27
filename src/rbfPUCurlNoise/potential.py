import numpy as np
from perlin_noise import PerlinNoise
from .interpolator import PUMInterpolator
from .triang_mesh import TriMesh
from .rbf import RBF
from .patches import PatchCoverage


class NoisePotential2D(PUMInterpolator):

    def __init__(self, mesh: TriMesh, coverage: PatchCoverage, rbf: RBF):
        super().__init__(mesh, coverage, rbf)

    def _smoothstep(self, x):
        return x*x*(3 - 2*x)

    def _ramp_boundary(self, phi, d0):
        A = np.zeros_like(phi)

        band = (phi > 0) & (phi < d0)
        r = phi[band] / d0
        A[band] = self._smoothstep(r)

        A[phi >= d0] = 1.0

        return A

    def noise_potential_ramp_mult_sdf(self, phi, sdf, d0=1.0):
        # phi is the potential noise field
        psi = np.zeros(phi.shape)
        ramp_field = self._ramp_boundary(sdf, d0)

        for ind, f_value in enumerate(phi):
            psi[ind]= f_value * ramp_field[ind]
            
        return psi

    def noise_potential_ramp_add_sdf(self, phi, sdf, phi_closest, C, d0=1.0):
        # phi is the noise field (potential)
        # phi_closest is the noise field of the closest boundary point of the current vertex

        psi = np.zeros(phi.shape)
        ramp_field = self._ramp_boundary(sdf, d0)

        for ind, _ in enumerate(phi):
            psi[ind] = phi[ind] + (1.0 - ramp_field[ind]) * (C - phi_closest[ind])
            
        return psi

    def noise_potential_rbf_layers(self, noise_field, layers):
        field = np.ones(layers.shape)

        field[layers == 0] = 0
        field[layers == 1] = 0
        field[layers == 2] = 0.4

        psi = np.zeros(noise_field.shape)

        for ind, f_value in enumerate(noise_field):
            psi[ind]= f_value * field[ind]
        
        return psi