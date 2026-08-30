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
    
    def compute_rbf_velocity(self, nbrs_dict):
        vels = []
        for ind, vertex in enumerate(self.mesh["vertices"]):
            curl_vel = self.rbf_curl_2D_constrained_mesh(ind, vertex, nbrs_dict)
            vels.append(curl_vel)

        vels = np.array(vels)

        return vels
    
    def compute_interp_vel_mesh(self, mesh_refined):
        vels = []
        for vertex in mesh_refined["vertices"]:
            #print(vertex)
            partial_x = self.predict_derivative(vertex, self.mesh["vertices"], self.w_rbf, self.w_poly, self.indices, self.s, 0)
            partial_y = self.predict_derivative(vertex, self.mesh["vertices"], self.w_rbf, self.w_poly, self.indices, self.s, 1)
            
            curl_vel = np.array([partial_y, -partial_x])
            vels.append(curl_vel)

        vels = np.array(vels)
        return vels

    def compute_interp_vel_mesh_v2(self, mesh_refined, mesh):
        vels = []
        print(self.w_rbf)
        for vertex in mesh_refined["vertices"]:
            partial_x = self.predict_derivative(vertex, mesh, self.w_rbf, self.w_poly, self.indices, self.s, 0)
            partial_y = self.predict_derivative(vertex, mesh, self.w_rbf, self.w_poly, self.indices, self.s, 1)
            
            curl_vel = np.array([partial_y, -partial_x])
            vels.append(curl_vel)

        vels = np.array(vels)
        return vels
    
    def get_velocity(self, vertex, velocity):
        partial_x = self.predict_derivative(vertex, self.mesh["vertices"], self.w_rbf, self.w_poly, self.indices, self.s, 0)
        partial_y = self.predict_derivative(vertex, self.mesh["vertices"], self.w_rbf, self.w_poly, self.indices, self.s, 1)

        velocity[:] = np.array([partial_y, -partial_x])

    def rbf_curl_2D_constrained_mesh(self, center_ind, center_coord, nbrs):
        x, y = center_coord
        indices = nbrs[center_ind]

        stencil_pts = self.mesh["vertices"][indices, :]
        stencil_pts = np.squeeze(stencil_pts)
        center = np.array(center_coord)
        weights = self.nd_rbf_fd_weights(stencil_pts, center)

        f_values = []
        for i, st in enumerate(stencil_pts):
            v = indices[i]
            f_value = self.psi[v]
            f_values.append(f_value)

        f_values = np.array(f_values)

        dp_rbf_list = []
        for d in range(self.dimension):
            #print(weights[:,d])
            dp_rbf = np.dot(weights[:,d], f_values)
            dp_rbf_list.append(dp_rbf)

        dx = dp_rbf_list[0]
        dy = dp_rbf_list[1]

        return np.array([dy, -dx])