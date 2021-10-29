import numpy as np

from fvm_advection_sphere.utils.cartesian_mesh import cartesian_mesh
from fvm_advection_sphere.advection import fvm_advect

# parameters
vel = np.array([1., 0.]) # velocity
δt = 1 # time step
niter = 1

# initialize mesh
Ni = 10
Nj = 10
mesh = cartesian_mesh(Ni, Nj, [0, 1], [0, 1])

# initialize fields
rho = np.zeros(mesh.num_vertices)
rho[0] = 1
#  todo: geometry: dual_volume, dual_normal, face_orientation

for i in range(niter):
    fvm_advect(mesh, rho, vel=vel, δt=δt)

bla = 1+1