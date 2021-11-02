import numpy as np

from fvm_advection_sphere.utils.cartesian_mesh import cartesian_mesh
from fvm_advection_sphere.advection import fvm_advect

import fvm_advection_sphere.utils.vis as vis

# parameters
vel = np.array([1, 0]) # velocity
δt = 1 # time step
niter = 10

# initialize mesh
Ni = 10
Nj = 10
mesh = cartesian_mesh(Ni, Nj, (0, Ni), (0, Nj))

# initialize fields
rho = np.zeros(mesh.num_vertices)
rho[0] = 1
#  todo: geometry: dual_volume, dual_normal, face_orientation

vis.start_pyvista()

ds = vis.make_dataset_from_arrays(mesh.points, edges=mesh.e2v, cells=mesh.c2v, vertex_fields={"rho": rho})
p = vis.plot_mesh(ds)
p.show(cpos="xy", interactive_update=True, auto_close=False)

for i in range(niter):
    fvm_advect(mesh, rho, vel=vel, δt=δt)
    # todo: fix
    ds["vertices"].point_data["rho"] = rho
    p.update()

bla = 1+1