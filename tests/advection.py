import numpy as np
import pyvista as pv

import fvm_advection_sphere.utils.regular_mesh as regular_mesh
import fvm_advection_sphere.utils.atlas_mesh as atlas_mesh
from fvm_advection_sphere.advection import fvm_advect

import fvm_advection_sphere.utils.vis as vis

from atlas4py import StructuredGrid

# initialize mesh
#Ni = 100
#Nj = 100
#mesh = regular_mesh.setup_mesh(Ni, Nj, (0, Ni), (0, Nj))
mesh = atlas_mesh.setup_mesh(StructuredGrid("O32"))

# parameters
vel = np.zeros((mesh.num_edges, 2))
vel[:, 0] = 60000
δt = 1 # time step
niter = 100

# initialize fields
rho = np.zeros(mesh.num_vertices)
origin = mesh.points.min(axis=0)
extent = mesh.points.max(axis=0)-mesh.points.min(axis=0)
xlim = (min(mesh.points[:, 0]), max(mesh.points[:, 0]))
ylim = (min(mesh.points[:, 1]), max(mesh.points[:, 1]))
for v in range(0, mesh.num_vertices):
    rel_distance_from_origin = np.linalg.norm((mesh.points[v, :]-origin))/np.linalg.norm(extent)
    if rel_distance_from_origin < 0.2:
        rho[v] = 1

vis.start_pyvista()

c2v = mesh.c2v
ds = vis.make_dataset_from_arrays(mesh.points, edges=mesh.e2v, cells=c2v, vertex_fields={"rho": rho})

p = vis.plot_mesh(ds, interpolate_before_map=True)
p.show(cpos="xy", interactive_update=True, auto_close=False) # non-blocking
#p.show(cpos="xy") # blocking

for i in range(niter):
    fvm_advect(mesh, rho, vel=vel, δt=δt)

    # todo: fix
    ds["vertices"].point_data["rho"] = rho
    ds["vertices_interpolated"].point_data["rho"] = rho
    p.update()

p.show(cpos="xy")