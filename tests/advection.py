import numpy as np
import pyvista as pv

import fvm_advection_sphere.utils.regular_mesh as regular_mesh
import fvm_advection_sphere.utils.atlas_mesh as atlas_mesh
from fvm_advection_sphere.advection import fvm_advect

import fvm_advection_sphere.utils.vis as vis

# parameters
vel = np.array([1, 0]) # velocity
δt = 1 # time step
niter = 40

# initialize mesh
#Ni = 100
#Nj = 100
#mesh = regular_mesh.setup_mesh(Ni, Nj, (0, Ni), (0, Nj))
mesh = atlas_mesh.setup_mesh()

# initialize fields
rho = np.zeros(mesh.num_vertices)
rho[1:10] = 1
#rho[100:110] = 1
#rho[200:210] = 1

vis.start_pyvista()

#c2v = mesh.c2v[np.invert(mesh.cflags_periodic)]
c2v = mesh.c2v
ds = vis.make_dataset_from_arrays(mesh.points, edges=mesh.e2v, cells=c2v, vertex_fields={"rho": rho})

#grid = ds["cells"]
#grid.point_data["rho"] = ds["vertices"].point_data["rho"]
#pl = pv.Plotter()
#pl.add_mesh(grid, show_edges=True, line_width=2, interpolate_before_map=True)
#pl.show(cpos="xy", interactive_update=True, auto_close=False)

p = vis.plot_mesh(ds, interpolate_before_map=True)
p.show(cpos="xy", interactive_update=True, auto_close=False)

for i in range(niter):
    fvm_advect(mesh, rho, vel=vel, δt=δt)

    # todo: fix
    ds["vertices"].point_data["rho"] = rho
    ds["vertices_interpolated"].point_data["rho"] = rho
    p.update()

p.show(cpos="xy")