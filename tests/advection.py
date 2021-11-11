import numpy as np

import fvm_advection_sphere.utils.regular_mesh as regular_mesh
import fvm_advection_sphere.utils.atlas_mesh as atlas_mesh
from fvm_advection_sphere.advection import fvm_advect, advector_in_edges

import fvm_advection_sphere.utils.vis as vis

from atlas4py import StructuredGrid

# initialize mesh
#Ni = 100
#Nj = 100
#mesh = regular_mesh.setup_mesh(Ni, Nj, (0, Ni), (0, Nj))
mesh = atlas_mesh.setup_mesh(StructuredGrid("O32"))

# parameters
δt = 1.0 # time step
niter = 100

# initialize fields
rho = np.zeros(mesh.num_vertices)
g11 = np.zeros(mesh.num_vertices)
g22 = np.zeros(mesh.num_vertices)
gac = np.zeros(mesh.num_vertices)
origin = mesh.xyarc.min(axis=0)
extent = mesh.xyarc.max(axis=0)-mesh.xyarc.min(axis=0)
xlim = (min(mesh.xyarc[:,0]), max(mesh.xyarc[:,0]))
ylim = (min(mesh.xyarc[:,1]), max(mesh.xyarc[:,1]))
rsina = np.sin(mesh.xyrad[:,1]) 
rcosa = np.cos(mesh.xyrad[:,1])
g11[:] = 1.0 / rcosa[:]
g22[:] = 1.0
gac[:] = rcosa[:]

uvel = np.zeros((mesh.num_vertices, 2))
u0 = 60000.0
flow_angle = np.deg2rad(0.0)  # radians

cosb = np.cos(flow_angle)
sinb = np.sin(flow_angle)
uvel[:,0] = u0 * (cosb*rcosa[:] + rsina[:]*np.cos(mesh.xyrad[:,0])*sinb)
uvel[:,1] = - u0 * np.sin(mesh.xyrad[:,0]) * sinb

vel = np.zeros((mesh.num_vertices, 2))
vel[:,0] = uvel[:,0]*g11[:]*gac[:]
vel[:,1] = uvel[:,1]*g22[:]*gac[:]

# advector in edges
vel_edges = np.zeros((mesh.num_edges, 2))
advector_in_edges(mesh, vel_nodes=vel, vel_edges=vel_edges)

for v in range(0, mesh.num_vertices):
    rel_distance_from_origin = np.linalg.norm((mesh.xyarc[v, :]-origin))/np.linalg.norm(extent)
    if rel_distance_from_origin < 0.2:
        rho[v] = 1

vis.start_pyvista()

c2v = mesh.c2v
ds = vis.make_dataset_from_arrays(mesh.xyarc, edges=mesh.e2v, cells=c2v, vertex_fields={"rho": rho})

p = vis.plot_mesh(ds, interpolate_before_map=True)
p.show(cpos="xy", interactive_update=True, auto_close=False) # non-blocking
#p.show(cpos="xy") # blocking

for i in range(niter):
    fvm_advect(mesh, rho, gac, vel=vel_edges, δt=δt)

    # todo: fix
    ds["vertices"].point_data["rho"] = rho
    ds["vertices_interpolated"].point_data["rho"] = rho
    p.update()

p.show(cpos="xy")