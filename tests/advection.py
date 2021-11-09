import numpy as np
import pyvista as pv

import fvm_advection_sphere.mesh.regular_mesh as regular_mesh
import fvm_advection_sphere.mesh.atlas_mesh as atlas_mesh
from fvm_advection_sphere.advection import fvm_advect

import fvm_advection_sphere.utils.vis as vis

# initialize mesh
mesh_type = "serialized" # atlas, atlas_from_file
# regular mesh
if mesh_type == "regular":
    Ni = 100
    Nj = 100
    mesh = regular_mesh.setup_mesh(Ni, Nj, (0, Ni), (0, Nj))
elif mesh_type == "atlas":
    # atlas mesh
    from atlas4py import StructuredGrid
    grid = StructuredGrid("O32")
    mesh = atlas_mesh.setup_mesh(grid)

    import bz2, pickle
    file = f"../mesh_data/atlas_{grid.name}.dat.bz2"
    with bz2.BZ2File(file, 'wb') as fd:
        pickle.dump(mesh, fd)
elif mesh_type == "serialized":
    import bz2, pickle
    file = "../mesh_data/atlas_O32.dat.bz2"
    with bz2.BZ2File(file, 'rb') as fd:
        mesh = pickle.load(fd)
else:
    raise ValueError()

# parameters
vel = np.zeros((mesh.num_edges, 2))
vel[:, 0] = 60000
δt = 1 # time step
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
    fvm_advect(mesh, rho, vel=vel, δt=δt)

    # todo: fix
    ds["vertices"].point_data["rho"] = rho
    ds["vertices_interpolated"].point_data["rho"] = rho
    p.update()

p.show(cpos="xy")