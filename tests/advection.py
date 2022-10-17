from atlas4py import Topology
import numpy as np
import pyvista as pv

from timeit import default_timer as timer

import fvm_advection_sphere.mesh.regular_mesh as regular_mesh
from fvm_advection_sphere.mesh.atlas_mesh import AtlasMesh, update_periodic_layers
from fvm_advection_sphere.advection import fvm_advect, advector_in_edges
from functional.iterator.embedded import NeighborTableOffsetProvider

import fvm_advection_sphere.utils.vis as vis
from fvm_advection_sphere.common import *
from fvm_advection_sphere.state import StateContainer

# initialize mesh
mesh_type = "atlas"  # atlas, atlas_from_file

# regular mesh
if mesh_type == "regular":
    Ni = 100
    Nj = 100
    mesh = regular_mesh.setup_mesh(Ni, Nj, (0, Ni), (0, Nj))
elif mesh_type == "atlas":
    # atlas mesh
    from atlas4py import StructuredGrid
    grid = StructuredGrid("O50")
    mesh = AtlasMesh.generate(grid)

    if False:
        import copy
        picklable_mesh = copy.copy(mesh)
        picklable_mesh._atlas_mesh = None
        import bz2, pickle
        file = f"../mesh_data/atlas_{grid.name}.dat.bz2"
        with bz2.BZ2File(file, 'wb') as fd:
            pickle.dump(picklable_mesh, fd)
elif mesh_type == "serialized":
    import bz2, pickle
    file = "../mesh_data/atlas_O32.dat.bz2"
    with bz2.BZ2File(file, 'rb') as fd:
        mesh = pickle.load(fd)
else:
    raise ValueError()

# write mesh to file for debugging purposes
with open('mesh.txt', 'w') as f:
    for key, value in mesh.__dict__.items():
        if key[0] == "_":
            continue
        if isinstance(value, np.ndarray):
            print(f"{key}:", file=f)
            print(value, file=f)
        else:
            print(f"{key}: {value}", file=f)

print(mesh.info())

# parameters
δt = 3600.0  # time step in s
niter = 1000

# initialize fields
state = StateContainer.from_mesh(mesh)

g11 = np.zeros(mesh.num_vertices)
g22 = np.zeros(mesh.num_vertices)
gac = np.zeros(mesh.num_vertices)
origin = mesh.xyarc.min(axis=0)
extent = mesh.xyarc.max(axis=0)-mesh.xyarc.min(axis=0)
xlim = (min(mesh.xyarc[:, 0]), max(mesh.xyarc[:, 0]))
ylim = (min(mesh.xyarc[:, 1]), max(mesh.xyarc[:, 1]))
rsina = np.sin(mesh.xyrad[:, 1])
rcosa = np.cos(mesh.xyrad[:, 1])
g11[:] = 1.0 / rcosa[:]
g22[:] = 1.0
gac[:] = rcosa[:]

#lonc = 60/180*np.pi
#latc = 60/180 * np.pi
lonc = 0.5 * np.pi
latc = 0
for jv in range(0, mesh.num_vertices):
    zdist = mesh.radius * np.arccos(np.sin(latc) * rsina[jv] + np.cos(latc) * rcosa[jv] * np.cos(mesh.xyrad[jv, 0] - lonc))
    rpr = (zdist / (mesh.radius / 2)) ** 2
    rpr = min(1.0, rpr)
    if not mesh.vertex_flags[jv] & Topology.GHOST:
        state.rho[jv] = 0.5 * (1.0 + np.cos(np.pi * rpr))

from functional.ffront.fbuiltins import arccos, sin, cos, where, minimum
@field_operator
def intial_rho(
    mesh_radius: float,
    mesh_xydeg_x: Field[[Vertex], float],
    mesh_xydeg_y: Field[[Vertex], float],
    vertex_ghost_mask: Field[[Vertex], float]
) -> Field[[Vertex], float]:
    PI = 3.141592653589793
    DEG2RAD = 2.0 * PI / 360.0
    lonc = 0.5 * PI
    latc = 0.

    rsina = sin(mesh_xyrad_y)
    rcosa = cos(mesh_xyrad_y)

    xyrad_x = mesh_xydeg * DEG2RAD
    xyrad_x = mesh_xydeg * DEG2RAD

    zdist = mesh_radius * arccos(sin(latc) * rsina + cos(latc) * rcosa * cos(xyrad_x - lonc))
    rpr = (zdist / (mesh_radius / 2)) ** 2
    rpr = minimum(1.0, rpr)
    return where(vertex_ghost_mask, 0., 0.5 * (1.0 + cos(PI * rpr)))

#initial_rho(mesh.radius, mesh.xyrad)

uvel = np.zeros((mesh.num_vertices, 2))
u0 = -30.0  # m/s
flow_angle = np.deg2rad(45.0)  # radians


cosb = np.cos(flow_angle)
sinb = np.sin(flow_angle)
uvel[:, 0] = u0 * (cosb * rcosa[:] + rsina[:] * np.cos(mesh.xyrad[:, 0]) * sinb)
uvel[:, 1] = - u0 * np.sin(mesh.xyrad[:, 0]) * sinb

state.vel_x[:] = uvel[:,0]*g11[:]*gac[:]
state.vel_y[:] = uvel[:,1]*g22[:]*gac[:]

#vel[:,0] = 30
#vel[:,1] = 0

offset_provider = {
    "E2V": mesh.e2v,
    "V2E": mesh.v2e,
    "V2VE": mesh.v2ve
}

# advector in edges
#vel_edges = advector_in_edges(mesh, vel, offset_provider=offset_provider)

vis.start_pyvista()

#c2v = mesh.c2v[np.invert(mesh.cflags_periodic)] # fixes visualization with regular mesh
c2v = mesh.c2v_np
ds = vis.make_dataset_from_arrays(mesh.xyarc, edges=mesh.e2v_np, cells=c2v, vertex_fields={"rho": np.asarray(state.rho)}) # use mesh.xyz for vis on the sphere
#ds = vis.make_dataset_from_arrays(mesh.xyz, edges=mesh.e2v, cells=c2v, vertex_fields={"rho": rho}) # use mesh.xyz for vis on the sphere
p = vis.plot_mesh(ds, interpolate_before_map=True)

#
# vertices
#
ds["vertices"]["indices"] = [f"v{idx}" for idx in range(mesh.num_vertices)]
ds["vertices"]["remote_indices"] = [f"v{idx}, {remote_idx}" for (idx, remote_idx) in zip(range(mesh.num_vertices), mesh.remote_indices[Vertex])]
ds["vertices"]["periodic_vertices"] = [f"{i}" for i in (mesh.vertex_flags&Topology.BC).astype(np.bool).astype(np.int)]
ds["vertices"]["conn_v2e"] = [f"{mesh.v2e_np[i, :]}" for i in range(mesh.num_vertices)]
ds["vertices"]["conn_v2c"] = [f"{mesh.v2c_np[i, :]}" for i in range(mesh.num_vertices)]

#
# edges
#
#p.add_point_labels(ds["vertices"], "conn", point_size=5, font_size=10)
def compute_edge_centers(mesh):
    """Compute edge centers for all non-pole edges."""
    pole_edge_mask = mesh.edge_flags&Topology.POLE==0
    edge_centers = (mesh.xyarc[mesh.e2v_np[pole_edge_mask, 0]]+mesh.xyarc[mesh.e2v_np[pole_edge_mask, 1]])/2
    edge_indices = np.arange(0, mesh.num_edges, 1, dtype=int)[pole_edge_mask]
    return edge_indices, np.concatenate(
        (
            edge_centers,
            np.zeros((edge_centers.shape[0], 3 - edge_centers.shape[1]),
                     dtype=edge_centers.dtype),
        ),
        axis=1,
    )
edge_center_indices, edge_centers = compute_edge_centers(mesh)
ds["edge_centers"] = pv.PolyData(edge_centers)
ds["edge_centers"]["indices"] = [f"e{i}" for i in edge_center_indices]
ds["edge_centers"]["remote_indices"] = [f"e{i}, {mesh.remote_indices[Edge][i]}" for i in edge_center_indices]

#
# cells
#
cell_centers = np.zeros((mesh.num_cells, 3))
for cell_id in range(mesh.num_cells):
    num_vertices_per_cell = 0
    for vertex_id in range(0, mesh.c2v_np.shape[1]):
        if mesh.c2v_np[cell_id, vertex_id] != -1:
            num_vertices_per_cell += 1
            cell_centers[cell_id, 0:2] += mesh.xyarc[mesh.c2v_np[cell_id, vertex_id], :]
    cell_centers[cell_id, :] /= num_vertices_per_cell
ds["cell_centers"] = pv.PolyData(cell_centers)
ds["cell_centers"]["indices"] = [f"c{idx}" for idx in range(mesh.num_cells)]
ds["cell_centers"]["conn_c2v"] = [f"{mesh.c2v_np[i, :]}" for i in range(mesh.num_cells)]
ds["cell_centers"]["conn_c2e"] = [f"{mesh.c2e_np[i, :]}" for i in range(mesh.num_cells)]


#p.add_point_labels(ds["edge_centers"], "indices", point_size=5, font_size=9)
#p.add_point_labels(ds["vertices"], "conn_v2e", point_size=5, font_size=10)
#p.add_point_labels(ds["vertices"], "remote_indices", point_size=5, font_size=10)
#p.add_point_labels(ds["vertices"], "periodic_vertices", point_size=5, font_size=10)
#p.add_point_labels(ds["cell_centers"], "indices", point_size=5, font_size=9)
#p.add_point_labels(ds["edge_centers"], "remote_indices", point_size=5, font_size=9)
p.show(cpos="xy", interactive_update=True, auto_close=False)  # non-blocking
#p.render()
#p.show(cpos="xy") # blocking

for i in range(niter):
    start = timer()

    state.rho = fvm_advect(
        mesh,
        state.rho,
        gac,
        (state.vel_x, state.vel_y),
        δt=δt,
        offset_provider=offset_provider
    )

    update_periodic_layers(mesh, state.rho)

    start_plotting = timer()
    # todo: avoid copy
    ds["vertices"].point_data["rho"] = np.asarray(state.rho)
    ds["vertices_interpolated"].point_data["rho"] = np.asarray(state.rho)
    p.render()
    #p.update()  # use p.render() otherwise (update sometimes hangs)
    end_plotting = timer()
    print(f"Plotting {i} ({end_plotting - start_plotting}s)")

    end = timer()
    print(f"Timestep {i} ({end - start}s)")

print("Done")
p.show(cpos="xy")
