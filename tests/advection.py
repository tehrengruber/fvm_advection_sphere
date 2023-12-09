import numpy as np

from gt4py.eve.utils import FrozenNamespace
from atlas4py import Topology

from timeit import default_timer as timer

from gt4py.next.common import GridType
from gt4py.next.ffront.decorator import field_operator
from gt4py.next.ffront.fbuiltins import arccos, sin, cos, where, minimum, Field, broadcast, int32
from fvm_advection_sphere.build_config import float_type
from fvm_advection_sphere.common import *
from fvm_advection_sphere import build_config
from fvm_advection_sphere.mesh.atlas_mesh import AtlasMesh, update_periodic_layers
import fvm_advection_sphere.mesh.regular_mesh as regular_mesh
from fvm_advection_sphere.state_container import StateContainer, allocate_field
from fvm_advection_sphere.advection import (
    mpdata_program, upwind_scheme, nabla_z
)  # , advect_density, mpdata_program
from fvm_advection_sphere.output import output_data
from fvm_advection_sphere.metric import Metric

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

    grid = StructuredGrid("O32")
    mesh = AtlasMesh.generate(grid, num_level=30)

    if False:
        import copy

        picklable_mesh = copy.copy(mesh)
        picklable_mesh._atlas_mesh = None
        import bz2, pickle

        file = f"../mesh_data/atlas_{grid.name}.dat.bz2"
        with bz2.BZ2File(file, "wb") as fd:
            pickle.dump(picklable_mesh, fd)
elif mesh_type == "serialized":
    import bz2, pickle

    file = "../mesh_data/atlas_O32.dat.bz2"
    with bz2.BZ2File(file, "rb") as fd:
        mesh = pickle.load(fd)
else:
    raise ValueError()

# write mesh to file for debugging purposes
with open("mesh.txt", "w") as f:
    for key, value in mesh.__dict__.items():
        if key[0] == "_":
            continue
        if isinstance(value, np.ndarray):
            print(f"{key}:", file=f)
            print(value, file=f)
        else:
            print(f"{key}: {value}", file=f)

print(mesh.info())

constants = FrozenNamespace(
    pi=np.pi,
    deg2rad=2.0 * np.pi / 360.0,
)

# parameters
# δt = 3600.0  # time step in s
# niter = 576
δt = 1800.0  # time step in s
niter = 1000
# niter = 300
# niter = 30
# model_endtime = 3600.0 * 24.0 * 24.0
eps = 1.0e-8

# some properties derived from the mesh
metric = Metric.from_mesh(mesh)
origin = mesh.xyarc.min(axis=0)
extent = mesh.xyarc.max(axis=0) - mesh.xyarc.min(axis=0)
xlim = (min(mesh.xyarc[:, 0]), max(mesh.xyarc[:, 0]))
ylim = (min(mesh.xyarc[:, 1]), max(mesh.xyarc[:, 1]))
level_indices = allocate_field(mesh, Field[[K], int32])
level_indices[...] = np.arange(0., mesh.num_level)

# initialize fields
state = StateContainer.from_mesh(mesh)
state_next = StateContainer.from_mesh(mesh)

# initialize temporaries
tmp_fields = {}
for i in range(6):
    tmp_fields[f"tmp_vertex_{i}"] = allocate_field(mesh, Field[[Vertex, K], float_type])
for j in range(3):
    tmp_fields[f"tmp_edge_{j}"] = allocate_field(mesh, Field[[Edge, K], float_type])


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def initial_rho(
    mesh_radius: float_type,
    mesh_xydeg_x: Field[[Vertex], float_type],
    mesh_xydeg_y: Field[[Vertex], float_type],
    mesh_vertex_ghost_mask: Field[[Vertex], bool],
) -> Field[[Vertex, K], float_type]:
    lonc = 0.5 * constants.pi
    latc = 0.0

    mesh_xyrad_x, mesh_xyrad_y = mesh_xydeg_x * constants.deg2rad, mesh_xydeg_y * constants.deg2rad

    rsina, rcosa = sin(mesh_xyrad_y), cos(mesh_xyrad_y)

    zdist = mesh_radius * arccos(sin(latc) * rsina + cos(latc) * rcosa * cos(mesh_xyrad_x - lonc))
    rpr = (zdist / (mesh_radius / 2.0)) ** 2.0
    rpr = minimum(1.0, rpr)
    return broadcast(where(mesh_vertex_ghost_mask, 0.0, 0.5 * (1.0 + cos(constants.pi * rpr))), (Vertex, K))


initial_rho(
    mesh.radius,
    mesh.xydeg_x,
    mesh.xydeg_y,
    mesh.vertex_ghost_mask,
    out=state.rho,
    offset_provider=mesh.offset_provider,
)

outstep = 0
output_data(mesh, state, outstep)


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def initial_velocity(
    mesh_xydeg_x: Field[[Vertex], float_type],
    mesh_xydeg_y: Field[[Vertex], float_type],
    metric_gac: Field[[Vertex], float_type],
    metric_g11: Field[[Vertex], float_type],
    metric_g22: Field[[Vertex], float_type],
) -> tuple[Field[[Vertex, K], float_type], Field[[Vertex, K], float_type], Field[[Vertex, K], float_type]]:
    mesh_xyrad_x, mesh_xyrad_y = mesh_xydeg_x * constants.deg2rad, mesh_xydeg_y * constants.deg2rad

    u0 = 22.238985328911745
    flow_angle = 0.0 * constants.deg2rad  # radians

    rsina, rcosa = sin(mesh_xyrad_y), cos(mesh_xyrad_y)

    cosb, sinb = cos(flow_angle), sin(flow_angle)
    uvel_x = u0 * (cosb * rcosa + rsina * cos(mesh_xyrad_x) * sinb)
    uvel_y = -u0 * sin(mesh_xyrad_x) * sinb

    vel_x = broadcast(uvel_x * metric_g11 * metric_gac, (Vertex, K))
    vel_y = broadcast(uvel_y * metric_g22 * metric_gac, (Vertex, K))
    vel_z = broadcast(0., (Vertex, K))
    return vel_x, vel_y, vel_z

initial_velocity(
    mesh.xydeg_x,
    mesh.xydeg_y,
    metric.gac,
    metric.g11,
    metric.g22,
    out=state.vel,
    offset_provider=mesh.offset_provider,
)

print(
    f"rho0 | min, max, avg : {np.min(state.rho.asnumpy())}, {np.max(state.rho.asnumpy())}, {np.average(state.rho.asnumpy())} | "
)

state_next.vel = state.vel  # constant velocity for now
start = timer()

# TODO(tehrengruber): use somewhere meaningful and remove from here
tmp_fields[f"tmp_vertex_0"].asnumpy()[...] = np.arange(0., mesh.num_level)[np.newaxis, :]
nabla_z(tmp_fields[f"tmp_vertex_0"], level_indices, mesh.num_level, out=tmp_fields[f"tmp_vertex_1"], offset_provider=mesh.offset_provider)

for i in range(niter):

    # advect_density(
    #    state.rho,
    #    δt,
    #    mesh.vol,
    #    metric.gac,
    #    state.vel[0],
    #    state.vel[1],
    #    mesh.pole_edge_mask,
    #    mesh.dual_face_orientation,
    #    mesh.dual_face_normal_weighted_x,
    #    mesh.dual_face_normal_weighted_y,
    #    out=state_next.rho,
    #    offset_provider=mesh.offset_provider,
    # )

    upwind_scheme(
       state.rho,
       δt,
       mesh.vol,
       metric.gac,
       state.vel[0],
       state.vel[1],
       state.vel[2],
       mesh.pole_edge_mask,
       mesh.dual_face_orientation,
       mesh.dual_face_normal_weighted_x,
       mesh.dual_face_normal_weighted_y,
       out=state_next.rho,
       offset_provider=mesh.offset_provider,
    )

    # mpdata_program(
    #     state.rho,
    #     state_next.rho,
    #     δt,
    #     eps,
    #     mesh.vol,
    #     metric.gac,
    #     state.vel[0],
    #     state.vel[1],
    #     state.vel[2],
    #     mesh.pole_edge_mask,
    #     mesh.dual_face_orientation,
    #     mesh.dual_face_normal_weighted_x,
    #     mesh.dual_face_normal_weighted_y,
    #     **tmp_fields,
    #     offset_provider=mesh.offset_provider,
    # )

    state, state_next = state_next, state  # "pointer swap"

    update_periodic_layers(mesh, state.rho)

    # start_plotting = timer()

    # end_plotting = timer()
    # print(f"Plotting {i} ({end_plotting - start_plotting}s)")
    print(
        f"rho | min, max, avg : {np.min(state.rho.asnumpy())}, {np.max(state.rho.asnumpy())}, {np.average(state.rho.asnumpy())} | "
    )

    print(f"Timestep {i}")

outstep = 1
output_data(mesh, state, outstep)
end = timer()
print(f"Timestep {i} ({end - start}s)")
print("Done")
