import numpy as np

from eve.utils import FrozenNamespace
from atlas4py import Topology

from timeit import default_timer as timer

from functional.ffront.decorator import field_operator
from functional.ffront.fbuiltins import arccos, sin, cos, where, minimum, Field
from functional.iterator.embedded import np_as_located_field

from fvm_advection_sphere.common import *
from fvm_advection_sphere import build_config
from fvm_advection_sphere.mesh.atlas_mesh import AtlasMesh, update_periodic_layers
import fvm_advection_sphere.mesh.regular_mesh as regular_mesh
from fvm_advection_sphere.state_container import StateContainer, allocate_field
from fvm_advection_sphere.advection import (
    mpdata_program,
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
    mesh = AtlasMesh.generate(grid)

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

# some properties derived from the mesh
metric = Metric.from_mesh(mesh)
origin = mesh.xyarc.min(axis=0)
extent = mesh.xyarc.max(axis=0) - mesh.xyarc.min(axis=0)
xlim = (min(mesh.xyarc[:, 0]), max(mesh.xyarc[:, 0]))
ylim = (min(mesh.xyarc[:, 1]), max(mesh.xyarc[:, 1]))

# initialize fields
state = StateContainer.from_mesh(mesh)
state_next = StateContainer.from_mesh(mesh)

# initialize temporaries
tmp_fields = {}
for i in range(4):
    tmp_fields[f"tmp_vertex_{i}"] = allocate_field(mesh, Field[[Vertex], float])
for j in range(2):
    tmp_fields[f"tmp_edge_{j}"] = allocate_field(mesh, Field[[Edge], float])


@field_operator(backend=build_config.backend)
def initial_rho(
    mesh_radius: float,
    mesh_xydeg_x: Field[[Vertex], float],
    mesh_xydeg_y: Field[[Vertex], float],
    mesh_vertex_ghost_mask: Field[[Vertex], bool],
) -> Field[[Vertex], float]:
    lonc = 0.5 * constants.pi
    latc = 0.0

    mesh_xyrad_x, mesh_xyrad_y = mesh_xydeg_x * constants.deg2rad, mesh_xydeg_y * constants.deg2rad

    rsina, rcosa = sin(mesh_xyrad_y), cos(mesh_xyrad_y)

    zdist = mesh_radius * arccos(sin(latc) * rsina + cos(latc) * rcosa * cos(mesh_xyrad_x - lonc))
    rpr = (zdist / (mesh_radius / 2.0)) ** 2.0
    rpr = minimum(1.0, rpr)
    return where(mesh_vertex_ghost_mask, 0.0, 0.5 * (1.0 + cos(constants.pi * rpr)))


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


@field_operator(backend=build_config.backend)
def initial_velocity(
    mesh_xydeg_x: Field[[Vertex], float],
    mesh_xydeg_y: Field[[Vertex], float],
    metric_gac: Field[[Vertex], float],
    metric_g11: Field[[Vertex], float],
    metric_g22: Field[[Vertex], float],
) -> tuple[Field[[Vertex], float], Field[[Vertex], float]]:
    mesh_xyrad_x, mesh_xyrad_y = mesh_xydeg_x * constants.deg2rad, mesh_xydeg_y * constants.deg2rad

    u0 = 22.238985328911745
    flow_angle = 0.0 * constants.deg2rad  # radians

    rsina, rcosa = sin(mesh_xyrad_y), cos(mesh_xyrad_y)

    cosb, sinb = cos(flow_angle), sin(flow_angle)
    uvel_x = u0 * (cosb * rcosa + rsina * cos(mesh_xyrad_x) * sinb)
    uvel_y = -u0 * sin(mesh_xyrad_x) * sinb

    vel_x = uvel_x * metric_g11 * metric_gac
    vel_y = uvel_y * metric_g22 * metric_gac
    return vel_x, vel_y


@field_operator(backend=build_config.backend)
def initial_velocity_x(
    mesh_xydeg_x: Field[[Vertex], float],
    mesh_xydeg_y: Field[[Vertex], float],
    metric_gac: Field[[Vertex], float],
    metric_g11: Field[[Vertex], float],
    metric_g22: Field[[Vertex], float],
) -> Field[[Vertex], float]:
    return initial_velocity(mesh_xydeg_x, mesh_xydeg_y, metric_gac, metric_g11, metric_g22)[0]


@field_operator(backend=build_config.backend)
def initial_velocity_y(
    mesh_xydeg_x: Field[[Vertex], float],
    mesh_xydeg_y: Field[[Vertex], float],
    metric_gac: Field[[Vertex], float],
    metric_g11: Field[[Vertex], float],
    metric_g22: Field[[Vertex], float],
) -> Field[[Vertex], float]:
    return initial_velocity(mesh_xydeg_x, mesh_xydeg_y, metric_gac, metric_g11, metric_g22)[1]


initial_velocity_x(
    mesh.xydeg_x,
    mesh.xydeg_y,
    metric.gac,
    metric.g11,
    metric.g22,
    out=state.vel[0],
    offset_provider=mesh.offset_provider,
)
initial_velocity_y(
    mesh.xydeg_x,
    mesh.xydeg_y,
    metric.gac,
    metric.g11,
    metric.g22,
    out=state.vel[1],
    offset_provider=mesh.offset_provider,
)

state_next.vel = state.vel  # constant velocity for now
start = timer()

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

    # upwind_scheme(
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

    mpdata_program(
        state.rho,
        state_next.rho,
        δt,
        mesh.vol,
        metric.gac,
        state.vel[0],
        state.vel[1],
        mesh.pole_edge_mask,
        mesh.dual_face_orientation,
        mesh.dual_face_normal_weighted_x,
        mesh.dual_face_normal_weighted_y,
        **tmp_fields,
        offset_provider=mesh.offset_provider,
    )

    # mpdata_scheme(
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

    state, state_next = state_next, state  # "pointer swap"

    update_periodic_layers(mesh, state.rho)

    # start_plotting = timer()

    # end_plotting = timer()
    # print(f"Plotting {i} ({end_plotting - start_plotting}s)")
    print(
        f"rho | min, max, avg : {np.min(state.rho)}, {np.max(state.rho)}, {np.average(state.rho)} | "
    )

    print(f"Timestep {i}")

outstep = 1
output_data(mesh, state, outstep)
end = timer()
print(f"Timestep {i} ({end - start}s)")
print("Done")
