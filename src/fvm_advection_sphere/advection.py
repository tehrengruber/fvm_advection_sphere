import numpy as np
import types
from timeit import default_timer as timer

from functional.ffront.fbuiltins import Field, float64, where, neighbor_sum
from functional.ffront.decorator import field_operator
from functional.iterator.embedded import np_as_located_field
from functional.ffront.fbuiltins import neighbor_sum

from fvm_advection_sphere.common import *


@field_operator
def upstream_flux_(
    rho : Field[[Vertex], float64],
    vel_x: Field[[Edge], float64],
    vel_y: Field[[Edge], float64],
    dual_face_normal_weighted_x: Field[[Edge], float64],
    dual_face_normal_weighted_y: Field[[Edge], float64]
) -> Field[[Edge], float64]:
    # weighted normal velocity
    wnv = vel_x*dual_face_normal_weighted_x+vel_y*dual_face_normal_weighted_y
    return where(wnv > 0.0, rho(E2V[0]) * wnv, rho(E2V[1]) * wnv)



@field_operator
def fluxdiv_(
    rho: Field[[Vertex], float64],
    dt: Field[[Vertex], float64],
    vol: Field[[Vertex], float64],
    gac: Field[[Vertex], float64],
    flux: Field[[Edge], float64],
    dual_face_orientation: Field[[Vertex, V2EDim], float64]
) -> Field[[Vertex], float64]:
    return rho - dt / (vol*gac) * neighbor_sum(flux(V2E) * dual_face_orientation, axis=V2EDim)


@field_operator
def fvm_advect_(
        rho: Field[[Vertex], float64],
        dt: Field[[Vertex], float64],
        vol: Field[[Vertex], float64],
        gac: Field[[Vertex], float64],
        vel_x: Field[[Edge], float64],
        vel_y: Field[[Edge], float64],
        dual_face_orientation: Field[[Vertex, V2EDim], float64],
        dual_face_normal_weighted_x: Field[[Edge], float64],
        dual_face_normal_weighted_y: Field[[Edge], float64]
) -> Field[[Vertex], float64]:
    flux = upstream_flux_(rho, vel_x, vel_y, dual_face_normal_weighted_x, dual_face_normal_weighted_y)
    return fluxdiv_(rho, dt, vol, gac, flux, dual_face_orientation)


def fvm_advect(
        mesh,
        rho: np.ndarray,  # field on vertices
        gac: np.ndarray,  # field on vertices
        vel: np.ndarray,  # 2d-vector on edges
        *,
        δt: float,
        offset_provider
):
    start = timer()

    rho_ = np_as_located_field(Vertex)(rho)
    rho_next = np_as_located_field(Vertex)(np.zeros(rho.shape))
    dt = np_as_located_field(Vertex)(np.ones(rho.shape) * δt)
    vol_ = np_as_located_field(Vertex)(mesh.vol)
    gac_ = np_as_located_field(Vertex)(gac)
    vel_x = np_as_located_field(Edge)(vel[:, 0])
    vel_y = np_as_located_field(Edge)(vel[:, 1])
    # TODO: check arguments type
    dual_face_orientation \
        = np_as_located_field(Vertex, V2EDim)(mesh.dual_face_orientation)
    dual_face_normal_weighted_x \
        = np_as_located_field(Edge)(mesh.dual_face_normal_weighted[:, 0])
    dual_face_normal_weighted_y \
        = np_as_located_field(Edge)(mesh.dual_face_normal_weighted[:, 1])

    #flux_t = np_as_located_field(Edge)(np.zeros(mesh.num_edges))
    #upstream_flux_(rho_, vel_x, vel_y, dual_face_normal_weighted_x, dual_face_normal_weighted_y, out=flux_t, offset_provider=offset_provider)
    #flux2 = upstream_flux(mesh, rho, vel)
    #assert np.allclose(flux_t.array(), flux2)

    #fluxdiv_t = np_as_located_field(Vertex)(np.zeros(mesh.num_vertices))
    #fluxdiv_(rho_, dt, vol_, gac_, flux_t, dual_face_orientation, out=fluxdiv_t,
    #               offset_provider=offset_provider)
    #fluxdiv_ref = fluxdiv_np(mesh, rho, gac, flux2, δt=δt)

    fvm_advect_(rho_, dt, vol_, gac_, vel_x, vel_y, dual_face_orientation,
                dual_face_normal_weighted_x, dual_face_normal_weighted_y, out=rho_next, offset_provider=offset_provider)

    #rho_ref = fvm_advect_np(mesh, rho, gac, vel, δt=δt)

    end = timer()
    print(end - start)

    return rho_next.array()



def advector_in_edges(
    mesh,
    vel_vertices
):
    vel_edges = np.zeros((mesh.num_edges, 2))

    for e in range(0, mesh.num_edges):
        v1, v2 = mesh.e2v[e,:]
        vel_edges[e,0] = 0.5 * (vel_vertices[v1,0] + mesh.pole_bc[e]*vel_vertices[v2,0])
        vel_edges[e,1] = 0.5 * (vel_vertices[v1,1] + mesh.pole_bc[e]*vel_vertices[v2,1])

    for ep in range(0, mesh.num_pole_edges):
        e = mesh.pole_edges[ep]
        vel_edges[e,1] = 0.0

    return vel_edges