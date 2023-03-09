from functional.ffront.fbuiltins import (
    Field,
    where,
    neighbor_sum,
    max_over,
    min_over,
    abs,
    maximum,
    minimum,
)
from functional.ffront.decorator import field_operator, program

from fvm_advection_sphere.common import *
from fvm_advection_sphere.build_config import float_type
from fvm_advection_sphere import build_config


@field_operator
def advector_in_edges(
    vel_x: Field[[Vertex], float_type],
    vel_y: Field[[Vertex], float_type],
    pole_edge_mask: Field[[Edge], bool],
) -> tuple[Field[[Edge], float_type], Field[[Edge], float_type]]:
    pole_bc = where(pole_edge_mask, -1.0, 1.0)
    vel_edges_x = 0.5 * (vel_x(E2V[0]) + pole_bc * vel_x(E2V[1]))
    vel_edges_y = 0.5 * (vel_y(E2V[0]) + pole_bc * vel_y(E2V[1]))
    return vel_edges_x, where(pole_edge_mask, 0.0, vel_edges_y)


@field_operator
def advector_normal(
    vel_x: Field[[Vertex], float_type],
    vel_y: Field[[Vertex], float_type],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_normal_weighted_x: Field[[Edge], float_type],
    dual_face_normal_weighted_y: Field[[Edge], float_type],
) -> Field[[Edge], float_type]:
    pole_bc = where(pole_edge_mask, -1.0, 1.0)
    vel_edges_x = 0.5 * (vel_x(E2V[0]) + pole_bc * vel_x(E2V[1]))
    vel_edges_y = 0.5 * (vel_y(E2V[0]) + pole_bc * vel_y(E2V[1]))
    vel_edges_y = where(pole_edge_mask, 0.0, vel_edges_y)
    # vel_edges_x = where(pole_edge_mask, 0.0, vel_edges_x)
    return vel_edges_x * dual_face_normal_weighted_x + vel_edges_y * dual_face_normal_weighted_y


@field_operator
def upstream_flux(
    rho: Field[[Vertex], float_type],
    vel_x: Field[[Vertex], float_type],
    vel_y: Field[[Vertex], float_type],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_normal_weighted_x: Field[[Edge], float_type],
    dual_face_normal_weighted_y: Field[[Edge], float_type],
) -> Field[[Edge], float_type]:
    vel_x_face, vel_y_face = advector_in_edges(vel_x, vel_y, pole_edge_mask)
    wnv = vel_x_face * dual_face_normal_weighted_x + vel_y_face * dual_face_normal_weighted_y
    return where(wnv > 0.0, rho(E2V[0]) * wnv, rho(E2V[1]) * wnv)


@field_operator
def upwind_flux(
    rho: Field[[Vertex], float_type],
    veln: Field[[Edge], float_type],
) -> Field[[Edge], float_type]:
    return where(veln > 0.0, rho(E2V[0]) * veln, rho(E2V[1]) * veln)


@field_operator
def centered_flux(
    rho: Field[[Vertex], float_type],
    veln: Field[[Edge], float_type],
) -> Field[[Edge], float_type]:
    return (
        0.5 * veln * (rho(E2V[1]) + rho(E2V[0]))
    )  # todo(ckuehnlein): polar flip for u and v transport later


@field_operator
def pseudo_flux(
    rho: Field[[Vertex], float_type],
    veln: Field[[Edge], float_type],
    grg: Field[[Vertex], float_type],
    cfluxdiv: Field[[Vertex], float_type],
    dt: float_type,
) -> Field[[Edge], float_type]:
    return 0.5 * abs(veln) * (rho(E2V[1]) - rho(E2V[0])) - dt * veln * 0.5 * (
        (cfluxdiv(E2V[1]) + cfluxdiv(E2V[0])) / (grg(E2V[1]) + grg(E2V[0]))
    )


@field_operator(backend=build_config.backend)
def flux_divergence(
    flux: Field[[Edge], float_type],
    vol: Field[[Vertex], float_type],
    gac: Field[[Vertex], float_type],
    dual_face_orientation: Field[[Vertex, V2EDim], float_type],
) -> Field[[Vertex], float_type]:
    return 1.0 / (vol * gac) * neighbor_sum(flux(V2E) * dual_face_orientation, axis=V2EDim)


@field_operator(backend=build_config.backend)
def local_min(
    psi: Field[[Vertex], float_type],
) -> Field[[Vertex], float_type]:
    return minimum(psi, min_over(psi(V2V), axis=V2VDim))


@field_operator(backend=build_config.backend)
def local_max(
    psi: Field[[Vertex], float_type],
) -> Field[[Vertex], float_type]:
    return maximum(psi, max_over(psi(V2V), axis=V2VDim))


# @field_operator(backend=build_config.backend)
# def add_ab2c_vertex(
#    a: Field[[Vertex], float_type],
#    b: Field[[Vertex], float_type],
# ) -> Field[[Vertex], float_type]:
#    return a + b


@field_operator(backend=build_config.backend)
def update_solution(
    rho: Field[[Vertex], float_type],
    flux: Field[[Edge], float_type],
    dt: float_type,
    vol: Field[[Vertex], float_type],
    gac: Field[[Vertex], float_type],
    dual_face_orientation: Field[[Vertex, V2EDim], float_type],
) -> Field[[Vertex], float_type]:
    return rho - dt / (vol * gac) * neighbor_sum(flux(V2E) * dual_face_orientation, axis=V2EDim)


@field_operator(backend=build_config.backend)
def advect_density(
    rho: Field[[Vertex], float_type],
    dt: float_type,
    vol: Field[[Vertex], float_type],
    gac: Field[[Vertex], float_type],
    vel_x: Field[[Vertex], float_type],
    vel_y: Field[[Vertex], float_type],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_orientation: Field[[Vertex, V2EDim], float_type],
    dual_face_normal_weighted_x: Field[[Edge], float_type],
    dual_face_normal_weighted_y: Field[[Edge], float_type],
) -> Field[[Vertex], float_type]:

    veln = advector_normal(
        vel_x,
        vel_y,
        pole_edge_mask,
        dual_face_normal_weighted_x,
        dual_face_normal_weighted_y,
    )

    flux = upwind_flux(rho, veln)
    rho = update_solution(rho, flux, dt, vol, gac, dual_face_orientation)

    cflux = centered_flux(rho, veln)
    cfluxdiv = flux_divergence(cflux, vol, gac, dual_face_orientation)

    pseudoflux = pseudo_flux(rho, veln, gac, cfluxdiv, dt)
    rho = update_solution(rho, pseudoflux, dt, vol, gac, dual_face_orientation)

    return rho


@program(backend=build_config.backend)
def mpdata_program(
    rho0: Field[[Vertex], float_type],
    rho1: Field[[Vertex], float_type],
    dt: float_type,
    vol: Field[[Vertex], float_type],
    gac: Field[[Vertex], float_type],
    vel_x: Field[[Vertex], float_type],
    vel_y: Field[[Vertex], float_type],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_orientation: Field[[Vertex, V2EDim], float_type],
    dual_face_normal_weighted_x: Field[[Edge], float_type],
    dual_face_normal_weighted_y: Field[[Edge], float_type],
    tmp_vertex_0: Field[[Vertex], float_type],
    tmp_vertex_1: Field[[Vertex], float_type],
    tmp_vertex_2: Field[[Vertex], float_type],
    tmp_vertex_3: Field[[Vertex], float_type],
    tmp_edge_0: Field[[Edge], float_type],
    tmp_edge_1: Field[[Edge], float_type],
):

    advector_normal(
        vel_x,
        vel_y,
        pole_edge_mask,
        dual_face_normal_weighted_x,
        dual_face_normal_weighted_y,
        out=tmp_edge_0,
    )
    upwind_flux(rho0, tmp_edge_0, out=tmp_edge_1)
    update_solution(
        rho0,
        tmp_edge_1,
        dt,
        vol,
        gac,
        dual_face_orientation,
        out=tmp_vertex_0
        # rho0,
        # tmp_edge_1,
        # dt,
        # vol,
        # gac,
        # dual_face_orientation,
        # out=rho1,
    )  # out is upwind solution (Vertex)

    local_min(rho0, out=tmp_vertex_2)
    local_max(rho0, out=tmp_vertex_3)

    centered_flux(tmp_vertex_0, tmp_edge_0, out=tmp_edge_1)  # out is centered flux (Edge)
    flux_divergence(
        tmp_edge_1, vol, gac, dual_face_orientation, out=tmp_vertex_1
    )  # out is fluxdiv of centered flux (Vertex)

    pseudo_flux(
        tmp_vertex_0, tmp_edge_0, gac, tmp_vertex_1, dt, out=tmp_edge_1
    )  # out is pseudo flux (Edge)

    # nonoscoefficients(tmp_edge_1, dual_face_orientation, out=())

    update_solution(
        tmp_vertex_0, tmp_edge_1, dt, vol, gac, dual_face_orientation, out=rho1
    )  # out is final solution (Vertex)


@field_operator(backend=build_config.backend)
def upwind_scheme(
    rho: Field[[Vertex], float_type],
    dt: float_type,
    vol: Field[[Vertex], float_type],
    gac: Field[[Vertex], float_type],
    vel_x: Field[[Vertex], float_type],
    vel_y: Field[[Vertex], float_type],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_orientation: Field[[Vertex, V2EDim], float_type],
    dual_face_normal_weighted_x: Field[[Edge], float_type],
    dual_face_normal_weighted_y: Field[[Edge], float_type],
) -> Field[[Vertex], float_type]:

    vn = advector_normal(
        vel_x,
        vel_y,
        pole_edge_mask,
        dual_face_normal_weighted_x,
        dual_face_normal_weighted_y,
    )
    flux = upwind_flux(rho, vn)
    rho = rho - dt / (vol * gac) * neighbor_sum(flux(V2E) * dual_face_orientation, axis=V2EDim)
    return rho
