from functional.ffront.fbuiltins import Field, float64, where, neighbor_sum, abs
from functional.ffront.decorator import field_operator

from fvm_advection_sphere.common import *
from fvm_advection_sphere import build_config


@field_operator
def advector_in_edges(
    vel_x: Field[[Vertex], float64],
    vel_y: Field[[Vertex], float64],
    pole_edge_mask: Field[[Edge], bool],
) -> tuple[Field[[Edge], float64], Field[[Edge], float64]]:
    """
    Interpolate advective velocity from vertices to edges.
    """
    pole_bc = where(pole_edge_mask, -1.0, 1.0)
    vel_edges_x = 0.5 * (vel_x(E2V[0]) + pole_bc * vel_x(E2V[1]))
    vel_edges_y = 0.5 * (vel_y(E2V[0]) + pole_bc * vel_y(E2V[1]))
    return vel_edges_x, where(pole_edge_mask, 0.0, vel_edges_y)


@field_operator
def advector_normal(
    vel_x: Field[[Vertex], float64],
    vel_y: Field[[Vertex], float64],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_normal_weighted_x: Field[[Edge], float64],
    dual_face_normal_weighted_y: Field[[Edge], float64],
) -> Field[[Edge], float64]:
    pole_bc = where(pole_edge_mask, -1.0, 1.0)
    vel_edges_x = 0.5 * (vel_x(E2V[0]) + pole_bc * vel_x(E2V[1]))
    vel_edges_y = 0.5 * (vel_y(E2V[0]) + pole_bc * vel_y(E2V[1]))
    vel_edges_y = where(pole_edge_mask, 0.0, vel_edges_y)
    return vel_edges_x * dual_face_normal_weighted_x + vel_edges_y * dual_face_normal_weighted_y


@field_operator
def upstream_flux(
    rho: Field[[Vertex], float64],
    vel_x: Field[[Vertex], float64],
    vel_y: Field[[Vertex], float64],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_normal_weighted_x: Field[[Edge], float64],
    dual_face_normal_weighted_y: Field[[Edge], float64],
) -> Field[[Edge], float64]:
    """
    Compute advective upwind flux.
    """
    vel_x_face, vel_y_face = advector_in_edges(vel_x, vel_y, pole_edge_mask)
    # weighted normal velocity
    wnv = vel_x_face * dual_face_normal_weighted_x + vel_y_face * dual_face_normal_weighted_y
    return where(wnv > 0.0, rho(E2V[0]) * wnv, rho(E2V[1]) * wnv)


@field_operator
def upwind_flux(
    rho: Field[[Vertex], float64],
    veln: Field[[Edge], float64],
) -> Field[[Edge], float64]:
    return where(veln > 0.0, rho(E2V[0]) * veln, rho(E2V[1]) * veln)


@field_operator
def centered_flux(
    rho: Field[[Vertex], float64],
    veln: Field[[Edge], float64],
) -> Field[[Edge], float64]:
    return (
        0.5 * veln * (rho(E2V[1]) + rho(E2V[0]))
    )  # todo(ckuehnlein): polar flip for u and v transport later


# @field_operator(backend=build_config.backend)
# def centered_flux_divergence(
#    rho: Field[[Vertex], float64],
#    vol: Field[[Vertex], float64],
#    gac: Field[[Vertex], float64],
#    vel_x: Field[[Vertex], float64],
#    vel_y: Field[[Vertex], float64],
#    pole_edge_mask: Field[[Edge], bool],
#    dual_face_orientation: Field[[Vertex, V2EDim], float64],
#    dual_face_normal_weighted_x: Field[[Edge], float64],
#    dual_face_normal_weighted_y: Field[[Edge], float64],
# ) -> Field[[Vertex], float64]:
#
#    flux = centered_flux(
#        rho, vel_x, vel_y, pole_edge_mask, dual_face_normal_weighted_x, dual_face_normal_weighted_y
#    )
#
#    flux_divergence = (
#        1.0 / (vol * gac) * neighbor_sum(flux(V2E) * dual_face_orientation, axis=V2EDim)
#    )
#    return flux_divergence


@field_operator
def pseudo_flux(
    rho: Field[[Vertex], float64],
    veln: Field[[Edge], float64],
) -> Field[[Edge], float64]:
    return 0.5 * abs(veln) * (rho(E2V[1]) - rho(E2V[0]))


@field_operator(backend=build_config.backend)
def update_solution(
    rho: Field[[Vertex], float64],
    flux: Field[[Edge], float64],
    dt: float64,
    vol: Field[[Vertex], float64],
    gac: Field[[Vertex], float64],
    dual_face_orientation: Field[[Vertex, V2EDim], float64],
) -> Field[[Vertex], float64]:
    return rho - dt / (vol * gac) * neighbor_sum(flux(V2E) * dual_face_orientation, axis=V2EDim)


@field_operator(backend=build_config.backend)
def advect_density(
    rho: Field[[Vertex], float64],
    dt: float64,
    vol: Field[[Vertex], float64],
    gac: Field[[Vertex], float64],
    vel_x: Field[[Vertex], float64],
    vel_y: Field[[Vertex], float64],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_orientation: Field[[Vertex, V2EDim], float64],
    dual_face_normal_weighted_x: Field[[Edge], float64],
    dual_face_normal_weighted_y: Field[[Edge], float64],
) -> Field[[Vertex], float64]:

    veln = advector_normal(
        vel_x,
        vel_y,
        pole_edge_mask,
        dual_face_normal_weighted_x,
        dual_face_normal_weighted_y,
    )

    flux = upwind_flux(rho, veln)
    rho = update_solution(rho, flux, dt, vol, gac, dual_face_orientation)

    # flux = upwind_flux(rho, veln)
    # rho = update_solution(rho, flux, dt, vol, gac, dual_face_orientation)

    pseudoflux = pseudo_flux(rho, veln)
    rho = update_solution(rho, pseudoflux, dt, vol, gac, dual_face_orientation)
    # rho2 = rho1 - dt / (vol * gac) * neighbor_sum(
    #    pseudoflux(V2E) * dual_face_orientation, axis=V2EDim
    # )

    return rho


@field_operator(backend=build_config.backend)
def upwind_scheme(
    rho: Field[[Vertex], float64],
    dt: float64,
    vol: Field[[Vertex], float64],
    gac: Field[[Vertex], float64],
    vel_x: Field[[Vertex], float64],
    vel_y: Field[[Vertex], float64],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_orientation: Field[[Vertex, V2EDim], float64],
    dual_face_normal_weighted_x: Field[[Edge], float64],
    dual_face_normal_weighted_y: Field[[Edge], float64],
) -> Field[[Vertex], float64]:

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


# @field_operator(backend=build_config.backend)
# def mpdata_scheme(
#    rho: Field[[Vertex], float64],
#    dt: float64,
#    vol: Field[[Vertex], float64],
#    gac: Field[[Vertex], float64],
#    vel_x: Field[[Vertex], float64],
#    vel_y: Field[[Vertex], float64],
#    pole_edge_mask: Field[[Edge], bool],
#    dual_face_orientation: Field[[Vertex, V2EDim], float64],
#    dual_face_normal_weighted_x: Field[[Edge], float64],
#    dual_face_normal_weighted_y: Field[[Edge], float64],
# ) -> Field[[Vertex], float64]:
#    """
#    Compute density at t+dt
#    """
#    pseudoflux = pseudo_flux(
#        rho,
#        vel_x,
#        vel_y,
#        pole_edge_mask,
#        dual_face_normal_weighted_x,
#        dual_face_normal_weighted_y,
#    )
#    rho = rho - dt / (vol * gac) * neighbor_sum(
#        pseudoflux(V2E) * dual_face_orientation, axis=V2EDim
#    )
#
#    return rho


# @field_operator(backend=build_config.backend)
# def flux_divergence(
#    rho: Field[[Vertex], float64],
#    dt: float64,
#    vol: Field[[Vertex], float64],
#    gac: Field[[Vertex], float64],
#    vel_x: Field[[Vertex], float64],
#    vel_y: Field[[Vertex], float64],
#    pole_edge_mask: Field[[Edge], bool],
#    dual_face_orientation: Field[[Vertex, V2EDim], float64],
#    dual_face_normal_weighted_x: Field[[Edge], float64],
#    dual_face_normal_weighted_y: Field[[Edge], float64],
# ) -> Field[[Vertex], float64]:
#
#    flux = centered_flux(
#        rho, vel_x, vel_y, pole_edge_mask, dual_face_normal_weighted_x, dual_face_normal_weighted_y
#    )
#    rho_update = rho - dt / (vol * gac) * neighbor_sum(
#        flux(V2E) * dual_face_orientation, axis=V2EDim
#    )
#    return rho_update
