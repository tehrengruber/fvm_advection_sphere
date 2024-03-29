from gt4py.next.common import GridType
from gt4py.next.ffront.fbuiltins import (
    Field,
    where,
    neighbor_sum,
    max_over,
    min_over,
    abs,
    maximum,
    minimum,
    int32
)
from gt4py.next.ffront.decorator import field_operator, program

from fvm_advection_sphere.common import *
from fvm_advection_sphere.build_config import float_type
from fvm_advection_sphere import build_config

@field_operator(backend=build_config.backend)
def with_boundary_values(
        lower: Field[[Vertex, K], float_type],
        interior: Field[[Vertex, K], float_type],
        upper: Field[[Vertex, K], float_type],
        level_indices: Field[[K], int32],
        num_level: int32
) -> Field[[Vertex, K], float_type]:
    return where(level_indices == 0, lower, where(level_indices == num_level - 1, upper, interior))

# TODO(tehrengruber): move to seperate file
@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def nabla_z(psi: Field[[Vertex, K], float_type], level_indices: Field[[K], int32], num_level: int32):
    return with_boundary_values(
        psi(Koff[1]) - psi(Koff[0]),
        psi(Koff[1]) - psi(Koff[-1]),
        psi(Koff[0]) - psi(Koff[-1]),
        level_indices, num_level # TODO(tehrengruber): use keyword args when supported
    )


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def advector_in_edges(
    vel_x: Field[[Vertex, K], float_type],
    vel_y: Field[[Vertex, K], float_type],
    pole_edge_mask: Field[[Edge], bool],
) -> tuple[Field[[Edge, K], float_type], Field[[Edge, K], float_type]]:
    pole_bc = where(pole_edge_mask, -1.0, 1.0)
    vel_edges_x = 0.5 * (vel_x(E2V[0]) + pole_bc * vel_x(E2V[1]))
    vel_edges_y = 0.5 * (vel_y(E2V[0]) + pole_bc * vel_y(E2V[1]))
    return vel_edges_x, where(pole_edge_mask, 0.0, vel_edges_y)


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def advector_normal(
    vel_x: Field[[Vertex, K], float_type],
    vel_y: Field[[Vertex, K], float_type],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_normal_weighted_x: Field[[Edge], float_type],
    dual_face_normal_weighted_y: Field[[Edge], float_type],
) -> Field[[Edge, K], float_type]:
    pole_bc = where(pole_edge_mask, -1.0, 1.0)
    vel_edges_x = 0.5 * (vel_x(E2V[0]) + pole_bc * vel_x(E2V[1]))
    vel_edges_y = 0.5 * (vel_y(E2V[0]) + pole_bc * vel_y(E2V[1]))
    vel_edges_y = where(pole_edge_mask, 0.0, vel_edges_y)
    # vel_edges_x = where(pole_edge_mask, 0.0, vel_edges_x)
    return vel_edges_x * dual_face_normal_weighted_x + vel_edges_y * dual_face_normal_weighted_y


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def upstream_flux(
    rho: Field[[Vertex, K], float_type],
    vel_x: Field[[Vertex, K], float_type],
    vel_y: Field[[Vertex, K], float_type],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_normal_weighted_x: Field[[Edge], float_type],
    dual_face_normal_weighted_y: Field[[Edge], float_type],
) -> Field[[Edge, K], float_type]:
    vel_x_face, vel_y_face = advector_in_edges(vel_x, vel_y, pole_edge_mask)
    wnv = vel_x_face * dual_face_normal_weighted_x + vel_y_face * dual_face_normal_weighted_y
    return where(wnv > 0.0, rho(E2V[0]) * wnv, rho(E2V[1]) * wnv)


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def upwind_flux(
    rho: Field[[Vertex, K], float_type],
    veln: Field[[Edge, K], float_type],
) -> Field[[Edge, K], float_type]:
    return where(veln > 0.0, rho(E2V[0]) * veln, rho(E2V[1]) * veln)


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def centered_flux(
    rho: Field[[Vertex, K], float_type],
    veln: Field[[Edge, K], float_type],
) -> Field[[Edge, K], float_type]:
    return (
        0.5 * veln * (rho(E2V[1]) + rho(E2V[0]))
    )  # todo(ckuehnlein): polar flip for u and v transport later


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def pseudo_flux(
    rho: Field[[Vertex, K], float_type],
    veln: Field[[Edge, K], float_type],
    grg: Field[[Vertex], float_type],
    cfluxdiv: Field[[Vertex, K], float_type],
    dt: float_type,
) -> Field[[Edge, K], float_type]:
    return 0.5 * abs(veln) * (rho(E2V[1]) - rho(E2V[0])) - dt * veln * 0.5 * (
        (cfluxdiv(E2V[1]) + cfluxdiv(E2V[0])) / (grg(E2V[1]) + grg(E2V[0]))
    )


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def limit_pseudo_flux(
    flux: Field[[Edge, K], float_type],
    cn: Field[[Vertex, K], float_type],
    cp: Field[[Vertex, K], float_type],
) -> Field[[Edge, K], float_type]:
    # pflux(jlev,jedge) =  max(0._wp,pflux(jlev,jedge))*min(plimit,cp(jlev,ip2),cn(jlev,ip1)) &
    #                    & +min(0._wp,pflux(jlev,jedge))*min(plimit,cn(jlev,ip2),cp(jlev,ip1))
    return maximum(0.0, flux) * minimum(1.0, minimum(cp(E2V[1]), cn(E2V[0]))) + minimum(
        0.0, flux
    ) * minimum(1.0, minimum(cn(E2V[1]), cp(E2V[0])))


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def flux_divergence(
    flux: Field[[Edge, K], float_type],
    vol: Field[[Vertex], float_type],
    gac: Field[[Vertex], float_type],
    dual_face_orientation: Field[[Vertex, V2EDim], float_type],
) -> Field[[Vertex, K], float_type]:
    return 1.0 / (vol * gac) * neighbor_sum(flux(V2E) * dual_face_orientation, axis=V2EDim)


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def nonoscoefficients_cn(
    psimin: Field[[Vertex, K], float_type],
    psi: Field[[Vertex, K], float_type],
    flux: Field[[Edge, K], float_type],
    vol: Field[[Vertex], float_type],
    gac: Field[[Vertex], float_type],
    dt: float_type,
    eps: float_type,
    dual_face_orientation: Field[[Vertex, V2EDim], float_type],
) -> Field[[Vertex, K], float_type]:
    # zrhout(jlev,jnode) = zrhout(jlev,jnode)+zsignp*zpos+zsignn*zneg
    # cn(jlev,jnode)     = (pD(jlev,jnode)-zDmin(jlev,jnode))  &
    #                   & *prho(jlev,jnode)/(zrhout(jlev,jnode)*pdt+eps)
    # zrhout = (1.0 / vol) * neighbor_sum(
    #    maximum(0.0, flux(V2E)) * maximum(0.0, dual_face_orientation)
    #    + minimum(0.0, flux(V2E)) * minimum(0.0, dual_face_orientation),
    #    axis=V2EDim,
    # )
    zrhout = (1.0 / vol) * neighbor_sum(
        (
            maximum(0.0, flux(V2E)) * maximum(0.0, dual_face_orientation)
            + minimum(0.0, flux(V2E)) * minimum(0.0, dual_face_orientation)
        ),
        axis=V2EDim,
    )
    return (psi - psimin) * gac / (zrhout * dt + eps)


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def nonoscoefficients_cp(
    psimax: Field[[Vertex, K], float_type],
    psi: Field[[Vertex, K], float_type],
    flux: Field[[Edge, K], float_type],
    vol: Field[[Vertex], float_type],
    gac: Field[[Vertex], float_type],
    dt: float_type,
    eps: float_type,
    dual_face_orientation: Field[[Vertex, V2EDim], float_type],
) -> Field[[Vertex, K], float_type]:
    # zrhin (jlev,jnode) = zrhin (jlev,jnode)-zsignp*zneg-zsignn*zpos
    # cp(jlev,jnode)     = (pDmax(jlev,jnode)-pD(jlev,jnode))  &
    #                   & *prho(jlev,jnode)/(zrhin(jlev,jnode)*pdt+eps)
    zrhin = (1.0 / vol) * neighbor_sum(
        -minimum(0.0, flux(V2E)) * maximum(0.0, dual_face_orientation)
        - maximum(0.0, flux(V2E)) * minimum(0.0, dual_face_orientation),
        axis=V2EDim,
    )
    return (psimax - psi) * gac / (zrhin * dt + eps)


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def local_min(
    psi: Field[[Vertex, K], float_type],
) -> Field[[Vertex, K], float_type]:
    return minimum(psi, min_over(psi(V2V), axis=V2VDim))


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def local_max(
    psi: Field[[Vertex, K], float_type],
) -> Field[[Vertex, K], float_type]:
    return maximum(psi, max_over(psi(V2V), axis=V2VDim))


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def update_solution(
    rho: Field[[Vertex, K], float_type],
    flux: Field[[Edge, K], float_type],
    dt: float_type,
    vol: Field[[Vertex], float_type],
    gac: Field[[Vertex], float_type],
    dual_face_orientation: Field[[Vertex, V2EDim], float_type],
) -> Field[[Vertex, K], float_type]:
    return rho - dt / (vol * gac) * neighbor_sum(flux(V2E) * dual_face_orientation, axis=V2EDim)


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def advect_density(
    rho: Field[[Vertex, K], float_type],
    dt: float_type,
    vol: Field[[Vertex], float_type],
    gac: Field[[Vertex], float_type],
    vel_x: Field[[Vertex, K], float_type],
    vel_y: Field[[Vertex, K], float_type],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_orientation: Field[[Vertex, V2EDim], float_type],
    dual_face_normal_weighted_x: Field[[Edge], float_type],
    dual_face_normal_weighted_y: Field[[Edge], float_type],
) -> Field[[Vertex, K], float_type]:

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


@program(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def mpdata_program(
    rho0: Field[[Vertex, K], float_type],
    rho1: Field[[Vertex, K], float_type],
    dt: float_type,
    eps: float_type,
    vol: Field[[Vertex], float_type],
    gac: Field[[Vertex], float_type],
    vel_x: Field[[Vertex, K], float_type],
    vel_y: Field[[Vertex, K], float_type],
    vel_z: Field[[Vertex, K], float_type],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_orientation: Field[[Vertex, V2EDim], float_type],
    dual_face_normal_weighted_x: Field[[Edge], float_type],
    dual_face_normal_weighted_y: Field[[Edge], float_type],
    tmp_vertex_0: Field[[Vertex, K], float_type],
    tmp_vertex_1: Field[[Vertex, K], float_type],
    tmp_vertex_2: Field[[Vertex, K], float_type],
    tmp_vertex_3: Field[[Vertex, K], float_type],
    tmp_vertex_4: Field[[Vertex, K], float_type],
    tmp_vertex_5: Field[[Vertex, K], float_type],
    tmp_edge_0: Field[[Edge, K], float_type],
    tmp_edge_1: Field[[Edge, K], float_type],
    tmp_edge_2: Field[[Edge, K], float_type],
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

    local_min(rho0, out=tmp_vertex_2)  # out is local min
    local_max(rho0, out=tmp_vertex_3)  # out is local max

    centered_flux(tmp_vertex_0, tmp_edge_0, out=tmp_edge_1)  # out is centered flux (Edge)
    flux_divergence(
        tmp_edge_1, vol, gac, dual_face_orientation, out=tmp_vertex_1
    )  # out is fluxdiv of centered flux (Vertex)

    pseudo_flux(
        tmp_vertex_0, tmp_edge_0, gac, tmp_vertex_1, dt, out=tmp_edge_1
    )  # out is pseudo flux (Edge)

    nonoscoefficients_cn(
        tmp_vertex_2,
        tmp_vertex_0,
        tmp_edge_1,
        vol,
        gac,
        dt,
        eps,
        dual_face_orientation,
        out=tmp_vertex_4,  # out is cn nonos coefficient
    )
    nonoscoefficients_cp(
        tmp_vertex_3,
        tmp_vertex_0,
        tmp_edge_1,
        vol,
        gac,
        dt,
        eps,
        dual_face_orientation,
        out=tmp_vertex_5,  # out is cp nonos coefficient
    )

    limit_pseudo_flux(
        tmp_edge_1, tmp_vertex_4, tmp_vertex_5, out=tmp_edge_2
    )  # out is limited pseudo flux (Edge)

    # todo(ckuehnlein): tmp_edge_2 must be used in case of nonos=True
    update_solution(
        tmp_vertex_0,
        tmp_edge_2,  # tmp_edge_1 without fct, tmp_edge_2 with fct
        dt,
        vol,
        gac,
        dual_face_orientation,
        out=rho1,
    )  # out is final solution (Vertex)


@field_operator(backend=build_config.backend, grid_type=GridType.UNSTRUCTURED)
def upwind_scheme(
    rho: Field[[Vertex, K], float_type],
    dt: float_type,
    vol: Field[[Vertex], float_type],
    gac: Field[[Vertex], float_type],
    vel_x: Field[[Vertex, K], float_type],
    vel_y: Field[[Vertex, K], float_type],
    vel_z: Field[[Vertex, K], float_type],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_orientation: Field[[Vertex, V2EDim], float_type],
    dual_face_normal_weighted_x: Field[[Edge], float_type],
    dual_face_normal_weighted_y: Field[[Edge], float_type],
) -> Field[[Vertex, K], float_type]:

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

