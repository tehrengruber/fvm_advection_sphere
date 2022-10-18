import numpy as np

from functional.ffront.fbuiltins import Field, float64, where, neighbor_sum
from functional.ffront.decorator import field_operator, program
from functional.iterator.embedded import np_as_located_field

from fvm_advection_sphere.common import *
from fvm_advection_sphere import build_config
from fvm_advection_sphere.mesh.atlas_mesh import AtlasMesh


@field_operator
def advector_in_edges(
    vel_x: Field[[Vertex], float64],
    vel_y: Field[[Vertex], float64],
    pole_edge_mask: Field[[Edge], bool]
) -> tuple[Field[[Edge], float64], Field[[Edge], float64]]:
    """
    Interpolate velocity from vertices to edges.
    """
    pole_bc = where(pole_edge_mask, -1., 1.)
    vel_edges_x = 0.5 * (vel_x(E2V[0]) + pole_bc * vel_x(E2V[1]))
    vel_edges_y = 0.5 * (vel_y(E2V[0]) + pole_bc * vel_y(E2V[1]))
    return vel_edges_x, where(pole_edge_mask, 0., vel_edges_y)


@field_operator
def upstream_flux(
    rho : Field[[Vertex], float64],
    vel_x: Field[[Vertex], float64],
    vel_y: Field[[Vertex], float64],
    pole_edge_mask: Field[[Edge], bool],
    dual_face_normal_weighted_x: Field[[Edge], float64],
    dual_face_normal_weighted_y: Field[[Edge], float64]
) -> Field[[Edge], float64]:
    """
    Compute flux density using an upwind scheme.
    """
    vel_x_face, vel_y_face = advector_in_edges(vel_x, vel_y, pole_edge_mask)
    # weighted normal velocity
    wnv = vel_x_face*dual_face_normal_weighted_x + vel_y_face*dual_face_normal_weighted_y
    return where(wnv > 0.0, rho(E2V[0]) * wnv, rho(E2V[1]) * wnv)


@field_operator(backend=build_config.backend)
def fvm_advect(
        rho: Field[[Vertex], float64],
        dt: float64,
        vol: Field[[Vertex], float64],
        gac: Field[[Vertex], float64],
        vel_x: Field[[Vertex], float64],
        vel_y: Field[[Vertex], float64],
        pole_edge_mask: Field[[Edge], bool],
        dual_face_orientation: Field[[Vertex, V2EDim], float64],
        dual_face_normal_weighted_x: Field[[Edge], float64],
        dual_face_normal_weighted_y: Field[[Edge], float64]
) -> Field[[Vertex], float64]:
    """
    Compute density at t+dt
    """
    flux = upstream_flux(
        rho,
        vel_x,
        vel_y,
        pole_edge_mask,
        dual_face_normal_weighted_x,
        dual_face_normal_weighted_y
    )
    return rho - dt / (vol*gac) * neighbor_sum(flux(V2E) * dual_face_orientation, axis=V2EDim)