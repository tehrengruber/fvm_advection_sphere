import numpy as np
import types

from functional.ffront.fbuiltins import Field, float64, where, neighbor_sum
from functional.ffront.decorator import field_operator, program
from functional.iterator.embedded import np_as_located_field

from fvm_advection_sphere.common import *
from fvm_advection_sphere.mesh.atlas_mesh import AtlasMesh

@field_operator
def advector_in_edges(
        vel_x: Field[[Vertex], float64],
        vel_y: Field[[Vertex], float64],
        pole_edge_mask: Field[[Edge], bool]
) -> tuple[Field[[Edge], float64], Field[[Edge], float64]]:
    """
    Interpolate velocity from vertices onto edges.
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
    vel_x_face, vel_y_face = advector_in_edges(vel_x, vel_y, pole_edge_mask)
    # weighted normal velocity
    wnv = vel_x_face*dual_face_normal_weighted_x + vel_y_face*dual_face_normal_weighted_y
    return where(wnv > 0.0, rho(E2V[0]) * wnv, rho(E2V[1]) * wnv)

from functional.ffront.fbuiltins import Dimension, FieldOffset
from functional.ffront.gtcallable import GTCallable
import functional.ffront.itir_makers as im
import functional.ffront.common_types as ct
import functional.iterator.ir as itir
import dataclasses

@dataclasses.dataclass(frozen=True)
class AsNeighborField(GTCallable):
    name: str
    source: Dimension
    target: tuple[Dimension, Dimension]
    offset: FieldOffset
    dtype: ct.ScalarType


from functional.fencil_processors.runners import gtfn_cpu

@field_operator(backend=gtfn_cpu.run_gtfn)
def fvm_advect_(
        rho: Field[[Vertex], float64],
        #dt: float64,
        vol: Field[[Vertex], float64],
        gac: Field[[Vertex], float64],
        vel_x: Field[[Vertex], float64],
        vel_y: Field[[Vertex], float64],
        pole_edge_mask: Field[[Edge], bool],
        dual_face_orientation_flat: Field[[VertexEdgeNb], float64],
        #dual_face_orientation: Field[[Vertex, V2EDim], float64],
        dual_face_normal_weighted_x: Field[[Edge], float64],
        dual_face_normal_weighted_y: Field[[Edge], float64]
) -> Field[[Vertex], float64]:
    dt = 0.1
    dual_face_orientation = as_vertex_v2e_field(dual_face_orientation_flat)
    flux = upstream_flux(
        rho,
        vel_x,
        vel_y,
        pole_edge_mask,
        dual_face_normal_weighted_x,
        dual_face_normal_weighted_y
    )
    return rho - dt / (vol*gac) * neighbor_sum(flux(V2E) * dual_face_orientation, axis=V2EDim)

def fvm_advect(
        mesh: AtlasMesh,
        rho: Field[[Vertex], float],  # field on vertices
        gac: np.ndarray,  # field on vertices
        vel: np.ndarray,  # 2d-vector on edges
        *,
        δt: float,
        offset_provider
):
    rho_next = np_as_located_field(Vertex)(np.zeros(mesh.num_vertices))
    vol_ = mesh.vol
    gac_ = np_as_located_field(Vertex)(gac)
    vel_x = np_as_located_field(Vertex)(vel[:, 0])
    vel_y = np_as_located_field(Vertex)(vel[:, 1])

    #flux_t = np_as_located_field(Edge)(np.zeros(mesh.num_edges))
    #upstream_flux_(rho_, vel_x, vel_y, dual_face_normal_weighted_x, dual_face_normal_weighted_y, out=flux_t, offset_provider=offset_provider)
    #flux2 = upstream_flux(mesh, rho, vel)
    #assert np.allclose(flux_t.array(), flux2)

    #fluxdiv_t = np_as_located_field(Vertex)(np.zeros(mesh.num_vertices))
    #fluxdiv_(rho_, dt, vol_, gac_, flux_t, dual_face_orientation, out=fluxdiv_t,
    #               offset_provider=offset_provider)
    #fluxdiv_ref = fluxdiv_np(mesh, rho, gac, flux2, δt=δt)

    fvm_advect_(rho,
                #δt,
                vol_, gac_, vel_x, vel_y,
                mesh.pole_edge_mask,
                #mesh.dual_face_orientation,
                mesh.dual_face_orientation_flat,
                mesh.dual_face_normal_weighted_x,
                mesh.dual_face_normal_weighted_y,
                out=rho_next,
                offset_provider=offset_provider)

    #rho_ref = fvm_advect_np(mesh, rho, gac, vel, δt=δt)

    return rho_next

@program
def _advector_in_edges_prg(
    vel_x: Field[[Vertex], float64],
    vel_y: Field[[Vertex], float64],
    pole_edge_mask: Field[[Edge], bool],
    vel_out_x: Field[[Edge], float64],
    vel_out_y: Field[[Edge], float64],
):
    advector_in_edges(vel_x, vel_y, pole_edge_mask, out=(vel_out_x, vel_out_y))

def advector_in_edges_np(
    mesh: AtlasMesh,
    vel_vertices,
    offset_provider
):
    vel_edges = np.zeros((mesh.num_edges, 2))

    for e in range(0, mesh.num_edges):
        v1, v2 = mesh.e2v_np[e,:]
        vel_edges[e,0] = 0.5 * (vel_vertices[v1,0] + mesh.pole_bc[e]*vel_vertices[v2,0])
        vel_edges[e,1] = 0.5 * (vel_vertices[v1,1] + mesh.pole_bc[e]*vel_vertices[v2,1])

    for e in mesh.pole_edges:
        vel_edges[e,1] = 0.0

    return vel_edges

def _advector_in_edges(
    mesh: AtlasMesh,
    vel_vertices,
    offset_provider
):
    vel_edges = np.zeros((mesh.num_edges, 2))

    vel_edges_x = np_as_located_field(Edge)(vel_edges[:, 0])
    vel_edges_y = np_as_located_field(Edge)(vel_edges[:, 1])

    _advector_in_edges_prg(
        np_as_located_field(Vertex)(vel_vertices[:, 0]),
        np_as_located_field(Vertex)(vel_vertices[:, 1]),
        mesh.pole_edge_mask,
        vel_edges_x,
        vel_edges_y,
        offset_provider=offset_provider
    )

    vel_edges_np = advector_in_edges_np(mesh, vel_vertices, offset_provider)
    assert np.allclose(vel_edges_x, vel_edges_np[:, 0])
    assert np.allclose(vel_edges_y, vel_edges_np[:, 1])

    return vel_edges