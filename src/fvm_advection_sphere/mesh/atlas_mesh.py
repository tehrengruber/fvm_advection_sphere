from typing import Final, Iterable, Optional, TypeVar, Union
import numpy as np
import math
import functools
from types import SimpleNamespace

from atlas4py import (
    StructuredGrid,
    Topology,
    Config,
    StructuredMeshGenerator,
    functionspace,
    build_edges,
    build_node_to_edge_connectivity,
    build_node_to_cell_connectivity,
    build_element_to_edge_connectivity,
    build_median_dual_mesh,
    build_periodic_boundaries,
    build_halo,
    build_parallel_fields,
    BlockConnectivity,
)

T = TypeVar("T")
OneOrMore = Union[T, Iterable[T]]

def assert_close(expected, actual):
    assert math.isclose(expected, actual), "expected={}, actual={}".format(
        expected, actual
    )

def _default_config():
    config = Config()
    config["triangulate"] = True
    config["angle"] = 20.0
    return config

rpi: Final[float] = 2.0 * math.asin(1.0)
deg2rad: Final[float] = 2.0 * rpi / 360.0

def _atlas_connectivity_to_numpy(atlas_conn, *, out=None, skip_neighbor_indicator=-1):
    if isinstance(atlas_conn, BlockConnectivity):
        shape = (atlas_conn.rows, atlas_conn.cols)
        out = np.zeros(shape, dtype=np.int64) if out is None else out
        assert out.shape == shape

        for i in range(atlas_conn.rows):
            for nb in range(atlas_conn.cols):
                out[i, nb] = atlas_conn[i, nb]

        return out

    shape = (atlas_conn.rows, atlas_conn.maxcols)
    out = np.zeros(shape, dtype=np.int64) if out is None else out
    assert out.shape == shape

    for i in range(atlas_conn.rows):
        cols = atlas_conn.cols(i)
        for nb in range(cols):
            out[i, nb] = atlas_conn[i, nb]
        out[i, cols:]=-1

    return out

def _build_atlas_mesh(config, grid, periodic_halos=None):
    # compare https://github.com/ckuehnlein/FVM_BESPOKE_NWP/blob/bespoke_nwp/src/fvm/core/datastruct_module.F90#L353
    mesh = StructuredMeshGenerator(config).generate(grid)

    # note: regularly the following calls are done implicitly using
    functionspace.EdgeColumns(mesh, halo=2)
    functionspace.NodeColumns(mesh, halo=2)
    #if periodic_halos:
    #    build_parallel_fields(mesh)
    #    build_periodic_boundaries(mesh)
    #    build_halo(mesh, periodic_halos)

    #build_edges(mesh, config)
    build_node_to_edge_connectivity(mesh)
    build_node_to_cell_connectivity(mesh)
    build_median_dual_mesh(mesh)

    return mesh


def setup_mesh(grid=StructuredGrid("O32"), radius=6371.22e03, config=None):
    if config is None:
        config = Config()
        config["triangulate"] = False
        config["angle"] = -1.0
        config["pole_edges"] = True

    # generate mesh from grid points
    #mesh = _build_atlas_mesh(config, grid)
    mesh = _build_atlas_mesh(config, grid, periodic_halos=2)

    num_cells = mesh.cells.size
    num_edges = mesh.edges.size
    num_vertices = mesh.nodes.size

    # flags
    vertex_flags = np.array(mesh.nodes.flags(), copy=False)
    edge_flags = np.array(mesh.edges.flags(), copy=False)
    cell_flags = np.array(mesh.cells.flags(), copy=False)

    #
    # connectivities
    v2e = _atlas_connectivity_to_numpy(mesh.nodes.edge_connectivity)[0:num_vertices, :]
    v2c = _atlas_connectivity_to_numpy(mesh.nodes.cell_connectivity)[0:num_vertices, :]
    e2v = _atlas_connectivity_to_numpy(mesh.edges.node_connectivity)[0:num_edges, :]
    e2c = _atlas_connectivity_to_numpy(mesh.edges.cell_connectivity)[0:num_edges, :]
    c2v = _atlas_connectivity_to_numpy(mesh.cells.node_connectivity)[0:num_cells, :]
    c2e = _atlas_connectivity_to_numpy(mesh.cells.edge_connectivity)[0:num_cells, :]

    v2e = _atlas_connectivity_to_numpy(mesh.nodes.edge_connectivity)
    v2c = _atlas_connectivity_to_numpy(mesh.nodes.cell_connectivity)
    e2v = _atlas_connectivity_to_numpy(mesh.edges.node_connectivity)
    e2c = _atlas_connectivity_to_numpy(mesh.edges.cell_connectivity)
    c2v = _atlas_connectivity_to_numpy(mesh.cells.node_connectivity)
    c2e = _atlas_connectivity_to_numpy(mesh.cells.edge_connectivity)

    vertex_remote_indices = np.array(mesh.nodes.field("remote_idx"), copy=False)
    edge_remote_indices = np.array(mesh.edges.field("remote_idx"), copy=False)

    assert v2e.shape[0] == num_vertices
    assert v2c.shape[0] == num_vertices
    assert e2v.shape[0] == num_edges
    assert e2c.shape[0] == num_edges
    assert c2v.shape[0] == num_cells
    assert c2e.shape[0] == num_cells

    #
    # geometrical properties
    #
    xydeg = np.array(mesh.nodes.lonlat, copy=False)
    xyrad = np.array(mesh.nodes.lonlat, copy=False) * deg2rad
    xyarc = np.array(mesh.nodes.lonlat, copy=False) * deg2rad * radius
    phi, theta = xyrad[:, 1], xyrad[:, 0]
    xyz = np.stack((np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)), axis=1)

    # face orientation
    edges_per_node = v2e.shape[1]
    dual_face_orientation = np.zeros((num_vertices, edges_per_node)) # formerly known as "sign field"

    def is_pole_edge(e):
        return Topology.check(edge_flags[e], Topology.POLE)

    num_pole_edges = 0
    pole_bc = np.ones(num_edges)
    for e in range(0, num_edges):
        if is_pole_edge(e):
            num_pole_edges += 1
            pole_bc[e] = -1.0

    pole_edges = np.zeros(num_pole_edges, dtype=np.int32)
    inum_pole_edge = -1 
    for e in range(0, num_edges):
        if is_pole_edge(e):
            inum_pole_edge += 1
            pole_edges[inum_pole_edge] = e


    for v in range(0, num_vertices):
        for e_nb in range(0, edges_per_node):
            e = v2e[v, e_nb]
            if v == e2v[e, 0]:
                dual_face_orientation[v, e_nb] = 1.0
            else:
                dual_face_orientation[v, e_nb] = -1.0
                if is_pole_edge(e):
                    dual_face_orientation[v, e_nb] = 1.0

    # dual normal
    dual_face_normal_weighted = np.array(mesh.edges.field("dual_normals"), copy=False) * radius * deg2rad

    # dual volume
    vol = np.array(mesh.nodes.field("dual_volumes"), copy=False) * deg2rad**2 * radius**2

    result = SimpleNamespace(
        num_vertices=num_vertices,
        num_edges=num_edges,
        num_pole_edges=num_pole_edges,
        num_cells=num_cells,
        # connectivities
        c2v=c2v,
        c2e=c2e,
        v2e=v2e,
        v2c=v2c,
        e2v=e2v,
        e2c=e2c,
        # poles
        pole_edges=pole_edges,
        pole_bc=pole_bc,
        edge_remote_indices=edge_remote_indices,
        # flags
        vertex_flags=vertex_flags,
        edge_flags=edge_flags,
        cell_flags=cell_flags,
        # geometry
        radius=radius,
        xydeg=xydeg,
        xyrad=xyrad,
        xyarc=xyarc,
        xyz=xyz,
        vol=vol,
        dual_face_normal_weighted=dual_face_normal_weighted,
        dual_face_orientation=dual_face_orientation,
        # for debugging
        _atlas_mesh=mesh
    )

    return result