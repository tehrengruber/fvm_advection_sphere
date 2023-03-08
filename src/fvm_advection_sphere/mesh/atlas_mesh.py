from typing import Final, Any
import numpy as np
import math
import dataclasses
import textwrap

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

from fvm_advection_sphere.common import dtype, Cell, Edge, Vertex, K, V2EDim, VertexEdgeNb, E2VDim

from functional.common import Dimension, Field, Connectivity, DimensionKind
from functional.iterator.embedded import NeighborTableOffsetProvider, np_as_located_field

rpi: Final[float] = 2.0 * math.asin(1.0)
deg2rad: Final[float] = 2.0 * rpi / 360.0

DIMENSION_TO_SIZE_ATTR: dict[Dimension, str] = {
    Vertex: "num_vertices",
    Edge: "num_edges",
    Cell: "num_cells",
}


def _atlas_connectivity_to_numpy(atlas_conn, *, out=None, skip_neighbor_indicator=-1):
    if isinstance(atlas_conn, BlockConnectivity):
        shape = (atlas_conn.rows, atlas_conn.cols)
        out = np.zeros(shape, dtype=np.int32) if out is None else out
        assert out.shape == shape

        for i in range(atlas_conn.rows):
            for nb in range(atlas_conn.cols):
                out[i, nb] = atlas_conn[i, nb]

        return out

    shape = (atlas_conn.rows, atlas_conn.maxcols)
    out = np.zeros(shape, dtype=np.int32) if out is None else out
    assert out.shape == shape

    for i in range(atlas_conn.rows):
        cols = atlas_conn.cols(i)
        for nb in range(cols):
            out[i, nb] = atlas_conn[i, nb]
        out[i, cols:] = -1

    return out


def _build_atlas_mesh(config, grid, periodic_halos=None):
    # compare https://github.com/ckuehnlein/FVM_BESPOKE_NWP/blob/bespoke_nwp/src/fvm/core/datastruct_module.F90#L353
    mesh = StructuredMeshGenerator(config).generate(grid)

    # note: regularly the following calls are done implicitly using
    functionspace.EdgeColumns(mesh, halo=periodic_halos)
    functionspace.NodeColumns(mesh, halo=periodic_halos)
    # if periodic_halos:
    #    build_parallel_fields(mesh)
    #    build_periodic_boundaries(mesh)
    #    build_halo(mesh, periodic_halos)

    # build_edges(mesh, config)
    build_node_to_edge_connectivity(mesh)
    build_node_to_cell_connectivity(mesh)
    build_median_dual_mesh(mesh)

    return mesh


@dataclasses.dataclass
class AtlasMesh:
    num_vertices: int
    num_edges: int
    num_cells: int
    num_pole_edges: int
    nb_vertices_ghost: int
    nb_vertices_noghost: int

    # connectivities
    c2v: Connectivity
    c2e: Connectivity
    v2e: Connectivity
    v2c: Connectivity
    v2v: Connectivity
    e2v: Connectivity
    e2c: Connectivity
    v2ve: Connectivity

    c2v_np: np.ndarray
    c2e_np: np.ndarray
    v2e_np: np.ndarray
    v2c_np: np.ndarray
    v2v_np: np.ndarray
    e2v_np: np.ndarray
    e2c_np: np.ndarray

    # poles
    pole_edges: np.ndarray  # list of all pole edges
    pole_edge_mask: Field[[Edge], bool]
    pole_edge_mask_np: np.ndarray

    # remote indices: for each geometric entity it's remote index
    remote_indices: dict[Dimension, np.ndarray]

    # flags
    vertex_flags: np.ndarray
    edge_flags: np.ndarray
    cell_flags: np.ndarray

    vertex_ghost_mask: Field[[Vertex], bool]

    # geometry
    radius: dtype
    xydeg_x: Field[[Vertex], dtype]
    xydeg_y: Field[[Vertex], dtype]
    xydeg_np: np.ndarray
    xyrad: np.ndarray
    xyarc: np.ndarray
    xyz: np.ndarray

    vol: Field[[Edge], dtype]
    vol_np: np.ndarray

    dual_face_normal_weighted_x: Field[[Edge], dtype]
    dual_face_normal_weighted_y: Field[[Edge], dtype]
    dual_face_normal_weighted_np: np.ndarray

    dual_face_orientation: Field[[Vertex, V2EDim], dtype]
    dual_face_orientation_flat: Field[[VertexEdgeNb], dtype]
    dual_face_orientation_np: np.ndarray

    offset_provider: dict[str, Connectivity | Dimension]

    grid_description: str  # string representation of the atlas grid instance

    _atlas_mesh: Any  # for debugging

    def info(self):
        n = math.ceil(math.log(self.num_edges + 1, 10))

        return textwrap.dedent(
            f"""
        Atlas mesh
          grid:     {self.grid_description}
          vertices: {str(self.num_vertices).rjust(n)}  (ghost: {str(self.nb_vertices_ghost).rjust(n)}, noghost: {str(self.nb_vertices_noghost).rjust(n)})
          edges:    {str(self.num_edges).rjust(n)}
          cells:    {str(self.num_cells).rjust(n)}
        """
        )

    @classmethod
    def generate(cls, grid=StructuredGrid("O32"), radius=6371.22e03, config=None) -> "AtlasMesh":
        if config is None:
            config = Config()
            config["triangulate"] = False
            config["angle"] = -1.0
            config["pole_edges"] = True
            config["ghost_at_end"] = True

        # generate mesh from grid points
        # mesh = _build_atlas_mesh(config, grid)
        mesh = _build_atlas_mesh(config, grid, periodic_halos=10)

        num_cells = mesh.cells.size
        num_edges = mesh.edges.size
        num_vertices = mesh.nodes.size

        # flags
        vertex_flags = np.array(mesh.nodes.flags(), copy=False)
        edge_flags = np.array(mesh.edges.flags(), copy=False)
        cell_flags = np.array(mesh.cells.flags(), copy=False)

        vertex_ghost_mask = np_as_located_field(Vertex)(
            (vertex_flags & Topology.GHOST).astype(bool)
        )

        nb_vertices_ghost = np.sum(np.where(vertex_ghost_mask, 1, 0), dtype=int)
        nb_vertices_noghost = num_vertices - nb_vertices_ghost
        assert nb_vertices_noghost == np.sum(grid.nx)

        #
        # connectivities
        v2e_np = _atlas_connectivity_to_numpy(mesh.nodes.edge_connectivity)
        v2c_np = _atlas_connectivity_to_numpy(mesh.nodes.cell_connectivity)
        v2v_np = np.zeros(v2e_np.shape, dtype=np.int32)  # initialized further below
        e2v_np = _atlas_connectivity_to_numpy(mesh.edges.node_connectivity)
        e2c_np = _atlas_connectivity_to_numpy(mesh.edges.cell_connectivity)
        c2v_np = _atlas_connectivity_to_numpy(mesh.cells.node_connectivity)
        c2e_np = _atlas_connectivity_to_numpy(mesh.cells.edge_connectivity)

        assert v2e_np.shape[0] == num_vertices
        assert v2c_np.shape[0] == num_vertices
        assert e2v_np.shape[0] == num_edges
        assert e2c_np.shape[0] == num_edges
        assert c2v_np.shape[0] == num_cells
        assert c2e_np.shape[0] == num_cells

        v2e = NeighborTableOffsetProvider(v2e_np, Vertex, Edge, v2e_np.shape[1])
        v2c = NeighborTableOffsetProvider(v2c_np, Vertex, Cell, v2c_np.shape[1])
        v2v = NeighborTableOffsetProvider(v2v_np, Vertex, Vertex, v2v_np.shape[1])
        e2v = NeighborTableOffsetProvider(e2v_np, Edge, Vertex, e2v_np.shape[1])
        e2c = NeighborTableOffsetProvider(e2c_np, Edge, Cell, e2c_np.shape[1])
        c2v = NeighborTableOffsetProvider(c2v_np, Cell, Vertex, c2v_np.shape[1])
        c2e = NeighborTableOffsetProvider(c2e_np, Cell, Edge, c2e_np.shape[1])
        v2ve = NeighborTableOffsetProvider(
            np.reshape(
                np.arange(0, num_vertices * v2e.max_neighbors, 1, dtype=np.int32),
                (num_vertices, v2e.max_neighbors),
            ),
            Vertex,
            VertexEdgeNb,
            v2e.max_neighbors,
        )

        vertex_remote_indices = np.array(mesh.nodes.field("remote_idx"), copy=False)
        edge_remote_indices = np.array(mesh.edges.field("remote_idx"), copy=False)
        cell_remote_indices = np.array(mesh.cells.field("remote_idx"), copy=False)

        #
        # geometrical properties
        #
        xydeg_np = np.array(mesh.nodes.lonlat, copy=False)
        xydeg_x = np_as_located_field(Vertex)(xydeg_np[:, 0])
        xydeg_y = np_as_located_field(Vertex)(xydeg_np[:, 1])
        xyrad = np.array(mesh.nodes.lonlat, copy=False) * deg2rad
        xyarc = np.array(mesh.nodes.lonlat, copy=False) * deg2rad * radius
        phi, theta = xyrad[:, 1], xyrad[:, 0]
        xyz = np.stack(
            (np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)), axis=1
        )

        # face orientation
        edges_per_node = v2e_np.shape[1]
        dual_face_orientation_np = np.zeros(
            (num_vertices, edges_per_node)
        )  # formerly known as "sign field"

        def is_pole_edge(e):
            return Topology.check(edge_flags[e], Topology.POLE)

        num_pole_edges = 0
        pole_edge_mask_np = np.zeros(num_edges, dtype=bool)
        for e in range(0, num_edges):
            if is_pole_edge(e):
                num_pole_edges += 1
                pole_edge_mask_np[e] = True
        pole_edge_mask = np_as_located_field(Edge)(pole_edge_mask_np)

        pole_edges = np.zeros(num_pole_edges, dtype=np.int32)
        inum_pole_edge = -1
        for e in range(0, num_edges):
            if is_pole_edge(e):
                inum_pole_edge += 1
                pole_edges[inum_pole_edge] = e

        dual_face_orientation_flat = np.zeros(num_vertices * edges_per_node)
        for v in range(0, num_vertices):
            for e_nb in range(0, edges_per_node):
                e = v2e_np[v, e_nb]
                if e != -1:
                    if v == e2v_np[e, 0]:
                        dual_face_orientation_np[v, e_nb] = 1.0
                        v2v_np[v, e_nb] = e2v_np[e, 1]
                    else:
                        dual_face_orientation_np[v, e_nb] = -1.0
                        v2v_np[v, e_nb] = e2v_np[e, 0]
                        if is_pole_edge(e):
                            dual_face_orientation_np[v, e_nb] = 1.0
                else:
                    dual_face_orientation_np[v, e_nb] = np.nan
                    v2v_np[v, e_nb] = -1
                dual_face_orientation_flat[edges_per_node * v + e_nb] = dual_face_orientation_np[
                    v, e_nb
                ]
        dual_face_orientation = np_as_located_field(Vertex, V2EDim)(dual_face_orientation_np)
        dual_face_orientation_flat = np_as_located_field(VertexEdgeNb)(dual_face_orientation_flat)

        # dual normal
        dual_face_normal_weighted_np = (
            np.array(mesh.edges.field("dual_normals"), copy=False) * radius * deg2rad
        )
        dual_face_normal_weighted_x = np_as_located_field(Edge)(dual_face_normal_weighted_np[:, 0])
        dual_face_normal_weighted_y = np_as_located_field(Edge)(dual_face_normal_weighted_np[:, 1])

        # dual volume
        vol_np = np.array(mesh.nodes.field("dual_volumes"), copy=False) * deg2rad**2 * radius**2
        vol = np_as_located_field(Vertex)(vol_np)

        # offset provider
        offset_provider = {
            "V2V": v2v,
            "V2E": v2e,
            "E2V": e2v,
            "V2EDim": V2EDim,
            "E2VDim": E2VDim,
        }

        return cls(
            num_vertices=num_vertices,
            num_edges=num_edges,
            num_pole_edges=num_pole_edges,
            num_cells=num_cells,
            nb_vertices_ghost=nb_vertices_ghost,
            nb_vertices_noghost=nb_vertices_noghost,
            # connectivities
            v2v=v2v,
            v2v_np=v2v_np,
            v2e=v2e,
            v2e_np=v2e_np,
            v2c=v2c,
            v2c_np=v2c_np,
            e2v=e2v,
            e2v_np=e2v_np,
            c2v=c2v,
            c2v_np=c2v_np,
            c2e=c2e,
            c2e_np=c2e_np,
            e2c=e2c,
            e2c_np=e2c_np,
            v2ve=v2ve,
            # poles
            pole_edge_mask=pole_edge_mask,
            pole_edge_mask_np=pole_edge_mask_np,
            pole_edges=pole_edges,
            remote_indices={
                Vertex: vertex_remote_indices,
                Edge: edge_remote_indices,
                Cell: cell_remote_indices,
            },
            # flags
            vertex_flags=vertex_flags,
            edge_flags=edge_flags,
            cell_flags=cell_flags,
            vertex_ghost_mask=vertex_ghost_mask,
            # geometry
            radius=radius,
            xydeg_x=xydeg_x,
            xydeg_y=xydeg_y,
            xydeg_np=xydeg_np,
            xyrad=xyrad,
            xyarc=xyarc,
            xyz=xyz,
            vol=vol,
            vol_np=vol_np,
            dual_face_normal_weighted_np=dual_face_normal_weighted_np,
            dual_face_normal_weighted_x=dual_face_normal_weighted_x,
            dual_face_normal_weighted_y=dual_face_normal_weighted_y,
            dual_face_orientation_np=dual_face_orientation_np,
            dual_face_orientation=dual_face_orientation,
            dual_face_orientation_flat=dual_face_orientation_flat,
            offset_provider=offset_provider,
            grid_description=str(grid),
            # for debugging
            _atlas_mesh=mesh,
        )


def update_periodic_layers(mesh: AtlasMesh, field: Field):
    # todo: generalize to other dimensions
    horizontal_dimension = field.axes[0]
    assert horizontal_dimension.kind == DimensionKind.HORIZONTAL
    remote_indices = mesh.remote_indices[horizontal_dimension]

    for hid in range(getattr(mesh, DIMENSION_TO_SIZE_ATTR[horizontal_dimension])):
        if remote_indices[hid] != hid:
            field[hid] = field[remote_indices[hid]]
