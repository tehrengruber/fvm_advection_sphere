from typing import Final, Iterable, Optional, TypeVar, Union
import numpy as np
import math
import functools

from atlas4py import (
    StructuredGrid,
    Topology,
    Config,
    StructuredMeshGenerator,
    functionspace,
    build_edges,
    build_node_to_edge_connectivity,
    build_median_dual_mesh,
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

class FVMMesh:
    def __init__(self, *, grid=StructuredGrid("O32"), radius=6371.22e03, config=_default_config()):
        mesh = StructuredMeshGenerator(config).generate(grid)

        fs_edges = functionspace.EdgeColumns(mesh, halo=1)
        fs_nodes = functionspace.NodeColumns(mesh, halo=1)

        build_edges(mesh)
        build_node_to_edge_connectivity(mesh)
        build_median_dual_mesh(mesh)

        edges_per_node = max([mesh.nodes.edge_connectivity.cols(node) for node in range(0, fs_nodes.size)])

        self.radius = radius
        self.mesh = mesh
        self.fs_edges = fs_edges
        self.fs_nodes = fs_nodes
        self.edges_per_node = edges_per_node

    @property
    def edges2node_connectivity(self):
        return self.mesh.edges.node_connectivity

    @property
    def nodes2edge_connectivity(self):
        return self.mesh.nodes.edge_connectivity

    @property
    def nodes_size(self):
        return self.fs_nodes.size

    @property
    def edges_size(self):
        return self.fs_edges.size

    @functools.cached_property
    def sign_field(self):
        node2edge_sign = np.zeros((self.nodes_size, self.edges_per_node))
        edge_flags = np.array(self.mesh.edges.flags())

        def is_pole_edge(e):
            return Topology.check(edge_flags[e], Topology.POLE)

        for jnode in range(0, self.nodes_size):
            node_edge_con = self.mesh.nodes.edge_connectivity
            edge_node_con = self.mesh.edges.node_connectivity
            for jedge in range(0, node_edge_con.cols(jnode)):
                iedge = node_edge_con[jnode, jedge]
                ip1 = edge_node_con[iedge, 0]
                if jnode == ip1:
                    node2edge_sign[jnode, jedge] = 1.0
                else:
                    node2edge_sign[jnode, jedge] = -1.0
                    if is_pole_edge(iedge):
                        node2edge_sign[jnode, jedge] = 1.0
        return node2edge_sign

    @functools.cached_property
    def S_fields(self):
        S = np.array(self.mesh.edges.field("dual_normals"), copy=False)
        S_MXX = np.zeros((self.edges_size))
        S_MYY = np.zeros((self.edges_size))

        MXX = 0
        MYY = 1

        for i in range(0, self.edges_size):
            S_MXX[i] = S[i, MXX] * self.radius * deg2rad
            S_MYY[i] = S[i, MYY] * self.radius * deg2rad

        assert math.isclose(min(S_MXX), -103437.60479272791)
        assert math.isclose(max(S_MXX), 340115.33913622628)
        assert math.isclose(min(S_MYY), -2001577.7946404363)
        assert math.isclose(max(S_MYY), 2001577.7946404363)

        return S_MXX, S_MYY

    @functools.cached_property
    def vol_field(self):
        vol_atlas = np.array(self.mesh.nodes.field("dual_volumes"), copy=False)
        # dual_volumes 4.6510228700066421    68.891611253882218    12.347560975609632
        assert_close(4.6510228700066421, min(vol_atlas))
        assert_close(68.891611253882218, max(vol_atlas))

        vol = np.zeros((vol_atlas.size))
        for i in range(0, vol_atlas.size):
            vol[i] = vol_atlas[i] * pow(deg2rad, 2) * pow(self.radius, 2)
        # VOL(min/max):  57510668192.214096    851856184496.32886
        assert_close(57510668192.214096, min(vol))
        assert_close(851856184496.32886, max(vol))
        return vol

    # todo
    def as_connectivity_arrays(self, edge_blocks: Optional[OneOrMore[int]] = None, cell_blocks: Optional[OneOrMore[int]] = None):
        points = np.asarray(self.mesh.nodes.lonlat)
        assert len(points) == self.mesh.nodes.size

        if edge_blocks is None:
            edge_blocks = range(self.mesh.edges.node_connectivity.blocks)
        elif isinstance(edge_blocks, int):
            edge_blocks = [edge_blocks]
        edges = []
        for block_id in edge_blocks:
            block = self.mesh.edges.node_connectivity.block(block_id)
            edges.extend([[block[i, j] for j in range(block.cols)] for i in range(block.rows)])
        edges = np.asarray(edges, dtype=int)

        if cell_blocks is None:
            cell_blocks = range(self.mesh.cells.node_connectivity.blocks)
        elif isinstance(cell_blocks, int):
            cell_blocks = [cell_blocks]
        cells = []
        for block_id in cell_blocks:
            block = self.mesh.cells.node_connectivity.block(block_id)
            cells.extend([[block[i, j] for j in range(block.cols)] for i in range(block.rows)])
        cells = np.asarray(cells, dtype=int)

        return points, edges, cells