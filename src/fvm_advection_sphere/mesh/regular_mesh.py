from typing import Tuple
from types import SimpleNamespace
import numpy as np


def setup_mesh(Ni: int, Nj: int, xlim: Tuple[float, float], ylim: Tuple[float, float]):
    """
    Generate a periodic cartesian mesh with Ni x Nj vertices

    Compass directions
            N
        W       E
            S

    Coordinate system
         J
         ^
         |
         | -- > I

    Neighboring conventions (counter-clock-wise)
      C2E:
            X  c2e[2]  X
            |          |
          c2e[3]  c  c2e[1]
            |          |
            X -c2e[0]- X
      C2V:
           c2v[3] --  c2v[2]
             |    c    |
           c2v[0] --  c2v[1]
      V2E:
                   v2e[1]
           v2e[2] -- v -- v2e[0]
                   v2e[3]

    Example mesh for Ni, Ny = 2, 2
       ylim[1]
          ↑      |        |
          |       e5  c2   e7   c3
          |      |        |
          |     v2 --e4-- v3 --e6--
          |      |        |
          |      e1  c0   e3   c1
          ↓      |        |
       ylim[0]   v0 --e0-- v1 --e2--

       xlim[0]   ←-----------------→  xlim[1]
    """

    def _llp_to_id(i, j):
        "Convert lat-long position to id"
        return i + Ni * j

    num_vertices = Ni * Nj
    num_edges = num_vertices * 2
    num_cells = num_vertices

    #
    # topology
    #
    c2v = np.zeros((num_cells, 4), dtype=int)
    c2e = np.zeros((num_cells, 4), dtype=int)
    v2e = np.zeros((num_vertices, 4), dtype=int)
    e2v = np.zeros((num_edges, 2), dtype=int)
    e2c = np.zeros((num_edges, 2), dtype=int)

    # c2v
    #  each cell has four vertices
    for j in range(0, Nj):
        for i in range(0, Ni):
            cell_id = _llp_to_id(i, j)
            # c2v
            c2v[cell_id, 0] = _llp_to_id(i, j)  # vertex bottom-left
            c2v[cell_id, 1] = _llp_to_id((i + Ni + 1) % Ni, j)  # vertex bottom-right
            c2v[cell_id, 2] = _llp_to_id((i + Ni + 1) % Ni, (j + Nj + 1) % Nj)  # vertex top-right
            c2v[cell_id, 3] = _llp_to_id(i, (j + Nj + 1) % Nj)  # vertex top-left

    # c2e, e2c
    # first uniquely attribute two edges to each cell (c2e[0] and c2e[3]) and allocate that edge
    edge_id = 0
    for j in range(0, Nj):
        for i in range(0, Ni):
            cell_id = _llp_to_id(i, j)

            c2e[cell_id, 0] = edge_id
            e2c[edge_id, 0] = _llp_to_id(i, (j + Nj - 1) % Nj)
            e2c[edge_id, 1] = _llp_to_id(i, j)
            edge_id += 1

            c2e[cell_id, 3] = edge_id
            e2c[edge_id, 0] = _llp_to_id(i, j)
            e2c[edge_id, 1] = _llp_to_id((i + Ni - 1) % Ni, j)
            edge_id += 1

    # then find the c2e[1] and c2e[2] edges
    for j in range(0, Nj):
        for i in range(0, Ni):
            cell_id = _llp_to_id(i, j)
            c2e[cell_id, 1] = c2e[_llp_to_id((i + Ni + 1) % Ni, j), 3]
            c2e[cell_id, 2] = c2e[_llp_to_id(i, (j + Nj + 1) % Nj), 0]

    # e2v
    for j in range(0, Nj):
        for i in range(0, Ni):
            cell_id = _llp_to_id(i, j)

            e2v[c2e[cell_id, 0], 0] = c2v[cell_id, 0]
            e2v[c2e[cell_id, 0], 1] = c2v[cell_id, 1]

            e2v[c2e[cell_id, 3], 0] = c2v[cell_id, 0]
            e2v[c2e[cell_id, 3], 1] = c2v[cell_id, 3]

    # v2e
    for j in range(0, Nj):
        for i in range(0, Ni):
            cell_id = _llp_to_id(i, j)
            cell_id_bl = _llp_to_id((i + Ni - 1) % Ni, (j + Nj - 1) % Nj)
            # v2e
            v2e[c2v[cell_id, 0], 0] = c2e[cell_id, 0]
            v2e[c2v[cell_id, 0], 1] = c2e[cell_id, 3]
            v2e[c2v[cell_id, 0], 2] = c2e[cell_id_bl, 2]
            v2e[c2v[cell_id, 0], 3] = c2e[cell_id_bl, 1]

    # flags
    cflags_periodic = np.zeros(num_cells, dtype=np.bool)
    for j in range(0, Nj):
        for i in range(0, Ni):
            cell_id = _llp_to_id(i, j)
            cflags_periodic[cell_id] = i + 1 == Ni or j + 1 == Nj

    #
    # geometry
    #
    dx, dy = (xlim[1] - xlim[0]) / Ni, (ylim[1] - ylim[0]) / Nj

    # vertex coordinates
    xs = np.linspace(xlim[0], xlim[1], num=Ni, endpoint=False)
    ys = np.linspace(ylim[0], ylim[1], num=Nj, endpoint=False)
    xycrd = np.zeros((num_vertices, 2))
    for j in range(0, Nj):
        for i in range(0, Ni):
            xycrd[_llp_to_id(i, j), :] = xs[i], ys[j]

    # cell barycenters (vertices in the dual mesh)
    xc = xs + dx / 2
    yc = ys + dy / 2
    barycenters = np.zeros((num_cells, 2))
    for j in range(0, Nj):
        for i in range(0, Ni):
            cell_id = _llp_to_id(i, j)
            barycenters[cell_id, 0] = xc[i]
            barycenters[cell_id, 1] = yc[j]

    # face properties
    dual_face_normal_weighted = np.zeros((num_edges, 2))
    dual_face_length = np.zeros(num_edges)
    dual_face_normal = np.zeros((num_edges, 2))
    for j in range(0, Nj):
        for i in range(0, Ni):
            cell_id = _llp_to_id(i, j)

            bc_c = barycenters[cell_id, :]
            bc_l = barycenters[_llp_to_id((i + Ni - 1) % Ni, j), :]
            bc_b = barycenters[_llp_to_id(i, (j + Nj - 1) % Nj), :]

            if i == 0:
                bc_l = np.copy(bc_l)
                bc_l[0] -= xlim[1] - xlim[0]
            if j == 0:
                bc_b = np.copy(bc_b)
                bc_b[1] -= ylim[1] - ylim[0]

            for e, bc0, bc1 in ((c2e[cell_id, 0], bc_b, bc_c), (c2e[cell_id, 3], bc_c, bc_l)):
                dual_face_normal_weighted[e] = (bc0 - bc1) @ np.array([[0, 1], [-1, 0]])
                dual_face_length[e] = np.linalg.norm(dual_face_normal_weighted[e])
                dual_face_normal[e, :] = dual_face_normal_weighted[e] / dual_face_length[e]

    dual_face_orientation = np.zeros((num_vertices, 4))
    for v in range(0, num_vertices):
        for i in range(0, 4):
            dual_face_orientation[v, i] = 1 if e2v[v2e[v, i], 0] == v else -1

    vol = np.ones(num_vertices)

    return SimpleNamespace(
        num_vertices=num_vertices,
        num_edges=num_edges,
        # connectivities
        c2v=c2v,
        c2e=c2e,
        v2e=v2e,
        e2v=e2v,
        e2c=e2c,
        # flags
        cflags_periodic=cflags_periodic,
        # geometry
        xycrd=xycrd,
        vol=vol,
        dual_face_normal_weighted=dual_face_normal_weighted,
        dual_face_length=dual_face_length,
        dual_face_normal=dual_face_normal,
        dual_face_orientation=dual_face_orientation,
    )
