import numpy as np

def assert_close(expected, actual):
    assert math.isclose(expected, actual), "expected={}, actual={}".format(
        expected, actual
    )

def nabla(mesh, pp):
    num_edges = mesh.edges_size
    num_vertices = mesh.nodes_size
    e2v = mesh.edges2node_connectivity
    v2e = mesh.nodes2edge_connectivity
    S_MXX, S_MYY = mesh.S_fields
    sign = mesh.sign_field
    vol = mesh.vol_field

    zavg = np.zeros(num_edges)
    for e in range(0, num_edges):
        zavg[e] = 0.5 * (pp[e2v[e, 0]] + pp[e2v[e, 1]])

    pnabla_MXX = np.zeros(num_vertices)
    pnabla_MYY = np.zeros(num_vertices)
    for v in range(0, num_vertices):
        pnabla_MXX[v] = sum(S_MXX[v2e[v, e]] * zavg[v2e[v, e]] * sign[v, e] for e in range(0, v2e.cols(v))) / vol[v]
        pnabla_MYY[v] = sum(S_MYY[v2e[v, e]] * zavg[v2e[v, e]] * sign[v, e] for e in range(0, v2e.cols(v))) / vol[v]

    return pnabla_MXX, pnabla_MYY

from fvm_advection_sphere.utils.atlas_mesh import setup_mesh
import math

def setup_nabla():
    mesh = setup_mesh()

    klevel = 0
    MXX = 0
    MYY = 1
    rpi = 2.0 * math.asin(1.0)
    radius = 6371.22e03
    deg2rad = 2.0 * rpi / 360.0

    zh0 = 2000.0
    zrad = 3.0 * rpi / 4.0 * radius
    zeta = rpi / 16.0 * radius
    zlatc = 0.0
    zlonc = 3.0 * rpi / 2.0

    m_rlonlatcr = mesh.fs_nodes.create_field(
        name="m_rlonlatcr",
        levels=1,
        dtype=np.float64,
        variables=mesh.edges_per_node,
    )
    rlonlatcr = np.array(m_rlonlatcr, copy=False)

    m_rcoords = mesh.fs_nodes.create_field(
        name="m_rcoords", levels=1, dtype=np.float64, variables=mesh.edges_per_node
    )
    rcoords = np.array(m_rcoords, copy=False)

    m_rcosa = mesh.fs_nodes.create_field(name="m_rcosa", levels=1, dtype=np.float64)
    rcosa = np.array(m_rcosa, copy=False)

    m_rsina = mesh.fs_nodes.create_field(name="m_rsina", levels=1, dtype=np.float64)
    rsina = np.array(m_rsina, copy=False)

    m_pp = mesh.fs_nodes.create_field(name="m_pp", levels=1, dtype=np.float64)
    rzs = np.array(m_pp, copy=False)

    rcoords_deg = np.array(mesh.mesh.nodes.field("lonlat"))

    for jnode in range(0, mesh.nodes_size):
        for i in range(0, 2):
            rcoords[jnode, klevel, i] = rcoords_deg[jnode, i] * deg2rad
            rlonlatcr[jnode, klevel, i] = rcoords[
                jnode, klevel, i
            ]  # This is not my pattern!
        rcosa[jnode, klevel] = math.cos(rlonlatcr[jnode, klevel, MYY])
        rsina[jnode, klevel] = math.sin(rlonlatcr[jnode, klevel, MYY])
    for jnode in range(0, mesh.nodes_size):
        zlon = rlonlatcr[jnode, klevel, MXX]
        zdist = math.sin(zlatc) * rsina[jnode, klevel] + math.cos(zlatc) * rcosa[
            jnode, klevel
        ] * math.cos(zlon - zlonc)
        zdist = radius * math.acos(zdist)
        rzs[jnode, klevel] = 0.0
        if zdist < zrad:
            rzs[jnode, klevel] = rzs[jnode, klevel] + 0.5 * zh0 * (
                1.0 + math.cos(rpi * zdist / zrad)
            ) * math.pow(math.cos(rpi * zdist / zeta), 2)

    assert_close(0.0000000000000000, min(rzs))
    assert_close(1965.4980340735883, max(rzs))

    return mesh, rzs[:, klevel]

mesh, pp = setup_nabla()

pnabla_MXX, pnabla_MYY = nabla(mesh, pp)

assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
assert_close(3.5455427772565435e-003, max(pnabla_MXX))
assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
assert_close(3.3540113705465301e-003, max(pnabla_MYY))

bla = 1+1