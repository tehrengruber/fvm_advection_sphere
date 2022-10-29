import math
import numpy as np

from atlas4py import StructuredGrid
from fvm_advection_sphere.mesh.atlas_mesh import AtlasMesh


def assert_close(expected, actual):
    assert math.isclose(expected, actual), "expected={}, actual={}".format(expected, actual)


def nabla(mesh, pp):
    deg2rad = 2.0 * np.pi / 360.0

    e2v = mesh.e2v_np
    v2e = mesh.v2e_np
    sign = mesh.dual_face_orientation_np
    vol = mesh.vol_np

    zavg = np.zeros(mesh.num_edges)
    for e in range(0, mesh.num_edges):
        zavg[e] = 0.5 * (pp[e2v[e, 0]] + pp[e2v[e, 1]])

    pnabla_MXX = np.zeros(mesh.num_vertices)
    pnabla_MYY = np.zeros(mesh.num_vertices)
    for v in range(0, mesh.num_vertices):
        pnabla_MXX[v] = (
            sum(
                mesh.dual_face_normal_weighted_np[v2e[v, e], 0] * zavg[v2e[v, e]] * sign[v, e]
                for e in range(0, v2e.shape[1])
                if v2e[v, e] != -1
            )
            / vol[v]
        )
        pnabla_MYY[v] = (
            sum(
                mesh.dual_face_normal_weighted_np[v2e[v, e], 1] * zavg[v2e[v, e]] * sign[v, e]
                for e in range(0, v2e.shape[1])
                if v2e[v, e] != -1
            )
            / vol[v]
        )

    return pnabla_MXX, pnabla_MYY


def setup_nabla():
    grid = StructuredGrid("O32")
    mesh = AtlasMesh.generate(grid)
    raw_mesh = mesh._atlas_mesh

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

    rlonlatcr = np.zeros((mesh.num_vertices, mesh.v2e_np.shape[1]))
    rcoords = np.zeros((mesh.num_vertices, mesh.v2e_np.shape[1]))
    rcosa = np.zeros((mesh.num_vertices,))
    rsina = np.zeros((mesh.num_vertices,))
    rzs = np.zeros((mesh.num_vertices,))

    rcoords_deg = np.array(raw_mesh.nodes.field("lonlat"))

    for jnode in range(0, mesh.num_vertices):
        for i in range(0, 2):
            rcoords[jnode, i] = rcoords_deg[jnode, i] * deg2rad
            rlonlatcr[jnode, i] = rcoords[jnode, i]  # This is not my pattern!
        rcosa[jnode] = math.cos(rlonlatcr[jnode, MYY])
        rsina[jnode] = math.sin(rlonlatcr[jnode, MYY])
    for jnode in range(0, mesh.num_vertices):
        zlon = rlonlatcr[jnode, MXX]
        zdist = math.sin(zlatc) * rsina[jnode] + math.cos(zlatc) * rcosa[jnode] * math.cos(
            zlon - zlonc
        )
        zdist = radius * math.acos(zdist)
        rzs[jnode] = 0.0
        if zdist < zrad:
            rzs[jnode] = rzs[jnode] + 0.5 * zh0 * (1.0 + math.cos(rpi * zdist / zrad)) * math.pow(
                math.cos(rpi * zdist / zeta), 2
            )

    assert_close(0.0000000000000000, min(rzs))
    assert_close(1965.4980340735883, max(rzs))

    return mesh, rzs


mesh, pp = setup_nabla()

pnabla_MXX, pnabla_MYY = nabla(mesh, pp)

assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
assert_close(3.5455427772565435e-003, max(pnabla_MXX))
assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
assert_close(3.3540113705465301e-003, max(pnabla_MYY))

bla = 1 + 1
