import numpy as np

def fvm_advect(
    mesh,
    rho: np.ndarray,    # field on vertices
    gac: np.ndarray,
    *,
    vel: np.ndarray,    # 2d-vector on edges
    δt: float,
):
    # compute flux density through the intersection of the two
    #  control volumes around the dual cells associated with
    #  the vertices of `e` using an upwind scheme
    flux = np.zeros(mesh.num_edges)
    for e in range(0, mesh.num_edges):
        # upwind flux (instructive)
        v1, v2 = mesh.e2v[e, :]
        weighted_normal_velocity = np.dot(vel[e], mesh.dual_face_normal_weighted[e])  # velocity projected onto the normal
        if weighted_normal_velocity > 0.0:
            flux[e] = rho[v1] * weighted_normal_velocity
        else:
            flux[e] = rho[v2] * weighted_normal_velocity

    # compute density in the next timestep
    for v in range(0, mesh.num_vertices):
        #rho[v] = rho - δt / mesh.volume[v] * sum(flux[e] * face_orientation[v, local_e] for local_e, e in enumerate(mesh.v2e[v, :]))
        rho[v] = rho[v] - δt / (mesh.vol[v]*gac[v]) * sum(flux[e] * mesh.dual_face_orientation[v, local_e] for local_e, e in enumerate(mesh.v2e[v, :]))



def advector_in_edges(
    mesh,
    *,
    vel_vertices,
    vel_edges):

    for e in range(0, mesh.num_edges):
        v1, v2 = mesh.e2v[e,:]
        vel_edges[e,0] = 0.5 * (vel_vertices[v1,0] + mesh.pole_bc[e]*vel_vertices[v2,0])
        vel_edges[e,1] = 0.5 * (vel_vertices[v1,1] + mesh.pole_bc[e]*vel_vertices[v2,1])

    for ep in range(0, mesh.num_pole_edges):
        e = mesh.pole_edges[ep]
        vel_edges[e,1] = 0.0