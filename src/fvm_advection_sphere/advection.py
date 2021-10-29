import numpy as np

def fvm_advect(
    mesh,
    rho: np.ndarray,    # field on vertices
    *,
    δt: float,
    vel: np.ndarray,    # 2d-vector
):
    # compute flux density through the intersection of the two
    #  control volumes around the dual cells associated with
    #  the vertices of `e` using an upwind scheme
    flux = np.zeros(mesh.num_edges)
    for e in range(0, mesh.num_edges):
        # upwind flux (instructive)
        v1, v2 = mesh.e2v[e, :]
        normal_velocity = np.dot(vel, mesh.dual_face_normal[e])  # velocity projected onto the normal
        if np.dot(vel, mesh.dual_face_normal[e]) > 0:
            flux[e] = rho[v1] * normal_velocity * mesh.dual_face_length[e]
        else:
            flux[e] = rho[v2] * normal_velocity * mesh.dual_face_length[e]

    # compute density in the next timestep
    for v in range(0, mesh.num_vertices):
        #rho[v] = rho - δt / mesh.volume[v] * sum(flux[e] * face_orientation[v, local_e] for local_e, e in enumerate(mesh.v2e[v, :]))
        rho[v] = rho[v] - δt * sum(flux[e] * mesh.face_orientation[v, local_e] for local_e, e in enumerate(mesh.v2e[v, :]))


