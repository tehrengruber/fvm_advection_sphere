import numpy as np

def upstream_flux(
    mesh,
    rho: np.ndarray,    # field on vertices
    vel: np.ndarray     # 2d-vector on edges
) -> np.ndarray:
    """
    compute flux density through the intersection of the two control volumes around the dual cells associated with
    the vertices of `e` using an upwind scheme
    """
    flux = np.zeros(mesh.num_edges)
    for e in range(0, mesh.num_edges):
        # upwind flux (instructive)
        v1, v2 = mesh.e2v[e, :]
        weighted_normal_velocity = np.dot(vel[e], mesh.dual_face_normal_weighted[e])  # velocity projected onto the normal
        if weighted_normal_velocity > 0.0:
            flux[e] = rho[v1] * weighted_normal_velocity
        else:
            flux[e] = rho[v2] * weighted_normal_velocity

    return flux

def fluxdiv(
    mesh,
    rho: np.ndarray,    # field on vertices
    gac: np.ndarray,    # field on vertices
    flux: np.ndarray,   # field on edges
    *,
    δt: float
):
    "compute density in the next timestep"
    rho_next = np.zeros(mesh.num_vertices)
    for v in range(0, mesh.num_vertices):
        rho_next[v] = rho[v] - δt / (mesh.vol[v] * gac[v]) * sum(flux[e] * mesh.dual_face_orientation[v, local_e] for local_e, e in enumerate(mesh.v2e[v, :]))
    return rho_next

def fvm_advect(
    mesh,
    rho: np.ndarray, # field on vertices
    gac: np.ndarray, # field on vertices
    vel: np.ndarray, # 2d-vector on edges
    *,
    δt: float,
):
    flux = upstream_flux(mesh, rho, vel)
    rho_next = fluxdiv(mesh, rho, gac, flux, δt=δt)

    return rho_next

def advector_in_edges(
    mesh,
    vel_vertices
):
    vel_edges = np.zeros((mesh.num_edges, 2))

    for e in range(0, mesh.num_edges):
        v1, v2 = mesh.e2v[e,:]
        vel_edges[e,0] = 0.5 * (vel_vertices[v1,0] + mesh.pole_bc[e]*vel_vertices[v2,0])
        vel_edges[e,1] = 0.5 * (vel_vertices[v1,1] + mesh.pole_bc[e]*vel_vertices[v2,1])

    for ep in range(0, mesh.num_pole_edges):
        e = mesh.pole_edges[ep]
        vel_edges[e,1] = 0.0

    return vel_edges