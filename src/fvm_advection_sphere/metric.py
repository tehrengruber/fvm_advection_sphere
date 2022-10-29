import dataclasses
import numpy as np

from fvm_advection_sphere.common import *
from fvm_advection_sphere.mesh.atlas_mesh import AtlasMesh
from functional.ffront.fbuiltins import Field
from functional.iterator.embedded import np_as_located_field


@dataclasses.dataclass
class Metric:
    g11: Field[[Vertex], float]
    g22: Field[[Vertex], float]
    gac: Field[[Vertex], float]

    @classmethod
    def from_mesh(cls, mesh: AtlasMesh):
        rsina = np.sin(mesh.xyrad[:, 1])
        rcosa = np.cos(mesh.xyrad[:, 1])

        g11 = np_as_located_field(Vertex)(1.0 / rcosa)
        g22 = np_as_located_field(Vertex)(1.0 * np.ones(mesh.num_vertices))
        gac = np_as_located_field(Vertex)(rcosa)

        return cls(g11=g11, g22=g22, gac=gac)
