import dataclasses
import numpy as np

from functional.iterator.embedded import LocatedField, np_as_located_field
from functional.ffront.fbuiltins import Field
from functional.ffront.symbol_makers import make_symbol_type_from_typing
import functional.ffront.common_types as ct

from fvm_advection_sphere.common import Vertex, Edge, Cell, K

_DIMENSION_TO_SIZE_ATTR = {
    Vertex: "num_vertices",
    Edge: "num_edges",
    Cell: "num_cells"
}

@dataclasses.dataclass
class StateContainer:
    rho: Field[[Vertex], float]
    vel_x: Field[[Vertex], float]
    vel_y: Field[[Vertex], float]

    @classmethod
    def from_mesh(cls, mesh):
        fields: dict[str, LocatedField] = {}
        for attr in dataclasses.fields(cls):
            type_ = make_symbol_type_from_typing(attr.type)
            if not isinstance(type_, ct.FieldType):
                raise ValueError(f"Expected a Field type for `{attr.name}` attribute.")
            shape = [getattr(mesh, _DIMENSION_TO_SIZE_ATTR[dim]) for dim in type_.dims]
            fields[attr.name] = np_as_located_field(*type_.dims)(np.zeros(shape))
        return cls(**fields)