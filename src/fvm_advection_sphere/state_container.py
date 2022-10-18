import dataclasses
import numpy as np

from functional.iterator.embedded import LocatedField, np_as_located_field
from functional.ffront.fbuiltins import Field
from functional.ffront.symbol_makers import make_symbol_type_from_typing
import functional.ffront.common_types as ct

from fvm_advection_sphere.mesh.atlas_mesh import DIMENSION_TO_SIZE_ATTR
from fvm_advection_sphere.common import Vertex, Edge, Cell, K

def allocate_field(mesh, type_: Field | ct.FieldType | ct.TupleType):
    """
    Allocate a field of zeros of given type.

    >>> allocate_field(mesh, Field[[Vertex], float])
    """
    if isinstance(type_, ct.TupleType):
        return tuple(allocate_field(mesh, el_type) for el_type in type_.types)
    elif isinstance(type_, ct.FieldType):
        shape = [getattr(mesh, DIMENSION_TO_SIZE_ATTR[dim]) for dim in type_.dims]
        return np_as_located_field(*type_.dims)(np.zeros(shape))

    try:
        type_ = make_symbol_type_from_typing(type_)
        return allocate_field(mesh, type_)
    except Exception as e:
        raise ValueError(f"Type `{type_}` not understood.") from e

@dataclasses.dataclass
class StateContainer:
    rho: Field[[Vertex], float]
    vel: tuple[Field[[Vertex], float], Field[[Vertex], float]]

    @classmethod
    def from_mesh(cls, mesh):
        """
        Initialize state container with all fields being zero.

        New fields can be added by simply adding the annotation above.
        """
        fields: dict[str, LocatedField] = {}
        for attr in dataclasses.fields(cls):
            type_ = make_symbol_type_from_typing(attr.type)
            fields[attr.name] = allocate_field(mesh, type_)
        return cls(**fields)