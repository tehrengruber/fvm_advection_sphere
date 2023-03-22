import dataclasses
import numpy as np

from gt4py.next.iterator.embedded import LocatedField, np_as_located_field
from gt4py.next.ffront.fbuiltins import Field
from gt4py.next.ffront import type_translation as type_translation
import gt4py.next.type_system.type_specifications as type_spec

from fvm_advection_sphere.build_config import float_type
from fvm_advection_sphere.mesh.atlas_mesh import DIMENSION_TO_SIZE_ATTR
from fvm_advection_sphere.common import Vertex, Edge, Cell, K


def allocate_field(mesh, type_: Field | type_spec.FieldType | type_spec.TupleType):
    """
    Allocate a field of zeros of given type.

    >>> allocate_field(mesh, Field[[Vertex], float_type])
    """
    if isinstance(type_, type_spec.TupleType):
        return tuple(allocate_field(mesh, el_type) for el_type in type_.types)
    elif isinstance(type_, type_spec.FieldType):
        shape = [getattr(mesh, DIMENSION_TO_SIZE_ATTR[dim]) for dim in type_.dims]
        return np_as_located_field(*type_.dims)(np.zeros(shape))

    try:
        type_ = type_translation.from_type_hint(type_)
        return allocate_field(mesh, type_)
    except Exception as e:
        raise ValueError(f"Type `{type_}` not understood.") from e


@dataclasses.dataclass
class StateContainer:
    rho: Field[[Vertex], float_type]
    vel: tuple[Field[[Vertex], float_type], Field[[Vertex], float_type]]

    @classmethod
    def from_mesh(cls, mesh):
        """
        Initialize state container with all fields being zero.

        New fields can be added by simply adding the annotation above.
        """
        fields: dict[str, LocatedField] = {}
        for attr in dataclasses.fields(cls):
            type_ = type_translation.from_type_hint(attr.type)
            fields[attr.name] = allocate_field(mesh, type_)
        return cls(**fields)
