import dataclasses

from functional.ffront.fbuiltins import Dimension, FieldOffset
from functional.ffront.gtcallable import GTCallable
import functional.ffront.itir_makers as im
import functional.ffront.common_types as ct
import functional.iterator.ir as itir

from fvm_advection_sphere.common import *

@dataclasses.dataclass(frozen=True)
class AsNeighborField(GTCallable):
    name: str
    source: Dimension
    target: tuple[Dimension, Dimension]
    offset: FieldOffset
    dtype: ct.ScalarType

    def __gt_itir__(self):
        return itir.FunctionDefinition(
            id=self.name,
            params=[itir.Sym(id="arg")],
            expr=im.shift_(self.offset.target[1].value)("arg")
        )

    def __gt_type__(self):
        # note: we are returning a function type instead of a field operator
        #  as we don't want the lowering to introduce a lift
        return ct.FunctionType(
            args=[ct.FieldType(dims=[self.source], dtype=self.dtype)],
            kwargs={},
            returns=ct.FieldType(dims=list(self.target), dtype=self.dtype)
        )

as_vertex_v2e_field = AsNeighborField(
        name="as_vertex_v2e_field",
        source=VertexEdgeNb,
        target=(Vertex, V2EDim),
        offset=V2VE,
        dtype=ct.ScalarType(kind=ct.ScalarKind.FLOAT64))