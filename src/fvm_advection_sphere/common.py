from fvm_advection_sphere.build_config import float_type
from gt4py.next.common import DimensionKind
from gt4py.next.ffront.fbuiltins import FieldOffset, Dimension

Vertex = Dimension("Vertex")
Edge = Dimension("Edge")
Cell = Dimension("Cell")
V2VDim = Dimension("V2V", kind=DimensionKind.LOCAL)
E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
V2EDim = Dimension("V2E", kind=DimensionKind.LOCAL)
V2V = FieldOffset("V2V", source=Vertex, target=(Vertex, V2VDim))
E2V = FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
V2E = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
K = Dimension("K")

VertexEdgeNb = Dimension("VertexEdgeNb")
V2VEDim = Dimension("V2VE", kind=DimensionKind.LOCAL)
V2VE = FieldOffset("V2Vertex_Edge_Nb", source=VertexEdgeNb, target=(Vertex, V2VEDim))

__all__ = [
    "Vertex",
    "Edge",
    "V2V",
    "V2E",
    "E2V",
    "V2VDim",
    "E2VDim",
    "V2EDim",
    "VertexEdgeNb",
    "V2VEDim",
    "V2VE",
]
