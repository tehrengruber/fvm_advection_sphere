from functional.common import DimensionKind
from functional.ffront.fbuiltins import FieldOffset, Dimension

dtype = float

Vertex = Dimension("Vertex")
Edge = Dimension("Edge")
Cell = Dimension("Cell")
E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
E2V = FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
V2EDim = Dimension("V2E", kind=DimensionKind.LOCAL)
V2E = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
K = Dimension("K")

VertexEdgeNb = Dimension("VertexEdgeNb")
V2VEDim = Dimension("V2VE", kind=DimensionKind.LOCAL)
V2VE = FieldOffset("V2Vertex_Edge_Nb", source=VertexEdgeNb, target=(Vertex, V2VEDim))

__all__ = [
    "Vertex",
    "Edge",
    "E2VDim",
    "E2V",
    "V2EDim",
    "V2E",
    "VertexEdgeNb",
    "V2VEDim",
    "V2VE"
]