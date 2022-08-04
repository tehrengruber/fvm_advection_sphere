from functional.common import DimensionKind
from functional.ffront.fbuiltins import FieldOffset, Dimension

Vertex = Dimension("Vertex")
Edge = Dimension("Edge")
E2VDim = Dimension("E2V", kind=DimensionKind.LOCAL)
E2V = FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
V2EDim = Dimension("V2E", kind=DimensionKind.LOCAL)
V2E = FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))

__all__ = [
    "Vertex",
    "Edge",
    "E2VDim",
    "E2V",
    "V2EDim",
    "V2E",
]