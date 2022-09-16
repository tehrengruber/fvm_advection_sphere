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

__all__ = [
    "Vertex",
    "Edge",
    "E2VDim",
    "E2V",
    "V2EDim",
    "V2E",
]