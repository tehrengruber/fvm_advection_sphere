from typing import Hashable, Optional, Any
from functools import cached_property
import dataclasses
import textwrap
import platform

import numpy as np
import pyvista as pv
from functional.ffront.fbuiltins import Field

from fvm_advection_sphere.mesh.atlas_mesh import AtlasMesh
from fvm_advection_sphere.utils.vis import make_dataset_from_arrays, start_pyvista

# TODO(tehrengruber): Remove
start_pyvista()


@dataclasses.dataclass
class Plotter:
    mesh: AtlasMesh
    fields: dict[Hashable, Field]
    layout: Optional[str] = None
    plotter_args: Optional[dict[str, Any]] = None

    @cached_property
    def _datasets(self):
        i = 0
        datasets = {}
        for name, field in self.fields.items():
            datasets[name] = make_dataset_from_arrays(
                self.mesh.xyarc,
                edges=self.mesh.e2v_np,
                cells=self.mesh.c2v_np,
                vertex_fields={name: np.asarray(field)},
                edge_fields={},
                cell_fields={},
            )
        return datasets

    @cached_property
    def _pv_plotter(self):
        # compute layout
        layout: list[str] = textwrap.dedent(self.layout).strip().split("\n")
        rows, columns = len(layout), len(layout[0])
        if not all(len(line) == columns for line in layout):
            raise ValueError("Invalid layout")

        subplot_indices: dict[str, tuple[int, int]] = {}
        groups_: list[tuple[set, set]] = [(set(), set()) for _ in range(len(self.fields))]
        for row in range(rows):
            for column in range(columns):
                field_index = int(layout[row][column])
                field_name = [*self.fields.keys()][field_index]
                if not field_name in subplot_indices:
                    subplot_indices[field_name] = (row, column)
                groups_[field_index][0].add(row)
                groups_[field_index][1].add(column)
        groups = [([*group[0]], [*group[1]]) for group in groups_]

        # create plotter
        plotter_args = self.plotter_args or {"interpolate_before_map": True}
        plotter = pv.Plotter(
            shape=(rows, columns), row_weights=[1] * rows, col_weights=[1] * columns, groups=groups
        )

        # plot data
        for i, (name, dataset) in enumerate(self._datasets.items()):
            plotter.subplot(*subplot_indices[name])
            plotter.add_mesh(dataset, **plotter_args)
            plotter.add_text(name)

        return plotter

    def show(self, title=None, font_size=None, **add_mesh_kwargs):
        # self._pv_plotter.show(cpos="xy")
        self._pv_plotter.show(cpos="xy", interactive_update=True, auto_close=False)

    def update(self):
        self._pv_plotter.render()

    def save(self, filename: str):
        self._pv_plotter.save_graphic(filename)

    def update_fields(self, fields: dict[Hashable, Field]):
        for name, field in fields.items():
            dataset = self._datasets[name]
            dataset["vertices"][name] = np.asarray(field)
            dataset["vertices_interpolated"][name] = np.asarray(field)
