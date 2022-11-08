# -*- coding: utf-8 -*-
from __future__ import annotations
import matplotlib.tri as tri
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

outstep = 1
data_format = "numpy"
data_format = "netCDF"

if data_format == "numpy":
    fields = np.load(f"./data_{outstep}.npz")
    print(fields.files)
    xyrad_x = fields["longitude"]
    xyrad_y = fields["latitude"]
    rho = fields["rho"]

if data_format == "netCDF":
    fields = nc.Dataset(f"./data_{outstep}.nc")
    print(fields)
    xyrad_x = fields["longitude"]
    xyrad_y = fields["latitude"]
    rho = fields["rho"]


xi = np.linspace(0.0, 2.0 * np.pi, 361)
yi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 181)
triang = tri.Triangulation(xyrad_x[:], xyrad_y[:])
interpolator = tri.LinearTriInterpolator(triang, rho)
Xi, Yi = np.meshgrid(xi, yi)
rho_plot = interpolator(Xi, Yi)
plt.figure(figsize=(15, 6))
plt.contourf(xi, yi, rho_plot, levels=14, cmap="RdBu_r")
plt.grid("on", linestyle="--")
plt.xlim(left=0.0, right=2.0 * np.pi)
plt.ylim(bottom=-0.5 * np.pi, top=0.5 * np.pi)
plt.colorbar()
plt.savefig(f"rho_{outstep}.pdf")
plt.close()
