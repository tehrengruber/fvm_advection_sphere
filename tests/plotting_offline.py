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
    velx = fields["velx"]
    vely = fields["vely"]

xydeg_x = np.rad2deg(xyrad_x)
xydeg_y = np.rad2deg(xyrad_y)

xi = np.linspace(0.0, 360.0, 361)
yi = np.linspace(-90.0, 90.0, 181)
triang = tri.Triangulation(xydeg_x[:], xydeg_y[:])
interpolator = tri.LinearTriInterpolator(triang, rho)
# interpolator = tri.LinearTriInterpolator(triang, velx)
# interpolator = tri.LinearTriInterpolator(triang, vely)
Xi, Yi = np.meshgrid(xi, yi)
rho_plot = interpolator(Xi, Yi)
plt.figure(figsize=(15, 6))
plt.contourf(xi, yi, rho_plot, levels=24, cmap="RdBu_r")
plt.grid("on", linestyle="--")
plt.xlim(left=0.0, right=360.0)
plt.ylim(bottom=-90.0, top=90.0)
plt.colorbar()
plt.savefig(f"rho_{outstep}.pdf")
plt.close()
