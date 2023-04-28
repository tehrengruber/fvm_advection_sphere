# -*- coding: utf-8 -*-
from __future__ import annotations
#import eccodes
import netCDF4 as nc
import numpy as np

from fvm_advection_sphere.mesh.atlas_mesh import AtlasMesh
from fvm_advection_sphere.state_container import StateContainer


def output_data(mesh: AtlasMesh, state: StateContainer, outstep: int, output_ghost=False) -> None:
    print("DISABLED OUTPUT")
    return # TODO
    if output_ghost:
        nb_vertices_output = mesh.nb_vertices
    else:
        nb_vertices_output = mesh.nb_vertices_noghost

    basedir = "/Users/nack/fvm_advection_sphere/tests/"
    basedir = "./"

    output_netcdf = True
    if output_netcdf:
        filename = f"{basedir}data_{outstep}.nc"
        print(f"Define netcdf data file for {nb_vertices_output} number of xy vertices...")
        filename = f"data_{outstep}.nc"
        with nc.Dataset(filename, mode="w") as ds:
            # _ = ds.createDimension("t", 1)
            # tv = ds.createVariable("t", float, ["t"])
            # tv[...] = 0.0
            _ = ds.createDimension("xy", nb_vertices_output)
            longitude = ds.createVariable("longitude", float, ("xy",))
            longitude[...] = mesh.xyrad[:nb_vertices_output, 0]
            latitude = ds.createVariable("latitude", float, ("xy",))
            latitude[...] = mesh.xyrad[:nb_vertices_output, 1]
            rho = ds.createVariable("rho", float, ("xy",))
            rho[...] = state.rho[:nb_vertices_output]
            vx = ds.createVariable("velx", float, ("xy",))
            vx[...] = state.vel[0][:nb_vertices_output]
            vy = ds.createVariable("vely", float, ("xy",))
            vy[...] = state.vel[1][:nb_vertices_output]

    output_numpy = False
    if output_numpy:
        filename = f"{basedir}data_{outstep}.npz"
        print(f"Define numpy array data file for {nb_vertices_output} number of xy vertices...")
        np.savez(
            filename,
            longitude=mesh.xyrad[:nb_vertices_output, 0],
            latitude=mesh.xyrad[:nb_vertices_output, 1],
            rho=state.rho[:nb_vertices_output],
            vx=state.vel[0][:nb_vertices_output],
            vy=state.vel[1][:nb_vertices_output],
        )

    # output_grib = False
    # if output_grib:
    #    gid = eccodes.codes_grib_new_from_samples("regular_gg_sfc_grib2")
    #    # gid = eccodes.codes_grib_new_from_samples("regular_gg_pl_grib1.tmpl")
    #    eccodes.codes_set(gid, "dataType", "fc")
    #    eccodes.codes_set(gid, "truncateDegrees", 1)
    #    eccodes.codes_set(gid, "numberOfPointsAlongAMeridian", grid.ny)
    #    eccodes.codes_set(gid, "latitudeOfFirstGridPointInDegrees", grid.y[0])
    #    eccodes.codes_set(gid, "latitudeOfLastGridPointInDegrees", grid.y[-1])
    #    eccodes.codes_set(gid, "longitudeOfFirstGridPointInDegrees", 0.0)
    #    zval = 360.0 - 360.0 / np.max(np.array(grid.nx))
    #    print(f"zval: {zval}")
    #    eccodes.codes_set(gid, "longitudeOfLastGridPointInDegrees", zval)
    #    eccodes.codes_set(gid, "numberOfParallelsBetweenAPoleAndTheEquator", grid.ny // 2)
    #    eccodes.codes_set(gid, "gridType", "reduced_gg")
    #    eccodes.codes_set_array(gid, "pl", grid.nx)
    #    # eccodes.codes_set_array(gid, "values", state.rho[:])
    #    eccodes.codes_set_values(gid, state.rho[:nb_vertices_output])
    #    output = f"data_{outstep}.grib"
    #    file = basedir + output
    #    print(f"file: {file}")
    #    with open(str(file), "wb") as fout:
    #        eccodes.codes_write(gid, fout)
    #    eccodes.codes_release(gid)
    #   # some more info printed
    #   print(f"grid.nx {grid.nx}")
    #   print(f"len(grid.nx) {len(grid.nx)}")
    #   print(f"grid.ny {grid.ny}")
    #   print(f"grid.nxmax {grid.nxmax}")
    #   print(f"grid.y {grid.y}")
    #   print(f"grid.reduced {grid.reduced}")
    #   print(f"grid.regular {grid.regular}")
