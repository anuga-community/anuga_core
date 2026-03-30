import numpy as np


def tif2array(filename, verbose=False,):

    import rasterio

    with rasterio.open(filename) as raster:
        ncols = raster.width
        nrows = raster.height
        transform = raster.transform
        x_origin = transform.c   # upper-left x
        x_res    = transform.a   # pixel width (positive)
        y_origin = transform.f   # upper-left y
        y_res    = transform.e   # pixel height (negative for north-up rasters)
        NODATA_value = raster.nodata
        Z = raster.read(1).astype(float)

    if NODATA_value is not None:
        Z = np.where(Z == NODATA_value, np.nan, Z)

    if y_res < 0:
        x = np.linspace(x_origin, x_origin + (ncols - 1) * x_res, ncols)
        y = np.linspace(y_origin + (nrows - 1) * y_res, y_origin, nrows)
        Z = np.flip(Z, axis=0)
        Z = Z.transpose()
    else:
        x = np.linspace(x_origin, x_origin + (ncols - 1) * x_res, ncols)
        y = np.linspace(y_origin, y_origin + (nrows - 1) * y_res, nrows)
        Z = Z.transpose()

    return x, y, Z
