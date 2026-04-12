# geostats/temporal/composites.py

import xarray as xr
from typing import Literal

ReducerType = Literal["mean", "median", "max", "min"]


def create_time_composites(
    da: xr.DataArray,
    freq: str = "1W",
    reducer: ReducerType = "mean",
) -> xr.DataArray:
    """
    Create temporal composites from an NDVI datacube.

    Parameters
    ----------
    da : xr.DataArray
        Input data with dimensions (time, y, x)
    freq : str, default="1W"
        Resampling frequency (e.g., '1W', '1M')
    reducer : str, default="mean"
        Aggregation method: 'mean', 'median', 'max', 'min'

    Returns
    -------
    xr.DataArray
        Resampled datacube with regular temporal resolution
    """

    if "time" not in da.dims:
        raise ValueError("Input DataArray must have a 'time' dimension.")

    resampler = da.resample(time=freq)

    if reducer == "mean":
        return resampler.mean(dim="time", skipna=True)
    elif reducer == "median":
        return resampler.median(dim="time", skipna=True)
    elif reducer == "max":
        return resampler.max(dim="time", skipna=True)
    elif reducer == "min":
        return resampler.min(dim="time", skipna=True)
    else:
        raise ValueError(f"Unsupported reducer: {reducer}")