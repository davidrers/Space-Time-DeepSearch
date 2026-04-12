# geostats/temporal/gap_filling.py

import xarray as xr


def fill_temporal_gaps(
    da: xr.DataArray,
    method: str = "linear",
    limit: int | None = None,
) -> xr.DataArray:
    """
    Fill temporal gaps in a datacube using interpolation.

    Parameters
    ----------
    da : xr.DataArray
        Input data with missing values (NaNs)
    method : str, default="linear"
        Interpolation method ('linear', 'nearest', 'spline')
    limit : int, optional
        Maximum number of consecutive NaNs to fill

    Returns
    -------
    xr.DataArray
        Gap-filled datacube
    """

    if "time" not in da.dims:
        raise ValueError("DataArray must have a 'time' dimension.")

    return da.interpolate_na(
        dim="time",
        method=method,
        limit=limit,
    )