"""
LandTrendr temporal segmentation for xarray datacubes.

Provides a high-level API to run the LandTrendr algorithm (Kennedy et al. 2010)
on xarray DataArrays with Dask-based parallelization across pixels.

Usage:
    from space_time_deepsearch.temporal import run_landtrendr, extract_change_map

    # Annual NDVI datacube with dims (time, y, x)
    result = run_landtrendr(ndvi_annual)
    change = extract_change_map(result, change_type="greatest", delta_filter="loss")
"""

import dataclasses
import numpy as np
import xarray as xr
from tqdm.auto import tqdm

from ._landtrendr_core import landtrendr_pixel, extract_change_pixel


@dataclasses.dataclass(frozen=True)
class LandTrendrParams:
    """Parameters controlling LandTrendr temporal segmentation.

    Defaults match the Google Earth Engine LandTrendr implementation.

    Attributes:
        max_segments: Maximum number of segments per pixel trajectory.
        spike_threshold: Noise removal sensitivity (0-1). Higher = less filtering.
        vertex_count_overshoot: Extra vertices allowed during initial fitting.
        prevent_one_year_recovery: If True, disallow single-year recovery segments.
        recovery_threshold: Maximum recovery rate (value change per year).
            1.0 = no constraint, 0.25 = max 1/4 of disturbance magnitude per year.
        pval_threshold: P-value threshold for F-test model selection.
        best_model_proportion: Proportion of best-fit RMSE for accepting simpler models.
        min_observations_needed: Minimum annual observations required per pixel.
    """
    max_segments: int = 6
    spike_threshold: float = 0.9
    vertex_count_overshoot: int = 3
    prevent_one_year_recovery: bool = True
    recovery_threshold: float = 0.25
    pval_threshold: float = 0.05
    best_model_proportion: float = 0.75
    min_observations_needed: int = 6


def annual_composite(data, method="median"):
    """Composite sub-annual observations into annual values.

    LandTrendr requires one observation per year. This utility groups
    observations by year and reduces them using the specified method.

    Args:
        data: xarray.DataArray with a ``time`` dimension (datetime64).
        method: Reduction method — "median" (default, robust to outliers)
            or "mean".

    Returns:
        xarray.DataArray with one value per year, time coordinate set to
        July 1 of each year.
    """
    if method == "median":
        result = data.groupby("time.year").median(dim="time")
    elif method == "mean":
        result = data.groupby("time.year").mean(dim="time")
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'median' or 'mean'.")

    # Rename 'year' dim back to 'time' for consistency
    result = result.rename({"year": "time"})
    return result


def run_landtrendr(spectral_index, params=None, progress=True):
    """Run LandTrendr temporal segmentation on every pixel of a datacube.

    Args:
        spectral_index: xarray.DataArray with dims ``(time, y, x)``.
            Must contain annual observations (one value per year). The
            ``time`` coordinate can be datetime64 (years extracted
            automatically) or integer years.
        params: LandTrendrParams instance. Defaults to LandTrendrParams()
            with GEE-matching defaults.
        progress: If True (default), display a progress bar tracking
            per-pixel segmentation.

    Returns:
        xr.Dataset with variables:
            - ``source_values`` (time, y, x): original spectral values
            - ``fitted_values`` (time, y, x): piecewise-linear fitted values
            - ``is_vertex`` (time, y, x): boolean, True at breakpoint years
            - ``rmse`` (y, x): root-mean-square error of the fit
    """
    if params is None:
        params = LandTrendrParams()

    # Extract integer years from time coordinate
    if np.issubdtype(spectral_index.time.dtype, np.datetime64):
        years = spectral_index.time.dt.year.values.astype(np.int32)
    else:
        years = np.asarray(spectral_index.time.values, dtype=np.int32)

    if len(years) < params.min_observations_needed:
        raise ValueError(
            f"Need at least {params.min_observations_needed} annual observations, "
            f"got {len(years)}."
        )

    # Count spatial pixels for progress bar
    spatial_dims = [d for d in spectral_index.dims if d != "time"]
    total_pixels = int(np.prod([spectral_index.sizes[d] for d in spatial_dims]))
    pbar = tqdm(total=total_pixels, desc="LandTrendr", unit="px",
                disable=not progress)

    def _pixel_func(values_1d):
        fitted, is_vtx, rmse = landtrendr_pixel(
            years, values_1d,
            max_segments=params.max_segments,
            spike_threshold=params.spike_threshold,
            vertex_count_overshoot=params.vertex_count_overshoot,
            prevent_one_year_recovery=params.prevent_one_year_recovery,
            recovery_threshold=params.recovery_threshold,
            pval_threshold=params.pval_threshold,
            best_model_proportion=params.best_model_proportion,
            min_observations_needed=params.min_observations_needed,
        )
        pbar.update(1)
        return fitted, is_vtx.astype(np.float64), rmse

    fitted, is_vertex, rmse = xr.apply_ufunc(
        _pixel_func,
        spectral_index,
        input_core_dims=[["time"]],
        output_core_dims=[["time"], ["time"], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64, np.float64, np.float64],
    )
    pbar.close()

    # Convert vertex mask back to bool
    is_vertex = is_vertex > 0.5

    return xr.Dataset({
        "source_values": spectral_index,
        "fitted_values": fitted.transpose("time", ...),
        "is_vertex": is_vertex.transpose("time", ...),
        "rmse": rmse,
    })


def extract_change_map(lt_result, change_type="greatest", delta_filter="loss"):
    """Derive a change map from LandTrendr segmentation results.

    Identifies segments between vertices and selects one per pixel based
    on the specified criteria, producing spatial maps of change attributes.

    Args:
        lt_result: xr.Dataset returned by ``run_landtrendr()``.
        change_type: Segment selection criterion:
            - ``"greatest"``: largest absolute magnitude (default)
            - ``"longest"``: longest duration
            - ``"steepest"``: largest absolute rate
            - ``"newest"``: most recent change
        delta_filter: Direction filter:
            - ``"loss"``: only negative spectral change (default)
            - ``"gain"``: only positive spectral change
            - ``"all"``: both directions

    Returns:
        xr.Dataset with variables (all dims ``(y, x)``):
            - ``yod``: Year of Detection (start year of the change segment)
            - ``mag``: Magnitude (spectral delta, start to end)
            - ``dur``: Duration in years
            - ``preval``: Pre-change fitted spectral value
            - ``rate``: Magnitude / duration
            - ``dsnr``: Delta Signal-to-Noise Ratio (|magnitude| / RMSE)
    """
    fitted = lt_result["fitted_values"]
    is_vertex = lt_result["is_vertex"].astype(np.float64)
    rmse = lt_result["rmse"]

    if np.issubdtype(fitted.time.dtype, np.datetime64):
        years_arr = fitted.time.dt.year.values.astype(np.int32)
    else:
        years_arr = np.asarray(fitted.time.values, dtype=np.int32)

    def _change_pixel(fitted_1d, vertex_1d, rmse_scalar):
        return extract_change_pixel(
            fitted_1d, vertex_1d > 0.5, rmse_scalar, years_arr,
            change_type=change_type, delta_filter=delta_filter,
        )

    yod, mag, dur, preval, rate, dsnr = xr.apply_ufunc(
        _change_pixel,
        fitted, is_vertex, rmse,
        input_core_dims=[["time"], ["time"], []],
        output_core_dims=[[], [], [], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float64] * 6,
    )

    return xr.Dataset({
        "yod": yod,
        "mag": mag,
        "dur": dur,
        "preval": preval,
        "rate": rate,
        "dsnr": dsnr,
    })
