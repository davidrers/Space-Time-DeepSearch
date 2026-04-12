# geostats/pipeline.py

import xarray as xr
from space_time_deepsearch.geostats.temporal.composites import create_time_composites
from space_time_deepsearch.geostats.temporal.gap_filling import fill_temporal_gaps


def process_ndvi_timeseries(
    ndvi: xr.DataArray,
    freq: str = "1W",
    reducer: str = "mean",
    gap_method: str = "linear",
) -> xr.DataArray:
    """
    Clean NDVI processing pipeline with proper ordering.

    Steps:
    1. Temporal compositing
    2. Spatial kriging (fill spatial gaps)
    3. Temporal interpolation (fill time gaps)
    """

    # -------------------------
    # Step 1: Temporal composite
    # -------------------------
    composites = create_time_composites(
        ndvi,
        freq=freq,
        reducer=reducer,
    )

    # -------------------------
    # Step 2: Temporal gap filling
    # -------------------------
    temporal_filled = fill_temporal_gaps(
        composites,
        method=gap_method,
    )

    return temporal_filled