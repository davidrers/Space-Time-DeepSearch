"""Integration tests for the LandTrendr xarray/dask API."""

import numpy as np
import pytest
import xarray as xr

from space_time_deepsearch.temporal import (
    run_landtrendr,
    extract_change_map,
    annual_composite,
    LandTrendrParams,
)


def _make_synthetic_cube(ny=5, nx=5, n_years=20, seed=42):
    """Create a synthetic DataArray with known change patterns.

    Pixel (0,0): flat-drop-flat (step change at year 7-11)
    Pixel (1,0): linear decline
    Pixel (2,0): constant
    All other pixels: random noise
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    data = rng.random((n_years, ny, nx)) * 0.1 + 0.5  # base noise

    # Pixel (0,0): step change
    data[:7, 0, 0] = 0.8
    data[7:11, 0, 0] = np.linspace(0.8, 0.2, 4)
    data[11:, 0, 0] = 0.2

    # Pixel (1,0): linear decline
    data[:, 1, 0] = np.linspace(1.0, 0.0, n_years)

    # Pixel (2,0): constant
    data[:, 2, 0] = 0.5

    da = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={
            "time": years,
            "y": np.arange(ny),
            "x": np.arange(nx),
        },
    )
    return da


class TestRunLandtrendr:
    def test_output_structure(self):
        """Output Dataset has expected variables and dimensions."""
        da = _make_synthetic_cube()
        result = run_landtrendr(da)

        assert isinstance(result, xr.Dataset)
        assert "source_values" in result
        assert "fitted_values" in result
        assert "is_vertex" in result
        assert "rmse" in result

        assert result["fitted_values"].dims == ("time", "y", "x")
        assert result["is_vertex"].dims == ("time", "y", "x")
        assert result["rmse"].dims == ("y", "x")

    def test_fitted_values_shape(self):
        """Fitted values have same shape as input."""
        da = _make_synthetic_cube()
        result = run_landtrendr(da)
        assert result["fitted_values"].shape == da.shape

    def test_step_change_pixel(self):
        """Pixel (0,0) with step change: fitted tracks the decline."""
        da = _make_synthetic_cube()
        result = run_landtrendr(da)
        fitted = result["fitted_values"].sel(y=0, x=0).values
        # Fitted should capture the decline: start high, end low
        assert fitted[0] > fitted[-1]
        assert np.sum(result["is_vertex"].sel(y=0, x=0).values) >= 2

    def test_constant_pixel_low_rmse(self):
        """Constant pixel (2,0) should have near-zero RMSE."""
        da = _make_synthetic_cube()
        result = run_landtrendr(da)
        rmse = float(result["rmse"].sel(y=2, x=0).values)
        assert rmse < 0.01

    def test_custom_params(self):
        """Custom parameters are respected."""
        da = _make_synthetic_cube()
        params = LandTrendrParams(max_segments=3, spike_threshold=0.5)
        result = run_landtrendr(da, params=params)
        # Should still produce valid output
        assert not np.all(np.isnan(result["fitted_values"].values))

    def test_insufficient_years_raises(self):
        """Fewer years than min_observations_needed raises ValueError."""
        years = np.arange(2000, 2004)
        data = np.random.default_rng(42).random((4, 3, 3))
        da = xr.DataArray(
            data, dims=["time", "y", "x"],
            coords={"time": years, "y": [0, 1, 2], "x": [0, 1, 2]},
        )
        with pytest.raises(ValueError, match="at least"):
            run_landtrendr(da)

    def test_datetime_time_coordinate(self):
        """Works with datetime64 time coordinates."""
        da = _make_synthetic_cube()
        # Convert integer years to datetime64
        dates = [np.datetime64(f"{y}-07-01") for y in da.time.values]
        da = da.assign_coords(time=dates)
        result = run_landtrendr(da)
        assert result["fitted_values"].shape == da.shape


class TestExtractChangeMap:
    def test_output_structure(self):
        """Change map has expected variables."""
        da = _make_synthetic_cube()
        lt = run_landtrendr(da)
        change = extract_change_map(lt)

        assert isinstance(change, xr.Dataset)
        for var in ["yod", "mag", "dur", "preval", "rate", "dsnr"]:
            assert var in change
            assert change[var].dims == ("y", "x")

    def test_step_change_detected(self):
        """Step change pixel (0,0) should have a loss detected."""
        da = _make_synthetic_cube()
        lt = run_landtrendr(da)
        change = extract_change_map(lt, change_type="greatest", delta_filter="loss")

        yod = float(change["yod"].sel(y=0, x=0).values)
        mag = float(change["mag"].sel(y=0, x=0).values)
        dur = float(change["dur"].sel(y=0, x=0).values)

        assert not np.isnan(yod)
        assert mag < 0  # loss
        assert dur > 0

    def test_gain_filter(self):
        """delta_filter='gain' should only show positive changes."""
        da = _make_synthetic_cube()
        lt = run_landtrendr(da)
        change = extract_change_map(lt, delta_filter="gain")
        # Non-NaN magnitudes should all be positive
        valid = ~np.isnan(change["mag"].values)
        if np.any(valid):
            assert np.all(change["mag"].values[valid] > 0)


class TestAnnualComposite:
    def test_median_composite(self):
        """Sub-annual data is composited to annual."""
        # Create monthly data for 3 years
        dates = [
            np.datetime64(f"{y}-{m:02d}-15")
            for y in range(2000, 2003)
            for m in range(1, 13)
        ]
        data = np.random.default_rng(42).random((36, 3, 3))
        da = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": dates, "y": [0, 1, 2], "x": [0, 1, 2]},
        )
        result = annual_composite(da, method="median")
        assert len(result.time) == 3

    def test_mean_composite(self):
        """Mean method works."""
        dates = [np.datetime64(f"2000-{m:02d}-15") for m in range(1, 7)]
        data = np.ones((6, 2, 2))
        da = xr.DataArray(
            data, dims=["time", "y", "x"],
            coords={"time": dates, "y": [0, 1], "x": [0, 1]},
        )
        result = annual_composite(da, method="mean")
        assert len(result.time) == 1
        np.testing.assert_array_almost_equal(result.values, 1.0)

    def test_invalid_method_raises(self):
        """Unknown method raises ValueError."""
        dates = [np.datetime64("2000-01-15")]
        da = xr.DataArray(
            [[[1.0]]], dims=["time", "y", "x"],
            coords={"time": dates, "y": [0], "x": [0]},
        )
        with pytest.raises(ValueError, match="Unknown method"):
            annual_composite(da, method="invalid")
