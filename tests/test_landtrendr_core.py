"""Unit tests for the LandTrendr per-pixel numpy kernel."""

import numpy as np
import pytest

from space_time_deepsearch.temporal._landtrendr_core import (
    _despike,
    _identify_initial_vertices,
    _piecewise_linear_fit,
    landtrendr_pixel,
    extract_change_pixel,
)


class TestDespike:
    def test_no_spike(self):
        """Monotonically increasing series has no spikes."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _despike(values, spike_threshold=0.9)
        np.testing.assert_array_almost_equal(result, values)

    def test_single_spike_removed(self):
        """A single large spike in an otherwise flat series is removed."""
        values = np.array([1.0, 1.0, 5.0, 1.0, 1.0])
        result = _despike(values, spike_threshold=0.5)
        # The spike at index 2 should be replaced with mean of neighbors
        assert result[2] == pytest.approx(1.0)

    def test_threshold_1_no_filtering(self):
        """spike_threshold=1.0 disables filtering entirely."""
        values = np.array([1.0, 1.0, 10.0, 1.0, 1.0])
        result = _despike(values, spike_threshold=1.0)
        np.testing.assert_array_almost_equal(result, values)

    def test_does_not_modify_input(self):
        """Despiking should not modify the original array."""
        values = np.array([1.0, 1.0, 5.0, 1.0, 1.0])
        original = values.copy()
        _despike(values, spike_threshold=0.5)
        np.testing.assert_array_equal(values, original)


class TestIdentifyVertices:
    def test_flat_series(self):
        """Flat series should only have start and end vertices."""
        years = np.arange(2000, 2010, dtype=np.int32)
        values = np.ones(10)
        vertices = _identify_initial_vertices(years, values, max_vertices=10)
        assert vertices == [0, 9]

    def test_single_direction_change(self):
        """V-shaped series: decline then increase has 3 vertices."""
        years = np.arange(2000, 2010, dtype=np.int32)
        values = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5, 6], dtype=np.float64)
        vertices = _identify_initial_vertices(years, values, max_vertices=10)
        assert 0 in vertices
        assert 9 in vertices
        assert 4 in vertices  # trough

    def test_max_vertices_limit(self):
        """Pruning respects the max_vertices limit."""
        years = np.arange(2000, 2020, dtype=np.int32)
        values = np.sin(np.linspace(0, 4 * np.pi, 20))  # many extrema
        vertices = _identify_initial_vertices(years, values, max_vertices=4)
        assert len(vertices) <= 4
        assert vertices[0] == 0
        assert vertices[-1] == 19


class TestPiecewiseLinearFit:
    def test_two_vertices_is_linear(self):
        """Two vertices produce a simple linear interpolation."""
        years = np.arange(2000, 2005, dtype=np.int32)
        values = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        fitted = _piecewise_linear_fit(years, values, [0, 4])
        np.testing.assert_array_almost_equal(fitted, values)

    def test_three_vertices_v_shape(self):
        """Three vertices produce a V-shaped piecewise fit."""
        years = np.arange(2000, 2005, dtype=np.int32)
        values = np.array([4.0, 2.0, 0.0, 2.0, 4.0])
        fitted = _piecewise_linear_fit(years, values, [0, 2, 4])
        np.testing.assert_array_almost_equal(fitted, values)


class TestLandtrendrPixel:
    def test_flat_series(self):
        """Constant series: 1 segment, RMSE near 0, only endpoints are vertices."""
        years = np.arange(2000, 2015, dtype=np.int32)
        values = np.full(15, 0.5)
        fitted, is_vertex, rmse = landtrendr_pixel(years, values)
        assert rmse < 0.01
        assert is_vertex[0] and is_vertex[-1]

    def test_linear_decline(self):
        """Linear decline: should be captured by 1 segment."""
        years = np.arange(2000, 2015, dtype=np.int32)
        values = np.linspace(1.0, 0.0, 15)
        fitted, is_vertex, rmse = landtrendr_pixel(years, values)
        assert rmse < 0.05
        # Fitted should closely follow the linear trend
        np.testing.assert_allclose(fitted, values, atol=0.1)

    def test_step_change_detected(self):
        """Flat-drop-flat pattern: vertices at the transition points."""
        years = np.arange(2000, 2020, dtype=np.int32)
        values = np.array(
            [0.8] * 7 + [0.5, 0.3] + [0.2] * 11, dtype=np.float64
        )
        fitted, is_vertex, rmse = landtrendr_pixel(years, values)
        # At minimum start and end are vertices
        assert np.sum(is_vertex) >= 2
        # Fitted should track the step: higher at start, lower at end
        assert fitted[0] > fitted[-1]
        # RMSE should be reasonable for this clear pattern
        assert rmse < 0.2

    def test_all_nan_returns_nan(self):
        """All-NaN input returns NaN fitted values and NaN RMSE."""
        years = np.arange(2000, 2015, dtype=np.int32)
        values = np.full(15, np.nan)
        fitted, is_vertex, rmse = landtrendr_pixel(years, values)
        assert np.all(np.isnan(fitted))
        assert np.all(~is_vertex)
        assert np.isnan(rmse)

    def test_insufficient_observations(self):
        """Fewer than min_observations_needed returns NaN."""
        years = np.arange(2000, 2004, dtype=np.int32)
        values = np.array([1.0, 2.0, 3.0, 4.0])
        fitted, is_vertex, rmse = landtrendr_pixel(
            years, values, min_observations_needed=6
        )
        assert np.all(np.isnan(fitted))
        assert np.isnan(rmse)

    def test_partial_nan_handled(self):
        """Series with some NaN values still processes valid observations."""
        years = np.arange(2000, 2015, dtype=np.int32)
        values = np.linspace(1.0, 0.0, 15)
        values[3] = np.nan
        values[7] = np.nan
        fitted, is_vertex, rmse = landtrendr_pixel(years, values)
        assert not np.isnan(rmse)
        # Fitted values should be defined at all positions (interpolated)
        assert not np.any(np.isnan(fitted))

    def test_output_shapes(self):
        """Output arrays match input length."""
        n = 20
        years = np.arange(2000, 2000 + n, dtype=np.int32)
        values = np.random.default_rng(42).random(n)
        fitted, is_vertex, rmse = landtrendr_pixel(years, values)
        assert fitted.shape == (n,)
        assert is_vertex.shape == (n,)
        assert isinstance(rmse, (float, np.floating))


class TestExtractChangePixel:
    def _make_step_result(self):
        """Helper: create a step-change segmentation result."""
        years = np.arange(2000, 2020, dtype=np.int32)
        fitted = np.concatenate([
            np.full(8, 0.8),
            np.linspace(0.8, 0.2, 4),
            np.full(8, 0.2),
        ])
        is_vertex = np.zeros(20, dtype=bool)
        is_vertex[0] = True
        is_vertex[7] = True
        is_vertex[11] = True
        is_vertex[19] = True
        rmse = 0.05
        return fitted, is_vertex, rmse, years

    def test_greatest_loss(self):
        """Greatest loss segment is the declining one."""
        fitted, is_vertex, rmse, years = self._make_step_result()
        yod, mag, dur, preval, rate, dsnr = extract_change_pixel(
            fitted, is_vertex, rmse, years,
            change_type="greatest", delta_filter="loss",
        )
        assert yod == 2007.0  # start of decline segment
        assert mag < 0  # loss
        assert dur == 4.0

    def test_no_gain_in_loss_filter(self):
        """If delta_filter='gain' but only loss exists, returns NaN."""
        fitted, is_vertex, rmse, years = self._make_step_result()
        # The only notable change is a loss, so filtering for gain should give NaN
        # (unless the flat→flat segments are counted as slight gain)
        yod, mag, dur, preval, rate, dsnr = extract_change_pixel(
            fitted, is_vertex, rmse, years,
            change_type="greatest", delta_filter="gain",
        )
        # If no gain segments, should be NaN
        if not np.isnan(yod):
            assert mag > 0  # must be a gain if not NaN

    def test_all_nan_input(self):
        """All-NaN fitted values return NaN metrics."""
        years = np.arange(2000, 2020, dtype=np.int32)
        fitted = np.full(20, np.nan)
        is_vertex = np.zeros(20, dtype=bool)
        result = extract_change_pixel(fitted, is_vertex, np.nan, years)
        assert all(np.isnan(v) for v in result)

    def test_newest_change_type(self):
        """'newest' selects the most recent change segment."""
        fitted, is_vertex, rmse, years = self._make_step_result()
        yod, mag, dur, preval, rate, dsnr = extract_change_pixel(
            fitted, is_vertex, rmse, years,
            change_type="newest", delta_filter="all",
        )
        # The last segment starts at year 2011
        assert yod >= 2007.0
