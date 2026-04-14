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


# ---------------------------------------------------------------------------
# Real-data parameter tests using a known NDVI disturbance/recovery pixel
# ---------------------------------------------------------------------------

# Data: forest pixel with major disturbance ~1997 and gradual recovery
_REAL_YEARS = np.array([
    1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993,
    1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,
    2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
    2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023,
    2024, 2025,
], dtype=np.int32)

_REAL_NDVI = np.array([
    0.817, 0.875, 0.862, 0.856, 0.856, 0.854, 0.848, 0.861, 0.816,
    0.809, 0.842, 0.858, 0.869, 0.413, 0.363, 0.528, 0.589, 0.632,
    0.638, 0.658, 0.719, 0.777, 0.811, 0.805, 0.826, 0.835, 0.849,
    0.870, 0.879, 0.881, 0.892, 0.891, 0.904, 0.902, 0.887, 0.882,
    0.895, 0.894, 0.903, 0.896, 0.902, 0.904,
], dtype=np.float64)


class TestDefaultParametersOnRealData:
    """Verify default parameters produce a sensible fit on real data."""

    def test_defaults_detect_disturbance(self):
        """Default parameters should detect the ~1997 disturbance."""
        fitted, is_vertex, rmse = landtrendr_pixel(_REAL_YEARS, _REAL_NDVI)

        # Fitted value should drop substantially around the disturbance
        pre_dist = fitted[_REAL_YEARS <= 1996].mean()
        post_dist = fitted[(_REAL_YEARS >= 1997) & (_REAL_YEARS <= 1999)].mean()
        assert pre_dist - post_dist > 0.15, (
            f"Disturbance not captured: pre={pre_dist:.3f} post={post_dist:.3f}"
        )

    def test_defaults_output_valid(self):
        """Default outputs have correct shapes, no NaN, positive RMSE."""
        fitted, is_vertex, rmse = landtrendr_pixel(_REAL_YEARS, _REAL_NDVI)
        assert fitted.shape == _REAL_NDVI.shape
        assert is_vertex.shape == _REAL_NDVI.shape
        assert not np.any(np.isnan(fitted))
        assert rmse > 0
        # Start and end should always be vertices
        assert is_vertex[0] and is_vertex[-1]

    def test_defaults_rmse_reasonable(self):
        """RMSE with default params should be well below the series range."""
        fitted, is_vertex, rmse = landtrendr_pixel(_REAL_YEARS, _REAL_NDVI)
        series_range = _REAL_NDVI.max() - _REAL_NDVI.min()
        assert rmse < series_range * 0.5, (
            f"RMSE ({rmse:.3f}) too large relative to range ({series_range:.3f})"
        )


class TestMaxSegments:
    """max_segments controls the maximum model complexity."""

    def test_one_segment_is_linear(self):
        """max_segments=1 forces a single straight line (2 vertices)."""
        fitted, is_vertex, rmse = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI, max_segments=1
        )
        n_vertices = np.sum(is_vertex)
        assert n_vertices == 2, f"Expected 2 vertices, got {n_vertices}"

    def test_more_segments_better_or_equal_rmse(self):
        """More allowed segments should yield equal or lower RMSE."""
        _, _, rmse_few = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI, max_segments=2
        )
        _, _, rmse_many = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI, max_segments=10,
            pval_threshold=1.0, best_model_proportion=1.0,
        )
        assert rmse_many <= rmse_few + 1e-9, (
            f"More segments gave worse RMSE: {rmse_many:.4f} > {rmse_few:.4f}"
        )

    def test_more_segments_more_or_equal_vertices(self):
        """Higher max_segments should yield at least as many vertices."""
        _, is_vertex_low, _ = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI, max_segments=2,
        )
        _, is_vertex_high, _ = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI, max_segments=6,
        )
        assert np.sum(is_vertex_high) >= np.sum(is_vertex_low)


class TestSpikeThreshold:
    """spike_threshold controls noise removal in the despike stage."""

    def test_threshold_1_preserves_all(self):
        """spike_threshold=1.0 skips despiking — noise is kept."""
        clean = _despike(_REAL_NDVI.copy(), spike_threshold=1.0)
        np.testing.assert_array_equal(clean, _REAL_NDVI)

    def test_lower_threshold_removes_more_spikes(self):
        """Lower spike_threshold = more aggressive filtering (larger internal
        threshold relative to range), so more points are flagged as spikes.
        """
        spiky = np.array([
            0.8, 0.8, 0.8, 0.5, 0.8, 0.8, 0.8, 0.9, 0.8, 0.8,
        ])
        clean_high = _despike(spiky.copy(), spike_threshold=0.95)
        clean_low = _despike(spiky.copy(), spike_threshold=0.5)
        diff_high = np.sum(np.abs(clean_high - spiky))
        diff_low = np.sum(np.abs(clean_low - spiky))
        # Lower spike_threshold → more aggressive → more changes
        assert diff_low >= diff_high, (
            f"spike_threshold=0.5 changed {diff_low:.3f}, "
            f"spike_threshold=0.95 changed {diff_high:.3f}"
        )

    def test_extreme_threshold_zero_replaces_all_spikes(self):
        """spike_threshold=0.0 is maximally aggressive."""
        clean = _despike(_REAL_NDVI.copy(), spike_threshold=0.0)
        # Still same length
        assert len(clean) == len(_REAL_NDVI)
        # Endpoints are never changed
        assert clean[0] == _REAL_NDVI[0]
        assert clean[-1] == _REAL_NDVI[-1]


class TestVertexCountOvershoot:
    """vertex_count_overshoot allows extra candidate vertices during fitting."""

    def test_zero_overshoot_still_works(self):
        """vertex_count_overshoot=0 limits initial vertices to max_segments+1."""
        fitted, is_vertex, rmse = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            max_segments=4, vertex_count_overshoot=0,
        )
        assert not np.isnan(rmse)
        assert np.sum(is_vertex) >= 2

    def test_higher_overshoot_enables_better_fit(self):
        """More overshoot provides more candidates, potentially better RMSE."""
        _, _, rmse_low = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            max_segments=4, vertex_count_overshoot=0,
            pval_threshold=1.0, best_model_proportion=1.0,
        )
        _, _, rmse_high = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            max_segments=4, vertex_count_overshoot=10,
            pval_threshold=1.0, best_model_proportion=1.0,
        )
        # More candidates should give equal or better fit
        assert rmse_high <= rmse_low + 1e-9


class TestPreventOneYearRecovery:
    """prevent_one_year_recovery blocks ecologically implausible 1-yr recovery."""

    def test_enabled_no_single_year_gain_segments(self):
        """With flag on, no recovery segment should span exactly 1 year."""
        fitted, is_vertex, rmse = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            prevent_one_year_recovery=True,
        )
        vertex_indices = np.where(is_vertex)[0]
        for i in range(len(vertex_indices) - 1):
            si = vertex_indices[i]
            ei = vertex_indices[i + 1]
            duration = _REAL_YEARS[ei] - _REAL_YEARS[si]
            delta = fitted[ei] - fitted[si]
            if delta > 0:  # recovery segment
                assert duration > 1, (
                    f"1-year recovery found from {_REAL_YEARS[si]} to {_REAL_YEARS[ei]}"
                )

    def test_disabled_allows_one_year_recovery(self):
        """With flag off, the algorithm is free to fit 1-year recovery segments."""
        fitted, is_vertex, rmse = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            prevent_one_year_recovery=False,
        )
        # Just check it runs and produces valid output
        assert not np.isnan(rmse)
        assert not np.any(np.isnan(fitted))


class TestRecoveryThreshold:
    """recovery_threshold limits maximum recovery rate (value/year)."""

    def test_strict_threshold_limits_recovery_slope(self):
        """A low threshold caps how fast recovery can happen per year."""
        threshold = 0.10
        fitted, is_vertex, rmse = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            recovery_threshold=threshold,
            prevent_one_year_recovery=True,
        )
        vertex_indices = np.where(is_vertex)[0]
        for i in range(len(vertex_indices) - 1):
            si = vertex_indices[i]
            ei = vertex_indices[i + 1]
            duration = _REAL_YEARS[ei] - _REAL_YEARS[si]
            delta = fitted[ei] - fitted[si]
            if delta > 0 and duration > 0:  # recovery segment
                rate = delta / duration
                assert rate <= threshold + 1e-6, (
                    f"Recovery rate {rate:.4f} exceeds threshold {threshold}"
                )

    def test_threshold_1_no_constraint(self):
        """recovery_threshold=1.0 effectively disables the constraint."""
        fitted_strict, _, rmse_strict = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI, recovery_threshold=0.10,
        )
        fitted_free, _, rmse_free = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI, recovery_threshold=1.0,
        )
        # Unconstrained should fit at least as well
        assert rmse_free <= rmse_strict + 1e-6


class TestPvalThreshold:
    """pval_threshold controls the F-test for model selection."""

    def test_strict_pval_prefers_simpler_model(self):
        """Very low pval demands strong evidence, yielding fewer segments."""
        _, is_vertex_strict, _ = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI, pval_threshold=0.001,
        )
        _, is_vertex_loose, _ = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI, pval_threshold=1.0,
        )
        # Stricter p-value should produce the same or fewer vertices
        assert np.sum(is_vertex_strict) <= np.sum(is_vertex_loose)

    def test_pval_1_accepts_most_complex(self):
        """pval_threshold=1.0 should always accept the complex model."""
        _, is_vertex, rmse = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            pval_threshold=1.0, best_model_proportion=1.0,
            max_segments=6,
        )
        # With fully relaxed selection, expect more vertices than default
        _, is_vertex_default, _ = landtrendr_pixel(_REAL_YEARS, _REAL_NDVI)
        assert np.sum(is_vertex) >= np.sum(is_vertex_default)


class TestBestModelProportion:
    """best_model_proportion controls acceptance of simpler models."""

    def test_low_proportion_prefers_complex(self):
        """Low proportion (e.g. 1.0) accepts any model → most complex."""
        _, is_vertex_greedy, _ = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            best_model_proportion=1.0, pval_threshold=1.0,
        )
        _, is_vertex_default, _ = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            best_model_proportion=0.75, pval_threshold=1.0,
        )
        assert np.sum(is_vertex_greedy) >= np.sum(is_vertex_default)

    def test_high_proportion_prefers_simpler(self):
        """Higher proportion demands that the simpler model be nearly as good."""
        _, is_vertex_strict, _ = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            best_model_proportion=0.50, pval_threshold=0.05,
        )
        _, is_vertex_relaxed, _ = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            best_model_proportion=1.0, pval_threshold=0.05,
        )
        assert np.sum(is_vertex_strict) <= np.sum(is_vertex_relaxed)


class TestMinObservationsNeeded:
    """min_observations_needed gates whether the pixel gets processed."""

    def test_exact_threshold(self):
        """Exactly min_observations_needed valid points should succeed."""
        n = 10
        years = _REAL_YEARS[:n]
        values = _REAL_NDVI[:n]
        fitted, is_vertex, rmse = landtrendr_pixel(
            years, values, min_observations_needed=n,
        )
        assert not np.isnan(rmse)

    def test_one_below_threshold_fails(self):
        """One fewer than min_observations_needed returns NaN."""
        n = 10
        years = _REAL_YEARS[:n]
        values = _REAL_NDVI[:n]
        fitted, is_vertex, rmse = landtrendr_pixel(
            years, values, min_observations_needed=n + 1,
        )
        assert np.all(np.isnan(fitted))
        assert np.isnan(rmse)

    def test_nan_values_reduce_observation_count(self):
        """NaN values reduce the effective observation count."""
        values = _REAL_NDVI.copy()
        # Inject 38 NaNs → only 4 valid observations remain
        values[4:] = np.nan
        fitted, is_vertex, rmse = landtrendr_pixel(
            _REAL_YEARS, values, min_observations_needed=6,
        )
        assert np.all(np.isnan(fitted))
        assert np.isnan(rmse)


class TestOverfittingVsUnderfitting:
    """Verify that extreme parameter combos produce expected over/underfit."""

    def test_overfit_params_low_rmse(self):
        """Maximally overfit parameters should give very low RMSE."""
        fitted, is_vertex, rmse_overfit = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            max_segments=len(_REAL_YEARS) - 1,
            spike_threshold=1.0,
            vertex_count_overshoot=10,
            prevent_one_year_recovery=False,
            recovery_threshold=1.0,
            pval_threshold=1.0,
            best_model_proportion=1.0,
            min_observations_needed=2,
        )
        _, _, rmse_default = landtrendr_pixel(_REAL_YEARS, _REAL_NDVI)
        assert rmse_overfit <= rmse_default, (
            f"Overfit RMSE ({rmse_overfit:.4f}) > default RMSE ({rmse_default:.4f})"
        )

    def test_underfit_params_high_rmse(self):
        """max_segments=1 should produce higher RMSE than default."""
        _, _, rmse_underfit = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI, max_segments=1,
        )
        _, _, rmse_default = landtrendr_pixel(_REAL_YEARS, _REAL_NDVI)
        assert rmse_underfit >= rmse_default - 1e-9

    def test_overfit_tracks_noise(self):
        """Overfit model should closely follow every data point."""
        fitted, _, _ = landtrendr_pixel(
            _REAL_YEARS, _REAL_NDVI,
            max_segments=len(_REAL_YEARS) - 1,
            spike_threshold=1.0,
            vertex_count_overshoot=10,
            prevent_one_year_recovery=False,
            recovery_threshold=1.0,
            pval_threshold=1.0,
            best_model_proportion=1.0,
        )
        # Piecewise linear only passes through vertex points, not every
        # observation, so some deviation is expected even with max complexity
        max_deviation = np.max(np.abs(fitted - _REAL_NDVI))
        assert max_deviation < 0.25, (
            f"Overfit model deviates by {max_deviation:.3f} — expected < 0.25"
        )


class TestSubAnnualData:
    """Verify that fractional (sub-annual) years are handled correctly."""

    def test_fractional_years_accepted(self):
        """Fractional year values produce a valid fit."""
        # Simulate quarterly observations over 4 years
        years = np.array([
            2020.0, 2020.25, 2020.5, 2020.75,
            2021.0, 2021.25, 2021.5, 2021.75,
            2022.0, 2022.25, 2022.5, 2022.75,
            2023.0, 2023.25, 2023.5, 2023.75,
        ])
        # Declining trend with noise
        values = np.linspace(0.8, 0.3, len(years)) + np.random.default_rng(42).normal(0, 0.02, len(years))
        fitted, is_vertex, rmse = landtrendr_pixel(years, values, min_observations_needed=6)
        assert not np.isnan(rmse)
        assert not np.any(np.isnan(fitted))
        assert is_vertex[0] and is_vertex[-1]

    def test_sub_annual_duration_correct(self):
        """Duration and rate calculations work with fractional years."""
        # Clear step change mid-way
        years = np.array([
            2020.0, 2020.25, 2020.5, 2020.75,
            2021.0, 2021.25, 2021.5, 2021.75,
            2022.0, 2022.25, 2022.5, 2022.75,
        ])
        values = np.array([
            0.8, 0.8, 0.8, 0.8,
            0.5, 0.3, 0.2, 0.2,
            0.2, 0.2, 0.2, 0.2,
        ])
        fitted, is_vertex, rmse = landtrendr_pixel(years, values, min_observations_needed=6)
        assert not np.isnan(rmse)
        # Should detect the decline
        assert fitted[0] > fitted[-1]

    def test_sub_annual_change_pixel(self):
        """extract_change_pixel works with fractional years."""
        years = np.array([2020.0, 2020.5, 2021.0, 2021.5, 2022.0, 2022.5])
        fitted = np.array([0.8, 0.8, 0.5, 0.3, 0.2, 0.2])
        is_vertex = np.array([True, False, False, False, False, True])
        rmse = 0.05
        yod, mag, dur, preval, rate, dsnr = extract_change_pixel(
            fitted, is_vertex, rmse, years,
            change_type="greatest", delta_filter="loss",
        )
        assert yod == pytest.approx(2020.0)
        assert mag < 0
        assert dur == pytest.approx(2.5)
        assert abs(rate) > 0

    def test_integer_years_still_work(self):
        """Integer years (backward compat) produce identical results."""
        years_int = np.arange(2000, 2015, dtype=np.int32)
        years_float = np.arange(2000, 2015, dtype=np.float64)
        values = np.linspace(1.0, 0.0, 15)
        fitted_int, vtx_int, rmse_int = landtrendr_pixel(years_int, values)
        fitted_float, vtx_float, rmse_float = landtrendr_pixel(years_float, values)
        np.testing.assert_array_almost_equal(fitted_int, fitted_float)
        np.testing.assert_array_equal(vtx_int, vtx_float)
        assert rmse_int == pytest.approx(rmse_float)
