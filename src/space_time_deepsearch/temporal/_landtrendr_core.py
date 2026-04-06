"""
Pure numpy implementation of the LandTrendr temporal segmentation algorithm.

This module contains the per-pixel segmentation kernel with no xarray or dask
dependencies, making it independently testable and compatible with numba.

Reference:
    Kennedy, R.E., Yang, Z., Cohen, W.B. (2010). Detecting trends in forest
    disturbance and recovery using yearly Landsat time series: 1. LandTrendr —
    Temporal segmentation algorithms. Remote Sensing of Environment, 114(12).
"""

import numpy as np
from scipy.stats import f as f_dist


def _despike(values, spike_threshold):
    """Remove ephemeral spikes from a time series.

    A spike is a single-year anomaly where the value deviates sharply from
    both neighbors in the same direction (up-then-down or down-then-up).

    Args:
        values: 1-D array of spectral values (modified in place).
        spike_threshold: Controls filtering severity. 1.0 = no filtering,
            lower values = more aggressive. A point is flagged as a spike if
            its deviation relative to the series range exceeds (1 - spike_threshold).
    """
    if spike_threshold >= 1.0:
        return values

    values = values.copy()
    val_range = np.nanmax(values) - np.nanmin(values)
    if val_range == 0:
        return values

    threshold = (1.0 - spike_threshold) * val_range

    for i in range(1, len(values) - 1):
        left_delta = values[i] - values[i - 1]
        right_delta = values[i] - values[i + 1]

        # Both deltas same sign means spike
        if left_delta * right_delta > 0:
            spike_mag = min(abs(left_delta), abs(right_delta))
            if spike_mag > threshold:
                values[i] = (values[i - 1] + values[i + 1]) / 2.0

    return values


def _identify_initial_vertices(years, values, max_vertices):
    """Identify candidate breakpoint indices in the time series.

    Always includes the first and last index. Interior vertices are local
    extrema (direction changes). If too many are found, the most prominent
    ones are kept based on angle magnitude.

    Args:
        years: 1-D integer array of years.
        values: 1-D float array of spectral values.
        max_vertices: Maximum number of vertices to return.

    Returns:
        Sorted list of indices into the years/values arrays.
    """
    n = len(values)
    if n <= 2:
        return list(range(n))

    # Start and end are always vertices
    vertices = [0, n - 1]

    # Find local extrema (peaks and troughs)
    for i in range(1, n - 1):
        d_left = values[i] - values[i - 1]
        d_right = values[i + 1] - values[i]
        # Direction change: product of consecutive differences is negative
        if d_left * d_right < 0:
            vertices.append(i)

    vertices = sorted(set(vertices))

    # If too many vertices, prune by angle prominence
    if len(vertices) > max_vertices:
        vertices = _prune_vertices(years, values, vertices, max_vertices)

    return vertices


def _compute_vertex_angle(years, values, vertices, pos):
    """Compute the angle deflection at an interior vertex position.

    Larger values indicate more important vertices (sharper direction change).

    Args:
        years: Full years array.
        values: Full values array.
        vertices: Sorted list of vertex indices.
        pos: Position in the vertices list (must be interior, not 0 or last).

    Returns:
        Absolute angle change at this vertex.
    """
    prev_idx = vertices[pos - 1]
    curr_idx = vertices[pos]
    next_idx = vertices[pos + 1]

    dy_before = years[curr_idx] - years[prev_idx]
    dy_after = years[next_idx] - years[curr_idx]

    if dy_before == 0 or dy_after == 0:
        return 0.0

    slope_before = (values[curr_idx] - values[prev_idx]) / dy_before
    slope_after = (values[next_idx] - values[curr_idx]) / dy_after

    return abs(slope_after - slope_before)


def _prune_vertices(years, values, vertices, max_vertices):
    """Keep only the most prominent vertices up to max_vertices.

    Iteratively removes the interior vertex with the smallest angle
    deflection until the vertex count is at or below max_vertices.
    """
    vertices = list(vertices)
    while len(vertices) > max_vertices:
        # Find interior vertex with smallest angle
        min_angle = np.inf
        min_pos = -1
        for pos in range(1, len(vertices) - 1):
            angle = _compute_vertex_angle(years, values, vertices, pos)
            if angle < min_angle:
                min_angle = angle
                min_pos = pos
        if min_pos > 0:
            vertices.pop(min_pos)
        else:
            break
    return vertices


def _piecewise_linear_fit(years, values, vertex_indices):
    """Compute piecewise linear interpolation through vertex points.

    Vertex y-values are taken directly from the original values at those
    positions, and linear interpolation fills in between.

    Args:
        years: 1-D integer array of all years.
        values: 1-D float array of all values.
        vertex_indices: Sorted list of indices that are vertices.

    Returns:
        1-D array of fitted values, same length as years.
    """
    v_years = years[vertex_indices].astype(np.float64)
    v_values = values[vertex_indices]
    return np.interp(years.astype(np.float64), v_years, v_values)


def _compute_rmse(observed, fitted):
    """Root mean square error between observed and fitted values."""
    residuals = observed - fitted
    return np.sqrt(np.mean(residuals ** 2))


def _apply_recovery_constraints(years, fitted, vertex_indices,
                                 prevent_one_year_recovery, recovery_threshold):
    """Apply constraints on post-disturbance recovery segments.

    For indices where the spectral value increases (recovery), enforces:
    - No single-year recovery if prevent_one_year_recovery is True
    - Maximum recovery rate limited by recovery_threshold

    Args:
        years: Integer years array.
        fitted: Fitted values array (modified in place).
        vertex_indices: Sorted vertex index list (may be modified).
        prevent_one_year_recovery: Block 1-year recovery segments.
        recovery_threshold: Max allowed recovery rate (value/year).
            1.0 = no constraint.

    Returns:
        Tuple of (fitted, vertex_indices) after applying constraints.
    """
    if recovery_threshold >= 1.0 and not prevent_one_year_recovery:
        return fitted, vertex_indices

    fitted = fitted.copy()
    vertex_indices = list(vertex_indices)
    changed = True

    while changed:
        changed = False
        i = 0
        while i < len(vertex_indices) - 1:
            start_idx = vertex_indices[i]
            end_idx = vertex_indices[i + 1]
            duration = years[end_idx] - years[start_idx]
            delta = fitted[end_idx] - fitted[start_idx]

            # Recovery = value increasing (for loss-oriented indices like NBR,
            # where disturbance decreases value and recovery increases it)
            is_recovery = delta > 0

            if is_recovery:
                # Block 1-year recovery
                if prevent_one_year_recovery and duration <= 1 and len(vertex_indices) > 2:
                    # Remove this segment's end vertex (merge with next segment)
                    if i + 1 < len(vertex_indices) - 1:
                        vertex_indices.pop(i + 1)
                        changed = True
                        continue

                # Cap recovery rate
                if recovery_threshold < 1.0 and duration > 0:
                    max_recovery = recovery_threshold * duration
                    if abs(delta) > max_recovery:
                        fitted[end_idx] = fitted[start_idx] + np.sign(delta) * max_recovery
                        changed = True
            i += 1

        # Re-interpolate after constraint application
        if changed:
            v_years = years[vertex_indices].astype(np.float64)
            v_values = fitted[vertex_indices]
            fitted = np.interp(years.astype(np.float64), v_years, v_values)

    return fitted, vertex_indices


def _fit_models(years, values, initial_vertices, prevent_one_year_recovery,
                recovery_threshold):
    """Generate a sequence of models from most complex to simplest.

    Starting with all initial vertices, iteratively removes the least
    important interior vertex, fitting a piecewise linear model at each step.

    Args:
        years: Integer years array.
        values: Float values array.
        initial_vertices: Starting vertex indices.
        prevent_one_year_recovery: Block 1-year recovery.
        recovery_threshold: Max recovery rate.

    Returns:
        List of (vertex_indices, fitted_values, rmse) tuples, ordered
        from most complex to simplest.
    """
    models = []
    current_vertices = list(initial_vertices)

    while len(current_vertices) >= 2:
        fitted = _piecewise_linear_fit(years, values, current_vertices)

        # Apply recovery constraints
        fitted, constrained_vertices = _apply_recovery_constraints(
            years, fitted, current_vertices,
            prevent_one_year_recovery, recovery_threshold,
        )

        rmse = _compute_rmse(values, fitted)
        models.append((list(constrained_vertices), fitted.copy(), rmse))

        if len(current_vertices) <= 2:
            break

        # Find and remove least important interior vertex
        min_angle = np.inf
        min_pos = -1
        for pos in range(1, len(current_vertices) - 1):
            angle = _compute_vertex_angle(years, values, current_vertices, pos)
            if angle < min_angle:
                min_angle = angle
                min_pos = pos

        if min_pos > 0:
            current_vertices.pop(min_pos)
        else:
            break

    return models


def _select_best_model(models, n_obs, pval_threshold, best_model_proportion):
    """Select the best model using F-test and proportion criterion.

    Walks from simplest to most complex model. Accepts the simplest model
    that is statistically significant (p < threshold) and whose RMSE is
    within best_model_proportion of the overall best fit.

    Args:
        models: List of (vertices, fitted, rmse) from _fit_models.
        n_obs: Number of observations.
        pval_threshold: P-value threshold for F-test.
        best_model_proportion: Proportion criterion for model acceptance.

    Returns:
        Index into models list for the selected model.
    """
    if len(models) <= 1:
        return 0

    # Find the model with lowest RMSE
    rmses = [m[2] for m in models]
    best_rmse = min(rmses)

    # The simplest model is the last one (2 vertices, 1 segment)
    null_sse = rmses[-1] ** 2 * n_obs

    # Walk from simplest to most complex
    selected = len(models) - 1  # default to simplest

    for i in range(len(models) - 1, -1, -1):
        n_params = len(models[i][0])  # number of vertices
        model_sse = rmses[i] ** 2 * n_obs

        # Need at least more params than null to do F-test
        if n_params <= 2:
            continue

        # Degrees of freedom
        df1 = n_params - 2  # additional params vs null
        df2 = n_obs - n_params  # residual df

        if df2 <= 0 or df1 <= 0:
            continue

        # F-statistic: improvement in fit relative to residual variance
        if model_sse > 0:
            f_stat = ((null_sse - model_sse) / df1) / (model_sse / df2)
        else:
            f_stat = np.inf

        if f_stat <= 0:
            continue

        p_value = f_dist.sf(f_stat, df1, df2)

        if p_value < pval_threshold:
            # Check proportion criterion
            if best_rmse > 0 and rmses[i] <= best_rmse / best_model_proportion:
                selected = i
                break

    return selected


def landtrendr_pixel(years, values, max_segments=6, spike_threshold=0.9,
                     vertex_count_overshoot=3, prevent_one_year_recovery=True,
                     recovery_threshold=0.25, pval_threshold=0.05,
                     best_model_proportion=0.75, min_observations_needed=6):
    """Run LandTrendr temporal segmentation on a single pixel time series.

    Args:
        years: 1-D array of integer years, sorted ascending.
        values: 1-D array of float spectral values corresponding to years.
        max_segments: Maximum number of segments (vertices - 1).
        spike_threshold: Spike removal sensitivity (0-1). Higher = less filtering.
        vertex_count_overshoot: Extra vertices allowed during initial fitting.
        prevent_one_year_recovery: If True, disallow 1-year recovery segments.
        recovery_threshold: Maximum recovery rate. 1.0 = no constraint.
        pval_threshold: P-value threshold for model selection F-test.
        best_model_proportion: Proportion criterion for accepting simpler models.
        min_observations_needed: Minimum observations required to run.

    Returns:
        Tuple of:
            fitted_values: 1-D array of piecewise-linear fitted values.
            is_vertex: 1-D boolean array, True at breakpoint years.
            rmse: Scalar root-mean-square error of the selected fit.

        If the pixel has insufficient valid observations, returns
        (all-NaN array, all-False array, NaN).
    """
    years = np.asarray(years, dtype=np.int32)
    values = np.asarray(values, dtype=np.float64)
    n_total = len(years)

    # Handle NaN values
    valid_mask = ~np.isnan(values)
    valid_years = years[valid_mask]
    valid_values = values[valid_mask]

    # Insufficient observations
    if len(valid_values) < min_observations_needed:
        return (np.full(n_total, np.nan),
                np.zeros(n_total, dtype=bool),
                np.nan)

    # Stage 1: Despike
    clean_values = _despike(valid_values, spike_threshold)

    # Stage 2: Identify initial vertices
    max_vertices = max_segments + 1 + vertex_count_overshoot
    max_vertices = min(max_vertices, len(clean_values))
    initial_vertices = _identify_initial_vertices(
        valid_years, clean_values, max_vertices
    )

    # Stage 3: Fit progression (complex to simple models)
    models = _fit_models(
        valid_years, clean_values, initial_vertices,
        prevent_one_year_recovery, recovery_threshold,
    )

    if not models:
        return (np.full(n_total, np.nan),
                np.zeros(n_total, dtype=bool),
                np.nan)

    # Stage 4: Model selection
    best_idx = _select_best_model(
        models, len(valid_values), pval_threshold, best_model_proportion
    )
    best_vertices, best_fitted, best_rmse = models[best_idx]

    # Map results back to original (possibly NaN-containing) time axis
    fitted_full = np.full(n_total, np.nan)
    vertex_full = np.zeros(n_total, dtype=bool)

    # Interpolate fitted values to all original years (including NaN positions)
    v_years = valid_years[best_vertices].astype(np.float64)
    v_values = best_fitted[best_vertices]
    fitted_full = np.interp(years.astype(np.float64), v_years, v_values)

    # Mark vertex positions in the original time axis
    for vi in best_vertices:
        original_year = valid_years[vi]
        orig_idx = np.searchsorted(years, original_year)
        if orig_idx < n_total and years[orig_idx] == original_year:
            vertex_full[orig_idx] = True

    return fitted_full, vertex_full, best_rmse


def extract_change_pixel(fitted_values, is_vertex, rmse, years,
                         change_type="greatest", delta_filter="loss"):
    """Extract change map metrics from a single pixel's segmentation result.

    Args:
        fitted_values: 1-D array of fitted spectral values.
        is_vertex: 1-D boolean array marking vertices.
        rmse: Scalar RMSE of the fit.
        years: 1-D integer array of years.
        change_type: Segment selection criterion. One of:
            "greatest" (largest magnitude), "longest" (longest duration),
            "steepest" (largest rate), "newest" (most recent).
        delta_filter: Direction filter. "loss" (negative change only),
            "gain" (positive change only), or "all".

    Returns:
        Tuple of (yod, mag, dur, preval, rate, dsnr).
        Returns all-NaN if no matching segment is found.
    """
    nan_result = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    if np.all(np.isnan(fitted_values)) or not np.any(is_vertex):
        return nan_result

    # Find vertex positions
    vertex_indices = np.where(is_vertex)[0]

    if len(vertex_indices) < 2:
        return nan_result

    # Build segments
    segments = []
    for i in range(len(vertex_indices) - 1):
        start_idx = vertex_indices[i]
        end_idx = vertex_indices[i + 1]
        start_year = int(years[start_idx])
        end_year = int(years[end_idx])
        start_val = fitted_values[start_idx]
        end_val = fitted_values[end_idx]
        delta = end_val - start_val
        duration = end_year - start_year

        if duration == 0:
            continue

        rate = delta / duration
        dsnr = abs(delta) / rmse if rmse > 0 else 0.0

        segments.append({
            "yod": start_year,
            "mag": delta,
            "dur": duration,
            "preval": start_val,
            "rate": rate,
            "dsnr": dsnr,
        })

    if not segments:
        return nan_result

    # Apply delta filter
    if delta_filter == "loss":
        segments = [s for s in segments if s["mag"] < 0]
    elif delta_filter == "gain":
        segments = [s for s in segments if s["mag"] > 0]

    if not segments:
        return nan_result

    # Select segment by change_type
    if change_type == "greatest":
        selected = max(segments, key=lambda s: abs(s["mag"]))
    elif change_type == "longest":
        selected = max(segments, key=lambda s: s["dur"])
    elif change_type == "steepest":
        selected = max(segments, key=lambda s: abs(s["rate"]))
    elif change_type == "newest":
        selected = max(segments, key=lambda s: s["yod"])
    else:
        selected = max(segments, key=lambda s: abs(s["mag"]))

    return (
        float(selected["yod"]),
        float(selected["mag"]),
        float(selected["dur"]),
        float(selected["preval"]),
        float(selected["rate"]),
        float(selected["dsnr"]),
    )
