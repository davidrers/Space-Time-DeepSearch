#!/usr/bin/env python3
"""
GEE vs Native LandTrendr Comparison — Figure & Metric Generator

Runs both the Google Earth Engine and native space_time_deepsearch LandTrendr
implementations on two test sites (Oregon forest, Brazil mining), generates
comparison figures, and prints quantitative metrics.

Usage:
    python generate_comparison_report.py --gee-project YOUR_PROJECT_ID

Prerequisites:
    pip install earthengine-api geemap
    earthengine authenticate
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless backend — no display needed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 — used via xr accessor
import xarray as xr
from pyproj import Transformer
from scipy.stats import pearsonr

import ee

# ── Native LandTrendr imports ──
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from space_time_deepsearch.core import SpaceTimeDeepSearch
from space_time_deepsearch.temporal import LandTrendrParams

# =============================================================================
# Configuration
# =============================================================================

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "figure.facecolor": "white"})

SITES = {
    "Oregon forest": {
        "lon": -122.8848,
        "lat": 43.7929,
        "start_day": "06-01",
        "end_day": "10-31",
        "expected": "Abrupt disturbance ~1997",
    },
    "Brazil mining": {
        "lon": -56.61152,
        "lat": -6.84313,
        "start_day": "06-01",
        "end_day": "10-31",
        "expected": "Continuous degradation",
    },
}

INDEX = "NBR"
START_YEAR = 1985
END_YEAR = 2024
AOI_HALF_WIDTH_DEG = 0.015  # ~3 km / 2 at mid-latitudes

# Matched LandTrendr parameters (native defaults applied to both)
LT_PARAMS = {
    "maxSegments": 6,
    "spikeThreshold": 0.9,
    "vertexCountOvershoot": 3,
    "preventOneYearRecovery": True,
    "recoveryThreshold": 0.25,
    "pvalThreshold": 0.25,
    "bestModelProportion": 0.75,
    "minObservationsNeeded": 6,
}

NATIVE_PARAMS = LandTrendrParams(
    max_segments=6,
    spike_threshold=0.9,
    vertex_count_overshoot=3,
    prevent_one_year_recovery=True,
    recovery_threshold=0.25,
    pval_threshold=0.25,
    best_model_proportion=0.75,
    min_observations_needed=6,
)

GEE_BAND_NAMES = ["yod", "mag", "dur", "preval", "rate", "dsnr"]


def make_bbox(lon, lat, hw=AOI_HALF_WIDTH_DEG):
    """Create a (west, south, east, north) bbox around a point."""
    return (lon - hw, lat - hw, lon + hw, lat + hw)


def savefig(fig, filename):
    """Save figure to FIGURES_DIR and close it."""
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# GEE Helper Functions
# =============================================================================


def apply_scale_factors(image):
    """Apply Landsat C2 L2 surface reflectance scale factors.

    C2 L2 SR bands are stored as uint16 with scale=0.0000275, offset=-0.174.
    Must be applied BEFORE computing spectral indices (normalizedDifference
    is non-linear, so the offset doesn't cancel out).
    """
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.174)
    return image.addBands(optical, overwrite=True)


def rename_tm_bands(image):
    """Rename TM/ETM+ bands to match OLI naming convention.

    TM/ETM+: B1=Blue, B2=Green, B3=Red, B4=NIR, B5=SWIR1, B7=SWIR2
    OLI:     B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR1, B7=SWIR2

    Without this, calc_nbr(['SR_B5','SR_B7']) computes (SWIR1-SWIR2)
    instead of (NIR-SWIR2) for Landsat 5/7.
    """
    return image.select(
        ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7", "QA_PIXEL"],
        ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL"],
    )


def mask_landsatsr(image):
    """Mask clouds, shadows, and snow from Landsat SR using QA_PIXEL."""
    qa = image.select("QA_PIXEL")
    cloud_shadow = qa.bitwiseAnd(1 << 3).eq(0)
    cloud = qa.bitwiseAnd(1 << 4).eq(0)
    snow = qa.bitwiseAnd(1 << 5).eq(0)
    mask = cloud_shadow.And(cloud).And(snow)
    return image.updateMask(mask)


def calc_nbr(image):
    """NBR = (NIR - SWIR2) / (NIR + SWIR2).

    Assumes bands have been renamed to OLI convention (SR_B5=NIR)
    and scale factors have been applied.
    """
    return image.normalizedDifference(["SR_B5", "SR_B7"]).rename("NBR")


def build_landsat_collection(aoi, start_year, end_year):
    """Merge LT05 / LE07 / LC08 / LC09 with band renaming, cloud masking,
    and scale factor application. No cross-sensor harmonization."""
    collections = []

    # Landsat 5 TM (1984-2012) — rename bands to OLI convention
    lt5 = (
        ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
        .filterBounds(aoi)
        .map(rename_tm_bands)
        .map(mask_landsatsr)
        .map(apply_scale_factors)
    )
    collections.append(lt5)

    # Landsat 7 ETM+ (1999-2003-05-31 only, exclude SLC-off) — rename bands
    lt7 = (
        ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate("1999-01-01", "2003-05-31")
        .map(rename_tm_bands)
        .map(mask_landsatsr)
        .map(apply_scale_factors)
    )
    collections.append(lt7)

    # Landsat 8 OLI (2013-present) — bands already in OLI convention
    lt8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(aoi)
        .map(mask_landsatsr)
        .map(apply_scale_factors)
    )
    collections.append(lt8)

    # Landsat 9 OLI-2 (2021-present) — bands already in OLI convention
    lt9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(aoi)
        .map(mask_landsatsr)
        .map(apply_scale_factors)
    )
    collections.append(lt9)

    merged = collections[0]
    for c in collections[1:]:
        merged = merged.merge(c)
    return merged


def build_annual_composites(collection, start_year, end_year, start_day, end_day):
    """Build annual NBR composites: median -> x1000 -> x-1 (flip for GEE LT).

    Handles years with no valid scenes by creating a masked empty image.
    """
    images = []
    for year in range(start_year, end_year + 1):
        date_start = f"{year}-{start_day}"
        date_end = f"{year}-{end_day}"

        annual = collection.filterDate(date_start, date_end)
        nbr = annual.map(lambda img: calc_nbr(img))

        # Handle empty collections: use a masked constant as fallback
        empty = ee.Image(0).rename("NBR").selfMask()  # fully masked
        composite = ee.Algorithms.If(
            nbr.size().gt(0),
            nbr.median(),
            empty,
        )
        composite = ee.Image(composite)

        # Scale x1000 and flip for GEE LandTrendr (loss = positive)
        composite = composite.multiply(-1000).toInt16()
        composite = composite.set("system:time_start", ee.Date(date_start).millis())
        images.append(composite)

    return ee.ImageCollection.fromImages(images)


def run_gee_landtrendr(composites, params):
    """Run ee.Algorithms.TemporalSegmentation.LandTrendr."""
    run_params = dict(params)
    run_params["timeSeries"] = composites
    return ee.Algorithms.TemporalSegmentation.LandTrendr(**run_params)


def extract_gee_pixel(lt_result, lon, lat):
    """Extract source/fitted/vertex arrays at a point, unflip and unscale.

    Returns a dict with keys: years, source, fitted, is_vertex, rmse.
    Values are in natural NBR orientation [-1, 1].
    """
    point = ee.Geometry.Point([lon, lat])

    pixel_data = (
        lt_result.select("LandTrendr")
        .reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=30)
        .getInfo()
    )
    lt_array = pixel_data["LandTrendr"]

    years = np.array(lt_array[0], dtype=np.int32)
    source = np.array(lt_array[1], dtype=np.float64)
    fitted = np.array(lt_array[2], dtype=np.float64)
    is_vertex = np.array(lt_array[3], dtype=bool)

    # Unflip (GEE flipped x-1) and unscale (GEE scaled x1000)
    source = -source / 1000.0
    fitted = -fitted / 1000.0

    rmse_data = (
        lt_result.select("rmse")
        .reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=30)
        .getInfo()
    )
    rmse = rmse_data["rmse"]

    return {
        "years": years,
        "source": source,
        "fitted": fitted,
        "is_vertex": is_vertex,
        "rmse": rmse,
    }


def download_gee_lt_and_build_change_map(lt_result, aoi, native_change, scale=30):
    """Download raw GEE LandTrendr arrays and build change map locally in numpy.

    This avoids GEE server-side array sorting issues by downloading the raw
    LandTrendr result (source, fitted, vertices per pixel) and processing
    locally — analogous to the native extract_change_map.

    Returns an xr.Dataset matching the native change map's grid and conventions
    (natural orientation, unscaled, YOD adjusted).
    """
    # The LandTrendr result has: LandTrendr (4 x nYears array) and rmse.
    # Its default projection is 1° — must reproject to 30m before sampleRectangle.
    lt_img = lt_result.select("LandTrendr")
    rmse_img = lt_result.select("rmse").reproject(crs="EPSG:4326", scale=scale)

    # Download RMSE (scalar band)
    rmse_sampled = rmse_img.clip(aoi).sampleRectangle(region=aoi, defaultValue=0)
    rmse_arr = np.array(rmse_sampled.getInfo()["properties"]["rmse"], dtype=np.float64)

    # Detect actual array length (LandTrendr may drop fully-masked trailing years)
    arr_len = lt_img.arrayLength(1).reproject(crs="EPSG:4326", scale=scale)
    arr_len_val = int(arr_len.reduceRegion(
        reducer=ee.Reducer.mode(), geometry=aoi, scale=scale
    ).getInfo()["LandTrendr"])
    year_range = list(range(START_YEAR, START_YEAR + arr_len_val))
    print(f"  LT array length: {arr_len_val} years ({year_range[0]}-{year_range[-1]})")

    # Extract fitted (row 2) and vertex flags (row 3), flatten to multi-band,
    # reproject to 30m so sampleRectangle returns a proper grid.
    fitted_img = (
        lt_img.arraySlice(0, 2, 3).arrayProject([1])
        .arrayFlatten([[f"f{y}" for y in year_range]])
        .reproject(crs="EPSG:4326", scale=scale)
    )
    vertex_img = (
        lt_img.arraySlice(0, 3, 4).arrayProject([1])
        .arrayFlatten([[f"v{y}" for y in year_range]])
        .reproject(crs="EPSG:4326", scale=scale)
    )

    # Download flattened images
    print("  Downloading GEE fitted values...")
    fitted_info = fitted_img.clip(aoi).sampleRectangle(region=aoi, defaultValue=0).getInfo()

    print("  Downloading GEE vertex flags...")
    vertex_info = vertex_img.clip(aoi).sampleRectangle(region=aoi, defaultValue=0).getInfo()

    years = np.array(year_range)

    # Build arrays: (height, width, n_years)
    fitted_list = [np.array(fitted_info["properties"][f"f{y}"], dtype=np.float64)
                   for y in year_range]
    vertex_list = [np.array(vertex_info["properties"][f"v{y}"], dtype=np.float64)
                   for y in year_range]

    height, width = fitted_list[0].shape
    fitted_3d = np.stack(fitted_list, axis=-1)   # (H, W, T)
    vertex_3d = np.stack(vertex_list, axis=-1)   # (H, W, T)

    # Unflip and unscale: GEE values are ×-1 ×1000
    fitted_3d = -fitted_3d / 1000.0

    # Build change map locally — extract greatest loss segment per pixel
    yod = np.full((height, width), np.nan)
    mag = np.full((height, width), np.nan)
    dur = np.full((height, width), np.nan)
    preval = np.full((height, width), np.nan)
    rate = np.full((height, width), np.nan)
    dsnr = np.full((height, width), np.nan)

    for r in range(height):
        for c in range(width):
            vtx_mask = vertex_3d[r, c, :] == 1
            vtx_idx = np.where(vtx_mask)[0]
            if len(vtx_idx) < 2:
                continue

            pix_rmse = rmse_arr[r, c] / 1000.0  # unscale RMSE
            if pix_rmse <= 0:
                pix_rmse = 1e-6

            best_mag = 0
            for i in range(len(vtx_idx) - 1):
                i0, i1 = vtx_idx[i], vtx_idx[i + 1]
                seg_mag = fitted_3d[r, c, i1] - fitted_3d[r, c, i0]
                seg_dur = years[i1] - years[i0]
                if seg_mag < 0 and abs(seg_mag) > abs(best_mag):  # loss (natural)
                    best_mag = seg_mag
                    yod[r, c] = years[i0]
                    mag[r, c] = seg_mag
                    dur[r, c] = seg_dur
                    preval[r, c] = fitted_3d[r, c, i0]
                    rate[r, c] = seg_mag / seg_dur if seg_dur > 0 else 0
                    dsnr[r, c] = abs(seg_mag) / pix_rmse

    # Resize GEE arrays to match native change map dimensions
    from scipy.ndimage import zoom

    native_ref = native_change["yod"]
    native_h, native_w = native_ref.shape

    change_vars = {"yod": yod, "mag": mag, "dur": dur,
                   "preval": preval, "rate": rate, "dsnr": dsnr}

    # Nearest-neighbor resize for categorical (yod, dur), bilinear for continuous
    resized = {}
    for name, arr in change_vars.items():
        if native_h == height and native_w == width:
            resized[name] = arr
        else:
            order = 0 if name in ("yod", "dur") else 1  # nearest vs bilinear
            resized[name] = zoom(arr, (native_h / height, native_w / width), order=order)

    gee_change = xr.Dataset(
        {name: (native_ref.dims, resized[name]) for name in change_vars},
        coords=native_ref.coords,
    )

    print(f"  GEE change map: {height}x{width} -> {native_h}x{native_w} (matched to native)")
    return gee_change


# =============================================================================
# Native Pipeline
# =============================================================================


def run_native_pipeline(bbox, start_year, end_year, start_day, end_day, params=None):
    """Run the full native LandTrendr pipeline.

    Returns:
        (lt_result, change_map, annual_nbr, stds)
    """
    if params is None:
        params = NATIVE_PARAMS

    stds = SpaceTimeDeepSearch(bbox=bbox)

    data = stds.get_landsat(
        start_date=f"{start_year}-01-01",
        end_date=f"{end_year}-12-31",
        cloud_cover_max=20,
        min_coverage=70,
        mask_clouds=True,
        composite_period="Y",
        composite_start=start_day,
        composite_end=end_day,
        add_nbr=True,
        exclude_slc_off=False,
        apply_srf_correction=False,
    )

    nbr = data.sel(band="NBR", drop=True)
    print(f"  Annual composites: {len(nbr.time)}")

    lt_result = stds.run_landtrendr(
        nbr,
        composite_to_annual=False,  # already annual composites
        params=params,
    )

    change_map = stds.extract_change_map(
        lt_result, change_type="greatest", delta_filter="loss"
    )

    annual_nbr = nbr  # already composited

    return lt_result, change_map, annual_nbr, stds


# =============================================================================
# Analysis Helpers
# =============================================================================


def compute_spatial_stats(native_change, gee_change):
    """Compute comparison statistics between native and GEE change maps."""
    stats = {}
    for var in ["yod", "mag", "dur"]:
        n_vals = native_change[var].values.flatten()
        g_vals = gee_change[var].values.flatten()
        valid = ~np.isnan(n_vals) & ~np.isnan(g_vals)
        if var == "yod":
            valid &= (n_vals > 0) & (g_vals > 0)

        n_valid = valid.sum()
        if n_valid < 2:
            stats[var] = {"n_valid": int(n_valid)}
            continue

        nv = n_vals[valid]
        gv = g_vals[valid]

        r_val = float(np.corrcoef(nv, gv)[0, 1]) if n_valid > 1 else np.nan
        mae = float(np.mean(np.abs(nv - gv)))

        extra = {}
        if var == "yod":
            extra["pct_agree_pm1"] = float(np.mean(np.abs(nv - gv) <= 1) * 100)
            extra["pct_agree_exact"] = float(np.mean(nv == gv) * 100)

        stats[var] = {
            "n_valid": int(n_valid),
            "correlation": r_val,
            "MAE": mae,
            "native_mean": float(np.mean(nv)),
            "gee_mean": float(np.mean(gv)),
            **extra,
        }

    return stats


def extract_gee_pixel_yod_mag_dur(gee_pixel):
    """Extract best loss segment's YOD/mag/dur from GEE pixel vertex data."""
    vtx_years = gee_pixel["years"][gee_pixel["is_vertex"]]
    vtx_fitted = gee_pixel["fitted"][gee_pixel["is_vertex"]]
    segments = []
    for i in range(len(vtx_years) - 1):
        delta = vtx_fitted[i + 1] - vtx_fitted[i]
        dur = vtx_years[i + 1] - vtx_years[i]
        if delta < 0:  # loss in natural orientation
            segments.append((vtx_years[i], delta, dur))
    if segments:
        best = max(segments, key=lambda s: abs(s[1]))
        return best[0], best[1], best[2]
    return np.nan, np.nan, np.nan


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_side_by_side_trajectories(
    native_years, native_source, native_fitted, native_vtx,
    gee_pixel, site_name, site_coords, filename,
):
    """Figure: Side-by-side native (left) vs GEE (right) pixel trajectories."""
    gee = gee_pixel
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    # Native (left)
    ax = axes[0]
    ax.scatter(native_years, native_source, c="gray", s=20, zorder=3, label="Source")
    ax.plot(native_years, native_fitted, c="tab:red", lw=2, label="Fitted")
    if native_vtx.any():
        ax.scatter(
            native_years[native_vtx], native_fitted[native_vtx],
            c="tab:red", marker="^", s=70, zorder=4,
            edgecolors="black", linewidths=0.5, label="Vertices",
        )
    ax.set_title("Native LandTrendr — NBR", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("NBR")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # GEE (right)
    ax = axes[1]
    ax.scatter(gee["years"], gee["source"], c="gray", s=20, zorder=3, label="Source")
    ax.plot(gee["years"], gee["fitted"], c="tab:blue", lw=2, label="Fitted")
    vtx = gee["is_vertex"]
    if vtx.any():
        ax.scatter(
            gee["years"][vtx], gee["fitted"][vtx],
            c="tab:blue", marker="^", s=70, zorder=4,
            edgecolors="black", linewidths=0.5, label="Vertices",
        )
    ax.set_title("GEE LandTrendr — NBR", fontsize=13)
    ax.set_xlabel("Year")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{site_name} ({site_coords[0]}, {site_coords[1]})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, filename)


def plot_overlay_trajectories(
    native_years, native_source, native_fitted,
    gee_pixel, site_name, filename,
):
    """Figure: Overlay of both fitted trajectories on common years."""
    gee = gee_pixel
    common_years = np.intersect1d(native_years, gee["years"])
    native_idx = np.isin(native_years, common_years)
    gee_idx = np.isin(gee["years"], common_years)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        native_years[native_idx], native_fitted[native_idx],
        c="tab:red", lw=2, label="Native fitted", zorder=2,
    )
    ax.plot(
        gee["years"][gee_idx], gee["fitted"][gee_idx],
        c="tab:blue", lw=2, label="GEE fitted", zorder=2,
    )
    ax.scatter(native_years, native_source, c="salmon", s=12, alpha=0.5, label="Native source")
    ax.scatter(gee["years"], gee["source"], c="lightskyblue", s=12, alpha=0.5, label="GEE source")

    ax.set_xlabel("Year")
    ax.set_ylabel("NBR")
    ax.set_title(f"{site_name} — Fitted Trajectory Overlay", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fig, filename)


def plot_change_map_grid(native_change, gee_change, site_name, filename):
    """Figure: 3x3 grid comparing YOD, magnitude, duration (native/GEE/diff)."""
    variables = ["yod", "mag", "dur"]
    cmaps = {"yod": "plasma", "mag": "Reds", "dur": "YlOrBr"}
    vlims = {
        "yod": (START_YEAR, END_YEAR),
        "mag": (None, None),
        "dur": (0, None),
    }
    titles_row = ["YOD", "Magnitude", "Duration"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    for col, var in enumerate(variables):
        native_data = native_change[var].values
        gee_data = gee_change[var].values
        diff_data = native_data - gee_data

        cmap = cmaps[var]
        vmin, vmax = vlims[var]

        # Row 0: Native
        ax = axes[0, col]
        im = ax.imshow(native_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        ax.set_title(f"Native — {titles_row[col]}", fontsize=11)
        if col == 0:
            ax.set_ylabel("Native")

        # Row 1: GEE
        ax = axes[1, col]
        im = ax.imshow(gee_data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        ax.set_title(f"GEE — {titles_row[col]}", fontsize=11)
        if col == 0:
            ax.set_ylabel("GEE")

        # Row 2: Difference (Native - GEE)
        ax = axes[2, col]
        abs_max = np.nanmax(np.abs(diff_data))
        if abs_max == 0:
            abs_max = 1
        im = ax.imshow(
            diff_data, cmap="RdBu", vmin=-abs_max, vmax=abs_max, interpolation="nearest"
        )
        plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        ax.set_title(f"Diff (Native - GEE) — {titles_row[col]}", fontsize=11)
        if col == 0:
            ax.set_ylabel("Difference")

    fig.suptitle(
        f"{site_name} — Change Map Comparison", fontsize=15, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    savefig(fig, filename)


def plot_cross_site_scatter(all_site_data, filename):
    """Figure: Native vs GEE scatter for YOD, magnitude, duration across sites."""
    variables = ["yod", "mag", "dur"]
    labels = ["Year of Detection", "Magnitude", "Duration (years)"]
    colors = {"Oregon forest": "tab:green", "Brazil mining": "tab:orange"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for col, (var, label) in enumerate(zip(variables, labels)):
        ax = axes[col]

        for site_name, site_data in all_site_data.items():
            native_vals = site_data["native_change"][var].values.flatten()
            gee_vals = site_data["gee_change"][var].values.flatten()
            valid = ~np.isnan(native_vals) & ~np.isnan(gee_vals)
            if var == "yod":
                valid &= (native_vals > 0) & (gee_vals > 0)

            if valid.sum() > 0:
                nv = native_vals[valid]
                gv = gee_vals[valid]
                ax.scatter(
                    gv, nv, c=colors[site_name], s=4, alpha=0.3, label=site_name,
                )

        # 1:1 line
        lims = [ax.get_xlim(), ax.get_ylim()]
        lo = min(lims[0][0], lims[1][0])
        hi = max(lims[0][1], lims[1][1])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"GEE {label}")
        ax.set_ylabel(f"Native {label}")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, markerscale=3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    fig.suptitle(
        "Cross-Site: Native vs GEE Spatial Agreement", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    savefig(fig, filename)


# =============================================================================
# Per-Site Pipeline Runner
# =============================================================================


def run_site(site_name, site_info, fig_prefix):
    """Run both pipelines for a site, generate figures, return results."""
    lon, lat = site_info["lon"], site_info["lat"]
    bbox = make_bbox(lon, lat)
    aoi = ee.Geometry.Rectangle(list(bbox))

    print(f"\n{'=' * 60}")
    print(f"Site: {site_name} ({lon}, {lat})")
    print(f"Expected: {site_info['expected']}")
    print(f"{'=' * 60}")

    # ── Native pipeline ──
    print("\n[Native] Running pipeline...")
    native_lt, native_change, native_annual, stds = run_native_pipeline(
        bbox=bbox,
        start_year=START_YEAR,
        end_year=END_YEAR,
        start_day=site_info["start_day"],
        end_day=site_info["end_day"],
    )

    # Project lon/lat to the native UTM CRS so we select the same physical pixel as GEE
    native_crs = native_lt.rio.crs
    transformer = Transformer.from_crs("EPSG:4326", native_crs, always_xy=True)
    cx, cy = transformer.transform(lon, lat)
    native_px = native_lt.sel(y=cy, x=cx, method="nearest")
    native_years_raw = native_px.time.values
    # Convert datetime64 to integer years for compatibility with GEE int32 years
    if np.issubdtype(native_years_raw.dtype, np.datetime64):
        native_years = native_years_raw.astype("datetime64[Y]").astype(int) + 1970
    else:
        native_years = native_years_raw.astype(np.int32)
    native_source = native_px["source_values"].values
    native_fitted = native_px["fitted_values"].values
    native_vtx = native_px["is_vertex"].values.astype(bool)
    native_rmse = float(native_px["rmse"].values)
    print(f"  Native years: {native_years[[0, -1]]}, RMSE: {native_rmse:.6f}, vertices: {native_vtx.sum()}")

    # ── GEE pipeline ──
    print("\n[GEE] Building collection...")
    gee_collection = build_landsat_collection(aoi, START_YEAR, END_YEAR)
    gee_composites = build_annual_composites(
        gee_collection, START_YEAR, END_YEAR, site_info["start_day"], site_info["end_day"]
    )

    print("[GEE] Running LandTrendr...")
    gee_lt = run_gee_landtrendr(gee_composites, LT_PARAMS)

    print("[GEE] Extracting pixel...")
    gee_pixel = extract_gee_pixel(gee_lt, lon, lat)
    print(f"  GEE years: {gee_pixel['years'][[0, -1]]}, RMSE: {gee_pixel['rmse']}, vertices: {gee_pixel['is_vertex'].sum()}")

    # ── Generate trajectory figures (don't need change map yet) ──
    print("\n[Figures] Generating trajectories...")

    plot_side_by_side_trajectories(
        native_years, native_source, native_fitted, native_vtx,
        gee_pixel, site_name, (lon, lat),
        f"{fig_prefix}_trajectories.png",
    )

    plot_overlay_trajectories(
        native_years, native_source, native_fitted,
        gee_pixel, site_name,
        f"{fig_prefix}_overlay.png",
    )

    # ── Download GEE LT arrays and build change map locally ──
    print("\n[GEE] Downloading LT arrays and building change map...")
    gee_change = download_gee_lt_and_build_change_map(gee_lt, aoi, native_change)

    # Figure: 3x3 change map grid
    plot_change_map_grid(
        native_change, gee_change, site_name,
        f"{fig_prefix}_change_maps.png",
    )

    # ── Quantitative comparison ──
    common_years = np.intersect1d(native_years, gee_pixel["years"])
    native_idx = np.isin(native_years, common_years)
    gee_idx = np.isin(gee_pixel["years"], common_years)

    native_fit_common = native_fitted[native_idx]
    gee_fit_common = gee_pixel["fitted"][gee_idx]

    r, _ = pearsonr(native_fit_common, gee_fit_common)
    rmse_between = np.sqrt(np.mean((native_fit_common - gee_fit_common) ** 2))

    native_yod = float(native_change["yod"].sel(y=cy, x=cx, method="nearest").values)
    native_mag = float(native_change["mag"].sel(y=cy, x=cx, method="nearest").values)
    native_dur = float(native_change["dur"].sel(y=cy, x=cx, method="nearest").values)

    gee_yod, gee_mag, gee_dur = extract_gee_pixel_yod_mag_dur(gee_pixel)

    print(f"\n{'=' * 60}")
    print(f"{site_name} — Pixel Comparison Metrics")
    print(f"{'=' * 60}")
    print(f"{'Metric':<30} {'Native':>12} {'GEE':>12}")
    print("-" * 60)
    print(f"{'Pearson r (fitted)':<30} {r:>12.4f}")
    print(f"{'RMSE between fits':<30} {rmse_between:>12.6f}")
    print(f"{'Pixel RMSE':<30} {native_rmse:>12.6f} {gee_pixel['rmse']:>12.1f}")
    print(f"{'YOD':<30} {native_yod:>12.0f} {gee_yod:>12.0f}")
    print(f"{'Magnitude':<30} {native_mag:>12.4f} {gee_mag:>12.4f}")
    print(f"{'Duration (years)':<30} {native_dur:>12.0f} {gee_dur:>12.0f}")
    print(f"{'Vertex count':<30} {native_vtx.sum():>12d} {gee_pixel['is_vertex'].sum():>12d}")
    print(f"{'YOD difference':<30} {native_yod - gee_yod:>12.0f}")
    print("=" * 60)

    # Spatial statistics
    spatial_stats = compute_spatial_stats(native_change, gee_change)

    print(f"\n{'=' * 65}")
    print(f"{site_name} — Spatial Change Map Statistics")
    print(f"{'=' * 65}")
    for var, s in spatial_stats.items():
        print(f"\n{var.upper()}:")
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k:<25} {v:>10.4f}")
            else:
                print(f"  {k:<25} {v:>10}")
    print("=" * 65)

    return {
        "native_lt": native_lt,
        "native_change": native_change,
        "gee_pixel": gee_pixel,
        "gee_change": gee_change,
        "pixel_r": r,
        "pixel_rmse_between": rmse_between,
        "spatial_stats": spatial_stats,
        "native_rmse": native_rmse,
        "native_yod": native_yod,
        "native_mag": native_mag,
        "native_dur": native_dur,
        "gee_yod": gee_yod,
        "gee_mag": gee_mag,
        "gee_dur": gee_dur,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="GEE vs Native LandTrendr comparison — generate figures and metrics"
    )
    parser.add_argument(
        "--gee-project",
        default=os.environ.get("GEE_PROJECT"),
        help="Google Earth Engine cloud project ID (or set GEE_PROJECT env var)",
    )
    args = parser.parse_args()

    print("Initializing Google Earth Engine...")
    if args.gee_project:
        ee.Initialize(project=args.gee_project)
    else:
        ee.Initialize()

    # ── Run both sites ──
    all_site_data = {}

    # Site 1: Oregon
    all_site_data["Oregon forest"] = run_site(
        "Oregon forest", SITES["Oregon forest"], "fig1"
    )

    # Site 2: Brazil
    all_site_data["Brazil mining"] = run_site(
        "Brazil mining", SITES["Brazil mining"], "fig2"
    )

    # ── Cross-site scatter plot ──
    print("\n[Cross-site] Generating scatter plot...")
    plot_cross_site_scatter(all_site_data, "fig7_cross_site_scatter.png")

    # ── Cross-site summary table ──
    summary_rows = []
    for site_name, data in all_site_data.items():
        row = {
            "Site": site_name,
            "Pixel r (fitted)": data["pixel_r"],
            "Pixel RMSE (fits)": data["pixel_rmse_between"],
        }
        for var in ["yod", "mag", "dur"]:
            if "correlation" in data["spatial_stats"][var]:
                row[f"{var.upper()} r (spatial)"] = data["spatial_stats"][var]["correlation"]
                row[f"{var.upper()} MAE"] = data["spatial_stats"][var]["MAE"]
            if "pct_agree_pm1" in data["spatial_stats"][var]:
                row["YOD agree +/-1yr (%)"] = data["spatial_stats"][var]["pct_agree_pm1"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("Site")
    print(f"\n{'=' * 80}")
    print("Cross-Site Summary")
    print(f"{'=' * 80}")
    print(summary_df.to_string(float_format="{:.4f}".format))
    print(f"{'=' * 80}")

    print(f"\nAll figures saved to: {os.path.abspath(FIGURES_DIR)}")
    print("Done.")


if __name__ == "__main__":
    main()
