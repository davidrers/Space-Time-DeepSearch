"""
Pixel-level LandTrendr test using the SpaceTimeDeepSearch core entry point.

Pipeline:
  1. Fetch multi-year Sentinel-2 NDVI datacube via SpaceTimeDeepSearch
  2. Annual-composite the dense time series to one value per year
  3. Extract a single centre pixel (1-D time series)
  4. Run landtrendr_pixel directly on it (no xarray/dask overhead)
  5. Plot the source values, fitted trajectory, and detected vertices

Usage:
    python examples/landtrendr_pixel_test.py

Requires that the package is installed (poetry install).
The script makes real API calls to Microsoft Planetary Computer, so
an internet connection is needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from space_time_deepsearch.core import SpaceTimeDeepSearch
from space_time_deepsearch.temporal._landtrendr_core import landtrendr_pixel

# ---------------------------------------------------------------------------
# 1. Area and date range
#    We use a small area over Amsterdam (Netherlands) and a 5-year window to
#    collect enough annual observations for LandTrendr.  Adjust to any bbox.
# ---------------------------------------------------------------------------
BBOX = (4.85, 52.32, 4.95, 52.40)   # (west, south, east, north) in WGS84
START = "2018-01-01"
END   = "2023-12-31"

print("=" * 60)
print("Step 1: Fetch Sentinel-2 NDVI datacube (2018-2023)")
print("=" * 60)

stds = SpaceTimeDeepSearch(bbox=BBOX)

# Retrieve Sentinel-2 with NDVI across the full period.
# Using composite_period="3M" here to get one composite per quarter —
# run_landtrendr will then collapse these to annual values internally.
s2 = stds.get_sentinel2(
    start_date=START,
    end_date=END,
    cloud_cover_max=30,
    min_coverage=60,
    mask_clouds=True,
    composite_period="3M",
    add_ndvi=True,
)

# Select just the NDVI band → dims become (time, y, x)
ndvi = s2.sel(band="NDVI", drop=True)
print(f"\nNDVI datacube: shape={ndvi.shape}  time steps={len(ndvi.time)}")

# ---------------------------------------------------------------------------
# 2. Run LandTrendr on the full datacube (annual composite internally)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Run LandTrendr (annual composite + segmentation)")
print("=" * 60)

lt_result = stds.run_landtrendr(
    ndvi,
    composite_to_annual=True,
    composite_method="median",
)

print("LandTrendr dataset:\n", lt_result)

# ---------------------------------------------------------------------------
# 3. Pick the centre pixel and extract its 1-D time series
# ---------------------------------------------------------------------------
# Spatial centre of the datacube
cy = float(ndvi.y.values[len(ndvi.y) // 2])
cx = float(ndvi.x.values[len(ndvi.x) // 2])

print(f"\nCentre pixel: y={cy:.1f}  x={cx:.1f}")

# Pull the annual source and fitted values for this pixel
source_1d  = lt_result["source_values"].sel(y=cy, x=cx, method="nearest").values
fitted_1d  = lt_result["fitted_values"].sel(y=cy, x=cx, method="nearest").values
vertex_1d  = lt_result["is_vertex"].sel(y=cy, x=cx, method="nearest").values.astype(bool)
rmse_val   = float(lt_result["rmse"].sel(y=cy, x=cx, method="nearest").values)

# Extract integer years from the time coordinate of the annual composite
time_coord = lt_result["source_values"].time
if np.issubdtype(time_coord.dtype, np.datetime64):
    years = time_coord.dt.year.values.astype(np.int32)
else:
    years = time_coord.values.astype(np.int32)

print(f"Years covered: {years[0]}–{years[-1]}  ({len(years)} annual observations)")
print(f"RMSE of fit: {rmse_val:.4f}")

# ---------------------------------------------------------------------------
# 4. Re-run landtrendr_pixel directly for inspection / debugging
#    (The result should match what run_landtrendr already computed above)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: Re-run landtrendr_pixel directly on this pixel")
print("=" * 60)

fitted_direct, is_vertex_direct, rmse_direct = landtrendr_pixel(
    years=years,
    values=source_1d,
    max_segments=6,
    spike_threshold=0.9,
    vertex_count_overshoot=3,
    prevent_one_year_recovery=True,
    recovery_threshold=0.25,
    pval_threshold=0.05,
    best_model_proportion=0.75,
    min_observations_needed=6,
)

print(f"Fitted values   : {np.round(fitted_direct, 3)}")
print(f"Vertex flags    : {is_vertex_direct.astype(int)}")
print(f"Vertex years    : {years[is_vertex_direct]}")
print(f"RMSE (direct)   : {rmse_direct:.4f}")

n_segments = np.sum(is_vertex_direct) - 1
print(f"Segments detected: {n_segments}")

# ---------------------------------------------------------------------------
# 5. Extract change map and report the greatest-loss segment
# ---------------------------------------------------------------------------
change = stds.extract_change_map(lt_result, change_type="greatest", delta_filter="loss")

yod_px  = float(change["yod"].sel(y=cy, x=cx, method="nearest").values)
mag_px  = float(change["mag"].sel(y=cy, x=cx, method="nearest").values)
dur_px  = float(change["dur"].sel(y=cy, x=cx, method="nearest").values)
rate_px = float(change["rate"].sel(y=cy, x=cx, method="nearest").values)
dsnr_px = float(change["dsnr"].sel(y=cy, x=cx, method="nearest").values)

print("\n" + "=" * 60)
print("Greatest-loss segment for this pixel:")
print("=" * 60)
if not np.isnan(yod_px):
    print(f"  Year of Detection : {int(yod_px)}")
    print(f"  Magnitude         : {mag_px:.4f}")
    print(f"  Duration (years)  : {int(dur_px)}")
    print(f"  Rate (mag/yr)     : {rate_px:.4f}")
    print(f"  dSNR (mag/rmse)   : {dsnr_px:.2f}")
else:
    print("  No loss segment detected at this pixel.")

# ---------------------------------------------------------------------------
# 6. Plot results
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# --- (a) Pixel trajectory ---
ax_traj = fig.add_subplot(gs[0, :])  # full top row

ax_traj.scatter(years, source_1d, color="steelblue", s=50, zorder=3,
                label="Annual NDVI (source)")
ax_traj.plot(years, fitted_direct, color="tomato", linewidth=2.5, zorder=2,
             label="LandTrendr fit")

vtx_years = years[is_vertex_direct]
vtx_vals  = fitted_direct[is_vertex_direct]
ax_traj.scatter(vtx_years, vtx_vals, marker="^", color="tomato", s=120,
                edgecolors="black", linewidths=0.8, zorder=4, label="Vertices")

if not np.isnan(yod_px):
    ax_traj.axvline(yod_px, color="gold", linestyle="--", linewidth=1.5,
                    label=f"YOD = {int(yod_px)}")

ax_traj.set_title(f"LandTrendr Pixel Trajectory  —  y={cy:.0f}, x={cx:.0f}  |  RMSE={rmse_direct:.4f}",
                  fontsize=12)
ax_traj.set_xlabel("Year")
ax_traj.set_ylabel("NDVI")
ax_traj.legend(fontsize=9)
ax_traj.grid(True, alpha=0.3)
ax_traj.set_xticks(years)

# --- (b) Year of Detection map ---
ax_yod = fig.add_subplot(gs[1, 0])
change["yod"].plot(ax=ax_yod, cmap="plasma", add_colorbar=True)
ax_yod.scatter([cx], [cy], marker="*", color="white", s=200, zorder=5,
               edgecolors="black", linewidths=0.8, label="Pixel")
ax_yod.set_title("Year of Detection (Greatest Loss)")
ax_yod.set_xlabel("Easting")
ax_yod.set_ylabel("Northing")

# --- (c) Magnitude map ---
ax_mag = fig.add_subplot(gs[1, 1])
change["mag"].plot(ax=ax_mag, cmap="RdBu", center=0, add_colorbar=True)
ax_mag.scatter([cx], [cy], marker="*", color="black", s=200, zorder=5,
               edgecolors="white", linewidths=0.8, label="Pixel")
ax_mag.set_title("Magnitude of Change")
ax_mag.set_xlabel("Easting")
ax_mag.set_ylabel("Northing")

plt.suptitle("LandTrendr Analysis — Sentinel-2 NDVI", fontsize=13, y=1.01)
plt.savefig("examples/landtrendr_pixel_result.png", dpi=150, bbox_inches="tight")
print("\nPlot saved to examples/landtrendr_pixel_result.png")
plt.show()
