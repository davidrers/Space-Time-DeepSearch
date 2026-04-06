"""
NDVI Timelapse Animation Example
=================================

Creates a GIF animation of monthly NDVI composites using the
SpaceTimeDeepSearch core class.

Pipeline:
  1. Fetch Sentinel-2 imagery with NDVI for a one-year window
  2. Select the NDVI band
  3. Generate a timelapse GIF via stds.animate()

Usage:
    poetry run python examples/ndvi_animation.py

Requires an internet connection (queries Microsoft Planetary Computer).
"""

from space_time_deepsearch.core import SpaceTimeDeepSearch

# ── 1. Define area and time range ──────────────────────────────────────
# Small bbox over Vondelpark / Amsterdam — good seasonal NDVI variation.
BBOX = (4.85, 52.34, 4.90, 52.37)   # (west, south, east, north) WGS-84
START = "2023-01-01"
END   = "2023-12-31"

print("=" * 60)
print("Step 1: Fetch Sentinel-2 NDVI datacube (monthly composites)")
print("=" * 60)

stds = SpaceTimeDeepSearch(bbox=BBOX)

# Monthly composites, with NDVI, cloud-masked
s2 = stds.get_sentinel2(
    start_date=START,
    end_date=END,
    cloud_cover_max=30,
    min_coverage=50,
    mask_clouds=True,
    composite_period="1M",   # one composite per month
    add_ndvi=True,
)

print(f"\nSentinel-2 datacube: {s2.dims}  shape={s2.shape}")
print(f"Bands available   : {list(s2.band.values)}")
print(f"Time steps        : {len(s2.time)}")

# ── 2. Animate NDVI via the core class ─────────────────────────────────
print("\n" + "=" * 60)
print("Step 2: Generate NDVI timelapse GIF")
print("=" * 60)

# Option A — Use the core animate() method with source key + band selection
stds.animate(
    source="sentinel2",
    output_path="examples/ndvi_timelapse.gif",
    bands=["NDVI"],          # select just the NDVI band
    cmap="RdYlGn",           # Red→Yellow→Green colormap (natural for NDVI)
    vmin=-0.2,
    vmax=0.9,
    fps=2,                   # slow enough to read dates
    add_text=True,
    figsize=(10, 8),
    dpi=120,
)

print("\n✓ Saved to examples/ndvi_timelapse.gif")

# ── 3. Alternative — animate a custom DataArray directly ───────────────
print("\n" + "=" * 60)
print("Step 3 (optional): Animate from a standalone DataArray")
print("=" * 60)

# Extract NDVI as a (time, y, x) DataArray
ndvi = s2.sel(band="NDVI", drop=True)
print(f"NDVI DataArray: {ndvi.dims}  shape={ndvi.shape}")

# Pass the DataArray directly instead of a source key
stds.animate(
    data=ndvi,
    output_path="examples/ndvi_timelapse_direct.gif",
    cmap="RdYlGn",
    vmin=-0.2,
    vmax=0.9,
    fps=3,
    add_text=True,
    add_basemap=True,        # overlay a basemap underneath
    alpha=0.75,              # semi-transparent NDVI layer
)

print("✓ Saved to examples/ndvi_timelapse_direct.gif")
print("\nDone!")
