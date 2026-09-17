"""
Sentinel-1 RTC backscatter retrieval and per-parcel time series.

The radar counterpart to `sentinel2.py`. Where Sentinel-2 measures how GREEN a
canopy is, Sentinel-1 measures its STRUCTURE: a standing grass canopy scatters
C-band in volume, a freshly cut one does not, so the cross-polarised return
(VH) drops within a day of a cut. Radar sees through cloud, which is the whole
point on mountain grassland where the optical series goes dark for weeks.

This module provides two levels:

    get_sentinel1_rtc_imagery(...)   AOI -> 4D DataArray (time, band, y, x)
    parcel_series(geometry, ...)     one polygon -> one row per acquisition

Per date, over the polygon's pixels, `parcel_series` produces:

    VV      co-pol gamma0            soil + surface scattering
    VH      cross-pol gamma0         volume scattering -> standing biomass
    VH/VV   cross-ratio              biomass, largely free of soil moisture
    RVI     4*VH / (VV + VH)         normalised vegetation scattering, 0..1

each as a MEDIAN and a STD over the polygon, plus the linear mean and a pixel
count, with per-track offsets removed.

Why `sentinel-1-rtc` and not `sentinel-1-grd`
---------------------------------------------
RTC is Radiometrically Terrain Corrected gamma0. Plain GRD carries sigma0
referenced to a flat ellipsoid, so on a slope the backscatter is modulated by
the local incidence angle - on mountain grassland that topographic term is far
larger than the mowing signal, and it differs between ascending and descending
passes over the SAME field. RTC divides it out. The collection is anonymously
readable on Planetary Computer despite advertising `requires_account: true`.

Four corrections a naive port of the Sentinel-2 loader gets wrong
-----------------------------------------------------------------
1. LINEAR POWER, NEVER dB. The assets are float32 gamma0 in linear power, not
   decibels. Every reduction - median, mean, std, the ratio, RVI - happens in
   linear power and is converted to dB only for display. Averaging in dB
   averages logarithms, which by Jensen's inequality sits below the true mean
   and drifts with speckle, i.e. with pixel count, i.e. with polygon size.
2. `fill_value=np.nan`, NOT 0. The RTC nodata value is -32768 (declared in the
   assets' `raster:bands`). `get_sentinel2_imagery` passes `fill_value=0`; 0 is
   a perfectly legal reflectance but in linear power it is -inf dB and it drags
   a linear mean to zero. Gaps must stay NaN.
3. MOSAIC BY (date, relative_orbit), not by exact timestamp. The Sentinel-2
   loader groups duplicate timestamps. S1 slices from one datatake have start
   times seconds apart, so exact-timestamp grouping never fires: a polygon on a
   slice boundary silently becomes two half-covered rows on the same day.
   Measured on a 1.6 x 0.9 degree AOI, June 2024: all 8 acquisitions arrived as
   2 slices. It bites small AOIs too, and the slicing differs by YEAR - a
   0.04 x 0.03 degree Salzburg AOI has no slice pairs at all in 2024 but 13 of
   them in 2025 (101 items collapsing to 88 acquisitions). Testing only on 2024
   never exercises this path.
4. NO SPECKLE FILTER. Speckle is multiplicative with unit mean; the spatial
   reduction over the polygon IS the multi-looking. Pre-filtering with Lee or
   similar would only blur the polygon edge inwards.

Median vs mean: for speckle the LINEAR MEAN is the unbiased estimator of
gamma0, while the median sits below it (ln 2 of the mean in the single-look
limit). The bias is near-constant across dates, so it cancels in change
detection - medians are the robust headline, but `*_mean` is there when an
absolute gamma0 is wanted.

2024 is the one thin year - read this before comparing years
-------------------------------------------------------------
Sentinel-1 is designed as a TWO-satellite constellation, phased 180 deg apart
in the same plane, giving a 6-day repeat per track. It was not one in 2024.
S1B failed on 2021-12-23 and S1C did not launch until 2024-12-05, so 2024 ran
on S1A alone. Measured over one Salzburg AOI, April-October of each year:

    year   platforms      same-track revisit   pooled gap   scenes
    2021   S1A + S1B            6.00 d           2.05 d       90
    2024   S1A alone           12.00 d           4.81 d       37
    2025   S1A + S1C            6.00 d           1.80 d      101

Anything fitted per-year - track offsets, window means, change statistics - is
NOT comparable across 2024 and 2025 without accounting for this: a 14-day
post-cut window holds ~1 scene per track in 2024 but ~2 in 2025. It also feeds
`normalize_orbit`, whose `min_obs=5` needs roughly a full season per track in
2024 but only half of one in 2025.

What actually carries a cut, honestly
--------------------------------------
20 grassland parcels with a dated mowing event, comparing the mean of the 21
days before the cut with the 14 days after: the CROSS-RATIO
`vh_vv_median_db_adj` drops -0.12 dB, falling on 14/20 parcels - binomial
p ~= 0.06, i.e. NOT significant at that sample size. `vv_median_db` RISES
(exposed soil), `vh_median_db` alone is at chance. The ratio works precisely
because its two halves move in opposite directions.

Re-drawing the random sample moved that from 14/20 to 13/20 and -0.269 to
-0.158 dB across two runs of the identical script - run-to-run noise is the
same size as the effect. At a parcel-to-parcel sd of ~0.4 dB, detecting it at
80% power needs ~44 parcels. Treat the cross-ratio as one weak feature for a
multi-date change detector, ideally alongside an optical series - never as a
single-date threshold.

RVI is not an extra feature. With r = VH/VV, RVI = 4r/(1+r) is strictly
increasing in r, and the median commutes with a monotone transform, so
`rvi_median` is the transform evaluated at `vh_vv_median`. The agreement is
exact only on an ODD number of valid pixels: with an even count numpy's median
AVERAGES the two middle values, and averaging does not commute with a nonlinear
transform. Measured here on synthetic gamma-distributed pixels -

    81 px, 225 px (odd)    max abs difference   1.1e-16
    64 px, 256 px (even)   max abs difference   2.0e-05, 4.4e-08

The even-count residual shrinks as the two middle values converge, so it is
negligible on any real polygon and vanishes entirely on odd ones. RANKS ARE
IDENTICAL IN EVERY CASE, which is the part that matters: for anything that only
sees order - trees, thresholds - the two are the SAME feature and feeding both
just duplicates a column. It is returned because it is the conventional bounded
reporting scale. The `*_std` columns are NOT redundant this way: a standard
deviation is not monotone-invariant.

Example
-------
    from space_time_deepsearch.io import parcel_series, get_sentinel1_rtc_imagery

    df = parcel_series("field.geojson", "2024-03-01", "2024-11-15")
    df[["date", "relative_orbit", "vh_vv_median_db_adj"]]

    cube = get_sentinel1_rtc_imagery(bbox=(13.28, 47.32, 13.32, 47.35),
                                     start_date="2024-06-01",
                                     end_date="2024-06-30")
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry
from shapely.geometry import box

COLLECTION = "sentinel-1-rtc"
STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"

# RTC nodata, as declared in each asset's `raster:bands`. stackstac honours it
# and hands back NaN, but it is named here so the intent is not implicit.
RTC_NODATA = -32768.0


def fix_proj_env() -> None:
    """Drop PROJ_LIB / PROJ_DATA when they point somewhere without a proj.db.

    The package `__init__` already unsets PROJ_LIB for the PostgreSQL/PostGIS
    case. This is the general form: any installation that sets PROJ_LIB to a
    directory this process cannot read - another user's conda env, a removed
    toolchain - makes rasterio fail with "Cannot find proj.db". Testing for the
    file itself catches all of them and is a no-op when PROJ_LIB is valid.
    """
    for var in ("PROJ_LIB", "PROJ_DATA"):
        path = os.environ.get(var)
        if path and not Path(path, "proj.db").is_file():
            print(f"note: {var}={path} has no proj.db - unsetting it "
                  "so the bundled PROJ data is used")
            del os.environ[var]


def _utm_epsg(west, south, east, north) -> int:
    """UTM EPSG for a lat/lon bbox, as `sentinel2._get_utm_epsg` does."""
    from pyproj.aoi import AreaOfInterest
    from pyproj.database import query_utm_crs_info

    zones = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(west_lon_degree=west,
                                        south_lat_degree=south,
                                        east_lon_degree=east,
                                        north_lat_degree=north),
    )
    return int(zones[0].code)


def _as_geometry(geometry) -> shapely.geometry.base.BaseGeometry:
    """Accept a vector path, a GeoDataFrame/GeoSeries or a Shapely geometry.

    Always returns a single geometry in EPSG:4326. A bare Shapely geometry is
    ASSUMED to be in EPSG:4326 already - it carries no CRS to check.
    """
    if isinstance(geometry, (str, Path)):
        gdf = gpd.read_file(geometry)
        if gdf.crs is not None and gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        return gdf.union_all()
    if isinstance(geometry, (gpd.GeoDataFrame, gpd.GeoSeries)):
        gs = geometry.geometry if isinstance(geometry, gpd.GeoDataFrame) else geometry
        if gs.crs is not None and gs.crs != "EPSG:4326":
            gs = gs.to_crs("EPSG:4326")
        return gs.union_all()
    return geometry


# ============================================================ the loader
def get_sentinel1_rtc_imagery(
    bbox: tuple[float, float, float, float] | None = None,
    custom_geometry: str | shapely.geometry.base.BaseGeometry | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    polarizations: list[str] | None = None,
    resolution: int = 10,
    orbit_state: str | None = None,
    relative_orbit: int | None = None,
    min_coverage: float = 0.0,
    add_indices: bool = True,
    collection: str = COLLECTION,
    chunksize: int = 2048,
    verbose: bool = True,
):
    """Fetch a Sentinel-1 RTC cube for an area of interest and time range.

    Mirrors `get_sentinel2_imagery`, but for radar: no cloud filtering (there
    are no clouds), no scene classification, and the reductions stay in linear
    power.

    Args:
        bbox (tuple, optional): (west, south, east, north) in EPSG:4326.
        custom_geometry (str or shapely.geometry.BaseGeometry, optional): path
            to a vector file, or a Shapely geometry. The cube is clipped to it
            (`all_touched`, extent preserved).
        start_date (str): ISO date, "YYYY-MM-DD".
        end_date (str): ISO date, "YYYY-MM-DD".
        polarizations (list, optional): subset of ["vv", "vh"]. Defaults to
            both. Land is covered in IW dual-pol VV+VH; HH/HV exist only over
            the poles.
        resolution (int, optional): output pixel size in metres. Native RTC is
            10 m. Defaults to 10.
        orbit_state (str, optional): keep only "ascending" or "descending".
            None keeps both. NOTE this does NOT remove the track offset - see
            `normalize_orbit`.
        relative_orbit (int, optional): keep only this track number.
        min_coverage (float, optional): drop dates whose valid-pixel fraction
            over the AOI is below this (0-1). Catches polygons clipped by a
            scene edge. Defaults to 0.
        add_indices (bool, optional): append per-pixel VH/VV and RVI as extra
            bands, computed in LINEAR POWER before any spatial reduction.
            Defaults to True.
        collection (str, optional): STAC collection id. Defaults to
            "sentinel-1-rtc".
        chunksize (int, optional): dask chunk size. Defaults to 2048.
        verbose (bool, optional): print progress. Defaults to True.

    Returns:
        xarray.DataArray: 4D DataArray (time, band, y, x) in linear-power
            gamma0 - NOT decibels - with `relative_orbit`, `orbit_state` and
            `platform` as coordinates along `time`.

    Raises:
        ValueError: if neither bbox nor custom_geometry is given, or both are,
            or the dates are missing, or the search returns nothing.
    """
    # Imported here rather than at module scope: `io/__init__` pulls every
    # loader in eagerly, and stackstac + pystac_client cost seconds to import.
    import planetary_computer
    import pystac_client
    import rioxarray  # noqa: F401  - registers the .rio accessor used below
    import stackstac

    if bbox is None and custom_geometry is None:
        raise ValueError("Either bbox or custom_geometry must be provided.")
    if bbox is not None and custom_geometry is not None:
        raise ValueError("Provide either bbox or custom_geometry, not both.")
    if start_date is None or end_date is None:
        raise ValueError("Both start_date and end_date must be provided.")

    if polarizations is None:
        polarizations = ["vv", "vh"]
    polarizations = [p.lower() for p in polarizations]
    if add_indices and not {"vv", "vh"} <= set(polarizations):
        raise ValueError("add_indices needs both 'vv' and 'vh'.")

    fix_proj_env()

    # ---- AOI -------------------------------------------------------------
    if custom_geometry is not None:
        aoi = _as_geometry(custom_geometry)
        search_bbox = list(aoi.bounds)
    else:
        aoi = box(*bbox)
        search_bbox = list(bbox)

    # ---- search ----------------------------------------------------------
    catalog = pystac_client.Client.open(
        STAC_API, modifier=planetary_computer.sign_inplace)

    query = {}
    if orbit_state is not None:
        query["sat:orbit_state"] = {"eq": orbit_state}
    if relative_orbit is not None:
        query["sat:relative_orbit"] = {"eq": int(relative_orbit)}

    items = catalog.search(
        collections=[collection],
        bbox=search_bbox,
        datetime=f"{start_date}/{end_date}",
        query=query or None,
    ).item_collection()

    if len(items) == 0:
        raise ValueError(
            f"No {collection} scenes for this AOI and date range. "
            "RTC coverage is not global - check the AOI, or widen the dates.")

    epsg = _utm_epsg(*search_bbox)
    if verbose:
        tracks = pd.Series(
            [f'{i.properties["sat:relative_orbit"]} '
             f'{i.properties["sat:orbit_state"][:3]}' for i in items]
        ).value_counts().to_dict()
        print(f"Found {len(items)} {collection} scenes; tracks {tracks}; "
              f"EPSG:{epsg}")

    # ---- stack -----------------------------------------------------------
    # fill_value=np.nan, deliberately: see correction 2 in the module docstring.
    cube = stackstac.stack(
        items,
        assets=polarizations,
        bounds_latlon=search_bbox,
        resolution=resolution,
        epsg=epsg,
        # float64, not float32: stackstac refuses float32 alongside its
        # rescaling step ("safe casting cannot be completed"). The assets are
        # float32 with scale 1 / offset 0, so this only widens them.
        dtype="float64",
        fill_value=np.nan,
        chunksize=chunksize,
    )

    # ---- orbit metadata as time coordinates ------------------------------
    props = pd.DataFrame({
        "time": [i.datetime for i in items],
        "relative_orbit": [i.properties["sat:relative_orbit"] for i in items],
        "orbit_state": [i.properties["sat:orbit_state"] for i in items],
        # Upper-cased on purpose: MPC is inconsistent about this field and the
        # SAME satellite arrives under two spellings within one 2025 search -
        # 'SENTINEL-1A' on 18 scenes and 'sentinel-1a' on 26. Grouping by the
        # raw string silently splits one platform into two.
        "platform": [i.properties.get("platform", "").upper() for i in items],
    }).sort_values("time").reset_index(drop=True)
    # stackstac sorts items by time, so positional assignment is aligned.
    for col in ("relative_orbit", "orbit_state", "platform"):
        cube = cube.assign_coords({col: ("time", props[col].to_numpy())})

    # ---- clip to the exact geometry --------------------------------------
    if custom_geometry is not None:
        clip = gpd.GeoDataFrame(geometry=[aoi], crs="EPSG:4326").to_crs(epsg=epsg)
        cube = cube.rio.clip(clip.geometry, crs=clip.crs,
                             drop=False, all_touched=True)

    # ---- mosaic slices of the same acquisition ---------------------------
    cube = _mosaic_slices(cube, verbose=verbose)

    # ---- coverage filter -------------------------------------------------
    if min_coverage > 0:
        ref = cube.isel(band=0, drop=True)
        frac = ref.notnull().mean(dim=("y", "x")).compute()
        keep = frac >= min_coverage
        if verbose:
            print(f"Coverage >= {min_coverage:.0%}: keeping "
                  f"{int(keep.sum())}/{len(keep)} dates")
        cube = cube.isel(time=keep.values)

    # ---- per-pixel indices, in LINEAR POWER ------------------------------
    if add_indices:
        cube = _add_indices(cube)

    cube.attrs.update(
        source="Microsoft Planetary Computer",
        collection=collection,
        units="gamma0, linear power (NOT dB)",
        nodata=RTC_NODATA,
        start_date=start_date,
        end_date=end_date,
        resolution=resolution,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import dask.diagnostics
        with dask.diagnostics.ProgressBar():
            if verbose:
                print("Loading backscatter...")
            return cube.compute()


def _mosaic_slices(cube, verbose: bool = True):
    """Merge scenes belonging to one acquisition (same date AND same track).

    Sentinel-1 splits a datatake into slices whose start times differ by
    seconds, so grouping on the exact timestamp - what `get_sentinel2_imagery`
    does - never fires. A polygon straddling a slice boundary would otherwise
    appear twice on the same day, each row covering part of it.

    Grouping on the date alone would be wrong in the other direction: an
    ascending and a descending pass can land on the same day and must stay
    separate observations.
    """
    times = pd.to_datetime(cube.time.values)
    key = pd.Index([f"{t:%Y%m%d}_{o}" for t, o
                    in zip(times, cube.relative_orbit.values)])
    if key.is_unique:
        return cube

    n_before = len(key)
    meta = {}
    for i, k in enumerate(key):
        meta.setdefault(k, (times[i], cube.relative_orbit.values[i],
                            cube.orbit_state.values[i],
                            cube.platform.values[i]))  # first slice wins

    cube = cube.assign_coords(acq=("time", key.to_numpy()))
    # nanmedian across the slices: in the overlap both carry the same
    # acquisition, elsewhere exactly one is valid and the other is NaN.
    merged = cube.groupby("acq").median(dim="time", skipna=True)

    order = list(merged.acq.values)
    merged = merged.assign_coords(
        time=("acq", np.array([meta[k][0] for k in order],
                              dtype="datetime64[ns]")),
        relative_orbit=("acq", np.array([meta[k][1] for k in order])),
        orbit_state=("acq", np.array([meta[k][2] for k in order])),
        platform=("acq", np.array([meta[k][3] for k in order])),
    ).swap_dims({"acq": "time"}).drop_vars("acq").sortby("time")

    if verbose:
        print(f"Mosaicked {n_before} slices into {len(merged.time)} acquisitions")
    return merged


def _add_indices(cube):
    """Append VH/VV and RVI as bands, computed per pixel in linear power.

    RVI = 4*VH / (VV + VH) is the dual-pol form of the Radar Vegetation Index.
    It runs 0 -> 1: near 0 for a bare, surface-scattering field, rising as the
    canopy's volume scattering takes over. Both indices are ratios of linear
    powers - forming them from dB values instead would silently turn them into
    differences and sums of logarithms.
    """
    import xarray as xr

    vv = cube.sel(band="vv", drop=True)
    vh = cube.sel(band="vh", drop=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = vh / vv
        rvi = 4.0 * vh / (vv + vh)

    # Strip the per-scene scalar coords so concat does not try to align them.
    keep = set(cube.dims) | {"spatial_ref", "epsg", "relative_orbit",
                             "orbit_state", "platform"}

    def _strip(a):
        return a.drop_vars([c for c in a.coords if c not in keep],
                           errors="ignore")

    extra = []
    for name, arr in (("vh_vv", ratio), ("rvi", rvi)):
        arr = arr.where(np.isfinite(arr)).expand_dims(band=[name])
        extra.append(_strip(arr))

    return xr.concat([_strip(cube)] + extra, dim="band")


# ============================================================ reduction
def to_db(x):
    """Linear power -> decibels, with non-positive values sent to NaN."""
    x = np.asarray(x, dtype="float64")
    out = np.full(x.shape, np.nan)
    good = np.isfinite(x) & (x > 0)
    out[good] = 10.0 * np.log10(x[good])
    return out


def compute_features(cube, min_pixels: int = 5) -> pd.DataFrame:
    """Per-date medians, means and stds over the polygon's pixels.

    Everything is reduced in LINEAR POWER; the `_db` columns are the dB of the
    reduced linear value, never a reduction of dB values (see the module
    docstring). `*_std_db` is the exception and is meant to be one: it is the
    std of the per-pixel dB, i.e. the within-polygon heterogeneity on the scale
    where speckle is roughly additive, which makes it comparable between
    polygons of different brightness.

    Args:
        cube (xarray.DataArray): the output of `get_sentinel1_rtc_imagery`.
        min_pixels (int, optional): drop dates with fewer valid pixels than
            this in any band. Defaults to 5.

    Returns:
        pandas.DataFrame: one row per acquisition, chronological.
    """
    bands = [str(b) for b in cube.band.values]
    rows = {"date": pd.to_datetime(cube.time.values)}

    for col in ("relative_orbit", "orbit_state", "platform"):
        if col in cube.coords:
            rows[col] = cube[col].values

    n_pixels = None
    for b in bands:
        arr = cube.sel(band=b).astype("float64")
        arr = arr.where(np.isfinite(arr))
        if b in ("vv", "vh"):
            arr = arr.where(arr > 0)            # 0 / negative are not power

        rows[f"{b}_median"] = arr.median(dim=("y", "x"), skipna=True).values
        rows[f"{b}_mean"] = arr.mean(dim=("y", "x"), skipna=True).values
        rows[f"{b}_std"] = arr.std(dim=("y", "x"), skipna=True).values

        if b in ("vv", "vh"):
            db = np.log10(arr) * 10.0
            rows[f"{b}_std_db"] = db.std(dim=("y", "x"), skipna=True).values

        valid = arr.notnull().sum(dim=("y", "x")).values
        n_pixels = valid if n_pixels is None else np.minimum(n_pixels, valid)

    rows["n_pixels"] = n_pixels
    df = pd.DataFrame(rows)

    # dB of the linear medians - display only, and for VH/VV the conventional
    # "cross-ratio in dB" that the mowing literature reports.
    for b in ("vv", "vh", "vh_vv"):
        if f"{b}_median" in df:
            df[f"{b}_median_db"] = to_db(df[f"{b}_median"])

    df = df[df.n_pixels >= min_pixels]
    return df.sort_values("date").reset_index(drop=True)


DB_COLS = ("vv_median_db", "vh_median_db", "vh_vv_median_db")
ORBIT_KEYS = ("relative_orbit", "platform")


# ============================================================ track offsets
def normalize_orbit(df: pd.DataFrame, cols=DB_COLS, by=ORBIT_KEYS,
                    min_obs: int = 5, verbose: bool = True) -> pd.DataFrame:
    """Remove each track's constant offset, adding `<col>_adj` columns.

    What a "track" is. Sentinel-1 flies an exact-repeat orbit: 175 orbits fit
    precisely into 12 days (98.74 min each), so the ground path retraces itself
    every cycle. `sat:relative_orbit` says which of those 175 an acquisition
    belongs to - it is just the cumulative orbit counter wrapped into the
    cycle, `(sat:absolute_orbit - 73) mod 175 + 1` for S1A (verified exact on
    37 scenes; the constant differs for S1C, which is why this reads the field
    rather than deriving it).

    One orbit circles the whole planet, so its ascending (S->N) and descending
    (N->S) halves fall on opposite sides of it. Over a fixed polygon a given
    track is therefore ALWAYS the same pass direction - measured: track 22
    descending, 44 ascending, 95 descending, never mixed. Local solar times
    come out at 6.15 h descending and 17.87 h ascending, the sun-synchronous
    18:00 node Sentinel-1 was designed around.

    At 47.5N the 175 tracks sit 155 km apart while the IW swath is 250 km, so
    swaths overlap 1.6x; with both dawn and evening passes a polygon is seen by
    3-4 tracks. Each hits it at a different incidence angle and compass
    direction, and on a slope that means probing the canopy along a different
    line. RTC normalises the illuminated AREA, not the viewing angle, which is
    why a per-track offset survives terrain correction and has to be removed
    here.

    Why this is needed even on RTC, and why `orbit_state` is the WRONG unit:
    measured on 8 alpine and 8 lowland grassland parcels, season 2024, counting
    only tracks with >=5 scenes, after detrending each series so phenology
    cannot masquerade as a track effect -

        terrain   spread ACROSS TRACKS   worst   within-track std
        alpine        1.16 dB            3.23 dB      0.57 dB
        lowland       0.64 dB            1.64 dB      0.51 dB

    BOTH exceed their own within-track noise, and both are several times the
    mowing signal (-0.12 dB), so raw pooling buries the cut on ANY terrain -
    alpine merely worse. Normalise everywhere; this is why the default is on.

    (An earlier 5+5 run that did not exclude thinly-sampled tracks put lowland
    at 0.32 dB and suggested pooling was harmless there. It is not - that
    figure was the small-sample artefact, not this one.)

    The grouping key is (relative_orbit, PLATFORM), not the track alone,
    because the two satellites are not cross-calibrated to better than the
    signal. Measured on 12 parcels x 40 track-parcel pairs, season 2025, each
    series detrended first: within the SAME track, S1A sits +0.29 dB above S1C
    in the cross-ratio (median +0.32, sd 0.23), the same sign in 37 of 40
    pairs. That is larger than the mowing signal itself. Because A and C
    alternate along a track, any pre/post window gets a different A/C mix, so
    leaving this in injects bias at exactly signal scale. Verified to cut the
    residual offset from +0.213 to +0.008 dB. For 2024 (S1A only) the extra key
    is a no-op.

    Crucially the offset is per-TRACK, not per-DIRECTION. One alpine parcel is
    seen by tracks 44 asc, 95 desc and 117 asc; the two ASCENDING tracks, both
    well sampled (11 and 12 scenes), sit 0.96 dB apart on their own - track 44
    at -5.34 dB, track 117 at -4.38 - on top of a separate ~1.5 dB ascending-
    vs-descending contrast. Passing `orbit_state="ascending"` to the loader
    would keep that 1 dB. `relative_orbit` is therefore part of the default.

    The offset is a per-track gain, which is multiplicative in linear power and
    so ADDITIVE in dB - which is why this subtracts in the dB domain, the one
    place in this module where working in dB is the correct choice.

    A track with fewer than `min_obs` observations is left ALONE rather than
    centred: subtracting a median estimated from one or two dates would force
    those dates onto the global level and delete the very anomaly being looked
    for. Such tracks stay biased, so prefer a longer window over a short one.

    DO NOT instead drop to a single track. Tested on 20 parcels with dated cuts
    (cross-ratio, 21 days before vs 14 days after):

        mixed, raw          mean -0.12 dB   fell 12/20   mean/sd -0.25
        mixed, normalised   mean -0.12 dB   fell 14/20   mean/sd -0.29
        single track only   mean -0.05 dB   fell 13/20   mean/sd -0.06

    Normalising is the best of the three and never costs anything. Restricting
    to one track is the WORST: it drops the revisit from 2-3 days to 12, so the
    before/after windows often hold a single scene each and speckle dominates.
    The cure for track bias is removing the offset, not discarding the data.

    Args:
        df (pandas.DataFrame): the output of `compute_features`.
        cols (tuple, optional): dB columns to correct. Defaults to the three
            medians.
        by (tuple or str, optional): grouping key. Defaults to
            (relative_orbit, platform). Keys absent from `df` are ignored.
        min_obs (int, optional): leave a group un-normalised below this many
            observations. Defaults to 5.
        verbose (bool, optional): report skipped groups. Defaults to True.

    Returns:
        pandas.DataFrame: a copy with `<col>_adj` columns added, plus
            `rvi_median_adj` re-derived from the corrected cross-ratio.
    """
    df = df.copy()
    keys = [by] if isinstance(by, str) else [k for k in by if k in df]
    if not keys:
        return df

    # One groupby object drives both the counts and the loop, so the group-key
    # type stays consistent (pandas hands back scalars vs 1-tuples for a
    # single-element list depending on version).
    grouped = df.groupby(keys, sort=False)
    counts = grouped.size()
    usable = {k for k, n in counts.items() if n >= min_obs}
    if verbose and len(usable) < len(counts):
        skipped = sorted(str(k) for k in set(counts.index) - usable)
        print(f"note: group(s) {skipped} have <{min_obs} observations - left "
              "un-normalised (too few dates to estimate an offset safely)")

    for col in cols:
        if col not in df:
            continue
        adj = df[col].astype(float).copy()
        overall = adj.median()
        for gkey, grp in grouped:
            if gkey in usable:
                adj.loc[grp.index] -= (df.loc[grp.index, col].median() - overall)
        df[f"{col}_adj"] = adj

    # RVI is a monotone transform of the cross-ratio, so it is re-derived from
    # the corrected ratio rather than normalised on its own scale.
    if "vh_vv_median_db_adj" in df:
        r = 10.0 ** (df["vh_vv_median_db_adj"] / 10.0)
        df["rvi_median_adj"] = 4.0 * r / (1.0 + r)
    return df


# ============================================================ orbit modes
# The viewing-geometry configurations worth comparing. Every mode ends up with
# the same `<col>_adj` columns so a caller never has to know which was used -
# "raw" simply copies the uncorrected values across.
ORBIT_MODES = {
    "all":       "every track, per-(track, satellite) offset removed  [default]",
    "all-raw":   "every track, NO correction - use this to SEE the offsets",
    "track-raw": "every track, corrected per track only - leaves the S1A/S1C gap",
    "asc":       "ascending only (~17.9 h local, dry canopy)",
    "desc":      "descending only (~6.2 h local, dew likely)",
    "track":     "ONE relative orbit, both satellites (6 d in 2025, 12 d in 2024)",
    "track+sat": "ONE relative orbit AND one satellite - cleanest, always 12 d",
}


def _most_observed(s: pd.Series):
    """The value with the most rows; ties broken by the SMALLEST value.

    `Series.value_counts().idxmax()` looks like the obvious way to do this and
    is a trap: on a tie it returns whichever value happens to come first, which
    depends on ROW ORDER. One parcel has tracks 44 and 95 both on 21
    acquisitions, and shuffling the frame flips the answer between them - so a
    re-fetch or a different date range could silently change which track "track
    mode" means, and two runs would disagree without saying why.
    """
    counts = s.value_counts()
    top = counts.max()
    tied = sorted(counts[counts == top].index)
    return int(tied[0]) if str(tied[0]).lstrip("-").isdigit() else str(tied[0])


def select_orbit(df: pd.DataFrame, mode: str = "all", track: int | None = None,
                 platform: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Subset a series to one viewing-geometry configuration and re-normalise.

    Normalisation is recomputed ON THE SUBSET, which is the whole point: an
    offset estimated from all tracks is not the right offset for a single-track
    series.

    Args:
        df (pandas.DataFrame): a series from `parcel_series`.
        mode (str, optional): one of `ORBIT_MODES`. Defaults to "all".
        track (int, optional): which relative orbit, for the track modes.
            Defaults to whichever has the most observations, printed rather
            than left implicit.
        platform (str, optional): which satellite, for "track+sat". Same
            default rule.
        verbose (bool, optional): print what was selected. Defaults to True.

    Returns:
        pandas.DataFrame: a copy with `<col>_adj` columns and `df.attrs`
            describing what was done, so a chart can label itself honestly.

    Raises:
        ValueError: if `mode` is not in `ORBIT_MODES`.
    """
    if mode not in ORBIT_MODES:
        raise ValueError(f"mode must be one of {list(ORBIT_MODES)}, got {mode!r}")

    out = df.copy()
    note = mode

    if mode in ("asc", "desc"):
        want = "ascending" if mode == "asc" else "descending"
        out = out[out.orbit_state == want]
    elif mode in ("track", "track+sat"):
        if track is None:
            track = _most_observed(out.relative_orbit)
        out = out[out.relative_orbit == track]
        note = f"{mode} {track}"
        if mode == "track+sat":
            if platform is None and len(out):
                platform = _most_observed(out.platform)
            out = out[out.platform == platform]
            note = f"{mode} {track}/{platform}"

    out = out.reset_index(drop=True)

    if mode == "all-raw":
        for c in DB_COLS:                       # pass the raw values straight
            if c in out:                        # through under the _adj name
                out[f"{c}_adj"] = out[c].astype(float)
        if "rvi_median" in out:
            out["rvi_median_adj"] = out["rvi_median"].astype(float)
        normalised = False
    else:
        keys = ("relative_orbit",) if mode == "track-raw" else ORBIT_KEYS
        out = normalize_orbit(out, by=keys, verbose=verbose)
        normalised = True

    out.attrs.update(s1_mode=mode, s1_note=note, s1_normalised=normalised,
                     s1_track=track, s1_platform=platform)
    if verbose:
        print(f"  mode {note}: {len(out)} of {len(df)} acquisitions"
              + ("" if normalised else "  (uncorrected)"))
    return out


# ============================================================ per-parcel API
def parcel_series(
    geometry,
    start_date: str,
    end_date: str,
    resolution: int = 10,
    orbit_state: str | None = None,
    relative_orbit: int | None = None,
    orbit_normalize: bool = True,
    min_pixels: int = 5,
    buffer_m: float = 0.0,
    return_cube: bool = False,
    verbose: bool = True,
):
    """Sentinel-1 RTC feature series for one polygon - the whole workflow.

    Fetch, clip, mosaic slices, reduce per date in linear power, and remove the
    per-(track, satellite) offset, in one call. This is the level most analyses
    want: a tidy frame with one row per acquisition.

    USE THE `_adj` COLUMNS. `vh_vv_median_db` still carries the track offset,
    which is 0.64-1.16 dB against a mowing signal of -0.12 dB - several times
    larger than what is being looked for. `vh_vv_median_db_adj` is the one with
    the offset removed. See `normalize_orbit`.

    Args:
        geometry: the polygon. A path to a vector file, a GeoDataFrame or
            GeoSeries (reprojected from its own CRS), or a Shapely geometry
            (ASSUMED to be EPSG:4326 - it carries no CRS to check).
        start_date (str): ISO date, "YYYY-MM-DD".
        end_date (str): ISO date, "YYYY-MM-DD".
        resolution (int, optional): output pixel size in metres. Defaults to 10.
        orbit_state (str, optional): restrict to "ascending" or "descending".
            Note this does NOT remove the track offset, which is per-track and
            not per-direction - two ascending tracks over one alpine parcel sat
            0.96 dB apart. Prefer leaving it None and normalising.
        relative_orbit (int, optional): restrict to one track. Measured to be
            the WORST of the three options - see `normalize_orbit`.
        orbit_normalize (bool, optional): add the `_adj` columns. Defaults to
            True. Turning it off is for inspecting the offsets, not for
            analysis.
        min_pixels (int, optional): drop dates with fewer valid pixels than
            this. Defaults to 5.
        buffer_m (float, optional): shrink the polygon by this many metres
            before sampling. At 10 m a boundary pixel mixes in the neighbouring
            field, hedge or track. Defaults to 0 so nothing is discarded
            silently; the buffer is skipped with a note if it would empty the
            polygon.
        return_cube (bool, optional): also return the DataArray, e.g. to
            compute spatial statistics over the pixels. Defaults to False.
        verbose (bool, optional): print progress. Defaults to True.

    Returns:
        pandas.DataFrame: one row per acquisition, or `(df, cube)` when
            `return_cube` is True.
    """
    # Before the first geopandas read, not just before the STAC call: reading
    # and reprojecting the geometry is itself a CRS lookup and trips a broken
    # PROJ_LIB.
    fix_proj_env()

    geom = _as_geometry(geometry)

    if buffer_m:
        # Buffer in a projected CRS: the geometry is in EPSG:4326, where a
        # buffer distance would be in DEGREES and anisotropic with latitude.
        epsg = _utm_epsg(*geom.bounds)
        gs = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(epsg=epsg)
        shrunk = gs.buffer(-abs(buffer_m))
        if not shrunk.is_empty.all():
            geom = shrunk.to_crs("EPSG:4326").union_all()
        elif verbose:
            print(f"note: -{buffer_m} m buffer empties this polygon - not applied")

    cube = get_sentinel1_rtc_imagery(
        custom_geometry=geom, start_date=start_date, end_date=end_date,
        resolution=resolution, orbit_state=orbit_state,
        relative_orbit=relative_orbit, verbose=verbose)

    df = compute_features(cube, min_pixels=min_pixels)
    if orbit_normalize:
        df = normalize_orbit(df, verbose=verbose)

    return (df, cube) if return_cube else df
