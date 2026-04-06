"""
Landsat imagery retrieval using STAC and stackstac.

This module provides functionality to fetch harmonized Landsat imagery
(Landsat 4/5 TM, Landsat 7 ETM+, Landsat 8/9 OLI) for a given area
of interest and time range, returning a 4D xarray DataArray with
dimensions (time, band, y, x).

Uses the Planetary Computer ``landsat-c2-l2`` collection which provides
pre-harmonized band names across all missions.
"""

from datetime import datetime, timezone

import geopandas as gpd
import shapely.geometry
from shapely.geometry import box
import pystac_client
import stackstac
import planetary_computer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import warnings
import numpy as np
import rioxarray
import dask.diagnostics
import xarray as xr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_BANDS = ["blue", "green", "red", "nir08"]

# Landsat Collection 2 Level-2 surface reflectance scale factors
_SCALE_FACTOR = 0.0000275
_OFFSET = -0.2

# QA_PIXEL bit positions (Landsat Collection 2)
_QA_DILATED_CLOUD = 1
_QA_CLOUD = 3
_QA_CLOUD_SHADOW = 4
_QA_SNOW = 5

# Landsat 7 SLC-off date (Scan Line Corrector failure)
_SLC_OFF_DATE = datetime(2003, 5, 31, tzinfo=timezone.utc)

# TM/ETM+ platforms (need SRF correction to match OLI)
_LEGACY_PLATFORMS = {"landsat-4", "landsat-5", "landsat-7"}

# Roy et al. (2016) spectral bandpass adjustment coefficients
# Transforms TM/ETM+ reflectance to OLI-equivalent reflectance
# Format: (slope, intercept)
_SRF_COEFFICIENTS = {
    "blue":   (0.9785, -0.0095),
    "green":  (0.9542, -0.0016),
    "red":    (0.9825,  0.0004),
    "nir08":  (0.8339,  0.0029),
    "swir16": (0.8639,  0.0039),
    "swir22": (0.9005,  0.0026),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_utm_epsg(west, south, east, north):
    """
    Get the UTM EPSG code for a bounding box using pyproj.

    Args:
        west, south, east, north: Bounding box coordinates in EPSG:4326.

    Returns:
        int: EPSG code for the appropriate UTM zone.
    """
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=west,
            south_lat_degree=south,
            east_lon_degree=east,
            north_lat_degree=north,
        ),
    )
    return int(utm_crs_list[0].code)


def _decode_qa_pixel(qa, mask_cloud=True, mask_shadow=True, mask_snow=False):
    """
    Decode Landsat QA_PIXEL band into a boolean mask.

    Args:
        qa: xarray DataArray of QA_PIXEL values.
        mask_cloud: Mask cloud and dilated cloud pixels.
        mask_shadow: Mask cloud shadow pixels.
        mask_snow: Mask snow/ice pixels.

    Returns:
        xarray DataArray of bool (True = bad pixel to mask out).
    """
    qa_int = qa.astype(int)
    bad = xr.zeros_like(qa, dtype=bool)
    if mask_cloud:
        bad = bad | ((qa_int >> _QA_CLOUD) & 1).astype(bool)
        bad = bad | ((qa_int >> _QA_DILATED_CLOUD) & 1).astype(bool)
    if mask_shadow:
        bad = bad | ((qa_int >> _QA_CLOUD_SHADOW) & 1).astype(bool)
    if mask_snow:
        bad = bad | ((qa_int >> _QA_SNOW) & 1).astype(bool)
    return bad


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_landsat_imagery(
    bbox: tuple[float, float, float, float] | None = None,
    custom_geometry: str | shapely.geometry.base.BaseGeometry | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    bands: list[str] | None = None,
    resolution: int = 30,
    cloud_cover_max: int = 30,
    min_coverage: int = 0,
    mask_clouds: bool = False,
    mask_snow: bool = False,
    composite_period: str | None = None,
    collection: str = "landsat-c2-l2",
    chunksize: int = 2048,
    add_ndvi: bool = False,
    add_ndbi: bool = False,
    missions: list[str] | None = None,
    exclude_slc_off: bool = False,
    apply_srf_correction: bool = False,
    apply_scale_factors: bool = True,
):
    """
    Fetch harmonized Landsat imagery for a given area of interest and time range.

    Uses Microsoft Planetary Computer STAC API to search and retrieve
    Landsat Collection 2 Level-2 (surface reflectance) imagery across
    all Landsat missions (4/5 TM, 7 ETM+, 8/9 OLI), returning a
    4D xarray DataArray.

    Band names are harmonized by the Planetary Computer collection:
    ``blue``, ``green``, ``red``, ``nir08``, ``swir16``, ``swir22``.

    Args:
        bbox (tuple, optional): Bounding box as (west, south, east, north)
            in EPSG:4326.
        custom_geometry (str or shapely.geometry.BaseGeometry, optional):
            Path to a GeoJSON file or a Shapely geometry.
        start_date (str): Start date in ISO format (YYYY-MM-DD).
        end_date (str): End date in ISO format (YYYY-MM-DD).
        bands (list, optional): List of harmonized band names to retrieve.
            Defaults to ["blue", "green", "red", "nir08"].
            Available: blue, green, red, nir08, swir16, swir22, coastal
            (coastal only on Landsat 8/9).
        resolution (int, optional): Output resolution in meters. Defaults to 30.
        cloud_cover_max (int, optional): Maximum cloud cover percentage.
            Defaults to 30.
        min_coverage (int, optional): Minimum spatial coverage percentage.
            Scenes covering less than this percentage of the AOI will be
            filtered out. Defaults to 0.
        mask_clouds (bool, optional): If True, sets cloudy pixels to NaN.
            Defaults to False.
        mask_snow (bool, optional): If True, also mask snow/ice pixels.
            Only effective when mask_clouds is True. Defaults to False.
        composite_period (str | None, optional): Temporal period for
            compositing (e.g., "1M", "1Y"). Defaults to None.
        collection (str, optional): STAC collection ID.
            Defaults to "landsat-c2-l2".
        chunksize (int, optional): Dask chunk size. Defaults to 2048.
        add_ndvi (bool, optional): If True, calculate NDVI and add as band.
            Requires red and nir08. Defaults to False.
        add_ndbi (bool, optional): If True, calculate NDBI and add as band.
            Requires swir16 and nir08. Defaults to False.
        missions (list, optional): Filter to specific platforms, e.g.
            ["landsat-8", "landsat-9"]. Defaults to None (all missions).
        exclude_slc_off (bool, optional): If True, drop Landsat 7 scenes
            after 2003-05-31 (SLC failure). Defaults to False.
        apply_srf_correction (bool, optional): If True, apply Roy et al.
            (2016) spectral bandpass adjustment to TM/ETM+ bands to
            harmonize with OLI. Defaults to False.
        apply_scale_factors (bool, optional): If True, convert raw DN to
            surface reflectance (DN * 0.0000275 - 0.2). Defaults to True.

    Returns:
        xarray.DataArray: 4D DataArray with dimensions (time, band, y, x).

    Raises:
        ValueError: If neither bbox nor custom_geometry is provided,
            or if both are provided, or if date range is not specified.
    """
    # --- Input validation ---
    if bbox is None and custom_geometry is None:
        raise ValueError("Either bbox or custom_geometry must be provided.")
    if bbox is not None and custom_geometry is not None:
        raise ValueError("Provide either bbox or custom_geometry, not both.")
    if start_date is None or end_date is None:
        raise ValueError("Both start_date and end_date must be provided.")

    if bands is None:
        bands = list(_DEFAULT_BANDS)

    # --- AOI geometry ---
    if custom_geometry is not None:
        if isinstance(custom_geometry, str):
            gdf_boundary = gpd.read_file(custom_geometry)
            if gdf_boundary.crs != "EPSG:4326":
                gdf_boundary = gdf_boundary.to_crs("EPSG:4326")
            aoi_geometry = gdf_boundary.union_all()
        else:
            aoi_geometry = custom_geometry
        aoi_bounds = aoi_geometry.bounds
        search_bbox = list(aoi_bounds)
    else:
        west, south, east, north = bbox
        aoi_geometry = box(west, south, east, north)
        search_bbox = list(bbox)

    # --- STAC search ---
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    datetime_range = f"{start_date}/{end_date}"
    search_query = {"eo:cloud_cover": {"lt": cloud_cover_max}}

    search = catalog.search(
        collections=[collection],
        bbox=search_bbox,
        datetime=datetime_range,
        query=search_query,
    )

    items = search.item_collection()

    if len(items) == 0:
        raise ValueError(
            "No Landsat scenes found for the specified parameters. "
            "Try increasing cloud_cover_max or expanding the date range."
        )

    print(f"Found {len(items)} candidate Landsat scenes")

    # Re-sign items to ensure SAS tokens are fresh before data access.
    # The modifier on the catalog signs during search, but tokens can expire
    # if there is a delay before the lazy data is actually read.
    items = [planetary_computer.sign(item) for item in items]

    # --- Filter STAC items by mission / SLC-off ---
    if missions is not None:
        items = [i for i in items if i.properties.get("platform") in missions]
        if len(items) == 0:
            raise ValueError(
                f"No scenes found for missions {missions}. "
                f"Check platform names (e.g. 'landsat-8', 'landsat-9')."
            )
        print(f"Filtered to {len(items)} scenes for missions: {missions}")

    if exclude_slc_off:
        before = len(items)
        items = [
            i for i in items
            if not (
                i.properties.get("platform") == "landsat-7"
                and i.datetime >= _SLC_OFF_DATE
            )
        ]
        dropped = before - len(items)
        if dropped > 0:
            print(f"Excluded {dropped} Landsat 7 SLC-off scenes")

    if len(items) == 0:
        raise ValueError("No scenes remaining after mission/SLC-off filtering.")

    # --- UTM projection ---
    target_epsg = _get_utm_epsg(
        search_bbox[0], search_bbox[1], search_bbox[2], search_bbox[3]
    )
    print(f"Using EPSG:{target_epsg} for projection")

    # --- Determine assets to load ---
    assets_to_load = list(bands)

    if "qa_pixel" not in assets_to_load:
        assets_to_load.append("qa_pixel")

    if add_ndvi:
        if "red" not in assets_to_load:
            assets_to_load.append("red")
        if "nir08" not in assets_to_load:
            assets_to_load.append("nir08")

    if add_ndbi:
        if "swir16" not in assets_to_load:
            assets_to_load.append("swir16")
        if "nir08" not in assets_to_load:
            assets_to_load.append("nir08")

    # --- Build data cube ---
    # rescale=False: Landsat C2 L2 items carry raster:bands scale/offset
    # metadata (0.0000275 / -0.2).  stackstac applies this automatically
    # when rescale=True (the default), but the code applies the same
    # conversion later (lines below) when apply_scale_factors=True.
    # Disabling auto-rescale avoids double-scaling that collapses all
    # reflectance values to zero.
    cube = stackstac.stack(
        items,
        assets=assets_to_load,
        bounds_latlon=search_bbox,
        resolution=resolution,
        epsg=target_epsg,
        dtype="float64",
        fill_value=np.nan,
        rescale=False,
        chunksize=chunksize,
        gdal_env=stackstac.DEFAULT_GDAL_ENV.updated(always=dict(
            GDAL_HTTP_MAX_RETRY=3,
            GDAL_HTTP_RETRY_DELAY=5,
        )),
    )

    # Note: Mosaicing of overlapping tiles is deferred until after scene
    # filtering to avoid processing scenes that will be discarded.

    # --- Clip to custom geometry ---
    if custom_geometry is not None and not isinstance(
        aoi_geometry, shapely.geometry.box.__class__
    ):
        gdf_clip = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
        gdf_clip_proj = gdf_clip.to_crs(epsg=target_epsg)
        cube = cube.rio.clip(
            gdf_clip_proj.geometry,
            crs=gdf_clip_proj.crs,
            drop=False,
            all_touched=True,
        )

    # --- Local Cloud and Coverage Filtering (using QA_PIXEL) ---

    # Mosaic only qa_pixel band for stats (avoids expensive all-band mosaic)
    qa = cube.sel(band="qa_pixel", drop=True)
    if len(qa.time) != len(np.unique(qa.time.values)):
        qa = qa.groupby("time").max(dim="time")

    is_cloudy = _decode_qa_pixel(qa, mask_cloud=True, mask_shadow=True, mask_snow=False)
    # fill_value=np.nan means missing pixels are NaN; QA_PIXEL nodata is 1
    is_geo_valid = qa.notnull() & (qa != 0)
    is_data_valid = is_geo_valid & (qa > 1)

    print(
        f"Filtering scenes: Cloud < {cloud_cover_max}%, "
        f"Coverage >= {min_coverage}%..."
    )

    # Downsample for faster statistics
    if qa.sizes["y"] * qa.sizes["x"] > 10000:
        qa_stats = qa[:, ::10, ::10]
        is_cloudy_d = _decode_qa_pixel(
            qa_stats, mask_cloud=True, mask_shadow=True, mask_snow=False
        )
        is_geo_valid_d = qa_stats.notnull() & (qa_stats != 0)
        is_data_valid_d = is_geo_valid_d & (qa_stats > 1)

        cloud_counts = is_cloudy_d.sum(dim=["y", "x"])
        geo_counts = is_geo_valid_d.sum(dim=["y", "x"])
        data_counts = is_data_valid_d.sum(dim=["y", "x"])
    else:
        cloud_counts = is_cloudy.sum(dim=["y", "x"])
        geo_counts = is_geo_valid.sum(dim=["y", "x"])
        data_counts = is_data_valid.sum(dim=["y", "x"])

    cloud_pct = (cloud_counts / data_counts) * 100
    coverage_pct = (data_counts / geo_counts) * 100

    stats_ds = xr.Dataset({"cloud_pct": cloud_pct, "coverage_pct": coverage_pct})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with dask.diagnostics.ProgressBar():
            print("Computing local cloud stats...")
            stats_computed = stats_ds.compute()

    # Filter: select valid timestamps from original (unmosaiced) cube
    valid_mask = (stats_computed.cloud_pct < cloud_cover_max) & (
        stats_computed.coverage_pct >= min_coverage
    )
    valid_time_values = stats_computed.time[valid_mask].values

    cube_filtered = cube.sel(time=cube.time.isin(valid_time_values))

    # Mosaic overlapping tiles (deferred so only retained scenes are processed)
    if len(cube_filtered.time) != len(np.unique(cube_filtered.time.values)):
        cube_filtered = cube_filtered.groupby("time").max(dim="time")

    print(
        f"Retained {len(cube_filtered.time)} scenes after local filtering "
        f"(max {cloud_cover_max}%)"
    )

    # --- Apply scale factors ---
    if apply_scale_factors:
        reflectance_bands = [b for b in cube_filtered.band.values if b != "qa_pixel"]
        refl = cube_filtered.sel(band=reflectance_bands)
        refl = refl * _SCALE_FACTOR + _OFFSET
        refl = refl.clip(0, 1)

        # Reassemble with qa_pixel unchanged
        if "qa_pixel" in cube_filtered.band.values:
            qa_da = cube_filtered.sel(band=["qa_pixel"])
            cube_filtered = xr.concat([refl, qa_da], dim="band")
        else:
            cube_filtered = refl

    # --- Cloud masking ---
    if mask_clouds:
        qa_filtered = cube_filtered.sel(band="qa_pixel", drop=True)
        cloud_mask = _decode_qa_pixel(
            qa_filtered, mask_cloud=True, mask_shadow=True, mask_snow=mask_snow
        )
        cube_filtered = cube_filtered.where(~cloud_mask)

    # --- SRF correction (Roy et al. 2016) ---
    if apply_srf_correction and "platform" in cube_filtered.coords:
        is_legacy = cube_filtered.platform.isin(list(_LEGACY_PLATFORMS))
        for band_name in list(cube_filtered.band.values):
            if band_name in _SRF_COEFFICIENTS:
                slope, intercept = _SRF_COEFFICIENTS[band_name]
                band_data = cube_filtered.sel(band=band_name)
                corrected = xr.where(
                    is_legacy, band_data * slope + intercept, band_data
                )
                cube_filtered.loc[dict(band=band_name)] = corrected

    # --- Index Calculation Helper ---
    def _add_index(cube, band_name, fn):
        try:
            idx = fn(cube)
            idx = idx.expand_dims(band=[band_name])

            dims = set(cube.dims)
            valid_coords = dims.union({"spatial_ref", "epsg"})
            coords_to_drop_cube = [c for c in cube.coords if c not in valid_coords]
            coords_to_drop_idx = [c for c in idx.coords if c not in valid_coords]

            cube_stripped = cube.drop_vars(coords_to_drop_cube, errors="ignore")
            idx_stripped = idx.drop_vars(coords_to_drop_idx, errors="ignore")

            return xr.concat([cube_stripped, idx_stripped], dim="band")
        except Exception as e:
            print(f"Warning: Could not calculate/append {band_name}: {e}")
            return cube

    # --- NDVI ---
    if add_ndvi:
        def calc_ndvi(c):
            red = c.sel(band="red", drop=True)
            nir = c.sel(band="nir08", drop=True)
            return (nir - red) / (nir + red)

        cube_filtered = _add_index(cube_filtered, "NDVI", calc_ndvi)

    # --- NDBI ---
    if add_ndbi:
        def calc_ndbi(c):
            swir = c.sel(band="swir16", drop=True)
            nir = c.sel(band="nir08", drop=True)
            return (swir - nir) / (swir + nir)

        cube_filtered = _add_index(cube_filtered, "NDBI", calc_ndbi)

    # --- Drop qa_pixel and select requested bands ---
    if "qa_pixel" not in bands:
        bands_to_keep = list(bands)
        if add_ndvi and "NDVI" in cube_filtered.band.values:
            if "NDVI" not in bands_to_keep:
                bands_to_keep.append("NDVI")
        if add_ndbi and "NDBI" in cube_filtered.band.values:
            if "NDBI" not in bands_to_keep:
                bands_to_keep.append("NDBI")
        cube_filtered = cube_filtered.sel(band=bands_to_keep)

    # --- Temporal Compositing ---
    if composite_period:
        print(f"Compositing data over {composite_period} using median...")
        cube_filtered = cube_filtered.resample(time=composite_period).median(
            dim="time", skipna=True
        )

    # --- Attributes ---
    cube_filtered.attrs["source"] = "Microsoft Planetary Computer"
    cube_filtered.attrs["collection"] = collection
    cube_filtered.attrs["start_date"] = start_date
    cube_filtered.attrs["end_date"] = end_date
    cube_filtered.attrs["cloud_cover_max"] = cloud_cover_max
    cube_filtered.attrs["min_coverage"] = min_coverage
    cube_filtered.attrs["mask_clouds"] = str(mask_clouds)
    cube_filtered.attrs["ndvi_added"] = str(add_ndvi)
    cube_filtered.attrs["ndbi_added"] = str(add_ndbi)
    cube_filtered.attrs["scale_factors_applied"] = str(apply_scale_factors)
    cube_filtered.attrs["srf_correction_applied"] = str(apply_srf_correction)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with dask.diagnostics.ProgressBar():
            data = cube_filtered.compute()

    return data
