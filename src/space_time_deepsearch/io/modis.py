"""
MODIS Land Surface Temperature retrieval.

This module provides functionality to fetch MODIS LST (MOD11A1) data
from Microsoft Planetary Computer and return it as an xarray DataArray.
"""

import geopandas as gpd
import shapely.geometry
from shapely.geometry import box
import pystac_client
import stackstac
import planetary_computer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import rioxarray
import dask.diagnostics
import xarray as xr

def _get_utm_epsg(west, south, east, north):
    """
    Get the UTM EPSG code for a bounding box using pyproj.
    (Duplicated from sentinel2.py for self-containment)
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

def get_modis_temperature(
    bbox: tuple[float, float, float, float] | None = None,
    custom_geometry: str | shapely.geometry.base.BaseGeometry | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    collection: str = "modis-11A1-061",
    layer: str = "LST_Day_1km",
    convert_to_celsius: bool = True,
    composite_period: str | None = "1W",
    resolution: int = 1000, # Approx 1km for MODIS
    chunksize: int = 2048,
):
    """
    Fetch MODIS Land Surface Temperature data.

    Args:
        bbox (tuple, optional): (west, south, east, north) in EPSG:4326.
        custom_geometry (str or geometry, optional): Path to GeoJSON or Shapely geometry.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        collection (str, optional): STAC collection ID. Defaults to "modis-11A1-061".
        layer (str, optional): Asset name for temperature. Defaults to "LST_Day_1km".
        convert_to_celsius (bool, optional): If True, converts Kelvin to Celsius. Defaults to True.
        composite_period (str | None, optional): Temporal period for compositing (e.g., "1W", "1M").
            Defaults to "1W". If None, returns daily data.
            Compositing takes the mean over the period to fill cloud gaps.
        resolution (int, optional): Output resolution in meters. Defaults to 1000.
        chunksize (int, optional): Dask chunk size. Defaults to 2048.

    Returns:
        xarray.DataArray: DataArray with dims (time, band, y, x).
    """
    # Validate inputs
    if bbox is None and custom_geometry is None:
        raise ValueError("Either bbox or custom_geometry must be provided.")
    if bbox is not None and custom_geometry is not None:
        raise ValueError("Provide either bbox or custom_geometry, not both.")
    if start_date is None or end_date is None:
        raise ValueError("Both start_date and end_date must be provided.")

    # Define AOI geometry
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

    # Calculate UTM EPSG
    target_epsg = _get_utm_epsg(search_bbox[0], search_bbox[1], search_bbox[2], search_bbox[3])
    print(f"Using EPSG:{target_epsg}")

    # STAC Search
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    date_range = f"{start_date}/{end_date}"
    search = catalog.search(
        collections=[collection],
        bbox=search_bbox,
        datetime=date_range,
    )
    items = search.item_collection()
    
    if len(items) == 0:
        raise ValueError("No MODIS items found for the specified parameters.")
    
    print(f"Found {len(items)} MODIS scenes")
    
    # Pre-process items: MPC MODIS items have datetime=None but start_datetime set.
    # stackstac needs a valid datetime.
    import dateutil.parser
    for item in items:
        if item.datetime is None:
            start_dt = item.properties.get("start_datetime")
            if start_dt:
                item.datetime = dateutil.parser.parse(start_dt)
    
    # Stack
    # Note: MODIS native projection is Sinusoidal. stackstac will reproject to target_epsg.
    cube = stackstac.stack(
        items,
        assets=[layer],
        bounds_latlon=search_bbox,
        resolution=resolution,
        epsg=target_epsg,
        chunksize=chunksize,
    )
    
    # Debug: Check time coordinate
    #print("Time coordinates:", cube.time.values)
    
    # Ensure time is sorted
    cube = cube.sortby("time")

    # Mosaic by time (max) to handle swaths if necessary, though MOD11A1 is gridded.
    # Daily MODIS is tiled, so duplicates for same day might exist if AOI crosses tiles.
    # Group by day and mosaic.
    # We use a string format to group by day safely even if times differ slightly
    cube = cube.groupby(cube.time.dt.date).max(dim="time")
    # Rename 'date' back to 'time' to maintain consistency
    cube = cube.rename({"date": "time"})
    
    # Ensure time is a DatetimeIndex for resampling
    import pandas as pd
    cube["time"] = pd.to_datetime(cube.time.values)


    # Clip logic (same as sentinel2)
    if custom_geometry is not None and not isinstance(aoi_geometry, shapely.geometry.box.__class__):
        gdf_clip = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
        gdf_clip_proj = gdf_clip.to_crs(epsg=target_epsg)
        cube = cube.rio.clip(
            gdf_clip_proj.geometry,
            crs=gdf_clip_proj.crs,
            drop=False,
            all_touched=True,
        )

    # Conversion
    if convert_to_celsius:
        # stackstac often applies the scale factor (0.02) automatically if metadata is present.
        # If the values are ~300, they are already Kelvin. 
        # If they were ~15000, they would be raw DNs.
        # Based on testing, they appear to be already in Kelvin (~280-300).
        cube = cube - 273.15
        cube.attrs["units"] = "Celsius"
    else:
        cube.attrs["units"] = "Kelvin"
        
    # Temporal Compositing (Lazy)
    # Default is "1W" to fill gaps
    if composite_period:
        # We assume the user wants the mean temperature over the period.
        # Resample requires a valid datetime index, which we ensured above.
        cube = cube.resample(time=composite_period).mean(dim="time", skipna=True)

    # Attributes
    cube.attrs["collection"] = collection
    cube.attrs["layer"] = layer
    
    # Compute
    with dask.diagnostics.ProgressBar():
        data = cube.compute()
        
    return data
