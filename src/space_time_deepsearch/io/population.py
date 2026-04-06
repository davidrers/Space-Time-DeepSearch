import os
import tempfile
import requests
import rasterio
import rasterio.mask
import numpy as np
import geopandas as gpd
import shapely.geometry
from shapely.geometry import box
import warnings
import xarray as xr
import rioxarray


def get_world_boundaries():
    """
    Load or download world boundaries (Natural Earth 110m).
    """
    cache_dir = os.path.expanduser("~/.cache/osmsatlab")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.join(cache_dir, "ne_110m_admin_0_countries.geojson")
    
    if not os.path.exists(filename):
        # URL for Natural Earth 110m Admin 0 Countries (GeoJSON)
        # Using a reliable mirror (e.g. from martynafford or similar, or direct S3 converted)
        url = "https://raw.githubusercontent.com/martynafford/natural-earth-geojson/master/110m/cultural/ne_110m_admin_0_countries.json"
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(filename, 'wb') as f:
                f.write(r.content)
        except Exception as e:
            raise RuntimeError(f"Failed to download world boundaries: {e}")
            
    return gpd.read_file(filename)

def get_country_iso3(geometry):
    """
    Identifies the country ISO3 code for a given geometry.
    
    Args:
        geometry (shapely.geometry.BaseGeometry): The geometry to check.
        
    Returns:
        str: ISO3 code of the country with the largest intersection.
    """
    world = get_world_boundaries()
    
    # spatial join or intersection
    # Create a small gdf for the input geometry
    gdf_input = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
    
    # Ensure CRS match
    if world.crs != gdf_input.crs:
        world = world.to_crs(gdf_input.crs)
    
    # Filter world countries that intersect with our geometry
    intersection = gpd.overlay(world, gdf_input, how='intersection')
    
    if intersection.empty:
        # Fallback for small points slightly off-shore or coordinate issues?
        # Maybe use nearest? For now, raise.
        raise ValueError("The provided geometry does not intersect with any country.")
    
    # Calculate area of intersection
    intersection['overlap_area'] = intersection.geometry.area
    best_match = intersection.loc[intersection['overlap_area'].idxmax()]
    
    # Natural Earth GeoJSON usually uses 'ISO_A3' or 'ADM0_A3'
    # Check available columns
    if 'ISO_A3' in best_match:
        return best_match['ISO_A3']
    elif 'ADM0_A3' in best_match:
        return best_match['ADM0_A3']
    elif 'iso_a3' in best_match: # Legacy
        return best_match['iso_a3']
    else:
        # Fallback, return first 3 chars of name? Unsafe.
        # Dump available cols if debug needed
        return "UNK"

def get_cached_country_file(iso3, year=2020):
    """
    Get path to cached WorldPop country file, downloading if necessary.
    Uses 1km Aggregated UN-adjusted data.
    """
    iso3_upper = iso3.upper()
    iso3_lower = iso3.lower()
    
    cache_dir = os.path.expanduser("~/.cache/osmsatlab/worldpop")
    os.makedirs(cache_dir, exist_ok=True)
    
    filename = f"{iso3_lower}_ppp_{year}_1km_Aggregated.tif"
    local_path = os.path.join(cache_dir, filename)
    print(local_path)
    if not os.path.exists(local_path):
        # URL for WorldPop 1km
        # Example: https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/BEL/bel_ppp_2020_1km_Aggregated.tif
        url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/{year}/{iso3_upper}/{filename}"
        print(url)
        print('not found, downloading...')
        print(f"Downloading WorldPop data for {iso3_upper}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            # Clean up partial file if failed
            if os.path.exists(local_path):
                os.remove(local_path)
            raise RuntimeError(f"Failed to download WorldPop data from {url}: {e}")
            
    return local_path

def get_population_data(bbox=None, custom_geometry=None, year=2020):
    """
    Download or load cached WorldPop data for a given bounding box.
    Extracts population counts (1km resolution).
    
    Args:
        bbox (tuple, optional): Bounding box as (west, south, east, north).
        custom_geometry (str or shapely.geometry.BaseGeometry, optional): Path to a GeoJSON file or a Shapely geometry.
        year (int, optional): Year of the dataset. Defaults to 2020.
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing points with 'population' count.
    """
    if bbox is None and custom_geometry is None:
        raise ValueError("Either bbox or custom_geometry must be provided.")
    if bbox is not None and custom_geometry is not None:
        raise ValueError("Provide either bbox or custom_geometry, not both.")
        
    # 1. Define AOI Polygon
    if custom_geometry is not None:
        if isinstance(custom_geometry, str):
            gdf_boundary = gpd.read_file(custom_geometry)
            if gdf_boundary.crs != "EPSG:4326":
                gdf_boundary = gdf_boundary.to_crs("EPSG:4326")
            msk = gdf_boundary.union_all()
        else:
            msk = custom_geometry
    else:
        west, south, east, north = bbox
        msk = box(west, south, east, north)
    
    # 2. Identify Country
    iso3 = get_country_iso3(msk)
    
    # 3. Get Cached File
    local_filename = get_cached_country_file(iso3, year=year)
            
    # 4. Clip Raster
    with rasterio.open(local_filename) as src:
        # Mask requires a list of geometries
        try:
            out_image, out_transform = rasterio.mask.mask(src, [msk], crop=True)
        except ValueError:
            # Can happen if bbox doesn't overlap with raster (wrong country identified?)
            warnings.warn(f"Area does not overlap with {iso3} population data.")
            return gpd.GeoDataFrame({'population': [], 'geometry': []}, crs="EPSG:4326")

        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # The out_image is numpy array of shape (bands, rows, cols)
        data = out_image[0]
        
        # 5. Convert to Points
        # WorldPop nodata is often -99999 or similar float
        nodata = src.nodata
        if nodata is None:
            nodata = -99999
            
        rows, cols = np.where((data != nodata) & (data > 0))
        
        if len(rows) == 0:
            warnings.warn("No population found in the specified area.")
            return gpd.GeoDataFrame({'population': [], 'geometry': []}, crs="EPSG:4326")
        
        values = data[rows, cols]
        
        # Transform to coordinates
        xs, ys = rasterio.transform.xy(out_transform, rows, cols, offset='center')
        
        points = gpd.points_from_xy(xs, ys)
        
        gdf_pop = gpd.GeoDataFrame(
            {'population': values}, 
            geometry=points, 
            crs=src.crs
        )
        
        if gdf_pop.crs != "EPSG:4326":
            gdf_pop = gdf_pop.to_crs("EPSG:4326")
            
        return gdf_pop

def get_population_raster(bbox: tuple[float, float, float, float] | None = None, 
                         custom_geometry: str | shapely.geometry.base.BaseGeometry | None = None, 
                         year: int = 2020) -> xr.DataArray:
    """
    Download or load cached WorldPop data for a given bounding box as an xarray DataArray.
    Useful for raster-based analysis like Heat Exposure Index.
    
    Args:
        bbox (tuple, optional): Bounding box as (west, south, east, north).
        custom_geometry (str or geometry, optional): Path to GeoJSON or Shapely geometry.
        year (int, optional): Year of the dataset. Defaults to 2020.
        
    Returns:
        xarray.DataArray: Population count raster. Dims: (y, x) or (band, y, x).
    """
    if bbox is None and custom_geometry is None:
        raise ValueError("Either bbox or custom_geometry must be provided.")
    if bbox is not None and custom_geometry is not None:
        raise ValueError("Provide either bbox or custom_geometry, not both.")
        
    # 1. Define AOI Polygon (Reusing logic)
    if custom_geometry is not None:
        if isinstance(custom_geometry, str):
            gdf_boundary = gpd.read_file(custom_geometry)
            if gdf_boundary.crs != "EPSG:4326":
                gdf_boundary = gdf_boundary.to_crs("EPSG:4326")
            msk = gdf_boundary.union_all()
        else:
            msk = custom_geometry
    else:
        west, south, east, north = bbox
        msk = box(west, south, east, north)
    
    # 2. Identify Country
    iso3 = get_country_iso3(msk)
    
    # 3. Get Cached File
    local_filename = get_cached_country_file(iso3, year=year)
    
    # 4. Open with rioxarray
    da = rioxarray.open_rasterio(local_filename, masked=True)
    
    # 5. Clip
    try:
        # rioxarray expects a list of geometries.
        # Ensure geometry is in correct CRS (WorldPop is usually EPSG:4326)
        if da.rio.crs != "EPSG:4326":
             # This assumes AOI is 4326. If raster is not, we project AOI to raster CRS.
             # But usually WorldPop is 4326.
             pass
             
        da_clipped = da.rio.clip([msk], "EPSG:4326", drop=True)
        
        # Set nodata to NaN if not already
        da_clipped = da_clipped.where(da_clipped != da_clipped.rio.nodata)
        
        # Squeeze band if single band
        if "band" in da_clipped.dims and da_clipped.sizes["band"] == 1:
            da_clipped = da_clipped.squeeze("band")
            
        return da_clipped
        
    except Exception as e:
        warnings.warn(f"Failed to clip raster, returning full country slice (Warning: {e})")
        return da
