import osmnx as ox
import geopandas as gpd

def download_osm_data(bbox=None, custom_geometry=None, tags=None):
    """
    Download OSM data for a given bounding box or custom geometry.

    Args:
        bbox (tuple, optional): Bounding box as (west, south, east, north).
        custom_geometry (str or shapely.geometry.BaseGeometry, optional): Path to a GeoJSON file or a Shapely geometry.
        tags (dict, optional): Dictionary of tags to query. Defaults to None, which queries 'building'.

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing the downloaded OSM features.
    
    Raises:
        ValueError: If neither bbox nor custom_geometry is provided, or if both are provided.
    """
    if bbox is None and custom_geometry is None:
        raise ValueError("Either bbox or custom_geometry must be provided.")
    if bbox is not None and custom_geometry is not None:
        raise ValueError("Provide either bbox or custom_geometry, not both.")

    if tags is None:
        tags = {'building': True}

    if custom_geometry is not None:
        # Handle custom geometry
        if isinstance(custom_geometry, str):
            # Assume it's a file path (e.g. GeoJSON)
            gdf_boundary = gpd.read_file(custom_geometry)
            # Combine all geometries into one (e.g. valid polygon or multipolygon)
            polygon = gdf_boundary.union_all()
        else:
            # Assume it is already a shapely geometry
            polygon = custom_geometry
            
        gdf = ox.features.features_from_polygon(polygon, tags=tags)
    else:
        # Handle bbox
        west, south, east, north = bbox
        # osmnx.features.features_from_bbox expects bbox=(west, south, east, north) in v2.0+
        gdf = ox.features.features_from_bbox(bbox=bbox, tags=tags)
    
    return gdf

def download_street_network(bbox=None, custom_geometry=None, network_type='drive'):
    """
    Download OSM street network for a given bounding box or custom geometry.

    Args:
        bbox (tuple, optional): Bounding box as (west, south, east, north).
        custom_geometry (str or shapely.geometry.BaseGeometry, optional): Path to a GeoJSON file or a Shapely geometry.
        network_type (str, optional): Type of street network to download (e.g., 'drive', 'walk'). Defaults to 'drive'.

    Returns:
        networkx.MultiDiGraph: The downloaded street network graph.
    
    Raises:
        ValueError: If neither bbox nor custom_geometry is provided, or if both are provided.
    """
    if bbox is None and custom_geometry is None:
        raise ValueError("Either bbox or custom_geometry must be provided.")
    if bbox is not None and custom_geometry is not None:
        raise ValueError("Provide either bbox or custom_geometry, not both.")

    G_raw = None

    if custom_geometry is not None:
        # Handle custom geometry
        polygon = None
        if isinstance(custom_geometry, str):
            # Assume it's a file path (e.g. GeoJSON)
            gdf_boundary = gpd.read_file(custom_geometry)
            # Combine all geometries into one
            polygon = gdf_boundary.union_all()
        else:
            # Assume it is already a shapely geometry
            polygon = custom_geometry
            
        if hasattr(polygon, 'bounds'):
            G_raw = ox.graph_from_polygon(polygon, network_type=network_type)
    else:
        # Handle bbox
        west, south, east, north = bbox
        G_raw = ox.graph_from_bbox(bbox=(west, south, east, north), network_type=network_type)
        
    if G_raw is not None:
        # Calculate travel time
        # 1. Add edge speeds (fills in missing maxspeed data based on road type)
        # We need a fallback because not all edges have maxspeed tags.
        # Use 30 km/h for driving (urban), 4.5 km/h for walking.
        fallback_speed = 4.5 if network_type == 'walk' else 35.0
        
        G_raw = ox.add_edge_speeds(G_raw, fallback=fallback_speed)
        # 2. Add edge travel times (length / speed)
        G_raw = ox.add_edge_travel_times(G_raw)
        
    return G_raw
