
"""
Visualization module for satellite data.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr
import numpy as np
import os
from typing import List, Optional
try:
    import contextily as ctx
except ImportError:
    ctx = None

def create_timelapse(
    data: xr.DataArray,
    output_path: str = "timelapse.gif",
    fps: int = 5,
    bands: Optional[List[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = "viridis",
    add_text: bool = True,
    add_basemap: bool = False,
    basemap_source: Optional[str] = None,
    alpha: float = 1.0,
    figsize: tuple = (10, 10),
    dpi: int = 100
):
    """
    Create a timelapse animation (GIF) from an xarray DataArray.

    Args:
        data (xr.DataArray): Input data with a 'time' dimension. 
                             Can be (time, y, x) or (time, band, y, x).
        output_path (str, optional): Output path for the GIF. Defaults to "timelapse.gif".
        fps (int, optional): Frames per second. Defaults to 5.
        bands (List[str], optional): For RGB visualization, list of [Red, Green, Blue] band names. 
                                     Only used if data has a 'band' dimension. 
                                     Defaults to None (uses cmap on first band/layer).
        vmin (float, optional): Min value for colormap scaling. Defaults to data min.
        vmax (float, optional): Max value for colormap scaling. Defaults to data max.
        vmax (float, optional): Max value for colormap scaling. Defaults to data max.
        cmap (str, optional): Matplotlib colormap name. Defaults to "viridis". Ignored for RGB.
        add_text (bool, optional): If True, adds the date to the image. Defaults to True.
        add_basemap (bool, optional): If True, adds a context basemap (requires contextily). Defaults to False.
        basemap_source (str, optional): Contextily provider source. Defaults to CartoDB.Positron.
        alpha (float, optional): Opacity of the data layer (0.0 to 1.0). Defaults to 1.0.
        figsize (tuple, optional): Figure size. Defaults to (10, 10).
        dpi (int, optional): DPI of the output. Defaults to 100.
    """
    if add_basemap and ctx is None:
        print("Warning: 'contextily' is not installed. Basemap will not be added.")
        add_basemap = False
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare plotting data
    is_rgb = False
    
    # Check dimensions
    dims = data.dims
    if "time" not in dims:
        raise ValueError("DataArray must have a 'time' dimension.")

    if "band" in dims:
        if bands and len(bands) == 3:
            # RGB Mode
            is_rgb = True
            try:
                # Select RGB bands
                plot_data = data.sel(band=bands)
                # Normalize for RGB display (0-1)
                # Simple normalization based on vmin/vmax or percentiles if not provided
                if vmin is None: 
                    vmin = plot_data.as_numpy().quantile(0.02)
                if vmax is None:
                    vmax = plot_data.as_numpy().quantile(0.98)
                
                # Clip and scale
                plot_data = (plot_data - vmin) / (vmax - vmin)
                plot_data = plot_data.clip(0, 1)
                
                # Transpose to (time, y, x, band) for matplotlib
                plot_data = plot_data.transpose("time", "y", "x", "band")
            except KeyError:
                print(f"Warning: Specified bands {bands} not found. Falling back to single band.")
                plot_data = data.isel(band=0)
        else:
            # Single Band Mode - take first band if not specified
            if bands:
               plot_data = data.sel(band=bands[0])
            else:
               plot_data = data.isel(band=0)
    else:
        # 3D array (time, y, x)
        plot_data = data

    # ── Drop all-NaN frames (e.g. fully cloudy months) ──────────────
    if is_rgb:
        # For RGB: check across (y, x, band)
        valid_mask = np.array([
            np.any(np.isfinite(plot_data.isel(time=t).values))
            for t in range(plot_data.sizes["time"])
        ])
    else:
        # For single-band: check across (y, x)
        valid_mask = np.array([
            np.any(np.isfinite(plot_data.isel(time=t).values))
            for t in range(plot_data.sizes["time"])
        ])
    
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        print(f"Skipping {n_dropped} all-NaN frames out of {len(valid_mask)} total.")
        valid_indices = np.where(valid_mask)[0]
        plot_data = plot_data.isel(time=valid_indices)
        data = data.isel(time=valid_indices)
    
    if plot_data.sizes["time"] == 0:
        print("Warning: No valid frames remaining after filtering. No animation created.")
        return

    # Setup Figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Determine Extent for plotting (Left, Right, Bottom, Top)
    # Allows overlaying with basemap
    extent = None
    if "x" in data.coords and "y" in data.coords:
        try:
            # Assuming x are longitudes/eastings and y are latitudes/northings
            # xarray plotting uses center-coordinates, but imshow needs edges? 
            # Actually imshow with extent treats them as edges usually.
            # Using min/max is roughly correct for large images.
            xs = data.x.values
            ys = data.y.values
            
            # Matplotlib imshow extent: [left, right, bottom, top]
            # Check orientation of Y
            if ys[1] < ys[0]: # Y is descending (North Up standard for rasters)
                 # top is max(ys), bottom is min(ys)
                 # But imshow with origin='upper' (default) expects top row to be at top coordinate.
                 # extent = [min_x, max_x, min_y, max_y]
                 extent = [xs.min(), xs.max(), ys.min(), ys.max()]
            else:
                 extent = [xs.min(), xs.max(), ys.min(), ys.max()]
            
            # CRITICAL: Set axis limits to match data extent immediately
            # This allows contextily to download the correct tiles for the basemap
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
                 
        except Exception as e:
            print(f"Warning: Could not determine extent from coordinates: {e}")

    
    # Initial Frame
    times = data.time.values
    
    
    # Handle NaNs in colormap for transparency
    if not is_rgb and cmap:
        try:
             cm = plt.get_cmap(cmap)
             cm.set_bad(alpha=0.0) # Make NaNs transparent
             cmap = cm
        except:
             pass

    if vmin is None and not is_rgb:
         vmin = float(plot_data.min())
    if vmax is None and not is_rgb:
         vmax = float(plot_data.max())


    # Add Basemap FIRST (so it is underneath)
    if add_basemap and extent is not None:
        try:
            # Check CRS
            crs = data.rio.crs
            if crs is None:
                print("Warning: DataArray has no CRS (.rio.crs). Basemap might be misaligned.")
                
            # Use default source if not provided
            if basemap_source is None:
                basemap_source = ctx.providers.CartoDB.Positron
                
            # Add basemap with low zorder
            ctx.add_basemap(ax, crs=crs, source=basemap_source)
        except Exception as e:
             print(f"Warning: Failed to add basemap: {e}")

    if is_rgb:
        im = ax.imshow(plot_data.isel(time=0).values, interpolation='nearest', extent=extent, alpha=alpha, zorder=10)
    else:
        im = ax.imshow(plot_data.isel(time=0).values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest', extent=extent, alpha=alpha, zorder=10)
        if not is_rgb:
             # Add colorbar only once
             cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
        
    if add_text:
        title = ax.set_title(str(times[0])[:10], fontsize=16)

    def update(frame):
        if is_rgb:
            im.set_data(plot_data.isel(time=frame).values)
        else:
            im.set_data(plot_data.isel(time=frame).values)
        
        if add_text:
            title.set_text(str(times[frame])[:10])
        return im,

    print(f"Generating animation with {len(times)} frames...")
    ani = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(times), 
        interval=1000/fps, 
        blit=False # blit=True can cause issues with title updates sometimes
    )
    
    ani.save(output_path, writer='pillow', dpi=dpi)
    plt.close(fig)
    print(f"Saved animation to {output_path}")

