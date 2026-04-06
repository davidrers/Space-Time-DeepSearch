"""
Visualization functions for LandTrendr segmentation results.

Provides plotting for change maps and per-pixel spectral trajectories.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_change_map(change_ds, variable="mag", cmap="RdBu", title=None,
                    ax=None, figsize=(10, 8), vmin=None, vmax=None):
    """Plot a single change map variable as a 2-D raster image.

    Args:
        change_ds: xr.Dataset from ``extract_change_map()``.
        variable: Variable name to plot. One of "yod", "mag", "dur",
            "preval", "rate", "dsnr".
        cmap: Matplotlib colormap name.
        title: Plot title. Defaults to the variable name.
        ax: Optional matplotlib Axes. If None, creates a new figure.
        figsize: Figure size if creating a new figure.
        vmin: Minimum value for colormap scaling.
        vmax: Maximum value for colormap scaling.

    Returns:
        matplotlib.figure.Figure
    """
    if variable not in change_ds:
        raise ValueError(
            f"Variable '{variable}' not in dataset. "
            f"Available: {list(change_ds.data_vars)}"
        )

    data = change_ds[variable]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Determine extent from coordinates
    extent = None
    if "x" in data.coords and "y" in data.coords:
        xs = data.x.values
        ys = data.y.values
        extent = [xs.min(), xs.max(), ys.min(), ys.max()]

    im = ax.imshow(
        data.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        interpolation="nearest",
    )

    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    ax.set_title(title or variable)

    if extent is not None:
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")

    return fig


def plot_pixel_trajectory(lt_result, y, x, ax=None, figsize=(12, 4)):
    """Plot source and fitted spectral trajectory for a single pixel.

    Shows the original annual values as dots, the piecewise-linear
    fitted line, and marks vertices with triangles.

    Args:
        lt_result: xr.Dataset from ``run_landtrendr()``.
        y: Y coordinate (northing or index) of the pixel.
        x: X coordinate (easting or index) of the pixel.
        ax: Optional matplotlib Axes. If None, creates a new figure.
        figsize: Figure size if creating a new figure.

    Returns:
        matplotlib.figure.Figure
    """
    source = lt_result["source_values"].sel(y=y, x=x, method="nearest")
    fitted = lt_result["fitted_values"].sel(y=y, x=x, method="nearest")
    vertices = lt_result["is_vertex"].sel(y=y, x=x, method="nearest")

    if np.issubdtype(source.time.dtype, np.datetime64):
        time_vals = source.time.dt.year.values
    else:
        time_vals = source.time.values

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Source values as dots
    ax.scatter(time_vals, source.values, color="gray", s=30, zorder=3,
               label="Source")

    # Fitted trajectory as line
    ax.plot(time_vals, fitted.values, color="tab:red", linewidth=2, zorder=2,
            label="Fitted")

    # Mark vertices
    vtx_mask = vertices.values.astype(bool)
    if np.any(vtx_mask):
        ax.scatter(time_vals[vtx_mask], fitted.values[vtx_mask],
                   color="tab:red", marker="^", s=80, zorder=4,
                   edgecolors="black", linewidths=0.5, label="Vertices")

    ax.set_xlabel("Year")
    ax.set_ylabel("Spectral Value")
    ax.set_title(f"LandTrendr Trajectory (y={y}, x={x})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
