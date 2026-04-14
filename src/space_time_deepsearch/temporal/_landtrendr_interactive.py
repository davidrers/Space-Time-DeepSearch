"""
Interactive LandTrendr change map inspector using HoloViews + Panel.

Provides an interactive dashboard that displays change maps (YOD, Magnitude,
Duration) with layer toggling and click-to-inspect pixel trajectories.

This module is fully isolated from the static matplotlib-based plots in
``_landtrendr_viz.py``.
"""

from pathlib import Path

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 — registers .rio accessor on xarray
from pyproj import Transformer

import param
import holoviews as hv
from holoviews.streams import Params
import hvplot.xarray  # noqa: F401 — registers .hvplot accessor on xarray
import panel as pn

hv.extension("bokeh")


class _ClickState(param.Parameterized):
    """Shared mutable state for the currently selected pixel coordinates."""
    x = param.Number(default=None, allow_None=True)
    y = param.Number(default=None, allow_None=True)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_CMAPS = {
    "yod": "plasma",
    "mag": "RdBu_r",
    "dur": "YlOrBr",
}

# Layer order: Magnitude (bottom), Year of Detection, Duration (top)
_LAYER_ORDER = ["mag", "yod", "dur"]

_LAYER_LABELS = {
    "mag": "Magnitude",
    "yod": "Year of Detection",
    "dur": "Duration",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_years(time_coord):
    """Return fractional year array from a time coordinate."""
    if np.issubdtype(time_coord.dtype, np.datetime64):
        from .landtrendr import _datetime_to_fractional_year
        return _datetime_to_fractional_year(time_coord.values)
    return time_coord.values


def _build_trajectory_plot(lt_result, change_ds, x, y, width=420, height=250):
    """Build a HoloViews overlay for a single pixel trajectory.

    Parameters
    ----------
    lt_result : xr.Dataset
        Output of ``run_landtrendr()``.
    change_ds : xr.Dataset
        Output of ``extract_change_map()``.
    x, y : float
        Geographic coordinates (easting, northing) of the pixel.
    width, height : int
        Plot dimensions in pixels.

    Returns
    -------
    hv.Overlay
    """
    source = lt_result["source_values"].sel(y=y, x=x, method="nearest")
    fitted = lt_result["fitted_values"].sel(y=y, x=x, method="nearest")
    vertices = lt_result["is_vertex"].sel(y=y, x=x, method="nearest")

    years = _extract_years(source.time)
    src_vals = source.values.astype(float)
    fit_vals = fitted.values.astype(float)

    # Source values — gray dots
    source_pts = hv.Scatter(
        (years, src_vals), "Year", "Spectral Value", label="Source"
    ).opts(color="gray", size=6, tools=["hover"])

    # Fitted trajectory — red line
    fitted_curve = hv.Curve(
        (years, fit_vals), "Year", "Spectral Value", label="Fitted"
    ).opts(color="red", line_width=2)

    # Vertices — red triangles
    vtx_mask = vertices.values.astype(bool)
    elements = [source_pts, fitted_curve]
    if np.any(vtx_mask):
        vertex_pts = hv.Scatter(
            (years[vtx_mask], fit_vals[vtx_mask]),
            "Year", "Spectral Value", label="Vertices",
        ).opts(color="red", marker="triangle", size=10,
               line_color="black", line_width=0.5)
        elements.append(vertex_pts)

    # YOD vertical line
    yod_val = change_ds["yod"].sel(y=y, x=x, method="nearest").values
    if not np.isnan(yod_val):
        vline = hv.VLine(float(yod_val)).opts(
            color="gold", line_dash="dashed", line_width=2,
        )
        elements.append(vline)

    overlay = hv.Overlay(elements).opts(
        hv.opts.Overlay(
            width=width, height=height, title="Fitted Trajectory",
            show_legend=False, show_grid=True,
        )
    )
    return overlay


def _build_info_markdown(lt_result, change_ds, x, y):
    """Build a Markdown string with key pixel parameters.

    Parameters
    ----------
    lt_result : xr.Dataset
        Output of ``run_landtrendr()``.
    change_ds : xr.Dataset
        Output of ``extract_change_map()``.
    x, y : float
        Geographic coordinates (easting, northing) of the pixel.

    Returns
    -------
    str
        Markdown-formatted inspector info.
    """
    def _fmt(val, decimals=4):
        if np.isnan(val):
            return "—"
        if decimals == 0:
            return str(int(val))
        return f"{val:.{decimals}f}"

    yod = float(change_ds["yod"].sel(y=y, x=x, method="nearest").values)
    mag = float(change_ds["mag"].sel(y=y, x=x, method="nearest").values)
    dur = float(change_ds["dur"].sel(y=y, x=x, method="nearest").values)
    rmse = float(lt_result["rmse"].sel(y=y, x=x, method="nearest").values)

    actual_x = float(change_ds.x.sel(x=x, method="nearest").values)
    actual_y = float(change_ds.y.sel(y=y, method="nearest").values)

    lines = [
        "**Inspector**",
        "",
        f"**Easting:** {actual_x:.0f}",
        f"**Northing:** {actual_y:.0f}",
        "",
        f"**Year:** {_fmt(yod, 0)}",
        f"**Magnitude:** {_fmt(mag)}",
        f"**Duration:** {_fmt(dur, 0)}",
        f"**Fit RMSE:** {_fmt(rmse)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LandTrendrInspector
# ---------------------------------------------------------------------------

class LandTrendrInspector:
    """Interactive LandTrendr change map inspector.

    Displays a single large map with Magnitude, Year of Detection, and
    Duration as toggleable overlaid layers (like a GIS viewer). The
    trajectory plot is shown to the right of the map. Export controls
    are placed below.

    Parameters
    ----------
    lt_result : xr.Dataset
        Output of ``run_landtrendr()`` containing ``source_values``,
        ``fitted_values``, ``is_vertex``, ``rmse``.
    change_ds : xr.Dataset
        Output of ``extract_change_map()`` containing at least ``yod``,
        ``mag``, ``dur``.
    cmaps : dict[str, str] | None
        Colormaps per variable. Defaults to
        ``{"yod": "plasma", "mag": "RdBu_r", "dur": "YlOrBr"}``.
    map_width : int
        Width of the map panel in pixels (default 600).
    map_height : int
        Height of the map panel in pixels (default 600).

    Examples
    --------
    >>> lt = run_landtrendr(ndvi_annual)
    >>> change = extract_change_map(lt, change_type="greatest", delta_filter="loss")
    >>> inspector = LandTrendrInspector(lt, change)
    >>> inspector.show()  # renders in Jupyter or opens browser
    """

    def __init__(
        self,
        lt_result: xr.Dataset,
        change_ds: xr.Dataset,
        cmaps: dict[str, str] | None = None,
        map_width: int = 600,
        map_height: int = 600,
    ):
        # Validate inputs
        for var in ("source_values", "fitted_values", "is_vertex", "rmse"):
            if var not in lt_result:
                raise ValueError(
                    f"lt_result is missing '{var}'. "
                    f"Available: {list(lt_result.data_vars)}"
                )
        for var in ("yod", "mag", "dur"):
            if var not in change_ds:
                raise ValueError(
                    f"change_ds is missing '{var}'. "
                    f"Available: {list(change_ds.data_vars)}"
                )

        self.lt_result = lt_result
        self.change_ds = change_ds
        self.cmaps = {**_DEFAULT_CMAPS, **(cmaps or {})}
        self.map_width = map_width
        self.map_height = map_height

        # Initial pixel — center of the dataset
        self._init_x = float(change_ds.x.values[len(change_ds.x) // 2])
        self._init_y = float(change_ds.y.values[len(change_ds.y) // 2])

        # Build the dashboard
        self._layout = self._build()

    # ------------------------------------------------------------------
    # Dashboard construction
    # ------------------------------------------------------------------

    def _build(self):
        """Construct the full Panel layout.

        Layout: single large map (left) with basemap tiles and a compact
        info panel (right) containing instructions, pixel parameters,
        small trajectory plot, and export controls. Layer checkboxes sit
        below the map.
        """
        panel_width = 350

        # --- Shared click state -------------------------------------------
        click_state = _ClickState(x=self._init_x, y=self._init_y)
        self._click_state = click_state  # keep reference

        # --- Detect CRS and prepare basemap reprojection ------------------
        src_crs = None
        to_webmerc = None   # Transformer: source CRS → EPSG:3857
        from_webmerc = None  # Transformer: EPSG:3857 → source CRS

        # Try to get CRS from the first change variable
        first_var = self.change_ds[_LAYER_ORDER[0]]
        try:
            src_crs = first_var.rio.crs
        except Exception:
            pass

        use_basemap = src_crs is not None
        if use_basemap:
            to_webmerc = Transformer.from_crs(
                src_crs, "EPSG:3857", always_xy=True,
            )
            from_webmerc = Transformer.from_crs(
                "EPSG:3857", src_crs, always_xy=True,
            )
            # Reproject initial click coords to Web Mercator
            init_x_wm, init_y_wm = to_webmerc.transform(
                self._init_x, self._init_y,
            )
        else:
            init_x_wm = self._init_x
            init_y_wm = self._init_y

        # Update click state to use display coordinates
        click_state.x = init_x_wm
        click_state.y = init_y_wm

        # --- Build individual layer images --------------------------------
        layer_images = {}

        for var in _LAYER_ORDER:
            data = self.change_ds[var]
            cmap = self.cmaps[var]

            # Compute clim from original data (before reprojection adds nodata)
            valid = data.values[~np.isnan(data.values)]
            if var == "mag" and len(valid) > 0:
                vabs = max(abs(valid.min()), abs(valid.max()))
                clim = (-vabs, vabs) if vabs > 0 else None
            elif len(valid) > 0:
                clim = (float(valid.min()), float(valid.max()))
            else:
                clim = None

            # Reproject to Web Mercator for basemap alignment
            if use_basemap:
                data = data.rio.reproject("EPSG:3857")

            img = data.hvplot.image(
                x="x", y="y",
                cmap=cmap, clim=clim,
                xlabel="Easting", ylabel="Northing",
                width=self.map_width, height=self.map_height,
                tools=["tap", "hover"],
                alpha=0.85,
            )
            layer_images[var] = img

        # --- Compute data extent for view locking --------------------------
        # Use the first layer's reprojected bounds (all layers share same AOI)
        ref = self.change_ds[_LAYER_ORDER[0]]
        if use_basemap:
            ref = ref.rio.reproject("EPSG:3857")
        data_xlim = (float(ref.x.min()), float(ref.x.max()))
        data_ylim = (float(ref.y.min()), float(ref.y.max()))

        # --- Shared Params stream from click state ------------------------
        params_stream = Params(click_state, ["x", "y"])

        # --- Helper: convert display coords to native coords -------------
        def _to_native(x, y):
            """Convert display (possibly Web Mercator) coords to native."""
            if from_webmerc is not None:
                return from_webmerc.transform(x, y)
            return x, y

        # --- Marker overlay (reactive to shared state) --------------------
        def _marker(x, y):
            if x is None or y is None:
                return hv.Points([])
            return hv.Points([(x, y)]).opts(
                color="white", marker="star", size=15,
                line_color="black", line_width=1.5,
            )

        marker_dmap = hv.DynamicMap(_marker, streams=[params_stream])

        # --- Layer toggle via reactive map --------------------------------
        layer_names = [_LAYER_LABELS[var] for var in _LAYER_ORDER]
        active_layers = pn.widgets.CheckBoxGroup(
            name="Layers",
            value=layer_names,
            options=layer_names,
            inline=True,
        )

        # Attach tap stream to each layer image for click capture
        for var in _LAYER_ORDER:
            t = hv.streams.SingleTap(
                source=layer_images[var],
                x=init_x_wm, y=init_y_wm,
            )
            t.add_subscriber(
                lambda x, y, _state=click_state: _state.param.update(x=x, y=y)
            )

        # Basemap tile layer (satellite imagery)
        tiles = hv.element.tiles.EsriImagery() if use_basemap else None

        def _build_map(active):
            """Rebuild the map overlay based on active layer checkboxes."""
            # Collect active data layers
            data_elements = []
            for var in _LAYER_ORDER:
                label = _LAYER_LABELS[var]
                if label in active:
                    data_elements.append(layer_images[var])

            # Colorbar: only on topmost active layer
            styled = []
            for i, el in enumerate(data_elements):
                if i < len(data_elements) - 1:
                    styled.append(el.opts(colorbar=False))
                else:
                    styled.append(el.opts(colorbar=True))

            # Build overlay: tiles (bottom) + data layers + marker (top)
            elements = []
            if tiles is not None:
                elements.append(tiles)
            elements.extend(styled)

            if not data_elements:
                # No data layers — show tiles only, locked to data extent
                base = tiles if tiles is not None else hv.Curve([])
                return base.opts(
                    width=self.map_width, height=self.map_height,
                    xlim=data_xlim, ylim=data_ylim,
                    title="All layers hidden",
                )

            elements.append(marker_dmap)
            overlay = elements[0]
            for el in elements[1:]:
                overlay = overlay * el
            return overlay.opts(
                hv.opts.Overlay(
                    width=self.map_width, height=self.map_height,
                    xlim=data_xlim, ylim=data_ylim,
                    title="LandTrendr Change Map",
                )
            )

        map_dmap = pn.bind(_build_map, active=active_layers)
        map_pane = pn.pane.HoloViews(map_dmap, linked_axes=False)

        # --- Right panel: instructions ------------------------------------
        instructions = pn.pane.Markdown(
            "**Instructions**\n\n"
            "1. Toggle layers with the checkboxes below the map\n"
            "2. Click a pixel on the map to inspect its trajectory\n"
            "3. Export change maps as GeoTIFF below",
            width=panel_width,
        )

        # --- Right panel: inspector info (reactive) -----------------------
        def _info(x, y):
            if x is None or y is None:
                return "**Inspector**\n\nNo pixel selected."
            native_x, native_y = _to_native(x, y)
            return _build_info_markdown(
                self.lt_result, self.change_ds, native_x, native_y,
            )

        info_pane = pn.pane.Markdown(
            pn.bind(_info, x=click_state.param.x, y=click_state.param.y),
            width=panel_width,
        )

        # --- Right panel: trajectory --------------------------------------
        traj_width = 420
        traj_height = 250

        def _trajectory(x, y):
            if x is None or y is None:
                return hv.Curve([]).opts(
                    width=traj_width, height=traj_height,
                    title="Click a pixel to see its trajectory",
                )
            native_x, native_y = _to_native(x, y)
            return _build_trajectory_plot(
                self.lt_result, self.change_ds, native_x, native_y,
                width=traj_width, height=traj_height,
            )

        traj_dmap = hv.DynamicMap(_trajectory, streams=[params_stream])
        traj_pane = pn.pane.HoloViews(traj_dmap, linked_axes=False)

        # --- Right panel: export controls ---------------------------------
        dir_input = pn.widgets.TextInput(
            name="Output directory", value=".", width=panel_width - 10,
        )
        save_btn = pn.widgets.Button(
            name="Save GeoTIFFs", button_type="primary",
            width=panel_width - 10,
        )
        save_status = pn.pane.Markdown("", width=panel_width)

        def _on_save(event):
            try:
                paths = save_change_map_geotiff(
                    self.change_ds, output_dir=dir_input.value,
                )
                names = ", ".join(Path(p).name for p in paths)
                save_status.object = f"Saved: {names}"
            except Exception as e:
                save_status.object = f"Error: {e}"

        save_btn.on_click(_on_save)

        # --- Assemble right panel -----------------------------------------
        right_panel = pn.Column(
            instructions,
            pn.layout.Divider(),
            info_pane,
            pn.layout.Divider(),
            traj_pane,
            pn.layout.Divider(),
            dir_input,
            save_btn,
            save_status,
            width=panel_width + 20,
        )

        # --- Assemble full layout -----------------------------------------
        main_row = pn.Row(map_pane, right_panel)

        layout = pn.Column(
            pn.pane.Markdown("### LandTrendr Interactive Inspector"),
            main_row,
            active_layers,
        )
        return layout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_geotiff(
        self,
        output_dir: str = ".",
        variables: list[str] | None = None,
        crs: str | None = None,
    ) -> list[str]:
        """Export change map variables as GeoTIFF files.

        Writes one file per variable (e.g. ``yod.tif``, ``mag.tif``).

        Parameters
        ----------
        output_dir : str
            Directory to write files into. Created if it doesn't exist.
        variables : list[str] | None
            Variables to export. Defaults to all in ``change_ds``.
        crs : str | None
            CRS string (e.g. ``"EPSG:32620"``). If None, uses the CRS
            already set on the data via rioxarray, if any.

        Returns
        -------
        list[str]
            Paths of written GeoTIFF files.
        """
        return save_change_map_geotiff(
            self.change_ds, output_dir=output_dir,
            variables=variables, crs=crs,
        )

    def panel(self) -> pn.Column:
        """Return the Panel layout for embedding in larger dashboards."""
        return self._layout

    def show(self):
        """Display the dashboard in Jupyter or open in a browser tab."""
        return self._layout.show()

    def servable(self):
        """Mark the layout as servable for ``panel serve``."""
        return self._layout.servable()

    def _repr_mimebundle_(self, **kwargs):
        """Enable direct rendering in Jupyter by returning the Panel repr."""
        return self._layout._repr_mimebundle_(**kwargs)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def inspect_landtrendr(
    lt_result: xr.Dataset,
    change_ds: xr.Dataset,
    cmaps: dict[str, str] | None = None,
    map_width: int = 600,
    map_height: int = 600,
) -> LandTrendrInspector:
    """Create and return an interactive LandTrendr inspector.

    This is a convenience wrapper around :class:`LandTrendrInspector`.

    Parameters
    ----------
    lt_result : xr.Dataset
        Output of ``run_landtrendr()``.
    change_ds : xr.Dataset
        Output of ``extract_change_map()``.
    cmaps : dict[str, str] | None
        Colormaps per variable.
    map_width, map_height : int
        Map panel dimensions in pixels.

    Returns
    -------
    LandTrendrInspector
        Keep a reference to prevent garbage collection of event handlers.
    """
    return LandTrendrInspector(
        lt_result, change_ds,
        cmaps=cmaps, map_width=map_width, map_height=map_height,
    )


# ---------------------------------------------------------------------------
# GeoTIFF export
# ---------------------------------------------------------------------------

def save_change_map_geotiff(
    change_ds: xr.Dataset,
    output_dir: str = ".",
    variables: list[str] | None = None,
    crs: str | None = None,
) -> list[str]:
    """Export change map variables as GeoTIFF files.

    Writes one ``.tif`` file per variable (e.g. ``yod.tif``, ``mag.tif``).
    Requires the Dataset to have ``x`` and ``y`` coordinates.

    Parameters
    ----------
    change_ds : xr.Dataset
        Output of ``extract_change_map()``.
    output_dir : str
        Directory to write files into. Created if it doesn't exist.
    variables : list[str] | None
        Variables to export. Defaults to all data variables in the Dataset.
    crs : str | None
        CRS string (e.g. ``"EPSG:32620"``). If None, uses the CRS
        already set on the data via rioxarray, if any.

    Returns
    -------
    list[str]
        Paths of written GeoTIFF files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if variables is None:
        variables = list(change_ds.data_vars)

    paths = []
    for var in variables:
        if var not in change_ds:
            raise ValueError(
                f"Variable '{var}' not in dataset. "
                f"Available: {list(change_ds.data_vars)}"
            )
        da = change_ds[var]

        # Ensure spatial dims are named x, y for rioxarray
        if "x" in da.dims and "y" in da.dims:
            da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

        if crs is not None:
            da = da.rio.write_crs(crs)

        filepath = str(out / f"{var}.tif")
        da.rio.to_raster(filepath)
        paths.append(filepath)

    return paths
