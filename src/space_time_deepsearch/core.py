"""
Space-Time DeepSearch: Core analysis entry point.

Provides a single class that wraps all data retrieval and temporal analysis
functions, accepting a geographic area of interest (bbox, geometry, or city name).
"""

import xarray as xr
import shapely.geometry
import geopandas as gpd

from .io.sentinel2 import get_sentinel2_imagery
from .io.modis import get_modis_temperature
from .io.landsat import get_landsat_imagery
from .temporal import run_landtrendr, extract_change_map, annual_composite, LandTrendrParams
from .temporal._landtrendr_viz import plot_change_map, plot_pixel_trajectory
from .vis.animation import create_timelapse


class SpaceTimeDeepSearch:
    """
    Entry point for satellite time series analysis.

    Manages the area of interest and provides methods to fetch satellite
    imagery and run temporal analyses including LandTrendr.

    Args:
        bbox: Bounding box as ``(west, south, east, north)`` in EPSG:4326.
        custom_geometry: Path to a GeoJSON file or a Shapely geometry.
        city: Name of a city to geocode automatically (requires osmnx).

    Raises:
        ValueError: If none or more than one of bbox/custom_geometry/city is given.
    """

    def __init__(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        custom_geometry: str | shapely.geometry.base.BaseGeometry | None = None,
        city: str | None = None,
    ):
        if city is not None:
            try:
                import osmnx as ox
                city_gdf = ox.geocode_to_gdf(city)
                custom_geometry = city_gdf.geometry.iloc[0]
                print(f"Located '{city}'")
            except Exception as e:
                raise ValueError(f"Could not locate city '{city}'.") from e

        if bbox is None and custom_geometry is None:
            raise ValueError("Provide bbox, custom_geometry, or city.")
        if bbox is not None and custom_geometry is not None:
            raise ValueError("Provide only one of bbox or custom_geometry.")

        self.bbox = bbox
        self.custom_geometry = custom_geometry

        # Storage for loaded datacubes
        self._data: dict[str, xr.DataArray] = {}

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def get_sentinel2(
        self,
        start_date: str,
        end_date: str,
        bands: list[str] | None = None,
        cloud_cover_max: int = 30,
        min_coverage: int = 0,
        resolution: int = 10,
        mask_clouds: bool = False,
        composite_period: str | None = None,
        composite_start: str | None = None,
        composite_end: str | None = None,
        add_ndvi: bool = False,
        add_ndbi: bool = False,
        add_nbr: bool = False,
        collection: str = "sentinel-2-l2a",
        chunksize: int = 2048,
        **kwargs,
    ) -> xr.DataArray:
        """
        Fetch Sentinel-2 L2A imagery for the area of interest.

        Args:
            start_date: ISO date string (YYYY-MM-DD).
            end_date: ISO date string (YYYY-MM-DD).
            bands: Band names to retrieve. Defaults to ["B02","B03","B04","B08"].
            cloud_cover_max: Maximum scene cloud cover percentage.
            min_coverage: Minimum spatial coverage percentage.
            resolution: Output resolution in metres.
            mask_clouds: Replace cloudy pixels with NaN.
            composite_period: Temporal compositing period (e.g. "1M", "1Y").
            composite_start: Start of annual date window for compositing
                in "MM-DD" format (e.g. "06-01"). Only scenes within this
                window are included in each composite.
            composite_end: End of annual date window for compositing
                in "MM-DD" format (e.g. "10-31").
            add_ndvi: Append NDVI as an extra band.
            add_ndbi: Append NDBI as an extra band.
            add_nbr: Append NBR as an extra band.
            collection: STAC collection ID.
            chunksize: Dask chunk size.

        Returns:
            xr.DataArray with dims ``(time, band, y, x)``.
        """
        data = get_sentinel2_imagery(
            bbox=self.bbox,
            custom_geometry=self.custom_geometry,
            start_date=start_date,
            end_date=end_date,
            bands=bands,
            cloud_cover_max=cloud_cover_max,
            min_coverage=min_coverage,
            resolution=resolution,
            mask_clouds=mask_clouds,
            composite_period=composite_period,
            composite_start=composite_start,
            composite_end=composite_end,
            add_ndvi=add_ndvi,
            add_ndbi=add_ndbi,
            add_nbr=add_nbr,
            collection=collection,
            chunksize=chunksize,
            **kwargs,
        )
        self._data["sentinel2"] = data
        return data

    def get_landsat(
        self,
        start_date: str,
        end_date: str,
        bands: list[str] | None = None,
        cloud_cover_max: int = 30,
        min_coverage: int = 0,
        resolution: int = 30,
        mask_clouds: bool = False,
        mask_snow: bool = False,
        composite_period: str | None = None,
        composite_start: str | None = None,
        composite_end: str | None = None,
        add_ndvi: bool = False,
        add_ndbi: bool = False,
        add_nbr: bool = False,
        missions: list[str] | None = None,
        exclude_slc_off: bool = False,
        apply_srf_correction: bool = False,
        apply_scale_factors: bool = True,
        collection: str = "landsat-c2-l2",
        chunksize: int = 2048,
        **kwargs,
    ) -> xr.DataArray:
        """
        Fetch harmonized Landsat imagery for the area of interest.

        Supports Landsat 4/5 TM, 7 ETM+, and 8/9 OLI via the Planetary
        Computer ``landsat-c2-l2`` collection with pre-harmonized band names.

        Args:
            start_date: ISO date string (YYYY-MM-DD).
            end_date: ISO date string (YYYY-MM-DD).
            bands: Harmonized band names. Defaults to
                ["blue", "green", "red", "nir08"].
            cloud_cover_max: Maximum scene cloud cover percentage.
            min_coverage: Minimum spatial coverage percentage.
            resolution: Output resolution in metres.
            mask_clouds: Replace cloudy pixels with NaN.
            mask_snow: Also mask snow/ice pixels (requires mask_clouds).
            composite_period: Temporal compositing period (e.g. "1M", "1Y").
            composite_start: Start of annual date window for compositing
                in "MM-DD" format (e.g. "06-01"). Only scenes within this
                window are included in each composite.
            composite_end: End of annual date window for compositing
                in "MM-DD" format (e.g. "10-31").
            add_ndvi: Append NDVI as an extra band.
            add_ndbi: Append NDBI as an extra band.
            add_nbr: Append NBR as an extra band.
            missions: Filter to specific platforms (e.g. ["landsat-8"]).
            exclude_slc_off: Drop Landsat 7 scenes after SLC failure (2003).
            apply_srf_correction: Apply Roy et al. (2016) TM/ETM+ → OLI
                spectral bandpass adjustment.
            apply_scale_factors: Convert DN to surface reflectance.
            collection: STAC collection ID.
            chunksize: Dask chunk size.

        Returns:
            xr.DataArray with dims ``(time, band, y, x)``.
        """
        data = get_landsat_imagery(
            bbox=self.bbox,
            custom_geometry=self.custom_geometry,
            start_date=start_date,
            end_date=end_date,
            bands=bands,
            cloud_cover_max=cloud_cover_max,
            min_coverage=min_coverage,
            resolution=resolution,
            mask_clouds=mask_clouds,
            mask_snow=mask_snow,
            composite_period=composite_period,
            composite_start=composite_start,
            composite_end=composite_end,
            add_ndvi=add_ndvi,
            add_ndbi=add_ndbi,
            add_nbr=add_nbr,
            missions=missions,
            exclude_slc_off=exclude_slc_off,
            apply_srf_correction=apply_srf_correction,
            apply_scale_factors=apply_scale_factors,
            collection=collection,
            chunksize=chunksize,
            **kwargs,
        )
        self._data["landsat"] = data
        return data

    def get_modis(
        self,
        start_date: str,
        end_date: str,
        layer: str = "LST_Day_1km",
        convert_to_celsius: bool = True,
        composite_period: str | None = "1W",
        composite_start: str | None = None,
        composite_end: str | None = None,
        resolution: int = 1000,
        collection: str = "modis-11A1-061",
        chunksize: int = 2048,
        **kwargs,
    ) -> xr.DataArray:
        """
        Fetch MODIS Land Surface Temperature (MOD11A1) data.

        Args:
            start_date: ISO date string (YYYY-MM-DD).
            end_date: ISO date string (YYYY-MM-DD).
            layer: MODIS asset name.
            convert_to_celsius: Convert Kelvin to Celsius.
            composite_period: Temporal compositing period.
            composite_start: Start of annual date window for compositing
                in "MM-DD" format (e.g. "06-01").
            composite_end: End of annual date window for compositing
                in "MM-DD" format (e.g. "10-31").
            resolution: Output resolution in metres.
            collection: STAC collection ID.
            chunksize: Dask chunk size.

        Returns:
            xr.DataArray with dims ``(time, band, y, x)``.
        """
        data = get_modis_temperature(
            bbox=self.bbox,
            custom_geometry=self.custom_geometry,
            start_date=start_date,
            end_date=end_date,
            layer=layer,
            convert_to_celsius=convert_to_celsius,
            composite_period=composite_period,
            composite_start=composite_start,
            composite_end=composite_end,
            resolution=resolution,
            collection=collection,
            chunksize=chunksize,
            **kwargs,
        )
        self._data["modis"] = data
        return data

    # ------------------------------------------------------------------
    # LandTrendr
    # ------------------------------------------------------------------

    def run_landtrendr(
        self,
        spectral_index: xr.DataArray,
        composite_to_annual: bool = True,
        composite_method: str = "median",
        params: LandTrendrParams | None = None,
    ) -> xr.Dataset:
        """
        Run LandTrendr temporal segmentation.

        Accepts any single-band annual (or sub-annual) DataArray with a
        ``time`` dimension. Sub-annual data is composited to annual values
        before segmentation when ``composite_to_annual=True``.

        If your input has a ``band`` dimension (e.g. from :meth:`get_sentinel2`),
        select the desired index first::

            ndvi = data.sel(band="NDVI", drop=True)
            result = stds.run_landtrendr(ndvi)

        Args:
            spectral_index: DataArray with dims ``(time, y, x)``.
            composite_to_annual: If True, resample to annual medians/means
                before running LandTrendr.
            composite_method: "median" or "mean" for annual compositing.
            params: LandTrendrParams. Uses GEE-equivalent defaults if None.

        Returns:
            xr.Dataset with ``source_values``, ``fitted_values``,
            ``is_vertex``, ``rmse``.
        """
        if composite_to_annual:
            spectral_index = annual_composite(spectral_index, method=composite_method)

        lt_result = run_landtrendr(spectral_index, params=params)
        self._data["lt_result"] = lt_result
        return lt_result

    def extract_change_map(
        self,
        lt_result: xr.Dataset | None = None,
        change_type: str = "greatest",
        delta_filter: str = "loss",
    ) -> xr.Dataset:
        """
        Derive change maps from LandTrendr results.

        Args:
            lt_result: Dataset from :meth:`run_landtrendr`. If None, uses the
                last result stored internally.
            change_type: ``"greatest"``, ``"longest"``, ``"steepest"``,
                or ``"newest"``.
            delta_filter: ``"loss"``, ``"gain"``, or ``"all"``.

        Returns:
            xr.Dataset with ``yod``, ``mag``, ``dur``, ``preval``,
            ``rate``, ``dsnr`` — all with dims ``(y, x)``.
        """
        if lt_result is None:
            if "lt_result" not in self._data:
                raise ValueError("No LandTrendr result available. Run run_landtrendr() first.")
            lt_result = self._data["lt_result"]

        return extract_change_map(lt_result, change_type=change_type, delta_filter=delta_filter)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_change_map(self, change_ds: xr.Dataset, variable: str = "mag", **kwargs):
        """Plot a change map variable (wraps :func:`plot_change_map`)."""
        return plot_change_map(change_ds, variable=variable, **kwargs)

    def plot_pixel_trajectory(
        self,
        lt_result: xr.Dataset,
        y: float,
        x: float,
        **kwargs,
    ):
        """Plot source + fitted trajectory for one pixel (wraps :func:`plot_pixel_trajectory`)."""
        return plot_pixel_trajectory(lt_result, y=y, x=x, **kwargs)

    def inspect_landtrendr(
        self,
        lt_result: xr.Dataset | None = None,
        change_ds: xr.Dataset | None = None,
        change_type: str = "greatest",
        delta_filter: str = "loss",
        **kwargs,
    ):
        """Open an interactive LandTrendr change map inspector.

        If ``change_ds`` is not provided, it is computed from ``lt_result``
        using the given ``change_type`` and ``delta_filter``.

        Args:
            lt_result: Dataset from :meth:`run_landtrendr`. If None, uses
                the last stored result.
            change_ds: Dataset from :meth:`extract_change_map`. If None,
                computed automatically.
            change_type: Passed to ``extract_change_map`` if computing.
            delta_filter: Passed to ``extract_change_map`` if computing.
            **kwargs: Forwarded to ``LandTrendrInspector``
                (cmaps, map_width, map_height).

        Returns:
            LandTrendrInspector instance.
        """
        if lt_result is None:
            if "lt_result" not in self._data:
                raise ValueError(
                    "No LandTrendr result available. "
                    "Run run_landtrendr() first."
                )
            lt_result = self._data["lt_result"]
        if change_ds is None:
            change_ds = extract_change_map(
                lt_result, change_type=change_type, delta_filter=delta_filter,
            )

        from .temporal._landtrendr_interactive import inspect_landtrendr as _inspect
        return _inspect(lt_result, change_ds, **kwargs)

    def save_change_map_geotiff(
        self,
        change_ds: xr.Dataset | None = None,
        output_dir: str = ".",
        variables: list[str] | None = None,
        crs: str | None = None,
        **kwargs,
    ) -> list[str]:
        """Export change map variables as GeoTIFF files.

        Args:
            change_ds: Dataset from :meth:`extract_change_map`. If None,
                computes from stored LandTrendr result.
            output_dir: Directory to write files into.
            variables: Variables to export. Defaults to all.
            crs: CRS string (e.g. ``"EPSG:32620"``).
            **kwargs: Forwarded to ``extract_change_map`` if computing
                (change_type, delta_filter).

        Returns:
            List of written file paths.
        """
        if change_ds is None:
            if "lt_result" not in self._data:
                raise ValueError(
                    "No LandTrendr result available. "
                    "Run run_landtrendr() first."
                )
            change_ds = extract_change_map(self._data["lt_result"], **kwargs)

        from .temporal._landtrendr_interactive import save_change_map_geotiff as _save
        return _save(change_ds, output_dir=output_dir,
                     variables=variables, crs=crs)

    # ------------------------------------------------------------------
    # Animation / Timelapse
    # ------------------------------------------------------------------

    def animate(
        self,
        source: str | None = None,
        data: xr.DataArray | None = None,
        output_path: str = "timelapse.gif",
        **kwargs,
    ) -> None:
        """
        Create a timelapse GIF from satellite data.

        Provide either ``source`` (a key like ``"sentinel2"``, ``"landsat"``,
        or ``"modis"`` that was previously loaded) **or** a raw ``data``
        DataArray.  All extra keyword arguments are forwarded to
        :func:`~space_time_deepsearch.vis.animation.create_timelapse`.

        Args:
            source: Key into the internal data store (e.g. ``"sentinel2"``).
            data: An xarray DataArray with a ``time`` dimension.
            output_path: Filename for the output GIF.
            **kwargs: Passed to ``create_timelapse`` (fps, cmap, bands,
                vmin, vmax, add_basemap, alpha, figsize, dpi, …).

        Raises:
            ValueError: If neither ``source`` nor ``data`` is provided,
                or if the requested source has not been loaded.
        """
        if data is None and source is None:
            raise ValueError(
                "Provide either 'source' (e.g. 'sentinel2') or a 'data' DataArray."
            )

        if data is None:
            if source not in self._data:
                available = list(self._data.keys()) or ["(none)"]
                raise ValueError(
                    f"No data loaded for '{source}'. "
                    f"Available sources: {available}. "
                    f"Fetch data first with get_sentinel2() / get_landsat() / get_modis()."
                )
            data = self._data[source]

        create_timelapse(data, output_path=output_path, **kwargs)
