"""Tests for the interactive LandTrendr inspector module."""

import os

import numpy as np
import pytest
import xarray as xr


def _make_synthetic_lt_data(ny=5, nx=5, n_years=20, seed=42):
    """Create synthetic lt_result and change_ds for testing.

    Returns (lt_result, change_ds) matching the structure of
    ``run_landtrendr()`` and ``extract_change_map()`` outputs.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    ys = np.arange(ny, dtype=float) * 30 + 500000  # fake UTM northing
    xs = np.arange(nx, dtype=float) * 30 + 600000  # fake UTM easting

    source = rng.random((n_years, ny, nx)) * 0.1 + 0.5
    fitted = source.copy()
    is_vertex = np.zeros((n_years, ny, nx), dtype=float)
    rmse = rng.random((ny, nx)) * 0.01

    # Pixel (0,0): step change with vertices at start, midpoint, end
    source[:7, 0, 0] = 0.8
    source[7:11, 0, 0] = np.linspace(0.8, 0.2, 4)
    source[11:, 0, 0] = 0.2
    fitted[:7, 0, 0] = 0.8
    fitted[7:11, 0, 0] = np.linspace(0.8, 0.2, 4)
    fitted[11:, 0, 0] = 0.2
    is_vertex[0, 0, 0] = 1.0
    is_vertex[7, 0, 0] = 1.0
    is_vertex[11, 0, 0] = 1.0
    is_vertex[-1, 0, 0] = 1.0

    lt_result = xr.Dataset(
        {
            "source_values": (["time", "y", "x"], source),
            "fitted_values": (["time", "y", "x"], fitted),
            "is_vertex": (["time", "y", "x"], is_vertex),
            "rmse": (["y", "x"], rmse),
        },
        coords={"time": years, "y": ys, "x": xs},
    )

    # Change map
    yod = np.full((ny, nx), np.nan)
    mag = np.full((ny, nx), np.nan)
    dur = np.full((ny, nx), np.nan)
    preval = np.full((ny, nx), np.nan)
    rate = np.full((ny, nx), np.nan)
    dsnr = np.full((ny, nx), np.nan)

    # Pixel (0,0): disturbance
    yod[0, 0] = 2007
    mag[0, 0] = -0.6
    dur[0, 0] = 4
    preval[0, 0] = 0.8
    rate[0, 0] = -0.15
    dsnr[0, 0] = 60.0

    change_ds = xr.Dataset(
        {
            "yod": (["y", "x"], yod),
            "mag": (["y", "x"], mag),
            "dur": (["y", "x"], dur),
            "preval": (["y", "x"], preval),
            "rate": (["y", "x"], rate),
            "dsnr": (["y", "x"], dsnr),
        },
        coords={"y": ys, "x": xs},
    )

    return lt_result, change_ds


# ---------------------------------------------------------------------------
# Import guard — skip all tests if holoviews/panel are not installed
# ---------------------------------------------------------------------------

hvp = pytest.importorskip("holoviews")
pn = pytest.importorskip("panel")

# Use a non-interactive renderer for CI
hvp.extension("bokeh")


class TestLandTrendrInspector:
    """Tests for the LandTrendrInspector class."""

    def test_construction(self):
        """Inspector creates without errors and produces a Panel layout."""
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            LandTrendrInspector,
        )

        lt_result, change_ds = _make_synthetic_lt_data()
        inspector = LandTrendrInspector(lt_result, change_ds)
        layout = inspector.panel()
        assert layout is not None
        assert isinstance(layout, pn.Column)

    def test_missing_lt_variable_raises(self):
        """Missing variables in lt_result raise ValueError."""
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            LandTrendrInspector,
        )

        lt_result, change_ds = _make_synthetic_lt_data()
        bad_lt = lt_result.drop_vars("fitted_values")
        with pytest.raises(ValueError, match="fitted_values"):
            LandTrendrInspector(bad_lt, change_ds)

    def test_missing_change_variable_raises(self):
        """Missing variables in change_ds raise ValueError."""
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            LandTrendrInspector,
        )

        lt_result, change_ds = _make_synthetic_lt_data()
        bad_change = change_ds.drop_vars("yod")
        with pytest.raises(ValueError, match="yod"):
            LandTrendrInspector(lt_result, bad_change)

    def test_custom_cmaps(self):
        """Custom colormaps are accepted."""
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            LandTrendrInspector,
        )

        lt_result, change_ds = _make_synthetic_lt_data()
        inspector = LandTrendrInspector(
            lt_result, change_ds, cmaps={"yod": "viridis"}
        )
        assert inspector.cmaps["yod"] == "viridis"
        # Others keep defaults
        assert inspector.cmaps["mag"] == "RdBu_r"


class TestBuildTrajectoryPlot:
    """Tests for the _build_trajectory_plot helper."""

    def test_returns_overlay(self):
        """Trajectory builder returns an hv.Overlay."""
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            _build_trajectory_plot,
        )

        lt_result, change_ds = _make_synthetic_lt_data()
        # Use pixel (0,0) which has known data
        x = float(change_ds.x.values[0])
        y = float(change_ds.y.values[0])
        overlay = _build_trajectory_plot(lt_result, change_ds, x, y)
        assert isinstance(overlay, hvp.Overlay)

    def test_nan_pixel_no_crash(self):
        """Clicking a pixel with all-NaN change data does not crash."""
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            _build_trajectory_plot,
        )

        lt_result, change_ds = _make_synthetic_lt_data()
        # Pixel (2,2) has NaN change values
        x = float(change_ds.x.values[2])
        y = float(change_ds.y.values[2])
        overlay = _build_trajectory_plot(lt_result, change_ds, x, y)
        assert isinstance(overlay, hvp.Overlay)

    def test_trajectory_with_vertices(self):
        """Pixel (0,0) trajectory includes vertex markers."""
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            _build_trajectory_plot,
        )

        lt_result, change_ds = _make_synthetic_lt_data()
        x = float(change_ds.x.values[0])
        y = float(change_ds.y.values[0])
        overlay = _build_trajectory_plot(lt_result, change_ds, x, y)
        # Should have at least 3 elements: scatter, curve, vertices
        # (possibly 4 with VLine)
        assert len(overlay) >= 3


class TestInspectLandtrendrFunction:
    """Tests for the convenience function."""

    def test_returns_inspector(self):
        """inspect_landtrendr returns a LandTrendrInspector instance."""
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            LandTrendrInspector,
            inspect_landtrendr,
        )

        lt_result, change_ds = _make_synthetic_lt_data()
        inspector = inspect_landtrendr(lt_result, change_ds)
        assert isinstance(inspector, LandTrendrInspector)


class TestSaveChangeMapGeotiff:
    """Tests for GeoTIFF export."""

    def test_writes_default_variables(self, tmp_path):
        """Exports all variables as separate .tif files."""
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            save_change_map_geotiff,
        )

        _, change_ds = _make_synthetic_lt_data()
        paths = save_change_map_geotiff(change_ds, output_dir=str(tmp_path))

        assert len(paths) == len(change_ds.data_vars)
        for p in paths:
            assert os.path.isfile(p)
            assert p.endswith(".tif")

    def test_writes_selected_variables(self, tmp_path):
        """Exports only specified variables."""
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            save_change_map_geotiff,
        )

        _, change_ds = _make_synthetic_lt_data()
        paths = save_change_map_geotiff(
            change_ds, output_dir=str(tmp_path), variables=["yod", "mag"],
        )

        assert len(paths) == 2
        names = {os.path.basename(p) for p in paths}
        assert names == {"yod.tif", "mag.tif"}

    def test_files_readable_with_rasterio(self, tmp_path):
        """Written files are valid rasterio-readable GeoTIFFs."""
        import rasterio
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            save_change_map_geotiff,
        )

        _, change_ds = _make_synthetic_lt_data()
        paths = save_change_map_geotiff(
            change_ds, output_dir=str(tmp_path), variables=["yod"],
        )

        with rasterio.open(paths[0]) as src:
            data = src.read(1)
            assert data.shape == (5, 5)  # ny=5, nx=5

    def test_crs_is_written(self, tmp_path):
        """CRS is embedded in the output GeoTIFF when specified."""
        import rasterio
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            save_change_map_geotiff,
        )

        _, change_ds = _make_synthetic_lt_data()
        paths = save_change_map_geotiff(
            change_ds, output_dir=str(tmp_path),
            variables=["yod"], crs="EPSG:32620",
        )

        with rasterio.open(paths[0]) as src:
            assert src.crs is not None
            assert src.crs.to_epsg() == 32620

    def test_invalid_variable_raises(self, tmp_path):
        """Unknown variable name raises ValueError."""
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            save_change_map_geotiff,
        )

        _, change_ds = _make_synthetic_lt_data()
        with pytest.raises(ValueError, match="nonexistent"):
            save_change_map_geotiff(
                change_ds, output_dir=str(tmp_path),
                variables=["nonexistent"],
            )

    def test_inspector_save_geotiff(self, tmp_path):
        """Inspector.save_geotiff delegates correctly."""
        from space_time_deepsearch.temporal._landtrendr_interactive import (
            LandTrendrInspector,
        )

        lt_result, change_ds = _make_synthetic_lt_data()
        inspector = LandTrendrInspector(lt_result, change_ds)
        paths = inspector.save_geotiff(
            output_dir=str(tmp_path), variables=["mag"],
        )

        assert len(paths) == 1
        assert os.path.isfile(paths[0])
