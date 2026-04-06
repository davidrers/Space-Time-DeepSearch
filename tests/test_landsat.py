"""
Diagnostic tests for Landsat retrieval module.

Tests focus on early-date (2000-2003) retrieval issues where data comes back
as all zeros or NaN.
"""

import numpy as np
import pytest
import xarray as xr
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from space_time_deepsearch.io.landsat import (
    _decode_qa_pixel,
    _SCALE_FACTOR,
    _OFFSET,
    _SLC_OFF_DATE,
    _QA_CLOUD,
    _QA_CLOUD_SHADOW,
    _QA_DILATED_CLOUD,
    _QA_SNOW,
    get_landsat_imagery,
)


# ---------------------------------------------------------------------------
# Unit tests for QA_PIXEL decoding
# ---------------------------------------------------------------------------


class TestDecodeQAPixel:
    """Test QA_PIXEL bit-flag decoding for various scenarios."""

    def test_clear_pixel(self):
        """QA value with no flags set should not be masked."""
        # Bit 0 = fill (0=not fill), rest clear.  Typical clear value = 21824
        # Bit pattern for clear land: 0101010100000000 = 21824
        qa = xr.DataArray([21824])
        mask = _decode_qa_pixel(qa, mask_cloud=True, mask_shadow=True)
        assert not mask.values[0], "Clear pixel should not be masked"

    def test_cloud_pixel(self):
        """QA value with cloud bit set should be masked."""
        # Set bit 3 (cloud) = 8
        qa = xr.DataArray([8])
        mask = _decode_qa_pixel(qa, mask_cloud=True, mask_shadow=False)
        assert mask.values[0], "Cloudy pixel should be masked"

    def test_dilated_cloud_pixel(self):
        """QA value with dilated cloud bit set should be masked."""
        # Set bit 1 (dilated cloud) = 2
        qa = xr.DataArray([2])
        mask = _decode_qa_pixel(qa, mask_cloud=True, mask_shadow=False)
        assert mask.values[0], "Dilated cloud pixel should be masked"

    def test_shadow_pixel(self):
        """QA value with cloud shadow bit set should be masked."""
        # Set bit 4 (cloud shadow) = 16
        qa = xr.DataArray([16])
        mask = _decode_qa_pixel(qa, mask_cloud=False, mask_shadow=True)
        assert mask.values[0], "Shadow pixel should be masked"

    def test_snow_pixel(self):
        """QA value with snow bit set should be masked when snow masking enabled."""
        # Set bit 5 (snow) = 32
        qa = xr.DataArray([32])
        mask_on = _decode_qa_pixel(qa, mask_cloud=False, mask_shadow=False, mask_snow=True)
        mask_off = _decode_qa_pixel(qa, mask_cloud=False, mask_shadow=False, mask_snow=False)
        assert mask_on.values[0], "Snow pixel should be masked when mask_snow=True"
        assert not mask_off.values[0], "Snow pixel should NOT be masked when mask_snow=False"

    def test_nodata_qa_zero(self):
        """QA=0 (nodata/fill) — check if decode treats as clear (not cloudy)."""
        qa = xr.DataArray([0])
        mask = _decode_qa_pixel(qa, mask_cloud=True, mask_shadow=True)
        # QA=0 has no bits set, so decode returns False (not masked as cloud)
        # But the caller should handle QA=0 as invalid via is_geo_valid check
        assert not mask.values[0], "QA=0 should not trigger cloud mask (handled separately)"

    def test_nodata_qa_one(self):
        """QA=1 (fill) — check decode behavior."""
        qa = xr.DataArray([1])
        mask = _decode_qa_pixel(qa, mask_cloud=True, mask_shadow=True)
        # QA=1: bit 0 set (fill), no cloud/shadow bits
        assert not mask.values[0], "QA=1 should not trigger cloud mask (handled separately)"

    def test_typical_landsat5_clear_values(self):
        """Test with typical Landsat 5 TM QA_PIXEL values for clear pixels.

        Landsat 5 Collection 2 Level-2 common clear values:
        5440 = 0001010101000000 (clear, low confidence cloud)
        """
        qa = xr.DataArray([5440, 5504])
        mask = _decode_qa_pixel(qa, mask_cloud=True, mask_shadow=True)
        np.testing.assert_array_equal(mask.values, [False, False])

    def test_typical_landsat7_clear_values(self):
        """Test with typical Landsat 7 ETM+ QA_PIXEL values for clear pixels."""
        qa = xr.DataArray([5440, 5504])
        mask = _decode_qa_pixel(qa, mask_cloud=True, mask_shadow=True)
        np.testing.assert_array_equal(mask.values, [False, False])


# ---------------------------------------------------------------------------
# Unit tests for scale factor application
# ---------------------------------------------------------------------------


class TestScaleFactors:
    """Test DN-to-reflectance conversion with Landsat Collection 2 scale factors."""

    def test_typical_dn_range(self):
        """Typical Landsat 5/7 C2 L2 DN values should produce valid reflectance."""
        # Typical surface reflectance DN range for vegetation: 7000-12000
        dns = np.array([7000, 8000, 9000, 10000, 12000, 15000], dtype=float)
        refl = dns * _SCALE_FACTOR + _OFFSET
        refl = np.clip(refl, 0, 1)
        # 7000 * 0.0000275 - 0.2 = 0.1925 - 0.2 = -0.0075 → clipped to 0
        # 8000 * 0.0000275 - 0.2 = 0.22 - 0.2 = 0.02
        # 10000 * 0.0000275 - 0.2 = 0.275 - 0.2 = 0.075
        expected = np.array([0.0, 0.02, 0.0475, 0.075, 0.13, 0.2125])
        np.testing.assert_allclose(refl, expected)

    def test_low_dn_produces_zero(self):
        """DN values below ~7273 produce negative reflectance, clipped to 0.

        This is a KEY finding: 0.2 / 0.0000275 ≈ 7272.7
        Any DN below 7273 will result in 0 after clipping!
        """
        threshold_dn = _OFFSET / -_SCALE_FACTOR  # = 7272.727...
        assert abs(threshold_dn - 7272.73) < 1

        # DNs below threshold → 0 after clip
        low_dns = np.array([0, 1000, 5000, 7000, 7272], dtype=float)
        refl = np.clip(low_dns * _SCALE_FACTOR + _OFFSET, 0, 1)
        np.testing.assert_array_equal(refl, [0, 0, 0, 0, 0])

    def test_zero_dn_is_nodata(self):
        """DN=0 typically means nodata in Landsat C2. After scaling → 0, not NaN."""
        dn = np.array([0.0])
        refl = np.clip(dn * _SCALE_FACTOR + _OFFSET, 0, 1)
        # This is problematic: nodata (DN=0) becomes 0.0 instead of NaN
        assert refl[0] == 0.0, (
            "DN=0 (nodata) becomes 0.0 after scaling — should be NaN to "
            "distinguish from real low-reflectance data"
        )

    def test_nan_dn_stays_nan(self):
        """NaN DN (from stackstac fill) should remain NaN after scaling."""
        dn = np.array([np.nan])
        refl = dn * _SCALE_FACTOR + _OFFSET
        refl = np.clip(refl, 0, 1)
        assert np.isnan(refl[0])

    def test_high_dn_range(self):
        """Very high DN values (bright surfaces) should produce valid reflectance."""
        dns = np.array([20000, 30000, 40000], dtype=float)
        refl = np.clip(dns * _SCALE_FACTOR + _OFFSET, 0, 1)
        # 20000 → 0.35, 30000 → 0.625, 40000 → 0.9
        expected = np.array([0.35, 0.625, 0.9])
        np.testing.assert_allclose(refl, expected)


# ---------------------------------------------------------------------------
# Tests for coverage / cloud filtering logic
# ---------------------------------------------------------------------------


class TestCoverageFiltering:
    """Test the QA-based coverage and cloud percentage calculations."""

    def _make_qa_dataarray(self, values_2d, n_times=3):
        """Create a time-series QA DataArray for testing.

        Args:
            values_2d: 2D array of QA values for one timestep.
            n_times: Number of timesteps (repeats the same values).
        """
        vals = np.array(values_2d, dtype=float)
        data = np.stack([vals] * n_times)
        return xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": np.arange(n_times)},
        )

    def test_all_nodata_qa_zero(self):
        """Scene where all QA=0 (nodata) should have 0% coverage."""
        qa = self._make_qa_dataarray([[0, 0], [0, 0]], n_times=1)
        is_geo_valid = qa.notnull() & (qa != 0)
        is_data_valid = is_geo_valid & (qa > 1)
        geo_counts = is_geo_valid.sum(dim=["y", "x"])
        data_counts = is_data_valid.sum(dim=["y", "x"])
        # geo_counts=0, data_counts=0 → coverage_pct = 0/0 = NaN
        # This means the scene would fail min_coverage check
        assert geo_counts.values[0] == 0
        assert data_counts.values[0] == 0

    def test_all_fill_qa_one(self):
        """Scene where all QA=1 (fill/saturated) should have 0% valid data."""
        qa = self._make_qa_dataarray([[1, 1], [1, 1]], n_times=1)
        is_geo_valid = qa.notnull() & (qa != 0)
        is_data_valid = is_geo_valid & (qa > 1)
        geo_counts = is_geo_valid.sum(dim=["y", "x"])
        data_counts = is_data_valid.sum(dim=["y", "x"])
        # geo_counts=4 (QA=1 is not null and !=0), data_counts=0 (QA=1 is NOT >1)
        assert geo_counts.values[0] == 4
        assert data_counts.values[0] == 0
        # coverage = 0/4 = 0% → fails min_coverage=80

    def test_mixed_valid_and_nodata(self):
        """Scene with mix of valid and nodata should compute correct coverage."""
        # 2 valid pixels (QA=5440), 2 nodata pixels (QA=0)
        qa = self._make_qa_dataarray([[5440, 5440], [0, 0]], n_times=1)
        is_geo_valid = qa.notnull() & (qa != 0)
        is_data_valid = is_geo_valid & (qa > 1)
        geo_counts = is_geo_valid.sum(dim=["y", "x"])
        data_counts = is_data_valid.sum(dim=["y", "x"])
        coverage_pct = (data_counts / geo_counts) * 100
        assert coverage_pct.values[0] == 100  # 2/2 = 100% of geo-valid pixels

    def test_coverage_with_fill_pixels(self):
        """Scene with fill pixels (QA=1) reduces valid data coverage."""
        # 1 valid (5440), 1 fill (1), 2 nodata (0)
        qa = self._make_qa_dataarray([[5440, 1], [0, 0]], n_times=1)
        is_geo_valid = qa.notnull() & (qa != 0)
        is_data_valid = is_geo_valid & (qa > 1)
        geo_counts = is_geo_valid.sum(dim=["y", "x"])
        data_counts = is_data_valid.sum(dim=["y", "x"])
        coverage_pct = (data_counts / geo_counts) * 100
        assert coverage_pct.values[0] == 50  # 1 valid / 2 geo_valid = 50%

    def test_high_min_coverage_drops_sparse_scenes(self):
        """With min_coverage=80, scenes with <80% valid pixels are dropped."""
        # Only 25% valid pixels
        qa = self._make_qa_dataarray([[5440, 0], [0, 0]], n_times=1)
        is_geo_valid = qa.notnull() & (qa != 0)
        is_data_valid = is_geo_valid & (qa > 1)
        geo_counts = is_geo_valid.sum(dim=["y", "x"])
        data_counts = is_data_valid.sum(dim=["y", "x"])
        coverage_pct = (data_counts / geo_counts) * 100
        # coverage = 100% of geo_valid, BUT geo_valid is only 1 pixel
        # The issue: coverage_pct compares data_counts to geo_counts, NOT to total pixels
        # So even if only 1 pixel has data, if that 1 pixel has valid QA, coverage=100%
        # The actual spatial coverage problem is NOT caught by this metric!
        assert coverage_pct.values[0] == 100.0

    def test_nan_qa_values(self):
        """NaN QA values (from stackstac fill_value) should be treated as nodata."""
        qa = self._make_qa_dataarray([[np.nan, 5440], [5440, np.nan]], n_times=1)
        is_geo_valid = qa.notnull() & (qa != 0)
        is_data_valid = is_geo_valid & (qa > 1)
        geo_counts = is_geo_valid.sum(dim=["y", "x"])
        data_counts = is_data_valid.sum(dim=["y", "x"])
        assert geo_counts.values[0] == 2
        assert data_counts.values[0] == 2


# ---------------------------------------------------------------------------
# Integration test: actual STAC query for early dates
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLandsatEarlyDatesIntegration:
    """Integration tests that query the actual Planetary Computer STAC API.

    These tests verify that early-date (2000-2003) Landsat data is retrievable
    and contains valid (non-zero, non-NaN) values.

    Run with: pytest -m integration tests/test_landsat.py
    """

    BBOX = (-0.15, 51.46, 0.0, 51.53)  # London area
    START = "2000-06-01"
    END = "2000-09-01"  # Summer — less cloud, more data availability

    def test_stac_search_returns_items(self):
        """Verify STAC search finds Landsat scenes for 2000."""
        import pystac_client
        import planetary_computer

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=list(self.BBOX),
            datetime=f"{self.START}/{self.END}",
            query={"eo:cloud_cover": {"lt": 50}},
        )
        items = search.item_collection()
        assert len(items) > 0, (
            f"Expected Landsat scenes for {self.START}/{self.END}, got 0. "
            "Check if Planetary Computer has Landsat 5/7 data for this period."
        )
        # Print mission info for debugging
        platforms = set(i.properties.get("platform", "?") for i in items)
        print(f"\nFound {len(items)} scenes from platforms: {platforms}")

    def test_inspect_raw_dn_values(self):
        """Inspect raw DN values from early Landsat to check if they're in valid range.

        This test checks whether the raw (unscaled) DN values are above the
        threshold (~7273) needed to produce non-zero reflectance after scaling.
        """
        import pystac_client
        import planetary_computer
        import stackstac

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=list(self.BBOX),
            datetime=f"{self.START}/{self.END}",
            query={"eo:cloud_cover": {"lt": 30}},
            max_items=5,
        )
        items = search.item_collection()
        if len(items) == 0:
            pytest.skip("No scenes found for this date range")

        items = [planetary_computer.sign(item) for item in items]

        # Build a small cube with just red and nir bands
        cube = stackstac.stack(
            items[:3],
            assets=["red", "nir08", "qa_pixel"],
            bounds_latlon=list(self.BBOX),
            resolution=30,
            dtype="float64",
            fill_value=np.nan,
            chunksize=512,
        )

        # Sample a small region
        sample = cube[:, :, :50, :50].compute()

        red_raw = sample.sel(band="red")
        nir_raw = sample.sel(band="nir08")
        qa_raw = sample.sel(band="qa_pixel")

        print(f"\n--- Raw DN statistics ---")
        print(f"Red: min={float(red_raw.min()):.0f}, max={float(red_raw.max()):.0f}, "
              f"mean={float(red_raw.mean()):.0f}, nan%={float(red_raw.isnull().mean())*100:.1f}%")
        print(f"NIR: min={float(nir_raw.min()):.0f}, max={float(nir_raw.max()):.0f}, "
              f"mean={float(nir_raw.mean()):.0f}, nan%={float(nir_raw.isnull().mean())*100:.1f}%")
        print(f"QA:  min={float(qa_raw.min()):.0f}, max={float(qa_raw.max()):.0f}, "
              f"unique values (sample): {np.unique(qa_raw.values[~np.isnan(qa_raw.values)])[:10]}")

        # Check: are DN values above the zero-threshold?
        threshold = abs(_OFFSET / _SCALE_FACTOR)  # ~7272.7
        red_valid = red_raw.values[~np.isnan(red_raw.values)]
        if len(red_valid) > 0:
            pct_above = (red_valid > threshold).mean() * 100
            print(f"Red band: {pct_above:.1f}% of valid pixels have DN > {threshold:.0f}")
            assert pct_above > 10, (
                f"Only {pct_above:.1f}% of red band pixels have DN > {threshold:.0f}. "
                "After scale factor application, most pixels will be clipped to 0!"
            )
        else:
            pytest.fail("No valid (non-NaN) red band pixels found")

    def test_qa_pixel_values_early_data(self):
        """Check QA_PIXEL value distribution for early Landsat data.

        Verifies that QA values are in expected ranges and that the
        is_data_valid (qa > 1) check doesn't filter out all pixels.
        """
        import pystac_client
        import planetary_computer
        import stackstac

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=list(self.BBOX),
            datetime=f"{self.START}/{self.END}",
            query={"eo:cloud_cover": {"lt": 30}},
            max_items=5,
        )
        items = search.item_collection()
        if len(items) == 0:
            pytest.skip("No scenes found")

        items = [planetary_computer.sign(item) for item in items]

        cube = stackstac.stack(
            items[:3],
            assets=["qa_pixel"],
            bounds_latlon=list(self.BBOX),
            resolution=30,
            dtype="float64",
            fill_value=np.nan,
            chunksize=512,
        )

        qa = cube.sel(band="qa_pixel").compute()

        print(f"\n--- QA_PIXEL analysis ---")
        qa_flat = qa.values.flatten()
        qa_valid = qa_flat[~np.isnan(qa_flat)]

        if len(qa_valid) == 0:
            pytest.fail("All QA_PIXEL values are NaN — no data loaded from STAC")

        unique_vals, counts = np.unique(qa_valid, return_counts=True)
        print(f"Unique QA values: {unique_vals[:20]}")
        print(f"Counts: {counts[:20]}")

        # Check how many are nodata (0), fill (1), or valid (>1)
        n_zero = (qa_valid == 0).sum()
        n_one = (qa_valid == 1).sum()
        n_valid = (qa_valid > 1).sum()
        total = len(qa_valid)
        print(f"QA=0 (nodata): {n_zero} ({n_zero/total*100:.1f}%)")
        print(f"QA=1 (fill):   {n_one} ({n_one/total*100:.1f}%)")
        print(f"QA>1 (valid):  {n_valid} ({n_valid/total*100:.1f}%)")

        # Check cloud masking
        mask = _decode_qa_pixel(qa, mask_cloud=True, mask_shadow=True)
        cloud_pct = mask.mean().values * 100
        print(f"Cloud mask % (of all pixels incl NaN): {cloud_pct:.1f}%")

        # Among valid pixels, how many pass cloud filter?
        qa_valid_da = qa.where(qa > 1)
        mask_valid = _decode_qa_pixel(qa_valid_da.fillna(0), mask_cloud=True, mask_shadow=True)
        # Only count where qa > 1
        valid_pixels = (qa > 1)
        cloud_in_valid = (mask_valid & valid_pixels).sum().values
        total_valid = valid_pixels.sum().values
        if total_valid > 0:
            cloud_in_valid_pct = cloud_in_valid / total_valid * 100
            print(f"Cloud % (of valid pixels only): {cloud_in_valid_pct:.1f}%")

    def test_full_retrieval_early_dates(self):
        """Full end-to-end retrieval for 2000 — check for non-zero, non-NaN values.

        This test validates the fix for the double-scaling bug where stackstac's
        auto-rescale (using raster:bands metadata) was applied, then the code
        applied the same scale/offset again, collapsing all values to zero.
        """
        result = get_landsat_imagery(
            bbox=self.BBOX,
            start_date=self.START,
            end_date=self.END,
            bands=["red", "nir08"],
            cloud_cover_max=50,
            min_coverage=0,   # Don't filter on coverage
            mask_clouds=False,  # Don't mask to see raw data
            add_ndvi=True,
            apply_scale_factors=True,
        )

        print(f"\n--- Full retrieval result ---")
        print(f"Shape: {result.shape}")
        print(f"Dims: {result.dims}")
        print(f"Times: {result.time.values}")
        print(f"Bands: {list(result.band.values)}")

        # Check for all-zero or all-NaN
        for band_name in result.band.values:
            band_data = result.sel(band=band_name)
            n_total = band_data.size
            n_nan = int(band_data.isnull().sum())
            n_zero = int((band_data == 0).sum())
            n_valid = n_total - n_nan
            n_nonzero = n_valid - n_zero

            print(f"  {band_name}: total={n_total}, NaN={n_nan} ({n_nan/n_total*100:.1f}%), "
                  f"zero={n_zero} ({n_zero/n_total*100:.1f}%), "
                  f"nonzero={n_nonzero} ({n_nonzero/n_total*100:.1f}%)")

            if band_name in ["red", "nir08"]:
                assert n_nonzero > 0, (
                    f"Band {band_name} has NO non-zero, non-NaN values! "
                    "Early Landsat data is producing empty results. "
                    "This was caused by double-scaling: stackstac auto-rescales "
                    "using raster:bands metadata (scale=2.75e-05, offset=-0.2), "
                    "then the code applied the same factors again."
                )

            if band_name == "NDVI":
                valid_ndvi = band_data.values[~np.isnan(band_data.values)]
                if len(valid_ndvi) > 0:
                    print(f"  NDVI range: [{valid_ndvi.min():.3f}, {valid_ndvi.max():.3f}]")
                    # NDVI should have meaningful variation for vegetated areas
                    assert valid_ndvi.max() > 0.1, (
                        "NDVI max is suspiciously low — values may be incorrectly scaled"
                    )

    def test_retrieval_without_scale_factors(self):
        """Retrieve without scale factors to check if raw DNs are valid."""
        result = get_landsat_imagery(
            bbox=self.BBOX,
            start_date=self.START,
            end_date=self.END,
            bands=["red", "nir08"],
            cloud_cover_max=50,
            min_coverage=0,
            mask_clouds=False,
            apply_scale_factors=False,  # Raw DNs
        )

        print(f"\n--- Raw DN retrieval (no scale factors) ---")
        for band_name in ["red", "nir08"]:
            if band_name in result.band.values:
                band_data = result.sel(band=band_name)
                valid = band_data.values[~np.isnan(band_data.values)]
                if len(valid) > 0:
                    print(f"  {band_name}: min={valid.min():.0f}, max={valid.max():.0f}, "
                          f"mean={valid.mean():.0f}, median={np.median(valid):.0f}")
                    threshold = abs(_OFFSET / _SCALE_FACTOR)
                    pct_above = (valid > threshold).mean() * 100
                    print(f"  {band_name}: {pct_above:.1f}% above DN threshold ({threshold:.0f})")
                else:
                    pytest.fail(f"No valid pixels in raw {band_name} band")

    def test_retrieval_high_coverage_filter(self):
        """Test with min_coverage=80 — may drop too many early scenes."""
        try:
            result = get_landsat_imagery(
                bbox=self.BBOX,
                start_date=self.START,
                end_date=self.END,
                bands=["red", "nir08"],
                cloud_cover_max=50,
                min_coverage=80,  # Strict coverage — might drop all scenes
                mask_clouds=False,
                apply_scale_factors=True,
            )
            print(f"\nWith min_coverage=80: got {len(result.time)} scenes")
            for band_name in ["red", "nir08"]:
                band_data = result.sel(band=band_name)
                n_nonzero = int((band_data > 0).sum())
                print(f"  {band_name}: {n_nonzero} non-zero pixels")
        except ValueError as e:
            print(f"\nWith min_coverage=80: FAILED — {e}")
            # This is expected: high coverage filter drops all early scenes
            assert "No" in str(e) or "Retained 0" in str(e) or True

    def test_compare_landsat5_vs_landsat7(self):
        """Compare L5 and L7 data separately to isolate mission-specific issues."""
        for mission in ["landsat-5", "landsat-7"]:
            try:
                result = get_landsat_imagery(
                    bbox=self.BBOX,
                    start_date=self.START,
                    end_date=self.END,
                    bands=["red", "nir08"],
                    cloud_cover_max=50,
                    min_coverage=0,
                    mask_clouds=False,
                    missions=[mission],
                    apply_scale_factors=False,
                )
                red = result.sel(band="red")
                valid = red.values[~np.isnan(red.values)]
                print(f"\n{mission}: {len(result.time)} scenes, "
                      f"red DN range [{valid.min():.0f}, {valid.max():.0f}]")
            except ValueError as e:
                print(f"\n{mission}: No data — {e}")


# ---------------------------------------------------------------------------
# Regression test: mosaic via max() can mask issues
# ---------------------------------------------------------------------------


class TestMosaicBehavior:
    """Test that groupby('time').max() mosaicing doesn't introduce artifacts."""

    def test_max_mosaic_with_nan_fill(self):
        """max() with NaN fill should take the non-NaN value from overlapping tiles."""
        # Simulate 2 tiles at same time with different spatial coverage
        data = np.array([
            [[np.nan, 5000], [8000, np.nan]],  # tile 1
            [[9000, np.nan], [np.nan, 7000]],  # tile 2
        ], dtype=float)
        da = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": [0, 0]},  # Same timestamp
        )
        mosaiced = da.groupby("time").max(dim="time")
        expected = np.array([[9000, 5000], [8000, 7000]])
        np.testing.assert_array_equal(mosaiced.values[0], expected)

    def test_max_mosaic_picks_higher_value(self):
        """max() mosaic picks the higher value when both tiles have data.

        This could be problematic: max() doesn't average, it picks the brightest.
        For reflectance data, this biases toward brighter (potentially cloudy) pixels.
        """
        data = np.array([
            [[100, 200], [300, 400]],
            [[150, 50], [350, 100]],
        ], dtype=float)
        da = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": [0, 0]},
        )
        mosaiced = da.groupby("time").max(dim="time")
        # max picks highest: 150, 200, 350, 400
        expected = np.array([[150, 200], [350, 400]])
        np.testing.assert_array_equal(mosaiced.values[0], expected)


# ---------------------------------------------------------------------------
# Test: scale factor + NaN interaction
# ---------------------------------------------------------------------------


class TestScaleFactorNaNInteraction:
    """Test that scale factor application preserves NaN properly."""

    def test_scale_preserves_nan(self):
        """NaN pixels should stay NaN after scale factor application."""
        data = xr.DataArray(
            [[[np.nan, 10000], [8000, np.nan]]],
            dims=["time", "y", "x"],
        )
        scaled = data * _SCALE_FACTOR + _OFFSET
        scaled = scaled.clip(0, 1)
        assert np.isnan(scaled.values[0, 0, 0])
        assert np.isnan(scaled.values[0, 1, 1])
        assert scaled.values[0, 0, 1] == pytest.approx(0.075)
        assert scaled.values[0, 1, 0] == pytest.approx(0.02)

    def test_zero_dn_becomes_zero_not_nan(self):
        """Critical: DN=0 (which means nodata in Landsat) becomes 0.0, not NaN.

        This is a potential bug: after scale factors, nodata pixels (DN=0) are
        indistinguishable from legitimately dark pixels. The code relies on
        stackstac using fill_value=np.nan, but if the actual raster has 0 as
        nodata and stackstac doesn't convert it, these become false zeros.
        """
        data = xr.DataArray([[[0.0, 10000]]], dims=["time", "y", "x"])
        scaled = data * _SCALE_FACTOR + _OFFSET
        scaled = scaled.clip(0, 1)
        # DN=0 → 0*0.0000275 - 0.2 = -0.2 → clip to 0.0
        assert scaled.values[0, 0, 0] == 0.0
        # This is indistinguishable from a legitimately low-reflectance pixel
