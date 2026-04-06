"""
Diagnostic test to identify the double-scaling issue in Landsat retrieval.

Tests whether stackstac auto-rescales data (rescale=True by default) and
whether the code then applies scale factors a second time.
"""

import numpy as np
import pytest
import pystac_client
import planetary_computer
import stackstac


BBOX = (-0.15, 51.46, 0.0, 51.53)
START = "2000-06-01"
END = "2000-09-01"


def _get_items(max_items=3):
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=list(BBOX),
        datetime=f"{START}/{END}",
        query={"eo:cloud_cover": {"lt": 50}},
        max_items=max_items,
    )
    items = search.item_collection()
    return [planetary_computer.sign(item) for item in items]


@pytest.mark.integration
class TestDoubleScaling:

    def test_check_stac_item_metadata(self):
        """Check if STAC items have raster:bands scale/offset metadata."""
        items = _get_items(1)
        assert len(items) > 0
        item = items[0]

        print(f"\nItem: {item.id}")
        print(f"Platform: {item.properties.get('platform')}")
        print(f"Datetime: {item.datetime}")

        for asset_key in ["red", "nir08", "qa_pixel"]:
            asset = item.assets.get(asset_key)
            if asset is None:
                print(f"  {asset_key}: NOT FOUND in assets")
                continue
            print(f"\n  Asset: {asset_key}")
            print(f"    href: {asset.href[:80]}...")

            # Check for raster:bands metadata (used by stackstac for rescaling)
            raster_bands = asset.extra_fields.get("raster:bands", [])
            if raster_bands:
                for rb in raster_bands:
                    scale = rb.get("scale")
                    offset = rb.get("offset")
                    nodata = rb.get("nodata")
                    print(f"    raster:bands scale={scale}, offset={offset}, nodata={nodata}")
            else:
                print(f"    raster:bands: NONE")

            # Check eo:bands
            eo_bands = asset.extra_fields.get("eo:bands", [])
            if eo_bands:
                for eb in eo_bands:
                    print(f"    eo:bands: {eb}")

    def test_stackstac_rescale_true_vs_false(self):
        """Compare stackstac output with rescale=True (default) vs rescale=False.

        If rescale=True produces values in [0, 1] range (reflectance), then
        the code should NOT apply scale factors again.
        If rescale=False produces raw DN values (7000-15000), then the code
        SHOULD apply scale factors.
        """
        from pyproj.aoi import AreaOfInterest
        from pyproj.database import query_utm_crs_info

        items = _get_items(2)
        assert len(items) > 0

        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=BBOX[0],
                south_lat_degree=BBOX[1],
                east_lon_degree=BBOX[2],
                north_lat_degree=BBOX[3],
            ),
        )
        target_epsg = int(utm_crs_list[0].code)

        common_kwargs = dict(
            assets=["red", "nir08"],
            bounds_latlon=list(BBOX),
            resolution=30,
            epsg=target_epsg,
            dtype="float64",
            fill_value=np.nan,
            chunksize=512,
        )

        # With rescale=True (default)
        cube_rescaled = stackstac.stack(items, rescale=True, **common_kwargs)
        sample_rescaled = cube_rescaled[:, :, :30, :30].compute()

        # With rescale=False
        cube_raw = stackstac.stack(items, rescale=False, **common_kwargs)
        sample_raw = cube_raw[:, :, :30, :30].compute()

        for band in ["red", "nir08"]:
            rescaled = sample_rescaled.sel(band=band).values
            raw = sample_raw.sel(band=band).values

            rescaled_valid = rescaled[~np.isnan(rescaled)]
            raw_valid = raw[~np.isnan(raw)]

            print(f"\n--- Band: {band} ---")
            if len(rescaled_valid) > 0:
                print(f"  rescale=True:  min={rescaled_valid.min():.6f}, "
                      f"max={rescaled_valid.max():.6f}, "
                      f"mean={rescaled_valid.mean():.6f}")
            else:
                print(f"  rescale=True:  ALL NaN ({rescaled.size} pixels)")

            if len(raw_valid) > 0:
                print(f"  rescale=False: min={raw_valid.min():.0f}, "
                      f"max={raw_valid.max():.0f}, "
                      f"mean={raw_valid.mean():.0f}")
            else:
                print(f"  rescale=False: ALL NaN ({raw.size} pixels)")

            # If rescaled values are in [0, 1], that's reflectance — double scaling!
            if len(rescaled_valid) > 0 and rescaled_valid.max() < 2.0:
                print(f"  >>> rescale=True produces reflectance-range values!")
                print(f"  >>> CODE SHOULD NOT apply scale factors again (double-scaling bug)")

            # If raw values are in [0, 65535] or [7000, 15000], those are DNs
            if len(raw_valid) > 0 and raw_valid.max() > 100:
                print(f"  >>> rescale=False produces raw DN values")
                print(f"  >>> Code should apply scale factors to these")

    def test_stackstac_default_rescale(self):
        """Test stackstac's DEFAULT behavior (no explicit rescale parameter).

        The main code doesn't pass rescale= at all, so we need to know the default.
        """
        from pyproj.aoi import AreaOfInterest
        from pyproj.database import query_utm_crs_info

        items = _get_items(2)
        assert len(items) > 0

        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=BBOX[0],
                south_lat_degree=BBOX[1],
                east_lon_degree=BBOX[2],
                north_lat_degree=BBOX[3],
            ),
        )
        target_epsg = int(utm_crs_list[0].code)

        # Default call (same as main code, no rescale param)
        cube = stackstac.stack(
            items,
            assets=["red", "nir08"],
            bounds_latlon=list(BBOX),
            resolution=30,
            epsg=target_epsg,
            dtype="float64",
            fill_value=np.nan,
            chunksize=512,
        )
        sample = cube[:, :, :30, :30].compute()

        for band in ["red", "nir08"]:
            data = sample.sel(band=band).values
            valid = data[~np.isnan(data)]
            print(f"\n{band} (default stackstac): ", end="")
            if len(valid) > 0:
                print(f"min={valid.min():.6f}, max={valid.max():.6f}, mean={valid.mean():.6f}")
                if valid.max() < 2.0:
                    print(f"  >>> Values are in reflectance range — stackstac IS auto-rescaling!")
                elif valid.max() > 100:
                    print(f"  >>> Values are in raw DN range — stackstac is NOT rescaling")
            else:
                print(f"ALL NaN")
