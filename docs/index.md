# Space-Time DeepSearch

A Python geospatial library for retrieving and analyzing satellite imagery with temporal segmentation.

## Features

- **Satellite imagery retrieval** from Microsoft Planetary Computer (Sentinel-2, Landsat, MODIS)
- **LandTrendr temporal segmentation** (Kennedy et al. 2010) with Dask parallelization
- **Interactive inspection** of LandTrendr results with HoloViews/Panel
- **Timelapse animation** generation from multi-temporal data
- **OpenStreetMap** feature and street network retrieval
- **WorldPop** population density rasters

## Installation

```bash
pip install space-time-deepsearch
```

## Quick Start

```python
from space_time_deepsearch import SpaceTimeDeepSearch

# Initialize with a bounding box (west, south, east, north)
stds = SpaceTimeDeepSearch(bbox=(5.0, 52.0, 6.0, 53.0))

# Retrieve Sentinel-2 imagery
s2 = stds.get_sentinel2(start_date="2023-01-01", end_date="2023-12-31")

# Retrieve Landsat imagery
ls = stds.get_landsat(start_date="2000-01-01", end_date="2023-12-31")

# Run LandTrendr temporal segmentation
result = stds.run_landtrendr(ls, spectral_index="NDVI")
```
