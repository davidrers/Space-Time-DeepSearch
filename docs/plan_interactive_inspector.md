# Interactive LandTrendr Visualization Module — Implementation Plan

## Context

The project has a native Python LandTrendr implementation with static matplotlib visualization. This plan adds an interactive inspector (similar to geeViz's inspector tool) that:
- Displays YOD, Magnitude, and Duration maps with **layer toggle on/off**
- **Clicking any pixel** on any map shows the LandTrendr trajectory (source dots + fitted line + vertices), matching the style in `docs/figures/fig1_trajectories.png`
- Works in Jupyter notebooks

**Library choice: HoloViews + hvPlot + Panel** — best xarray integration, native layer toggling, scalable to large rasters via datashader.

## Map Size / Spatial Extent

The maps' spatial dimensions come from the xarray Dataset passed to the inspector. This is determined upstream:
- `SpaceTimeDeepSearch(bbox=..., custom_geometry=...)` defines the AOI
- Satellite retrieval + `run_landtrendr()` produces `(time, y, x)` data
- `extract_change_map()` reduces to `(y, x)` — these are the map dimensions
- No separate "map size" input is needed; it's inherited from the data

## Implementation

### 1. Dependencies added to `pyproject.toml`

```
"holoviews (>=1.19.0,<2.0.0)",
"hvplot (>=0.11.0,<1.0.0)",
"panel (>=1.5.0,<2.0.0)",
"bokeh (>=3.5.0,<4.0.0)",
```

### 2. New module: `src/space_time_deepsearch/temporal/_landtrendr_interactive.py`

Fully isolated from existing `_landtrendr_viz.py` (matplotlib-based static plots).

**Layout:**

```
┌─────────────────────────────────────────────┐
│  [✓] YOD   [✓] Magnitude   [✓] Duration    │  ← CheckBoxGroup widget
├──────────┬──────────┬───────────────────────┤
│  YOD map │ Mag map  │  Duration map         │  ← hvplot.image() panels
│          │          │                       │
│          │          │                       │
├──────────┴──────────┴───────────────────────┤
│  Trajectory: Source ● + Fitted ── + Vtx ▲   │  ← DynamicMap (reactive)
│  + YOD vertical line + RMSE/mag annotation  │
└─────────────────────────────────────────────┘
```

**Key mechanics:**
- Maps rendered with `hvplot.image()` using native xarray coords
- Click via `hv.streams.SingleTap` on each map
- Trajectory via `hv.DynamicMap` (reactive to clicks)
- Layer toggling via `pn.widgets.CheckBoxGroup`
- Crosshair marker via `hv.Points` overlay

### 3. Integration points

- `temporal/__init__.py` — exports `LandTrendrInspector`, `inspect_landtrendr`
- `core.py` — `SpaceTimeDeepSearch.inspect_landtrendr()` convenience method

### 4. Tests: `tests/test_landtrendr_interactive.py`

## Usage

```python
from space_time_deepsearch.temporal import run_landtrendr, extract_change_map, inspect_landtrendr

lt = run_landtrendr(ndvi_annual)
change = extract_change_map(lt, change_type="greatest", delta_filter="loss")
dashboard = inspect_landtrendr(lt, change)
dashboard.show()
```
