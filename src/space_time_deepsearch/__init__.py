
import os
import sys

# Check for PROJ_LIB conflict (e.g., PostGIS installation on Windows)
# The error typically manifests as a version mismatch in proj.db
if sys.platform == "win32" and "PROJ_LIB" in os.environ:
    proj_lib = os.environ["PROJ_LIB"]
    # Common conflict is with PostgreSQL/PostGIS
    if "PostgreSQL" in proj_lib and "proj" in proj_lib.lower():
        # Unset PROJ_LIB to allow rasterio/pyproj to use their own bundled PROJ data
        del os.environ["PROJ_LIB"]

from .vis.animation import create_timelapse  # noqa: E402

__all__ = ["create_timelapse"]