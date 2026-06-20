"""
Example: Compute spectral indices from a GeoTIFF without any GUI.

Usage:
    python examples/headless_indices.py path/to/scene.tif

This demonstrates the pure-computation API — no Qt required.
"""
import sys
import numpy as np
from paravis.api import compute_indices, list_available_indices


def main():
    if len(sys.argv) < 2:
        print("Usage: python headless_indices.py <raster_path>")
        print("\nAvailable indices:")
        for name in list_available_indices()[:10]:
            print(f"  • {name}")
        print("  … and more")
        sys.exit(1)

    raster_path = sys.argv[1]
    indices = ["NDVI", "EVI", "SAVI", "NDWI"]

    print(f"Reading {raster_path} ...")
    results = compute_indices(raster_path, indices=indices)

    for name, data in results.items():
        print(f"  {name}: shape={data.shape}, "
              f"min={np.nanmin(data):.4f}, max={np.nanmax(data):.4f}")


if __name__ == "__main__":
    main()
