"""
Example: Batch process multiple GeoTIFFs with Rao's Q in a loop.

Demonstrates headless/script usage for HPC or batch pipelines.
"""
import os
import numpy as np
from paravis.api import compute_rao_q


def batch_process(input_dir: str, output_dir: str, window_size: int = 15):
    """Compute Rao's Q for all .tif files in input_dir."""
    os.makedirs(output_dir, exist_ok=True)

    tif_files = [f for f in os.listdir(input_dir) if f.endswith((".tif", ".tiff"))]
    print(f"Found {len(tif_files)} raster files in {input_dir}")

    for filename in tif_files:
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, f"raoq_{filename}")

        print(f"Processing {filename} ...")
        result = compute_rao_q(in_path, window_size=window_size, backend="cpu")

        # Save as GeoTIFF
        from paravis.core.raster.writer import write_geotiff
        write_geotiff(result, out_path)
        print(f"  → Saved to {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python batch_processing.py <input_dir> <output_dir> [window_size]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    ws = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    batch_process(input_dir, output_dir, ws)
