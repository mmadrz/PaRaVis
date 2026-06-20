"""
Example: Register and use a custom spectral index.

PaRaVis allows you to define your own indices via the
@register_index decorator.
"""
import numpy as np
from paravis.core.indices.registry import register_index, compute_custom_index


# Register a custom index — a simple ratio
@register_index(name="SR", bands=["N", "R"], description="Simple Ratio NIR/RED")
def simple_ratio(nir, red):
    """Simple Ratio: NIR / RED"""
    return nir / (red + 1e-10)


# Register a modified NDVI with a custom offset
@register_index(name="NDVI_OFFSET", bands=["N", "R"], description="NDVI with offset")
def ndvi_offset(nir, red):
    """NDVI plus a constant offset."""
    return (nir - red) / (nir + red + 1e-10) + 0.1


def main():
    # Simulate some band data
    nir = np.random.rand(100, 100).astype(np.float32) * 0.5 + 0.3
    red = np.random.rand(100, 100).astype(np.float32) * 0.3 + 0.1

    band_arrays = {"N": nir, "R": red}

    # Compute the custom indices
    sr = compute_custom_index("SR", band_arrays)
    ndvi_off = compute_custom_index("NDVI_OFFSET", band_arrays)

    print(f"SR shape: {sr.shape}, mean: {sr.mean():.4f}")
    print(f"NDVI_OFFSET shape: {ndvi_off.shape}, mean: {ndvi_off.mean():.4f}")


if __name__ == "__main__":
    main()
