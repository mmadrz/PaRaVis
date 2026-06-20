"""
Constants management for spectral index computation.

Handles default values from spyndex and user overrides.
"""
from typing import Dict, Optional

import spyndex


def get_default_constants() -> Dict[str, float]:
    """Return all default constant values from spyndex.

    Returns
    -------
    Dict[str, float]
        Mapping from constant name to default value.
    """
    constants: Dict[str, float] = {}
    for name, const in spyndex.constants.items():
        try:
            constants[name] = float(getattr(const, "default", 0.0))
        except (ValueError, TypeError):
            constants[name] = 0.0
    return constants


def merge_constants(
    defaults: Dict[str, float],
    overrides: Dict[str, Optional[float]],
) -> Dict[str, float]:
    """Merge user overrides into default constants.

    Parameters
    ----------
    defaults : Dict[str, float]
        Default constant values.
    overrides : Dict[str, Optional[float]]
        User overrides. A value of None removes the override (fallback to default).

    Returns
    -------
    Dict[str, float]
        Merged constants dictionary.
    """
    merged = dict(defaults)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
        elif key in merged:
            del merged[key]
    return merged
