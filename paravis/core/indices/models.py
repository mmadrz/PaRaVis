"""
Data models for spectral index definitions and band mapping.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SpectralIndex:
    """Definition of a spectral index."""
    name: str
    formula: str
    bands: List[str]
    constants: List[str] = field(default_factory=list)
    reference: str = ""
    long_name: str = ""
    platform: str = "generic"
    computable: bool = False


@dataclass
class BandMapping:
    """Mapping from band number to spectral code (e.g. {1: 'A', 2: 'B', ...})."""
    mapping: Dict[int, str] = field(default_factory=dict)
    name: str = "Custom"

    @classmethod
    def landsat_8_9(cls) -> "BandMapping":
        return cls(
            mapping={1: "A", 2: "B", 3: "G", 4: "R", 5: "N", 6: "S1", 7: "S2", 8: "T"},
            name="Landsat 8/9",
        )

    @classmethod
    def sentinel2(cls) -> "BandMapping":
        return cls(
            mapping={1: "A", 2: "B", 3: "G", 4: "R", 5: "RE1", 6: "RE2",
                     7: "RE3", 8: "N2", 9: "WV", 10: "S1", 11: "S2"},
            name="Sentinel-2",
        )

    @classmethod
    def sentinel1(cls) -> "BandMapping":
        return cls(
            mapping={1: "VV", 2: "VH"},
            name="Sentinel-1",
        )

    def get_code(self, band_num: int) -> str:
        return self.mapping.get(band_num, "")

    def to_dict(self) -> Dict[int, str]:
        return dict(self.mapping)
