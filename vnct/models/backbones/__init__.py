"""Vision backbones."""

from typing import Any

from vnct.models.backbones.vnct import VNCTBackbone, vnct_debug, vnct_tiny
from vnct.models.backbones.vssd import VSSDBackbone, vssd_debug, vssd_tiny


def vssd_small_ncm3(**kwargs: Any):
    """Lazily build the checkpoint-compatible camera-ready Small model."""
    from vnct.models.backbones.vssd_small_ncm3 import vssd_small_ncm3 as build

    return build(**kwargs)


def vssd_small_original(**kwargs: Any):
    """Lazily build the unchanged camera-ready VSSD-Small model."""
    from vnct.models.backbones.vssd_small_ncm3 import vssd_small_original as build

    return build(**kwargs)

__all__ = [
    "VNCTBackbone",
    "VSSDBackbone",
    "vnct_debug",
    "vnct_tiny",
    "vssd_debug",
    "vssd_small_ncm3",
    "vssd_small_original",
    "vssd_tiny",
]
