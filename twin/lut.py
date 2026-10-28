"""Invert twin/material_lut.json: the liquid diffuse color that renders a target pixel color.

numpy only, so it runs in uv and in Isaac Sim's Python.
"""
import numpy as np


def diffuse_for(lut: dict, rgb: list[float]) -> list[float]:
    """Per channel, the diffuse level whose gray swatch rendered rgb's value in the measurement box.

    lut: {"diffuse": [levels], "box_rgb": [[r, g, b] per level]} as written by render.py --calibrate.
    Targets outside the rendered range clip to the darkest or brightest swatch.
    """
    levels, box = np.array(lut["diffuse"]), np.array(lut["box_rgb"])
    return [float(np.interp(rgb[c], np.maximum.accumulate(box[:, c]), levels)) for c in range(3)]
