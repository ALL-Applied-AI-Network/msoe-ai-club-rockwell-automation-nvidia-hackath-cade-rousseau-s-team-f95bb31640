import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "twin"))
from lut import diffuse_for


def test_diffuse_for_inverts_each_channel():
    levels = [0.0, 0.25, 0.5, 1.0]
    box = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90], [130, 140, 150]])
    lut = {"diffuse": levels, "box_rgb": box.tolist()}
    assert diffuse_for(lut, [40, 80, 150]) == [0.25, 0.5, 1.0]
    assert diffuse_for(lut, [25, 20, 120]) == [0.125, 0.0, 0.75]
    assert diffuse_for(lut, [0, 255, 30]) == [0.0, 1.0, 0.0]      # out of range clips to the ends
