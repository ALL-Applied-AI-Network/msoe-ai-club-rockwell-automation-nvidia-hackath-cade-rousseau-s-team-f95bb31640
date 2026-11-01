import cv2
import numpy as np
import pandas as pd

from src.color_monitor import overlay


def test_paired_overlay_writes_empty_and_filled_measurement_frames(
    monkeypatch, tmp_path
):
    empty = np.full((480, 640, 3), 180, dtype=np.uint8)
    filled = empty.copy()
    filled[375:397, 295:327] = (20, 80, 180)

    def fake_grab(_video_id, t_s):
        return empty.copy() if t_s == 10.0 else filled.copy()

    monkeypatch.setattr(overlay, "_grab", fake_grab)
    vials = pd.DataFrame(
        {
            "video_id": [5],
            "vial_id": [7],
            "recipe_id": [3],
            "empty_t_measure": [10.0],
            "stable_t_measure": [20.0],
        }
    )
    output = tmp_path / "paired.png"

    result = overlay.save_paired_measurement_overlays(vials, output)

    assert result == output
    contact = cv2.imread(str(output))
    assert contact is not None
    assert contact.shape[1] > contact.shape[0]
    assert int(contact.max()) == 255
