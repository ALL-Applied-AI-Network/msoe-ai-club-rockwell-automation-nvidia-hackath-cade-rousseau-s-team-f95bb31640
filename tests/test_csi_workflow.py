import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from changeover_guard.diagnostic_baselines import OCT29_TANK_ROIS
from changeover_guard.diagnostic_workflow import run_operator_workflow


def _software_inputs(root):
    """Controlled I/O fixtures; never passed off as real camera evaluation."""
    camera = root / "data/csi-vial-fill-dataset/vial_fill"
    labels = camera / "labels"
    labels.mkdir(parents=True)
    (camera / "videos").mkdir()
    pd.DataFrame(
        [
            {
                "vial": i + 1,
                "recipe_id": 1,
                "t100_ml": 50,
                "t200_ml": 0,
                "t400_ml": 0,
                "total_ml": 50,
                "dispense_order": "T100",
                "start_s": i * 6.0,
                "end_s": (i + 1) * 6.0,
                "status": "ok",
            }
            for i in range(3)
        ]
    ).to_csv(labels / "video5_vial_recipes.csv", index=False)
    video = camera / "videos/vial_fill_video_5.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 480)
    )
    if not writer.isOpened():
        pytest.skip("mp4v codec unavailable")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[375:397, 295:327] = [45, 70, 100]
    for _ in range(180):
        writer.write(frame)
    writer.release()
    photo = root / "data/10_29_26/top_view/images/tank_top_level_10_29_26.jpg"
    photo.parent.mkdir(parents=True)
    tank_image = np.full((700, 920, 3), 120, dtype=np.uint8)
    for rgb, (x1, y1, x2, y2) in zip(
        ([170, 160, 150], [160, 45, 25], [25, 45, 160]), OCT29_TANK_ROIS.values()
    ):
        tank_image[
            round(y1 * 700) : round(y2 * 700), round(x1 * 920) : round(x2 * 920)
        ] = rgb[::-1]
    assert cv2.imwrite(str(photo), tank_image)
    # Invalid October 29 production data deliberately exist: default must never read them.
    old_labels = root / "data/10_29_26/vial_filled/labels"
    old_labels.mkdir(parents=True)
    (old_labels / "vial_recipes.csv").write_text("this is not a valid dataset\n")
    return camera, photo


def test_default_uses_photo_and_csi_exclusively_even_when_october_vials_exist(tmp_path):
    _camera, photo = _software_inputs(tmp_path)
    result = run_operator_workflow(tmp_path, plots=False)
    assert result["tank"]["status"] == "measured"
    assert Path(result["tank"]["path"]) == photo
    measured = result["measurement_result"]["measurements"]
    assert len(measured) == 3
    assert measured.session_id.eq("csi/video5").all()
    assert measured.usable.all()
    assert len(result["operator_report"]) == 3
    assert result["summary"]["october29_vials_used_for_training_or_testing"] is False
    assert result["summary"]["prefill_subtraction_applied"] is False
    assert (
        result["summary"]["training_and_test_dataset"] == "data/csi-vial-fill-dataset"
    )
    assert result["benchmark"]["selection"]["status"] == "insufficient_data"
    assert (result["output_dir"] / "training_folds.json").is_file()
    assert (result["output_dir"] / "training_metrics.csv").is_file()
    assert (result["output_dir"] / "csi_trained_model.json").is_file()
    assert result["operator_report"].production_status.eq("PENDING").all()
    folds = json.loads((result["output_dir"] / "training_folds.json").read_text())
    assert all(
        key.startswith("csi/") for fold in folds for key in fold["test_vial_keys"]
    )


def test_exact_user_labels_path_is_an_accepted_dataset_root(tmp_path):
    camera, _ = _software_inputs(tmp_path)
    result = run_operator_workflow(
        tmp_path, dataset_root=camera / "labels", plots=False
    )
    assert len(result["measurement_result"]["measurements"]) == 3
    assert result["summary"]["training_and_test_dataset"].endswith("vial_fill/labels")


def test_missing_csi_never_falls_back_to_actual_october_collection(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = run_operator_workflow(
        root, dataset_root=tmp_path / "missing-csi", plots=False
    )
    assert result["summary"]["status"] == "missing_data"
    assert result["operator_report"].empty
    assert result["benchmark"]["folds"] == []
    assert result["summary"]["october29_vials_used_for_training_or_testing"] is False
    assert any("missing-csi" in value for value in result["summary"]["missing_inputs"])


def test_csi_notebook_path_has_a_real_filled_frame_gallery(tmp_path):
    pytest.importorskip("matplotlib")
    _software_inputs(tmp_path)
    result = run_operator_workflow(tmp_path, plots=True)
    assert (result["output_dir"] / "csi_vial_roi_01.png").stat().st_size > 1000
    assert (result["output_dir"] / "tank_roi_verification.png").stat().st_size > 1000
    assert (result["output_dir"] / "operator_dashboard.html").is_file()


def test_labels_without_videos_do_not_report_successful_measurement(tmp_path):
    camera, _ = _software_inputs(tmp_path)
    (camera / "videos/vial_fill_video_5.mp4").unlink()
    result = run_operator_workflow(tmp_path, plots=False)
    assert result["summary"]["status"] == "missing_data"
    assert result["summary"]["labeled_vials"] == 3
    assert (
        result["operator_report"].diagnostic_state.eq("measurement_sensor_reject").all()
    )
