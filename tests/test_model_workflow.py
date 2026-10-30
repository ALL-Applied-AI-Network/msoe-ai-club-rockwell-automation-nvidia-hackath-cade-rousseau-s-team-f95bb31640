import hashlib
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from changeover_guard.model_workflow import (
    default_config,
    make_contact_sheets,
    plot_recovery,
    run_workflow,
)
from changeover_guard.transition_analysis import (
    measure_timestamped_stills,
    validate_vials,
)
from changeover_guard.vial_color import VialCalibration


def test_missing_data_emits_honest_empty_artifacts(tmp_path):
    config = default_config(tmp_path)
    result = run_workflow(config, tmp_path, plots=False)
    output = result["output_dir"]
    assert result["report"]["status"] == "missing_data"
    assert result["measurements"].empty
    assert not result["repeatability"]["proven"]
    assert not result["report"]["production_decisions_enabled"]
    assert (
        json.loads((output / "clean_optical_model.json").read_text())["model"] is None
    )
    assert (
        json.loads((output / "carryover_model_parameters.json").read_text())["tau_ml"]
        is None
    )
    assert pd.read_csv(output / "vial_color_measurements.csv").empty
    next_run = run_workflow(config, tmp_path, plots=False)
    assert next_run["output_dir"] != output


def test_real_video_decode_path_with_controlled_software_fixture(tmp_path):
    # This small generated video tests decode/timing/ROI/export integration.
    # It is never used as a substitute for a measured real-data evaluation.
    video = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480)
    )
    if not writer.isOpened():
        pytest.skip("MJPG test codec unavailable")
    for frame in range(60):
        color = (60, 100, 140) if frame < 30 else (80, 100, 120)
        writer.write(np.full((480, 640, 3), color, dtype=np.uint8))
    writer.release()
    recipes = tmp_path / "vial_recipes.csv"
    pd.DataFrame(
        [
            {
                "vial": i + 1,
                "recipe_id": 1 if i < 3 else 2,
                "t100_ml": 40 if i < 3 else 20,
                "t200_ml": 0 if i < 3 else 20,
                "t400_ml": 0,
                "start_s": i,
                "end_s": i + 1,
            }
            for i in range(6)
        ]
    ).to_csv(recipes, index=False)
    config = default_config(tmp_path)
    config["sessions"] = [
        {
            "session_id": "software_fixture",
            "recipe_csv": str(recipes),
            "video": str(video),
        }
    ]
    result = run_workflow(config, tmp_path, plots=False)
    assert result["report"]["status"] == "measured"
    assert result["measurements"].usable.all()
    assert len(result["measurements"]) == 6 and len(result["summary"]) == 1
    assert result["summary"].right_censored.all()
    assert not result["support"]["shared_decay_supported"]
    assert result["measurements"].production_status.eq("PENDING").all()
    make_contact_sheets(result["measurements"], result["output_dir"])
    assert list(result["output_dir"].glob("alignment_roi*.png"))
    # Plots are an optional analysis dependency; decoding is tested regardless.
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return
    plot_recovery(result["curves"], result["summary"], result["output_dir"])
    assert list((result["output_dir"] / "recovery_plots").glob("*.png"))


def test_labels_without_video_still_report_the_transition_inventory(tmp_path):
    recipes = tmp_path / "vial_recipes.csv"
    recipes.write_text(
        "vial,recipe_id,t100_ml,t200_ml,t400_ml,start_s,end_s\n"
        "1,1,40,0,0,0,1\n2,2,20,20,0,2,3\n"
    )
    config = default_config(tmp_path)
    config["sessions"] = [
        {"session_id": "labels_only", "recipe_csv": str(recipes), "video": "absent.mp4"}
    ]
    result = run_workflow(config, tmp_path, plots=False)
    assert result["report"]["label_inventory"]["transitions"] == 1
    assert result["report"]["status"] == "missing_data"
    assert not result["report"]["production_decisions_enabled"]


def test_timestamped_stills_use_wall_clock_and_never_container_seconds(tmp_path):
    camera = tmp_path / "vial_filled"
    labels = camera / "labels"
    images = camera / "images"
    labels.mkdir(parents=True)
    images.mkdir()
    pd.DataFrame(
        [
            {
                "vial": i + 1,
                "recipe_id": 1,
                "t100_ml": 40,
                "t200_ml": 0,
                "t400_ml": 0,
                "start_time": f"00:00:0{2 * i}",
                "end_time": f"00:00:0{2 * i + 1}",
                "status": "ok",
            }
            for i in range(3)
        ]
    ).to_csv(labels / "vial_recipes.csv", index=False)
    frames = []
    for i in range(3):
        name = f"vial_{i + 1}.png"
        assert cv2.imwrite(
            str(images / name), np.full((480, 640, 3), (80, 100, 140), dtype=np.uint8)
        )
        frames.append(
            {
                "filename": name,
                "timestamp": f"2026-10-30 00:00:0{2 * i}.500000",
                "video_s": 2 * i + 0.5,
                "vial": i + 1,
                "recipe_id": 1,
                "in_fill": 1,
            }
        )
    pd.DataFrame(frames).to_csv(labels / "vial_fill_frame_labels.csv", index=False)
    config = default_config(tmp_path)
    config["sessions"] = [
        {
            "session_id": "still_fixture",
            "recipe_csv": str(labels / "vial_recipes.csv"),
            "frame_csv": str(labels / "vial_fill_frame_labels.csv"),
            "source_mode": "auto",
        }
    ]
    result = run_workflow(config, tmp_path, plots=False)
    assert result["report"]["measured_vials"] == 3
    assert (
        result["report"]["label_audits"][0]["measurement_source"]
        == "timestamped_stills"
    )
    assert result["measurements"].sample_frames.eq(1).all()
    assert result["measurements"].actual_source_times_s.tolist() == [
        [0.5],
        [2.5],
        [4.5],
    ]
    assert result["measurements"].decoded_times_s.tolist() == [[], [], []]
    assert result["measurements"].temporal_delta_e_max.isna().all()
    assert result["measurements"].production_status.eq("PENDING").all()


def test_container_clock_mismatch_is_not_silently_used(tmp_path):
    video = tmp_path / "short.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480)
    )
    if not writer.isOpened():
        pytest.skip("MJPG test codec unavailable")
    for _ in range(10):
        writer.write(np.full((480, 640, 3), 100, dtype=np.uint8))
    writer.release()
    recipes, frames = tmp_path / "recipes.csv", tmp_path / "frames.csv"
    recipes.write_text(
        "vial,recipe_id,t100_ml,t200_ml,t400_ml,start_time,end_time,status\n"
        "1,1,40,0,0,00:00:02,00:00:03,ok\n"
    )
    frames.write_text(
        "filename,timestamp,video_s,in_fill,vial,recipe_id\n"
        "absent.jpg,2026-10-30 00:00:00,0,0,,\n"
    )
    config = default_config(tmp_path)
    config["sessions"] = [
        {
            "session_id": "clock_mismatch",
            "recipe_csv": str(recipes),
            "frame_csv": str(frames),
            "video": str(video),
        }
    ]
    result = run_workflow(config, tmp_path, plots=False)
    assert result["report"]["status"] == "invalid_inputs"
    assert "clock exceeds video" in result["report"]["errors"][0]["error"]
    assert result["measurements"].empty


def test_stale_camera_frame_and_changed_audit_image_are_rejected(tmp_path):
    labels, images = tmp_path / "labels", tmp_path / "images"
    labels.mkdir()
    images.mkdir()
    source = images / "one.png"
    assert cv2.imwrite(str(source), np.full((480, 640, 3), 100, dtype=np.uint8))
    frames = labels / "frames.csv"
    frames.write_text(
        "filename,timestamp,video_s,in_fill,vial,recipe_id\n"
        "one.png,2026-10-30 00:00:00.500,0.5,1,1,1\n"
    )
    table = validate_vials(
        pd.DataFrame(
            [
                {
                    "vial": 1,
                    "recipe_id": 1,
                    "t100_ml": 40,
                    "t200_ml": 0,
                    "t400_ml": 0,
                    "start_s": 0,
                    "end_s": 1,
                }
            ]
        )
    )
    audit = pd.DataFrame(
        [
            {
                "filename": source.name,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "camera_timestamp": "2026-10-30 00:00:00.480",
            }
        ]
    )
    result = measure_timestamped_stills(
        table, frames, VialCalibration(), tmp_path / "contacts", clock_audit=audit
    )
    assert result.usable.iloc[0] and result.source_clock_verified.iloc[0]
    audit.loc[0, "camera_timestamp"] = "2026-10-29 23:59:59.500"
    stale = measure_timestamped_stills(
        table, frames, VialCalibration(), tmp_path / "contacts", clock_audit=audit
    )
    assert (
        not stale.usable.iloc[0] and "stale_camera_frame" in stale.quality_flags.iloc[0]
    )
    audit.loc[0, "source_sha256"] = "changed_image"
    changed = measure_timestamped_stills(
        table, frames, VialCalibration(), tmp_path / "contacts", clock_audit=audit
    )
    assert (
        not changed.usable.iloc[0] and "hash_mismatch" in changed.quality_flags.iloc[0]
    )
