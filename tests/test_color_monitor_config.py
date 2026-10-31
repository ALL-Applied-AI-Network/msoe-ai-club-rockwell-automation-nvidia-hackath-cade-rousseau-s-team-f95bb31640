from pathlib import Path

from src.color_monitor import config


def test_csi_dataset_path_is_relative_to_repository_root():
    repository_root = Path(__file__).resolve().parent.parent

    assert config.ROOT == repository_root
    assert config.DATASET == repository_root / "data" / "csi-vial-fill-dataset"
