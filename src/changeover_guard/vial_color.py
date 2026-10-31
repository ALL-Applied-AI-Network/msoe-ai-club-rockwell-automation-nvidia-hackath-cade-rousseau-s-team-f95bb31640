"""Vial-camera color correction, measurement diagnostics, and repeatability.

RGB values at public boundaries are sRGB 0..255. Matrix fitting and white
normalization happen in linear RGB; Lab is D65 / 2 degrees. An identity
calibration is useful for exploration but cannot authorize production decisions.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd
from skimage.color import deltaE_ciede2000, rgb2lab


def rgb_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0 or array.shape[-1] != 3 or not np.isfinite(array).all():
        raise ValueError("colors must contain finite RGB triples")
    if np.any((array < 0) | (array > 255)):
        raise ValueError("sRGB channels must be in 0..255")
    return array


def linear_rgb(rgb) -> np.ndarray:
    rgb = rgb_array(rgb) / 255.0
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def encoded_rgb(linear) -> np.ndarray:
    linear = np.clip(np.asarray(linear, dtype=float), 0, 1)
    return 255 * np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055
    )


def lab_values(rgb) -> np.ndarray:
    return rgb2lab(rgb_array(rgb) / 255.0)


def color_errors(actual, reference) -> np.ndarray:
    observed_lab, reference_lab = np.broadcast_arrays(
        lab_values(actual), lab_values(reference)
    )
    return deltaE_ciede2000(observed_lab, reference_lab)


@dataclass(frozen=True)
class VialCalibration:
    camera_id: str = "vial_fill"
    matrix: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    reference_size: tuple = (640, 480)
    liquid_roi: tuple = (300, 335, 324, 370)
    reference_roi: tuple | None = None
    reference_rgb: tuple | None = None
    settings_locked: bool = False
    alignment_reviewed: bool = False
    validation_delta_e: tuple = ()
    chart_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.settings_locked, bool) or not isinstance(
            self.alignment_reviewed, bool
        ):
            raise TypeError(
                "settings_locked and alignment_reviewed must be JSON booleans"
            )
        matrix = np.asarray(self.matrix, dtype=float)
        if (
            not self.camera_id
            or matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
        ):
            raise ValueError(
                "camera_id and a finite 3x3 linear RGB matrix are required"
            )
        width, height = self.reference_size
        if width <= 0 or height <= 0:
            raise ValueError("reference_size must be positive")
        for box in (self.liquid_roi, self.reference_roi):
            if box is None:
                continue
            if len(box) != 4 or not np.isfinite(box).all():
                raise ValueError("ROI must contain four finite coordinates")
            x0, y0, x1, y1 = box
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height * 450 / 480):
                raise ValueError(
                    "ROI must be inside the image and above the burned-in timestamp"
                )
        if (self.reference_roi is None) != (self.reference_rgb is None):
            raise ValueError(
                "reference_roi and reference_rgb must be supplied together"
            )
        if self.reference_rgb is not None and np.any(
            linear_rgb(self.reference_rgb) <= 0
        ):
            raise ValueError("reference_rgb must have positive channels")
        errors = np.asarray(self.validation_delta_e, dtype=float)
        if not np.isfinite(errors).all() or np.any(errors < 0):
            raise ValueError("invalid chart validation errors")

    @property
    def calibration_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:20]

    def correct(self, rgb, reference_rgb=None) -> np.ndarray:
        values = linear_rgb(rgb)
        if self.reference_roi is not None:
            if reference_rgb is None or np.any(linear_rgb(reference_rgb) <= 1e-6):
                raise ValueError("a usable reference patch is required in every frame")
            values = values * linear_rgb(self.reference_rgb) / linear_rgb(reference_rgb)
        return encoded_rgb(values @ np.asarray(self.matrix))

    def to_dict(self) -> dict:
        return {**asdict(self), "calibration_id": self.calibration_id}

    @classmethod
    def from_dict(cls, data: dict):
        fields = dict(data)
        claimed = fields.pop("calibration_id", None)
        calibration = cls(**fields)
        if claimed is not None and claimed != calibration.calibration_id:
            raise ValueError("calibration_id does not match the calibration contents")
        return calibration


def fit_camera_chart(chart: pd.DataFrame, config: VialCalibration) -> VialCalibration:
    """Fit paired camera/chart colors with separately acquired held-out patches.

    Columns: patch_id, split (train/validation), observed_r/g/b, reference_r/g/b.
    When using per-frame normalization, observed colors must already be normalized
    to config.reference_rgb, using the same procedure as the frame reader.
    """
    required = {
        "patch_id",
        "split",
        *[f"{prefix}_{c}" for prefix in ("observed", "reference") for c in "rgb"],
    }
    if not required.issubset(chart):
        raise ValueError(f"chart missing columns: {sorted(required - set(chart))}")
    if chart[list(required)].isna().any().any() or chart.patch_id.duplicated().any():
        raise ValueError(
            "chart patches must be unique, complete, and disjoint between splits"
        )
    if not chart.split.isin(["train", "validation"]).all():
        raise ValueError("chart split must be train or validation")
    observed = chart[[f"observed_{c}" for c in "rgb"]].to_numpy(float)
    reference = chart[[f"reference_{c}" for c in "rgb"]].to_numpy(float)
    train = (chart.split == "train").to_numpy()
    if min(train.sum(), (~train).sum()) < 6:
        raise ValueError("need at least six training and six held-out chart patches")
    source, target = linear_rgb(observed), linear_rgb(reference)
    if (
        min(np.linalg.matrix_rank(source[train]), np.linalg.matrix_rank(source[~train]))
        < 3
    ):
        raise ValueError("chart must span all three color channels in each split")
    matrix = np.linalg.lstsq(source[train], target[train], rcond=None)[0]
    predicted = encoded_rgb(source[~train] @ matrix)
    digest = hashlib.sha256(chart.to_csv(index=False).encode()).hexdigest()
    return replace(
        config,
        matrix=tuple(map(tuple, matrix)),
        validation_delta_e=tuple(color_errors(predicted, reference[~train])),
        chart_sha256=digest,
    )


def scaled_roi(shape, calibration: VialCalibration, box=None) -> tuple[int, ...]:
    height, width = shape[:2]
    ref_width, ref_height = calibration.reference_size
    if abs(width / height / (ref_width / ref_height) - 1) > 0.03:
        raise ValueError(
            "image aspect ratio differs from the calibrated camera; review the ROI"
        )
    x0, y0, x1, y1 = box if box is not None else calibration.liquid_roi
    result = (
        round(x0 * width / ref_width),
        round(y0 * height / ref_height),
        round(x1 * width / ref_width),
        round(y1 * height / ref_height),
    )
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError("scaled ROI is empty")
    return result


def measure_vial_color(image: np.ndarray, calibration: VialCalibration) -> dict:
    """Read a reviewed liquid patch; retain diagnostics even when it is unusable.

    This measures color only. Pixel height is not a calibrated fill-volume sensor.
    Spatial spread is a diagnostic, never a confidence interval.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError("image must be a nonempty uint8 BGR image")
    if min(image.shape[:2]) <= 0:
        raise ValueError("image must not be empty")
    x0, y0, x1, y1 = scaled_roi(image.shape, calibration)
    pixels = image[y0:y1, x0:x1, ::-1].reshape(-1, 3).astype(float)
    valid = (pixels.max(axis=1) > 3) & (pixels.max(axis=1) < 252)
    flags = []
    if valid.sum() < 40 or valid.mean() < 0.5:
        return {
            "usable": False,
            "flags": ["insufficient_liquid_pixels"],
            "rgb": None,
            "lab": None,
            "valid_fraction": float(valid.mean()),
            "roi": (x0, y0, x1, y1),
        }
    pixels = pixels[valid]
    raw = np.median(pixels, axis=0)
    reference = None
    if calibration.reference_roi is not None:
        a, b, c, d = scaled_roi(image.shape, calibration, calibration.reference_roi)
        patch = image[b:d, a:c, ::-1].reshape(-1, 3)
        if len(patch) < 40 or np.mean((patch <= 3) | (patch >= 252)) > 0.05:
            return {
                "usable": False,
                "flags": ["unusable_reference_patch"],
                "rgb": None,
                "lab": None,
                "valid_fraction": float(valid.mean()),
                "roi": (x0, y0, x1, y1),
            }
        reference = np.median(patch, axis=0)
    corrected = calibration.correct(raw, reference)
    spread = float(
        np.percentile(
            color_errors(calibration.correct(pixels, reference), corrected), 95
        )
    )
    if np.mean(pixels <= 1) > 0.1:
        flags.append("clipped_dark_channels")
    if spread > 10:
        flags.append("uneven_liquid_patch")
    unbounded = linear_rgb(raw)
    if reference is not None:
        unbounded *= linear_rgb(calibration.reference_rgb) / linear_rgb(reference)
    unbounded = unbounded @ np.asarray(calibration.matrix)
    if np.any((unbounded < 0) | (unbounded > 1)):
        flags.append("correction_out_of_gamut")
    return {
        "usable": not flags,
        "flags": flags,
        "raw_rgb": raw.tolist(),
        "rgb": corrected.tolist(),
        "lab": lab_values(corrected).tolist(),
        "spatial_delta_e_p95": spread,
        "valid_fraction": float(valid.mean()),
        "roi": (x0, y0, x1, y1),
    }


def repeatability_report(
    samples: pd.DataFrame | None, calibration: VialCalibration, threshold=2.2
) -> dict:
    """Require five stable controls, five captures each in two sessions.

    Rows contain corrected RGB, stable physical sample_id, session_id, unique
    capture_id, camera_id and calibration_id. Adjacent production frames are
    deliberately excluded from this study. The worst within-control pairwise
    CIEDE2000 must be below one third of the decision threshold.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    result = {
        "proven": False,
        "camera_id": calibration.camera_id,
        "calibration_id": calibration.calibration_id,
        "threshold": threshold,
        "budget_delta_e": threshold / 3,
        "reasons": [],
        "controls": [],
    }
    reasons = result["reasons"]
    if not calibration.alignment_reviewed:
        reasons.append(
            "Camera calibration has no saved alignment/ROI review acknowledgment."
        )
    if not calibration.settings_locked and calibration.reference_roi is None:
        reasons.append("Lock camera settings or supply a per-frame reference patch.")
    chart_errors = calibration.validation_delta_e
    if len(chart_errors) < 6 or calibration.chart_sha256 is None:
        reasons.append("Independent held-out color-chart validation is missing.")
    elif np.percentile(chart_errors, 95) >= threshold / 3:
        reasons.append("Held-out chart error exceeds the camera accuracy budget.")
    if samples is None or samples.empty:
        reasons.append("Independent stable-vial repeatability controls are missing.")
        return result
    required = {
        "sample_id",
        "session_id",
        "capture_id",
        "camera_id",
        "calibration_id",
        "r",
        "g",
        "b",
    }
    if not required.issubset(samples) or samples[list(required)].isna().any().any():
        raise ValueError(
            "repeatability rows require complete sample/session/capture/camera/calibration IDs and r,g,b"
        )
    if samples.capture_id.duplicated().any():
        raise ValueError("repeatability capture_id must be globally unique")
    if (
        not samples.camera_id.eq(calibration.camera_id).all()
        or not samples.calibration_id.eq(calibration.calibration_id).all()
    ):
        raise ValueError(
            "repeatability controls belong to a different camera/calibration"
        )
    rgb_array(samples[["r", "g", "b"]])
    if samples.sample_id.nunique() < 5:
        reasons.append("Need at least five stable physical color controls.")
    for sample_id, rows in samples.groupby("sample_id", sort=False):
        counts = rows.groupby("session_id").size()
        if (counts >= 5).sum() < 2:
            reasons.append(
                f"Control {sample_id} needs five independent captures in each of two sessions."
            )
        labs = lab_values(rows[["r", "g", "b"]].to_numpy())
        worst = float(deltaE_ciede2000(labs[:, None, :], labs[None, :, :]).max())
        result["controls"].append(
            {
                "sample_id": str(sample_id),
                "captures": len(rows),
                "sessions": len(counts),
                "worst_pair_delta_e": worst,
            }
        )
    worst = max(item["worst_pair_delta_e"] for item in result["controls"])
    result["worst_pair_delta_e"] = worst
    if worst >= threshold / 3:
        reasons.append("Stable-vial variation exceeds the repeatability budget.")
    result["proven"] = not reasons
    return result


def camera_evidence_ready(
    calibration: VialCalibration, report: dict | None, threshold=2.2
) -> bool:
    """Check the complete matching evidence before decisions or uncertainty."""
    if not report or not report.get("proven"):
        return False
    budget = threshold / 3
    controls = report.get("controls", [])
    study_complete = len(controls) >= 5 and all(
        c.get("sessions", 0) >= 2
        and c.get("captures", 0) >= 10
        and c.get("worst_pair_delta_e", float("inf")) < budget
        for c in controls
    )
    return bool(
        study_complete
        and report.get("calibration_id") == calibration.calibration_id
        and report.get("threshold") == threshold
        and report.get("camera_id") == calibration.camera_id
        and calibration.alignment_reviewed
        and (calibration.settings_locked or calibration.reference_roi is not None)
        and calibration.chart_sha256 is not None
        and len(calibration.validation_delta_e) >= 6
        and np.percentile(calibration.validation_delta_e, 95) < budget
    )
