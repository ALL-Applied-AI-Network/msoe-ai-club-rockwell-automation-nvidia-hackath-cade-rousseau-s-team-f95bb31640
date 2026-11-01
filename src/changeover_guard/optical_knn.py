"""Frame-level optical features and vectorized, training-only standardized k-NN."""

from enum import StrEnum

import numpy as np

from changeover_guard.vial_color import lab_values


class ObservationState(StrEnum):
    VALID = "VALID"
    SENSOR_CLIPPED = "SENSOR_CLIPPED"
    STALE_FRAME = "STALE_FRAME"
    UNSETTLED_MOTION = "UNSETTLED_MOTION"


FEATURES = ["L", "a", "lab_b", "od_r", "od_g", "od_b"]
ML = ["t100_ml", "t200_ml", "t400_ml"]
FRACTIONS = ["f100", "f200", "f400"]


def optical_features(rgb):
    """Fixed 245 intensity normalization; no measured pre-fill subtraction.

    OD on encoded camera RGB is an optical descriptor, not calibrated chemical
    absorbance. Signed OD is retained for channels brighter than the reference.
    """
    rgb = np.asarray(rgb, dtype=float)
    return np.concatenate(
        [lab_values(rgb), -np.log10(np.maximum(rgb, 1.0) / 245.0)], axis=-1
    )


def observation_state(flags):
    if any("CLIPPED" in flag for flag in flags):
        return ObservationState.SENSOR_CLIPPED.value
    if any(
        flag.startswith(("STALE", "MISSING", "DECODE", "INVALID")) for flag in flags
    ):
        return ObservationState.STALE_FRAME.value
    if flags:
        return ObservationState.UNSETTLED_MOTION.value
    return ObservationState.VALID.value


def inspect_patch(rgb_patch, *, previous=None, timestamp_offset_s=0.0):
    """Keep raw features even when a quality gate fails; never impute a color."""
    pixels = np.asarray(rgb_patch, dtype=float).reshape(-1, 3)
    if not len(pixels) or not np.isfinite(pixels).all():
        raise ValueError("ROI must contain finite RGB pixels")
    flags, details = [], []
    med = np.median(pixels, axis=0)
    dark = (pixels < 8).mean(axis=0)
    bright = (pixels > 250).mean(axis=0)
    for j, name in enumerate("RGB"):
        if dark[j] > 0.15 or pixels[:, j].mean() < 5:
            flags.append("DARK_CHANNEL_CLIPPED")
            details.append(f"{name}-Ch clipped (med={med[j]:.1f}; dark={dark[j]:.0%})")
        if bright[j] > 0.15:
            flags.append("SATURATION_CLIPPED")
            details.append(f"{name}-Ch saturated ({bright[j]:.0%})")
    spatial_std = pixels.std(axis=0)
    if spatial_std.max() > 35:
        flags.append("UNEVEN_PATCH")
        details.append(f"ROI spatial std={spatial_std.max():.1f} > 35")
    difference = np.nan
    if previous is not None and previous.shape == rgb_patch.shape:
        difference = float(
            np.abs(rgb_patch.astype(float) - previous.astype(float)).mean()
        )
        if difference == 0.0:
            flags.append("STALE_FRAME")
            details.append("Consecutive sampled ROI pixels identical (MAD=0)")
    if not np.isfinite(timestamp_offset_s) or abs(timestamp_offset_s) > 2:
        flags.append("STALE_FRAME")
        details.append(f"Decoded timestamp offset={timestamp_offset_s:.3f}s")
    flags = sorted(set(flags))
    return {
        **dict(zip(["r", "g", "b"], med)),
        **dict(zip(FEATURES, optical_features(med))),
        "ObservationState": observation_state(flags),
        "quality_flags": ";".join(flags),
        "quality_details": "; ".join(details),
        "usable": not flags,
        "pixel_difference": difference,
        "spatial_std_max": float(spatial_std.max()),
    }


class VectorizedOpticalKNN:
    """No Python loops over query/training pairs or classes; k defaults to five."""

    def __init__(self, k=5):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k

    def fit(self, X, y_class, Y_vol):
        X, y, volumes = (
            np.asarray(X, float),
            np.asarray(y_class, int),
            np.asarray(Y_vol, float),
        )
        if X.ndim != 2 or X.shape[1] != 6 or len(X) == 0:
            raise ValueError("Training features must have shape (N, 6), N > 0")
        if y.shape != (len(X),) or volumes.shape != (len(X), 3):
            raise ValueError("Targets must align with training frames")
        if not np.isfinite(X).all() or not np.isfinite(volumes).all():
            raise ValueError("Nonfinite training data")
        if (volumes < 0).any() or not np.allclose(volumes.sum(axis=1), 1):
            raise ValueError("Volume fractions must be nonnegative and sum to one")
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        self.X_ = (X - self.mean_) / self.scale_
        self.y_, self.volumes_ = y, volumes
        self.classes_ = np.unique(y)
        return self

    def distances(self, X):
        if not hasattr(self, "X_"):
            raise ValueError("Fit the optical model before prediction")
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] != 6 or not np.isfinite(X).all():
            raise ValueError("Query features must be finite (M, 6)")
        query = (X - self.mean_) / self.scale_
        squared = (
            (query * query).sum(axis=1)[:, None]
            + (self.X_ * self.X_).sum(axis=1)[None, :]
            - 2 * query @ self.X_.T
        )
        return np.sqrt(np.maximum(squared, 0))

    def predict(self, X):
        distances = self.distances(X)
        k = min(self.k, len(self.X_))
        # kth is zero-based, including when N == k.
        neighbors = np.argpartition(distances, k - 1, axis=1)[:, :k]
        near = np.take_along_axis(distances, neighbors, axis=1)
        weights = 1 / (near + 1e-6)
        weights /= weights.sum(axis=1, keepdims=True)
        votes = self.y_[neighbors, None] == self.classes_[None, None, :]
        probabilities = (weights[:, :, None] * votes).sum(axis=1)
        winner = probabilities.argmax(axis=1)
        return {
            "recipe_id": self.classes_[winner],
            "confidence": probabilities[np.arange(len(winner)), winner],
            "probabilities": probabilities,
            "fractions": (weights[:, :, None] * self.volumes_[neighbors]).sum(axis=1),
            "nearest_distance": distances.min(axis=1),
        }

    def predict_recipe(self, X):
        result = self.predict(X)
        return result["recipe_id"], result["confidence"]

    def predict_volumes(self, X):
        return self.predict(X)["fractions"]

    def to_dict(self):
        return {
            "k": self.k,
            "features": FEATURES,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "standardized_X": self.X_.tolist(),
            "recipe_ids": self.y_.tolist(),
            "fractions": self.volumes_.tolist(),
            "reference_intensity": [245.0] * 3,
        }
