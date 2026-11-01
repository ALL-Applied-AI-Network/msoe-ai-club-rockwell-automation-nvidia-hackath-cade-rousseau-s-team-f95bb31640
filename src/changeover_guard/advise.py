"""Basic contract adapters; sequence attribution lives in operator_diagnostics."""

import numpy as np

from changeover_guard.contracts import (
    COLOR_WARN_DE,
    FILL_TOL_ML,
    Advice,
    Check,
    Prediction,
    Recipe,
    TankReading,
    VialMeasurement,
    delta_e,
)


class InspectionUnavailable(NotImplementedError):
    """A boolean legacy Check cannot encode pending/unmeasured fill."""


def check(
    prediction: Prediction,
    measurement: VialMeasurement,
    *,
    threshold=COLOR_WARN_DE,
    fill_tolerance_ml=FILL_TOL_ML,
) -> Check:
    """Compare to CLEAN color; absent fill must never become an automatic pass."""
    if (
        not np.isfinite(threshold)
        or threshold <= 0
        or not np.isfinite(fill_tolerance_ml)
        or fill_tolerance_ml <= 0
    ):
        raise ValueError("color and fill tolerances must be finite and positive")
    if not np.isfinite(measurement.fill_ml):
        raise InspectionUnavailable(
            "Vial fill volume has not been independently measured."
        )
    error = delta_e(measurement.color, prediction.clean_color)
    fill_error = float(measurement.fill_ml - sum(prediction.recipe.ml.values()))
    return Check(
        error, fill_error, error < threshold and abs(fill_error) <= fill_tolerance_ml
    )


def advise(
    prediction: Prediction,
    check: Check,
    queue: list[Recipe],
    tanks: TankReading,
) -> Advice:
    """Recommend investigation; one Check cannot prove a root cause or flush need."""
    if check.ok:
        return Advice(None, False, None)
    messages = []
    if abs(check.fill_error_ml) > FILL_TOL_ML:
        messages.append(
            "Measured fill differs from requested volume; inspect dispensing and the fill sensor."
        )
    if check.color_delta_e >= COLOR_WARN_DE:
        messages.append(
            "Color differs from the clean reference; inspect measurement quality, tank identity and recipe parameters."
        )
        if (
            prediction.prev_recipe is not None
            and prediction.prev_recipe.ml != prediction.recipe.ml
        ):
            messages.append(
                "Carryover is possible after this change; verify trajectory and recovery across subsequent vials."
            )
        messages.append(
            "Inspect the same vial before dispensing on a repeat; do not subtract pre-fill color."
        )
    if not messages:
        messages.append(
            "An inspection tolerance was exceeded; review the configured color and fill limits."
        )
    return Advice(" ".join(messages), False, None)
