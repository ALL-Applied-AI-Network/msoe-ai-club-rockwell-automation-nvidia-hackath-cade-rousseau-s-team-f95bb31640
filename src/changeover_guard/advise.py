"""Explanations and advice. Owners: Adrian, Niko."""

from changeover_guard.contracts import (
    Advice,
    Check,
    Prediction,
    Recipe,
    TankReading,
    VialMeasurement,
)


def check(prediction: Prediction, measurement: VialMeasurement) -> Check:
    """Compare a measured vial against its prediction."""
    raise NotImplementedError


def advise(
    prediction: Prediction,
    check: Check,
    queue: list[Recipe],
    tanks: TankReading,
) -> Advice:
    """Explain a failed check and advise on flushing and the next recipe; queue is in run order."""
    raise NotImplementedError
