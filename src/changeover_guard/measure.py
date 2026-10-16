"""Camera and color measurement. Owners: Braedon, Kal."""

from datetime import datetime

import numpy as np

from changeover_guard.contracts import TankReading, VialMeasurement


def vial(image: np.ndarray, timestamp: datetime) -> VialMeasurement:
    """Measure the color and fill level of a vial from a BGR image.

    timestamp is the frame's capture time (the `timestamp` column from `dataset.frames`).
    """
    raise NotImplementedError


def tanks(image: np.ndarray, timestamp: datetime) -> TankReading:
    """Read the color of each tank (T100, T200, T400) from a BGR image.

    timestamp is the frame's capture time (the `timestamp` column from `dataset.frames`).
    """
    raise NotImplementedError
