from datetime import datetime

import numpy as np

from changeover_guard import advise, measure, mixing
from changeover_guard.contracts import Recipe, TankReading, VialMeasurement

R6 = Recipe(6, {"T100": 30.0, "T200": 10.0})
R7 = Recipe(7, {"T100": 20.0, "T400": 50.0})
R12 = Recipe(12, {"T100": 10.0, "T200": 50.0, "T400": 10.0})
TANKS = TankReading(datetime(2026, 1, 1), {})


def test_vial_reads_the_calibration_box_as_rgb():
    image = np.zeros((480, 640, 3), np.uint8)
    image[335:370, 300:324] = (30, 20, 10)                  # BGR inside the box
    m = measure.vial(image, datetime(2026, 1, 1))
    assert m.color == (10.0, 20.0, 30.0) and np.isnan(m.fill_ml)


def test_no_changeover_predicts_the_clean_color():
    p = mixing.predict(R7, R7, TANKS)
    assert p.first_vial_color == p.clean_color and p.delta_e == 0.0 and not p.warn


def test_changeover_prediction_moves_off_the_clean_color():
    p = mixing.predict(R6, R7, TANKS)
    assert p.delta_e > 0 and p.first_vial_color != p.clean_color


def test_check_and_advice_explain_an_off_color_first_vial():
    p = mixing.predict(R6, R7, TANKS)
    off = VialMeasurement(datetime(2026, 1, 1), (60.0, 30.0, 5.0), float("nan"))
    c = advise.check(p, off)
    assert not c.ok
    a = advise.advise(p, c, [R12, R6], TANKS)
    assert "Recipe 6" in a.cause and not a.flush and a.next_recipe_id in (12, 6)


def test_clean_vial_passes_without_a_cause():
    p = mixing.predict(R7, R7, TANKS)
    c = advise.check(p, VialMeasurement(datetime(2026, 1, 1), p.clean_color, float("nan")))
    assert c.ok and advise.advise(p, c, [], TANKS).cause is None
