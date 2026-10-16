import pytest

from changeover_guard.contracts import COLOR_WARN_DE, Recipe, delta_e

VIDEO5 = {5: {"T100": 10, "T200": 10, "T400": 10}, 6: {"T100": 30, "T200": 10}, 7: {"T100": 20, "T400": 50}, 8: {"T100": 20, "T200": 20, "T400": 20}, 9: {"T100": 10, "T200": 35, "T400": 35}, 10: {"T100": 45, "T200": 15, "T400": 15}, 11: {"T100": 10, "T200": 10, "T400": 50}, 12: {"T100": 10, "T200": 50, "T400": 10}, 13: {"T200": 50, "T400": 20}, 14: {"T100": 60, "T400": 10}}


def test_video5_recipes_valid():
    for rid, ml in VIDEO5.items():
        Recipe(rid, ml).validate()


@pytest.mark.parametrize(
    "ml",
    [{"T200": 60}, {"T300": 10}, {"T100": 5}, {"T100": 40, "T200": 30, "T400": 20}],
)
def test_invalid_recipes_raise(ml):
    with pytest.raises(ValueError):
        Recipe(1, ml).validate()


def test_delta_e_identity():
    assert delta_e((120, 80, 40), (120, 80, 40)) == pytest.approx(0, abs=1e-6)


def test_delta_e_orange_blue():
    assert delta_e((255, 165, 0), (0, 0, 255)) > COLOR_WARN_DE


def test_delta_e_black_white():
    assert delta_e((0, 0, 0), (255, 255, 255)) == pytest.approx(100, abs=0.5)
