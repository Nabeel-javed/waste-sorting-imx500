import pytest

from src.zone_logic import check_zone_correctness, expected_zone, get_object_center, get_zone_from_center


def test_object_center_midpoint():
    assert get_object_center((100, 100, 200, 300)) == (150, 200)


@pytest.mark.parametrize(
    ("center_x", "frame_width", "zone"),
    [
        (50, 900, "left"),
        (450, 900, "center"),
        (800, 900, "right"),
        (299, 900, "left"),
        (301, 900, "center"),
        (599, 900, "center"),
        (601, 900, "right"),
    ],
)
def test_get_zone_from_center(center_x, frame_width, zone):
    assert get_zone_from_center(center_x, frame_width) == zone


def test_correct_zone_plastic():
    assert check_zone_correctness("plastic_bottle", "left") is True
    assert check_zone_correctness("plastic_bottle", "right") is False


def test_correct_zone_paper_cardboard():
    assert check_zone_correctness("paper", "center") is True
    assert check_zone_correctness("cardboard", "center") is True


def test_correct_zone_glass_metal_right():
    assert check_zone_correctness("glass_jar", "right") is True
    assert check_zone_correctness("can", "right") is True


def test_unknown_class_never_correct():
    assert check_zone_correctness("bottle", "left") is False
    assert expected_zone("bottle") is None

