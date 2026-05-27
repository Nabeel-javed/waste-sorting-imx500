from src.bin_logic import UNKNOWN_BIN, format_recommendation, get_recommended_bin, is_known_class


def test_known_class_maps_correctly():
    assert get_recommended_bin("plastic_bottle") == "Plastic / Packaging"
    assert get_recommended_bin("can") == "Metal / Recycling"
    assert get_recommended_bin("glass_jar") == "Glass"


def test_all_six_classes_mapped():
    for class_name in ["plastic_bottle", "can", "paper", "cardboard", "glass_jar", "food_wrapper"]:
        assert get_recommended_bin(class_name) != UNKNOWN_BIN


def test_unknown_class_returns_unknown_bin():
    assert get_recommended_bin("bottle") == UNKNOWN_BIN
    assert get_recommended_bin("person") == UNKNOWN_BIN
    assert get_recommended_bin("") == UNKNOWN_BIN


def test_is_known_class():
    assert is_known_class("plastic_bottle") is True
    assert is_known_class("bottle") is False


def test_format_recommendation_known():
    output = format_recommendation("plastic_bottle", 0.873)
    assert "plastic_bottle" in output
    assert "0.87" in output
    assert "Plastic" in output


def test_format_recommendation_unknown_does_not_raise():
    output = format_recommendation("bottle", 0.5)
    assert UNKNOWN_BIN in output

