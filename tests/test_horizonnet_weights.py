import importlib.util
from pathlib import Path

import pytest

TRAINING = Path(__file__).resolve().parents[1] / "tools" / "3dfront_training"


@pytest.fixture(scope="module")
def convert():
    spec = importlib.util.spec_from_file_location(
        "convert_horizonnet_weights", TRAINING / "convert_horizonnet_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_weight_gets_the_module_path_of_the_target(convert):
    renamed = convert.rename({"feature_extractor.conv.weight": 1, "linear.bias": 2},
                             "layout_estimation.horizon_net.module.")
    assert sorted(renamed) == [
        "layout_estimation.horizon_net.module.feature_extractor.conv.weight",
        "layout_estimation.horizon_net.module.linear.bias"]


def test_a_wrong_prefix_is_caught_instead_of_silently_dropping_weights(convert):
    renamed = convert.rename({"conv.weight": 1}, "layout_estimation.")
    with pytest.raises(ValueError, match="no place in the target model"):
        convert.check_against(renamed, ["layout_estimation.horizon_net.module.conv.weight"])


def test_a_target_left_untrained_is_caught_too(convert):
    renamed = convert.rename({"conv.weight": 1}, "net.")
    with pytest.raises(ValueError, match="stay untrained"):
        convert.check_against(renamed, ["net.conv.weight", "net.conv.bias"])


def test_a_full_match_passes_and_counts(convert):
    renamed = convert.rename({"a": 1, "b": 2}, "net.")
    assert convert.check_against(renamed, ["net.a", "net.b"]) == 2
