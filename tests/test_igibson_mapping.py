import json
from pathlib import Path

IG56 = ['basket','bathtub','bed','bench','bottom_cabinet','bottom_cabinet_no_top','carpet',
        'chair','chest','coffee_machine','coffee_table','console_table','cooktop','counter',
        'crib','cushion','dishwasher','door','dryer','fence','floor_lamp','fridge',
        'grandfather_clock','guitar','heater','laptop','loudspeaker','microwave','mirror',
        'monitor','office_chair','oven','piano','picture','plant','pool_table','range_hood',
        'shelf','shower','sink','sofa','sofa_chair','speaker_system','standing_tv','stool',
        'stove','table','table_lamp','toilet','top_cabinet','towel_rack','trash_can',
        'treadmill','wall_clock','wall_mounted_tv','washer','window']

MAPPING = json.loads(
    (Path(__file__).resolve().parents[1] / "configs" / "protocols"
     / "front3d_to_igibson.json").read_text(encoding="utf-8"))


def test_every_target_exists_in_the_igibson_vocabulary():
    for group in ("confident", "debatable"):
        for source, target in MAPPING[group].items():
            assert target in IG56, f"{source} maps to an unknown class {target}"


def test_a_class_belongs_to_exactly_one_group():
    confident = set(MAPPING["confident"])
    debatable = set(MAPPING["debatable"])
    absent = set(MAPPING["no_counterpart"])
    assert not confident & debatable
    assert not confident & absent
    assert not debatable & absent


def test_the_control_row_keeps_only_confident_pairs():
    targets = set(MAPPING["confident"].values())
    assert "sofa_chair" not in targets, "an arguable pair must not enter the control row"
    assert {"bed", "chair", "sofa", "table"} <= targets


def test_the_mapping_is_not_empty_on_either_side():
    assert len(MAPPING["confident"]) >= 15
    assert MAPPING["no_counterpart"]
