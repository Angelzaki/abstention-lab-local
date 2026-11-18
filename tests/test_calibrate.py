from pathlib import Path

from abstention_lab.calibrate import calibrate

ROOT = Path(__file__).resolve().parents[1]


def test_calibration_finds_good_threshold():
    report = calibrate(ROOT / "samples" / "docs", ROOT / "samples" / "cases.jsonl")
    assert report["best"]["accuracy"] >= 0.8
    assert "min_score" in report["best"]
