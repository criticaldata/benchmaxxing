"""Imaging-lane entry point for the matched-conditions blind metric, so the MIMIC-CXR battery
(experiments/mimic_cxr_image/run_battery.py) can drive it like every other imaging arm.

The runner itself lives in experiments/blind_metric/matched_conditions.py and serves both lanes;
this module fixes ``--lane imaging`` and passes everything else through.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blind_metric"))
from matched_conditions import main as _main  # noqa: E402


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--lane" not in argv:
        argv = ["--lane", "imaging"] + argv
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
