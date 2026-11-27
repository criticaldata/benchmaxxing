"""The MedQA knowledge control reproduces its committed output from the committed rows, with no call."""
import json
from pathlib import Path

from experiments.medqa import knowledge_control

COMMITTED = Path(__file__).resolve().parents[1] / "experiments" / "medqa" / "results" / "knowledge_control.json"


def test_knowledge_control_reproduces_the_committed_file(tmp_path):
    out = tmp_path / "knowledge_control.json"
    assert knowledge_control.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(COMMITTED.read_text(encoding="utf-8"))


def test_the_strict_definition_is_never_larger_than_the_loose_ones():
    d = json.loads(COMMITTED.read_text(encoding="utf-8"))["definitions"]
    assert d["every"]["cases_known_by_all"] <= d["majority"]["cases_known_by_all"] <= d["union"]["cases_known_by_all"]
