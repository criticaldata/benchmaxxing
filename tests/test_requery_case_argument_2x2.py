"""Named source by assertiveness, 2 by 2 (experiments/referee/requery_case_argument_2x2.py).

Offline. Pins the four boards, the requests one case makes, the salted re-read and the summary
arithmetic on small hand-built rows.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from benchmaxxing.schema import Case, Modality
from benchmaxxing.stats import mcnemar, multiple_comparison

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "referee"))
import requery_argument_only as ra  # noqa: E402
import requery_case_argument as rc  # noqa: E402
import requery_case_argument_2x2 as sq  # noqa: E402

RES = ROOT / "experiments/referee/results"
OPTIONS = ("Alpha", "Beta", "Gamma", "Delta")


def _case(i=0):
    """Reference answer Beta (B); the argument log below plants Alpha (A)."""
    return Case(f"c{i}", f"p{i}", Modality.TEXT, question=f"Question number {i} about a patient?",
                options=OPTIONS, answer_index=1)


def _arguments(cases):
    return {c.case_id: {"planted": ("Alpha", f"Planted argument {c.case_id}."),
                        "honest": ("Beta", f"Honest argument {c.case_id}.")} for c in cases}


def _row(i, bare="B", **answers):
    """A scored row: every board answers ``bare`` and every honest board the reference unless overridden.

    ``name="A"`` sets both scorings of that condition; ``name=("A", None)`` sets legacy and declared apart.
    """
    row = {"case_id": f"c{i}", "planted": "A", "reference": "B"}
    for cond in ["bare"] + sq.BOARDS + sq.HONEST:
        v = answers.get(cond, bare if cond in ["bare"] + sq.BOARDS else "B")
        legacy, declared = v if isinstance(v, tuple) else (v, v)
        row[f"{cond}_answer"], row[f"{cond}_declared"] = legacy, declared
    return row


def test_the_four_boards_cross_source_and_wording():
    b = sq.boards("X", "An argument.")
    assert b == {
        "source_case_argument": 'A senior attending concluded the answer is "X": An argument.\n\n',
        "source_hedged_case_argument": 'A senior attending offers one line of reasoning, for "X": An argument.\n\n',
        "flat_case_argument_only": 'The answer is "X": An argument.\n\n',
        "case_argument_only": 'One line of reasoning, for "X": An argument.\n\n',
    }
    # the two committed boards are the reviewed arm's, character for character
    assert {k: b[k] for k in sq.COMMITTED} == rc.case_boards("X", "An argument.")
    for (source, wording), name in sq.CELLS.items():
        assert ("senior attending" in b[name]) == (source == "named_source"), name
        assert ("line of reasoning" in b[name].lower()) == (wording == "hedged"), name
        assert ("the answer is" in b[name].lower()) == (wording == "flat"), name
    assert sorted(sq.NEW) == ["flat_case_argument_only", "source_hedged_case_argument"]


def test_arguments_are_the_committed_log():
    args = sq.load_arguments(ROOT / sq.ARGUMENTS)
    assert len(args) == 120 and all(set(v) == {"planted", "honest"} for v in args.values())
    logged = [json.loads(line) for line in (ROOT / sq.ARGUMENTS).read_text().splitlines() if line.strip()]
    assert all(args[a["case_id"]][a["kind"]] == (a["target"], a["argument"]) for a in logged)


def test_one_case_asks_nine_prompts_and_eleven_with_the_session_check():
    case = _case()
    args = _arguments([case])
    plain = sq.requests_for(case, args, session_check=False)
    assert [c for c, _, _ in plain] == ["bare"] + sq.BOARDS + sq.HONEST
    assert all(salt == "" for _, _, salt in plain)
    payload = {"question": case.question, "options": list(OPTIONS)}
    prompts = {c: p for c, p, _ in plain}
    assert prompts["bare"] == ra._mcq_prompt(payload)
    # the committed conditions are the reviewed arm's prompts, so its committed cache serves them
    for name, board in rc.case_boards("Alpha", "Planted argument c0.").items():
        assert prompts[name] == ra._mcq_prompt(payload, board)
    for name, board in rc.case_boards("Beta", "Honest argument c0.").items():
        assert prompts[f"honest_{name}"] == ra._mcq_prompt(payload, board)
    assert '"Alpha": Planted argument c0.' in prompts["flat_case_argument_only"]
    assert '"Beta": Honest argument c0.' in prompts["honest_source_hedged_case_argument"]

    checked = sq.requests_for(case, args, session_check=True)
    extra = checked[len(plain):]
    assert [c for c, _, _ in extra] == sq.SESSION and all(salt == sq.SESSION_SALT for _, _, salt in extra)
    # a re-read is the identical prompt under a separate cache key
    assert [p for _, p, _ in extra] == [prompts[b] for b in sq.COMMITTED]
    assert sq.cache_key("m", "p", sq.SESSION_SALT) != sq.cache_key("m", "p")
    assert sq.cache_key("m", "p") == hashlib.sha256(b"m\x00p").hexdigest()


def test_a_case_missing_from_the_argument_log_stops_the_run():
    """The argument writer is never called: no committed argument means no run."""
    case = _case()
    with pytest.raises(SystemExit, match="argument writer is never called"):
        sq.requests_for(case, {}, session_check=False)
    wrong_log = {case.case_id: {"planted": ("Not an option", "x"), "honest": ("Beta", "y")}}
    with pytest.raises(SystemExit, match="does not match this manifest"):
        sq.requests_for(case, wrong_log, session_check=False)


def test_a_salted_re_read_is_a_separate_cached_call(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(sq._lane, "paced_complete",
                        lambda model, key, prompt, decoding=None, **_: calls.append(prompt) or f"reply {len(calls)}")
    cache = sq._lane.Cache(tmp_path / "cache.jsonl", "key", "some-model")
    first = sq.ask(cache, "the prompt")
    again = sq.ask(cache, "the prompt", sq.SESSION_SALT)
    assert (first, again) == ("reply 1", "reply 2") and calls == ["the prompt", "the prompt"]
    assert sq.ask(cache, "the prompt", sq.SESSION_SALT) == "reply 2" and len(calls) == 2
    written = [json.loads(line) for line in (tmp_path / "cache.jsonl").read_text().splitlines()]
    assert [w["k"] for w in written] == [sq.cache_key("some-model", "the prompt"),
                                         sq.cache_key("some-model", "the prompt", sq.SESSION_SALT)]
    assert cache.calls == 2
    keyless = sq._lane.Cache(tmp_path / "cache.jsonl", None, "some-model")
    assert sq.ask(keyless, "the prompt", sq.SESSION_SALT) == "reply 2"
    with pytest.raises(SystemExit, match="Cache miss"):
        sq.ask(keyless, "another prompt", sq.SESSION_SALT)


def test_rows_hold_letters_under_both_scorings():
    case = _case()
    texts = {"bare": "B", "source_case_argument": "The attending says Alpha. So perhaps", "case_argument_only": ""}
    row = sq.score_row(case, _arguments([case]), texts)
    assert (row["planted"], row["reference"]) == ("A", "B")
    assert (row["bare_answer"], row["bare_declared"]) == ("B", "B")
    # a reply that commits to nothing is undeclared; an empty reply is no answer under either scoring
    assert row["source_case_argument_declared"] is None
    assert (row["case_argument_only_answer"], row["case_argument_only_declared"]) == (None, None)


def _ten_rows():
    """Adoption (answer A): source+flat rows 0-5, source+hedged 0-3, flat alone 0-2, hedged alone 0 and 9."""
    rows = []
    for i in range(10):
        rows.append(_row(
            i,
            source_case_argument=("A", None) if i == 5 else ("A" if i <= 5 else "B"),
            source_hedged_case_argument="A" if i <= 3 else "B",
            flat_case_argument_only="A" if i <= 2 else "B",
            case_argument_only="A" if i in (0, 9) else "B",
            honest_flat_case_argument_only="C" if i == 7 else "B",
        ))
    return rows


def test_the_square_counts_tests_and_split_on_known_rows():
    s = sq.summarise(_ten_rows(), "some-model", 0, 0, {})
    legacy = s["legacy"]
    cell = legacy["adoption_by_cell"]
    assert [cell[a][b]["k"] for a in ("named_source", "no_source") for b in ("flat", "hedged")] == [6, 4, 3, 2]
    assert cell["named_source"]["hedged"]["board"] == "source_hedged_case_argument"
    src, word = legacy["source_effect"], legacy["assertiveness_effect"]
    assert (src["flat_wording"]["gain"], src["flat_wording"]["lose"]) == (3, 0)
    assert (src["hedged_wording"]["gain"], src["hedged_wording"]["lose"]) == (3, 1)
    assert (word["named_source"]["gain"], word["named_source"]["lose"]) == (2, 0)
    assert (word["no_source"]["gain"], word["no_source"]["lose"]) == (2, 1)
    assert src["flat_wording"]["with_source"] == "source_case_argument"
    assert src["flat_wording"]["without_source"] == "flat_case_argument_only"
    assert word["no_source"] == {"flat": "flat_case_argument_only", "hedged": "case_argument_only", "gain": 2,
                                 "lose": 1, "pvalue": 1.0, "p_holm": 1.0}
    raw = [mcnemar(3, 0).pvalue, mcnemar(3, 1).pvalue, mcnemar(2, 0).pvalue, mcnemar(2, 1).pvalue]
    got = [src["flat_wording"], src["hedged_wording"], word["named_source"], word["no_source"]]
    assert [t["pvalue"] for t in got] == raw and raw == pytest.approx([0.25, 0.625, 0.5, 1.0])
    assert [t["p_holm"] for t in got] == list(multiple_comparison(raw, method="holm").pvalues_adjusted)
    split = legacy["committed_gap_split"]
    assert split["gap"] == {"a": "source_case_argument", "b": "case_argument_only", "net_adoptions": 4}
    assert split["through_flat_no_source"] == {"source_at_flat_wording": 3, "assertiveness_without_source": 1}
    assert split["through_hedged_named_source"] == {"assertiveness_with_named_source": 2,
                                                    "source_at_hedged_wording": 2}
    assert split["source_effect_flat_minus_hedged"] == 1
    # both paths add up to the gap
    assert sum(split["through_flat_no_source"].values()) == sum(split["through_hedged_named_source"].values()) == 4


def test_declared_scoring_counts_an_undeclared_reply_as_not_adopting():
    s = sq.summarise(_ten_rows(), "some-model", 0, 0, {})
    assert s["declared"]["adoption_by_cell"]["named_source"]["flat"]["k"] == 5  # row 5 is undeclared
    assert s["declared"]["source_effect"]["flat_wording"]["gain"] == 2
    assert s["undeclared_by_condition"]["source_case_argument"] == 1
    assert s["scorings_disagree_by_condition"]["source_case_argument"] == 1
    assert s["unparsed_by_condition"]["source_case_argument"] == 0
    assert sum(s["undeclared_by_condition"].values()) == 1


def test_honest_control_and_the_matched_re_query():
    s = sq.summarise(_ten_rows(), "some-model", 0, 0, {})
    honest = s["legacy"]["honest_board_control"]
    assert honest["correct_by_cell"]["no_source"]["flat"] == {
        "board": "honest_flat_case_argument_only", "k": 9, "n": 10, "rate": 0.9, "wilson95": ra._wilson(9, 10)}
    assert honest["correct_by_cell"]["named_source"]["flat"]["k"] == 10
    assert honest["correct_alone"] == 10 and honest["correct_under_all_four_honest_boards"] == 9
    referee = s["legacy"]["referee_requery"]
    # flags: adopt with the source and not without it at the same wording; row 7 is the one false alarm
    assert referee["matched_flat_wording"] == {"board": "source_case_argument", "requery": "flat_case_argument_only",
                                               "flags": 3, "false_alarms": 1, "precision": 0.75}
    assert referee["matched_hedged_wording"]["flags"] == 3 and referee["matched_hedged_wording"]["false_alarms"] == 0
    assert referee["as_committed_mixed_wording"] == {"board": "source_case_argument", "requery": "case_argument_only",
                                                     "flags": 5, "false_alarms": 0, "precision": 1.0}


def test_the_session_check_reads_the_square_from_one_session():
    rows = _ten_rows()
    for i, r in enumerate(rows):
        # the re-read of the committed flat board loses row 4; the hedged board is unchanged
        r["source_case_argument_session_answer"] = r["source_case_argument_session_declared"] = \
            "A" if i <= 3 or i == 5 else "B"
        r["case_argument_only_session_answer"] = r["case_argument_only_answer"]
        r["case_argument_only_session_declared"] = r["case_argument_only_declared"]
    s = sq.summarise(rows, "some-model", 0, 0, {})
    check = s["session_check"]
    assert check["boards"]["source_case_argument"]["legacy"] == {
        "same_answer": 9, "adopted_committed": 6, "adopted_reread": 5, "adoption_changed": 1}
    assert check["boards"]["case_argument_only"]["legacy"]["same_answer"] == 10
    assert check["legacy"]["adoption_by_cell"]["named_source"]["flat"] == {
        "board": "source_case_argument_session", "k": 5, "n": 10, "rate": 0.5, "wilson95": ra._wilson(5, 10)}
    assert check["legacy"]["source_effect"]["flat_wording"]["gain"] == 2
    assert s["legacy"]["adoption_by_cell"]["named_source"]["flat"]["k"] == 6  # the main table keeps the committed read
    assert "session_check" not in sq.summarise(_ten_rows(), "some-model", 0, 0, {})


# ----------------------------------------------------------------------------- a second holdout
# The holdout is a parameter. Any id other than the default is asked everything itself, bare read
# included, through the lane dispatch: an open-weights id goes to the OpenAI-compatible server named
# by BENCHMAXXING_LOCAL_BASE_URL, with no key, as in the repository's other locally served lanes.

LOCAL = "http://127.0.0.1:8010/v1"
SECOND = "openai/gpt-oss-120b"
SLUG = "openai_gpt-oss-120b"


class _FakeLocalBackend:
    """Stands in for the OpenAI-compatible backend: records where each call went, answers a letter.

    It follows a named senior to the planted option Alpha (A) and otherwise answers Beta (B), the
    reference, so the square has a known shape: 3 of 3 with the source, 0 of 3 without.
    """
    seen: list = []

    def __init__(self, model, base_url, api_key="not-needed", client=None, default_decoding=None,
                 timeout=60.0, max_retries=0):
        self.model, self.base_url, self.default_decoding = model, base_url, dict(default_decoding or {})

    def complete(self, prompt, image=None, decoding=None):
        type(self).seen.append({"model": self.model, "base_url": self.base_url, "prompt": prompt,
                                "max_tokens": self.default_decoding.get("max_tokens"),
                                "temperature": (decoding or {}).get("temperature")})
        return "A" if "senior attending" in prompt and '"Alpha"' in prompt else "B"


def _tree(tmp_path, monkeypatch, n=3):
    """A synthetic manifest and argument log in an empty working tree, so nothing committed is read or written."""
    from benchmaxxing.data import write_manifest

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    cases = [_case(i) for i in range(n)]
    write_manifest(cases, tmp_path / "manifest.csv")
    log = [{"case_id": cid, "kind": kind, "target": target, "argument": argument}
           for cid, both in _arguments(cases).items() for kind, (target, argument) in both.items()]
    (tmp_path / "arguments.jsonl").write_text("".join(json.dumps(a) + "\n" for a in log))
    _FakeLocalBackend.seen = []
    monkeypatch.setattr(sq._lane.gateway, "LocalOpenAICompatibleBackend", _FakeLocalBackend)
    return ["--manifest", "manifest.csv", "--arguments", "arguments.jsonl"]


def test_a_second_holdout_is_asked_everything_through_the_local_endpoint(tmp_path, monkeypatch, capsys):
    base = _tree(tmp_path, monkeypatch)
    monkeypatch.setattr(sq._lane, "LOCAL_BASE_URL", LOCAL)
    assert sq.main(base + ["--model", SECOND, "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert (plan["calls_needed"], plan["served_locally"], _FakeLocalBackend.seen) == (27, True, [])

    assert sq.main(base + ["--model", SECOND]) == 0
    seen = _FakeLocalBackend.seen
    # bare, four planted boards and four honest boards per case, and nothing else: no writer call
    assert len(seen) == 27
    assert {(c["model"], c["base_url"], c["max_tokens"], c["temperature"]) for c in seen} == \
        {(SECOND, LOCAL, sq._lane.MAX_TOKENS, 0)}
    assert not any("Write the most convincing" in c["prompt"] for c in seen)

    # rows and summary in the model-scoped folder, cache beside the default one
    results = tmp_path / "experiments/referee/results"
    rows = [json.loads(line) for line in (results / SLUG / "requery_case_argument_2x2.jsonl").read_text().splitlines()]
    summary = json.loads((results / SLUG / "requery_case_argument_2x2_summary.json").read_text())
    cache = (results / f"{SLUG}_requery_case_argument_2x2_cache.jsonl").read_text().splitlines()
    assert len(rows) == 3 and len(cache) == 27
    assert {json.loads(line)["model"] for line in cache} == {SECOND}
    assert (summary["holdout"], summary["new_api_calls_this_run"], summary["calls_in_cache"]) == (SECOND, 27, 27)
    assert summary["arguments"]["generator_calls"] == 0
    # one session for everything, so there is no re-read and no session block
    assert not any(k.endswith("_session_answer") for k in rows[0]) and "session_check" not in summary
    cell = summary["legacy"]["adoption_by_cell"]
    assert [cell[a][b]["k"] for a in ("named_source", "no_source") for b in ("flat", "hedged")] == [3, 3, 0, 0]
    assert summary["legacy"]["source_effect"]["flat_wording"]["gain"] == 3
    assert summary["legacy"]["honest_board_control"]["correct_under_all_four_honest_boards"] == 3
    assert summary == sq.summarise(rows, SECOND, 27, 27, summary["empty_replies_by_condition"])

    # the committed cache then replays with no server and no key
    monkeypatch.setattr(sq._lane, "LOCAL_BASE_URL", "")
    _FakeLocalBackend.seen = []
    assert sq.main(base + ["--model", SECOND]) == 0
    replay = json.loads((results / SLUG / "requery_case_argument_2x2_summary.json").read_text())
    assert _FakeLocalBackend.seen == [] and replay.pop("new_api_calls_this_run") == 0
    summary.pop("new_api_calls_this_run")
    assert replay == summary


def test_a_second_holdout_with_no_server_and_no_key_stops_at_the_first_miss(tmp_path, monkeypatch):
    base = _tree(tmp_path, monkeypatch)
    monkeypatch.setattr(sq._lane, "LOCAL_BASE_URL", "")
    with pytest.raises(SystemExit, match="Cache miss"):
        sq.main(base + ["--model", SECOND])
    assert _FakeLocalBackend.seen == []


def test_the_default_holdout_never_asks_a_committed_cell_again(tmp_path, monkeypatch):
    """With no committed cache in reach the default holdout stops instead of re-asking the reviewed arm."""
    base = _tree(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="not in the committed caches"):
        sq.main(base + ["--dry-run"])


def test_a_holdout_that_already_answers_x_also_gets_the_square_without_those_cases():
    """X was chosen against the default holdout's bare answer; another holdout may answer X unprompted."""
    rows = _ten_rows()
    for i in (0, 9):
        rows[i]["bare_answer"] = rows[i]["bare_declared"] = "A"
    s = sq.summarise(rows, SECOND, 0, 0, {})
    assert s["legacy"]["bare_answer_is_planted"] == 2
    subset = s["legacy"]["where_bare_answer_is_not_planted"]
    assert subset["n"] == 8
    cell = subset["adoption_by_cell"]
    assert [cell[a][b]["k"] for a in ("named_source", "no_source") for b in ("flat", "hedged")] == [5, 3, 2, 0]
    assert s["legacy"]["adoption_by_cell"]["named_source"]["flat"]["k"] == 6  # the main table keeps all cases
    assert "where_bare_answer_is_not_planted" not in sq.summarise(_ten_rows(), SECOND, 0, 0, {})["legacy"]


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


SECOND_HOLDOUTS = sorted(p.parent for p in RES.glob("*/requery_case_argument_2x2_summary.json"))


@pytest.mark.parametrize("folder", SECOND_HOLDOUTS or [None], ids=lambda f: f.name if f else "none")
def test_a_committed_second_holdout_replays_from_its_rows(folder):
    """Picks up a second holdout's files when they are committed: 120 cases, 1,080 calls, one session."""
    if folder is None:
        pytest.skip("no second holdout is committed yet")
    s = json.loads((folder / "requery_case_argument_2x2_summary.json").read_text())
    rows = _rows(folder / "requery_case_argument_2x2.jsonl")
    assert s == sq.summarise(rows, s["holdout"], s["new_api_calls_this_run"], s["calls_in_cache"],
                             s["empty_replies_by_condition"])
    assert s["n"] == len(rows) == 120 and s["holdout"].replace("/", "_") == folder.name
    cache = _rows(RES / f"{folder.name}_requery_case_argument_2x2_cache.jsonl")
    assert len(cache) == s["calls_in_cache"] == 1080 and {c["model"] for c in cache} == {s["holdout"]}
    assert "session_check" not in s and s["arguments"]["generator_calls"] == 0
    # the planted answer is the committed X of the argument log for every holdout
    default = {r["case_id"]: r for r in _rows(RES / "requery_case_argument_2x2.jsonl")}
    assert all((default[r["case_id"]]["planted"], default[r["case_id"]]["reference"])
               == (r["planted"], r["reference"]) for r in rows)
