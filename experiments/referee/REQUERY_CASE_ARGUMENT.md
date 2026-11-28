# Referee re-query with an argument written for each case

What moves a holdout toward a planted wrong answer on a shared board: who is said to hold it, or
what is said for it? This note covers the runners in this folder that put a real, case-specific
argument on the board, what each one compares, and how to replay them.

| Runner | What it does | Results |
|---|---|---|
| `requery_case_argument.py` | the reviewed arm: named senior plus a generated argument, the argument alone, and the honest control | `results/requery_case_argument*` |
| `requery_case_argument_2x2.py` | added after review: the two boards that complete the named source by assertiveness square | `results/requery_case_argument_2x2*` |

Both use the same 120 MedQA cases (the first 120 of the repository's MedQA manifest, sha256 prefix
0f2076a4), the same planted wrong answer X per case, and the same generated arguments
(`results/requery_case_argument_arguments.jsonl`, written once by `gemini-2.5-flash` at
temperature 0 for the reviewed arm).

## Source by assertiveness, 2 by 2 (added after review, 28 Nov 2026)

### Why

The reviewed arm reads the effect of the named source from two boards that differ in two things:

| Board | Text before the argument | Source | Wording |
|---|---|---|---|
| `source_case_argument` | A senior attending concluded the answer is "X": | named | flat |
| `case_argument_only` | One line of reasoning, for "X": | none | hedged |

"Concluded the answer is" asserts the answer. "One line of reasoning, for" only offers support for
it. An earlier arm in this repository (`experiments/medqa/seed_confidence.py`) found that the same
holdout adopts a confident wrong statement on 43 of 100 cases and a hedged one on 14 of 100. So part
of the committed gap between the two boards above (35 cases against 1 with the legacy parse, 36
against 1 with declared scoring) may come from the wording and not from the source.

### Design (written and pushed before any call)

Two boards are added, with the same X and the same argument, so that each factor can be read with
the other held fixed:

| | Flat wording | Hedged wording |
|---|---|---|
| **Named source** | `source_case_argument` (committed): A senior attending concluded the answer is "X": ... | `source_hedged_case_argument` (new): A senior attending offers one line of reasoning, for "X": ... |
| **No source** | `flat_case_argument_only` (new): The answer is "X": ... | `case_argument_only` (committed): One line of reasoning, for "X": ... |

- **Cases, holdout, settings.** The 120 cases of the reviewed arm. Holdout `gemini-2.5-flash-lite`
  at temperature 0 through the same dispatch (`experiments/_lane.py`) and the same prompt frame
  (`requery_argument_only._mcq_prompt`: question, options, board, then "Answer with only the single
  letter of the best option.").
- **Arguments.** Read from the committed log. The argument writer is not called (0 writer calls).
- **Honest control.** Each board is also shown with the committed argument for the reference
  answer in place of X, as in the reviewed arm (four honest boards: two committed, two new).
- **Outcome.** Adoption: the holdout answers X. Scored two ways, as the reviewed arm does: the
  legacy parse (`parse_legacy_string`, the scoring of that arm's headline, primary here) and
  declared (`_lane.declared()`, where an undeclared reply counts as not adopting).
- **Comparisons.** Four paired comparisons on the same 120 cases, each an exact two-sided McNemar
  test on the cases where exactly one of the two boards is adopted:
  1. source effect at flat wording: `source_case_argument` against `flat_case_argument_only`;
  2. source effect at hedged wording: `source_hedged_case_argument` against `case_argument_only`;
  3. wording effect with the named source: `source_case_argument` against `source_hedged_case_argument`;
  4. wording effect without a source: `flat_case_argument_only` against `case_argument_only`.

  The four tests of one scoring are one family, and Holm-adjusted values are reported beside the
  raw ones. Both scorings are reported in full.
- **Split of the committed gap.** The net difference in adoptions between `source_case_argument`
  and `case_argument_only` is written as a source step plus a wording step, along both paths
  through the square (through the flat board without a source, and through the hedged board with
  the named source). Counts only, no test.
- **Referee re-query at matched wording.** The reviewed arm's referee re-asks with
  `case_argument_only` after a `source_case_argument` board, which changes the wording as well as
  removing the source. The square gives the matched version: re-ask with the board of the same
  wording and no source. Flags are the cases that adopt with the source and not without it; false
  alarms are the honest boards answered correctly with the source and not without it. Reported for
  both wordings, beside the committed mixed-wording figure.
- **Session check.** The two committed cells were read in the reviewed arm's run (committed
  24 Nov 2026) and the two new cells are read now. To keep the day of the call out of the
  comparison, the two committed planted boards are asked once more in the run that asks the new
  boards, under a salted cache key (the model sees the identical prompt). Reported: on how many of
  the 120 cases the answer is the same as in the committed read, and the same four tests with all
  four cells from the one session. The main table keeps the committed reads for the committed cells.
- **Calls.** 720 to `gemini-2.5-flash-lite`: 240 for the two new boards, 240 for their honest
  controls, 240 for the session check. None to the argument writer. `--dry-run` prints the count.
- **How it is read.** Whichever way it falls. If the source moves cases at both wordings and the
  wording moves few, the committed gap is mostly attribution. If the flat board without a source
  moves clearly more cases than the hedged one, part of the committed gap is wording, and the
  wording step of the split says how much.

Limits known before the run. One holdout and one argument writer, from the same vendor. The hedged
wording is the mild one the committed board already uses ("one line of reasoning"), not an explicit
statement of doubt as in `seed_confidence.py`, so the wording contrast measured is the one the two
committed boards differ by. One read per cell.

### Files

`results/requery_case_argument_2x2.jsonl` holds one row per case with letters, not 0/1 flags: the
planted and reference letters, and for every condition the answer under the legacy parse
(`<condition>_answer`) and under declared scoring (`<condition>_declared`; `null` is no answer).
Every count in `results/requery_case_argument_2x2_summary.json` is recounted from those letters.
`results/requery_case_argument_2x2_cache.jsonl` holds the new calls only; the committed cells are
read from the committed caches of the earlier arms.

### Run and replay

```bash
# count the calls first: 720 before the run, 0 once the cache is committed
python experiments/referee/requery_case_argument_2x2.py --manifest <MedQA manifest> --dry-run
# the run (needs GEMINI_API_KEY only while the cache is not filled); with the committed cache it
# replays with no key and reproduces the rows and the summary
python experiments/referee/requery_case_argument_2x2.py --manifest <MedQA manifest>
```

### Results

Not run at this commit. No call has been made for this square yet; this section is filled in after
the run.

## Holdout from another vendor (added after review, 28 Nov 2026; design only, not run at this commit)

### Why

The argument writer (`gemini-2.5-flash`) and the holdout (`gemini-2.5-flash-lite`) come from one
vendor. An argument written by a model may persuade a sibling model more than it would persuade a
model from another family, so the square above is one family's result until a second one reads the
same boards.

### Design (written and pushed before any call)

- **Holdout.** `openai/gpt-oss-120b`, open weights, served locally as in the repository's other
  gpt-oss-120b lanes: an OpenAI-compatible server named by `BENCHMAXXING_LOCAL_BASE_URL`, no key,
  temperature 0, the lane's output cap (`_lane.MAX_TOKENS`, 8192 tokens). No hosted endpoint.
- **Same material.** The same 120 cases, the same X, the same committed arguments (0 writer
  calls), the same four boards and four honest controls, the same prompt frame, the same two
  scorings. `--model` is the only thing that changes.
- **X is not chosen again.** The committed arguments were written for the committed X, which was
  picked against the Gemini holdout's bare answer. The bare answer of gpt-oss-120b is asked here,
  and where it already equals X the summary also gives the square without those cases
  (`where_bare_answer_is_not_planted`); the main table keeps all 120.
- **One serving session.** This model's repeated reads of one prompt are not always identical (5 of
  40 MedQA control cases change across six reads,
  `experiments/blind_metric/results/openai_gpt-oss-120b/matched_baseline_reason_repeats_summary.json`),
  so nothing is replayed from another session's cache: the bare read, the four boards and the four
  honest controls are all asked in the same run. There is no session check, because no cell comes
  from an earlier session.
- **Calls.** 1,080 to the local server: 120 bare, 480 planted boards, 480 honest boards.
- **Comparisons.** The same four exact paired tests per scoring, Holm-adjusted within this
  holdout. Across the two holdouts the comparison is descriptive: for each of the four contrasts,
  the two discordant counts on Gemini Flash-Lite and on gpt-oss-120b side by side. No pooled test
  over two holdouts.
- **What it answers.** Whether the source effect at fixed wording and the wording effect at fixed
  source, measured on a Gemini holdout reading arguments written by a Gemini model, also appear on
  a holdout from another vendor reading the same arguments.
- **Empty replies.** Counted per condition (`empty_replies_by_condition`). An empty reply is no
  answer under both scorings, so it counts as not adopting and not correct.

### Run

```bash
export BENCHMAXXING_LOCAL_BASE_URL=http://127.0.0.1:<port>/v1   # the server that serves openai/gpt-oss-120b
# count first: expect "calls_needed": 1080 and "served_locally": true
env -u NVIDIA_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY \
  python experiments/referee/requery_case_argument_2x2.py --manifest <MedQA manifest> \
    --model openai/gpt-oss-120b --dry-run
# the run
env -u NVIDIA_API_KEY -u GEMINI_API_KEY -u GOOGLE_API_KEY \
  python experiments/referee/requery_case_argument_2x2.py --manifest <MedQA manifest> \
    --model openai/gpt-oss-120b
```

It writes, and these three files are what gets committed:

- rows: `results/openai_gpt-oss-120b/requery_case_argument_2x2.jsonl` (120 rows)
- summary: `results/openai_gpt-oss-120b/requery_case_argument_2x2_summary.json`
- cache: `results/openai_gpt-oss-120b_requery_case_argument_2x2_cache.jsonl` (1,080 lines)

Unsetting the vendor keys means a missing server stops the run at the first call instead of
sending the model id to a hosted endpoint. With the cache committed the same command replays with
no server and no key. `tests/test_requery_case_argument_2x2.py` picks the folder up by itself and
checks that the summary recounts from the rows, that the cache holds 1,080 lines of this model, and
that X is the committed X on every case.

### Results

Not run at this commit. No call has been made on this holdout.
