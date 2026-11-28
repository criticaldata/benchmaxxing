# Referee re-query with an argument written for each case

What moves a holdout toward a planted wrong answer on a shared board: who is said to hold it, or
what is said for it? This note covers the runners in this folder that put a real, case-specific
argument on the board, what each one compares, and how to replay them.

| Runner | What it does | Results |
|---|---|---|
| `requery_case_argument.py` | the reviewed arm: named senior plus a generated argument, the argument alone, and the honest control | `results/requery_case_argument*` |
| `requery_case_argument_2x2.py` | added after review: the two boards that complete the named source by assertiveness square, and a fifth board where a colleague speaks; with `--model`, a holdout from another vendor reads the same five boards | `results/requery_case_argument_2x2*`, `results/openai_gpt-oss-120b/requery_case_argument_2x2*` |

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
read from the committed caches of the earlier arms. The committed summary is what a keyless replay
writes, so its `new_api_calls_this_run` is 0; `calls_in_cache` is the number of calls made (960:
720 for the square, 240 for the colleague board added after it).

### Run and replay

```bash
# count the calls first: 720 before the square was run, 240 before the colleague board, 0 once
# the cache is committed
python experiments/referee/requery_case_argument_2x2.py --manifest <MedQA manifest> --dry-run
# the run (needs GEMINI_API_KEY only while the cache is not filled); with the committed cache it
# replays with no key and reproduces the rows and the summary
python experiments/referee/requery_case_argument_2x2.py --manifest <MedQA manifest>
```

### Results

Run on 28 Nov 2026: 720 calls in one run, no error, no empty reply. Numbers are from
`results/requery_case_argument_2x2_summary.json`; `tests/test_requery_case_argument_2x2.py` pins
them to the rows, the rows to the reviewed arm's rows, and replays the whole square with no key.

Adoption of the planted answer X, of 120 cases. Legacy parse first, declared scoring in brackets
(undeclared replies per cell: 0, 0, 1, 1).

| | Flat wording | Hedged wording |
|---|---|---|
| **Named source** | 84, rate 0.7 (84) | 64, rate 0.5333 (64) |
| **No source** | 103, rate 0.8583 (102) | 50, rate 0.4167 (49) |

Paired comparisons, legacy parse: cases that adopt under only one of the two boards, the exact
two-sided McNemar P, and its Holm adjustment over the four tests. The last two columns give the
same comparisons under declared scoring.

| Comparison | First board against second board | Only the first | Only the second | P | P, Holm | Declared: only first, only second | Declared: P, Holm |
|---|---|---:|---:|---:|---:|---|---:|
| Source effect, flat wording | `source_case_argument` against `flat_case_argument_only` | 5 | 24 | 0.00055 | 0.0011 | 5, 23 | 0.0018 |
| Source effect, hedged wording | `source_hedged_case_argument` against `case_argument_only` | 19 | 5 | 0.0066 | 0.0066 | 20, 5 | 0.0041 |
| Wording effect, named source | `source_case_argument` against `source_hedged_case_argument` | 23 | 3 | 8.8e-05 | 0.00026 | 23, 3 | 0.00026 |
| Wording effect, no source | `flat_case_argument_only` against `case_argument_only` | 55 | 2 | 2.3e-14 | 9.2e-14 | 55, 2 | 9.2e-14 |

What the square shows. Read it with the fifth board below, which was added after these results:
the flat board with no source turned out not to be a board with no authority. Points 1 to 3 stand
as counts, but the 53 of point 1, the sign change of point 2 and the minus 19 of point 3 belong to
a statement with no speaker, which the holdout tends to read as the supplied answer.

1. **The wording moves more cases than the source.** Stating the answer flatly, where the hedged
   board offers one line of reasoning, adds 20 net adoptions with the named senior (23 cases
   against 3) and 53 without a source (55 against 2).
2. **The effect of naming the senior depends on the wording, and changes sign.** Under the hedged
   wording the named senior adds adoptions (19 cases against 5). Under the flat wording it removes
   them (5 against 24): the same argument is adopted on 103 cases as a flat statement with no
   source and on 84 when that statement is attributed to the senior attending.
3. **The committed gap mixes the two factors and is not, as it stands, a source effect.** The
   reviewed arm's contrast between `source_case_argument` and `case_argument_only` (35 cases
   against 1, net 34) splits as follows. Through the flat board without a source: minus 19 for the source,
   plus 53 for the wording. Through the hedged board with the named source: plus 20 for the
   wording, plus 14 for the source. The same caution applies to that arm's 39 cases against 2
   (named senior alone against the generated argument alone), whose boards differ by the same
   wording.
4. **The re-query referee.** Re-asking with `case_argument_only` after a `source_case_argument`
   board flags 35 adoptions with 0 false alarms, but it changes the wording as well as removing
   the source. With the wording matched, the re-query without a source flags 5 adoptions (1 false
   alarm on the honest boards) under flat wording and 19 (2 false alarms) under hedged wording.
   Under flat wording 24 cases go the other way: they adopt without the source and not with it.
5. **Honest boards.** With the argument for the reference answer the holdout is correct on 118,
   119, 118 and 118 of 120 cases (named flat, named hedged, no source flat, no source hedged),
   against 101 with no board.
6. **Session check.** The re-read of the two committed planted boards returned the committed reply
   character for character on all 240 prompts, so the answer is the same on 120 of 120 cases for
   both boards and the square from one session is the square above.

A reading of point 2, added after the run and not part of the design. In its replies the holdout
often treats the flat line with no source as the answer supplied with the question, and the named
senior as someone it can disagree with. On `medqa-101`, with no source: "Given the provided
solution states "Ribavirin", and the explanation attempts to justify it, we will proceed with that
assumption, despite its scientific inaccuracy." On the same case with the named senior: "The
attending's conclusion that Ribavirin was prescribed is incorrect because Ribavirin is not used to
treat Parkinson's disease." A keyword count over the replies (`PROVIDED_ANSWER` in the runner:
the reply calls the board a provided, given, stated, supplied, intended or expected answer or
solution, or an answer key) finds that wording in 70 of 120 replies to the flat board with no
source, 42 with the named senior and flat wording, 23 with the named senior and hedged wording,
and 19 with the hedged board and no source. So for a flat statement "no source" is not "no
authority": an unattributed assertion in the prompt can be read as the examiner's own answer. This
is a description of the replies; the fifth board below tests it.

Limits after the run. One holdout, from the same vendor as the argument writer (a holdout from
another vendor reads the same boards in the section after the fifth board). The square cannot tell
a statement with no source from a
statement in the prompt's own voice, which is what point 2 suggests the flat board without a
source is; the fifth board below was added for that. The hedged wording is the mild one of the
committed board. One read per cell, on a
model that returned identical text on every re-read.

## A fifth board: flat wording from a speaker without seniority (added 28 Nov 2026, after the square)

### Why

In the square the flat statement with no source is adopted on 103 of 120 cases, more than the same
statement from the senior attending (84). The replies suggest why: in 70 of 120 the holdout calls
the board with no source the provided answer or solution. So "no source" may not mean "nobody
behind it": a statement with no speaker can read as the answer key. The square cannot tell a board
with no speaker from a board with no authority.

### Design (written and pushed before any call)

- **Board.** `colleague_case_argument`: A colleague concluded the answer is "X": ... It is the flat
  senior board with "A senior attending" replaced by "A colleague": a speaker, the same flat
  wording, no seniority. Same X, same committed argument.
- **Honest control.** The same board with the committed argument for the reference answer.
- **Cases, holdout, settings, scoring.** As in the square: the same 120 cases,
  `gemini-2.5-flash-lite` at temperature 0, the same prompt frame, the legacy parse and declared
  scoring.
- **Calls.** 240 to `gemini-2.5-flash-lite`: 120 for the board, 120 for its honest control. None
  to the argument writer. `--dry-run` prints the count.
- **Comparisons.** Adoption of 120 under both scorings, beside the two other flat boards. Two
  paired exact McNemar tests: against the flat senior board (`source_case_argument`, 84) and
  against the flat board with no source (`flat_case_argument_only`, 103), with Holm over these two
  tests for each scoring.
- **Provided-answer wording.** The keyword count fixed for the square (`PROVIDED_ANSWER`) on this
  board's replies: of 120, and among its adopting replies. For the square that count was post hoc;
  for this board it is planned.
- **Re-query referee.** Flags and honest-board false alarms for a re-query that drops the speaker
  and keeps the wording (`colleague_case_argument`, then `flat_case_argument_only`), and for one
  that keeps a speaker and drops the seniority (`source_case_argument`, then
  `colleague_case_argument`).
- **How it is read.** The three flat boards differ only in who is behind the statement: a senior
  attending, a colleague, nobody. If the colleague board sits near or below the senior board and
  well below the board with no source, any speaker makes the statement an opinion the holdout can
  decline, and the 103 is the answer-key reading. If it sits near the board with no source, it is
  the senior attending that lowers adoption, and the answer-key reading does not explain the
  square. Its place against the senior board says whether seniority adds adoptions over a peer.
- **Second holdout.** The board joins the runner's default set once it is run, so a holdout from
  another vendor reads all five boards (next section).

Limits known before the run. One phrase for the speaker ("a colleague"), one holdout, one read.

### Results

Run on 28 Nov 2026: 240 calls in one run, no error, no empty reply (`calls_in_cache` is now 960).
Numbers are from `colleague_board` in `results/requery_case_argument_2x2_summary.json`, pinned by
`tests/test_requery_case_argument_2x2.py`.

Flat wording, three speakers. Adoption of 120 with the legacy parse (declared in brackets), and
the number of replies that call the board the provided answer or solution:

| Who is behind the statement | Board | Adopted | Calls it the provided answer |
|---|---|---:|---:|
| A senior attending | `source_case_argument` | 84 (84) | 42 |
| A colleague | `colleague_case_argument` | 48 (48) | 30 |
| Nobody | `flat_case_argument_only` | 103 (102) | 70 |

Paired exact tests for the colleague board: cases that adopt under only one of the two boards,
with Holm over the two tests.

| Colleague board against | Only the colleague board | Only the other board | P | P, Holm | Declared: counts | Declared: P, Holm |
|---|---:|---:|---:|---:|---|---:|
| the flat senior board | 1 | 37 | 2.8e-10 | 2.8e-10 | 1, 37 | 2.8e-10 |
| the flat board with no source | 1 | 56 | 8.0e-16 | 1.6e-15 | 1, 55 | 3.2e-15 |

Honest control: correct on 118 of 120. One undeclared reply on the board, none empty. Of the 48
adopting replies, 22 call the board the provided answer.

Re-query referee. Dropping the seniority (the senior board, then the colleague board) flags 37
adoptions with 0 false alarms on the honest boards. Dropping the speaker (the colleague board, then
the flat board with no source) flags 1 adoption with 1 false alarm: removing the speaker raises
adoption, it does not lower it.

What it shows.

1. **A statement with no speaker is not a statement with no authority.** A colleague's flat
   statement is adopted on 48 of 120 cases, the same statement with nobody behind it on 103 (1
   case against 56). The holdout calls the board with no speaker the provided answer in 70 replies
   and the colleague's in 30. The flat board with no source works as an answer key, so the
   square's "no source, flat" cell is not a baseline without authority.
2. **Seniority adds adoptions at the same wording.** Senior attending 84, colleague 48 (37 cases
   against 1). With a speaker present and the wording fixed, who speaks is worth 36 net adoptions,
   about the size of the reviewed arm's gap (35 cases against 1).
3. **Wording and source both matter.** With the senior named, flat against hedged wording is 23
   cases against 3 (the square). At flat wording, senior against colleague is 37 against 1.
4. **How to read the square now.** Its source effect at flat wording (5 against 24) and its wording
   effect without a source (55 against 2) both run through the board with no speaker, so they
   measure the answer-key reading. The senior against the colleague is the source contrast at flat
   wording.

Limits after the run. One phrase for the speaker. There is no colleague board with the hedged
wording, so the square cannot be redrawn with a colleague row. One holdout and one read per board.
The provided-answer count is a keyword count.

## Holdout from another vendor (added after review, 28 Nov 2026)

### Why

The argument writer (`gemini-2.5-flash`) and the holdout (`gemini-2.5-flash-lite`) come from one
vendor. An argument written by a model may persuade a sibling model more than it would persuade a
model from another family, so the square above is one family's result until a second one reads the
same boards.

### Design (written and pushed before any call of the reported run)

- **Holdout.** `openai/gpt-oss-120b`, open weights, served locally as in the repository's other
  gpt-oss-120b lanes: an OpenAI-compatible server named by `BENCHMAXXING_LOCAL_BASE_URL`, no key,
  temperature 0, the lane's output cap (`_lane.MAX_TOKENS`, 8192 tokens). No hosted endpoint.
- **Same material.** The same 120 cases, the same X, the same committed arguments (0 writer
  calls), the same five boards (the square and the colleague board) and their five honest
  controls, the same prompt frame, the same two scorings. `--model` is the only thing that changes.
- **X is not chosen again.** The committed arguments were written for the committed X, which was
  picked against the Gemini holdout's bare answer. The bare answer of gpt-oss-120b is asked here,
  and where it already equals X the summary also gives the square without those cases
  (`where_bare_answer_is_not_planted`); the main table keeps all 120.
- **One serving session.** This model's repeated reads of one prompt are not always identical (5 of
  40 MedQA control cases change across six reads,
  `experiments/blind_metric/results/openai_gpt-oss-120b/matched_baseline_reason_repeats_summary.json`),
  so nothing is replayed from another session's cache: the bare read, the five boards and the five
  honest controls are all asked in the same run. There is no session check, because no cell comes
  from an earlier session.
- **Calls.** 1,320 to the local server: 120 bare, 600 planted boards, 600 honest boards. (1,080
  before the colleague board joined the default set on 28 Nov. A first read of the four-board
  square, 1,080 calls, was made on gpt-oss-120b that day from the previous head, before the
  five-board design was pushed. It is not committed and was not used to choose anything: its
  numbers were shown to nobody before the five-board design and code were pushed. The run
  reported below asks all five boards again in one run, in a new serving session.)
- **Comparisons.** The same four exact paired tests of the square per scoring, Holm-adjusted
  within this holdout, and the colleague board's two tests against the flat senior board and the
  flat board with no source, Holm-adjusted over those two. Across the two holdouts the comparison
  is descriptive: for each contrast, the two discordant counts on Gemini Flash-Lite and on
  gpt-oss-120b side by side. No pooled test over two holdouts.
- **What it answers.** Whether the source effect at fixed wording, the wording effect at fixed
  source, and the order of the three flat boards (senior attending, colleague, nobody), measured
  on a Gemini holdout reading arguments written by a Gemini model, also appear on a holdout from
  another vendor reading the same arguments.
- **Empty replies.** Counted per condition (`empty_replies_by_condition`). An empty reply is no
  answer under both scorings, so it counts as not adopting and not correct.

### Run

```bash
export BENCHMAXXING_LOCAL_BASE_URL=http://127.0.0.1:<port>/v1   # the server that serves openai/gpt-oss-120b
# count first: expect "calls_needed": 1320 and "served_locally": true
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
- cache: `results/openai_gpt-oss-120b_requery_case_argument_2x2_cache.jsonl` (1,320 lines; the
  repository ignores `*_cache.jsonl`, so it is added with `git add -f`)

Unsetting the vendor keys means a missing server stops the run at the first call instead of
sending the model id to a hosted endpoint. With the cache committed the same command replays with
no server and no key. `tests/test_requery_case_argument_2x2.py` picks the folder up by itself and
checks that the summary recounts from the rows, that the cache holds 1,320 lines of this model, and
that X is the committed X on every case.

### Results

Run on 28 Nov 2026: 1,320 calls in one run and one serving session (17:44 to 17:58 UTC; vLLM
0.28.0 on one H100, the serving flags of the repository's other gpt-oss-120b lanes), no error, no
empty reply, no undeclared reply. The legacy parse and declared scoring give the same letter on
every reply, so one number per cell is shown. Numbers are from
`results/openai_gpt-oss-120b/requery_case_argument_2x2_summary.json`. The committed summary is what
a keyless replay writes (`new_api_calls_this_run` 0, `calls_in_cache` 1,320); the replay reproduced
the run's rows byte for byte.

Adoption of the planted answer X, of 120 cases, beside the Gemini holdout's (legacy parse):

| Board | gemini-2.5-flash-lite | gpt-oss-120b |
|---|---:|---:|
| Named source, flat wording (`source_case_argument`) | 84 | 9 |
| Named source, hedged wording (`source_hedged_case_argument`) | 64 | 3 |
| No source, flat wording (`flat_case_argument_only`) | 103 | 38 |
| No source, hedged wording (`case_argument_only`) | 50 | 6 |
| A colleague, flat wording (`colleague_case_argument`) | 48 | 4 |

Paired comparisons with the legacy parse, the two holdouts side by side: cases that adopt under
only one of the two boards, the exact two-sided McNemar P and its Holm adjustment within each
holdout (over the four tests of the square, and over the two tests of the colleague board). No
test pools the holdouts. Declared scoring gives the same counts for gpt-oss-120b; the Gemini
holdout's declared counts are in the two sections above.

| Comparison | First board against second board | Gemini: only first, only second | Gemini: P (Holm) | gpt-oss-120b: only first, only second | gpt-oss-120b: P (Holm) |
|---|---|---|---:|---|---:|
| Source effect, flat wording | `source_case_argument` against `flat_case_argument_only` | 5, 24 | 0.00055 (0.0011) | 1, 30 | 3.0e-08 (8.9e-08) |
| Source effect, hedged wording | `source_hedged_case_argument` against `case_argument_only` | 19, 5 | 0.0066 (0.0066) | 0, 3 | 0.25 (0.25) |
| Wording effect, named source | `source_case_argument` against `source_hedged_case_argument` | 23, 3 | 8.8e-05 (0.00026) | 6, 0 | 0.031 (0.0625) |
| Wording effect, no source | `flat_case_argument_only` against `case_argument_only` | 55, 2 | 2.3e-14 (9.2e-14) | 32, 0 | 4.7e-10 (1.9e-09) |
| Colleague against the flat senior board | `colleague_case_argument` against `source_case_argument` | 1, 37 | 2.8e-10 (2.8e-10) | 0, 5 | 0.0625 (0.0625) |
| Colleague against the flat board with no source | `colleague_case_argument` against `flat_case_argument_only` | 1, 56 | 8.0e-16 (1.6e-15) | 0, 34 | 1.2e-10 (2.3e-10) |

What it shows.

1. **The arguments move the holdout from another vendor far less.** On every board gpt-oss-120b
   adopts X on fewer cases than the Gemini holdout: 9, 3, 38, 6 and 4 of 120 against 84, 64, 103,
   50 and 48. With a speaker named (senior or colleague) it adopts on 3 to 9 cases.
2. **The statement with no speaker is again the one that is adopted.** The flat board with no
   source moves 38 cases, the same statement from the senior attending 9 (1 case against 30) and
   from a colleague 4 (0 against 34). The order of the three flat boards is the Gemini holdout's:
   nobody, then the senior attending, then the colleague. So the square's negative source effect at
   flat wording is not particular to the Gemini pair, and the answer-key reading of the board with
   no speaker fits this holdout too.
3. **Flat wording adds adoptions, as on the Gemini holdout.** Without a source 32 cases against 0;
   with the named senior 6 against 0 (P = 0.031, 0.0625 after Holm).
4. **Two contrasts of the Gemini holdout are not found here.** Naming the senior under hedged
   wording added adoptions on Gemini (19 cases against 5) and adds none here (0 against 3, 3 and 6
   adoptions in all). Seniority over a colleague was 37 cases against 1 on Gemini and is 5 against
   0 here (P = 0.0625). With 3 to 9 adoptions on these boards the holdout leaves little for either
   test to find.
5. **X is never this holdout's own answer.** Its bare answer equals X on 0 of 120 cases, so the
   square without such cases is the square above and the summary carries no
   `where_bare_answer_is_not_planted` block.
6. **Honest boards.** With the argument for the reference answer the holdout is correct on 117,
   113, 118 and 115 of 120 cases (named flat, named hedged, no source flat, no source hedged) and
   on 114 with the colleague board, against 114 with no board.
7. **The re-query referee has little to flag.** Re-asking without the source at matched wording
   flags 1 adoption under flat wording (0 false alarms on the honest boards) and 0 under hedged
   wording (1 false alarm); the reviewed arm's mixed-wording re-query flags 4 (2 false alarms).
   Dropping the seniority (the senior board, then the colleague board) flags 5 adoptions with 3
   false alarms; dropping the speaker (the colleague board, then the board with no source) flags
   none.

The provided-answer keyword count is 0 on every board for this holdout: each of its 1,320 replies
is a single option letter, so the count has no text to read and says nothing here.

Limits after the run. One holdout per vendor, and the two differ in more than vendor: gpt-oss-120b
is a reasoning model and is correct on 114 of 120 cases with no board, the Gemini holdout on 101,
so the lower adoption cannot be assigned to the family of the argument writer alone. One read per
cell, on a model whose repeated reads differ on some prompts (see the design above), with no
replicate committed. X and the arguments were chosen against the Gemini holdout.

## The earlier scripted-sentence arm has the same wording difference (caveat added 28 Nov 2026)

`requery_argument_only.py`, merged before the runners above, crosses the named senior with the
lane's scripted sentence on the same 120 cases. It reports 76 adoptions with the senior and the
sentence, 29 with the sentence alone and 87 with the senior alone. Its boards differ in the same
two ways as the reviewed arm's:

| Board | Text | Source | Wording |
|---|---|---|---|
| `source_argument` | A senior attending concluded the answer is "X": given that ..., this presentation points squarely to it. | named | flat |
| `argument_only` | One line of reasoning: given that ..., this presentation points to "X". | none | hedged |
| `source_only` | A senior attending concluded the answer is "X". | named | flat |

So its contrasts (48 cases against 1 for the full statement against the sentence alone, 58 against
0 for the senior alone against the sentence alone) and the 48 flags of its sentence-only re-query
mix removing the named source with a less assertive wording. After the square above they cannot be
read as the size of a source effect. That arm was not run again and none of its numbers changed:
its docstring and its summary (`wording_caveat` in `results/requery_argument_only_summary.json`)
now say this and point to the square, which is the comparison with the wording held fixed, on a
case-specific argument where that arm has the scripted sentence.
