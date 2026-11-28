# Paraphrase noise band of the matched control

`paraphrase_band.py` asks how often the matched control answer changes when the case is reworded
with the same facts. Read 0 is the committed control prompt, replayed from the lane's committed
cache. Reads 1 to 5 replace the question text with one of five committed paraphrases (written once
by `gemini-2.5-flash` at temperature 0, `paraphrases.jsonl` in each lane); the options, the
instruction line and the decoy are unchanged. Each lane and lineage folder holds
`paraphrase_band.jsonl` (one row per case) and `paraphrase_band_summary.json`.

Lanes: MedQA (`experiments/blind_metric/results`), MedMCQA (`experiments/medmcqa/results`) and
SUPPORT2 (`experiments/support2/results`). Lineages: Gemini 2.5 Flash-Lite in the lane folder, and
gpt-oss-120b, Qwen2.5-VL-72B and the FP8 nemotron, all served locally, in a subfolder each.

## Bands without read 0 (added after review, 28 Nov 2026)

Read 0 comes from the committed control cache. For a locally served lineage that cache was filled
in an earlier serving session than reads 1 to 5, and two of these lineages do not always repeat an
answer. So for them a band over reads 0 to 5 mixes two things: the rewording, and the change of
session between read 0 and the other reads. The committed same-prompt band
(`matched_baseline_reason_repeats`) has the same structure: its read 0 is the main control run and
its reads 1 to 5 are the repeats.

`without_read_0` in every `paraphrase_band_summary.json` drops read 0 on both sides:

- `paraphrase_reads`: the band over reads 1 to 5 of the paraphrase band (five rewordings);
- `same_session_repeats`: the band over reads 1 to 5 of the committed same-prompt band (the same
  prompt asked five more times), on the cases of the paraphrase band.

That is five reads against five reads, without the replayed read, on the same cases. It is
computed from the committed rows with no call (`paraphrase_band.band_of_reads`).

| Cohort | Lineage | Cases | Any disagreement, reads 0 to 5: paraphrases | Any disagreement, reads 1 to 5: paraphrases | Any disagreement, reads 1 to 5: same-prompt repeats | Mean pairwise, reads 1 to 5: paraphrases / repeats | Flip band, reads 1 to 5: paraphrases | Flip band, reads 1 to 5: repeats | Decoy band, reads 1 to 5: paraphrases | Decoy band, reads 1 to 5: repeats |
|---|---|---:|---:|---|---|---|---|---|---|---|
| MedQA | Gemini Flash-Lite | 40 | 12 | 12 (0.3, Wilson 0.1807 to 0.4543) | 0 (0, Wilson 0 to 0.0876) | 0.1525 / 0 | 0.175 to 0.275 | 0.2 to 0.2 | 0.075 to 0.125 | 0.075 to 0.075 |
| MedQA | gpt-oss-120b | 40 | 8 | 7 (0.175, Wilson 0.0875 to 0.3195) | 3 (0.075, Wilson 0.0258 to 0.1986) | 0.085 / 0.045 | 0 to 0.075 | 0.025 to 0.1 | 0 to 0 | 0 to 0.025 |
| MedQA | Qwen2.5-VL-72B | 40 | 10 | 9 (0.225, Wilson 0.1232 to 0.375) | 0 (0, Wilson 0 to 0.0876) | 0.1325 / 0 | 0.15 to 0.2 | 0.125 to 0.125 | 0 to 0.025 | 0 to 0 |
| MedQA | nemotron FP8 | 40 | 7 | 7 (0.175, Wilson 0.0875 to 0.3195) | 5 (0.125, Wilson 0.0546 to 0.2611) | 0.095 / 0.05 | 0.05 to 0.1 | 0.025 to 0.075 | 0 to 0.025 | 0 to 0 |
| MedMCQA | Gemini Flash-Lite | 40 | 18 | 17 (0.425, Wilson 0.2851 to 0.5781) | 0 (0, Wilson 0 to 0.0876) | 0.23 / 0 | 0.125 to 0.25 | 0.15 to 0.15 | 0.05 to 0.125 | 0.075 to 0.075 |
| MedMCQA | gpt-oss-120b | 40 | 11 | 8 (0.2, Wilson 0.105 to 0.3476) | 5 (0.125, Wilson 0.0546 to 0.2611) | 0.095 / 0.065 | 0.1 to 0.15 | 0.1 to 0.175 | 0 to 0.05 | 0.025 to 0.025 |
| MedMCQA | Qwen2.5-VL-72B | 40 | 12 | 12 (0.3, Wilson 0.1807 to 0.4543) | 0 (0, Wilson 0 to 0.0876) | 0.1625 / 0 | 0.225 to 0.3 | 0.225 to 0.225 | 0.025 to 0.05 | 0 to 0 |
| MedMCQA | nemotron FP8 | 40 | 12 | 11 (0.275, Wilson 0.1611 to 0.4284) | 8 (0.2, Wilson 0.105 to 0.3476) | 0.155 / 0.11 | 0.1 to 0.225 | 0.15 to 0.175 | 0 to 0.05 | 0 to 0.025 |
| SUPPORT2 | Gemini Flash-Lite | 31 | 4 | 4 (0.129, Wilson 0.0513 to 0.2885) | 0 (0, Wilson 0 to 0.1103) | 0.071 / 0 | 0.0323 to 0.129 | 0 to 0 | 0.0323 to 0.129 | 0 to 0 |
| SUPPORT2 | gpt-oss-120b | 26 | 1 | 1 (0.0385, Wilson 0.0068 to 0.1889) | 1 (0.0385, Wilson 0.0068 to 0.1889) | 0.0231 / 0.0154 | 0.0385 to 0.0769 | 0.0385 to 0.0769 | 0.0385 to 0.0769 | 0.0385 to 0.0769 |
| SUPPORT2 | Qwen2.5-VL-72B | 31 | 6 | 6 (0.1935, Wilson 0.0919 to 0.3628) | 0 (0, Wilson 0 to 0.1103) | 0.0968 / 0 | 0.0323 to 0.0968 | 0.0968 to 0.0968 | 0.0323 to 0.0968 | 0.0968 to 0.0968 |
| SUPPORT2 | nemotron FP8 | 26 | 2 | 2 (0.0769, Wilson 0.0214 to 0.2414) | 0 (0, Wilson 0 to 0.1287) | 0.0385 / 0 | 0 to 0.0769 | 0 to 0 | 0 to 0.0769 | 0 to 0 |

What it shows.

- Read 0 did carry part of the count on the two lineages whose answers are not stable. On
  gpt-oss-120b the cases with any disagreement go from 8, 11 and 1 over reads 0 to 5 to 7, 8 and 1
  over the paraphrase reads alone (MedQA, MedMCQA, SUPPORT2): on 1 MedQA case and 3 MedMCQA cases
  the five paraphrase reads agree with each other and only the replayed read 0 differs. On the FP8
  nemotron they go from 7, 12 and 2 to 7, 11 and 2.
- The same holds for the same-prompt band. Over its repeats alone, gpt-oss-120b disagrees on 3, 5
  and 1 cases and the FP8 nemotron on 5, 8 and 0, where the committed six-read band gave 5, 5 and 2
  of 30 and 6, 10 and 2 of 33.
- With read 0 out of both, the paraphrase reads still disagree on at least as many cases as the
  same-prompt repeats on every lane: gpt-oss-120b 7 against 3, 8 against 5, 1 against 1; FP8
  nemotron 7 against 5, 11 against 8, 2 against 0. The Wilson intervals of the two counts overlap
  on every lane for both lineages, as they did over reads 0 to 5.
- Gemini Flash-Lite and Qwen2.5-VL-72B repeat every same-prompt read exactly, so for them the
  comparison is unchanged: 12, 17 and 4 (Gemini) and 9, 12 and 6 (Qwen) cases against 0.
- SUPPORT2 on the local lineages: the paraphrase band covers only the band cases that have a
  committed paraphrase (26 of 30 for gpt-oss-120b, 31 of 35 for Qwen, 26 of 33 for the FP8
  nemotron), and the committed same-prompt band covers all of them. `same_session_repeats` uses the
  shared cases, which is why its counts differ from the committed band's: the two nemotron cases
  that disagree in the committed band (`support2-14`, `support2-17`) and one of the two
  gpt-oss-120b cases (`support2-14`) have no paraphrase.

## Semantic check of the paraphrases (added after review, 28 Nov 2026)

### Why

`paraphrase_band.py` checks each paraphrase only for the numbers of the original. A paraphrase
that dropped a negation ("all of the following except" reworded as "which of the following")
would pass that check, change the right answer, and be counted as a disagreement that is not
noise. That would widen the band and make a real cue effect look like rewording.

### Design (written and pushed before any call)

`paraphrase_semantic_check.py` puts two checks on every committed paraphrase: 200 for MedQA, 200
for MedMCQA and 155 for SUPPORT2, 555 in all. The local lineages read the same paraphrases, so one
check per lane covers every lineage.

- **Rule check, keyless.** It reads the question part of each text: the sentences that end with a
  question mark (each with the sentence before it when it has fewer than five words), or the last
  sentence when there is no question mark. A marker of negation or exception must be present in
  the paraphrase's question part if and only if one is present in the original's. A marker that is
  dropped fails, and so does one that is added. The rule compares presence, not the word, because
  a faithful rewording may turn "except" into "with one exception" or "is not true". The markers:
  - negation: not, no, never, none, neither, nor, without, cannot, and any contraction in n't;
  - exception: except, excepting, exception(s), excluding, exclude(s), excluded, other than,
    apart from, aside from, save for, all but;
  - least: least, lowest, fewest;
  - false: false, incorrect(ly), untrue, wrong(ly), inaccurate, unlikely, inappropriate, uncommon,
    contraindicated.
- **Judge.** `gemini-2.5-flash` at temperature 0, one call per paraphrase. It is shown the
  original, the paraphrase and the answer options, and asked whether the paraphrase asks the same
  question: every fact, value and unit kept and none added or changed, the same thing asked
  including any negation or exception, so that each option is right or wrong exactly as for the
  original. It replies yes or no and one line of reason. A reply with no verdict counts as not
  confirmed.
- **Failure.** A paraphrase fails when either check fails. Each lane's failures are listed in
  `paraphrase_semantic_failures.jsonl` with the paraphrase text, the markers found and the judge's
  reason; `paraphrase_semantic_check.jsonl` has one row for every paraphrase.
- **Bands.** Read i of a case is its paraphrase i, so leaving a paraphrase out removes that read
  from the case. For every lane and lineage `paraphrase_band_semantic_summary.json` gives the
  committed band recomputed from the same rows (`all_paraphrases`) and the band with the failed
  paraphrases left out (`failed_left_out`): read 0 is always kept, each read's rate is over the
  cases that keep it, and disagreement is over the cases that keep at least two reads. It also
  gives the band with only the rule failures or only the judge failures left out, and the band on
  the cases whose five paraphrases all pass, as the existing `fidelity_clean_subset` does for the
  numbers check. Each is given over reads 0 to 5 and over reads 1 to 5.
- **What is compared.** For each lane and lineage, the band with all paraphrases against the band
  with the failed ones left out: cases with any disagreement, mean pairwise disagreement, flip
  band and decoy band. No test, since it is the same rows with some reads removed.
- **How it is read.** If few paraphrases fail and the bands barely move, the paraphrase band
  stands as a band of rewording. If the failed paraphrases carry a large share of the
  disagreement, the band was partly measuring changed questions, and the version with them left
  out is the one to quote.
- **Calls.** 555 to `gemini-2.5-flash`. `--dry-run` prints the count for a lane.

Limits known before the run. The judge is the model that wrote the paraphrases, so it may share
their blind spots. The rule check does not depend on it, but it is lexical: it reads only the
question part, it can fail a faithful rewording that carries the negation in a word outside the
list, and it can pass a paraphrase that keeps one marker and drops another. The marker list was
fixed before the judge was called and before any band was recomputed.

### Judge control (added after the 555 judge calls, written and pushed before its own calls)

The 555 judge calls came back with one no. A judge that says yes to nearly everything shows little
unless it can also say no, and the judge here is the model that wrote the paraphrases. So the judge
is also shown rewordings that are known to ask something else.

- **Items.** Every committed paraphrase whose question part kept a marker of negation or exception
  (the rule check's "kept" rows), turned around by rule (`reverse_question`): "not" and "n't" are
  removed, "false", "incorrect", "least", "unlikely" and the other words of `OPPOSITE` are swapped
  for their opposites, "with one exception" becomes "with no exception", and an "except" that ends
  the question is dropped. The first marker of the question part that the table can reverse
  decides; a paraphrase with none is left out and counted. Nothing outside the question part is
  touched.
- **Call.** The same judge, the same prompt, the original against the reversed paraphrase: 38
  calls (5 MedQA, 33 MedMCQA, none for SUPPORT2, whose questions carry no marker).
- **Expected.** A judge that works says no to all of them. Reported per lane: how many it refused,
  how many the keyless rule check also refuses, and each reversed question with the verdict
  (`paraphrase_semantic_judge_control.jsonl`, `judge_control` in the summary).
- **Limits.** The reversals are built by rule, so they are cruder than a paraphrase that quietly
  loses a negation, and they cover only questions that carry a marker. The control says the judge
  can refuse a reversed question; it does not measure every way a paraphrase could drift.

### Run and replay

```bash
# per lane; --dry-run first prints "calls_needed" (200, 200, 155 before the run, 0 after)
python experiments/matched_baseline/paraphrase_semantic_check.py \
    --manifest <MedQA manifest> --results experiments/blind_metric/results
python experiments/matched_baseline/paraphrase_semantic_check.py \
    --manifest <MedMCQA manifest> --results experiments/medmcqa/results
python experiments/matched_baseline/paraphrase_semantic_check.py \
    --manifest experiments/support2/results/support2_manifest.csv --results experiments/support2/results
```

The judge calls are cached in each lane's `paraphrase_semantic_judge_cache.jsonl`, the control's in
`paraphrase_semantic_judge_control_cache.jsonl`. With the caches committed the same commands replay
with no key.

### Results

Run on 28 Nov 2026: 555 judge calls and 38 control calls, no error, no reply without a verdict.
Numbers are from each lane's `paraphrase_band_semantic_summary.json`; the committed summaries are
what a keyless replay writes (`new_api_calls_this_run` 0, `calls_in_cache` the calls made), and
`tests/test_paraphrase_semantic_check.py` pins them to the rows and replays each lane.

**How many fail.** 10 of the 555 paraphrases fail a check, on 6 of the 111 cases: 9 fail the rule
check, 1 fails the judge, none fails both.

| Cohort | Paraphrases | Fail the rule | Fail the judge | Fail either | Cases touched |
|---|---:|---|---:|---:|---:|
| MedQA | 200 | 1 (a marker added) | 0 | 1 | 1 |
| MedMCQA | 200 | 8 (6 dropped, 2 added) | 1 | 9 | 5 |
| SUPPORT2 | 155 | 0 | 0 | 0 | 0 |

**What the ten are.** Each lane's `paraphrase_semantic_failures.jsonl` lists them with their text.
Read one by one, none changes what is asked:

| Cohort | Case, paraphrase | Failed | What happened |
|---|---|---|---|
| MedQA | `medqa-7`, 2 | rule, added | "In addition to aspirin" is reworded "Aside from aspirin"; "aside from" is on the exception list |
| MedMCQA | `67add9f9`, 3 | rule, dropped | "not involved" is reworded "absent from involvement" |
| MedMCQA | `b64a9cd7`, 1 to 4 | rule, dropped | the stem has no question mark, so its last sentence, which holds the fact "She does not want another down syndrome kid", is read as the question; the paraphrases say "avoid" or "prevent", or move the fact out of the question |
| MedMCQA | `b944ada9`, 5 | rule, dropped | "not true" is reworded "a misstatement" |
| MedMCQA | `f0f91922`, 3 and 5 | rule, added | the fact "received food supplements or not" moves into the question sentence |
| MedMCQA | `e845e625`, 5 | judge | the stem's "cailage" is spelled "cartilage" in the paraphrase |

So the nine rule failures are the two limits stated before the run (a negation carried by a word
outside the list, and a negated fact inside or outside the question part), and the judge's one
refusal is a corrected spelling. No paraphrase was found that drops a negation or an exception.

**The bands with the ten left out.** They are left out all the same, whatever the reading above.

| Cohort | Lineage | Paraphrase reads left out | Any disagreement, reads 0 to 5: all / failed left out | Any disagreement, reads 1 to 5: all / failed left out | Flip band, reads 0 to 5: all | Flip band, reads 0 to 5: failed left out | Decoy band, reads 0 to 5: all | Decoy band, reads 0 to 5: failed left out |
|---|---|---:|---|---|---|---|---|---|
| MedQA | Gemini Flash-Lite | 1 of 200 | 12 of 40 / 12 of 40 | 12 of 40 / 12 of 40 | 0.175 to 0.275 | 0.175 to 0.275 | 0.075 to 0.125 | 0.075 to 0.125 |
| MedQA | gpt-oss-120b | 1 of 200 | 8 of 40 / 8 of 40 | 7 of 40 / 7 of 40 | 0 to 0.075 | 0 to 0.075 | 0 to 0.025 | 0 to 0.025 |
| MedQA | Qwen2.5-VL-72B | 1 of 200 | 10 of 40 / 10 of 40 | 9 of 40 / 9 of 40 | 0.125 to 0.2 | 0.125 to 0.2 | 0 to 0.025 | 0 to 0.0256 |
| MedQA | nemotron FP8 | 1 of 200 | 7 of 40 / 7 of 40 | 7 of 40 / 7 of 40 | 0.05 to 0.1 | 0.05 to 0.1026 | 0 to 0.025 | 0 to 0.025 |
| MedMCQA | Gemini Flash-Lite | 9 of 200 | 18 of 40 / 18 of 40 | 17 of 40 / 17 of 39 | 0.125 to 0.25 | 0.1282 to 0.2564 | 0.05 to 0.125 | 0.0541 to 0.1282 |
| MedMCQA | gpt-oss-120b | 9 of 200 | 11 of 40 / 11 of 40 | 8 of 40 / 7 of 39 | 0.075 to 0.15 | 0.075 to 0.1538 | 0 to 0.05 | 0 to 0.0541 |
| MedMCQA | Qwen2.5-VL-72B | 9 of 200 | 12 of 40 / 12 of 40 | 12 of 40 / 12 of 39 | 0.225 to 0.3 | 0.225 to 0.3243 | 0 to 0.05 | 0 to 0.0541 |
| MedMCQA | nemotron FP8 | 9 of 200 | 12 of 40 / 12 of 40 | 11 of 40 / 11 of 39 | 0.1 to 0.225 | 0.1026 to 0.2308 | 0 to 0.05 | 0 to 0.0541 |
| SUPPORT2 | Gemini Flash-Lite | 0 of 155 | 4 of 31 / 4 of 31 | 4 of 31 / 4 of 31 | 0 to 0.129 | 0 to 0.129 | 0 to 0.129 | 0 to 0.129 |
| SUPPORT2 | gpt-oss-120b | 0 of 130 | 1 of 26 / 1 of 26 | 1 of 26 / 1 of 26 | 0.0385 to 0.0769 | 0.0385 to 0.0769 | 0.0385 to 0.0769 | 0.0385 to 0.0769 |
| SUPPORT2 | Qwen2.5-VL-72B | 0 of 155 | 6 of 31 / 6 of 31 | 6 of 31 / 6 of 31 | 0.0323 to 0.0968 | 0.0323 to 0.0968 | 0.0323 to 0.0968 | 0.0323 to 0.0968 |
| SUPPORT2 | nemotron FP8 | 0 of 130 | 2 of 26 / 2 of 26 | 2 of 26 / 2 of 26 | 0 to 0.0769 | 0 to 0.0769 | 0 to 0.0769 | 0 to 0.0769 |

Over reads 0 to 5 the number of cases with any disagreement is the same on every lane and lineage.
Over reads 1 to 5 it changes once: on MedMCQA the case `b64a9cd7` keeps a single paraphrase read
and leaves the count, which takes gpt-oss-120b from 8 of 40 to 7 of 39. No edge of a flip or decoy
band moves by more than 0.0243 (Qwen2.5-VL-72B on MedMCQA, 0.3 to 0.3243, a smaller denominator on
one read), and the mean pairwise disagreement by no more than 0.0085. The summary also holds the
bands with only the rule failures or only the judge failures left out, and on the cases whose five
paraphrases all pass (39 MedQA cases, 35 MedMCQA cases, every SUPPORT2 case).

**Judge control.** 38 paraphrases whose question kept a marker could be turned around by rule
(11 more carried a marker the table does not reverse).

| Cohort | Kept a marker | Reversed | Judge refused | Judge accepted | Rule also refused | Refused by judge or rule |
|---|---:|---:|---:|---:|---:|---:|
| MedQA | 10 | 5 | 5 | 0 | 5 | 5 |
| MedMCQA | 39 | 33 | 32 | 1 | 30 | 32 |
| SUPPORT2 | 0 | 0 | 0 | 0 | 0 | 0 |

The judge refused 37 of the 38 reversed questions, with reasons such as "The original asks for the
false statement, while the reworded asks for the true statement." The one it accepted is "all
statements are accurate with no exception" for "all are true except", which it read as still
asking for the exception. That reversal keeps a marker, so the rule check does not see it either:
the three "with no exception" reversals are the ones the rule passes, and the judge refused the
other two.

**What this supports.** The paraphrase bands were not widened by paraphrases that ask something
else: the checks find none, and leaving out every flagged paraphrase leaves the bands where they
were. The judge can refuse a reversed question (37 of 38), and the rule check catches a removed
negation without it (35 of 38).

Limits after the run. The judge is the model that wrote the paraphrases. The control uses crude,
rule-built reversals of 38 of the 49 paraphrases whose question carries a marker; it does not show
what the judge does with a subtler drift, or with a changed fact that is not a number (numbers are
covered by the committed fidelity check). One reversal in 38 escaped both checks.
