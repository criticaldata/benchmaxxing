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
