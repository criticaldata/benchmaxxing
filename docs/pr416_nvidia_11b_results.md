# PR #416: final NVIDIA NIM results and model fallback

## Final status

The original 90B target could not be completed with the `N_2` NVIDIA credential. The
11B model was run as an operational fallback after confirming that the same credential
could return successful responses for it. This change is a fallback for availability;
it does not claim that Llama 3.2 11B and 90B are equivalent models.

## Why the model changed

### Llama 3.2 90B Vision

The 90B preflight never produced a model response. The N2 watchdog recorded 23
consecutive preflight timeouts between `2026-09-26T04:18:04Z` and
`2026-09-26T12:46:33Z`:

- HTTP status observed: `0` (no HTTP response received)
- Error: `timeout`
- Target block: 20 cases
- Cache rows: 0
- Cases started: 0

The older direct runner log also ends with `openai.APITimeoutError` after its three
request attempts. Therefore, the 90B run was stopped before any case was counted as
complete. Evidence: `experiments/chexpert/results/api_watchdog_n2_incremental.jsonl`
and `experiments/chexpert/results/phase2_n100/run_n2.log`.

### Llama 3.2 11B Vision

The same N2 credential returned HTTP 200 during the 11B run. The incremental runner
completed the staged targets `20`, `40`, `60`, `80`, `100`, `300`, and `600`. The
initial transient runner failure was recovered without counting a partial checkpoint.

The final 600-case checkpoint contains 600 rows for each of the three phases and has
zero unparseable responses. The final expansion from the existing 300-case state
added 900 new API calls; the complete cache contains 1,800 responses (600 cases x 3
phases).

### Microsoft Phi-3 Vision

The requested model identifier `microsoft/phi-3-vision-128k-instruct` was also not
available through the tested NVIDIA endpoint: the availability probe returned HTTP
`404`. It was therefore excluded from the comparison and produced no usable
experimental results. This is an endpoint/model availability failure, not evidence
about the model's image quality.

## Llama 3.2 11B results (`n=600`)

| Metric | Result | Count |
|---|---:|---:|
| Baseline decoy uptake | 0.0% | 0 / 600 |
| Blind decoy uptake | 80.5% | 483 / 600 |
| Test-aware decoy uptake | 52.5% | 315 / 600 |
| Test-awareness suppression | 28.0 percentage points | 168 / 600 |
| Silent drift among drifted cases | 100.0% | 483 / 483 |
| Unparseable responses | 0.0% | 0 / 1,800 |

The baseline is zero by construction: each decoy is defined as the opposite of the
model's baseline answer. Lower blind and test-aware uptake is better; higher
blind-minus-aware suppression means the warning helped more. These metrics measure
robustness to the synthetic decoy/rubric prompt, not clinical diagnostic accuracy.

## Gemini comparison

The existing Gemini CheXpert result covers only 35 cases. On the same 35 case IDs,
Llama 11B had 65.7% blind uptake and 42.9% test-aware uptake, while Gemini 2.5 Flash
had 85.7% and 17.1%, respectively. Gemini showed stronger suppression on this small
matched subset (68.6 vs. 22.9 percentage points). This is context only; a definitive
model comparison requires running Gemini on the same 600-case protocol.

## Reproducibility artifacts

- Summary: `experiments/chexpert/results/phase2_11b/phase2_n600/meta_llama-3.2-11b-vision-instruct/imaging_blind_metric_summary.json`
- Combined per-case results: `experiments/chexpert/results/phase2_11b/phase2_n600/meta_llama-3.2-11b-vision-instruct/imaging_blind_metric.jsonl`
- Phase checkpoints: the `imaging_blind_metric_phase*.jsonl` files in the same directory
- Cache: `experiments/chexpert/results/600_runs/img_cache_llama11b_n2.jsonl`
- Incremental execution log: `experiments/chexpert/results/api_watchdog_n2_11b_incremental.jsonl`

No API key values are stored in these artifacts. The manifest and prompt family were
kept unchanged from the CheXpert experiment.
