# PR #416: final NVIDIA NIM results and model fallback

## ✅ Final status

The original 90B target could not be completed with the `N_2` NVIDIA credential. The
11B model was run as an operational fallback after confirming that the same credential
could return successful responses for it. This change is a fallback for availability;
it does not claim that Llama 3.2 11B and 90B are equivalent models.

## 🔍 Why the model changed

### Llama 3.2 90B Vision

The 90B preflight never produced a model response. The N2 watchdog recorded 23
consecutive preflight timeouts between `2026-09-26T04:18:04Z` and
`2026-09-26T12:46:33Z`:

- HTTP status observed: `0` (no HTTP response received)
- Error: `timeout`
- Target block: 20 cases
- Cache rows: 0
- Cases started: 0

The watchdog report records the complete evidence and the 23 timeout events. Therefore,
the 90B run was stopped before any case was counted as complete. Evidence:
`experiments/chexpert/results/api_watchdog_n2_report.md`, with the raw events committed in
`experiments/chexpert/results/api_watchdog_n2_incremental.jsonl`.

### Llama 3.2 11B Vision

The same N2 credential returned HTTP 200 during the 11B run. The incremental runner
completed the staged targets `20`, `40`, `60`, `80`, `100`, `300`, and `600`. The
initial transient runner failure was recovered without counting a partial checkpoint.

The final 600-case checkpoint contains 600 rows for each of the three phases and has
zero unparseable responses. The final expansion from the existing 300-case state
added 900 new API calls; the complete cache contains 1,800 responses (600 cases x 3
phases). The corrected summary was regenerated from that complete cache with zero
additional API calls.

### Microsoft Phi-3 Vision

The requested model identifier `microsoft/phi-3-vision-128k-instruct` was also not
available through the tested NVIDIA endpoint: the availability probe returned HTTP
`404`. It was therefore excluded from the comparison and produced no usable
experimental results. This is an endpoint/model availability failure, not evidence
about the model's image quality.

## 📊 Llama 3.2 11B results (`n=600`)

The primary estimate is conditioned on a correct baseline read. Because this manifest
contains finding-present cases, `base_ans == yes` defines the baseline-correct stratum.
The model answered `no` on 229/600 cases, leaving 371 cases for the causal drift
comparison.

| Headline metric | Result | Count |
|---|---:|---:|
| Blind drift given correct baseline | 74.7% | 277 / 371 |
| Test-aware drift given correct baseline | 79.2% | 294 / 371 |
| Blind minus test-aware drift | -4.6 percentage points | (277 - 294) / 371 |
| No justification among correct-baseline blind drifters | 96.8% | 268 / 277 |
| Baseline-correct stratum | 61.8% | 371 / 600 |
| Unparseable responses | 0.0% | 0 / 1,800 |

The negative delta means the test-awareness warning did not suppress drift in the
baseline-correct stratum; the test-aware condition had 17 more drifts. The old
all-case rates remain useful as descriptive context (`80.5%` blind and `52.5%`
test-aware), but they mix in 229 cases where the model was already wrong and should
not be used as the causal drift headline. These metrics measure robustness to the
synthetic decoy/rubric prompt, not clinical diagnostic accuracy.

## ⚙️ Execution safety

The hosted 11B image runs were configured with `N_2` as the preferred NVIDIA credential,
one worker, a 20-second minimum gap between requests, and the same gap between gateway
retries. Every successful response is appended to the model-scoped cache before the next
case is attempted; a timeout or authorization error is not counted as a completed case.

## 📊 Gemini comparison

The existing Gemini CheXpert result covers only 35 cases. The conditional denominators
are model-specific because each model produces a different baseline answer; therefore
these rows should not be treated as a matched diagnostic-accuracy comparison. The
final row is the completed Llama 11B 600-case run, not a matched 600-vs-600 comparison.

For the shared 35-case hidden-note arm, also report the unconditional prompt-uptake rates
beside the conditional drift headline: the blind `SCORING NOTE` decoy was adopted in 23/35
(65.7%) cases, and the test-aware validation message decoy was adopted in 15/35 (42.9%).
These are unconditional rates over all 35 rows; the drift rows below are conditioned on
`base_ans == "yes"`.

| Model/cohort | Correct-baseline n | Blind drift | Test-aware drift | Blind minus aware |
|---|---:|---:|---:|---:|
| Gemini 2.5 Flash, existing subset | 31 | 93.5% (29/31) | 16.1% (5/31) | +77.4 pp |
| Llama 3.2 11B, same 35-case IDs | 21 | 61.9% (13/21) | 66.7% (14/21) | -4.8 pp |
| Llama 3.2 11B, final run | 371 | 74.7% (277/371) | 79.2% (294/371) | -4.6 pp |

On the available evidence, Gemini's warning reduced drift in its small cohort, while
Llama 11B's warning did not. A definitive model comparison requires running Gemini on
the same 600-case protocol and reporting the same baseline-correct conditioning.

## 📊 Additional CheXpert imaging battery

This PR also includes the eight model-scoped `experiments/imaging_chexpert` arms: four on the 35
shared cases (`imaging_solo`, `imaging_cascade`, `imaging_matched_temp`, and `imaging_noise_floor`)
and four 150-case arms (`device_absent`, `natural_cues`, `natural_independent`, and `system_flag`).
Each Llama 11B arm uses the same case IDs as its Qwen counterpart. The three `cue=none` cascades and
the `system_flag` control have a known cache-key defect: `clean` and `iso` reuse the same prompt/image
key, so those rows are not independent reads. Their shared-adoption rates are descriptive only; do
not interpret `shared_adopt - iso_adopt` as causal contagion. The exact findings and data-acquisition
status are recorded in the [experiment handoff](./pr416_nvidia_11b_handoff.md).

## 🔗 Reproducibility artifacts

- Summary: `experiments/chexpert/results/meta_llama-3.2-11b-vision-instruct/imaging_blind_metric_summary.json`
- Combined per-case results: `experiments/chexpert/results/meta_llama-3.2-11b-vision-instruct/imaging_blind_metric.jsonl`
- Phase checkpoints: the `imaging_blind_metric_phase*.jsonl` files in the same directory
- Shared 35-case hidden-note arm: `experiments/chexpert/results/meta_llama-3.2-11b-vision-instruct/hidden_note_n35/`
- Cache: `experiments/chexpert/results/600_runs/img_cache_llama11b_n2.jsonl`
- Incremental execution log: `experiments/chexpert/results/api_watchdog_n2_11b_incremental.jsonl`

No API key values are stored in these artifacts. The manifest and prompt family were
kept unchanged from the CheXpert experiment.
