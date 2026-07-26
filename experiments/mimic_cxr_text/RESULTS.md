# MIMIC-CXR text (Lane B): solo susceptibility + noise floor

Real-API results for #316, using `benchmaxxing/datasets/mimic_cxr_text.py` (adapter design per
#330) and `experiments/medqa/reproduce.py` reused as-is (per #316: "do not rewrite the method,
replicate it"), `--stage solo`.

Per #334/#316: an `n=35` smoke run first (new adapter, never exercised against the real API),
then `n=600` (sizing rationale and precedent in #334, in the same range as MedQA's own real-API
counts).

Manifests and raw model-call caches are **not committed**: report text is credentialed
PhysioNet data under a Data Use Agreement and must not be redistributed, even in a private repo.
Only the aggregate JSON below (flip rates, noise floor) is committed.

## n=35 (smoke run)

```json
{
  "n_records": 210,
  "noise_floor_by_model": {
    "gemini-2.5-flash": 0.13333333333333333,
    "gemini-2.5-flash-lite": 0.0
  },
  "flip_rate_by_model": {
    "gemini-2.5-flash": {"overall": 0.09523809523809523, "n": 105},
    "gemini-2.5-flash-lite": {"overall": 0.12380952380952381, "n": 105}
  }
}
```

Full record: [`results/solo_results.json`](results/solo_results.json).

Flag: at n=35, `gemini-2.5-flash`'s noise floor (0.133) came out *above* its flip rate (0.095) —
the run_solo noise-floor control always samples a fixed 15 cases, so this is a small-sample
noise artifact, not a design problem. Resolved at n=600 below.

## n=600

```json
{
  "n_records": 3600,
  "noise_floor_by_model": {
    "gemini-2.5-flash": 0.0,
    "gemini-2.5-flash-lite": 0.0
  },
  "flip_rate_by_model": {
    "gemini-2.5-flash": {
      "overall": 0.07888888888888888,
      "per_cue": {
        "lexical_overlap": 0.09333333333333334,
        "longest_option": 0.08333333333333333,
        "option_order": 0.06
      },
      "n": 1800
    },
    "gemini-2.5-flash-lite": {
      "overall": 0.1238888888888889,
      "per_cue": {
        "lexical_overlap": 0.14833333333333334,
        "longest_option": 0.135,
        "option_order": 0.08833333333333333
      },
      "n": 1800
    }
  }
}
```

Full record: [`results_n600/solo_results.json`](results_n600/solo_results.json).

## Cross-dataset table row (MIMIC-CXR text, n=600)

| model | overall flip | noise floor | flip-above-noise |
|---|---|---|---|
| gemini-2.5-flash | 0.079 | 0.000 | +0.079 |
| gemini-2.5-flash-lite | 0.124 | 0.000 | +0.124 |

Both models show a genuine (noise-floor-clean at n=600), smaller-than-MedQA shortcut effect on
report-text MCQs; `lexical_overlap` is the strongest cue for both tiers. Consistent between
n=35 (0.095/0.124) and n=600 (0.079/0.124) samples.
