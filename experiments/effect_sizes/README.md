# Effect sizes, bootstrap CIs, and achieved power, text lane (issue #192)

Several text-lane arms in this project report only a point estimate plus a McNemar/Fisher
p-value, with no interval and no power annotation. This adds a risk difference, odds ratio,
bootstrap 95% CI, and achieved statistical power (or the pairs needed for 80% power) to every
text-lane arm whose source data has been independently re-verified as current (post-parser-fix)
this session, using the same conservative scoping principle as #191's family-wide correction:
sources not re-verified this session, or confirmed tainted by the historical parser bug, are
listed under `excluded` with a reason rather than risking a mix of stale and corrected numbers.

The imaging-lane half of #192 is delivered separately, stacked on the imaging results branch
(PR #226).

## Two different bootstrap designs, chosen per arm

- **Paired bootstrap** (`_rate_diff_bootstrap_paired`): used only for `scale_c`, where the same
  85 cases are scored under both the generic and anchored conditions (matched per-case rows in
  `scale_c_per_case.jsonl`). Resamples one shared set of case indices and applies it to both
  conditions, preserving the pairing.
- **Independent-groups bootstrap** (`_rate_diff_bootstrap_independent`): used for `clean_a` and
  the `contamination_audit` split, where the two groups being compared are different, disjoint
  sets of cases (e.g. baseline-wrong cases vs baseline-correct cases) of different sizes. Each
  group is resampled independently with its own length.

An earlier draft of this script called the paired bootstrap on all arms, including the
independent-groups ones. Because the paired implementation resamples both arrays with an index
range sized to the *first* array's length, and `clean_a`/`contamination_audit`'s two groups are
different sizes (e.g. 11 vs 89 cases), it silently truncated the larger group to the smaller
group's index range instead of resampling it properly. This produced a bootstrap CI that did not
even bracket the point estimate (e.g. `[-0.45, 0.0]` around a point estimate of `+0.376`), caught
by comparing the interval against the point estimate before shipping. Fixed by splitting into two
explicit, differently-named functions and calling the correct one per arm.

Where only aggregate counts are committed (`clean_a`, `contamination_audit`), the bootstrap
reconstructs the exact empirical Bernoulli array implied by the counts (e.g. 6 ones and 17 zeros
for "6 of 23"); bootstrapping this array is identical to bootstrapping the real per-case data,
since only the empirical distribution, which the counts already fully specify, matters for the
resample.

## Arms covered

| Arm | Design | Risk difference | Bootstrap 95% CI | Odds ratio | n |
|---|---|---|---|---|---|
| `scale_c_anchored_vs_generic` | paired (real per-case rows) | 0.1176 | [0.0118, 0.2118] | 2.05 | 85 |
| `clean_a_flash` | independent (reconstructed) | 0.2609 | [0.087, 0.4348] | undefined (zero cell) | 23 |
| `clean_a_flash_lite` | independent (reconstructed) | 0.3043 | [0.1304, 0.4783] | undefined (zero cell) | 23 |
| `contamination_flash` | independent (reconstructed) | 0.3759 | [0.0919, 0.6711] | 9.76 | 11 vs 89 |
| `contamination_flash_lite` | independent (reconstructed) | 0.5466 | [0.3322, 0.7471] | 17.75 | 22 vs 78 |

`model_dependence_flash` and `multi_round_text_round1_vs_roundK` have no natural risk-difference
framing (single gain/lose/n triples, not two comparable groups), so only `achieved_power` and
`required_pairs_for_power_0.8` are reported for those.

## Read

`scale_c`'s anchored-vs-generic increment has a bootstrap interval that comes close to, but does
not include, 0 (lower bound 0.0118) and an achieved power of only 0.61 at n=85, consistent with
#191's finding that this effect does not survive a family-wide correction. Clean-A's risk
difference is large for both tiers and its interval is tighter for flash-lite than flash,
matching flash-lite's stronger Fisher p. The contamination audit's risk differences are large,
fully positive (both intervals now exclude 0), and one-directional for both tiers: the most
robust text-lane finding across every check this session, exactly as #191's family-wide
correction independently concluded.

`model_dependence_flash` and the multi-round text arm are both badly underpowered at their
current n (achieved power 0.056 and 0.166; would need 1537 and 312 paired cases respectively for
80% power), which is itself informative: neither arm should be read as a null result, only as an
inconclusive one at current sample size.

## Excluded sources

- `experiments/medqa/results/break_it_summary.json` / `push_c_summary.json`
  (`results/medqa-break-it`, PR #140): not independently re-verified for parser-fix currency this
  session.
- `experiments/medqa/results/cascade_v2_per_case.jsonl` (`results/medqa-experiments`, PR #135):
  confirmed tainted by the pre-fix stray-article parser bug.

## Reproduce

```bash
python3 -m experiments.effect_sizes.effect_sizes_text
```

Pure computation over already-committed per-case rows (`results/scale_c_per_case.jsonl`, copied
unchanged from `results/cascade-at-scale`) and already-committed aggregate counts; no API calls,
no key needed. Writes `results/effect_sizes_text.json`.

## Files

- `effect_sizes_text.py`, the analysis script.
- `results/scale_c_per_case.jsonl`, the real per-case rows (copied from
  `results/cascade-at-scale`, unchanged).
- `results/effect_sizes_text.json`, the computed output.
