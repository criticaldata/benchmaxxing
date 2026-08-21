# Cross-Dataset Comparison: Blind-Metric Substitution (Issue #294)

This table tracks the susceptibility to a misaligned "blind metric" (a hidden rubric rewarding a specific decoy answer or feature) across text and imaging datasets, specifically asking whether the gaming is "aware" (the model names the rubric in its justification) or "silent" (the model hides the rubric from its clinical justification). Under the shared detector it is near-silent in every lane.

| Dataset | Modality | Decoy Uptake &Delta; (Blind - Baseline) | Rubric-Naming Rate (among drifters) | Gaming Style |
|---|---|---|---|---|
| **MedQA** | Text | +0.275 (27.5%) | 9% (1/11) | Silent Gaming / Confabulation |
| **NIH ChestX-ray14** | Imaging | +0.830 (83.0%) | 0% (0/29) | Silent Gaming / Confabulation |
| **CheXpert** | Imaging | +0.857 (85.7%) | 0% (0/30) | Silent Gaming / Confabulation |

## Key Takeaways
- **Text vs. Imaging Drift:** Imaging models (NIH, CheXpert) show significantly higher susceptibility (83-88% drift) compared to text-only medical reasoning (27% drift).
- **The Silent Gaming Phenomenon:** Drift is near-silent in every lane under the shared naming detector: 1/11 drifters name the rubric on MedQA, 0/29 on NIH and 0/30 on CheXpert; the rest confabulate a purely clinical justification for the decoy. An earlier version of this table reported MedQA as fully self-declared (11/11); that figure came from a degenerate detector disjunct in the MedQA runner (see its `#NAMING-DETECTOR` note), and the paper's 1/11 vs 1/134 contrast (Fisher p=0.15) is the corrected comparison.

## Solo Susceptibility & Matched-Temperature Floor

All imaging lane datasets (NIH ChestX-ray14, CheXpert, and MIMIC-CXR) are processed through the centralized `experiments/imaging/` orchestrator scripts. 

Specifically, the "Baseline" (solo) numbers for all three datasets are **matched-temperature-corrected** via `experiments/imaging/imaging_matched_temp_floor.py`. This ensures that any measured delta in decoy uptake is isolated to the systemic pressure, formally preventing temperature artifacts from contaminating the cross-dataset comparisons (Resolves #343 and #344).
