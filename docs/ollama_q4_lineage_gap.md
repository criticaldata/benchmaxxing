# Ollama Q4 lineage parity closure

Lineage: ollama_llama-3.2-11b-vision-instruct-q4_k_m. Hosted NIM lineage: meta_llama-3.2-11b-vision-instruct.

The 30 previously declared Q4 path gaps are closed by model-scoped analyses, manifests, provenance, and explicit aliases generated from audited Q4 rows and read-only caches. The gap ledger is empty. Generation made no model/API calls.

NIH reports include baseline accuracy: **1 of 35 correct (2.9%)** for the shared cohort. Q4 has watermark opacity levels 0.15, 0.30, and 0.45. Where one level is needed for a sibling comparison, reports use 0.30, the closest saved value to the standard 0.25, and label the 0.05 difference as an approximate match. The full sweep is also reported. MIMIC-CXR remains excluded. The 30 artifact paths are closed; the closure manifest preserves their original gap reasons and output hashes. CheXpert byte provenance is partial: 94 natural_cues case IDs lack source hashes in the imported artifacts and remain explicitly listed.

SUPPORT2 status was checked from five committed summaries and result files; original per-arm timestamps were not in the imported audit. The positional report gives a pooled descriptive fit because optional statsmodels is absent; it makes no mixed-effects inference.

## Analysis workflow reference

Kassis T, Agarwal V, He Y, Patel D, Brueckner AM. Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065 (2026). The skill informed evidence-bounded reporting and is not a source for experiment measurements.
