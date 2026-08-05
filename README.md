<p align="center">
  <img src="assets/benchmaxxing.gif" alt="benchmaxxing: a committee shares context, one agent's shortcut cascades, a referee flags it" width="760">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.03744"><img src="https://img.shields.io/badge/arXiv-2608.03744-b31b1b.svg" alt="Paper on arXiv"></a>
  <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6676818"><img src="https://img.shields.io/badge/SSRN-6676818-b31b1b.svg" alt="DOJO paper on SSRN"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-CC%20BY%204.0-blue.svg" alt="Licensed CC BY 4.0"></a>
</p>

# benchmaxxing

Referee agents that catch benchmark gaming in clinical multi-agent systems: shortcut cascades, blind metrics, and mutual oversight, measured.

**Paper:** [Agents Catching Agents: Shortcut Cascades and Benchmark Gaming in Clinical Multi-Agent
Systems](https://arxiv.org/abs/2608.03744) (arXiv:2608.03744). Every number in it is computed from the
artifacts in this repository.

A benchmark score tells you an agent hit the target, not whether the target was right or whether it got there honestly. When agents share context, one agent's shortcut can propagate through the whole group before any human notices. This repo builds **referee agents** (agents whose task is the assessment of other agents) and a reproducible sandbox to measure three failure patterns: shortcut reliance, conformity cascades, and emergent dominance.

## Design principle: reuse the originals

Every line we write ourselves is a line that can be wrong and a line a reviewer can attribute the result to instead of the phenomenon. So the pipeline is thin wrappers over established, version-pinned libraries (numpy/scipy/scikit-learn for statistics, a change-point library for cascade onset, image libraries for cue injection, a unified gateway for models), and the bespoke core is kept small and tested hardest. The measure of progress is how little of the pipeline is ours.

## Model backend

Agents use the **Gemini API** (multimodal) for now, behind one gateway wrapper so the roster can be extended to other model APIs later without changing experiment code. No fine-tuning; models are used off-the-shelf. Model lineage is a first-class variable: Gemini-only committees are the same-lineage control, and Gemini-plus-open-weights committees are the cross-lineage arm.

## Install

```bash
pip install -e ".[dev]"                 # pure-Python core + test tooling
pip install -e ".[dev,stats,changepoint,image,config]"   # add the reuse extras
```

## Layout

- `benchmaxxing/schema.py`: the shared data contract every module builds against.
- `benchmaxxing/stats.py`: statistical tests (wrappers over scipy/sklearn/statsmodels).
- `benchmaxxing/onset.py`: cascade-onset (change-point), contagion, deference.
- `benchmaxxing/cues/`: image and text cue injection, twin-pair builder.
- `benchmaxxing/blackboard.py`: the shared-context committee harness and referee hooks.
- `benchmaxxing/gateway.py`, `roster.py`: model access and same/cross-lineage committees.
- `benchmaxxing/datasets/`: dataset adapters into the shared case schema; dataset staged/coded/blocked status lives in `benchmaxxing/datasets/status.py` and is surfaced by `benchmaxxing datasets`.
- `benchmaxxing/referee.py`, `blind_metric.py`: the referee duties and the blind-metric probe.

The build plan and the issue tracker for this repo mirror a six-stage program (stages 0-5).

## Licence

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), the same licence as the paper. Full
text in [`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0), scope in [`LICENSE`](LICENSE).

No dataset is redistributed here. MedQA-USMLE, MedMCQA, NIH ChestX-ray14, CheXpert, SUPPORT2 and
MIMIC-CXR each carry their own terms and must be obtained from their providers. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for access requirements and for how to reproduce the results.

## Citation

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders it as a "Cite this
repository" button. To cite this work, cite the paper:

```bibtex
@misc{cajasordonez2026agents,
  title         = {Agents Catching Agents: Shortcut Cascades and Benchmark Gaming
                   in Clinical Multi-Agent Systems},
  author        = {Cajas Ord{\'o}{\~n}ez, Sebasti{\'a}n Andr{\'e}s and Munnangi, Agastya and
                   Marzullo, Aldo and Ocampo Osorio, Felipe and Bui, Quang and
                   Shahin, Mohammad and Grewal, Armaan and Kwesiga, Emmanuel Paul and
                   Li, Anqi Peter and Nanyonjo, Josephine and Panchal, Aaditya and
                   Bhutani, Arshnoor and Jaiswal, Nikhil and Patel, Milit S. and
                   Lange, Maximin and Celi, Leo Anthony},
  year          = {2026},
  eprint        = {2608.03744},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2608.03744},
  url           = {https://arxiv.org/abs/2608.03744}
}
```

This work sits under the DOJO programme, which sets out the community-driven adversarial
evaluation platform these referee experiments instantiate:

```bibtex
@article{dojo2026,
  title   = {Distributed Open Justice Oversight (DOJO): A Community-Driven,
             Modality-Agnostic Platform for Adversarial Evaluation of Health AI},
  author  = {Xiang, Alexa Q. and Tohyama, Takeshi and Bank, Alexander Cole and
             Bui, Quang and Gorijavolu, Rahul and Garcia Henao, John Anderson and
             Jaiswal, Nikhil and Kelshiker, Akshay and Madapati, Kaushik and
             Caj{\'a}s Ord{\'o}{\~n}ez, Sebastian A. and Patel, Milit and
             Prakash, Nina and Celi, Leo Anthony},
  year    = {2026},
  month   = may,
  note    = {SSRN preprint 6676818, posted 11 May 2026},
  url     = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6676818}
}
```
