<p align="center">
  <img src="assets/benchmaxxing.gif" alt="benchmaxxing: a committee shares context, one agent's shortcut cascades, a referee flags it" width="760">
</p>

<p align="center">
  <a href="https://huggingface.co/papers/2608.03744"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow.svg" alt="Paper on Hugging Face"></a>
  <a href="https://arxiv.org/abs/2608.03744"><img src="https://img.shields.io/badge/arXiv-2608.03744-b31b1b.svg" alt="Paper on arXiv"></a>
  <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6676818"><img src="https://img.shields.io/badge/SSRN-6676818-b31b1b.svg" alt="DOJO paper on SSRN"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/licence-MIT-blue.svg" alt="MIT licensed"></a>
  <a href="#reproduce-it-without-an-api-key"><img src="https://img.shields.io/badge/replay-0%20API%20calls-brightgreen.svg" alt="Replays from cache with no API key"></a>
</p>

# benchmaxxing

### An AI agent ignores the shortcut when it works alone. Put two confident peers next to it and it takes the bait.

**Read the paper:** [Hugging Face](https://huggingface.co/papers/2608.03744) | [arXiv:2608.03744](https://arxiv.org/abs/2608.03744) | 2nd Agentic AI for Medicine Workshop @ MICCAI 2026

Clinical AI is moving to **committees** of language-model agents that deliberate on a shared
workspace. The safety story is redundancy: more agents, more chances someone catches the error. We
tested that story across seven cohorts and six public datasets, and it does not hold.

## What we found

**Alone, agents mostly resist the shortcut. In committee, they adopt it.**

| | Result |
|---|---|
| Two peers assert the same wrong answer | adopted on **15 of 40** medical exam cases |
| Same manipulation on chest X-rays | **+0.54 or more**, including cues that did nothing alone |
| Two scripted senior peers on ICU records | adoption hits **1.0**, zero resisters of 115 |

**The strongest version adds nothing to the image at all.** On 150 chest films that already carry a
support device, no pixels altered, the model reads them correctly on 143 by itself. Two peers
introduced as senior radiologists move it to the wrong read on **61%** of films. A wrong automated
flag with no peer behind it moves it on **0.7%**. The peer is the whole manipulation.

**What travels is agreement and confidence, not argument.** Hold the claim and the source fixed and
vary only how hedged the peer sounds: a categorical assertion is adopted on **0.42** of cases
against **0.14** for a hedged one. One peer carrying a case-specific rationale is statistically
indistinguishable from two carrying none.

**No single monitor catches it everywhere.** We compared three overseers on the same runs:

| Detector | What it sees | Text | Imaging |
|---|---|---|---|
| **gate** | flags any agreement | useless by construction | useless by construction |
| **judge** | reads the deliberation | precision 1.0, recall 0.93 | collapses onto the gate |
| **referee** | privately re-asks the agent, transcript withheld | 0.68 precision | **0.66 to 0.88 precision, 0.13 to 0.36 false alarms** |

The referee is the only one that transfers, and we quote its worse end on purpose, because quoting
the better end is the behaviour this paper is about.

**And we threw away five of our own results before publishing**, because their predicates could not
have come out otherwise. Two referee flags reduced algebraically to the label they were scored
against. That screen ships with the release, not just its output, so you can run the check that
would have caught us.

## Reproduce it without an API key

Every model call is cached and content-addressed, so the committed runs replay offline.

```bash
pip install -e ".[dev]"
pytest -q                                   # 1,190+ tests, no network
python experiments/blind_metric/blind_metric.py --manifest <manifest> \
    --cache experiments/blind_metric/results/call_cache.jsonl
# -> new_api_calls_this_run: 0
```

MedQA, NIH ChestX-ray14, CheXpert and SUPPORT2 replay from cache at zero cost. The MIMIC-CXR lanes
cannot, under PhysioNet terms, and ship de-identified per-case rows instead. Two imaging cue arms
reproduce only under a pinned font version, a cache-invalidation defect we document rather than
paper over; the per-case rows are the artefact of record there.

## Design principle: reuse the originals

Every line we write ourselves is a line that can be wrong, and a line a reviewer can attribute the
result to instead of the phenomenon. So the pipeline is thin wrappers over established,
version-pinned libraries (numpy/scipy/scikit-learn for statistics, a change-point library for
cascade onset, image libraries for cue injection, one gateway for models), and the bespoke core is
kept small and tested hardest. The measure of progress is how little of the pipeline is ours.

## Model backend

Agents use the **Gemini API** (multimodal), behind one gateway wrapper so the roster extends to
other model APIs without touching experiment code. No fine-tuning; models are off-the-shelf at
temperature 0. Model lineage is a first-class variable: Gemini-only committees are the
same-lineage control, Gemini-plus-open-weights the cross-lineage arm.

## Install

```bash
pip install -e ".[dev]"                                  # pure-Python core + test tooling
pip install -e ".[dev,stats,changepoint,image,config]"   # add the reuse extras
```

## Layout

- `benchmaxxing/schema.py`: the shared data contract every module builds against.
- `benchmaxxing/stats.py`: statistical tests (wrappers over scipy/sklearn/statsmodels).
- `benchmaxxing/onset.py`: cascade-onset (change-point), contagion, deference.
- `benchmaxxing/cues/`: image and text cue injection, twin-pair builder.
- `benchmaxxing/blackboard.py`: the shared-context committee harness and referee hooks.
- `benchmaxxing/gateway.py`, `roster.py`: model access and same/cross-lineage committees.
- `benchmaxxing/datasets/`: dataset adapters into the shared case schema; staged/coded/blocked status lives in `benchmaxxing/datasets/status.py` and is surfaced by `benchmaxxing datasets`.
- `benchmaxxing/referee.py`, `blind_metric.py`: the referee duties and the blind-metric probe.
- `experiments/`: one directory per lane, each with the runner that regenerates its committed results.

## Licence

Two licences, split by content type:

- **Code** (`benchmaxxing/`, `experiments/`, `scripts/`, `tests/`): [MIT](LICENSE). Creative
  Commons [recommends against](https://creativecommons.org/faq/#can-i-apply-a-creative-commons-license-to-software)
  applying a CC licence to software, so the code does not carry one.
- **Data, figures and prose** (per-case rows under `experiments/*/results/`, `assets/figures/`, the
  Markdown docs): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), full text in
  [`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0). Attribution is satisfied by citing the paper.

The vendored DejaVu font under `benchmaxxing/cues/assets/` keeps its own licence.

**No dataset is redistributed here.** MedQA-USMLE, MedMCQA, NIH ChestX-ray14, CheXpert, SUPPORT2
and MIMIC-CXR each carry their own terms and must be obtained from their providers. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for access requirements and for how to reproduce the results.

## Citation

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders it as a "Cite this
repository" button. To cite this work, cite the paper:

```bibtex
@misc{cajasordonez2026agents,
  title         = {Agents Catching Agents: Shortcut Cascades and Benchmark Gaming
                   in Clinical Multi-Agent Systems},
  author        = {Cajas Ord{\'o}{\~n}ez, Sebasti{\'a}n Andr{\'e}s and
                   Rodr{\'i}guez Moran, Yehudhah Kennedy and Munnangi, Agastya and
                   Marzullo, Aldo and Ocampo Osorio, Felipe and Bui, Quang and
                   Shahin, Mohammad and Grewal, Armaan and Kwesiga, Emmanuel Paul and
                   Li, Anqi Peter and Nanyonjo, Josephine and Panchal, Aaditya and
                   Bhutani, Arshnoor and Jaiswal, Nikhil and Patel, Milit S. and
                   Lange, Maximin and Umeton, Renato and Celi, Leo Anthony},
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
