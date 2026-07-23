# Paper figures

Source-controlled figures for the write-up, each generated from a script so the
numbers stay in sync with the experiments.

## figure1 (overview)

Three panels across the text lane (MedQA-USMLE) and the imaging lane (NIH ChestX-ray14):

- **(a)** a shortcut resisted by a model alone is adopted far more once the same model
  deliberates in a committee;
- **(b)** a naive agreement gate over-fires while a targeted, deployable referee moves toward
  the low-false-positive, high-precision corner;
- **(c)** reward-hacking toward a hidden rubric is larger and *silent* in imaging (0/29 name the
  rubric) versus fully self-declared in text (11/11).

Rebuild:

```bash
python3 figure1_gen.py             # writes figure1.svg
cairosvg figure1.svg -o figure1.pdf   # or: rsvg-convert -f pdf / inkscape --export-type=pdf
```

`figure1_gen.py` has no third-party imports (it emits raw SVG); only the SVG-to-PDF step needs
an external tool. All plotted values are defined at the top of the script; edit them there and
re-run to keep the figure aligned with the result tables.
