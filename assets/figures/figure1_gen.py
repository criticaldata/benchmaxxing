#!/usr/bin/env python3
"""
Generator for figure1.svg (the overview figure in miccai.tex).

Reproduce (from this directory, data/paper/figures/):
    python3 figure1_gen.py                 # writes figure1.svg next to this script
    cairosvg figure1.svg -o figure1.pdf    # vector PDF that LaTeX embeds
    # alternatives to cairosvg: rsvg-convert -f pdf figure1.svg -o figure1.pdf
    #                            inkscape figure1.svg --export-type=pdf

No third-party imports: the script emits raw SVG, so it runs on a bare Python 3.
Only the SVG->PDF step needs an external tool (cairosvg / rsvg-convert / inkscape).

All numbers below are the corrected-parser results reported in miccai.tex; edit them
here and re-run to keep the figure in sync with the tables. Print/CVD-safe by design:
blue = text lane, orange = imaging lane; marker shape and texture carry method/state
so nothing is distinguished by colour alone.
"""
import os

BLUE, BLUE_LT = "#2a78d6", "#9ec5f4"
ORNG, ORNG_LT = "#eb6834", "#f7c6a9"
INK, SEC, MUT = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, GOOD = "#e1e0d9", "#c3c2b7", "#0ca30c"
SURF = "#fcfcfb"

def yv(v):            # value 0..0.9  -> y  (baseline 300, top 60)
    return 300 - (v / 0.9) * 240

out = []
def a(s): out.append(s)

a('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 350" '
  'font-family="system-ui, -apple-system, \'Segoe UI\', Helvetica, Arial, sans-serif">')
# opaque white background so dark ink/axes are visible in any viewer (SVG has no default bg).
a('<rect x="0" y="0" width="960" height="350" fill="#ffffff"/>')

# ---- shared lane legend (top right) ----
a(f'<g font-size="12.5" fill="{SEC}">')
a(f'<rect x="470" y="9" width="13" height="13" rx="2" fill="{BLUE}"/>'
  f'<text x="489" y="20">Text lane (MedQA-USMLE)</text>')
a(f'<rect x="690" y="9" width="13" height="13" rx="2" fill="{ORNG}"/>'
  f'<text x="709" y="20">Imaging lane (NIH ChestX-ray14)</text>')
a('</g>')

def y_axis_09(x0, x1, xlab):
    g = [f'<g stroke="{GRID}" stroke-width="1">']
    for v in (0, .2, .4, .6, .8):
        g.append(f'<line x1="{x0}" y1="{yv(v):.1f}" x2="{x1}" y2="{yv(v):.1f}"/>')
    g.append('</g>')
    g.append(f'<g font-size="10.5" fill="{MUT}" text-anchor="end">')
    for v, t in ((0, "0"), (.2, ".2"), (.4, ".4"), (.6, ".6"), (.8, ".8")):
        g.append(f'<text x="{x0-6}" y="{yv(v)+4:.1f}">{t}</text>')
    g.append('</g>')
    g.append(f'<text x="{x0-40}" y="184" font-size="11.5" fill="{SEC}" '
             f'transform="rotate(-90 {x0-40} 184)" text-anchor="middle">{xlab}</text>')
    g.append(f'<line x1="{x0}" y1="300" x2="{x1}" y2="300" stroke="{BASE}" stroke-width="1.4"/>')
    return "\n".join(g)

def title(x, l1, l2):
    return (f'<text x="{x}" y="44" font-size="13.5" font-weight="700" fill="{INK}">{l1}</text>'
            f'<text x="{x}" y="60" font-size="11" fill="{SEC}">{l2}</text>')

def bar(x, w, v, fill, vlab, vcol, stroke=None):
    y = yv(v)
    h = 300 - y
    st = f' stroke="{stroke}" stroke-width="1.4"' if stroke else ""
    return (f'<rect x="{x}" y="{y:.1f}" width="{w}" height="{h:.1f}" rx="3" fill="{fill}"{st}/>'
            f'<text x="{x+w/2}" y="{y-6:.1f}" font-size="11" font-weight="700" '
            f'fill="{vcol}" text-anchor="middle">{vlab}</text>')

# ==================== PANEL A ====================
a(title(40, "(a) Alone vs. in a committee", "adoption jumps once peers deliberate"))
a(y_axis_09(60, 290, "shortcut adoption (above noise)"))
a(bar(66, 34, .09, BLUE_LT, ".09", BLUE))
a(bar(104, 34, .30, BLUE, ".30", BLUE))
a(bar(190, 34, .11, ORNG_LT, ".11", ORNG))
a(bar(228, 34, .63, ORNG, ".63", ORNG))
a(f'<g font-size="10" fill="{MUT}" text-anchor="middle">')
for x, t in ((83, "solo"), (121, "committee"), (207, "solo"), (245, "committee")):
    a(f'<text x="{x}" y="313">{t}</text>')
a('</g>')
a(f'<g font-size="11.5" fill="{SEC}" text-anchor="middle" font-weight="600">'
  f'<text x="102" y="330">Text</text><text x="226" y="330">Imaging</text></g>')

# ==================== PANEL B ====================
# plot: x 385..595 (FPR 0..1), y 300..90 (precision 0..1)
def bx(fpr): return 385 + fpr * 210
def by(p):   return 300 - p * 210
a(title(340, "(b) Referee vs. naive gate", "referee: few false alarms, high precision"))
a(f'<g stroke="{GRID}" stroke-width="1">'
  f'<line x1="385" y1="195" x2="595" y2="195"/>'
  f'<line x1="490" y1="90" x2="490" y2="300"/></g>')
a(f'<line x1="385" y1="300" x2="595" y2="300" stroke="{BASE}" stroke-width="1.4"/>')
a(f'<line x1="385" y1="300" x2="385" y2="90" stroke="{BASE}" stroke-width="1.4"/>')
a(f'<g font-size="10.5" fill="{MUT}" text-anchor="middle">'
  f'<text x="385" y="315">0</text><text x="490" y="315">0.5</text><text x="595" y="315">1.0</text></g>')
a(f'<text x="490" y="332" font-size="11.5" fill="{SEC}" text-anchor="middle">'
  f'false-positive rate  (lower is better)</text>')
a(f'<g font-size="10.5" fill="{MUT}" text-anchor="end">'
  f'<text x="379" y="304">0</text><text x="379" y="199">.5</text><text x="379" y="94">1</text></g>')
a(f'<text x="349" y="195" font-size="11.5" fill="{SEC}" '
  f'transform="rotate(-90 349 195)" text-anchor="middle">precision</text>')
# ideal corner
a(f'<circle cx="385" cy="90" r="15" fill="none" stroke="{GOOD}" stroke-width="1" stroke-dasharray="2 2"/>')
a(f'<text x="404" y="86" font-size="10" fill="{GOOD}" font-weight="600">ideal</text>')
# arrows naive -> referee
# text: naive (1.0, .375) -> referee (0, 1.0)
a(f'<line x1="{bx(1.0):.1f}" y1="{by(.375):.1f}" x2="{bx(.03):.1f}" y2="{by(.985):.1f}" '
  f'stroke="{BLUE}" stroke-width="1.6" stroke-dasharray="4 3" '
  f'marker-end="url(#arB)"/>')
# imaging: naive (.92,.65) -> referee (.23,.86)
a(f'<line x1="{bx(.92):.1f}" y1="{by(.65):.1f}" x2="{bx(.26):.1f}" y2="{by(.855):.1f}" '
  f'stroke="{ORNG}" stroke-width="1.6" stroke-dasharray="4 3" '
  f'marker-end="url(#arO)"/>')
# defs for arrowheads
a(f'<defs>'
  f'<marker id="arB" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">'
  f'<path d="M0,0 L7,3.2 L0,6.4 Z" fill="{BLUE}"/></marker>'
  f'<marker id="arO" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">'
  f'<path d="M0,0 L7,3.2 L0,6.4 Z" fill="{ORNG}"/></marker></defs>')
# points
a(f'<circle cx="{bx(1.0):.1f}" cy="{by(.375):.1f}" r="6.5" fill="{SURF}" stroke="{BLUE}" stroke-width="2"/>')
a(f'<circle cx="{bx(.92):.1f}" cy="{by(.65):.1f}" r="6.5" fill="{SURF}" stroke="{ORNG}" stroke-width="2"/>')
a(f'<circle cx="{bx(0):.1f}" cy="{by(1.0):.1f}" r="6.5" fill="{BLUE}" stroke="{SURF}" stroke-width="1.5"/>')
a(f'<circle cx="{bx(.23):.1f}" cy="{by(.86):.1f}" r="6.5" fill="{ORNG}" stroke="{SURF}" stroke-width="1.5"/>')
# compact shape key (marker shape = method; colour = lane, per the top legend)
a(f'<g font-size="10.5" fill="{SEC}">')
a(f'<circle cx="472" cy="100" r="5.5" fill="{SURF}" stroke="{MUT}" stroke-width="2"/>'
  f'<text x="483" y="104">naive gate</text>')
a(f'<circle cx="472" cy="117" r="5.5" fill="{MUT}" stroke="{SURF}" stroke-width="1.5"/>'
  f'<text x="483" y="121">referee</text>')
a('</g>')

# ==================== PANEL C ====================
a(title(660, "(c) Gaming: named vs. silent", "imaging drifts more, and names nothing"))
a(y_axis_09(700, 900, "decoy uptake &#916; (rubric on&#8722;off)"))
# text bar: uptake .275 solid (named)
a(bar(728, 52, .275, BLUE, ".28", BLUE))
a('<text x="754" y="266" font-size="10.5" font-weight="700" fill="#ffffff" text-anchor="middle">named</text>')
a('<text x="754" y="280" font-size="9.5" fill="#ffffff" text-anchor="middle">11/11</text>')
# imaging bar: uptake .83 light fill + diagonal hatch (silent).
# Hatch lines are clipped ANALYTICALLY to the bar box (no clipPath: renderer-robust).
by0 = yv(.83)
bx0, bx1, byT, byB = 822, 874, by0, 300.0
a(f'<rect x="{bx0}" y="{byT:.1f}" width="{bx1-bx0}" height="{byB-byT:.1f}" rx="3" '
  f'fill="{ORNG_LT}" stroke="{ORNG}" stroke-width="1.4"/>')
a(f'<g stroke="{ORNG}" stroke-width="1.8">')
c = 901
while c <= 1174:
    xA = max(bx0, c - byB)
    yA = c - xA
    xB = min(bx1, c - byT)
    yB = c - xB
    if xA <= xB:
        a(f'<line x1="{xA:.1f}" y1="{yA:.1f}" x2="{xB:.1f}" y2="{yB:.1f}"/>')
    c += 8
a('</g>')
a(f'<text x="848" y="{by0-6:.1f}" font-size="11" font-weight="700" fill="{ORNG}" text-anchor="middle">.83</text>')
a(f'<text x="848" y="188" font-size="10.5" font-weight="700" fill="{INK}" text-anchor="middle">silent</text>')
a(f'<text x="848" y="202" font-size="9.5" fill="{INK}" text-anchor="middle">0/29</text>')
a(f'<g font-size="11.5" fill="{SEC}" text-anchor="middle" font-weight="600">'
  f'<text x="754" y="316">Text</text><text x="848" y="316">Imaging</text></g>')
# condition legend
a(f'<g font-size="10" fill="{MUT}">'
  f'<rect x="700" y="330" width="11" height="11" rx="2" fill="{BLUE}"/>'
  f'<text x="715" y="339">names the rubric</text>'
  f'<rect x="808" y="330" width="11" height="11" rx="2" fill="{ORNG_LT}" stroke="{ORNG}"/>'
  f'<line x1="808" y1="341" x2="819" y2="330" stroke="{ORNG}" stroke-width="1.6"/>'
  f'<text x="823" y="339">silent (confabulates)</text></g>')

a('</svg>')

_here = os.path.dirname(os.path.abspath(__file__))
_dst = os.path.join(_here, "figure1.svg")
with open(_dst, "w") as f:
    f.write("\n".join(out))
print(f"wrote {_dst}")
