"""Shared style of the paper's figures (IEEE column 3.5 in, page 7.16 in).

Text in TeX Gyre Termes (the Times of the paper body, files in ``fonts/``, GUST Font License) and
math in Computer Modern, as in the paper, so that the figures read like the text around them.
Sizes: 8 pt for labels and titles, 7 pt for ticks, legends and annotations.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

UAVSAR, S1, SAMEDAY = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1, 2, 3
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#d9d8d2"
COLUMN, PAGE = 3.5, 7.16
BASE, SMALL = 8, 7
FONT_DIR = Path(__file__).with_name("fonts")


def _truetype(otf):
    """TrueType copy of an OpenType (CFF) font in a local cache, so that the PDFs embed it as a
    Type 42 font (CFF outlines converted to quadratic curves with fontTools)."""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable

    out = Path.home() / ".cache" / "closure_glrt_fonts" / (otf.stem + ".ttf")
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    font = TTFont(otf)
    gs = font.getGlyphSet()
    glyf = newTable("glyf")
    glyf.glyphs, glyf.glyphOrder = {}, font.getGlyphOrder()
    for name in glyf.glyphOrder:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, max_err=1.0, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    font["glyf"], font["loca"] = glyf, newTable("loca")
    font["maxp"] = newTable("maxp")
    m = font["maxp"]
    m.tableVersion, m.maxZones = 0x00010000, 1
    for k in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
              "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
        setattr(m, k, 0)
    font["head"].glyphDataFormat = 0
    font["post"].formatType, font["post"].extraNames, font["post"].mapping = 2.0, [], {}
    del font["CFF "]
    font.sfntVersion = "\x00\x01\x00\x00"
    font.save(out)
    return out


def style():
    fm = font_manager.fontManager
    for f in FONT_DIR.glob("*.otf"):
        fm.addfont(str(_truetype(f)))
    # keep only the TrueType copies (a system OpenType Termes would be embedded as CFF)
    fm.ttflist = [e for e in fm.ttflist
                  if not (e.name == "TeX Gyre Termes" and not e.fname.endswith(".ttf"))]
    font_manager.findfont.cache_clear() if hasattr(font_manager.findfont, "cache_clear") \
        else None
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["TeX Gyre Termes"], "mathtext.fontset": "cm",
        "axes.unicode_minus": True,
        "font.size": BASE, "axes.labelsize": BASE, "axes.titlesize": BASE,
        "xtick.labelsize": SMALL, "ytick.labelsize": SMALL, "legend.fontsize": SMALL,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "pdf.fonttype": 42, "image.interpolation": "none", "savefig.dpi": 600})


def panel_label(ax, label, color=INK, bg=None, corner="left"):
    box = dict(fc=bg, ec="none", pad=1.0) if bg else None
    x, ha = (0.01, "left") if corner == "left" else (0.99, "right")
    ax.text(x, 0.99, label, transform=ax.transAxes, fontsize=BASE, va="top", ha=ha,
            fontweight="bold", color=color, bbox=box)
