"""
Generate an animated graphical abstract GIF for PaRaVis.

Produces a looping animation illustrating the PaRaVis workflow:
  Raster Input -> Spectral Indices + Rao's Q -> Visualisation -> GUI

Usage:
    python scripts/generate_graphical_abstract.py

Output:
    docs/graphical_abstract.gif
"""

import io
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

# ── Configuration ──────────────────────────────────────────────────
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "graphical_abstract.gif"
FPS = 1.2
W, H = 640, 400
DPI = 120
BG_COLOR = "#0d1b2a"
ACCENT = "#009688"
ACCENT2 = "#00695c"
TITLE_COLOR = "#ffffff"
SUBTITLE_COLOR = "#b2dfdb"
TEXT_COLOR = "#b0bec5"
MUTED = "#546e7a"
FONT_FAMILY = "sans-serif"
MONO_FONT = "monospace"


def mpl_to_pil(fig) -> Image.Image:
    """Convert a Matplotlib figure to a PIL Image (RGBA)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI,
                facecolor=BG_COLOR, edgecolor="none")
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def _add_frame_border(ax, alpha=1.0):
    """Add a subtle unified border around the frame."""
    border = FancyBboxPatch(
        (0.03, 0.025), 0.94, 0.95,
        facecolor="none", edgecolor=ACCENT2,
        alpha=alpha * 0.25, transform=ax.transAxes,
        linewidth=0.8, boxstyle="round,pad=0.01"
    )
    ax.add_patch(border)


def _add_step(ax, current: int, total: int = 7, alpha=1.0):
    """Add a step indicator at the bottom edge."""
    for i in range(total):
        x = 0.30 + i * (0.40 / (total - 1))
        color = ACCENT if i < current else MUTED
        circle = plt.Circle(
            (x, 0.035), 0.005, color=color,
            alpha=alpha * (0.9 if i < current else 0.3),
            transform=ax.transAxes
        )
        ax.add_patch(circle)


# ── Frame generators ───────────────────────────────────────────────

def frame_title(alpha: float = 1.0) -> Image.Image:
    """Title / hero card."""
    fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Decorative glow circles
    for cx, cy, r, c in [(0.08, 0.92, 0.30, "#009688"),
                          (0.92, 0.08, 0.22, "#004d40")]:
        circle = plt.Circle((cx, cy), r, color=c, alpha=0.06,
                            transform=ax.transAxes)
        ax.add_patch(circle)

    # Central "orb" glow
    glow = plt.Circle((0.5, 0.5), 0.20, color=ACCENT, alpha=0.03,
                      transform=ax.transAxes)
    ax.add_patch(glow)

    ax.text(0.5, 0.55, "PaRaVis", fontsize=48, fontweight="bold",
            color=TITLE_COLOR, ha="center", va="center",
            fontfamily=FONT_FAMILY, alpha=alpha)
    ax.text(0.5, 0.38, "Parallel Rao's Q . Visualization . Analysis",
            fontsize=14, color=SUBTITLE_COLOR, ha="center", va="center",
            fontfamily=FONT_FAMILY, alpha=max(0, alpha - 0.2))
    ax.text(0.5, 0.24, "Spectral Indices  .  Diversity  .  Raster Viz",
            fontsize=11, color=TEXT_COLOR, ha="center", va="center",
            fontfamily=FONT_FAMILY, alpha=max(0, alpha - 0.3))

    badge = FancyBboxPatch((0.38, 0.06), 0.24, 0.06,
                           facecolor=ACCENT, edgecolor="none",
                           alpha=alpha * 0.85, transform=ax.transAxes,
                           linewidth=0, boxstyle="round,pad=0.02")
    ax.add_patch(badge)
    ax.text(0.5, 0.09, "v2.0.0  .  MIT License", fontsize=9,
            color=TITLE_COLOR, ha="center", va="center",
            fontfamily=FONT_FAMILY, alpha=alpha)

    _add_frame_border(ax, alpha)
    _add_step(ax, 1, alpha=alpha)
    img = mpl_to_pil(fig)
    plt.close(fig)
    return img


def frame_raster_input(alpha: float = 1.0) -> Image.Image:
    """Raster data input concept."""
    fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    np.random.seed(42)
    grid = np.random.rand(20, 30) * 0.8 + 0.2
    grid[5:8, 10:15] = 0.9
    grid[12:16, 5:9] = 0.1

    # Raster panel with border
    raster_bg = FancyBboxPatch(
        (0.055, 0.32), 0.41, 0.44,
        facecolor="#0a1628", edgecolor=ACCENT2,
        alpha=alpha * 0.3, transform=ax.transAxes,
        linewidth=0.5, boxstyle="round,pad=0.01"
    )
    ax.add_patch(raster_bg)

    cmap = plt.cm.RdYlGn
    ax.imshow(grid, extent=[0.08, 0.44, 0.34, 0.74], cmap=cmap,
              aspect="auto", alpha=alpha)

    ax.text(0.26, 0.28, "Multi-band Raster Input", fontsize=10,
            color=SUBTITLE_COLOR, ha="center", fontfamily=FONT_FAMILY, alpha=alpha)

    # Arrow
    ax.annotate("", xy=(0.55, 0.54), xytext=(0.47, 0.54),
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2.5, alpha=alpha))

    # Band insets panel
    bands_bg = FancyBboxPatch(
        (0.57, 0.32), 0.38, 0.44,
        facecolor="#0a1628", edgecolor=ACCENT2,
        alpha=alpha * 0.3, transform=ax.transAxes,
        linewidth=0.5, boxstyle="round,pad=0.01"
    )
    ax.add_patch(bands_bg)

    for i, (label, cm) in enumerate([("Band 1", "Blues"), ("Band 2", "Greens"),
                                      ("Band N", "Oranges")]):
        x = 0.62 + i * 0.11
        arr = np.random.rand(10, 10) * 0.6 + 0.2
        axi = ax.inset_axes([x, 0.36, 0.09, 0.32])
        axi.imshow(arr, cmap=cm, aspect="auto", alpha=alpha)
        axi.axis("off")
        ax.text(x + 0.045, 0.32, label, fontsize=7, color=SUBTITLE_COLOR,
                ha="center", fontfamily=FONT_FAMILY, alpha=alpha)

    _add_frame_border(ax, alpha)
    _add_step(ax, 2, alpha=alpha)
    img = mpl_to_pil(fig)
    plt.close(fig)
    return img


def frame_indices(alpha: float = 1.0) -> Image.Image:
    """Spectral indices computation visual with card grid layout."""
    fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.88, "Spectral Indices Engine", fontsize=18, fontweight="bold",
            color=TITLE_COLOR, ha="center", fontfamily=FONT_FAMILY, alpha=alpha)

    # Subtle divider
    ax.plot([0.15, 0.85], [0.80, 0.80], color=ACCENT2, lw=1, alpha=alpha * 0.4)

    # 2x2 grid of index cards
    index_data = [
        ("NDVI", "(NIR - R) / (NIR + R)", "#00897b"),
        ("EVI",  "2.5(N-R)/(N+6R-7.5B+1)", "#00695c"),
        ("SAVI", "(NIR-R)/(NIR+R+L)*(1+L)", "#00796b"),
        ("NDWI", "(G - NIR) / (G + NIR)", "#004d40"),
    ]
    for idx, (name, formula, color) in enumerate(index_data):
        col = idx % 2
        row = idx // 2
        cx = 0.12 + col * 0.45
        cy = 0.62 - row * 0.22

        # Card background
        card = FancyBboxPatch(
            (cx - 0.02, cy - 0.08), 0.41, 0.18,
            facecolor="#0a1628", edgecolor=color,
            alpha=alpha * 0.6, transform=ax.transAxes,
            linewidth=1.2, boxstyle="round,pad=0.015"
        )
        ax.add_patch(card)

        # Badge
        badge = FancyBboxPatch(
            (cx + 0.01, cy + 0.04), 0.08, 0.045,
            facecolor=color, edgecolor="none",
            alpha=alpha * 0.9, transform=ax.transAxes,
            linewidth=0, boxstyle="round,pad=0.012"
        )
        ax.add_patch(badge)
        ax.text(cx + 0.05, cy + 0.062, name, fontsize=8, color=TITLE_COLOR,
                ha="center", va="center", fontfamily=MONO_FONT, alpha=alpha,
                fontweight="bold")

        # Formula
        ax.text(cx + 0.105, cy + 0.062, formula, fontsize=6.5, color=TEXT_COLOR,
                ha="left", va="center", fontfamily=MONO_FONT, alpha=alpha)

        # Small decorative dot
        ax.plot(cx + 0.32, cy + 0.062, "o", color=color, markersize=4, alpha=alpha)
        ax.plot([cx + 0.28, cx + 0.36], [cy + 0.062, cy + 0.062],
                color=color, lw=1.5, alpha=alpha * 0.5)

    # Bottom info bar
    code_bg = FancyBboxPatch(
        (0.05, 0.04), 0.90, 0.06,
        facecolor="#0a1628", edgecolor=ACCENT2,
        alpha=alpha * 0.35, transform=ax.transAxes,
        linewidth=0.5, boxstyle="round,pad=0.01"
    )
    ax.add_patch(code_bg)
    ax.text(0.5, 0.07, "Custom indices via @register_index  .  100+ built-in formulas",
            fontsize=9, color=SUBTITLE_COLOR, ha="center",
            fontfamily=FONT_FAMILY, alpha=alpha)

    _add_frame_border(ax, alpha)
    _add_step(ax, 3, alpha=alpha)
    img = mpl_to_pil(fig)
    plt.close(fig)
    return img


def frame_raoq(alpha: float = 1.0) -> Image.Image:
    """Rao's Q diversity computation visual."""
    fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.88, "Rao's Q Diversity Analysis", fontsize=18, fontweight="bold",
            color=TITLE_COLOR, ha="center", fontfamily=FONT_FAMILY, alpha=alpha)

    # Subtle divider
    ax.plot([0.15, 0.85], [0.80, 0.80], color=ACCENT2, lw=1, alpha=alpha * 0.4)

    np.random.seed(1)
    big = np.random.rand(15, 25) * 0.8 + 0.2
    big[3:8, 8:14] = np.random.rand(5, 6) * 0.5 + 0.5
    big[9:12, 4:7] = 0.1

    # ---- Left: Raster with scanning window ----
    raster_bg = FancyBboxPatch(
        (0.04, 0.28), 0.46, 0.46,
        facecolor="#0a1628", edgecolor=ACCENT2,
        alpha=alpha * 0.3, transform=ax.transAxes,
        linewidth=0.5, boxstyle="round,pad=0.01"
    )
    ax.add_patch(raster_bg)

    ax.imshow(big, extent=[0.06, 0.48, 0.30, 0.72], cmap="viridis",
              aspect="auto", alpha=alpha)

    # Scan line effect
    ax.plot([0.06, 0.48], [0.48, 0.48], color=ACCENT, lw=1.5, alpha=alpha * 0.6,
            transform=ax.transAxes, linestyle="--")

    # Moving window with highlight
    rect = plt.Rectangle((0.22, 0.44), 0.06, 0.08,
                         facecolor=ACCENT, edgecolor=ACCENT, lw=2,
                         alpha=alpha * 0.20, transform=ax.transAxes)
    ax.add_patch(rect)
    rect2 = plt.Rectangle((0.22, 0.44), 0.06, 0.08,
                          facecolor="none", edgecolor="#ffffff", lw=2,
                          alpha=alpha * 0.7, transform=ax.transAxes)
    ax.add_patch(rect2)

    ax.text(0.25, 0.41, "sliding window", fontsize=6, color=ACCENT,
            ha="center", va="top", fontfamily=FONT_FAMILY, alpha=alpha)

    # ---- Formula (centered below raster) ----
    formula_bg = FancyBboxPatch(
        (0.08, 0.20), 0.38, 0.07,
        facecolor="#0a1628", edgecolor=ACCENT,
        alpha=alpha * 0.5, transform=ax.transAxes,
        linewidth=1.0, boxstyle="round,pad=0.01"
    )
    ax.add_patch(formula_bg)
    ax.text(0.27, 0.235, "Q = Sum_i Sum_j d_ij / n(n-1)",
            fontsize=9.5, color=SUBTITLE_COLOR, ha="center",
            fontfamily=FONT_FAMILY, alpha=alpha)

    # ---- Right column: Backends + Parameters ----
    # Single unified card
    unified_bg = FancyBboxPatch(
        (0.52, 0.20), 0.44, 0.56,
        facecolor="#0a1628", edgecolor=ACCENT2,
        alpha=alpha * 0.3, transform=ax.transAxes,
        linewidth=0.8, boxstyle="round,pad=0.015"
    )
    ax.add_patch(unified_bg)

    # Section: Backends
    ax.text(0.74, 0.72, "Backends", fontsize=8, color=MUTED,
            ha="center", fontfamily=FONT_FAMILY, alpha=alpha)
    backends = [("CPU", "#1565c0"), ("Parallel", "#00695c"), ("GPU", "#4a148c")]
    for i, (name, c) in enumerate(backends):
        x = 0.57 + i * 0.13
        badge = FancyBboxPatch((x - 0.04, 0.62), 0.08, 0.04,
                                facecolor=c, edgecolor="none",
                                alpha=alpha * 0.85, transform=ax.transAxes,
                                linewidth=0, boxstyle="round,pad=0.012")
        ax.add_patch(badge)
        ax.text(x, 0.64, name, fontsize=6.5, color=TITLE_COLOR,
                ha="center", va="center", fontfamily=FONT_FAMILY, alpha=alpha,
                fontweight="bold")

    # Divider
    ax.plot([0.55, 0.93], [0.58, 0.58], color=ACCENT2, lw=0.8, alpha=alpha * 0.3)

    # Section: Distance Metrics (vertical list)
    ax.text(0.74, 0.545, "Distance Metrics", fontsize=7, color=MUTED,
            ha="center", fontfamily=FONT_FAMILY, alpha=alpha)
    metrics = ["Euclidean", "Manhattan", "Chebyshev", "Minkowski"]
    for i, m in enumerate(metrics):
        xb = 0.56 + (i % 2) * 0.18
        yb = 0.49 - (i // 2) * 0.045
        ax.plot([xb, xb + 0.02], [yb, yb], color=ACCENT, lw=2, alpha=alpha * 0.5)
        ax.text(xb + 0.025, yb, m, fontsize=6.5, color=TEXT_COLOR,
                ha="left", va="center", fontfamily=FONT_FAMILY, alpha=alpha)

    # Divider
    ax.plot([0.55, 0.93], [0.39, 0.39], color=ACCENT2, lw=0.8, alpha=alpha * 0.3)

    # Section: Parameters (vertical list)
    ax.text(0.74, 0.355, "Parameters", fontsize=7, color=MUTED,
            ha="center", fontfamily=FONT_FAMILY, alpha=alpha)
    params = ["Batch Processing", "NA Tolerance", "Step Size"]
    for i, p in enumerate(params):
        xb = 0.56 + (i % 2) * 0.18
        yb = 0.30 - (i // 2) * 0.045
        ax.plot([xb, xb + 0.02], [yb, yb], color=ACCENT, lw=2, alpha=alpha * 0.5)
        ax.text(xb + 0.025, yb, p, fontsize=6.5, color=TEXT_COLOR,
                ha="left", va="center", fontfamily=FONT_FAMILY, alpha=alpha)

    _add_frame_border(ax, alpha)
    _add_step(ax, 4, alpha=alpha)
    img = mpl_to_pil(fig)
    plt.close(fig)
    return img


def frame_visualization(alpha: float = 1.0) -> Image.Image:
    """Visualization capabilities."""
    fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.93, "Visualization Suite", fontsize=20, fontweight="bold",
            color=TITLE_COLOR, ha="center", fontfamily=FONT_FAMILY, alpha=alpha)

    np.random.seed(7)
    data = np.random.randn(1000)

    # Left panel background
    left_bg = FancyBboxPatch(
        (0.04, 0.04), 0.56, 0.86,
        facecolor="#0a1628", edgecolor=ACCENT2,
        alpha=alpha * 0.25, transform=ax.transAxes,
        linewidth=0.5, boxstyle="round,pad=0.01"
    )
    ax.add_patch(left_bg)

    # Histogram (left upper)
    axi1 = ax.inset_axes([0.06, 0.54, 0.22, 0.30])
    axi1.hist(data, bins=25, color=ACCENT, alpha=alpha * 0.8, edgecolor="none")
    axi1.set_facecolor("none")
    for spine in axi1.spines.values():
        spine.set_visible(False)
    axi1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.text(0.17, 0.51, "Statistics", fontsize=8, color=SUBTITLE_COLOR,
            ha="center", fontfamily=FONT_FAMILY, alpha=alpha)

    # Difference map (middle upper)
    axi2 = ax.inset_axes([0.36, 0.54, 0.22, 0.30])
    diff = np.random.randn(20, 30)
    axi2.imshow(diff, cmap="RdBu", aspect="auto", alpha=alpha)
    axi2.axis("off")
    ax.text(0.47, 0.51, "Diff + Heatmap", fontsize=8, color=SUBTITLE_COLOR,
            ha="center", fontfamily=FONT_FAMILY, alpha=alpha)

    # Time series (bottom-left)
    axi3 = ax.inset_axes([0.06, 0.06, 0.50, 0.30])
    t = np.linspace(0, 4 * np.pi, 100)
    for j in range(3):
        axi3.plot(t, np.sin(t - j * 0.5) + np.random.randn(100) * 0.1,
                  color=["#e53935", "#43a047", "#1e88e5"][j],
                  lw=1.5, alpha=alpha * 0.8)
    axi3.set_facecolor("none")
    for spine in axi3.spines.values():
        spine.set_visible(False)
    axi3.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.text(0.31, 0.03, "Time Series . GIF Export", fontsize=8, color=SUBTITLE_COLOR,
            ha="center", fontfamily=FONT_FAMILY, alpha=alpha)

    # Right panel -- feature list
    right_bg = FancyBboxPatch(
        (0.64, 0.04), 0.34, 0.86,
        facecolor="#0a1628", edgecolor=ACCENT2,
        alpha=alpha * 0.25, transform=ax.transAxes,
        linewidth=0.5, boxstyle="round,pad=0.01"
    )
    ax.add_patch(right_bg)
    ax.text(0.81, 0.86, "Plot Types", fontsize=9, color=MUTED,
            ha="center", fontfamily=FONT_FAMILY, alpha=alpha)

    types = ["Individual", "Statistics", "Split View",
             "Diff Map", "Time Series", "GIF Export"]
    for i, t in enumerate(types):
        y = 0.78 - i * 0.09
        if y < 0.08:
            break
        ax.plot([0.68, 0.71], [y, y], color=ACCENT, lw=2, alpha=alpha)
        ax.text(0.72, y, t, fontsize=9, color=TEXT_COLOR,
                ha="left", va="center", fontfamily=FONT_FAMILY, alpha=alpha)

    _add_frame_border(ax, alpha)
    _add_step(ax, 5, alpha=alpha)
    img = mpl_to_pil(fig)
    plt.close(fig)
    return img


def frame_gui(alpha: float = 1.0) -> Image.Image:
    """GUI application concept."""
    fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.95, "Desktop GUI", fontsize=20, fontweight="bold",
            color=TITLE_COLOR, ha="center", fontfamily=FONT_FAMILY, alpha=alpha)

    # Menu bar
    menu = FancyBboxPatch((0.06, 0.76), 0.88, 0.04,
                           facecolor="#1a2a3a", edgecolor="#2a3a4a",
                           alpha=alpha * 0.9, transform=ax.transAxes,
                           linewidth=0.5, boxstyle="round,pad=0.008")
    ax.add_patch(menu)
    for i, label in enumerate(["File", "View", "Processing", "Help"]):
        ax.text(0.10 + i * 0.08, 0.78, label, fontsize=6, color=TEXT_COLOR,
                ha="center", va="center", fontfamily=FONT_FAMILY, alpha=alpha)

    # Three columns
    col_y = 0.12
    col_h = 0.56
    col_data = [
        ("Indices", 0.06, col_y, 0.28, col_h, ACCENT,
         ["NDVI", "EVI", "SAVI", "NDWI"]),
        ("Rao's Q", 0.36, col_y, 0.28, col_h, "#00796b",
         ["Window: 15", "Distance: Euclid", "Backend: GPU"]),
        ("Viz",     0.66, col_y, 0.28, col_h, "#00695c",
         ["Plot: NDVI", "Color: Viridis", "Save: PNG"]),
    ]
    for title, x, y, w, h, color, items in col_data:
        panel = FancyBboxPatch((x, y), w, h, facecolor="#112233",
                                edgecolor=color, alpha=alpha * 0.85,
                                transform=ax.transAxes, linewidth=1.5,
                                boxstyle="round,pad=0.015")
        ax.add_patch(panel)
        hdr = FancyBboxPatch((x, y + h - 0.04), w, 0.04,
                              facecolor=color, alpha=alpha * 0.8,
                              transform=ax.transAxes, linewidth=0,
                              boxstyle="round,pad=0.008")
        ax.add_patch(hdr)
        ax.text(x + w / 2, y + h - 0.02, title, fontsize=7, color=TITLE_COLOR,
                ha="center", va="center", fontfamily=FONT_FAMILY, alpha=alpha,
                fontweight="bold")
        for j, item in enumerate(items):
            ax.text(x + 0.02, y + h - 0.08 - j * 0.05, f". {item}",
                    fontsize=6.5, color=TEXT_COLOR, ha="left", va="center",
                    fontfamily=MONO_FONT, alpha=alpha)

    # Status bar
    status = FancyBboxPatch((0.06, 0.03), 0.88, 0.03,
                             facecolor="#1a2a3a", edgecolor="none",
                             alpha=alpha * 0.8, transform=ax.transAxes,
                             linewidth=0, boxstyle="round,pad=0.008")
    ax.add_patch(status)
    ax.text(0.10, 0.045, "Ready  .  GPU: RTX 3090  .  32 Cores  .  64 GB RAM",
            fontsize=6, color=TEXT_COLOR, ha="left", va="center",
            fontfamily=FONT_FAMILY, alpha=alpha)

    _add_frame_border(ax, alpha)
    _add_step(ax, 6, alpha=alpha)
    img = mpl_to_pil(fig)
    plt.close(fig)
    return img


def frame_summary(alpha: float = 1.0) -> Image.Image:
    """Summary / closing card."""
    fig, ax = plt.subplots(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Decorative wave
    for i in range(40):
        x = i / 40
        y = 0.88 + 0.03 * np.sin(i * 0.8)
        circle = plt.Circle((x, y), 0.007, color=ACCENT, alpha=alpha * 0.08,
                            transform=ax.transAxes)
        ax.add_patch(circle)

    ax.text(0.5, 0.72, "PaRaVis", fontsize=36, fontweight="bold",
            color=TITLE_COLOR, ha="center", fontfamily=FONT_FAMILY, alpha=alpha)
    ax.text(0.5, 0.60, "Parallel Rao's Q -- Visualization & Analysis",
            fontsize=13, color=SUBTITLE_COLOR, ha="center",
            fontfamily=FONT_FAMILY, alpha=max(0, alpha - 0.1))
    ax.text(0.5, 0.48, "pip install paravis", fontsize=11, color=ACCENT,
            ha="center", fontfamily=MONO_FONT, alpha=alpha)

    features = ["[I] Indices", "[Q] Rao's Q", "[V] Viz", "[G] GUI", "[P] GPU"]
    for i, f in enumerate(features):
        x = 0.15 + i * 0.17
        badge = FancyBboxPatch(
            (x - 0.055, 0.33), 0.11, 0.045,
            facecolor="#0a1628", edgecolor=ACCENT2,
            alpha=alpha * 0.5, transform=ax.transAxes,
            linewidth=0.5, boxstyle="round,pad=0.008"
        )
        ax.add_patch(badge)
        ax.text(x, 0.352, f, fontsize=8, color=TEXT_COLOR,
                ha="center", va="center", fontfamily=FONT_FAMILY, alpha=alpha)

    ax.text(0.5, 0.12, "MIT License  .  github.com/mmadrz/paravis",
            fontsize=8, color=MUTED, ha="center",
            fontfamily=FONT_FAMILY, alpha=alpha)

    _add_frame_border(ax, alpha)
    _add_step(ax, 7, alpha=alpha)
    img = mpl_to_pil(fig)
    plt.close(fig)
    return img


# -- Easing & transition helpers ------------------------------------

def ease_in_out_sine(t: float) -> float:
    """Smooth sinusoidal ease-in-out curve (more natural than linear)."""
    return -(math.cos(math.pi * t) - 1) / 2


# -- Build the animation --------------------------------------------
def build_gif(output_path: Path, fps: float = FPS):
    """Generate all frames and assemble the GIF with smooth transitions."""
    print("[GENERATING] PaRaVis graphical abstract...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_defs = [
        (frame_title, 10),
        (frame_raster_input, 8),
        (frame_indices, 10),
        (frame_raoq, 10),
        (frame_visualization, 10),
        (frame_gui, 10),
        (frame_summary, 14),
    ]

    # Smooth transition parameters
    crossfade_steps = 8
    hold_end_steps = 2

    all_frames: list[Image.Image] = []

    for idx, (frame_func, hold) in enumerate(frame_defs):
        print(f"  Frame {idx + 1}/{len(frame_defs)}: {frame_func.__name__} ...")
        full_img = frame_func(alpha=1.0)

        for _ in range(hold):
            all_frames.append(full_img.copy())

        if idx < len(frame_defs) - 1:
            next_func = frame_defs[idx + 1][0]
            next_img = next_func(alpha=1.0)

            for _ in range(hold_end_steps):
                all_frames.append(full_img.copy())

            for step in range(1, crossfade_steps + 1):
                t = ease_in_out_sine(step / (crossfade_steps + 1))
                blended = Image.blend(full_img, next_img, t)
                all_frames.append(blended)

            for _ in range(hold_end_steps):
                all_frames.append(next_img.copy())

    print(f"  Total frames: {len(all_frames)}")

    rgb_frames = [frame.convert("RGB") for frame in all_frames]

    paletted = []
    for frame in rgb_frames:
        q = frame.quantize(
            method=Image.Quantize.FASTOCTREE,
            dither=Image.Dither.FLOYDSTEINBERG,
            colors=96,
        )
        paletted.append(q)

    total_frames = len(paletted)
    target_duration_ms = 10000
    ms_per_frame = max(20, target_duration_ms // total_frames)

    print(f"  Saving GIF ({total_frames} frames, {ms_per_frame}ms each) "
          f"to {output_path} ...")
    paletted[0].save(
        output_path,
        save_all=True,
        append_images=paletted[1:],
        duration=ms_per_frame,
        loop=0,
        optimize=False,
    )

    size_kb = output_path.stat().st_size / 1024
    print(f"  Done! Saved to {output_path} ({size_kb:.1f} KB)")


# -- Entry point ----------------------------------------------------
if __name__ == "__main__":
    build_gif(OUTPUT_PATH)
