"""Render the website's matching training-reward panels.

Run with Python, NumPy, and Matplotlib. SafeMaze and LLM-Math use coordinates
recovered from the paper's vector figures; provenance is in the JSON file.
The Minesweeper trace is digitized from the researcher's W&B screenshot and
rescaled to the requested 0.4 plateau and 250 steps. Provenance is in the JSON.
"""

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "grace-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"
DATA = json.loads((ASSETS / "training-reward-data.json").read_text())
TEXT = "#352b24"
MUTED = "#756457"
PANEL = "#fffdf9"
GRID = "#ded3c5"
COLORS = {"RLOO": "#8a629e"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    # Outline glyphs so figures retain their fonts on phones without DejaVu.
    "svg.fonttype": "path",
    "svg.hashsalt": "graceful-training-reward",
    "pdf.fonttype": 42,
})


def draw(ax, title, series, steps, ticks):
    ax.set_facecolor(PANEL)
    ax.set_title(title, fontsize=15, fontweight="medium", pad=13)
    ax.set_xlabel("Training step", fontsize=13)
    ax.set_ylabel("Training reward", fontsize=13)
    ax.set_xlim(0, steps)
    ax.set_ylim(0, 0.5)
    ax.set_xticks(ticks)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.tick_params(labelsize=11.5, length=3, width=0.7)
    ax.grid(axis="y", color=GRID, linewidth=0.65, alpha=0.75)
    ax.set_axisbelow(True)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color("#a48d76")
        ax.spines[edge].set_linewidth(0.7)
    for method, coordinates in series.items():
        xy = np.array(coordinates)
        ax.plot(xy[:, 0], xy[:, 1], label=method, color=COLORS[method],
                linewidth=2.4)
    ax.axhline(0.4, color=MUTED, linewidth=1.15, linestyle=(0, (5, 4)))
    ax.text(0.97, 0.88, "Fallback = 0.4", transform=ax.transAxes,
            ha="right", color=MUTED, fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=11, handlelength=2.2)


def main():
    mine = DATA["minesweeper"]["series"]
    assert len(mine["RLOO"]) == 251
    assert mine["RLOO"][0] == [0, 0.02] and mine["RLOO"][-1] == [250, 0.4]
    panels = [
        ("safemaze", "SafeMaze", {"RLOO": DATA["safemaze"]["series"]["RLOO"]}, 30, [0, 10, 20, 30]),
        ("llm-math", "LLM-Math", DATA["llm_math"]["series"], 1000, [0, 500, 1000]),
        ("minesweeper", "Minesweeper", mine, 250, [0, 125, 250]),
    ]
    for slug, title, series, limit, ticks in panels:
        fig, ax = plt.subplots(figsize=(3.8, 3.25), facecolor=PANEL)
        fig.subplots_adjust(left=0.20, right=0.95, bottom=0.20, top=0.85)
        draw(ax, title, series, limit, ticks)
        description = ("Minesweeper RLOO reward curve traced from the researcher's W&B screenshot, "
                       "rescaled to start at 0.02 and reach 0.4 over 250 steps."
                       if slug == "minesweeper" else
                       "Redrawn from the paper's plotted vector coordinates. "
                       "See training-reward-data.json for provenance.")
        fig.savefig(ASSETS / f"training-reward-{slug}.svg",
                    metadata={"Title": f"{title}: training reward", "Description": description, "Date": None})
        fig.savefig(ASSETS / f"training-reward-{slug}.png", dpi=240,
                    metadata={"Description": description})
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.6), facecolor=PANEL)
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.25, top=0.85, wspace=0.42)
    for ax, (_, title, series, limit, ticks) in zip(axes, panels):
        draw(ax, title, series, limit, ticks)
    fig.text(0.5, 0.035,
             "SafeMaze and LLM-Math: redrawn from paper figures. Minesweeper: screenshot trace rescaled to 0.4.",
             ha="center", fontsize=10, color=MUTED)
    fig.savefig(ASSETS / "training-reward-comparison.pdf",
                metadata={"Title": "Fallback collapse: training-reward comparison",
                          "Subject": "Minesweeper trace digitized and rescaled from a W&B screenshot; SafeMaze and LLM-Math are redrawn from paper figures."})
    plt.close(fig)


if __name__ == "__main__":
    main()
