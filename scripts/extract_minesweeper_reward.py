"""Digitize the user's W&B screenshot for the website's Minesweeper panel.

Run: python scripts/extract_minesweeper_reward.py path/to/screenshot.png
The screenshot's 0.5 plateau is rescaled to 0.4, and 200 displayed steps
are stretched to 250. The extracted points are an approximate reconstruction.
"""

from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


EXPECTED_SHA256 = "c63cf835d571ea8834d68f3396faf10c7b68b1bb6939a5a0dcc7d098535d82f5"
DATA = Path(__file__).resolve().parents[1] / "docs" / "assets" / "training-reward-data.json"


def main(source: Path):
    if sha256(source.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("This extractor is calibrated to the supplied screenshot.")
    image = np.asarray(Image.open(source).convert("RGB"))
    assert image.shape == (630, 1118, 3)

    # The cyan trace spans pixels x=94..1022 and y=135..488 in this export.
    # Its solid core color appears in every column, avoiding UI and grid lines.
    left, right, top, bottom = 94, 1022, 135, 488
    color = np.array([151, 204, 191])
    pixels = image[top:bottom + 1, left:right + 1]
    line = np.all(pixels == color, axis=2)
    if not np.all(line.any(axis=0)):
        raise ValueError("The screenshot line could not be traced in every column.")
    y_pixels = np.array([top + np.median(np.flatnonzero(line[:, x]))
                         for x in range(line.shape[1])])

    steps = np.arange(251)
    original_x = np.linspace(left, right, len(steps))
    traced_y = np.interp(original_x, np.arange(left, right + 1), y_pixels)
    screenshot_reward = (bottom - traced_y) * 0.5 / (bottom - 136)
    # Preserve the screenshot's rise, dips, and late plateau while applying
    # the requested endpoints. The source does not provide raw W&B samples.
    reward = 0.02 + 0.38 * (screenshot_reward - screenshot_reward[0]) / (
        screenshot_reward[-1] - screenshot_reward[0])
    reward[0], reward[-1] = 0.02, 0.4

    data = json.loads(DATA.read_text())
    data["minesweeper"] = {
        "source": {"file": source.name, "sha256": EXPECTED_SHA256},
        "note": ("Teal line digitized from a W&B screenshot supplied by the researcher. "
                 "The screenshot spans about 200 steps and plateaus at 0.5; x is stretched "
                 "to 250 steps and y is linearly rescaled to start at 0.02 and plateau at 0.4. "
                 "These are approximate image-derived values, not raw log samples."),
        "series": {"RLOO": [[int(step), round(float(value), 6)]
                            for step, value in zip(steps, reward)]},
    }
    DATA.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/extract_minesweeper_reward.py path/to/screenshot.png")
    main(Path(sys.argv[1]))
