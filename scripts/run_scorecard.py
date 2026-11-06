"""Run the full scorecard build and validation end to end.

    python scripts/run_scorecard.py            # writes outputs/*.csv and outputs/charts/*.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scorecard.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
