"""Run the evaluation-only multi-geometry Junction trigger benchmark."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pygame_simulator.junction_trigger_multigeometry_evaluation import main


if __name__ == "__main__":
    main()
