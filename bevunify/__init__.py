"""bevunify — one Hydra project to train/eval every BEV-seg model on GaussianLSS GT."""
import os
import sys
from pathlib import Path

# Put the GaussianLSS host repo on the path so `from GaussianLSS...` works.
# Default = vendored copy under third_party/ (override with GAUSSIANLSS_ROOT).
_DEFAULT_HOST = str(Path(__file__).resolve().parents[1] / "third_party" / "GaussianLSS")
_HOST = os.environ.get("GAUSSIANLSS_ROOT") or _DEFAULT_HOST
if os.path.isdir(_HOST) and _HOST not in sys.path:
    sys.path.insert(0, _HOST)
