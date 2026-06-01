"""bevunify — one Hydra project to train/eval every BEV-seg model on GaussianLSS GT."""
import os
import sys

# Put the GaussianLSS host repo on the path so `from GaussianLSS...` works.
_HOST = os.environ.get("GAUSSIANLSS_ROOT", "/home/jeongtae/bevseg/GaussianLSS")
if os.path.isdir(_HOST) and _HOST not in sys.path:
    sys.path.insert(0, _HOST)
