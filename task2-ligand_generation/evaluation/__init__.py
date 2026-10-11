"""Evaluation of generated ligands: molecular quality, AutoDock Vina, PoseCheck and PoseBusters.

Run from task2-ligand_generation/ as `python -m evaluation.eval <run_dir>`; see eval.py.

TargetDiff's evaluation code is vendored unchanged under targetdiff/ and keeps its own package
names (utils.evaluation.*, utils.reconstruct), so that folder goes on sys.path here.
"""
import sys
from pathlib import Path

TARGETDIFF = Path(__file__).resolve().parent / "targetdiff"
if str(TARGETDIFF) not in sys.path:
    sys.path.insert(0, str(TARGETDIFF))
