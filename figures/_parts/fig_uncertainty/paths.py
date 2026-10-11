"""Canonical output locations for uncertainty figures and their supporting tables.

Plot/analysis entry points stay in figures/. This module performs no file I/O.
Raw predictions, feature caches and trained models remain under results/.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "figures" / "fig-uncertainty"
MOTIVATION = BASE / "motivation"
RANKING_METRICS = BASE / "affinity" / "metric-review" / "ranking-by-metric"
MOTIVATION_ARCHIVE = BASE / "exploratory" / "motivation-layouts-20260921"
AFFINITY = BASE / "affinity"
CL3 = AFFINITY / "cl3-5seed"
LP_CL3 = AFFINITY / "lp-trained-cl3-filtered"
METRIC_REVIEW = AFFINITY / "metric-review"
REPRESENTATION = BASE / "representation"
INPUT_DENSITY = BASE / "input-density"
ATOMIC_B = BASE / "controls" / "atomic-b-spatial"
STANDARDIZED_B = BASE / "controls" / "standardized-b-concat"
EXPLORATORY_CL3 = BASE / "exploratory" / "cl3-filtered"
MAP_DIAGNOSTICS = EXPLORATORY_CL3 / "density-uncertainty"
