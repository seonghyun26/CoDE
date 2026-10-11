"""Conda environments found by name, for code that calls another environment's tools.

No path is assumed: conda (CONDA_EXE, else `conda` on PATH) is asked where the environment lives,
so the same code works wherever the envs are installed. Callers keep their own env-var override.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional


# conda's executable: CONDA_EXE, conda on PATH, else a standard install in the home directory
# (a non-interactive shell has not run conda init).
def _conda_exe() -> Optional[str]:
    exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if exe:
        return exe
    for base in ("miniforge3", "mambaforge", "miniconda3", "anaconda3"):
        candidate = Path.home() / base / "bin" / "conda"
        if candidate.is_file():
            return str(candidate)
    return None


# Prefix of conda environment NAME, or None when conda is missing or does not know it.
@lru_cache(maxsize=None)
def conda_env_prefix(name: str) -> Optional[Path]:
    exe = _conda_exe()
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "env", "list", "--json"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    for prefix in json.loads(out).get("envs", []):
        if Path(prefix).name == name:
            return Path(prefix)
    return None


# Python of conda environment NAME; `fallback` when the environment is not found.
def conda_env_python(name: str, fallback: str = "python") -> str:
    prefix = conda_env_prefix(name)
    return str(prefix / "bin" / "python") if prefix else fallback
