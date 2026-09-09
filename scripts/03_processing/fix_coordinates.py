"""Thin wrapper: recompute feet + project (same as recompute_feet_and_project)."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("recompute_feet_and_project.py")), run_name="__main__")
