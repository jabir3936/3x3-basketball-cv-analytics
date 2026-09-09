"""
Run mapping fix pipeline: refit H (optional) → recompute feet → project → smooth.

Usage:
  python scripts/03_processing/run_mapping_pipeline.py
  python scripts/03_processing/run_mapping_pipeline.py --skip-refit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd):
    print("\n==>", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-refit", action="store_true", help="keep current calibration JSON")
    ap.add_argument("--skip-smooth", action="store_true")
    ap.add_argument("--skip-feet-overlay", action="store_true")
    ap.add_argument("--k1", type=float, default=-0.16, help="passed to refit when not skipped")
    args = ap.parse_args()

    py = sys.executable

    if not args.skip_refit:
        run([py, "scripts/02_calibration/refit_official_homography.py", "--k1", str(args.k1)])

    run([py, "scripts/03_processing/recompute_feet_and_project.py"])

    if not args.skip_smooth:
        run([py, "scripts/03_processing/match_tracking_smooth.py"])

    if not args.skip_feet_overlay:
        run([py, "scripts/diagnostics/overlay_feet_check.py"])

    print("\nPipeline done.")
    print("  Check: media/checks/official_calibration_checkv2.jpg")
    print("  Check: media/checks/feet_check_frame_*.jpg")
    print("  Data:  data/tracking/latest/match_tracking_data_official.json")
    print("  Data:  data/tracking/latest/match_tracking_data_smooth.json")
    print("  Render short clip:")
    print("    python scripts/diagnostics/render_mapping_preview.py")


if __name__ == "__main__":
    main()
