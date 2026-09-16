"""
Run mapping pipeline: camera motion → feet/filter → project → smooth.

Usage:
  python scripts/03_processing/run_mapping_pipeline.py
  python scripts/03_processing/run_mapping_pipeline.py --legacy-refit
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
    ap.add_argument(
        "--legacy-refit", action="store_true",
        help="Overwrite the current calibration with the old fixed five-point refit (not recommended).",
    )
    ap.add_argument("--skip-smooth", action="store_true")
    ap.add_argument("--skip-feet-overlay", action="store_true")
    ap.add_argument("--k1", type=float, default=0.0, help="measured k1 passed only to --legacy-refit")
    args = ap.parse_args()

    py = sys.executable

    if args.legacy_refit:
        run([py, "scripts/02_calibration/refit_official_homography.py", "--k1", str(args.k1)])
    else:
        print("Keeping current validated calibration JSON.")

    run([py, "scripts/03_processing/estimate_camera_motion.py"])
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
