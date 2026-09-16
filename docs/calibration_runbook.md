# 2D Map Calibration Runbook

Run each command in the VS Code/Cursor terminal from the project root:

```bash
cd /Users/jay/Desktop/Portfolio/Sport_Analytics
```

## 1. Create a validated homography

```bash
python scripts/02_calibration/calibrate_court_interactive.py
```

- Click the six green `FIT` landmarks as precisely as possible.
- Click the orange `CHECK` landmarks only if they are clearly visible; press `n` to skip one that is not visible.
- Press `s` after entering the landmarks.
- The script saves only if held-out check points meet both limits: mean error <= 0.50 m and worst point <= 0.75 m.
- Start without `--k1`. Do not use a guessed lens-distortion value.

Inspect:

```text
media/checks/official_calibration_check_interactive.jpg
data/calibration/calibration_validation_report.json
```

If validation fails, press `u` to undo the relevant fit point and re-click it. Do not continue with a failed report.

## 2. Mark the playable-floor boundary

```bash
python scripts/02_calibration/select_court_floor_polygon.py
```

Click 4–12 points just inside the playable court boundary, moving around it in one direction. Exclude advertising boards, photographers, benches, and referee areas. Save with `s`.

Inspect:

```text
media/checks/court_floor_polygon_check.jpg
```

## 3. Reproject existing detections; no RT-DETR retraining is required

```bash
python scripts/03_processing/run_mapping_pipeline.py
```

This preserves the validated calibration, estimates camera motion using court-floor features, rejects feet outside the selected court polygon, reprojects your existing tracking JSON, and smooths it.

Inspect:

```text
media/checks/feet_check_frame_*.jpg
media/checks/dynamic_court_check_frame_*.jpg
```

## 4. Test, then render the full map

```bash
python scripts/diagnostics/render_mapping_preview.py
python scripts/06_visualization/render_with_2dmap.py
```

The renderer uses `data/tracking/latest/match_tracking_data_smooth.json`, shows a maximum of three reliable players per team, and writes:

```text
media/renders/match_video_diagnostic_with_map.mp4
```

## Do not use

Avoid `scripts/02_calibration/refit_official_homography.py` unless you are deliberately comparing the legacy calibration. It uses the prior fixed points and can overwrite the validated calibration.
