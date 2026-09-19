# Court-vision-3x3

<img width="800" height="288" alt="ezgif-55a7b6abacccf723" src="https://github.com/user-attachments/assets/36e1ec2c-471f-47d8-8503-36d688411408" />


3x3 basketball analytics from a single broadcast camera: player & ball tracking, metric court mapping, team classification, a possession/clearance FSM, and live top-down 2D court visualization.

## Pipeline

1. **Detection & Tracking** — RT-DETR + BoT-SORT on broadcast footage; a circuit breaker suspends analytics during cuts/replays/close-ups.
2. **Interpolation** — fills short detection gaps for players and ball.
3. **Court Calibration** — 6-point pixel→meter homography anchored to official FIBA 3x3 dimensions (15×11 m court, 4.90×5.80 m key, 6.75 m arc, hoop at (0, 1.575)); validated by re-projecting the court overlay onto the video (≤0.15 m error).
4. **Team Mapping** — manual segment labels + HSV jersey-color auto-labeling + 3-vs-3 game-rule constraint repair.
5. **Possession FSM** — ball state (detected/interpolated/occluded/lost), debounced possession with ID-switch handover, shot zone (inside/outside 6.75 m arc), and 3x3 clearance state.
6. **Visualization** — side-by-side slow-motion video: diagnostic overlay (boxes, IDs, trails) + 2D top-down court map with FSM HUD.

## Key Scripts

| Script | Purpose |
|---|---|
| `court_calibration.py` | interactive homography calibration + async frame streamer |
| `diagnostic_visualizer.py` | tracking playback with boxes, IDs, trajectory tails |
| `team_map_place.py` | applies team-mapping CSV to tracking JSON |
| `render_with_2dmap.py` | final side-by-side 2D-map renderer (slow motion) |

Supporting steps (tracking, interpolation, re-projection, auto-labeling, FSM, sanity checks) run as notebook cells / utility scripts over the JSON telemetry chain.

## Usage

```bash
python court_calibration.py        # click 6 court landmarks -> calibration_official.json
# tracking + interpolation -> match_tracking_data_interpolated.json
# team labeling + team_map_place.py -> match_tracking_data_teams.json
# re-projection -> match_tracking_data_official.json
# FSM -> match_fsm_data.json
python render_with_2dmap.py        # -> match_video_diagnostic_with_map.mp4
