# Ball trajectory interpolation stage for 3x3 basketball tracking JSON.
#
# This script reads tracking output JSON, finds short missing ball gaps,
# interpolates the ball position, and writes a new JSON file.
#
# It is designed to be run after RT-DETR + BoT-SORT tracking.

import json
import math
import numpy as np

# Optional cubic/PCHIP interpolation.
# If SciPy is not installed, the script falls back to linear interpolation.
try:
    from scipy.interpolate import PchipInterpolator
except Exception:
    PchipInterpolator = None


def interpolate_2d_series(frame_ids, values, query_ids, use_cubic=True):
    """
    Interpolate a 2D coordinate sequence.

    Args:
        frame_ids: list of known frame IDs
        values: list of [x, y] values at those frame IDs
        query_ids: frame IDs to interpolate
        use_cubic: use PCHIP cubic interpolation if available

    Returns:
        interpolated_points, method
    """
    if len(frame_ids) < 2 or len(query_ids) == 0:
        return None, "none"

    x = np.array(frame_ids, dtype=float)
    y = np.array(values, dtype=float)
    q = np.array(query_ids, dtype=float)

    # Sort by frame ID.
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Remove duplicate frame IDs if any.
    unique_mask = np.concatenate(([True], np.diff(x) > 0))
    x = x[unique_mask]
    y = y[unique_mask]

    if len(x) < 2:
        return None, "none"

    # If SciPy is unavailable or interpolation is too short, use linear.
    if (
        len(x) == 2 or
        not use_cubic or
        PchipInterpolator is None or
        len(x) < 3
    ):
        x_interp = np.interp(q, x, y[:, 0])
        y_interp = np.interp(q, x, y[:, 1])
        method = "linear"

    else:
        try:
            px = PchipInterpolator(x, y[:, 0], extrapolate=False)
            py = PchipInterpolator(x, y[:, 1], extrapolate=False)

            x_interp = px(q)
            y_interp = py(q)

            # If PCHIP returns NaN outside support, fall back to linear.
            if np.any(np.isnan(x_interp)) or np.any(np.isnan(y_interp)):
                x_linear = np.interp(q, x, y[:, 0])
                y_linear = np.interp(q, x, y[:, 1])

                x_interp = np.where(np.isnan(x_interp), x_linear, x_interp)
                y_interp = np.where(np.isnan(y_interp), y_linear, y_interp)

            method = "pchip"

        except Exception:
            x_interp = np.interp(q, x, y[:, 0])
            y_interp = np.interp(q, x, y[:, 1])
            method = "linear"

    interpolated_points = [
        [float(x_interp[i]), float(y_interp[i])]
        for i in range(len(q))
    ]

    return interpolated_points, method


def interpolate_ball_payload(
    payload,
    fps=30.0,
    max_gap_frames=12,
    max_gap_seconds=None,
    occlusion_radius_m=1.2,
    occlusion_pixel_radius=90,
    max_speed_mps=25.0,
    hard_speed_limit_mps=45.0,
    use_cubic=True,
    support_window_frames=80
):
    """
    Interpolate missing ball locations inside a tracking payload.

    Args:
        payload: tracking JSON payload
        fps: video frame rate
        max_gap_frames: maximum gap length to interpolate
        max_gap_seconds: optional override for max_gap_frames
        occlusion_radius_m: court-meter distance to mark occlusion
        occlusion_pixel_radius: fallback pixel distance for occlusion
        max_speed_mps: soft speed threshold for warning
        hard_speed_limit_mps: skip interpolation if speed exceeds this
        use_cubic: use cubic/PCHIP interpolation if available
        support_window_frames: local support window for cubic interpolation

    Returns:
        updated payload
    """

    frames = payload.get("frames", [])

    frame_by_id = {
        f["frame_id"]: f
        for f in frames
    }

    # Allow configuration by seconds instead of frames.
    if max_gap_seconds is not None:
        max_gap_frames = int(max_gap_seconds * fps)

    def is_court_coord_valid(ball_obj):
        """
        Determine whether ball court coordinate should be trusted.
        """
        if ball_obj is None:
            return False

        # If newer class-aware filtering fields exist, use them.
        if "court_coords_valid" in ball_obj:
            return bool(ball_obj.get("court_coords_valid"))

        if "court_coords_usable_for_analytics" in ball_obj:
            return bool(ball_obj.get("court_coords_usable_for_analytics"))

        # Fallback: only check that coordinate exists.
        coord = ball_obj.get("court_coords_meters")
        return isinstance(coord, (list, tuple)) and len(coord) == 2

    def get_best_detected_ball(objects):
        """
        Select best original detected ball object in a frame.

        Ignores previously interpolated ball objects so this function
        can be safely re-run.
        """
        balls = [
            o for o in objects
            if o.get("class") == "ball"
        ]

        if not balls:
            return None

        # Ignore previously interpolated balls.
        original_balls = [
            b for b in balls
            if not b.get("is_interpolated", False)
            and float(b.get("confidence", 0.0)) > 0.0
        ]

        if not original_balls:
            return None

        # Choose best ball by confidence, then by court usability.
        best = max(
            original_balls,
            key=lambda b: (
                float(b.get("confidence", 0.0)),
                bool(b.get("court_coords_usable_for_analytics", True))
            )
        )

        return best

    def nearest_player_info(frame, ball_court, ball_px):
        """
        Find nearest player to interpolated ball.

        Returns:
            distance_meters, distance_pixels, player_track_id
        """
        best_dist_m = float("inf")
        best_dist_px = float("inf")
        best_player_id = None

        for obj in frame.get("objects", []):
            if obj.get("class") != "player":
                continue

            player_id = obj.get("track_id")

            # Court distance.
            if ball_court is not None:
                player_court = obj.get("court_coords_meters")

                # If player validity flag exists, use it.
                player_court_valid = obj.get("court_coords_valid", True)

                if (
                    player_court is not None and
                    len(player_court) == 2 and
                    player_court_valid
                ):
                    dist_m = math.hypot(
                        player_court[0] - ball_court[0],
                        player_court[1] - ball_court[1]
                    )

                    if dist_m < best_dist_m:
                        best_dist_m = dist_m
                        best_player_id = player_id

            # Pixel distance fallback.
            if ball_px is not None:
                player_px = obj.get("feet_coord_px")

                if player_px is None and obj.get("bbox") is not None:
                    x1, y1, x2, y2 = obj["bbox"]
                    player_px = [
                        int((x1 + x2) / 2),
                        int(y2)
                    ]

                if player_px is not None and len(player_px) == 2:
                    dist_px = math.hypot(
                        player_px[0] - ball_px[0],
                        player_px[1] - ball_px[1]
                    )

                    if dist_px < best_dist_px:
                        best_dist_px = dist_px

                        if best_player_id is None:
                            best_player_id = player_id

        final_dist_m = best_dist_m if math.isfinite(best_dist_m) else None
        final_dist_px = best_dist_px if math.isfinite(best_dist_px) else None

        return final_dist_m, final_dist_px, best_player_id

    def estimate_ball_bbox(center_px, prev_ball=None, next_ball=None):
        """
        Create a synthetic bbox around interpolated ball center.
        Uses previous/next ball bbox size when possible.
        """
        cx, cy = center_px

        sizes = []

        for ball in [prev_ball, next_ball]:
            if ball is None:
                continue

            bbox = ball.get("bbox")
            if bbox is not None and len(bbox) == 4:
                w = max(4, int(bbox[2] - bbox[0]))
                h = max(4, int(bbox[3] - bbox[1]))
                sizes.append((w, h))

        if sizes:
            w = int(sum(s[0] for s in sizes) / len(sizes))
            h = int(sum(s[1] for s in sizes) / len(sizes))
        else:
            w = 26
            h = 26

        half_w = max(3, w // 2)
        half_h = max(3, h // 2)

        return [
            max(0, int(cx - half_w)),
            max(0, int(cy - half_h)),
            int(cx + half_w),
            int(cy + half_h)
        ]

    # ------------------------------------------------------------------
    # STEP 1:
    # Clean existing ball objects and mark original detected balls.
    # ------------------------------------------------------------------

    ball_by_frame = {}

    for frame in frames:
        # Do not modify invalid court views.
        if not frame.get("is_valid_court_view", True):
            continue

        frame_id = frame["frame_id"]
        objects = frame.get("objects", [])

        balls = [
            o for o in objects
            if o.get("class") == "ball"
        ]

        # Remove all ball objects first.
        # We will insert only one clean ball object per frame.
        frame["objects"] = [
            o for o in objects
            if o.get("class") != "ball"
        ]

        best_ball = get_best_detected_ball(balls)

        if best_ball is not None:
            best_ball.update({
                "detected": True,
                "is_interpolated": False,
                "is_occluded": False,
                "tracking_state": "DETECTED",
                "interpolation_method": None,
                "frames_since_last_detection": 0
            })

            frame["objects"].append(best_ball)
            ball_by_frame[frame_id] = best_ball

    detected_ids = sorted(ball_by_frame.keys())

    # ------------------------------------------------------------------
    # STEP 2:
    # Interpolate short missing gaps.
    # ------------------------------------------------------------------

    interpolated_frames_added = 0
    occluded_interpolated_frames = 0

    all_gap_lengths = []
    filled_gap_lengths = []
    skipped_gap_lengths = []

    for idx in range(len(detected_ids) - 1):
        prev_id = detected_ids[idx]
        next_id = detected_ids[idx + 1]

        gap_ids = [
            fid for fid in range(prev_id + 1, next_id)
            if fid in frame_by_id
        ]

        if not gap_ids:
            continue

        all_gap_lengths.append(len(gap_ids))

        prev_ball = ball_by_frame[prev_id]
        next_ball = ball_by_frame[next_id]

        # Skip gaps that cross invalid court views.
        invalid_gap = any(
            not frame_by_id[fid].get("is_valid_court_view", True)
            for fid in gap_ids
        )

        if invalid_gap:
            skipped_gap_lengths.append(len(gap_ids))
            continue

        # Skip long gaps.
        if len(gap_ids) > max_gap_frames:
            skipped_gap_lengths.append(len(gap_ids))
            continue

        # ------------------------------------------------------------------
        # Speed sanity check using court coordinates.
        # This helps avoid impossible teleport interpolation.
        # ------------------------------------------------------------------
        prev_court = (
            prev_ball.get("court_coords_meters")
            if is_court_coord_valid(prev_ball)
            else None
        )

        next_court = (
            next_ball.get("court_coords_meters")
            if is_court_coord_valid(next_ball)
            else None
        )

        if prev_court is not None and next_court is not None:
            gap_duration_seconds = (len(gap_ids) + 1) / float(fps)

            court_distance = math.hypot(
                next_court[0] - prev_court[0],
                next_court[1] - prev_court[1]
            )

            if gap_duration_seconds > 0:
                estimated_speed = court_distance / gap_duration_seconds
            else:
                estimated_speed = 0.0

            if (
                hard_speed_limit_mps is not None and
                estimated_speed > hard_speed_limit_mps
            ):
                skipped_gap_lengths.append(len(gap_ids))
                continue

        # ------------------------------------------------------------------
        # Build local support frames for interpolation.
        # This gives cubic/PCHIP more context than only two endpoints.
        # ------------------------------------------------------------------
        center_gap = (prev_id + next_id) / 2.0

        support_ids = detected_ids[
            max(0, idx - 2): min(len(detected_ids), idx + 4)
        ]

        support_ids = [
            fid for fid in support_ids
            if abs(fid - center_gap) <= support_window_frames
        ]

        # Ensure endpoints are included.
        support_ids = sorted(set([prev_id, next_id] + support_ids))

        # ------------------------------------------------------------------
        # Interpolate pixel coordinates.
        # ------------------------------------------------------------------
        px_support_ids = []
        px_support_values = []

        for fid in support_ids:
            ball = ball_by_frame.get(fid)
            if ball is None:
                continue

            px = ball.get("center_coord_px")
            if px is not None and len(px) == 2:
                px_support_ids.append(fid)
                px_support_values.append(px)

        px_interpolated, px_method = interpolate_2d_series(
            px_support_ids,
            px_support_values,
            gap_ids,
            use_cubic=use_cubic
        )

        if px_interpolated is None:
            skipped_gap_lengths.append(len(gap_ids))
            continue

        # ------------------------------------------------------------------
        # Interpolate court coordinates if endpoints are usable.
        # ------------------------------------------------------------------
        court_interpolated = None
        court_method = None
        court_valid_for_gap = False

        if prev_court is not None and next_court is not None:
            court_support_ids = []
            court_support_values = []

            for fid in support_ids:
                ball = ball_by_frame.get(fid)
                if ball is None:
                    continue

                if not is_court_coord_valid(ball):
                    continue

                court = ball.get("court_coords_meters")
                if court is not None and len(court) == 2:
                    court_support_ids.append(fid)
                    court_support_values.append(court)

            court_interpolated, court_method = interpolate_2d_series(
                court_support_ids,
                court_support_values,
                gap_ids,
                use_cubic=use_cubic
            )

            if court_interpolated is not None:
                court_valid_for_gap = True

        # If court interpolation is not possible, hold last known court point.
        if court_interpolated is None:
            hold_court = prev_court if prev_court is not None else [0.0, 0.0]
            court_interpolated = [
                hold_court.copy()
                for _ in gap_ids
            ]
            court_method = "hold_last_known"
            court_valid_for_gap = False

        # ------------------------------------------------------------------
        # Insert interpolated ball objects.
        # ------------------------------------------------------------------
        for gap_index, fid in enumerate(gap_ids):
            frame = frame_by_id[fid]

            # Remove any existing ball object in this frame.
            frame["objects"] = [
                o for o in frame.get("objects", [])
                if o.get("class") != "ball"
            ]

            center_px = [
                int(round(px_interpolated[gap_index][0])),
                int(round(px_interpolated[gap_index][1]))
            ]

            court_m = [
                round(float(court_interpolated[gap_index][0]), 3),
                round(float(court_interpolated[gap_index][1]), 3)
            ]

            ball_court_for_occlusion = court_m if court_valid_for_gap else None

            dist_m, dist_px, nearest_player_id = nearest_player_info(
                frame,
                ball_court_for_occlusion,
                center_px
            )

            is_occluded = False

            if dist_m is not None and dist_m <= occlusion_radius_m:
                is_occluded = True
            elif dist_m is None and dist_px is not None and dist_px <= occlusion_pixel_radius:
                is_occluded = True

            bbox = estimate_ball_bbox(
                center_px,
                prev_ball=prev_ball,
                next_ball=next_ball
            )

            interpolated_ball = {
                "track_id": prev_ball.get("track_id", -1),
                "class": "ball",
                "bbox": bbox,
                "confidence": 0.0,

                "center_coord_px": center_px,
                "court_coords_meters": court_m,

                "detected": False,
                "is_interpolated": True,
                "is_occluded": bool(is_occluded),

                "tracking_state": (
                    "INTERPOLATED_OCCLUDED"
                    if is_occluded
                    else "INTERPOLATED"
                ),

                "interpolation_method": f"{px_method}/{court_method}",

                "gap_length_frames": len(gap_ids),
                "gap_index": gap_index + 1,
                "frames_since_last_detection": gap_index + 1,

                "court_coords_valid": court_valid_for_gap,
                "court_coords_interpolated": court_valid_for_gap,

                "nearest_player_track_id": nearest_player_id,
                "nearest_player_distance_m": (
                    round(dist_m, 3)
                    if dist_m is not None
                    else None
                ),
                "nearest_player_distance_px": (
                    round(dist_px, 1)
                    if dist_px is not None
                    else None
                )
            }

            frame["objects"].append(interpolated_ball)

            interpolated_frames_added += 1

            if is_occluded:
                occluded_interpolated_frames += 1

        filled_gap_lengths.append(len(gap_ids))

    # ------------------------------------------------------------------
    # STEP 3:
    # Add interpolation metadata.
    # ------------------------------------------------------------------

    interpolation_metadata = {
        "fps": fps,
        "max_gap_frames": max_gap_frames,
        "occlusion_radius_m": occlusion_radius_m,
        "occlusion_pixel_radius": occlusion_pixel_radius,
        "use_cubic": use_cubic,
        "support_window_frames": support_window_frames,

        "original_detected_ball_frames": len(detected_ids),

        "gap_count": len(all_gap_lengths),
        "filled_gap_count": len(filled_gap_lengths),
        "skipped_gap_count": len(skipped_gap_lengths),

        "average_gap_frames": (
            round(sum(all_gap_lengths) / len(all_gap_lengths), 2)
            if all_gap_lengths
            else 0.0
        ),

        "max_observed_gap_frames": (
            max(all_gap_lengths)
            if all_gap_lengths
            else 0
        ),

        "average_filled_gap_frames": (
            round(sum(filled_gap_lengths) / len(filled_gap_lengths), 2)
            if filled_gap_lengths
            else 0.0
        ),

        "interpolated_frames_added": interpolated_frames_added,
        "occluded_interpolated_frames": occluded_interpolated_frames
    }

    payload.setdefault("metadata", {})
    payload["metadata"]["ball_interpolation"] = interpolation_metadata

    return payload


def interpolate_ball_json_file(
    input_json_path,
    output_json_path,
    fps=30.0,
    max_gap_frames=12,
    max_gap_seconds=None,
    occlusion_radius_m=1.2,
    occlusion_pixel_radius=90,
    max_speed_mps=25.0,
    hard_speed_limit_mps=45.0,
    use_cubic=True,
    support_window_frames=80
):
    """
    File wrapper:
    Reads tracking JSON, interpolates ball gaps, writes new JSON.
    """
    print(f"Reading tracking JSON from: {input_json_path}")

    with open(input_json_path, "r") as f:
        payload = json.load(f)

    payload = interpolate_ball_payload(
        payload=payload,
        fps=fps,
        max_gap_frames=max_gap_frames,
        max_gap_seconds=max_gap_seconds,
        occlusion_radius_m=occlusion_radius_m,
        occlusion_pixel_radius=occlusion_pixel_radius,
        max_speed_mps=max_speed_mps,
        hard_speed_limit_mps=hard_speed_limit_mps,
        use_cubic=use_cubic,
        support_window_frames=support_window_frames
    )

    with open(output_json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Interpolated tracking JSON saved to: {output_json_path}")

    if "ball_interpolation" in payload.get("metadata", {}):
        print(json.dumps(payload["metadata"]["ball_interpolation"], indent=2))


# ---------------------------------------------------------
# EXECUTE INTERPOLATION
# ---------------------------------------------------------
interpolate_ball_json_file(
    input_json_path="match_tracking_data21.08.json",
    output_json_path="match_tracking_data_interpolated.json",
    fps = 30.0,
    max_gap_frames = 6,
    use_cubic = False,
    support_window_frames = 25,
    occlusion_radius_m = 1.0,
    hard_speed_limit_mps = 20.0
)