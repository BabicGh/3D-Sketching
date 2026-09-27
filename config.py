"""
config.py — every tunable in one place.

Distances that depend on how far the hand is from the camera are expressed as
RATIOS of the hand's own "palm scale" so thresholds keep working near or far.
"""
from dataclasses import dataclass


@dataclass
class GestureConfig:
    # ---- Step 1: Claw clutch -------------------------------------------------
    # ratio = mean(|fingertip - crown_center|) / palm_scale
    # Open hand is roughly 1.3-1.5, a tight claw/fist roughly 0.5-0.8.
    # Use the on-screen ratio readout to tune these for your hands + webcam.
    claw_enter_ratio: float = 1.05      # below this  -> claw candidate
    claw_exit_ratio: float = 1.25       # above this  -> release candidate (hysteresis gap)
    claw_engage_frames: int = 3         # consecutive frames required to engage
    claw_release_frames: int = 4        # consecutive frames required to release
    hand_lost_grace_frames: int = 3     # tolerate this many missed detections

    # ---- Smoothing: EMA weight on the NEW sample (1.0 = no smoothing) --------
    wrist_alpha: float = 0.6
    normal_alpha: float = 0.6
    distance_alpha: float = 0.5

    # ---- Step 2: one-hand branching (Orbit vs Pan) ---------------------------
    # "intent"     : after the claw engages, whichever motion (palm-normal
    #                rotation vs wrist translation) crosses its threshold first
    #                decides the mode. No extra gesture needed.
    # "handedness" : orbit_hand orbits, the other hand pans.
    one_hand_policy: str = "intent"
    orbit_hand: str = "Right"
    mode_arm_frames: int = 3            # min dwell before locking (lets a 2nd hand join -> Zoom)
    intent_rot_threshold_rad: float = 0.10   # net palm-normal rotation (~6 deg)
    intent_pan_threshold: float = 0.30       # net wrist travel, in palm-scale units
    rot_deadzone_rad: float = 0.0015    # per-frame rotation below this is ignored
    pan_deadzone_px: float = 0.4        # per-frame wrist travel below this is ignored

    # ---- Step 3: dual-hand zoom ----------------------------------------------
    zoom_deadzone: float = 0.003        # |D_cur/D_prev - 1| must exceed this
    zoom_gain: float = 1.5              # zoom_factor = (D_cur / D_prev) ** gain

    # ---- Step 6: tool wheel (point to open, swipe to browse, fist to select) -
    # score = index finger's ratio minus the MOST-EXTENDED of the other three (thumb excluded,
    # it behaves inconsistently across grips); a relaxed/open hand scores near 0, a clean point
    # scores roughly 0.6-0.9. Select itself reuses the claw thresholds above, but gated on the
    # MAX fingertip ratio rather than the mean — see ToolMenu's docstring in gesture_engine.py
    # for why that's what correctly tells "pointing" apart from "fisting".
    point_enter_score: float = 0.55     # score must exceed this to open the wheel
    point_exit_score: float = 0.30      # (kept for HysteresisGate's shape; see note below)
    point_engage_frames: int = 3
    point_release_frames: int = 4       # (also currently inert — see note below)
    menu_swipe_unit: float = 1.4        # wrist travel, in palm-scale units, per one tool step
    menu_select_cooldown_frames: int = 6  # nav stays suppressed this long after a selecting fist,
                                           # so the same fist can't also kick off an orbit/pan
    # Note: once open, the wheel stays open regardless of hand shape/visibility — only a fist
    # (select) or five straight fingers (cancel) close it, both using the claw_* thresholds
    # above. point_exit_score/point_release_frames are only exercised while closed, which never
    # happens in the current design (opening and leaving CLOSED happen in the same frame), but
    # are left wired through HysteresisGate in case you want a "must keep pointing" mode later.


@dataclass
class ViewportConfig:
    # camera / projection
    cam_dist: float = 6.0               # model radius ~1, so this leaves headroom for zoom
    focal_factor: float = 1.7           # focal length = factor * frame height

    # navigation gains
    orbit_gain: float = 1.5             # model rotation per unit palm-normal rotation
    grid_sync_gain: float = 1.5         # keep equal to orbit_gain for lock-step grid
    pan_gain: float = 1.2
    min_scale: float = 0.25
    max_scale: float = 2.5

    # Step 4: bounding box leads, geometry follows (time constants, seconds)
    tau_bbox: float = 0.05
    tau_geom: float = 0.20

    # floor grid (model units, relative to the normalised sketch)
    grid_half: float = 2.0
    grid_div: int = 10
    grid_gap: float = 0.12              # gap between sketch bottom and floor

    # look & cost knobs
    show_backdrop: bool = True          # dimmed webcam feed behind the hologram
    backdrop_gain: float = 0.30
    scanlines: bool = True
    bloom: bool = False                 # extra blur pass; costs ~1 ms at 640x480
    bloom_gain: float = 0.8
    antialias: bool = True
    glow_lines: bool = True             # cheap glow: wide dim pass + thin bright pass
    glow_strength: float = 0.28
    draw_landmarks: bool = True
    bg_color: tuple = (18, 10, 4)       # BGR

    # Step 6: tool wheel (fixed screen anchor, not hand-anchored, so it doesn't jitter)
    menu_tau: float = 0.10              # lag: wheel visually "falls into" the new position
    menu_center: tuple = (0.5, 0.26)    # fraction of (width, height) — the center socket
    menu_radius_x: float = 150.0
    menu_radius_y: float = 46.0
    menu_slot_r_max: float = 32.0
    menu_slot_r_min: float = 7.0
    menu_flash_frames: int = 40         # how long the "X SELECTED" / "CANCELLED" text lingers
    menu_color: tuple = (255, 210, 60)  # BGR
    menu_zoom_start: float = 0.75       # depth (0-1) past which the socket zoom-in + glow kicks in
    menu_zoom_boost: float = 14.0       # extra radius, on top of the normal perspective size, at depth=1

    # Step 7: exploded view (Disassemble / Assemble tool actions)
    tau_explode: float = 0.35           # seconds; lag for the explode/collapse animation
    explode_distance: float = 1.3       # model units each part travels outward at full explode

    # Step 7: per-part editing (Select / Move / Rotate / Scale tool actions)
    select_color: tuple = (255, 255, 255)   # BGR — the selected part's outline
    part_move_gain: float = 1.0             # extra feel adjustment on top of pan_gain
    part_scale_min: float = 0.2
    part_scale_max: float = 4.0

    # Step 9: Extrude (stretch a part along its own local Y) and Measure (read-only ruler)
    extrude_gain: float = 1.0
    part_stretch_min: float = 0.2
    part_stretch_max: float = 6.0
    measure_color: tuple = (120, 255, 190)  # BGR

    # Step 8: "realistic" projector artifacts — flicker, RGB channel split, a scanning highlight
    # band, and fine grain. All cheap per-frame numpy/cv2 ops (see _apply_hologram_fx); toggle
    # with main.py's 'h' key if you'd rather have the cleaner Step-4 look, or just to save the
    # ~1 ms/frame on the slowest hardware.
    hologram_fx: bool = True
    flicker_strength: float = 0.06      # +/- brightness modulation, fraction of full brightness
    flicker_speed: float = 6.0          # roughly Hz; a little per-frame jitter is layered on top
    aberration_px: int = 1              # R/B channel horizontal split, in pixels (0 disables)
    scan_band_speed: float = 220.0      # px/second the highlight band travels down and wraps
    scan_band_height: float = 46.0
    scan_band_gain: float = 0.22
    grain_gain: float = 0.05            # 0 disables
