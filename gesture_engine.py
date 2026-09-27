"""
gesture_engine.py — Steps 1, 2, 3 and 5 (pure NumPy, no OpenCV / MediaPipe imports).

Coordinate convention used everywhere in this file (right-handed, "screen space"):
    +X -> right on the mirrored display
    +Y -> UP            (image y is flipped)
    +Z -> toward viewer (MediaPipe z is flipped: smaller z = closer to camera)
Units are camera-frame pixels; MediaPipe's z is scaled by frame width so all
three axes share the same units.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import GestureConfig
from tools import Tool, DEFAULT_TOOLS

# --- Tracking blueprint: only these 8 of 21 landmarks are ever touched --------
FINGERTIP_IDS = (4, 8, 12, 16, 20)      # thumb, index, middle, ring, pinky
CROWN_IDS = (0, 5, 17)                  # wrist, index MCP, pinky MCP
TRACKED_IDS = FINGERTIP_IDS + CROWN_IDS  # row order of every (8, 3) array below

_EPS = 1e-9


# =============================================================================
# Data containers
# =============================================================================
@dataclass
class HandObservation:
    """One hand, reduced to the tracked subset, in both math and pixel space."""
    label: str                # "Left" / "Right" (user's hand, mirrored view)
    tips3: np.ndarray         # (5, 3) fingertip positions, screen-space 3D
    crown3: np.ndarray        # (3, 3) rows: p0 wrist, p5 index MCP, p17 pinky MCP
    tips_px: np.ndarray       # (5, 2) fingertip pixel coords (for drawing)
    crown_px: np.ndarray      # (3, 2)
    center3: np.ndarray = field(init=False)   # (3,)  crown center
    center_px: np.ndarray = field(init=False)  # (2,)

    def __post_init__(self) -> None:
        # Crown Center = centroid of the three base knuckles = (p0 + p5 + p17) / 3
        self.center3 = self.crown3.mean(axis=0)
        self.center_px = self.crown_px.mean(axis=0)

    @classmethod
    def from_normalized(cls, label: str, lm: np.ndarray, width: int, height: int) -> "HandObservation":
        """lm: (8, 3) MediaPipe normalized (x, y, z) in TRACKED_IDS order."""
        x = lm[:, 0] * width
        y = lm[:, 1] * height
        z = lm[:, 2] * width                       # MediaPipe: z shares x's scale
        px = np.stack((x, y), axis=1)
        p3 = np.stack((x, -y, -z), axis=1)         # flip Y and Z -> right-handed, Y up, Z to viewer
        return cls(label, p3[:5], p3[5:], px[:5], px[5:])


class NavMode(Enum):
    IDLE = auto()     # no claw
    ARMED = auto()    # claw engaged, mode not decided yet
    ORBIT = auto()
    PAN = auto()
    ZOOM = auto()


@dataclass
class ToolMenuOutput:
    """Step 6: state of the point-to-open / swipe-to-browse / fist-to-select tool wheel."""
    active: bool = False                    # wheel should be drawn this frame
    just_opened: bool = False               # one-shot: wheel opened this frame (let the viewport snap, not spin)
    index: int = 0                          # tool nearest the center this frame
    position: float = 0.0                   # continuous, unbounded wheel angle in "tool units" (index = round(position) % n)
    tool: Optional[Tool] = None             # tools[index], for convenience
    selected_tool: Optional[Tool] = None    # one-shot: set on the frame a fist confirms a choice
    cancelled: bool = False                 # one-shot: set on the frame all five fingers go straight


@dataclass
class NavigationOutput:
    """Everything the viewport needs for one frame."""
    mode: NavMode = NavMode.IDLE
    rotvec: np.ndarray = field(default_factory=lambda: np.zeros(3))   # delta rotation, axis*angle (rad)
    pan_px: np.ndarray = field(default_factory=lambda: np.zeros(2))   # delta (X right, Y up), pixels
    zoom_factor: float = 1.0                                          # >1 zoom in, <1 zoom out
    claws: Dict[str, bool] = field(default_factory=dict)
    claw_ratios: Dict[str, Optional[float]] = field(default_factory=dict)
    hands: List[HandObservation] = field(default_factory=list)
    active_tips_px: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    palm_normal: Optional[np.ndarray] = None       # driver hand's N (smoothed)
    palm_anchor_px: Optional[np.ndarray] = None    # where to draw N
    menu: ToolMenuOutput = field(default_factory=ToolMenuOutput)


# =============================================================================
# Vector math helpers
# =============================================================================
def palm_scale(crown3: np.ndarray) -> float:
    """Hand-size yardstick: mean side length of the wrist/index-MCP/pinky-MCP triangle."""
    p0, p5, p17 = crown3
    return float((np.linalg.norm(p5 - p0) + np.linalg.norm(p17 - p0) + np.linalg.norm(p5 - p17)) / 3.0)


def finger_ratios(tips3: np.ndarray, center3: np.ndarray, scale: float) -> np.ndarray:
    """Per-fingertip 3D distance to the crown center, / palm scale. Order: thumb, index, middle, ring, pinky."""
    return np.linalg.norm(tips3 - center3, axis=1) / max(scale, _EPS)


def claw_ratio(tips3: np.ndarray, center3: np.ndarray, scale: float) -> float:
    """Step 1 metric: mean of the five per-finger ratios (curled hand -> low ratio)."""
    return float(finger_ratios(tips3, center3, scale).mean())


def palm_normal(crown3: np.ndarray) -> Optional[np.ndarray]:
    """Step 2: N = normalize(v1 x v2), v1 = p5 - p0, v2 = p17 - p0."""
    p0, p5, p17 = crown3
    n = np.cross(p5 - p0, p17 - p0)
    length = np.linalg.norm(n)
    return None if length < 1e-6 else n / length


def rotvec_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector a onto unit vector b, as an axis*angle vector.

    axis  = a x b  (direction), |a x b| = sin(theta), a . b = cos(theta)
    angle = atan2(sin, cos)  -> numerically stable for tiny and near-180 deg angles
    """
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < _EPS:
        return np.zeros(3)
    theta = math.atan2(s, float(np.dot(a, b)))
    return axis * (theta / s)


# =============================================================================
# Generic debounced threshold gate (Step 1's claw hysteresis, generalized so
# Step 6's "point" gesture can reuse the exact same flicker-free logic)
# =============================================================================
class HysteresisGate:
    """Debounced boolean, engaged/released by a scalar crossing two thresholds with a gap.

    direction="below": engages once `value` stays under `enter` for `engage_frames`
        frames straight; releases once it stays over `exit_` for `release_frames`.
        (This is Step 1's claw: low ratio = curled fingers = grabbing.)
    direction="above": the mirror image — engages on staying OVER `enter`, releases
        on staying UNDER `exit_`. (Step 6's point: high index ratio, low others.)
    Either way `enter`/`exit_` sit on opposite sides of the switch point with a gap
    between them, and only a run of consecutive frames can flip the state — together
    that's what kills single-frame flicker.
    """

    def __init__(self, enter: float, exit_: float, engage_frames: int, release_frames: int,
                 lost_grace: int, direction: str = "below") -> None:
        assert direction in ("below", "above")
        self.enter, self.exit_ = enter, exit_
        self.engage_frames, self.release_frames, self.lost_grace = engage_frames, release_frames, lost_grace
        self.direction = direction
        self.active = False
        self._streak = 0
        self._missing = 0

    def reset(self) -> None:
        self.active, self._streak, self._missing = False, 0, 0

    def update(self, value: Optional[float]) -> bool:
        if value is None:                                    # hand not detected this frame
            self._missing += 1
            if self._missing > self.lost_grace:
                self.active, self._streak = False, 0
            return self.active
        self._missing = 0

        if self.direction == "below":
            enter_cond, release_cond = value < self.enter, value > self.exit_
        else:
            enter_cond, release_cond = value > self.enter, value < self.exit_

        if self.active:
            self._streak = self._streak + 1 if release_cond else 0
            if self._streak >= self.release_frames:
                self.active, self._streak = False, 0
        else:
            self._streak = self._streak + 1 if enter_cond else 0
            if self._streak >= self.engage_frames:
                self.active, self._streak = True, 0
        return self.active


# =============================================================================
# Step 1 — the Claw clutch (with per-hand hysteresis)
# =============================================================================
class ClawDetector:
    """Debounced boolean `is_clawing` for a single hand — a `HysteresisGate` underneath.

    Engages after `claw_engage_frames` consecutive frames below `claw_enter_ratio`;
    releases after `claw_release_frames` consecutive frames above `claw_exit_ratio`.
    """

    def __init__(self, cfg: GestureConfig) -> None:
        self._gate = HysteresisGate(cfg.claw_enter_ratio, cfg.claw_exit_ratio,
                                     cfg.claw_engage_frames, cfg.claw_release_frames,
                                     cfg.hand_lost_grace_frames, direction="below")

    @property
    def is_clawing(self) -> bool:
        return self._gate.active

    def update(self, ratio: Optional[float]) -> bool:
        return self._gate.update(ratio)


class _HandTrack:
    """Per-hand smoothed signals + claw state. Internal to the engine."""

    def __init__(self, cfg: GestureConfig) -> None:
        self.cfg = cfg
        self.claw = ClawDetector(cfg)
        self.obs: Optional[HandObservation] = None
        self.ratio: Optional[float] = None              # mean finger ratio (claw metric)
        self.finger_ratios: Optional[np.ndarray] = None  # (5,) thumb,index,middle,ring,pinky
        self.scale: float = 1.0
        self.wrist: Optional[np.ndarray] = None         # EMA-smoothed p0 (3,)
        self.normal: Optional[np.ndarray] = None        # EMA-smoothed N (3,)
        self.prev_wrist: Optional[np.ndarray] = None
        self.prev_normal: Optional[np.ndarray] = None

    def update(self, obs: Optional[HandObservation]) -> None:
        self.obs = obs
        if obs is None:
            self.claw.update(None)
            self.ratio = None
            self.finger_ratios = None
            # Forget history so a re-detected hand doesn't cause a jump.
            self.wrist = self.normal = self.prev_wrist = self.prev_normal = None
            return

        self.scale = palm_scale(obs.crown3)
        self.finger_ratios = finger_ratios(obs.tips3, obs.center3, self.scale)
        self.ratio = float(self.finger_ratios.mean())
        self.claw.update(self.ratio)

        self.prev_wrist, self.prev_normal = self.wrist, self.normal
        a = self.cfg.wrist_alpha
        w = obs.crown3[0]
        self.wrist = w.copy() if self.wrist is None else a * w + (1.0 - a) * self.wrist

        n = palm_normal(obs.crown3)
        if n is not None:
            if self.normal is None:
                self.normal = n
            else:
                m = self.cfg.normal_alpha * n + (1.0 - self.cfg.normal_alpha) * self.normal
                length = np.linalg.norm(m)
                if length > _EPS:
                    self.normal = m / length                 # keep it a unit vector

    def rotation_delta(self) -> np.ndarray:
        """Frame-to-frame rotation of the palm normal (axis*angle, rad)."""
        if self.prev_normal is None or self.normal is None:
            return np.zeros(3)
        return rotvec_between(self.prev_normal, self.normal)

    def pan_delta(self) -> np.ndarray:
        """Frame-to-frame wrist translation in (X, Y) pixels."""
        if self.prev_wrist is None or self.wrist is None:
            return np.zeros(2)
        return (self.wrist - self.prev_wrist)[:2]


# =============================================================================
# Step 6 — point-to-open / swipe-to-browse / fist-to-select tool wheel
# =============================================================================
class _MenuState(Enum):
    CLOSED = auto()
    OPEN = auto()


class ToolMenu:
    """Right hand only, by design (see `HolographicGestureEngine._arbitrate`'s
    caller for why left is left free): index finger out, other three curled
    opens the wheel; wrist swipe left/right browses it.

    Once open, the wheel stays open — through the hand leaving frame, through
    any hand shape while swiping — until one of two explicit gestures closes
    it: a fist confirms the centered tool, all five fingers straight cancels.

    `position` is a single unbounded float — no modulo bookkeeping, so a swipe
    that wraps past the last tool back to the first is just the same smooth
    motion continuing, exactly like a real wheel, instead of a visual jump.
    """

    def __init__(self, tools: Sequence[Tool], cfg: GestureConfig) -> None:
        self.tools = list(tools)
        self.cfg = cfg
        self._gate = HysteresisGate(cfg.point_enter_score, cfg.point_exit_score,
                                     cfg.point_engage_frames, cfg.point_release_frames,
                                     cfg.hand_lost_grace_frames, direction="above")
        # Select: every finger curled (MAX ratio, not mean) — deliberately separate from the
        # navigation ClawDetector's mean-based test. With only 4 of 5 fingers curled while
        # pointing, the MEAN ratio can already read as a (looser) nav claw, but the MAX ratio
        # (index still extended) can't, so this is what correctly tells point apart from fist.
        self._select_gate = HysteresisGate(cfg.claw_enter_ratio, cfg.claw_exit_ratio,
                                            cfg.claw_engage_frames, cfg.claw_release_frames,
                                            cfg.hand_lost_grace_frames, direction="below")
        # Cancel: every finger straight (MEAN ratio, high). Deliberately the mirror image of the
        # claw's own two thresholds — claw_exit_ratio is already "basically open", claw_enter_ratio
        # is already "basically closed" — so this reuses numbers you've already tuned by feel.
        self._open_gate = HysteresisGate(cfg.claw_exit_ratio, cfg.claw_enter_ratio,
                                          cfg.claw_engage_frames, cfg.claw_release_frames,
                                          cfg.hand_lost_grace_frames, direction="above")
        self.state = _MenuState.CLOSED
        self.position = 0.0                      # persists across opens, so the wheel doesn't reset every time
        self._prev_wrist_x: Optional[float] = None

    def update(self, track: "_HandTrack", nav_is_idle: bool) -> ToolMenuOutput:
        # Point score: index finger extended (high ratio) while the other three are curled (low
        # ratio). Thumb is deliberately excluded — its ratio behaves inconsistently across grips.
        score = max_ratio = mean_ratio = None
        if track.finger_ratios is not None:
            r = track.finger_ratios                                    # thumb, index, middle, ring, pinky
            score = float(r[1] - max(r[2], r[3], r[4]))
            max_ratio, mean_ratio = float(r.max()), float(r.mean())

        n = len(self.tools)
        index = int(round(self.position)) % n
        out = ToolMenuOutput(active=self.state is _MenuState.OPEN, index=index, position=self.position,
                              tool=self.tools[index])

        if self.state is _MenuState.CLOSED:
            # Only let the gate accumulate evidence when navigation isn't already running — this
            # keeps an in-progress orbit/pan/zoom from being interrupted by a stray point shape.
            pointing = self._gate.update(score if nav_is_idle else None)
            if pointing:
                self.state = _MenuState.OPEN
                self._select_gate.reset()
                self._open_gate.reset()
                self._prev_wrist_x = float(track.wrist[0]) if track.wrist is not None else None
                out.active, out.just_opened = True, True
            return out

        # --- OPEN: stays open regardless of hand visibility or shape — swiping doesn't require
        # holding the exact point pose anymore, and losing the hand out of frame no longer closes
        # it. Only the two explicit gestures below do: a fist confirms, a fully open hand cancels.
        is_fist = self._select_gate.update(max_ratio)
        if is_fist:
            self.state = _MenuState.CLOSED
            self._gate.reset()                                 # fresh debounce for the next point
            self._prev_wrist_x = None
            out.active, out.selected_tool = False, self.tools[index]
            return out

        is_open_hand = self._open_gate.update(mean_ratio)
        if is_open_hand:
            self.state = _MenuState.CLOSED
            self._gate.reset()
            self._select_gate.reset()
            self._prev_wrist_x = None
            out.active, out.cancelled = False, True
            return out

        # --- OPEN: wrist swipe, in palm-scale units so it doesn't depend on distance to camera.
        # A lost-then-recovered hand doesn't register a swipe from the jump: the None branch drops
        # the reference point, so the first frame back only re-primes it instead of moving the wheel.
        if track.wrist is not None and track.scale > _EPS:
            x = float(track.wrist[0])
            if self._prev_wrist_x is not None:
                self.position += (x - self._prev_wrist_x) / track.scale / self.cfg.menu_swipe_unit
            self._prev_wrist_x = x
        else:
            self._prev_wrist_x = None

        index = int(round(self.position)) % n
        out.index, out.position, out.tool = index, self.position, self.tools[index]
        return out


# =============================================================================
# Steps 2, 3, 5 — navigation state machine
# =============================================================================
class HolographicGestureEngine:
    """Turns per-frame hand observations into navigation deltas.

    State machine (Step 5 lockout):
        IDLE --claw--> ARMED --(intent | 2nd hand)--> ORBIT | PAN | ZOOM --all claws dropped--> IDLE
    Once ORBIT / PAN / ZOOM is locked, nothing but releasing every claw leaves it.
    """

    def __init__(self, cfg: Optional[GestureConfig] = None, tools: Optional[Sequence[Tool]] = None) -> None:
        self.cfg = cfg or GestureConfig()
        self.tracks: Dict[str, _HandTrack] = {"Left": _HandTrack(self.cfg), "Right": _HandTrack(self.cfg)}
        self.tool_menu = ToolMenu(tools if tools is not None else DEFAULT_TOOLS, self.cfg)
        self._menu_cooldown = 0             # frames of suppressed nav right after a tool selection
        self.reset()

    def reset(self) -> None:
        self.mode = NavMode.IDLE
        self._driver: Optional[str] = None
        self._arm_frames = 0
        self._pending_rot = np.zeros(3)     # motion accumulated while the mode is undecided
        self._pending_pan = np.zeros(2)
        self._prev_d: Optional[float] = None
        self._d_smooth: Optional[float] = None

    # ------------------------------------------------------------------ public
    def update(self, observations: Sequence[HandObservation]) -> NavigationOutput:
        by_label = self._assign_labels(list(observations))
        for label, track in self.tracks.items():
            track.update(by_label.get(label))

        # Step 6 first: the wheel only opens while nav is idle, and while it's open (or for a
        # short cooldown right after a selection) it holds nav at IDLE so a closing fist can't
        # also register as the start of an orbit/pan.
        menu_out = self.tool_menu.update(self.tracks["Right"], nav_is_idle=self.mode is NavMode.IDLE)
        if menu_out.selected_tool is not None:
            self._menu_cooldown = self.cfg.menu_select_cooldown_frames
        suppress_nav = menu_out.active or self._menu_cooldown > 0
        if self._menu_cooldown > 0:
            self._menu_cooldown -= 1

        clawing = [l for l, t in self.tracks.items() if t.claw.is_clawing]
        prev_mode = self.mode

        if suppress_nav:
            if self.mode is not NavMode.IDLE:
                self.reset()
        elif not clawing:
            self.reset()                                    # claw fully dropped -> unlock
        elif self.mode in (NavMode.IDLE, NavMode.ARMED):
            self._arbitrate(clawing)
        # else: ORBIT / PAN / ZOOM stay locked regardless of claw count

        rotvec, pan, zoom = np.zeros(3), np.zeros(2), 1.0
        just_locked = prev_mode in (NavMode.IDLE, NavMode.ARMED)

        if self.mode is NavMode.ORBIT:
            if just_locked:                                 # flush motion gathered while deciding
                rotvec = self._pending_rot.copy()
            else:
                rotvec = self._driver_rotation()
        elif self.mode is NavMode.PAN:
            pan = self._pending_pan.copy() if just_locked else self._driver_pan()
        elif self.mode is NavMode.ZOOM:
            zoom = self._zoom_factor()
        if just_locked and self.mode in (NavMode.ORBIT, NavMode.PAN):
            self._pending_rot[:] = 0.0
            self._pending_pan[:] = 0.0

        return self._package(rotvec, pan, zoom, list(by_label.values()), menu_out)

    # -------------------------------------------------------- Step 2: arbitration
    def _arbitrate(self, clawing: List[str]) -> None:
        cfg = self.cfg
        if self.mode is NavMode.IDLE:
            self.mode = NavMode.ARMED
            self._arm_frames = 0
            self._driver = None
            self._pending_rot[:] = 0.0
            self._pending_pan[:] = 0.0
        self._arm_frames += 1

        if len(clawing) >= 2:                               # Step 3 entry: both hands clawing
            self._lock(NavMode.ZOOM)
            return

        driver = clawing[0]
        if driver != self._driver:                          # (re)start accumulation for this hand
            self._driver = driver
            self._pending_rot[:] = 0.0
            self._pending_pan[:] = 0.0
        track = self.tracks[driver]
        self._pending_rot += track.rotation_delta()
        self._pending_pan += track.pan_delta()

        if self._arm_frames < cfg.mode_arm_frames:          # dwell so a 2nd hand can still join -> Zoom
            return

        if cfg.one_hand_policy == "handedness":
            self._lock(NavMode.ORBIT if driver == cfg.orbit_hand else NavMode.PAN)
            return

        # "intent": net rotation of N vs net wrist travel, each normalised by its threshold.
        # (Net displacement, not path length, so jitter does not count as intent.)
        rot_score = float(np.linalg.norm(self._pending_rot)) / cfg.intent_rot_threshold_rad
        pan_score = float(np.linalg.norm(self._pending_pan)) / max(cfg.intent_pan_threshold * track.scale, _EPS)
        if max(rot_score, pan_score) >= 1.0:
            self._lock(NavMode.ORBIT if rot_score >= pan_score else NavMode.PAN)

    def _lock(self, mode: NavMode) -> None:
        self.mode = mode
        if mode is NavMode.ZOOM:
            self._prev_d = None
            self._d_smooth = None

    def _driver_rotation(self) -> np.ndarray:
        t = self.tracks.get(self._driver or "")
        if t is None or not t.claw.is_clawing:
            return np.zeros(3)
        rv = t.rotation_delta()
        return rv if np.linalg.norm(rv) >= self.cfg.rot_deadzone_rad else np.zeros(3)

    def _driver_pan(self) -> np.ndarray:
        t = self.tracks.get(self._driver or "")
        if t is None or not t.claw.is_clawing:
            return np.zeros(2)
        d = t.pan_delta()
        return d if np.linalg.norm(d) >= self.cfg.pan_deadzone_px else np.zeros(2)

    # ------------------------------------------------------------- Step 3: zoom
    def _zoom_factor(self) -> float:
        """Ignore hand rotations; use only D = |wrist_L - wrist_R| (3D Euclidean)."""
        L, R = self.tracks["Left"], self.tracks["Right"]
        if not (L.claw.is_clawing and R.claw.is_clawing and L.wrist is not None and R.wrist is not None):
            self._prev_d = self._d_smooth = None            # pause; re-baseline when both return
            return 1.0

        d = float(np.linalg.norm(L.wrist - R.wrist))
        a = self.cfg.distance_alpha
        self._d_smooth = d if self._d_smooth is None else a * d + (1.0 - a) * self._d_smooth

        if self._prev_d is None or self._prev_d < _EPS:
            self._prev_d = self._d_smooth
            return 1.0

        ratio = self._d_smooth / self._prev_d               # >1: D_current > D_previous -> zoom in
        if abs(ratio - 1.0) <= self.cfg.zoom_deadzone:
            return 1.0                                      # keep baseline so slow drift accumulates
        self._prev_d = self._d_smooth
        return ratio ** self.cfg.zoom_gain

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _assign_labels(obs: List[HandObservation]) -> Dict[str, HandObservation]:
        if len(obs) >= 2 and obs[0].label == obs[1].label:
            # MediaPipe occasionally gives both hands the same label -> fall back to screen position.
            a, b = sorted(obs[:2], key=lambda o: o.crown_px[0, 0])
            a.label, b.label = "Left", "Right"
        return {o.label: o for o in obs[:2]}

    def _package(self, rotvec, pan, zoom, hands, menu: ToolMenuOutput) -> NavigationOutput:
        claws = {l: t.claw.is_clawing for l, t in self.tracks.items()}
        ratios = {l: t.ratio for l, t in self.tracks.items()}
        tips = [t.obs.tips_px for t in self.tracks.values() if t.claw.is_clawing and t.obs is not None]
        n = anchor = None
        drv = self.tracks.get(self._driver or "")
        if drv is not None and drv.obs is not None and drv.normal is not None:
            n, anchor = drv.normal, drv.obs.center_px
        return NavigationOutput(
            mode=self.mode, rotvec=rotvec, pan_px=pan, zoom_factor=zoom,
            claws=claws, claw_ratios=ratios, hands=hands,
            active_tips_px=np.concatenate(tips, axis=0) if tips else np.zeros((0, 2)),
            palm_normal=n, palm_anchor_px=anchor, menu=menu,
        )
