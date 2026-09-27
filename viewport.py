"""
viewport.py — Step 4: the holographic viewport.

Rendering is a tiny NumPy projection + cv2.polylines on preallocated buffers.
A sketch is a handful of polylines, so this is far cheaper than standing up a
GL context, and it sidesteps Wayland/GLX/EGL window-system headaches entirely.
All neon elements are drawn on a black "light layer" and blended ADDITIVELY
(cv2.add) over the dimmed webcam backdrop, which is what sells the hologram look.

Frame: right-handed, X right, Y up, Z toward viewer (same as gesture_engine).
"""
from __future__ import annotations

import itertools
import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

import shapes
from config import ViewportConfig
from gesture_engine import NavMode, NavigationOutput
from tools import Tool, DEFAULT_TOOLS

# BGR neon palette per mode
MODE_COLORS = {
    NavMode.IDLE: (255, 200, 40),
    NavMode.ARMED: (40, 220, 255),
    NavMode.ORBIT: (255, 240, 0),
    NavMode.PAN: (90, 255, 90),
    NavMode.ZOOM: (255, 70, 255),
}
SKETCH_COLOR = (255, 255, 120)
BBOX_COLOR = (255, 170, 30)
GRID_COLOR = (120, 85, 10)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ----------------------------------------------------------------------------- math
def rodrigues(rotvec: np.ndarray) -> np.ndarray:
    """Axis*angle -> 3x3 rotation matrix.  R = I + sin(t) K + (1 - cos(t)) K^2, K = [k]_x."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-9:
        return np.eye(3)
    kx, ky, kz = rotvec / theta
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def orthonormalize(R: np.ndarray) -> np.ndarray:
    """Snap back to the nearest rotation matrix (kills float drift from repeated products)."""
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def _rot_x(deg: float) -> np.ndarray:
    return rodrigues(np.array([math.radians(deg), 0.0, 0.0]))


def _rot_y(deg: float) -> np.ndarray:
    return rodrigues(np.array([0.0, math.radians(deg), 0.0]))


def make_demo_assembly() -> Tuple[List[List[np.ndarray]], List[str]]:
    """A small mechanical assembly — base plate, housing, shaft, gear, cap, and
    four corner bolts — built from shapes.py primitives, so Disassemble/Assemble
    and the per-part Select/Move/Rotate/Scale/Extrude tools have real, varied
    geometry to work with instead of one abstract curve. Replace via set_sketch()
    with your own parts: each is just a list of (N, 3) polyline segments."""
    plate_h = 0.08
    plate = shapes.box((1.6, plate_h, 1.0))
    plate_top = plate_h

    housing_half = (0.45, 0.42, 0.45)
    housing = shapes.translate(shapes.box(housing_half), (0, plate_top + housing_half[1], 0))

    shaft_r, shaft_hh = 0.12, 0.85
    shaft_y = plate_top + shaft_hh - 0.15
    shaft = shapes.translate(shapes.cylinder(shaft_r, shaft_hh, n=14), (0, shaft_y, 0))

    gear_y = plate_top + 2 * housing_half[1] + 0.10
    gear = shapes.translate(shapes.cylinder(0.55, 0.09, n=20, teeth=1.0), (0, gear_y, 0))

    cap_y = shaft_y + shaft_hh + 0.18
    cap = shapes.translate(shapes.sphere(0.18, n=18), (0, cap_y, 0))

    bolt_r, bolt_hh = 0.055, 0.16
    bolt_y = plate_top + bolt_hh - 0.02
    corners = [(1.35, 0.82), (1.35, -0.82), (-1.35, -0.82), (-1.35, 0.82)]
    bolts = [shapes.translate(shapes.cylinder(bolt_r, bolt_hh, n=10), (bx, bolt_y, bz)) for bx, bz in corners]

    parts = [plate, housing, shaft, gear, cap, *bolts]
    names = ["Base Plate", "Housing", "Shaft", "Gear", "Cap", "Bolt 1", "Bolt 2", "Bolt 3", "Bolt 4"]
    return parts, names


# ---------------------------------------------------------------------------- viewport
class HolographicViewport:
    def __init__(self, width: int, height: int, cfg: Optional[ViewportConfig] = None,
                 tools: Optional[Sequence[Tool]] = None) -> None:
        self.cfg = cfg or ViewportConfig()
        # Pass the SAME list you give HolographicGestureEngine(..., tools=...), so the wheel's
        # index always lines up with the engine's; defaults to the same DEFAULT_TOOLS otherwise.
        self.tool_menu_tools: List[Tool] = list(tools) if tools is not None else list(DEFAULT_TOOLS)
        self.w, self.h = width, height
        self.cx, self.cy = width / 2.0, height / 2.0
        self.focal = self.cfg.focal_factor * height
        self._aa = cv2.LINE_AA if self.cfg.antialias else cv2.LINE_8

        self._canvas = np.zeros((height, width, 3), np.uint8)   # reused every frame
        self._layer = np.zeros_like(self._canvas)               # additive light layer

        # Step 8: wall-clock accumulator for the flicker/scan-band effects, and a fixed noise
        # tile reused (rolled to a new offset) every frame rather than regenerated from scratch.
        self._t = 0.0
        self._grain = np.random.default_rng(0).integers(0, 40, (height, width, 3), dtype=np.uint8)

        # 12 box edges = corner pairs that differ in exactly one axis
        signs = np.array(list(itertools.product((-1, 1), repeat=3)), np.float32)   # index bits = axes
        self._signs = signs
        self._edges = np.array([(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j) in (1, 2, 4)])

        self.reset_view()
        self.set_sketch(*make_demo_assembly())

    # ------------------------------------------------------------------ scene setup
    def reset_view(self) -> None:
        self.R_model = _rot_x(-22) @ _rot_y(-30)      # pleasant 3/4 starting angle
        self.R_grid = self.R_model.copy()
        self.pan = np.zeros(2)
        self.target_scale = self.bbox_scale = self.geom_scale = 1.0
        self.menu_pos = 0.0                           # smoothed wheel position (Step 6)
        self.active_tool = None                       # last tool a fist confirmed
        self._menu_flash = None                        # (text, color, frames_left) or None
        self.explode_target = 0.0                     # Step 7: 0 = assembled, 1 = fully exploded
        self.explode_amount = 0.0                      # animated toward explode_target
        if hasattr(self, "n_parts"):                  # not yet known on the very first call
            self._reset_parts()

    def _reset_parts(self) -> None:
        """Step 7: per-part live transform, reset whenever the scene (re)loads or the view resets."""
        self.selected_part: Optional[int] = None
        self.part_pos = np.zeros((self.n_parts, 3), np.float64)
        self.part_rot = [np.eye(3) for _ in range(self.n_parts)]
        self.part_scale = np.ones(self.n_parts, np.float64)
        self.part_stretch = np.ones(self.n_parts, np.float64)   # Step 9: Extrude — local-Y-only stretch

    def set_sketch(self, parts: Sequence[Sequence[np.ndarray]], names: Optional[Sequence[str]] = None,
                    normalize: bool = True) -> None:
        """parts: one entry per PART, each a list of (N, 3) polyline segments in model space
        (Y up) — shapes.py's box/cylinder/sphere all return this shape, and a single legacy
        stroke still works as [stroke]. `names` (optional) labels each part for the HUD/Select
        tool; defaults to "Part 1", "Part 2", ... if omitted."""
        parts = [[np.asarray(seg, np.float32) for seg in part if len(seg) >= 2] for part in parts]
        parts = [part for part in parts if part]
        self.n_parts = len(parts)

        # Flatten to one point array and one segment list for drawing, but remember which part
        # each segment belongs to (_seg_part) and each part's own point range (_part_ranges) —
        # segment = what gets drawn as one continuous polyline, part = what Select/Move/Rotate/
        # Scale/Extrude/explode act on as a rigid unit (can be many segments, e.g. a box's 12 edges).
        all_segs: List[np.ndarray] = []
        seg_part: List[int] = []
        part_ranges: List[Tuple[int, int]] = []
        cursor = 0
        for pi, segs in enumerate(parts):
            start = cursor
            for seg in segs:
                all_segs.append(seg)
                seg_part.append(pi)
                cursor += len(seg)
            part_ranges.append((start, cursor))

        pts = np.concatenate(all_segs) if all_segs else np.zeros((0, 3), np.float32)
        lo, hi = (pts.min(axis=0), pts.max(axis=0)) if len(pts) else (np.zeros(3), np.zeros(3))
        center, half = (lo + hi) / 2.0, (hi - lo) / 2.0
        s = 1.0 / max(float(half.max()), 1e-6) if normalize else 1.0
        self._pts = (pts - center) * s                               # AABB centred at origin, half-extent <= 1
        self._seg_offsets = np.cumsum([0] + [len(seg) for seg in all_segs])
        self._seg_part = np.array(seg_part, np.int64)
        self.bbox_half = (half * s).astype(np.float32)
        self._corners9 = np.vstack((self._signs * self.bbox_half, np.zeros((1, 3), np.float32)))  # 8 corners + centre

        # floor grid segments (model frame, plane just below the sketch)
        g, n = self.cfg.grid_half, self.cfg.grid_div
        y = -float(self.bbox_half[1]) - self.cfg.grid_gap
        ticks = np.linspace(-g, g, n + 1, dtype=np.float32)
        seg_x = [[(-g, y, z), (g, y, z)] for z in ticks]
        seg_z = [[(x, y, -g), (x, y, g)] for x in ticks]
        self._grid = np.array(seg_x + seg_z, np.float32).reshape(-1, 3)

        # Step 7: each part's explode direction is its own centroid direction from the model
        # center (the origin, post-normalize), so a part sitting near the middle barely moves
        # and one out at the edge flies out the most — the standard radial CAD exploded-view
        # look, with no extra per-part authoring needed. The same centroids double as pick
        # targets for Select/Measure. _part_local_size is each part's own bounding size, in its
        # own centroid-relative frame — Measure reads it (scaled by that part's live scale/stretch).
        self._part_centroids = np.zeros((self.n_parts, 3), np.float32)
        self._part_local_size = np.zeros((self.n_parts, 3), np.float32)
        for i, (start, end) in enumerate(part_ranges):
            chunk = self._pts[start:end]
            self._part_centroids[i] = chunk.mean(axis=0)
            self._part_local_size[i] = chunk.max(axis=0) - chunk.min(axis=0)
        norms = np.linalg.norm(self._part_centroids, axis=1, keepdims=True)
        self._part_centroid_dirs = np.divide(self._part_centroids, norms,
                                              out=np.zeros_like(self._part_centroids), where=norms > 1e-6)
        self._part_names = list(names) if names is not None else [f"Part {i + 1}" for i in range(self.n_parts)]
        self._reset_parts()

    def set_explode(self, exploded: bool) -> None:
        """Step 7: the Disassemble / Assemble tool actions. Animated in apply_navigation()."""
        self.explode_target = 1.0 if exploded else 0.0

    # -------------------------------------------------------------------- navigation
    def apply_navigation(self, nav: NavigationOutput, dt: float) -> None:
        c = self.cfg
        active = self.active_tool.name if self.active_tool is not None else None

        # Select AND Measure both track whichever part is nearest the clawing hand's screen
        # position, live, for as long as a claw is held — neither intercepts the claw, so the
        # camera keeps orbiting/panning/zooming normally while you point out a part with the
        # other hand (or the same one, between grabs).
        if active in ("Select", "Measure") and self.n_parts > 0 and nav.mode in (NavMode.ORBIT, NavMode.PAN):
            # Gated on an actually-arbitrated one-hand drag (not just "a claw is down somewhere"):
            # nav.claws is the RAW per-hand detector and stays true through things like the exact
            # fist that CONFIRMS a different tool in the wheel — using nav.mode instead means this
            # only fires during a genuine claw-and-move gesture in the viewport.
            driver = next((o for o in nav.hands if nav.claws.get(o.label)), None)
            if driver is not None:
                self.selected_part = self._nearest_part(driver.center_px)

        # Move / Rotate / Scale / Extrude redirect the SAME claw deltas that would normally fly
        # the camera onto the selected part instead, so precise part edits don't also spin the
        # whole view. Measure redirects nothing — it only reads (see _draw_measurement) — so a
        # measurement never nudges the camera or the geometry being measured.
        part_edit = active in ("Move", "Rotate", "Scale", "Extrude") and self.selected_part is not None
        if active == "Measure":
            pass
        elif part_edit:
            i = self.selected_part
            if active == "Move" and np.any(nav.pan_px):
                # nav.pan_px is a screen-pixel wrist delta; convert to model units using the same
                # pixels-per-model-unit relationship the projection itself uses (focal/cam_dist).
                self.part_pos[i] += np.array([nav.pan_px[0], nav.pan_px[1], 0.0]) * \
                    (c.cam_dist / self.focal) * c.part_move_gain
            elif active == "Rotate" and np.any(nav.rotvec):
                self.part_rot[i] = orthonormalize(rodrigues(nav.rotvec * c.orbit_gain) @ self.part_rot[i])
            elif active == "Scale" and nav.zoom_factor != 1.0:
                self.part_scale[i] = float(np.clip(self.part_scale[i] * nav.zoom_factor,
                                                     c.part_scale_min, c.part_scale_max))
            elif active == "Extrude" and nav.pan_px[1] != 0.0:
                # Vertical drag only, stretching the part along its own local Y — a plate gets
                # thicker, a shaft gets longer — rather than the uniform stretch Scale gives you.
                delta = nav.pan_px[1] * (c.cam_dist / self.focal) * c.extrude_gain
                self.part_stretch[i] = float(np.clip(self.part_stretch[i] + delta,
                                                       c.part_stretch_min, c.part_stretch_max))
        else:
            # No part tool in play (Disassemble/Assemble/none, or a part tool with nothing
            # selected yet) -> original free camera orbit/pan/zoom, unchanged.
            if np.any(nav.rotvec):
                # Palm-normal delta -> model rotation. Deltas are measured in screen space,
                # so they compose on the LEFT of the current orientation.
                self.R_model = orthonormalize(rodrigues(nav.rotvec * c.orbit_gain) @ self.R_model)
                # Floor grid receives the identical delta -> tilts/rotates in lock-step with N.
                self.R_grid = orthonormalize(rodrigues(nav.rotvec * c.grid_sync_gain) @ self.R_grid)
            if np.any(nav.pan_px):
                self.pan = np.clip(self.pan + nav.pan_px * c.pan_gain, -self.w, self.w)
            if nav.zoom_factor != 1.0:
                self.target_scale = float(np.clip(self.target_scale * nav.zoom_factor, c.min_scale, c.max_scale))

        # Chain of first-order lags: target -> bbox (fast) -> geometry (slower).
        # The wireframe therefore visibly leads and the sketch catches up.
        dt = max(dt, 1e-3)
        self._t += dt
        self.bbox_scale += (self.target_scale - self.bbox_scale) * (1.0 - math.exp(-dt / c.tau_bbox))
        self.geom_scale += (self.bbox_scale - self.geom_scale) * (1.0 - math.exp(-dt / c.tau_geom))
        self.explode_amount += (self.explode_target - self.explode_amount) * (1.0 - math.exp(-dt / c.tau_explode))

        # Step 6: tool wheel. Snap silently on open (no spin-up from a stale position), then
        # ease continuously toward the live position — this IS the "falls into center" motion.
        menu = nav.menu
        if menu.just_opened:
            self.menu_pos = menu.position
        if menu.active:
            self.menu_pos += (menu.position - self.menu_pos) * (1.0 - math.exp(-dt / c.menu_tau))
        if menu.selected_tool is not None:
            self.active_tool = menu.selected_tool
            self._menu_flash = (f"{menu.selected_tool.name.upper()} SELECTED", c.menu_color, c.menu_flash_frames)
        elif menu.cancelled:
            self._menu_flash = ("MENU CANCELLED", (140, 140, 140), c.menu_flash_frames)

    # ---------------------------------------------------------------------- rendering
    def render(self, frame_bgr: Optional[np.ndarray], nav: NavigationOutput, fps: float = 0.0) -> np.ndarray:
        c = self.cfg
        canvas, layer = self._canvas, self._layer
        if frame_bgr is not None and c.show_backdrop:
            cv2.convertScaleAbs(frame_bgr, dst=canvas, alpha=c.backdrop_gain)   # dim + keep uint8
        else:
            canvas[:] = c.bg_color
        layer[:] = 0

        mode_color = MODE_COLORS[nav.mode]
        self._draw_floor(layer)
        box_pts = self._draw_bbox(layer)
        self._draw_sketch(layer)
        self._draw_tethers(layer, nav, box_pts, mode_color)
        if c.draw_landmarks:
            self._draw_hands(layer, nav, mode_color)
        self._draw_tool_menu(layer, nav)
        self._draw_measurement(layer, nav)
        if c.hologram_fx:
            self._apply_hologram_fx(layer)
        if c.bloom:
            small = cv2.resize(layer, (self.w // 4, self.h // 4), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), 1.5)
            glow = cv2.resize(small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(layer, 1.0, glow, c.bloom_gain, 0, dst=layer)

        cv2.add(canvas, layer, dst=canvas)                  # additive blend
        if c.scanlines:
            rows = canvas[::3]
            rows -= rows >> 2                               # darken every 3rd row by 25%, in place
        if c.hologram_fx and c.grain_gain > 0:
            oy, ox = int(self._t * 131) % self.h, int(self._t * 227) % self.w
            noise = np.roll(self._grain, shift=(oy, ox), axis=(0, 1))
            cv2.addWeighted(canvas, 1.0, noise, c.grain_gain, 0, dst=canvas)
        self._draw_hud(canvas, nav, fps, mode_color)
        return canvas

    def _apply_hologram_fx(self, layer: np.ndarray) -> None:
        """Step 8: cheap 'real projector' artifacts, applied to the light layer only (so the
        dimmed backdrop and the HUD text drawn afterward stay clean). Each piece is a single
        numpy/cv2 op on the whole layer — no per-pixel Python — so the total cost stays well
        under a millisecond at 640x480 on integrated graphics:
          - chromatic aberration: split R/B a pixel apart, the classic lens-fringe look
          - scan band: a bright horizontal band drifting down and wrapping, like a refresh sweep
          - flicker: whole-layer brightness wobble, a slow sine plus a little per-frame jitter
        """
        c = self.cfg
        if c.aberration_px:
            px = c.aberration_px
            layer[:, :, 2] = np.roll(layer[:, :, 2], px, axis=1)     # R channel right
            layer[:, :, 0] = np.roll(layer[:, :, 0], -px, axis=1)    # B channel left
        if c.scan_band_gain > 0:
            period = self.h + c.scan_band_height
            y0 = int((self._t * c.scan_band_speed) % period - c.scan_band_height)
            ys, ye = max(y0, 0), min(y0 + int(c.scan_band_height), self.h)
            if ye > ys:
                band = layer[ys:ye]
                cv2.addWeighted(band, 1.0, band, c.scan_band_gain, 0, dst=band)
        if c.flicker_strength > 0:
            jitter = float(np.random.uniform(-0.4, 0.4)) * c.flicker_strength
            wave = math.sin(2.0 * math.pi * c.flicker_speed * self._t) * 0.5
            factor = max(1.0 + c.flicker_strength * wave + jitter, 0.0)
            cv2.convertScaleAbs(layer, dst=layer, alpha=factor)

    # ---- projection ---------------------------------------------------------------
    def _project(self, pts_model: np.ndarray, R: np.ndarray, scale: float) -> np.ndarray:
        """model (N,3) -> screen (N,2) float. Rotate, scale, perspective divide, pan."""
        p = (pts_model @ R.T) * scale                       # row-vector form of R @ p
        depth = np.maximum(self.cfg.cam_dist - p[:, 2], 0.25)   # camera on +Z looking down -Z
        k = self.focal / depth
        sx = self.cx + self.pan[0] + p[:, 0] * k
        sy = self.cy - self.pan[1] - p[:, 1] * k            # screen Y grows downward
        return np.stack((sx, sy), axis=1)

    def _neon(self, layer, polys, color, thickness=1) -> None:
        if not polys:
            return
        c = self.cfg
        if c.glow_lines:
            dim = tuple(int(v * c.glow_strength) for v in color)
            cv2.polylines(layer, polys, False, dim, thickness + 2, cv2.LINE_8)
        cv2.polylines(layer, polys, False, color, thickness, self._aa)

    def _glow_circle(self, layer, center, radius, color, thickness=2) -> None:
        """Same dim-wide-pass + bright-thin-pass trick as _neon, for a filled/ringed circle
        instead of a polyline — used for the tool wheel's centered/zoomed-in slot."""
        c = self.cfg
        if c.glow_lines:
            dim = tuple(int(v * c.glow_strength) for v in color)
            cv2.circle(layer, center, radius + 3, dim, thickness + 5, self._aa)
        cv2.circle(layer, center, radius, color, thickness, self._aa)

    # ---- elements -------------------------------------------------------------------
    def _draw_floor(self, layer) -> None:
        scr = np.rint(self._project(self._grid, self.R_grid, self.bbox_scale)).astype(np.int32)
        self._neon(layer, list(scr.reshape(-1, 2, 2)), GRID_COLOR, 1)

    def _draw_bbox(self, layer) -> np.ndarray:
        """Box follows bbox_scale (leads); returns the 9 projected vertices (8 corners + centre)."""
        scr = self._project(self._corners9, self.R_model, self.bbox_scale)
        ip = np.rint(scr).astype(np.int32)
        self._neon(layer, list(ip[:8][self._edges]), BBOX_COLOR, 1)
        cv2.drawMarker(layer, (int(ip[8, 0]), int(ip[8, 1])), BBOX_COLOR, cv2.MARKER_CROSS, 10, 1)
        return scr

    def _current_part_centroids(self) -> np.ndarray:
        """Each part's live centroid: base position + its explode offset + any Move edit."""
        return (self._part_centroids + self._part_centroid_dirs * (self.explode_amount * self.cfg.explode_distance)
                + self.part_pos)

    def _nearest_part(self, screen_px: np.ndarray) -> int:
        """Step 7: Select tool — index of whichever part's projected centroid is closest on screen."""
        scr = self._project(self._current_part_centroids(), self.R_model, self.geom_scale)
        return int(np.argmin(((scr - screen_px) ** 2).sum(axis=1)))

    def _part_dims(self, i: int) -> np.ndarray:
        """Step 9: Measure — a part's current (x, y, z) size in model units, live scale/stretch
        included. Reads _part_local_size, the bounding size captured once in set_sketch()."""
        return self._part_local_size[i] * np.array([1.0, self.part_stretch[i], 1.0]) * self.part_scale[i]

    def _draw_sketch(self, layer) -> None:
        """Geometry follows geom_scale (lags behind the box). Step 7/9: each part carries its own
        rotation/scale/stretch (from Rotate/Scale/Extrude, about its own centroid), translation
        (Move), and radial explode offset (Disassemble/Assemble) — all in MODEL space, so the
        whole assembly still rotates/orbits together as one rigid body regardless of what any
        individual part is doing. A part can be several segments (e.g. a box's 12 edges); they
        all share that part's transform via `_seg_part`."""
        stretch_vec = np.stack([np.array([1.0, self.part_stretch[i], 1.0]) for i in range(self.n_parts)]) \
            if self.n_parts else np.zeros((0, 3))
        o = self._seg_offsets
        chunks = []
        for seg_idx in range(len(o) - 1):
            i = int(self._seg_part[seg_idx])
            rel = self._pts[o[seg_idx]:o[seg_idx + 1]] - self._part_centroids[i]
            rel = (rel @ self.part_rot[i].T) * stretch_vec[i] * self.part_scale[i]
            explode_off = self._part_centroid_dirs[i] * (self.explode_amount * self.cfg.explode_distance)
            chunks.append(self._part_centroids[i] + rel + self.part_pos[i] + explode_off)
        pts = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), np.float32)
        scr = np.rint(self._project(pts, self.R_model, self.geom_scale)).astype(np.int32)

        normal, selected = [], []
        for seg_idx in range(len(o) - 1):
            i = int(self._seg_part[seg_idx])
            poly = scr[o[seg_idx]:o[seg_idx + 1]]
            (selected if i == self.selected_part else normal).append(poly)
        self._neon(layer, normal, SKETCH_COLOR, 1)
        self._neon(layer, selected, self.cfg.select_color, 2)

    def _draw_tethers(self, layer, nav: NavigationOutput, box_pts: np.ndarray, color) -> None:
        tips = nav.active_tips_px
        if len(tips) == 0:
            return
        # Each fingertip -> nearest of the 9 box vertices (8 corners + centre), one vectorised argmin.
        d2 = ((tips[:, None, :] - box_pts[None, :, :]) ** 2).sum(axis=-1)      # (K, 9) squared distances
        targets = box_pts[d2.argmin(axis=1)]                                    # (K, 2)
        segs = np.rint(np.stack((tips, targets), axis=1)).astype(np.int32)      # (K, 2, 2)
        self._neon(layer, list(segs), color, 1)
        for tx, ty in np.rint(targets).astype(int):
            cv2.circle(layer, (int(tx), int(ty)), 3, color, -1, self._aa)

    def _draw_hands(self, layer, nav: NavigationOutput, mode_color) -> None:
        for obs in nav.hands:
            active = nav.claws.get(obs.label, False)
            col = mode_color if active else (150, 110, 40)
            cv2.polylines(layer, [np.rint(obs.crown_px).astype(np.int32)], True, col, 1, self._aa)
            for j, (x, y) in enumerate(np.rint(obs.tips_px).astype(int)):
                # j == 1 is the index fingertip (FINGERTIP_IDS order: thumb, index, middle, ring,
                # pinky) — drawn a couple pixels bigger since it's the one that drives the tool
                # wheel and points at parts, so it's worth being able to spot at a glance.
                r = (4 if active else 3) + (2 if j == 1 else 0)
                cv2.circle(layer, (int(x), int(y)), r, col, -1 if active else 1, self._aa)
            cx, cy = (int(v) for v in np.rint(obs.center_px))
            cv2.circle(layer, (cx, cy), 2, col, -1)
        if nav.palm_normal is not None and nav.palm_anchor_px is not None:
            a = nav.palm_anchor_px
            tip = a + 70.0 * np.array([nav.palm_normal[0], -nav.palm_normal[1]])   # N on screen (Y flipped back)
            cv2.arrowedLine(layer, (int(a[0]), int(a[1])), (int(tip[0]), int(tip[1])),
                            (255, 80, 255), 1, self._aa, tipLength=0.25)

    def _draw_tool_menu(self, layer, nav: NavigationOutput) -> None:
        """Step 6: the point/swipe/fist tool wheel, plus its selection/cancel feedback text."""
        c = self.cfg
        menu = nav.menu
        cx, cy = c.menu_center[0] * self.w, c.menu_center[1] * self.h

        if menu.active:
            tools = self.tool_menu_tools
            n = len(tools) if tools else 1
            # center socket the front slot animates into
            cv2.circle(layer, (int(cx), int(cy)), int(c.menu_slot_r_max + 8), c.menu_color, 1, self._aa)
            order = sorted(range(n), key=lambda i: math.cos(2 * math.pi * (i - self.menu_pos) / n))
            for i in order:                                          # back-to-front, so fronts overpaint
                theta = 2 * math.pi * (i - self.menu_pos) / n
                z = math.cos(theta)                                  # +1 front/center, -1 back
                depth = max((z + 1) * 0.5, 0.0)
                sx = cx + c.menu_radius_x * math.sin(theta)
                sy = cy - c.menu_radius_y * (z - 1.0) * 0.4          # front lands exactly on (cx, cy)
                # Zoom-in: past menu_zoom_start depth, ease sharply up to menu_zoom_boost extra
                # radius on top of the normal near/far interpolation below — a distinct "landing
                # in the socket" pop rather than just the smooth perspective falloff.
                zoom_t = max(depth - c.menu_zoom_start, 0.0) / max(1.0 - c.menu_zoom_start, 1e-6)
                zoom_t = zoom_t * zoom_t * (3.0 - 2.0 * zoom_t)      # smoothstep
                r = c.menu_slot_r_min + (c.menu_slot_r_max - c.menu_slot_r_min) * depth + c.menu_zoom_boost * zoom_t
                col = tuple(int(v * max(depth, 0.18)) for v in c.menu_color)
                if zoom_t > 0.05:                                    # glow while parked in the socket
                    self._glow_circle(layer, (int(sx), int(sy)), int(r), col, 2)
                else:
                    cv2.circle(layer, (int(sx), int(sy)), int(r), col, 1 if depth < 0.6 else 2, self._aa)
                if tools and depth > 0.35:
                    tool = tools[i % len(tools)]
                    font_scale = 0.35 + 0.25 * depth + 0.15 * zoom_t
                    tsize = cv2.getTextSize(tool.glyph, FONT, font_scale, 1)[0]
                    cv2.putText(layer, tool.glyph, (int(sx - tsize[0] / 2), int(sy + tsize[1] / 2)),
                                FONT, font_scale, col, 1, cv2.LINE_AA)
            if tools:
                idx = int(round(self.menu_pos)) % n
                cv2.putText(layer, tools[idx].name, (int(cx - 40), int(cy + c.menu_slot_r_max + 26)),
                            FONT, 0.5, c.menu_color, 1, cv2.LINE_AA)
            hint = "FIST = SELECT   OPEN HAND = CANCEL"
            hsize = cv2.getTextSize(hint, FONT, 0.4, 1)[0]
            cv2.putText(layer, hint, (int(cx - hsize[0] / 2), int(cy + c.menu_slot_r_max + 46)),
                        FONT, 0.4, tuple(v // 2 for v in c.menu_color), 1, cv2.LINE_AA)

        if self._menu_flash is not None:
            text, color, ttl = self._menu_flash
            fade = max(ttl / c.menu_flash_frames, 0.0)
            tsize = cv2.getTextSize(text, FONT, 0.7, 2)[0]
            col = tuple(int(v * fade) for v in color)
            cv2.putText(layer, text, (int(cx - tsize[0] / 2), int(cy)), FONT, 0.7, col, 2, cv2.LINE_AA)
            ttl -= 1
            self._menu_flash = (text, color, ttl) if ttl > 0 else None

    def _draw_measurement(self, layer, nav: NavigationOutput) -> None:
        """Step 9: the Measure tool. Two independent readouts, either or both showing at once:
          - a live ruler between the two wrists, in model units, whenever both hands are visible
          - the selected part's own (x, y, z) size, once a claw has pointed one out (same
            point-at-a-part mechanic Select uses)
        Purely a readout — see apply_navigation's `if active == "Measure": pass` — so it never
        nudges the camera or the part it's measuring.
        """
        if self.active_tool is None or self.active_tool.name != "Measure":
            return
        c = self.cfg
        by_label = {o.label: o for o in nav.hands}
        if "Left" in by_label and "Right" in by_label:
            pL, pR = by_label["Left"].crown_px[0], by_label["Right"].crown_px[0]   # wrist pixels
            dist = float(np.linalg.norm(pL - pR)) * (c.cam_dist / self.focal)      # -> model units
            mid = (pL + pR) / 2.0
            self._neon(layer, [np.rint(np.array([pL, pR])).astype(np.int32)], c.measure_color, 1)
            for p in (pL, pR):
                cv2.drawMarker(layer, (int(p[0]), int(p[1])), c.measure_color, cv2.MARKER_TILTED_CROSS, 10, 1)
            label = f"{dist:.2f} units"
            tsize = cv2.getTextSize(label, FONT, 0.5, 1)[0]
            cv2.putText(layer, label, (int(mid[0] - tsize[0] / 2), int(mid[1] - 12)),
                        FONT, 0.5, c.measure_color, 1, cv2.LINE_AA)
        if self.selected_part is not None:
            x, y, z = self._part_dims(self.selected_part)
            text = f"{self._part_names[self.selected_part]}: {x:.2f} x {y:.2f} x {z:.2f}"
            cv2.putText(layer, text, (16, 132), FONT, 0.45, c.measure_color, 1, cv2.LINE_AA)

    def _draw_hud(self, img, nav: NavigationOutput, fps: float, color) -> None:
        cv2.putText(img, f"MODE {nav.mode.name}", (16, 28), FONT, 0.6, color, 1, cv2.LINE_AA)
        cv2.putText(img, f"SCALE x{self.geom_scale:4.2f}   FPS {fps:4.1f}", (16, 50), FONT, 0.45,
                    (220, 200, 90), 1, cv2.LINE_AA)
        if self.active_tool is not None:
            cv2.putText(img, f"TOOL: {self.active_tool.name}", (16, 72), FONT, 0.45,
                        self.cfg.menu_color, 1, cv2.LINE_AA)
        if self.selected_part is not None:
            cv2.putText(img, f"PART: {self._part_names[self.selected_part]}", (16, 92), FONT, 0.45,
                        self.cfg.select_color, 1, cv2.LINE_AA)
        if self.explode_amount > 0.5:
            cv2.putText(img, "EXPLODED VIEW", (16, 112), FONT, 0.45, (140, 255, 255), 1, cv2.LINE_AA)
        parts = []
        for label in ("Left", "Right"):
            r = nav.claw_ratios.get(label)
            # The nav ClawDetector still runs (and may read "CLAW") while the tool wheel owns
            # this hand, since suppression only holds nav's MODE at IDLE, not the raw detector.
            # Hide that label here so a pointing/fisting hand doesn't show a misleading "CLAW".
            show_claw = nav.claws.get(label) and not (label == "Right" and nav.menu.active)
            parts.append(f"{label[0]} --" if r is None else f"{label[0]} {r:0.2f}{' CLAW' if show_claw else ''}")
        cv2.putText(img, "   ".join(parts), (16, self.h - 16), FONT, 0.45, (220, 200, 90), 1, cv2.LINE_AA)
        m, L = 8, 22                                            # corner brackets
        for (x, y, sx, sy) in ((m, m, 1, 1), (self.w - m, m, -1, 1), (m, self.h - m, 1, -1), (self.w - m, self.h - m, -1, -1)):
            cv2.line(img, (x, y), (x + sx * L, y), color, 1, cv2.LINE_AA)
            cv2.line(img, (x, y), (x, y + sy * L), color, 1, cv2.LINE_AA)
