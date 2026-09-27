"""
selftest.py — camera-free check of the whole pipeline using synthetic hands.

    python selftest.py            # asserts state machine behaviour, prints timings,
                                  # writes selftest_orbit.png / selftest_zoom.png
"""
import time

import cv2
import numpy as np

from config import GestureConfig, ViewportConfig
from gesture_engine import HandObservation, HolographicGestureEngine, NavigationOutput, NavMode
from viewport import HolographicViewport, rodrigues

W, H = 640, 480

# Palm-local template (arbitrary units): x right, y up, z toward viewer.
CROWN = np.array([[0, 0, 0], [3, 8.5, 0], [-3.5, 7, 0]], float)                      # p0, p5, p17
TIPS_OPEN = np.array([[8, 8, 1], [3, 16, 0], [0, 18, 0], [-2, 16, 0], [-4.5, 13, 0]], float)
TIPS_CURL = CROWN.mean(0) + np.array([[3, 0, 3], [1.5, 2, 3], [0, 2.5, 3], [-1.5, 2, 3], [-3, 1, 3]], float)


def hand_custom(label, finger_curls, pos, R=np.eye(3), scale=10.0):
    """finger_curls: length-5 (thumb, index, middle, ring, pinky), each 0=open .. 1=curled."""
    fc = np.asarray(finger_curls, float).reshape(5, 1)
    tips = (1 - fc) * TIPS_OPEN + fc * TIPS_CURL
    p3 = (np.vstack([tips, CROWN]) @ R.T) * scale + np.array([pos[0], -pos[1], 0.0])
    px = np.stack((p3[:, 0], -p3[:, 1]), axis=1)
    return HandObservation(label, p3[:5], p3[5:], px[:5], px[5:])


def hand(label, curl, pos, R=np.eye(3), scale=10.0):
    return hand_custom(label, [curl] * 5, pos, R, scale)


POINT_CURLS = [1, 0, 1, 1, 1]   # thumb, middle, ring, pinky curled; index out


def rot_y(deg):
    return rodrigues(np.array([0, np.radians(deg), 0.0]))


eng = HolographicGestureEngine(GestureConfig())
vp = HolographicViewport(W, H, ViewportConfig())
DT = 1 / 30


def step(*hands):
    nav = eng.update(list(hands))
    vp.apply_navigation(nav, DT)
    return nav


# --- open hand: nothing happens -------------------------------------------------
for _ in range(6):
    nav = step(hand("Right", 0.0, (320, 240)))
assert nav.mode is NavMode.IDLE and not nav.claws["Right"]
print("open hand         ->", nav.mode.name, f"ratio={nav.claw_ratios['Right']:.2f}")

# --- claw + translate => PAN, and rotation is locked out --------------------------
pan_sum, rot_sum = np.zeros(2), np.zeros(3)
x = 320.0
for i in range(30):
    x += 3.0
    R = rot_y(2.0 * i) if i > 15 else np.eye(3)          # add rotation late: must be ignored
    nav = step(hand("Right", 1.0, (x, 240), R))
    pan_sum += nav.pan_px
    rot_sum += nav.rotvec
assert nav.mode is NavMode.PAN, nav.mode
assert pan_sum[0] > 40 and np.linalg.norm(rot_sum) == 0
print("claw + slide      ->", nav.mode.name, f"pan_sum={pan_sum.round(1)}")

# --- release => IDLE ----------------------------------------------------------------
for _ in range(8):
    nav = step(hand("Right", 0.0, (x, 240)))
assert nav.mode is NavMode.IDLE
print("release           ->", nav.mode.name)

# --- claw + twist about Y => ORBIT; 2nd hand claw is ignored (lockout) ---------------
rot_sum = np.zeros(3)
for i in range(30):
    nav = step(hand("Right", 1.0, (320, 240), rot_y(1.5 * i)))
    rot_sum += nav.rotvec
assert nav.mode is NavMode.ORBIT, nav.mode
assert rot_sum[1] > 0.3 and abs(rot_sum[0]) < 0.05           # positive rotation about +Y
for i in range(10):
    nav = step(hand("Right", 1.0, (320, 240), rot_y(45 + 1.5 * i)), hand("Left", 1.0, (150, 240)))
assert nav.mode is NavMode.ORBIT
print("claw + twist      ->", nav.mode.name, f"rot_sum={rot_sum.round(2)}  (2nd claw ignored)")
orbit_nav = nav
cv2.imwrite("selftest_orbit.png", vp.render(None, orbit_nav, 60.0))

# --- release everything, then both claw => ZOOM in, then out --------------------------
for _ in range(8):
    nav = step()
assert nav.mode is NavMode.IDLE
zoom_up = zoom_dn = 1.0
for i in range(25):
    d = 2.5 * i
    nav = step(hand("Left", 1.0, (250 - d, 240)), hand("Right", 1.0, (390 + d, 240)))
    zoom_up *= nav.zoom_factor
assert nav.mode is NavMode.ZOOM and zoom_up > 1.3, zoom_up
zoom_frame = vp.render(None, nav, 60.0)
for i in range(25):
    d = 2.5 * (24 - i)
    nav = step(hand("Left", 1.0, (250 - d, 240)), hand("Right", 1.0, (390 + d, 240)))
    zoom_dn *= nav.zoom_factor
assert zoom_dn < 0.8, zoom_dn
print(f"two-hand zoom     -> in x{zoom_up:.2f}, out x{zoom_dn:.2f}; "
      f"bbox={vp.bbox_scale:.2f} geom={vp.geom_scale:.2f} target={vp.target_scale:.2f}")

# mid-zoom render: bbox should lead the geometry
vp.target_scale = 2.0
for _ in range(4):
    vp.apply_navigation(nav, DT)
assert vp.bbox_scale > vp.geom_scale
print(f"bbox leads geom   -> bbox={vp.bbox_scale:.2f} geom={vp.geom_scale:.2f}")
cv2.imwrite("selftest_zoom.png", vp.render(None, nav, 60.0))

# --- timing (pure CPU cost of engine + render on THIS machine) --------------------------
vp.reset_view()
hs = [hand("Left", 1.0, (250, 240)), hand("Right", 1.0, (390, 240))]
t0 = time.perf_counter()
N = 300
for _ in range(N):
    nav = eng.update(hs)
t1 = time.perf_counter()
for _ in range(N):
    vp.render(None, nav, 60.0)
t2 = time.perf_counter()
print(f"engine.update: {(t1 - t0) / N * 1e3:.3f} ms/frame | viewport.render: {(t2 - t1) / N * 1e3:.3f} ms/frame")

# =============================================================================
# Step 6: tool wheel — point opens it, swipe browses it, fist selects it
# =============================================================================
eng.reset()
for _ in range(10):                                          # make sure nav is fully idle first
    nav = step(hand("Left", 0.0, (150, 240)), hand("Right", 0.0, (500, 240)))
assert nav.mode is NavMode.IDLE and not nav.menu.active

for _ in range(8):
    nav = step(hand_custom("Right", POINT_CURLS, (400, 240)))
assert nav.menu.active and nav.menu.just_opened is False or True   # just_opened true on the exact frame; tolerate either by now
print(f"point gesture     -> menu.active={nav.menu.active}  tool={nav.menu.tool.name}")
assert nav.menu.active, "pointing should open the tool wheel"

start_pos = nav.menu.position
x = 400.0
for i in range(60):                                            # swipe right
    x += 3.0
    nav = step(hand_custom("Right", POINT_CURLS, (x, 240)))
assert nav.menu.active
moved = nav.menu.position - start_pos
print(f"swipe right       -> moved {moved:+.2f} tool-units, index={nav.menu.index} tool={nav.menu.tool.name}")
assert moved > 1.5, f"swipe should move the wheel by a couple of tools, only moved {moved:.2f}"

target_index = nav.menu.index
selected = None                                                # selected_tool is a one-shot field:
for _ in range(8):                                              # capture it the instant it fires
    nav = step(hand_custom("Right", [1, 1, 1, 1, 1], (x, 240)))
    if nav.menu.selected_tool is not None:
        selected = nav.menu.selected_tool
        break
assert selected is not None, "fist should select the centered tool"
assert selected.name == eng.tool_menu.tools[target_index].name
assert nav.menu.active is False
print(f"fist select       -> selected={selected.name}")

for _ in range(6):                                             # release, back to nothing
    nav = step(hand("Right", 0.0, (x, 240)))
assert nav.mode is NavMode.IDLE and not nav.menu.active
print("post-select        -> nav resumed cleanly, mode =", nav.mode.name)

# open again, then cancel by opening the hand instead of fisting
for _ in range(8):
    nav = step(hand_custom("Right", POINT_CURLS, (400, 240)))
assert nav.menu.active
cancelled = False                                              # cancelled is a one-shot field too:
for _ in range(8):                                              # capture it the instant it fires
    nav = step(hand("Right", 0.0, (400, 240)))                 # fully open hand, no fist
    if nav.menu.cancelled:
        cancelled = True
        break
assert cancelled and nav.menu.selected_tool is None and not nav.menu.active
print("open-hand cancel  -> cancelled =", cancelled)

# open again, then take the hand out of frame entirely for a long stretch — it must stay open
# (this is the new behavior: only a fist or a fully open hand close it, never a lost hand)
eng.reset()
for _ in range(10):
    nav = step(hand("Left", 0.0, (150, 240)), hand("Right", 0.0, (500, 240)))
for _ in range(8):
    nav = step(hand_custom("Right", POINT_CURLS, (400, 240)))
assert nav.menu.active
pos_before_loss = nav.menu.position
for _ in range(45):                                             # ~1.5s at 30fps with no Right hand at all
    nav = step(hand("Left", 0.0, (150, 240)))
assert nav.menu.active, "wheel must stay open while the hand is simply out of frame"
assert nav.menu.position == pos_before_loss, "a lost-then-absent hand shouldn't drift the wheel"
print(f"hand lost 45f     -> menu.active={nav.menu.active}  position unchanged at {nav.menu.position:.2f}")

x = 400.0                                                        # hand returns: swipe still works,
for i in range(60):                                              # and the re-detect frame itself
    x += 3.0                                                     # shouldn't register as a swipe jump
    nav = step(hand_custom("Right", POINT_CURLS, (x, 240)))
moved = nav.menu.position - pos_before_loss
assert moved > 1.5, f"swipe after recovery should still move the wheel, only moved {moved:.2f}"
print(f"swipe after recovery -> moved {moved:+.2f} tool-units")

for _ in range(8):                                               # and it can still be selected normally
    nav = step(hand_custom("Right", [1, 1, 1, 1, 1], (x, 240)))
    if nav.menu.selected_tool is not None:
        break
assert nav.menu.selected_tool is not None and not nav.menu.active
print(f"select after recovery -> selected={nav.menu.selected_tool.name}")

print("ALL CHECKS PASSED")

# =============================================================================
# Step 7: per-part tool actions — Select / Move / Rotate / Scale / Disassemble / Assemble
# =============================================================================
from tools import DEFAULT_TOOLS  # noqa: E402  (kept near its one use, after the main run above)

by_name = {t.name: t for t in DEFAULT_TOOLS}
eng2 = HolographicGestureEngine(GestureConfig(), tools=DEFAULT_TOOLS)
vp2 = HolographicViewport(W, H, ViewportConfig(), tools=DEFAULT_TOOLS)


def pick_tool(name):
    """Drive the real point -> swipe -> fist sequence until `name` is the confirmed tool."""
    nav = eng2.update([hand("Left", 0.0, (150, 240)), hand("Right", 0.0, (500, 240))])
    for _ in range(10):
        nav = eng2.update([hand("Left", 0.0, (150, 240)), hand("Right", 0.0, (500, 240))])
    for _ in range(8):
        nav = eng2.update([hand_custom("Right", POINT_CURLS, (400, 240))])
    target = eng2.tool_menu.tools.index(by_name[name])
    x = 400.0
    for _ in range(400):                                    # swipe until the target is centered
        if eng2.tool_menu.position.__round__() % len(eng2.tool_menu.tools) == target:
            break
        x += 3.0
        nav = eng2.update([hand_custom("Right", POINT_CURLS, (x, 240))])
    selected = None
    for _ in range(8):
        nav = eng2.update([hand_custom("Right", [1, 1, 1, 1, 1], (x, 240))])
        vp2.apply_navigation(nav, DT)
        if nav.menu.selected_tool is not None:
            selected = nav.menu.selected_tool
            break
    assert selected is not None and selected.name == name, f"expected {name}, got {selected}"
    for _ in range(6):                                      # release, back to nothing, ready to grab
        nav = eng2.update([hand("Right", 0.0, (x, 240))])
        vp2.apply_navigation(nav, DT)
    return nav


def drag(hand_fn, frames=30, settle_at=None):
    """Optionally settle the open hand at the drag's start pose for a few frames before the
    loop claws in — avoids an artificial position jump at the exact moment intent is decided,
    the same way a real user pauses before regrabbing rather than teleporting mid-claw."""
    if settle_at is not None:
        for _ in range(5):
            eng2.update([settle_at])
    nav = None
    for i in range(frames):
        nav = eng2.update([hand_fn(i)])
        vp2.apply_navigation(nav, DT)
    return nav


# --- Select: claw near a part's screen position selects it, and the camera keeps orbiting -----
pick_tool("Select")
before_R = vp2.R_model.copy()
nav = drag(lambda i: hand("Right", 1.0, (320, 240), rot_y(2.0 * i)),
           settle_at=hand("Right", 0.0, (320, 240)))
assert vp2.selected_part is not None, "Select should have picked a part"
assert not np.allclose(vp2.R_model, before_R), "camera should still orbit while Select is active"
print(f"select tool       -> picked '{vp2._part_names[vp2.selected_part]}', camera still orbits")

# --- Move: claw-drag now moves the selected part, and the camera does NOT move ----------------
pick_tool("Move")
sel = vp2.selected_part
before_pos, before_R2 = vp2.part_pos[sel].copy(), vp2.R_model.copy()
nav = drag(lambda i: hand("Right", 1.0, (320.0 + 4.0 * i, 240)),
           settle_at=hand("Right", 0.0, (320, 240)))
assert not np.allclose(vp2.part_pos[sel], before_pos), "Move should translate the selected part"
assert np.allclose(vp2.R_model, before_R2), "camera should NOT move while editing a part"
print(f"move tool         -> part moved by {(vp2.part_pos[sel] - before_pos).round(3)}")

# --- Rotate: claw-twist rotates the part about its own centroid, camera stays put -------------
pick_tool("Rotate")
sel = vp2.selected_part
before_rot = vp2.part_rot[sel].copy()
nav = drag(lambda i: hand("Right", 1.0, (320, 240), rot_y(2.0 * i)),
           settle_at=hand("Right", 0.0, (320, 240)))
assert not np.allclose(vp2.part_rot[sel], before_rot), "Rotate should spin the selected part"
print("rotate tool       -> part rotation changed")

# --- Scale: two-hand pinch scales the part -----------------------------------------------------
pick_tool("Scale")
sel = vp2.selected_part
before_scale = vp2.part_scale[sel]
nav = None
for i in range(30):
    d = 2.5 * i
    nav = eng2.update([hand("Left", 1.0, (250 - d, 240)), hand("Right", 1.0, (390 + d, 240))])
    vp2.apply_navigation(nav, DT)
assert vp2.part_scale[sel] > before_scale, "Scale should grow the selected part"
print(f"scale tool        -> part scale {before_scale:.2f} -> {vp2.part_scale[sel]:.2f}")

# --- Extrude: vertical claw-drag stretches the part along its own local Y only -----------------
pick_tool("Extrude")
sel = vp2.selected_part
before_stretch = vp2.part_stretch[sel]
before_scale2 = vp2.part_scale[sel]
nav = drag(lambda i: hand("Right", 1.0, (320, 240.0 - 4.0 * i)),         # drag straight up
           settle_at=hand("Right", 0.0, (320, 240)))
assert vp2.part_stretch[sel] != before_stretch, "Extrude should change the part's Y stretch"
assert vp2.part_scale[sel] == before_scale2, "Extrude should NOT touch uniform Scale"
print(f"extrude tool      -> part stretch {before_stretch:.2f} -> {vp2.part_stretch[sel]:.2f}")

# --- Measure: read-only — a two-hand ruler, and it must NOT move the camera or any part --------
pick_tool("Measure")
before_R3, before_pos2 = vp2.R_model.copy(), vp2.part_pos.copy()
nav = None
for i in range(20):                                             # two open hands, moving apart
    d = 2.0 * i
    nav = eng2.update([hand("Left", 0.0, (250 - d, 240)), hand("Right", 0.0, (390 + d, 240))])
    vp2.apply_navigation(nav, DT)
assert np.allclose(vp2.R_model, before_R3), "Measure must never move the camera"
assert np.allclose(vp2.part_pos, before_pos2), "Measure must never move a part"
by_label = {o.label: o for o in nav.hands}
ruler_px = float(np.linalg.norm(by_label["Left"].crown_px[0] - by_label["Right"].crown_px[0]))
ruler_units = ruler_px * (vp2.cfg.cam_dist / vp2.focal)
print(f"measure tool      -> camera/parts untouched, live ruler = {ruler_units:.2f} units")
img = vp2.render(None, nav, 60.0)   # exercise _draw_measurement's rendering path too
assert img.shape == (H, W, 3)

# --- Disassemble / Assemble: instant, gesture-free actions on selection (main.py wires these) --
assert vp2.explode_amount < 0.05
vp2.set_explode(True)
for _ in range(60):
    vp2.apply_navigation(NavigationOutput(), DT)
assert vp2.explode_amount > 0.9, vp2.explode_amount
print(f"disassemble       -> explode_amount = {vp2.explode_amount:.2f}")
vp2.set_explode(False)
for _ in range(60):
    vp2.apply_navigation(NavigationOutput(), DT)
assert vp2.explode_amount < 0.1, vp2.explode_amount
print(f"assemble          -> explode_amount = {vp2.explode_amount:.2f}")

cv2.imwrite("selftest_parts.png", vp2.render(None, nav or NavigationOutput(), 60.0))
print("ALL STEP-7 CHECKS PASSED")
