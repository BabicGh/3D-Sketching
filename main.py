"""
main.py — execution loop:  camera -> MediaPipe -> gesture engine -> viewport -> window.

Run:  python main.py [--camera 0|/dev/videoN] [--width 640] [--height 480] [--bloom] [--no-backdrop]
                     [--window-scale 1.6] [--fullscreen]
Keys: q/Esc quit | r reset view+wheel | b toggle backdrop | l toggle landmarks
      g toggle bloom | h toggle hologram fx (flicker/aberration/scan/grain)
      f toggle fullscreen

Tool wheel (right hand by default, see ToolWheelConfig.tool_hand):
    point (index out, other 4 curled) -> opens it
    swipe the same hand left/right    -> spins it
    close into a fist while it's open -> confirms the centered tool
"""
import os

# opencv wheels bundle only Qt's X11 plugin. Under Wayland that runs through XWayland;
# forcing it avoids the "could not load the Qt platform plugin wayland" failure.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import time

import cv2

from config import GestureConfig, ViewportConfig
from gesture_engine import HolographicGestureEngine
from hand_tracker import HandTracker, LatestFrameGrabber
from tools import DEFAULT_TOOLS
from viewport import HolographicViewport

WINDOW = "HOLOGRAPHIC VIEWPORT"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0", help="index (0, 2, ...) or device path (/dev/v4l/by-id/...-video-index0)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bloom", action="store_true")
    ap.add_argument("--no-backdrop", action="store_true")
    ap.add_argument("--no-swap-hands", action="store_true",
                     help="disable the Left/Right relabel in hand_tracker.py, if yours comes in correct already")
    ap.add_argument("--window-scale", type=float, default=2.0,
                     help="display window size as a multiple of --width/--height (processing stays at "
                          "the camera's own resolution either way — only the displayed window gets bigger)")
    ap.add_argument("--fullscreen", action="store_true", help="start fullscreen (toggle anytime with 'f')")
    args = ap.parse_args()

    gcfg, vcfg = GestureConfig(), ViewportConfig()
    vcfg.bloom = args.bloom
    vcfg.show_backdrop = not args.no_backdrop

    cam = int(args.camera) if args.camera.isdigit() else args.camera
    grabber = LatestFrameGrabber(cam, args.width, args.height, args.fps)
    tracker = HandTracker(swap_handedness=not args.no_swap_hands)
    tools = list(DEFAULT_TOOLS)                            # same list object given to both, so
    engine = HolographicGestureEngine(gcfg, tools=tools)    # the wheel's index always lines up
    viewport = HolographicViewport(grabber.width, grabber.height, vcfg, tools=tools)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    fullscreen = args.fullscreen
    window_size = (int(grabber.width * args.window_scale), int(grabber.height * args.window_scale))

    def apply_window_mode() -> None:
        # WINDOW_NORMAL always stretches whatever's imshow'n to fill the window, so making the
        # window bigger is enough — the 640x480 (or whatever --width/--height is) processing
        # resolution never changes, which is what keeps MediaPipe fast on this hardware.
        if fullscreen:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, *window_size)

    window_ready = False   # on Linux/Qt, resizing before the window is actually mapped (i.e.
                            # before its first imshow) is silently ignored — see below
    fps, last = 0.0, time.perf_counter()
    try:
        while True:
            frame = grabber.read()
            if frame is None:                                   # camera hiccup: keep the window responsive
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
                continue

            frame = cv2.flip(frame, 1)                          # mirror: hologram behaves like a mirror
            observations = tracker.process(frame)               # 8 landmarks per hand
            nav = engine.update(observations)                   # Steps 1, 2, 3, 5

            now = time.perf_counter()
            dt, last = now - last, now
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-3))
            viewport.apply_navigation(nav, dt)                  # Step 4 + Step 6 state
            if nav.menu.selected_tool is not None:              # Step 6: a fist just confirmed a tool
                tool = nav.menu.selected_tool
                print(f"[tool] selected: {tool.name}")
                if tool.name == "Disassemble":                  # Step 7: explode/collapse are the
                    viewport.set_explode(True)                  # two tools with an instant, gesture-free
                elif tool.name == "Assemble":                   # action — see tools.py for why this is
                    viewport.set_explode(False)                 # done by name here rather than on_select
                if tool.on_select is not None:
                    tool.on_select()
            cv2.imshow(WINDOW, viewport.render(frame, nav, fps))
            if not window_ready:               # window only exists once shown once — resize/fullscreen now
                apply_window_mode()
                window_ready = True

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("r"):
                viewport.reset_view()
                engine.reset()
            elif key == ord("b"):
                vcfg.show_backdrop = not vcfg.show_backdrop
            elif key == ord("l"):
                vcfg.draw_landmarks = not vcfg.draw_landmarks
            elif key == ord("g"):
                vcfg.bloom = not vcfg.bloom
            elif key == ord("h"):
                vcfg.hologram_fx = not vcfg.hologram_fx
            elif key == ord("f"):
                fullscreen = not fullscreen
                apply_window_mode()
    finally:
        grabber.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
