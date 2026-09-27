"""
hand_tracker.py — I/O front-end: threaded low-latency capture + MediaPipe HandLandmarker (Tasks API).

Kept separate from the math so the engine can be tested without a camera and so
this file is the only place to touch if MediaPipe's API changes (see HandTracker).
"""
from __future__ import annotations

import os
import threading
import time
import urllib.request
from typing import List, Optional

import cv2
import numpy as np

from gesture_engine import TRACKED_IDS, HandObservation


class LatestFrameGrabber:
    """Reads the camera on a background thread and only ever hands out the NEWEST frame.

    A plain cap.read() in the main loop queues stale frames whenever inference is
    slower than the camera; that queue is pure added latency. Here old frames are dropped.
    """

    def __init__(self, index=0, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # MJPG: far higher fps than raw YUYV on USB2
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index!r}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._fresh = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame
            self._fresh.set()

    def read(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Block until a frame newer than the last one returned exists (or timeout)."""
        if not self._fresh.wait(timeout):
            return None
        self._fresh.clear()
        with self._lock:
            return self._frame

    def release(self) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)
        self.cap.release()


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")


def ensure_model(path: str = DEFAULT_MODEL_PATH, url: str = MODEL_URL) -> str:
    """Download the HandLandmarker model bundle once (~7 MB) and reuse it afterwards."""
    if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading hand landmarker model -> {path}")
    tmp = path + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)                       # atomic: never leave a half-written model behind
    except Exception as exc:                        # noqa: BLE001 - any network failure gets the same advice
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"Could not download the model ({exc}). Fetch it manually:\n"
                           f"  mkdir -p {os.path.dirname(path)} && curl -L -o {path} {url}") from exc
    return path


class HandTracker:
    """MediaPipe HandLandmarker (Tasks API) -> list[HandObservation] with ONLY the 8 tracked landmarks.

    mediapipe 1.0 removed the legacy `mp.solutions` API, so this uses the Tasks API in VIDEO
    mode (tracks between frames instead of re-detecting every frame, which is the fast path).
    Everything downstream only ever sees HandObservation, so this is the only file that
    depends on MediaPipe.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, num_hands: int = 2,
                 min_detection: float = 0.6, min_presence: float = 0.5, min_tracking: float = 0.5,
                 swap_handedness: bool = True) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._t0 = time.monotonic()
        self._last_ts = -1
        # MediaPipe's handedness assumes the frame it receives IS the mirror image (its docs say
        # to swap the label yourself if that assumption doesn't hold). We do feed it the mirrored
        # frame, which should already be correct in theory — but in practice this flips per
        # camera/driver/MediaPipe build, so it's a toggle rather than a fixed assumption. Default
        # True matches what's been observed to be backwards; pass False if yours comes in right.
        self._swap = swap_handedness
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def process(self, frame_bgr_mirrored: np.ndarray) -> List[HandObservation]:
        """Expects the MIRRORED frame so 'Left'/'Right' match the user's hands and the display."""
        h, w = frame_bgr_mirrored.shape[:2]
        rgb = cv2.cvtColor(frame_bgr_mirrored, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts = int((time.monotonic() - self._t0) * 1000.0)
        ts = max(ts, self._last_ts + 1)             # VIDEO mode requires strictly increasing timestamps
        self._last_ts = ts
        return self._convert(self._landmarker.detect_for_video(image, ts), w, h)

    def _convert(self, result, w: int, h: int) -> List[HandObservation]:
        out = []
        for lm, handed in zip(result.hand_landmarks, result.handedness):
            arr = np.array([(lm[i].x, lm[i].y, lm[i].z) for i in TRACKED_IDS], np.float32)   # 8 of 21 only
            label = handed[0].category_name
            if self._swap:
                label = "Left" if label == "Right" else "Right"
            out.append(HandObservation.from_normalized(label, arr, w, h))
        return out

    def close(self) -> None:
        self._landmarker.close()
