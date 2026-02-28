from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.face_detection import FaceDetector
from core.live_recognition import LABEL_MAX_AGE, label_for_box


class CameraCapture:
    """Read frames off the GUI thread; the worker owns the camera handle."""

    def __init__(self, index=0):
        self.index = index
        self.messages = Queue(maxsize=1)
        self._stop = Event()
        self._frame_lock = Lock()
        self._latest_frame = None
        self._recognition = (0, [])
        self._recognition_enabled = False
        self._thread = Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def snapshot(self):
        """Copy the latest unannotated frame for background processing."""
        with self._frame_lock:
            if self._stop.is_set() or self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def set_recognition(self, results, captured_at):
        with self._frame_lock:
            if self._recognition_enabled:
                self._recognition = (captured_at, results)

    def enable_recognition(self, enabled):
        with self._frame_lock:
            self._recognition_enabled = enabled
            self._recognition = (0, [])

    def _publish(self, kind, value):
        # Keep only the latest frame so a slow preview cannot build a backlog.
        try:
            self.messages.put_nowait((kind, value))
        except Full:
            try:
                self.messages.get_nowait()
            except Empty:
                pass
            self.messages.put_nowait((kind, value))

    def _run(self):
        capture = None
        error = None
        try:
            detector = FaceDetector()
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except OSError:
                font = ImageFont.load_default()
            capture = cv2.VideoCapture(self.index)
            if not capture.isOpened():
                raise RuntimeError("Cannot open webcam. Check connection and camera permissions.")
            while not self._stop.is_set():
                success, frame = capture.read()
                if self._stop.is_set():
                    break
                if not success or frame is None or frame.size == 0:
                    raise RuntimeError("Cannot read webcam frames. Check the camera connection.")
                with self._frame_lock:
                    self._latest_frame = frame.copy()
                    captured_at, results = self._recognition
                    recognition_enabled = self._recognition_enabled
                if monotonic() - captured_at > LABEL_MAX_AGE:
                    results = []
                labels = []
                for box in detector.detect_faces(frame):
                    x, y, width, height = box
                    label = label_for_box(box, results)
                    cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), (0, 255, 0), 2)
                    if recognition_enabled:
                        labels.append(((x, max(0, y - 24)), label))
                if labels:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(image)
                    for position, label in labels:
                        draw.text(position, label, font=font, fill=(0, 255, 0),
                                  stroke_width=1, stroke_fill=(0, 0, 0))
                    frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                if not self._stop.is_set():
                    self._publish("frame", frame)
        except (cv2.error, RuntimeError) as exc:
            error = str(exc)
        finally:
            with self._frame_lock:
                self._latest_frame = None
            if capture is not None:
                capture.release()
            self._publish("stopped", error)
