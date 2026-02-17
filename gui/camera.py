from queue import Empty, Full, Queue
from threading import Event, Lock, Thread

import cv2

from core.face_detection import FaceDetector


class CameraCapture:
    """Read frames off the GUI thread; the worker owns the camera handle."""

    def __init__(self, index=0):
        self.index = index
        self.messages = Queue(maxsize=1)
        self._stop = Event()
        self._frame_lock = Lock()
        self._latest_frame = None
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
                frame = detector.draw_faces(frame)
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
