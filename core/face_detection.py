from pathlib import Path

import cv2


class FaceDetector:
    """Detect frontal faces using OpenCV's bundled Haar cascade."""

    def __init__(self):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError("Cannot load the OpenCV face detector.")

    def detect_faces(self, frame):
        """Return (x, y, width, height) boxes in original-frame coordinates."""
        height, width = frame.shape[:2]
        # Bound detection cost independently of the camera's capture resolution.
        scale = min(1.0, 640 / width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale < 1:
            gray = cv2.resize(
                gray, (640, max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        scale_x = width / gray.shape[1]
        scale_y = height / gray.shape[0]
        boxes = []
        for x, y, w, h in faces:
            left, top = round(x * scale_x), round(y * scale_y)
            right = min(width, round((x + w) * scale_x))
            bottom = min(height, round((y + h) * scale_y))
            boxes.append((left, top, right - left, bottom - top))
        return boxes

    def draw_faces(self, frame):
        """Draw boxes in place and return the BGR frame."""
        for x, y, w, h in self.detect_faces(frame):
            cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)
        return frame
