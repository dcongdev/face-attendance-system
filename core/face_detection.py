from pathlib import Path

import cv2


class FaceDetector:
    """Detect frontal faces using OpenCV's bundled Haar cascade."""

    def __init__(self):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError("Cannot load the OpenCV face detector.")

    def draw_faces(self, frame):
        """Draw boxes in place and return the BGR frame, including when no face is found."""
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
        for x, y, w, h in faces:
            top_left = (round(x * scale_x), round(y * scale_y))
            bottom_right = (
                min(width - 1, round((x + w) * scale_x)),
                min(height - 1, round((y + h) * scale_y)),
            )
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        return frame
