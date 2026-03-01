from dataclasses import dataclass

from core.face_detection import FaceDetector
from core.face_embedding import extract_embedding
from core.face_matching import match_embedding
from database.db import DEFAULT_DB_PATH, initialize_database, list_students


# Initial cosine-similarity threshold; calibrate with representative webcam samples.
RECOGNITION_THRESHOLD = 0.7
RECOGNITION_INTERVAL = 1.0
LABEL_MAX_AGE = 2.0


@dataclass(frozen=True)
class RecognizedFace:
    box: tuple
    label: str
    student_id: str | None = None


def recognize_frame(frame, *, threshold=RECOGNITION_THRESHOLD, db_path=DEFAULT_DB_PATH):
    """Recognize each face independently; return labels and a diagnostic message.

    Single-face extraction is reused on each crop. Failed/ambiguous crops stay
    Unknown, while other faces can still be recognized. No attendance is written.
    """
    boxes = FaceDetector().detect_faces(frame)
    initialize_database(db_path)
    students = list_students(db_path)
    known = {student.student_id: student.embedding for student in students}
    names = {student.student_id: student.full_name for student in students}
    if not known:
        return [RecognizedFace(box, "Unknown") for box in boxes], "No registered students."
    results, errors = [], 0
    for box in boxes:
        x, y, width, height = box
        label = "Unknown"
        matched_id = None
        try:
            embedding = extract_embedding(frame[y:y + height, x:x + width].copy())
            student_id, score = match_embedding(embedding, known, threshold=threshold)
            if student_id != "Unknown":
                matched_id = student_id
                label = f"{names[student_id]} ({student_id})"
        except Exception:
            errors += 1
        results.append(RecognizedFace(box, label, matched_id))
    message = f"Recognition failed for {errors} face(s); showing Unknown." if errors else "Live recognition running"
    return results, message


def label_for_box(box, results):
    """Use a recent result only for a single unambiguous overlapping face box."""
    x, y, w, h = box
    matches = []
    for result in results:
        rx, ry, rw, rh = result.box
        overlap = max(0, min(x + w, rx + rw) - max(x, rx)) * max(0, min(y + h, ry + rh) - max(y, ry))
        union = w * h + rw * rh - overlap
        if union > 0 and overlap / union >= 0.7:
            matches.append(result.label)
    return matches[0] if len(matches) == 1 else "Unknown"
