import numpy as np

from core.face_embedding import extract_embedding
from database.db import DEFAULT_DB_PATH, StudentNotFoundError, get_student, update_student_embedding


def update_student_face(student_id, frame, db_path=DEFAULT_DB_PATH):
    """Extract exactly one face and atomically replace its student's embedding.

    Run this synchronous operation in a worker. Existing extraction validates
    face count; SQLite validates the resulting vector before writing. Errors
    propagate to the caller, and unsuccessful operations retain the old vector.
    """
    if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
            or frame.size == 0 or frame.dtype != np.uint8):
        raise ValueError("Provide a valid uint8 BGR webcam frame with exactly one face.")
    if get_student(student_id, db_path) is None:
        raise StudentNotFoundError(f"Student ID {student_id!r} is not registered.")
    embedding = extract_embedding(frame)
    return update_student_embedding(student_id, embedding, db_path)
